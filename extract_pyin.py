#!/home/msieur-gab/soniq-lab/venv/bin/python3
"""Extract pYIN + chroma_major_corr for all tracks missing these features.

Adds 5 new scalars to existing scalars_json (does NOT overwrite):
  - voiced_ratio, voiced_conf, f0_mean, f0_std, chroma_major_corr

chroma_major_corr is computed from existing DB data (no audio needed).
pYIN requires loading audio (~2-3s per track).

Usage:
    python3 extract_pyin.py                    # v0.6 DB (default)
    python3 extract_pyin.py --db soniq_0.5.db  # specific DB
    python3 extract_pyin.py --dry-run           # preview only
    python3 extract_pyin.py --count-only        # just show counts
"""

import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress librosa PySoundFile warnings — expected for m4a/AAC files,
# audioread fallback via ffmpeg works correctly
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


DB_DEFAULT = [
    Path(__file__).parent / "soniq_0.6.db",
]

PYIN_KEYS = {"voiced_ratio", "voiced_conf", "f0_mean", "f0_std"}
CHROMA_KEY = "chroma_major_corr"

# Krumhansl-Schmuckler major key profile
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])


def compute_chroma_major_corr(chroma_mean, key):
    """Compute chroma-major-key correlation from stored vectors."""
    chroma_avg = np.array(chroma_mean)
    rotated = np.roll(chroma_avg, -int(key))
    corr = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
    return max(-1.0, min(1.0, corr))


def extract_pyin(filepath, max_duration=300):
    """Extract pYIN features from audio file. Returns dict or None.

    Uses 30s segment from track center — pYIN is O(n²), so shorter
    segments are dramatically faster (~6s vs ~28s for 90s load) while
    voiced_ratio stabilizes within 2% of full-length extraction.
    """
    import librosa

    try:
        duration = librosa.get_duration(path=filepath)
    except Exception:
        return None

    if duration < 3:
        return None

    # Load 30s from 15% into the track (avoids intros/outros)
    load_offset = max(0, duration * 0.15)
    load_duration = min(30, duration - load_offset)

    try:
        y_full, sr = librosa.load(filepath, sr=22050, offset=load_offset,
                                  duration=load_duration, mono=True)
    except Exception:
        return None

    if len(y_full) < sr:
        return None

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_full, fmin=80, fmax=800, sr=sr
        )
        voiced_ratio = float(np.mean(voiced_flag))
        voiced_confidence = float(np.mean(voiced_prob[voiced_flag])) if np.any(voiced_flag) else 0.0
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
    except Exception:
        voiced_ratio = 0.0
        voiced_confidence = 0.0
        f0_mean = 0.0
        f0_std = 0.0

    return {
        "voiced_ratio": round(voiced_ratio, 4),
        "voiced_conf": round(voiced_confidence, 4),
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
    }


def process_db(db_path, dry_run=False, count_only=False):
    """Process a single database — add pYIN + chroma_major_corr."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"  DB not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all done tracks
    cur.execute("""
        SELECT id, path, scalars_json, vectors_json
        FROM tracks WHERE status = 'done' AND scalars_json IS NOT NULL
    """)
    rows = cur.fetchall()
    total = len(rows)

    # Count which need pYIN vs chroma_major_corr
    need_pyin = []
    need_chroma = []
    for row in rows:
        scalars = json.loads(row["scalars_json"])
        if "voiced_ratio" not in scalars:
            need_pyin.append(row)
        if CHROMA_KEY not in scalars:
            need_chroma.append(row)

    print(f"\n  DB: {db_path.name}")
    print(f"  Total done: {total}")
    print(f"  Need pYIN: {len(need_pyin)}")
    print(f"  Need chroma_major_corr: {len(need_chroma)}")

    if not need_pyin and not need_chroma:
        print("  Nothing to do.")
        conn.close()
        return

    if count_only:
        conn.close()
        return

    # --- Phase A: chroma_major_corr from existing DB data (fast) ---
    chroma_done = 0
    chroma_skip = 0
    for row in need_chroma:
        scalars = json.loads(row["scalars_json"])
        vectors = json.loads(row["vectors_json"]) if row["vectors_json"] else {}

        chroma_mean = vectors.get("chroma")
        key = scalars.get("key", 0)

        if chroma_mean and len(chroma_mean) == 12:
            corr = compute_chroma_major_corr(chroma_mean, key)
            scalars[CHROMA_KEY] = round(corr, 4)

            if not dry_run:
                cur.execute("UPDATE tracks SET scalars_json = ? WHERE id = ?",
                            (json.dumps(scalars, separators=(",", ":")), row["id"]))
            chroma_done += 1
        else:
            chroma_skip += 1

    if not dry_run and chroma_done > 0:
        conn.commit()
    print(f"  chroma_major_corr: {chroma_done} computed, {chroma_skip} skipped (no chroma vector)")

    # --- Phase B: pYIN from audio (slow, ~2-3s per track) ---
    if not need_pyin:
        print("  pYIN: nothing to do.")
        conn.close()
        return

    print(f"\n  Extracting pYIN for {len(need_pyin)} tracks...")
    t_start = time.time()
    done = 0
    errors = 0
    missing = 0
    batch_size = 50  # commit every N tracks

    for i, row in enumerate(need_pyin):
        filepath = row["path"]

        if not os.path.isfile(filepath):
            missing += 1
            continue

        pyin_feats = extract_pyin(filepath)

        if pyin_feats is None:
            errors += 1
            continue

        # Re-read current scalars from DB (not the stale cached value)
        # to preserve chroma_major_corr added in Phase A
        cur.execute("SELECT scalars_json FROM tracks WHERE id = ?", (row["id"],))
        current_sj = cur.fetchone()[0]
        scalars = json.loads(current_sj)
        scalars.update(pyin_feats)

        if not dry_run:
            cur.execute("UPDATE tracks SET scalars_json = ? WHERE id = ?",
                        (json.dumps(scalars, separators=(",", ":")), row["id"]))

        done += 1

        # Progress
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = done / (elapsed + 1e-8)
            remaining = (len(need_pyin) - i - 1) / (rate + 1e-8)
            print(f"  [{i + 1}/{len(need_pyin)}] {done} done, {errors} err, "
                  f"{missing} missing — {rate:.1f} t/s — ETA {remaining / 60:.0f}m")

        # Batch commit
        if not dry_run and (i + 1) % batch_size == 0:
            conn.commit()

    # Final commit
    if not dry_run:
        conn.commit()

    elapsed = time.time() - t_start
    print(f"\n  pYIN done: {done} extracted, {errors} errors, {missing} missing files")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Verify
    cur.execute("SELECT COUNT(*) FROM tracks WHERE status='done' AND scalars_json LIKE '%voiced_ratio%'")
    count = cur.fetchone()[0]
    print(f"  Verified: {count}/{total} tracks now have pYIN features")

    conn.close()


if __name__ == "__main__":
    dry_run = False
    count_only = False
    dbs = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            dry_run = True
        elif args[i] == "--count-only":
            count_only = True
        elif args[i] in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        elif args[i] == "--db" and i + 1 < len(args):
            dbs.append(Path(args[i + 1]))
            i += 1
        i += 1

    if not dbs:
        dbs = DB_DEFAULT

    mode = "count-only" if count_only else ("dry-run" if dry_run else "write to DB")
    print(f"\n  pYIN + chroma_major_corr extraction")
    print(f"  Mode: {mode}")

    for db in dbs:
        process_db(db, dry_run=dry_run, count_only=count_only)

    print("\n  All done.\n")
