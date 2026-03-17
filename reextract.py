#!/home/msieur-gab/soniq-lab/venv/bin/python3
"""Re-extract all tracks with current pipeline (v0.7: 60s centered, ffmpeg, trimmed).

Reads paths from DB, runs full feature extraction, updates scalars_json
and vectors_json. Does NOT touch cls_json — run reclassify.py after.

Usage:
    python3 reextract.py                     # soniq_0.7.db
    python3 reextract.py --db soniq_0.6.db   # specific DB
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from soniq.librosa_features import extract_track_features
from soniq.tags import SCALAR_SHORT

DB_DEFAULT = Path(__file__).parent / "soniq_0.7.db"

# Reverse SCALAR_SHORT: long → short for DB storage
LONG_TO_SHORT = {v_long: v_short for v_long, v_short in SCALAR_SHORT.items()}

# Vector keys: long → short
VEC_MAP = {
    "mfcc_mean": "mfcc_m",
    "chroma_mean": "chroma",
    "tonnetz_mean": "tonnetz",
}


def build_db_scalars(features):
    """Convert librosa features dict to DB-format scalars dict (short keys)."""
    scalars = {}
    for long_key, val in features.items():
        if isinstance(val, (list, np.ndarray)):
            continue  # vectors handled separately
        short_key = LONG_TO_SHORT.get(long_key, long_key)
        if isinstance(val, float):
            scalars[short_key] = round(val, 4)
        else:
            scalars[short_key] = val
    return scalars


def build_db_vectors(features):
    """Convert librosa features dict to DB-format vectors dict (short keys)."""
    vectors = {}
    for long_key, short_key in VEC_MAP.items():
        if long_key in features:
            val = features[long_key]
            if isinstance(val, list):
                vectors[short_key] = [round(v, 4) for v in val]
    return vectors


def reextract(db_path):
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"  ERROR: {db_path} not found")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, path, artist, title FROM tracks WHERE status = 'done'")
    rows = cur.fetchall()
    total = len(rows)

    print(f"\n  Re-extracting {total} tracks in {db_path.name}")
    print(f"  Pipeline: 60s centered, ffmpeg decode, trimmed features\n")

    t_start = time.time()
    done = 0
    errors = 0
    missing = 0

    for i, row in enumerate(rows):
        filepath = row["path"]

        if not os.path.isfile(filepath):
            missing += 1
            continue

        try:
            features = extract_track_features(filepath)
            if features is None:
                raise ValueError("Extraction returned None")

            scalars = build_db_scalars(features)
            vectors = build_db_vectors(features)

            cur.execute(
                "UPDATE tracks SET scalars_json = ?, vectors_json = ?, tag_version = '0.7' WHERE id = ?",
                (json.dumps(scalars, separators=(",", ":")),
                 json.dumps(vectors, separators=(",", ":")),
                 row["id"])
            )
            done += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERR [{row['artist'][:20]}] {row['title'][:25]}: {e}")

        if (i + 1) % 50 == 0:
            conn.commit()
            elapsed = time.time() - t_start
            rate = done / (elapsed + 1e-8)
            remaining = (total - i - 1) / (rate + 1e-8)
            print(f"  [{i + 1}/{total}] {done} done, {errors} err, {missing} miss "
                  f"— {rate:.1f} t/s — ETA {remaining / 60:.0f}m")

    conn.commit()
    elapsed = time.time() - t_start
    print(f"\n  Done: {done} extracted, {errors} errors, {missing} missing")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Verify
    cur.execute("SELECT COUNT(*) FROM tracks WHERE tag_version = '0.7'")
    count = cur.fetchone()[0]
    print(f"  Verified: {count}/{total} tracks re-extracted")

    conn.close()


if __name__ == "__main__":
    # numpy needed by build_db_scalars for isinstance check
    import numpy as np

    db_path = DB_DEFAULT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--db" and i < len(sys.argv):
            db_path = Path(sys.argv[i + 1])
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)

    reextract(db_path)
