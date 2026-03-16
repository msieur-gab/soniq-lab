#!/home/msieur-gab/soniq-lab/venv/bin/python3
"""Re-classify all tracks using current formula classifiers.

Reads stored features from DB, runs predict_all, updates cls_json.
Does NOT re-extract audio features — uses existing scalars/vectors.

Usage:
    python3 reclassify.py                    # soniq_0.6.db
    python3 reclassify.py --db soniq_0.5.db  # specific DB
    python3 reclassify.py --dry-run          # preview, no writes
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from py.classifiers import predict_all
from py.validate_formulas import _reconstruct_librosa_features

DB_DEFAULT = Path(__file__).parent / "soniq_0.6.db"


def reclassify(db_path, dry_run=False):
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"  ERROR: {db_path} not found")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, artist, title, scalars_json, vectors_json
        FROM tracks WHERE status = 'done' AND scalars_json IS NOT NULL
    """)
    rows = cur.fetchall()
    total = len(rows)

    print(f"\n  Re-classifying {total} tracks in {db_path.name}")
    print(f"  Mode: {'dry-run' if dry_run else 'write to DB'}\n")

    t_start = time.time()
    done = 0
    errors = 0

    for i, row in enumerate(rows):
        try:
            scalars = json.loads(row["scalars_json"])
            vectors = json.loads(row["vectors_json"]) if row["vectors_json"] else {}
            feats = _reconstruct_librosa_features(scalars, vectors)
            cls = predict_all(feats)

            # Build cls_json
            cls_out = {}
            for key in ("happy", "sad", "relaxed", "aggressive", "party", "acoustic",
                        "danceable", "instrumental", "vocal", "tonal", "atonal",
                        "arousal", "valence",
                        "radiant", "somber", "brilliant", "warm",
                        "energetic", "still",
                        "hypnotic", "varied",
                        "contemplative", "restless"):
                if key in cls:
                    cls_out[key] = cls[key]

            if "_energy_components" in cls:
                cls_out["nrg"] = cls["_energy_components"]
            if "_hypnotic_path" in cls:
                cls_out["hypnotic_path"] = cls["_hypnotic_path"]

            cls_json = json.dumps(cls_out, separators=(",", ":"))

            if not dry_run:
                cur.execute("UPDATE tracks SET cls_json = ?, radiant = ? WHERE id = ?",
                            (cls_json, cls.get("radiant", 0.5), row["id"]))

            done += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{row['artist']}] {row['title']}: {e}")

        if (i + 1) % 200 == 0:
            if not dry_run:
                conn.commit()
            elapsed = time.time() - t_start
            rate = done / (elapsed + 1e-8)
            print(f"  [{i + 1}/{total}] {done} done — {rate:.0f} t/s")

    if not dry_run:
        conn.commit()

    elapsed = time.time() - t_start
    print(f"\n  Done: {done} classified, {errors} errors, {elapsed:.1f}s")

    # Quick distribution check
    if not dry_run:
        cur.execute("SELECT cls_json FROM tracks WHERE status='done' AND cls_json IS NOT NULL LIMIT 500")
        sample = [json.loads(r[0]) for r in cur.fetchall()]
        print(f"\n  Distribution sample (first 500):")
        for key in ["sad", "relaxed", "danceable", "party", "instrumental", "radiant", "contemplative"]:
            vals = [s.get(key, 0) for s in sample if key in s]
            if vals:
                mn, mx = min(vals), max(vals)
                avg = sum(vals) / len(vals)
                print(f"    {key:<16} min={mn:.3f} max={mx:.3f} mean={avg:.3f}")

    conn.close()


if __name__ == "__main__":
    db_path = DB_DEFAULT
    dry_run = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            dry_run = True
        elif args[i] == "--db" and i + 1 < len(args):
            db_path = Path(args[i + 1])
            i += 1
        elif args[i] in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        i += 1

    reclassify(db_path, dry_run=dry_run)
