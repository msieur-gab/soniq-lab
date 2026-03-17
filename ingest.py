#!/home/msieur-gab/soniq-lab/venv/bin/python3
"""Ingest new tracks into the soniq pipeline.

Scans a music folder, finds files not yet in the DB, runs librosa feature
extraction + classifiers, and marks them done. Skips already-processed tracks.

Usage:
    python3 ingest.py                          # scans default folder
    python3 ingest.py ~/soniq-player/music/    # scans specific folder
    python3 ingest.py --dry-run                # scan + classify but don't write tags
"""

import json
import os
import sys
import time
from pathlib import Path

# Add soniq-lab to path so we can import the pipeline modules
sys.path.insert(0, str(Path(__file__).parent))

from py import db as dbmod
from soniq.librosa_features import extract_track_features
from soniq.classifiers import predict_all
from soniq.genre import init as genre_init, fetch_album_genre
from soniq.tags import build_tag, write_tag

# Defaults
DEFAULT_MUSIC = os.path.expanduser("~/soniq-player/music")
DB_PATH = Path(__file__).parent / "soniq_0.6.db"
GENRE_CACHE = Path(__file__).parent / "genre_cache.json"


def ingest(music_folder, dry_run=False):
    music_folder = str(Path(music_folder).resolve())
    print(f"\n  soniq ingest")
    print(f"  music:  {music_folder}")
    print(f"  db:     {DB_PATH}")
    print(f"  mode:   {'dry-run' if dry_run else 'write tags + DB'}\n")

    # Init
    conn = dbmod.init_db(DB_PATH)
    genre_init(str(GENRE_CACHE))

    # Scan for new tracks
    new_count = dbmod.scan_folder(conn, music_folder)
    recovered = dbmod.recover_stuck(conn)

    stats = dbmod.get_stats(conn)
    print(f"  library: {stats['total']} total, {stats['done']} done, {stats['pending']} pending")
    if new_count:
        print(f"  new:     {new_count} tracks found")
    if recovered:
        print(f"  recovered: {recovered} stuck tracks")

    if stats["pending"] == 0:
        print("\n  Nothing to process. All tracks are classified.\n")
        conn.close()
        return

    print(f"\n  Processing {stats['pending']} tracks...\n")
    dbmod.log_event(conn, "ingest_start", f"folder={music_folder}, pending={stats['pending']}")

    processed = 0
    errors = 0
    t_start = time.time()

    while True:
        track = dbmod.get_next_pending(conn)
        if not track:
            break

        track_id = track["id"]
        dbmod.mark_processing(conn, track_id)

        t0 = time.time()
        try:
            # 1. Librosa features
            features = extract_track_features(track["path"], max_duration=300)
            if not features:
                raise ValueError("Librosa extraction returned None")

            # 2. Classify
            cls = predict_all(features)

            # 3. Genre (non-blocking)
            genre = fetch_album_genre(track["artist"], track["album"])

            # 4. Build tag
            tag = build_tag(features, cls, genre=genre)
            elapsed = time.time() - t0

            # 5. Write tag to file
            if not dry_run:
                write_tag(track["path"], tag)

            # 6. Mark done in DB
            dbmod.mark_done(conn, track_id, tag, elapsed)
            processed += 1

            # Top 3 cls for display
            top = sorted(
                [(k, v) for k, v in cls.items() if isinstance(v, (int, float)) and k not in ("arousal", "valence")],
                key=lambda x: x[1], reverse=True
            )[:3]
            top_str = ", ".join(f"{k}={v:.0%}" for k, v in top)
            print(f"  ✓ [{track['artist']}] {track['title']} — {elapsed:.1f}s — {top_str}")

        except Exception as e:
            elapsed = time.time() - t0
            dbmod.mark_error(conn, track_id, str(e), elapsed)
            errors += 1
            print(f"  ✗ [{track['artist']}] {track['title']} — {e}")

    total_time = time.time() - t_start
    dbmod.log_event(conn, "ingest_done", f"processed={processed}, errors={errors}, time={total_time:.0f}s")
    conn.close()

    print(f"\n  Done. {processed} classified, {errors} errors, {total_time:.1f}s total.\n")


if __name__ == "__main__":
    folder = DEFAULT_MUSIC
    dry_run = False

    for arg in sys.argv[1:]:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            sys.exit(0)
        else:
            folder = arg

    if not Path(folder).is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    ingest(folder, dry_run=dry_run)
