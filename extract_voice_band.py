#!/home/msieur-gab/soniq-lab/venv/bin/python3
"""Extract voice_band_ratio for all tracks in a DB.

Computes energy in the 300-3000 Hz voice band as a fraction of total energy.
Uses a single 10s segment from track center for speed (~1-2s/track).

This is the classic VAD (Voice Activity Detection) feature: human voice
concentrates energy in 300-3000 Hz. Instruments and electronic sounds
often have energy below 300 Hz or above 3000 Hz.

Usage:
    python3 extract_voice_band.py                    # soniq_0.6.db
    python3 extract_voice_band.py --db soniq_0.5.db  # specific DB
"""

import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

DB_DEFAULT = Path(__file__).parent / "soniq_0.6.db"


def extract_voice_band(filepath):
    """Extract voice_band_ratio from audio file.

    Loads 10s from track center, computes STFT, returns energy ratio
    in 300-3000 Hz band vs total.
    """
    import librosa

    try:
        duration = librosa.get_duration(path=filepath)
    except Exception:
        return None

    if duration < 3:
        return None

    # Load 10s from center of track
    center = duration * 0.5
    offset = max(0, center - 5)
    load_dur = min(10, duration - offset)

    try:
        y, sr = librosa.load(filepath, sr=22050, offset=offset,
                             duration=load_dur, mono=True)
    except Exception:
        return None

    if len(y) < sr:
        return None

    # STFT
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Voice band: 300-3000 Hz (covers F0 + formants F1-F3)
    voice_mask = (freqs >= 300) & (freqs < 3000)
    total = S_power.sum()

    if total < 1e-12:
        return 0.0

    voice_energy = S_power[voice_mask].sum()
    return round(float(voice_energy / total), 4)


def process_db(db_path):
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"  ERROR: {db_path} not found")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get tracks that need the feature
    cur.execute("SELECT id, path, scalars_json FROM tracks WHERE status='done' AND scalars_json IS NOT NULL")
    rows = cur.fetchall()

    need = []
    for row in rows:
        scalars = json.loads(row["scalars_json"])
        if "voice_band_ratio" not in scalars:
            need.append(row)

    total = len(rows)
    print(f"\n  DB: {db_path.name}")
    print(f"  Total done: {total}")
    print(f"  Need voice_band_ratio: {len(need)}")

    if not need:
        print("  Nothing to do.")
        conn.close()
        return

    print(f"  Extracting...\n")
    t_start = time.time()
    done = 0
    errors = 0
    missing = 0

    for i, row in enumerate(need):
        filepath = row["path"]

        if not os.path.isfile(filepath):
            missing += 1
            continue

        ratio = extract_voice_band(filepath)

        if ratio is None:
            errors += 1
            continue

        # Re-read current scalars from DB and merge
        cur.execute("SELECT scalars_json FROM tracks WHERE id = ?", (row["id"],))
        current_sj = cur.fetchone()[0]
        scalars = json.loads(current_sj)
        scalars["voice_band_ratio"] = ratio

        cur.execute("UPDATE tracks SET scalars_json = ? WHERE id = ?",
                    (json.dumps(scalars, separators=(",", ":")), row["id"]))
        done += 1

        if (i + 1) % 50 == 0:
            conn.commit()
            elapsed = time.time() - t_start
            rate = done / (elapsed + 1e-8)
            remaining = (len(need) - i - 1) / (rate + 1e-8)
            print(f"  [{i + 1}/{len(need)}] {done} done, {errors} err — "
                  f"{rate:.1f} t/s — ETA {remaining / 60:.0f}m")

    conn.commit()
    elapsed = time.time() - t_start
    print(f"\n  Done: {done} extracted, {errors} errors, {missing} missing")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Verify
    cur.execute("SELECT COUNT(*) FROM tracks WHERE status='done' AND scalars_json LIKE '%voice_band_ratio%'")
    count = cur.fetchone()[0]
    print(f"  Verified: {count}/{total} tracks now have voice_band_ratio")

    conn.close()


if __name__ == "__main__":
    db_path = DB_DEFAULT
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--db" and i < len(sys.argv):
            db_path = Path(sys.argv[i + 1])

    process_db(db_path)
