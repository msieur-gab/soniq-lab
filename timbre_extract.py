#!/usr/bin/env python3
"""
Full timbre vector extraction using librosa.

Computes the 34-dim timbre vector per Gab's spec:
  centroid mean+std, rolloff mean+std, bandwidth mean+std,
  flatness mean, flux mean+std, mfcc 1-13 mean+std

Reads: m4a files (audio only, no tags modified)
Writes: output/timbre_vectors.json (new file, nothing else touched)

Does NOT modify any DB or m4a tag.
"""

import json
import sqlite3
import time
from pathlib import Path

import librosa
import numpy as np

MUSIC_DB = Path.home() / "music-player" / ".data" / ".audio_features.db"
PIPELINE_DB = Path(__file__).parent / "pipeline.db"
OUTPUT = Path(__file__).parent / "output" / "timbre_vectors.json"

# Same test set as timbre_test.py
TEST_ARTISTS = [
    "Ballaké Sissoko",
    "Four Tet",
    "GoGo Penguin",
    "Jay-Jay Johanson",
    "Hidden Orchestra",
    "Mammal Hands",
    "Grandbrothers",
    "Flying Lotus, S.Ellison",
    "Esbjörn Svensson Trio",
    "The Cinematic Orchestra",
    "Portico Quartet",
    "Jaga Jazzist",
    "Matthew Halsall",
]


def get_test_tracks(db, artists, limit_per_artist=3):
    """Get file paths for test tracks from the librosa DB."""
    tracks = []
    for artist in artists:
        rows = db.execute("""
            SELECT artist, title, album, file
            FROM tracks
            WHERE artist LIKE ?
            ORDER BY title
            LIMIT ?
        """, (f"{artist}%", limit_per_artist)).fetchall()
        tracks.extend(rows)
    return tracks


def extract_timbre_vector(audio_path):
    """
    Extract the full 34-dim timbre vector from an audio file.

    Vector layout:
      [0-1]   centroid mean, std
      [2-3]   rolloff mean, std
      [4-5]   bandwidth mean, std
      [6]     flatness mean
      [7-8]   flux mean, std
      [9-34]  mfcc 1-13 mean, std (26 dims)
    Total: 35 dimensions
    """
    y, sr = librosa.load(audio_path, sr=22050)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Spectral features (per-frame, then aggregate)
    centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def ms(x):
        return [float(np.mean(x)), float(np.std(x))]

    # Build the vector
    vec = (
        ms(centroid) +
        ms(rolloff) +
        ms(bandwidth) +
        [float(np.mean(flatness))] +
        ms(flux) +
        [val for mfcc_row in mfccs for val in ms(mfcc_row)]
    )

    # Also return named breakdown for human inspection
    breakdown = {
        "centroid_mean": round(float(np.mean(centroid)), 2),
        "centroid_std": round(float(np.std(centroid)), 2),
        "rolloff_mean": round(float(np.mean(rolloff)), 2),
        "rolloff_std": round(float(np.std(rolloff)), 2),
        "bandwidth_mean": round(float(np.mean(bandwidth)), 2),
        "bandwidth_std": round(float(np.std(bandwidth)), 2),
        "flatness_mean": round(float(np.mean(flatness)), 6),
        "flux_mean": round(float(np.mean(flux)), 4),
        "flux_std": round(float(np.std(flux)), 4),
    }
    for i in range(13):
        breakdown[f"mfcc{i+1}_mean"] = round(float(np.mean(mfccs[i])), 2)
        breakdown[f"mfcc{i+1}_std"] = round(float(np.std(mfccs[i])), 2)

    return vec, breakdown


def get_effnet_timbre(db, artist, title):
    """Get EffNet bright/dark from pipeline DB."""
    row = db.execute("""
        SELECT cls_json FROM tracks
        WHERE artist LIKE ? AND title LIKE ?
        AND status='done' LIMIT 1
    """, (f"{artist}%", f"%{title}%")).fetchone()
    if row and row[0]:
        cls = json.loads(row[0])
        return {"bright": cls.get("bright"), "dark": cls.get("dark")}
    return None


def compute_brightness_score(breakdown):
    """
    Derive a brightness score (0=dark, 1=bright) from the full timbre vector.

    Uses z-score normalization against the batch, so this score is
    relative to the test set — not absolute.
    """
    # We'll compute this after collecting all tracks (needs corpus stats)
    pass


def main():
    librosa_db = sqlite3.connect(str(MUSIC_DB))
    pipeline_db = sqlite3.connect(str(PIPELINE_DB))

    test_tracks = get_test_tracks(librosa_db, TEST_ARTISTS, limit_per_artist=3)
    print(f"Extracting timbre vectors for {len(test_tracks)} tracks...\n")

    results = []
    vectors = []

    for i, (artist, title, album, filepath) in enumerate(test_tracks):
        # Resolve path — file column might be relative
        audio_path = filepath
        if not Path(audio_path).exists():
            audio_path = str(Path.home() / "music-player" / "music" / filepath)
        if not Path(audio_path).exists():
            print(f"  [{i+1}/{len(test_tracks)}] SKIP — file not found: {filepath}")
            continue

        t0 = time.time()
        try:
            vec, breakdown = extract_timbre_vector(audio_path)
            elapsed = time.time() - t0

            effnet = get_effnet_timbre(pipeline_db, artist, title)

            entry = {
                "artist": artist,
                "title": title,
                "album": album,
                "timbre_vector": [round(v, 6) for v in vec],
                "breakdown": breakdown,
                "effnet": effnet,
                "extraction_time_s": round(elapsed, 2),
            }
            results.append(entry)
            vectors.append(vec)

            print(f"  [{i+1}/{len(test_tracks)}] {artist:35s} {title:30s} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1}/{len(test_tracks)}] ERROR {artist} — {title}: {e}")

    if not results:
        print("No tracks processed!")
        return

    # === Z-score normalize across the batch and derive brightness ===
    vecs = np.array(vectors)
    means = np.mean(vecs, axis=0)
    stds = np.std(vecs, axis=0)
    stds[stds == 0] = 1  # avoid division by zero

    normalized = (vecs - means) / stds

    # Brightness score: weighted combination of normalized features
    # Indices: 0=centroid_mean, 1=centroid_std, 2=rolloff_mean, 4=bandwidth_mean, 6=flatness_mean
    weights = {
        0: 0.30,   # centroid_mean — where spectral mass sits
        1: 0.10,   # centroid_std — brightness variability
        2: 0.20,   # rolloff_mean — where 85% of energy lives
        4: 0.15,   # bandwidth_mean — spectral spread
        6: 0.10,   # flatness_mean — noise vs tone
        7: 0.05,   # flux_mean — spectral dynamics
        9: 0.10,   # mfcc1_mean — spectral slope
    }

    for i, result in enumerate(results):
        score = sum(normalized[i, idx] * w for idx, w in weights.items())
        # Sigmoid to map to 0-1
        brightness = 1 / (1 + np.exp(-score))
        result["brightness_score"] = round(float(brightness), 4)
        result["brightness_label"] = "bright" if brightness > 0.5 else "dark"

    # Save
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} tracks → {OUTPUT}")

    # === Summary ===
    print("\n=== Brightness scores (z-norm + sigmoid) ===")
    print(f"  {'Artist':40s} {'Title':30s} {'Score':>7s} {'Label':>7s}  {'EffNet':>7s}")
    print(f"  {'-'*40} {'-'*30} {'-'*7} {'-'*7}  {'-'*7}")

    sorted_results = sorted(results, key=lambda r: r["brightness_score"])
    for r in sorted_results:
        effnet_str = f"{r['effnet']['bright']:.4f}" if r["effnet"] else "   —"
        print(f"  {r['artist']:40s} {r['title']:30s} {r['brightness_score']:7.4f} {r['brightness_label']:>7s}  {effnet_str}")

    # Spread comparison
    scores = [r["brightness_score"] for r in results]
    effnet_scores = [r["effnet"]["bright"] for r in results if r["effnet"]]
    print(f"\n=== Spread ===")
    print(f"  Full vector brightness: min={min(scores):.4f} max={max(scores):.4f} spread={max(scores)-min(scores):.4f}")
    if effnet_scores:
        print(f"  EffNet bright:          min={min(effnet_scores):.4f} max={max(effnet_scores):.4f} spread={max(effnet_scores)-min(effnet_scores):.4f}")

    # Sanity check
    print(f"\n=== Sanity check ===")
    for r in sorted_results:
        if "Ballaké" in r["artist"] or "Four Tet" in r["artist"]:
            print(f"  {r['artist']:40s} {r['title']:30s} {r['brightness_score']:.4f} ({r['brightness_label']})")


if __name__ == "__main__":
    main()
