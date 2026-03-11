#!/usr/bin/env python3
"""
Timbre proxy investigation — can a DSP feature vector predict neural bright/dark?

Builds a 34-dim timbre vector per track using librosa:
  - centroid (mean, std)
  - rolloff (mean, std)
  - bandwidth (mean, std)
  - flatness (mean)
  - flux (mean, std)
  - MFCCs 1-13 (mean, std each = 26 dims)

Pulls neural timbre scores from pipeline.db (read-only).
Outputs: tests/timbre-proxy-results.json
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import librosa
import numpy as np

DB_PATH = Path(__file__).parent.parent / "pipeline.db"
OUTPUT = Path(__file__).parent / "timbre-proxy-results.json"


def get_tracks_with_timbre(one_per_artist=False):
    """Pull tracks that have neural timbre scores from the DB (read-only)."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    if one_per_artist:
        rows = db.execute("""
            SELECT path, artist, title, cls_json FROM tracks
            WHERE status='done' AND cls_json IS NOT NULL
            GROUP BY artist
            ORDER BY artist
        """).fetchall()
    else:
        rows = db.execute(
            "SELECT path, artist, title, cls_json FROM tracks WHERE status='done' AND cls_json IS NOT NULL"
        ).fetchall()
    db.close()

    tracks = []
    for r in rows:
        cls = json.loads(r["cls_json"])
        bright = cls.get("bright")
        dark = cls.get("dark")
        if bright is None or dark is None:
            continue
        tracks.append({
            "path": r["path"],
            "artist": r["artist"],
            "title": r["title"],
            "neural_bright": bright,
            "neural_dark": dark,
        })
    return tracks


FEATURE_NAMES = (
    ["centroid_mean", "centroid_std",
     "rolloff_mean", "rolloff_std",
     "bandwidth_mean", "bandwidth_std",
     "flatness_mean",
     "flux_mean", "flux_std"]
    + [f"mfcc_{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]
)


def timbre_vector(path):
    """Build 34-dim timbre vector using librosa."""
    y, sr = librosa.load(path, sr=22050)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def ms(x):
        return [float(np.mean(x)), float(np.std(x))]

    vec = (
        ms(centroid) + ms(rolloff) + ms(bandwidth)
        + [float(np.mean(flatness))]
        + ms(flux)
        + [val for mfcc in mfccs for val in ms(mfcc)]
    )
    return np.array(vec)


def pearson(x, y):
    mx, my = np.mean(x), np.mean(y)
    cov = np.mean((x - mx) * (y - my))
    sx, sy = np.std(x), np.std(y)
    return cov / (sx * sy) if sx * sy > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="One track per artist")
    args = parser.parse_args()

    tracks = get_tracks_with_timbre(one_per_artist=args.sample)
    print(f"Found {len(tracks)} tracks with neural timbre scores")

    if not tracks:
        print("No tracks found!")
        return

    all_vectors = []
    neural_bright = []
    track_info = []
    errors = 0

    t0 = time.time()
    for i, t in enumerate(tracks):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1) if i > 0 else 0
            eta = rate * (len(tracks) - i) / 60
            print(f"  [{i+1}/{len(tracks)}] {t['artist']} — {t['title']}  (ETA ~{eta:.1f}m)")

        try:
            vec = timbre_vector(t["path"])
            all_vectors.append(vec)
            neural_bright.append(t["neural_bright"])
            track_info.append(t)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR: {t['title']}: {e}")

    elapsed = time.time() - t0
    print(f"\nExtracted {len(FEATURE_NAMES)}-dim vectors for {len(all_vectors)} tracks in {elapsed:.1f}s ({errors} errors)")

    X = np.array(all_vectors)
    y = np.array(neural_bright)

    # Individual correlations
    print(f"\n{'Feature':<20} {'Pearson r':>10} {'Direction':>12}")
    print("-" * 45)
    correlations = {}
    for j, name in enumerate(FEATURE_NAMES):
        r = pearson(X[:, j], y)
        direction = "bright↑" if r > 0 else "dark↑"
        print(f"  {name:<20} {r:>10.4f} {direction:>10}")
        correlations[name] = round(r, 4)

    # Linear regression with all features
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    X_aug = np.column_stack([np.ones(len(X_norm)), X_norm])

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\nLinear regression (all {len(FEATURE_NAMES)} features):")
        print(f"  R² = {r_squared:.4f}")

        weights = list(zip(FEATURE_NAMES, beta[1:]))
        weights.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top 10 features by weight:")
        print(f"  {'Feature':<20} {'Weight':>10}")
        print(f"  {'-'*32}")
        for name, w in weights[:10]:
            print(f"  {name:<20} {w:>10.6f}")

    except Exception as e:
        r_squared = None
        weights = []
        print(f"\nRegression failed: {e}")

    # Save
    results = {
        "n_tracks": len(all_vectors),
        "n_features": len(FEATURE_NAMES),
        "extraction_time_s": round(elapsed, 1),
        "neural_bright_range": [round(float(y.min()), 4), round(float(y.max()), 4)],
        "neural_bright_spread": round(float(y.max() - y.min()), 4),
        "correlations": correlations,
        "regression_r_squared": round(r_squared, 4) if r_squared else None,
        "regression_weights": {n: round(float(w), 6) for n, w in weights} if weights else {},
        "feature_names": FEATURE_NAMES,
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
