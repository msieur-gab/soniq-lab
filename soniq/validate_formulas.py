"""Validate formula-based classifiers against soniq_0.5_reference.db.

Reads stored librosa features, runs new formula classifiers, and compares
against old Ridge regression scores stored in cls_json.

Checks:
1. Spearman rank correlation between old and new scores
2. Distribution spread (should use >60% of 0-1 range)
3. Spot checks on known tracks
"""

import json
import math
import os
import sqlite3
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soniq.classifiers import _features, predict_all


DB_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "soniq_0.5_reference.db")
DB_PATH = DB_DEFAULT

# Classifiers that were Ridge/logistic in v0.5
RIDGE_CLASSIFIERS = [
    "arousal", "valence", "acoustic", "tonal",
    "sad", "relaxed", "happy",
    "aggressive", "danceable", "party", "instrumental",
]

# Expected quality tiers for correlation targets
GOOD_CLASSIFIERS = {"arousal", "valence", "danceable", "sad", "tonal", "relaxed"}
BROKEN_CLASSIFIERS = {"party", "instrumental", "aggressive"}


def spearman_rho(x, y):
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d_sq_sum) / (n * (n * n - 1))


def load_tracks():
    """Load tracks with both scalars and cls from reference DB."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT path, artist, title, scalars_json, vectors_json, cls_json
        FROM tracks
        WHERE scalars_json IS NOT NULL
          AND cls_json IS NOT NULL
          AND status = 'done'
    """)
    tracks = []
    for path, artist, title, sj, vj, cj in cur.fetchall():
        try:
            scalars = json.loads(sj)
            vectors = json.loads(vj) if vj else {}
            old_cls = json.loads(cj)
        except (json.JSONDecodeError, TypeError):
            continue

        # Reconstruct librosa_features from stored tag scalars + vectors
        # The DB stores short names; we need to reverse-map for _features.prepare()
        librosa_feats = _reconstruct_librosa_features(scalars, vectors)
        tracks.append({
            "path": path,
            "artist": artist,
            "title": title,
            "librosa_features": librosa_feats,
            "old_cls": old_cls,
        })
    conn.close()
    return tracks


def _reconstruct_librosa_features(scalars, vectors):
    """Reconstruct librosa_features dict from stored tag scalars/vectors.

    The DB stores short names (same as _features.py output). We need to
    provide the long names that extract_track_features() would return,
    since _features.prepare() maps from long to short.
    """
    # Reverse mapping from short → long (based on tags.py SCALAR_SHORT)
    short_to_long = {
        "duration": "duration", "tempo": "tempo", "key": "key", "mode": "mode",
        "rms_mean": "rms_mean", "rms_max": "rms_max", "rms_var": "rms_variance",
        "dyn_range": "dynamic_range", "centroid": "centroid_mean",
        "centroid_std": "centroid_std",
        "rolloff": "rolloff_mean", "rolloff_std": "rolloff_std",
        "bandwidth": "bandwidth_mean", "bandwidth_std": "bandwidth_std",
        "flatness": "flatness_mean", "flux": "spectral_flux", "flux_std": "flux_std",
        "onset": "onset_strength", "beat": "beat_strength",
        "vocal": "vocal_proxy", "zcr": "zcr_mean",
        "low_energy_rate": "low_energy_rate",
        "energy_skew": "energy_skew", "energy_kurtosis": "energy_kurtosis",
        "bass_ratio": "bass_ratio", "mid_ratio": "mid_ratio",
        "treble_ratio": "treble_ratio", "bass_mid_ratio": "bass_mid_ratio",
        "spectral_skew": "spectral_skew", "spectral_kurtosis": "spectral_kurtosis",
        "spectral_entropy": "spectral_entropy", "spectral_crest": "spectral_crest",
        "mfcc_delta_var": "mfcc_delta_var", "mfcc_delta2_var": "mfcc_delta2_var",
        "mod_flatness": "mod_flatness", "mod_crest": "mod_crest",
        "mod_centroid": "mod_centroid",
        "harm_energy": "harm_energy", "perc_energy": "perc_energy",
        "harm_perc_ratio": "harm_perc_ratio", "harm_fraction": "harm_fraction",
        "beat_regularity": "beat_regularity", "rhythm_complexity": "rhythm_complexity",
        "plp_mean": "plp_mean", "plp_stability": "plp_stability",
        "onset_rate": "onset_rate",
        "voice_band_ratio": "voice_band_ratio",
        # v0.6 pYIN additions
        "voiced_ratio": "voiced_ratio",
        "voiced_conf": "voiced_confidence",
        "f0_mean": "f0_mean",
        "f0_std": "f0_std",
        "chroma_major_corr": "chroma_major_corr",
    }

    result = {}
    for short_key, val in scalars.items():
        long_key = short_to_long.get(short_key, short_key)
        result[long_key] = val

    # Vectors — map from tag short names to librosa long names
    vec_map = {
        "mfcc_m": "mfcc_mean", "mfcc_s": "mfcc_std",
        "contrast": "contrast_mean", "chroma": "chroma_mean",
        "tonnetz": "tonnetz_mean",
        "mfcc_d": "mfcc_delta_mean", "mfcc_d2": "mfcc_delta2_mean",
    }
    for short_key, long_key in vec_map.items():
        if short_key in vectors:
            result[long_key] = vectors[short_key]

    # pYIN features won't exist in old DB — defaults will be used
    # chroma_major_corr also won't exist — will be computed if chroma available

    return result


def validate():
    """Run validation and print results."""
    print(f"Loading tracks from {DB_PATH}...")
    tracks = load_tracks()
    print(f"Loaded {len(tracks)} tracks with scalars + classifications\n")

    if not tracks:
        print("ERROR: No tracks found!")
        return

    # Collect old vs new scores per classifier
    old_scores = {c: [] for c in RIDGE_CLASSIFIERS}
    new_scores = {c: [] for c in RIDGE_CLASSIFIERS}

    for track in tracks:
        new_cls = predict_all(track["librosa_features"])
        old_cls = track["old_cls"]

        for cls_name in RIDGE_CLASSIFIERS:
            if cls_name in old_cls and cls_name in new_cls:
                old_val = old_cls[cls_name]
                new_val = new_cls[cls_name]
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    old_scores[cls_name].append(float(old_val))
                    new_scores[cls_name].append(float(new_val))

    # Report per classifier
    print("=" * 72)
    print(f"{'Classifier':<15} {'N':>5} {'Rho':>7} {'Old range':>12} "
          f"{'New range':>12} {'New spread':>10} {'Status':>8}")
    print("-" * 72)

    for cls_name in RIDGE_CLASSIFIERS:
        old = old_scores[cls_name]
        new = new_scores[cls_name]
        n = len(old)

        if n < 10:
            print(f"{cls_name:<15} {n:>5}  INSUFFICIENT DATA")
            continue

        rho = spearman_rho(old, new)

        old_min, old_max = min(old), max(old)
        new_min, new_max = min(new), max(new)
        new_spread = new_max - new_min

        # Determine status
        if cls_name in GOOD_CLASSIFIERS:
            target_rho = 0.70
        elif cls_name in BROKEN_CLASSIFIERS:
            target_rho = 0.50
        else:
            target_rho = 0.60

        spread_ok = new_spread >= 0.60
        rho_ok = rho >= target_rho

        if rho_ok and spread_ok:
            status = "OK"
        elif rho_ok:
            status = "NARROW"
        elif spread_ok:
            status = "LOW_RHO"
        else:
            status = "FAIL"

        print(f"{cls_name:<15} {n:>5} {rho:>7.3f} "
              f"{old_min:>5.2f}-{old_max:>5.2f} "
              f"{new_min:>5.2f}-{new_max:>5.2f} "
              f"{new_spread:>10.3f} {status:>8}")

    print("=" * 72)

    # Distribution analysis
    print("\nDistribution analysis (new scores):")
    print("-" * 50)
    for cls_name in RIDGE_CLASSIFIERS:
        new = new_scores[cls_name]
        if len(new) < 10:
            continue
        mean = sum(new) / len(new)
        std = math.sqrt(sum((v - mean) ** 2 for v in new) / len(new))
        p5 = sorted(new)[int(len(new) * 0.05)]
        p95 = sorted(new)[int(len(new) * 0.95)]
        print(f"  {cls_name:<15} mean={mean:.3f}  std={std:.3f}  "
              f"p5={p5:.3f}  p95={p95:.3f}")

    # Spot checks
    print("\nSpot checks:")
    print("-" * 50)
    spot_checks = {
        "Around the World": ["danceable", "party", "energetic"],
        "Prayer": ["relaxed", "sad", "instrumental"],
    }

    for track in tracks:
        title = track.get("title", "")
        for keyword, expected_high in spot_checks.items():
            if keyword.lower() in (title or "").lower():
                new_cls = predict_all(track["librosa_features"])
                print(f"\n  {track['artist']} - {title}")
                for key in sorted(new_cls.keys()):
                    if key.startswith("_"):
                        continue
                    val = new_cls[key]
                    if isinstance(val, float):
                        marker = " <<<" if key in expected_high else ""
                        print(f"    {key:<18} {val:.3f}{marker}")


if __name__ == "__main__":
    # Support --db flag
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--db" and i < len(sys.argv):
            DB_PATH = os.path.abspath(sys.argv[i + 1])
    validate()
