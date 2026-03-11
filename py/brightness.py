"""Brightness scoring — z-score against fixed corpus stats + sigmoid."""

import numpy as np

# Fixed reference stats from 681 tracks in music-player DB.
# These never change — brightness scores are comparable across runs.
CORPUS_STATS = {
    "centroid":  {"mean": 1374.4, "std": 528.7},
    "flatness":  {"mean": 0.005678, "std": 0.009095},
    "flux":      {"mean": 40.0, "std": 21.6},
    "zcr":       {"mean": 0.0523, "std": 0.0267},
    "mfcc1":     {"mean": 134.4, "std": 31.6},
}

BRIGHTNESS_WEIGHTS = {
    "centroid": 0.35,
    "mfcc1":    0.25,
    "flatness": 0.15,
    "flux":     0.15,
    "zcr":      0.10,
}


def compute_brightness(features):
    """Compute brightness score from librosa features.

    Z-score normalizes each feature against the 681-track corpus,
    then weighted combination through sigmoid → 0 (dark) to 1 (bright).
    """
    values = {
        "centroid": features.get("centroid_mean", 0),
        "flatness": features.get("flatness_mean", 0),
        "flux": features.get("spectral_flux", 0),
        "zcr": features.get("zcr_mean", 0),
        "mfcc1": features.get("mfcc_mean", [0, 0])[1] if len(features.get("mfcc_mean", [])) > 1 else 0,
    }

    score = 0
    for key, weight in BRIGHTNESS_WEIGHTS.items():
        stats = CORPUS_STATS[key]
        z = (values[key] - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
        score += z * weight

    return round(float(1 / (1 + np.exp(-score))), 4)
