"""Classifier: aggressive — perceived aggression/harshness in music.

Formula-based. Components: harshness (high centroid + flatness + treble),
intensity (high flux + onset rate + percussive energy),
loudness (high RMS energy).

0 = gentle (soft, warm, smooth)
1 = aggressive (harsh, intense, loud)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict aggression from prepared features dict.

    Returns dict with aggressive (0-1) and component scores.
    """
    # Harshness: high centroid + flatness (noise-like) + treble
    harshness = (
        _norm(prepared.get("centroid", 0), "centroid") * 0.35
        + _norm(prepared.get("flatness", 0), "flatness") * 0.35
        + _norm(prepared.get("treble_ratio", 0), "treble_ratio") * 0.30
    )

    # Intensity: high spectral flux + onset density + percussive energy
    intensity = (
        _norm(prepared.get("flux", 0), "flux") * 0.30
        + _norm(prepared.get("onset_rate", 0), "onset_rate") * 0.35
        + _norm(prepared.get("perc_energy", 0), "perc_energy") * 0.35
    )

    # Loudness: high RMS energy
    loudness = _norm(prepared.get("rms_mean", 0), "rms_mean")

    # Combine
    raw = harshness * 0.35 + intensity * 0.35 + loudness * 0.30

    # Sigmoid stretch
    aggressive = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    aggressive = round(max(0.0, min(1.0, aggressive)), 4)

    return {
        "aggressive": aggressive,
        "harshness": round(harshness, 4),
        "intensity": round(intensity, 4),
        "loudness": round(loudness, 4),
    }
