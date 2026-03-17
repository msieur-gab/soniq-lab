"""Classifier: relaxed — perceived calm/restful quality in music.

Formula-based. Components: calm (low transient density and flux),
gentle (low centroid, low flatness), quiet (low RMS), harmonic (high
harmonic fraction).

0 = not relaxed (aggressive, loud, busy)
1 = relaxed (calm, gentle, quiet, harmonic)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict relaxed quality from prepared features dict.

    Returns dict with relaxed (0-1) and component scores.
    """
    # Calm: low transient density + low percussive energy + low flux
    calm = (
        (1 - _norm(prepared.get("onset_rate", 0), "onset_rate")) * 0.35
        + (1 - _norm(prepared.get("perc_energy", 0), "perc_energy")) * 0.35
        + (1 - _norm(prepared.get("flux", 0), "flux")) * 0.30
    )

    # Gentle: low spectral centroid + low flatness (warm, not harsh)
    gentle = (
        (1 - _norm(prepared.get("centroid", 0), "centroid")) * 0.5
        + (1 - _norm(prepared.get("flatness", 0), "flatness")) * 0.5
    )

    # Quiet: low RMS energy
    quiet = 1 - _norm(prepared.get("rms_mean", 0), "rms_mean")

    # Harmonic: high harmonic fraction (tonal, not noisy)
    harmonic = _norm(prepared.get("harm_fraction", 0), "harm_fraction")

    # Combine
    raw = calm * 0.40 + gentle * 0.25 + quiet * 0.20 + harmonic * 0.15

    # Sigmoid stretch
    relaxed = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    relaxed = round(max(0.0, min(1.0, relaxed)), 4)

    return {
        "relaxed": relaxed,
        "calm": round(calm, 4),
        "gentle": round(gentle, 4),
        "quiet": round(quiet, 4),
        "harmonic": round(harmonic, 4),
    }
