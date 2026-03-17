"""Classifier: arousal — perceived urgency and activation level.

Formula-based. Components: urgency (transient density), drive (tempo + flux),
loudness (RMS energy), simplicity (inverse rhythm complexity).

0 = low arousal (calm, quiet, sparse)
1 = high arousal (urgent, driving, loud)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict arousal from prepared features dict.

    Returns dict with arousal (0-1) and component scores.
    """
    # Urgency: transient density + percussive hits
    urgency = (
        _norm(prepared.get("onset_rate", 0), "onset_rate") * 0.5
        + _norm(prepared.get("perc_energy", 0), "perc_energy") * 0.5
    )

    # Drive: tempo + spectral flux
    drive = (
        _norm(prepared.get("tempo", 0), "tempo") * 0.5
        + _norm(prepared.get("flux", 0), "flux") * 0.5
    )

    # Loudness: RMS energy
    loudness = _norm(prepared.get("rms_mean", 0), "rms_mean")

    # Simplicity: inverse rhythm complexity (simple = more driving)
    simplicity = 1 - _norm(prepared.get("rhythm_complexity", 0), "rhythm_complexity")

    # Combine
    raw = urgency * 0.35 + drive * 0.30 + loudness * 0.20 + simplicity * 0.15

    # Sigmoid stretch for distribution spread
    arousal = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    arousal = round(max(0.0, min(1.0, arousal)), 4)

    return {
        "arousal": arousal,
        "urgency": round(urgency, 4),
        "drive": round(drive, 4),
        "loudness": round(loudness, 4),
        "simplicity": round(simplicity, 4),
    }
