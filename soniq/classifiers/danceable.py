"""Classifier: danceable — perceived danceability of music.

Formula-based. Components: percussive (perc_energy — strongest correlate,
rho +0.72), beat lock (regularity + PLP stability), tempo zone (gaussian
centered at 120 BPM), groove (rhythm simplicity + beat strength).

0 = not danceable (arrhythmic, ambient, irregular)
1 = danceable (percussive, locked beat, ~120 BPM, groovy)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def _tempo_zone(tempo, center=120.0, width=30.0):
    """Gaussian preference for dance-friendly tempo range."""
    return math.exp(-0.5 * ((tempo - center) / width) ** 2)


def predict(prepared):
    """Predict danceability from prepared features dict.

    Returns dict with danceable (0-1) and component scores.
    """
    # Percussive: perc_energy is the MVP (strongest single correlate)
    percussive = _norm(prepared.get("perc_energy", 0), "perc_energy")

    # Beat lock: regular beat + stable pulse
    beat_lock = (
        _norm(prepared.get("beat_regularity", 0), "beat_regularity") * 0.5
        + _norm(prepared.get("plp_stability", 0), "plp_stability") * 0.5
    )

    # Tempo zone: gaussian preference for ~120 BPM
    tempo = prepared.get("tempo", 0)
    tempo_zone = _tempo_zone(tempo)

    # Groove: rhythm simplicity + beat strength
    groove = (
        (1 - _norm(prepared.get("rhythm_complexity", 0), "rhythm_complexity")) * 0.5
        + _norm(prepared.get("beat", 0), "beat") * 0.5
    )

    # Combine — percussive energy dominates
    raw = (
        percussive * 0.30
        + beat_lock * 0.30
        + tempo_zone * 0.20
        + groove * 0.20
    )

    # Sigmoid stretch
    danceable = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    danceable = round(max(0.0, min(1.0, danceable)), 4)

    return {
        "danceable": danceable,
        "percussive": round(percussive, 4),
        "beat_lock": round(beat_lock, 4),
        "tempo_zone": round(tempo_zone, 4),
        "groove": round(groove, 4),
    }
