"""Classifier: party — perceived party/club energy of music.

Formula-based, feature-only (no inter-classifier dependencies).

Captures what makes music feel like it belongs at a party: strong percussive
drive, locked rhythm, dance-friendly tempo, loud, bright spectral character.

Components: percussive drive, rhythmic lock (regularity + pulse stability),
activity (onset density), loudness, spectral brightness, tempo zone.

0 = not party (quiet, slow, dark, ambient)
1 = party (percussive, locked groove, loud, bright, ~120 BPM)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def _tempo_zone(tempo, center=120.0, width=30.0):
    """Gaussian preference for party-friendly tempo range."""
    return math.exp(-0.5 * ((tempo - center) / width) ** 2)


def predict(prepared):
    """Predict party from prepared features dict.

    Returns dict with party (0-1) and component scores.
    """
    # Percussive drive — the core party signal
    percussive = _norm(prepared.get("perc_energy", 0), "perc_energy")

    # Rhythmic lock — regular beat + stable pulse + strong beat
    rhythmic = (
        _norm(prepared.get("beat_regularity", 0), "beat_regularity") * 0.4
        + _norm(prepared.get("plp_stability", 0), "plp_stability") * 0.3
        + _norm(prepared.get("beat", 0), "beat") * 0.3
    )

    # Activity — onset density (urgency)
    activity = _norm(prepared.get("onset_rate", 0), "onset_rate")

    # Loudness
    loudness = _norm(prepared.get("rms_mean", 0), "rms_mean")

    # Spectral brightness — bright sounds feel more party
    brightness = _norm(prepared.get("centroid", 0), "centroid")

    # Tempo zone — gaussian preference for ~120 BPM
    tempo = prepared.get("tempo", 0)
    tempo_zone = _tempo_zone(tempo)

    # Combine
    raw = (
        percussive * 0.25
        + rhythmic * 0.25
        + activity * 0.15
        + loudness * 0.15
        + brightness * 0.10
        + tempo_zone * 0.10
    )

    # Sigmoid stretch
    party = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    party = round(max(0.0, min(1.0, party)), 4)

    return {
        "party": party,
        "percussive": round(percussive, 4),
        "rhythmic": round(rhythmic, 4),
        "activity": round(activity, 4),
        "loudness": round(loudness, 4),
        "brightness": round(brightness, 4),
        "tempo_zone": round(tempo_zone, 4),
    }
