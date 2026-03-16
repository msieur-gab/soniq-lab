"""Classifier: valence — perceived positivity/negativity of music.

Formula-based. Grounded in Friberg 2014 (modality = strongest predictor,
R²=0.87 with 4 features) and Grekow 2018 (tonal features critical).

Key insight: valence ceiling with audio-only is r=0.67 (Yang 2008) —
lyrics carry the rest. Within that ceiling, tonal/key features dominate.
Energy features (tempo, onset) contribute but must not override key signal.

Components:
  - tonal positivity: chroma_major_corr (dominant signal per literature)
  - spectral warmth: centroid position + treble balance
  - vitality: tempo + beat strength (moderate contribution)

0 = negative valence (dark, minor, subdued)
1 = positive valence (bright, major, lively)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict valence from prepared features dict.

    Returns dict with valence (0-1) and component scores.
    """
    # Tonal positivity: major key correlation (Friberg: strongest predictor)
    chroma_corr = prepared.get("chroma_major_corr", 0)
    tonal_pos = max(0.0, min(1.0, (chroma_corr + 1) / 2))

    # Spectral warmth: centroid + treble balance
    spectral = (
        _norm(prepared.get("centroid", 0), "centroid") * 0.5
        + _norm(prepared.get("treble_ratio", 0), "treble_ratio") * 0.5
    )

    # Vitality: tempo + beat (energy contribution, kept moderate)
    vitality = (
        _norm(prepared.get("tempo", 0), "tempo") * 0.5
        + _norm(prepared.get("beat", 0), "beat") * 0.5
    )

    # Combine — tonal positivity dominates per literature
    raw = tonal_pos * 0.45 + spectral * 0.30 + vitality * 0.25

    # Sigmoid stretch
    valence = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    valence = round(max(0.0, min(1.0, valence)), 4)

    return {
        "valence": valence,
        "tonal_pos": round(tonal_pos, 4),
        "spectral": round(spectral, 4),
        "vitality": round(vitality, 4),
    }
