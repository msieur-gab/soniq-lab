"""Classifier: happy — perceived happiness/joy in music.

Formula-based. Uses chroma_major_corr (Krumhansl key strength) instead of
binary mode detection, per Friberg 2014: modality is the STRONGEST predictor
of valence/happiness (R²=0.87).

The key fix: binary mode(0/1) loses key strength. "bad guy" detects mode=1
(major) but chroma_major_corr=0.24 (weak). Real happy tracks like "1234"
have chroma_major_corr=0.74. This continuous measure correctly separates them.

Components:
  - tonal positivity: chroma_major_corr (strong major key = happy)
  - brightness: centroid + treble (bright spectrum = perceived positivity)
  - rhythmic joy: tempo + beat strength (upbeat feel)
  - liveliness: onset density (activity)

0 = not happy (dark, minor, subdued, ambiguous key)
1 = happy (bright, clear major key, lively, upbeat)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict happiness from prepared features dict.

    Returns dict with happy (0-1) and component scores.
    """
    # Tonal positivity: major key correlation (continuous, not binary)
    # Friberg 2014: modality = strongest happiness predictor
    chroma_corr = prepared.get("chroma_major_corr", 0)
    # chroma_major_corr is -1..1, map to 0..1
    tonal_pos = max(0.0, min(1.0, (chroma_corr + 1) / 2))

    # Brightness: high centroid + treble (perceived lightness)
    brightness = (
        _norm(prepared.get("centroid", 0), "centroid") * 0.5
        + _norm(prepared.get("treble_ratio", 0), "treble_ratio") * 0.5
    )

    # Rhythmic joy: upbeat tempo + strong beat
    rhythmic_joy = (
        _norm(prepared.get("tempo", 0), "tempo") * 0.5
        + _norm(prepared.get("beat", 0), "beat") * 0.5
    )

    # Liveliness: onset density (activity, not stillness)
    liveliness = _norm(prepared.get("onset_rate", 0), "onset_rate")

    # Combine — tonal positivity dominates (Friberg)
    raw = (
        tonal_pos * 0.35
        + brightness * 0.25
        + rhythmic_joy * 0.20
        + liveliness * 0.20
    )

    # Sigmoid stretch
    happy = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    happy = round(max(0.0, min(1.0, happy)), 4)

    return {
        "happy": happy,
        "tonal_pos": round(tonal_pos, 4),
        "brightness": round(brightness, 4),
        "rhythmic_joy": round(rhythmic_joy, 4),
        "liveliness": round(liveliness, 4),
    }
