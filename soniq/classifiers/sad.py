"""Classifier: sad — perceived sadness/melancholy in music.

Formula-based. Grounded in Friberg 2014 (modality = strongest valence
predictor) and Grekow 2018 (tonal features critical for valence).

Key fix: uses (1 - chroma_major_corr) instead of binary (1 - mode).
Sakamoto "Merry Christmas Mr. Lawrence" has mode=1 (binary: major) but
chroma_major_corr=0.52 (moderate). The continuous measure captures the
tonal ambiguity that makes melancholic music feel sad.

Components:
  - tonal darkness: low/ambiguous major key correlation
  - stillness: low percussive energy, sparse onsets, slow feel
  - spectral darkness: low centroid, low treble
  - harmonic weight: sustained harmonics (legato, held notes)

0 = not sad (bright, clear major key, energetic)
1 = sad (still, dark, minor/ambiguous key, sustained)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict sadness from prepared features dict.

    Returns dict with sad (0-1) and component scores.
    """
    # Tonal darkness: low/ambiguous major key = sadder
    # chroma_major_corr -1..1, high = clear major = not sad
    chroma_corr = prepared.get("chroma_major_corr", 0)
    corr_norm = max(0.0, min(1.0, (chroma_corr + 1) / 2))
    tonal_dark = 1.0 - corr_norm

    # Stillness: low percussive energy + sparse onsets
    stillness = (
        (1 - _norm(prepared.get("perc_energy", 0), "perc_energy")) * 0.5
        + (1 - _norm(prepared.get("onset_rate", 0), "onset_rate")) * 0.5
    )

    # Spectral darkness: low centroid + low treble
    darkness = (
        (1 - _norm(prepared.get("centroid", 0), "centroid")) * 0.5
        + (1 - _norm(prepared.get("treble_ratio", 0), "treble_ratio")) * 0.5
    )

    # Harmonic weight: sustained harmonic content (held notes, legato)
    harmonic_weight = _norm(prepared.get("harm_fraction", 0), "harm_fraction")

    # Combine — tonal darkness gets highest weight (Friberg 2014)
    raw = (
        tonal_dark * 0.35
        + stillness * 0.25
        + darkness * 0.20
        + harmonic_weight * 0.20
    )

    # Sigmoid stretch
    sad = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    sad = round(max(0.0, min(1.0, sad)), 4)

    return {
        "sad": sad,
        "tonal_dark": round(tonal_dark, 4),
        "stillness": round(stillness, 4),
        "darkness": round(darkness, 4),
        "harmonic_weight": round(harmonic_weight, 4),
    }
