"""Classifier: tonal — perceived tonal/melodic clarity in music.

Formula-based. Measures actual musical tonality: clear pitch, harmonic
structure, key definition. NOT the MusiCNN "tonal" which correlated with
percussive energy (measuring "produced" not "tonal").

Grounded in: Grekow 2018 (Key Strength, HPCP Entropy for tonality),
Peeters 2011 (spectral flatness as noise measure).

Components:
  - harmonic purity: high harm_fraction + low flatness (pitched, not noise)
  - key clarity: chroma_major_corr (Krumhansl profile match)
  - spectral focus: low spectral_entropy + high spectral_crest (concentrated energy)
  - pitch stability: low centroid_std (stable pitch center)

0 = atonal (noise, texture, no clear key or melody)
1 = tonal (clear pitch, strong key, melodic/harmonic structure)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict tonal quality from prepared features dict.

    Returns dict with tonal (0-1) and component scores.
    """
    # Harmonic purity: high harmonic fraction + low spectral flatness
    # (pitched instrument/voice vs noise/percussion)
    harmonic = _norm(prepared.get("harm_fraction", 0), "harm_fraction")
    low_noise = 1 - _norm(prepared.get("flatness", 0), "flatness")
    harmonic_purity = harmonic * 0.6 + low_noise * 0.4

    # Key clarity: how well the pitch content matches a key profile
    # chroma_major_corr is -1..1, normalize to 0..1
    chroma_corr = prepared.get("chroma_major_corr", 0)
    key_clarity = max(0.0, min(1.0, (chroma_corr + 1) / 2))

    # Spectral focus: concentrated energy (peaked, not diffuse)
    low_entropy = 1 - _norm(prepared.get("spectral_entropy", 0), "spectral_entropy")
    high_crest = _norm(prepared.get("spectral_crest", 0), "spectral_crest")
    spectral_focus = low_entropy * 0.5 + high_crest * 0.5

    # Pitch stability: stable spectral centroid = consistent pitch content
    pitch_stability = 1 - _norm(prepared.get("centroid_std", 0), "centroid_std")

    # Combine — harmonic purity is the backbone
    raw = (
        harmonic_purity * 0.35
        + key_clarity * 0.25
        + spectral_focus * 0.20
        + pitch_stability * 0.20
    )

    # Sigmoid stretch
    tonal = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    tonal = round(max(0.0, min(1.0, tonal)), 4)

    return {
        "tonal": tonal,
        "harmonic_purity": round(harmonic_purity, 4),
        "key_clarity": round(key_clarity, 4),
        "spectral_focus": round(spectral_focus, 4),
        "pitch_stability": round(pitch_stability, 4),
    }
