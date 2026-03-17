"""Classifier: acoustic — perceived acoustic (vs electronic/synthetic) quality.

Formula-based. Components: harmonic dominance (high harm_fraction, low flatness),
low percussive energy, peaked spectrum (high spectral crest, low flatness),
timbral texture (bandwidth stability, mfcc variability).

0 = electronic/synthetic (flat spectrum, processed, percussive)
1 = acoustic (harmonic, peaked spectrum, natural timbre)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict acoustic quality from prepared features dict.

    Returns dict with acoustic (0-1) and component scores.
    """
    # Harmonic dominance: high harmonic fraction, low flatness
    harmonic_dom = (
        _norm(prepared.get("harm_fraction", 0), "harm_fraction") * 0.6
        + (1 - _norm(prepared.get("flatness", 0), "flatness")) * 0.4
    )

    # Low percussive: acoustic instruments have less percussive energy
    low_perc = 1 - _norm(prepared.get("perc_energy", 0), "perc_energy")

    # Peaked spectrum: acoustic instruments have clear spectral peaks
    peaked = (
        _norm(prepared.get("spectral_crest", 0), "spectral_crest") * 0.5
        + (1 - _norm(prepared.get("flatness", 0), "flatness")) * 0.5
    )

    # Timbral texture: acoustic instruments have varied, natural timbre
    texture = (
        _norm(prepared.get("mfcc_delta_var", 0), "mfcc_delta_var") * 0.5
        + (1 - _norm(prepared.get("mod_flatness", 0), "mod_flatness")) * 0.5
    )

    # Combine
    raw = (
        harmonic_dom * 0.35
        + low_perc * 0.25
        + peaked * 0.25
        + texture * 0.15
    )

    # Sigmoid stretch
    acoustic = 1 / (1 + math.exp(-6 * (raw - 0.5)))
    acoustic = round(max(0.0, min(1.0, acoustic)), 4)

    return {
        "acoustic": acoustic,
        "harmonic_dom": round(harmonic_dom, 4),
        "low_perc": round(low_perc, 4),
        "peaked": round(peaked, 4),
        "texture": round(texture, 4),
    }
