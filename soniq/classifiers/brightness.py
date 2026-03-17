"""Classifier: radiant/somber — perceived atmospheric brightness of music.

Formula-based, feature-only (no inter-classifier dependencies).

Radiant/somber is about PERCEPTION — euphoric, inviting, warm atmosphere vs
scary, ominous, void-like. NOT acoustic spectral brightness (that's brilliant/warm)
and NOT emotional valence (that's happy/sad).

Components:
  - fullness: acoustic body from mfcc0 (loudness proxy)
  - melodic: harmonic richness weighted by spectral centroid
  - atmosphere dimming: still + spectrally dark + minor key → somber pull

Key test tracks:
  - Plastikman (Kriket, Rekall): somber (~0.08-0.19) — dark, void-like
  - Daft Punk (Technologic, Get Lucky): radiant (~0.63-0.83) — euphoric
  - Sakamoto tama: moderate somber (~0.35)

0 = somber (dark, sparse, ominous, void-like)
1 = radiant (bright, full, euphoric, inviting)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict radiant/somber from prepared features dict.

    Returns dict with radiant (0-1), somber (0-1), and component scores.
    """
    mfcc0 = prepared.get("mfcc0", -200)
    harm_fraction = prepared.get("harm_fraction", 0.5)
    centroid = prepared.get("centroid", 1500)

    # --- Base acoustic radiance ---

    # Fullness: how much acoustic "body" — sigmoid of mfcc0
    fullness = 1 / (1 + math.exp(-0.015 * (mfcc0 + 150)))

    # Centroid warmth factor (caps at 1.0)
    centroid_factor = min(1.0, centroid / 1500)

    # Melodic richness: harmonic content × spectral position × fullness
    melodic = harm_fraction * centroid_factor * min(1.0, fullness * 2.5)

    # Base radiance from acoustic body
    acoustic = fullness * 0.6 + melodic * 0.4

    # --- Atmosphere dimming (replaces Phase 2 sad dependency) ---
    # Tracks that are still + spectrally dark + minor feel perceptually somber
    # This captures the same signal as the old "sad penalty" but from raw features

    stillness = (
        (1 - _norm(prepared.get("onset_rate", 0), "onset_rate")) * 0.5
        + (1 - _norm(prepared.get("perc_energy", 0), "perc_energy")) * 0.5
    )

    darkness = (
        (1 - _norm(prepared.get("centroid", 0), "centroid")) * 0.5
        + (1 - _norm(prepared.get("treble_ratio", 0), "treble_ratio")) * 0.5
    )

    minor_pull = 1.0 - prepared.get("mode", 0.5)

    atmosphere_dim = stillness * 0.30 + darkness * 0.30 + minor_pull * 0.40

    # Apply dimming to base radiance
    radiant = acoustic * (1 - 0.4 * atmosphere_dim) + 0.08

    radiant = round(max(0.0, min(1.0, radiant)), 4)
    somber = round(1 - radiant, 4)

    return {
        "radiant": radiant,
        "somber": somber,
        "fullness": round(fullness, 4),
        "melodic": round(melodic, 4),
        "atmosphere_dim": round(atmosphere_dim, 4),
    }
