"""Classifier: contemplative/restless — perceived reflective depth of music.

Formula-based (no ground truth). Distinct from energy/still — contemplative
requires emotional resonance (serene or melancholic) AND spaciousness, not
just absence of kinetic drive. A cold, mechanical quiet track is still but
not contemplative.

Components: spaciousness (sparse events, low spectral flux), emotional depth
(sad + relaxed signals), tonal richness (harmonic, not noisy), unhurried pace
(slow tempo, gentle beat).

0 = restless (urgent, busy, surface-level)
1 = contemplative (reflective, spacious, emotionally deep)
"""

import math

# Corpus statistics (N=1359, soniq_0.5.db 2026-03-12)
_STATS = {
    "onset_rate":       (3.9,   1.7),
    "flux":             (40.6,  25.5),
    "harm_fraction":    (0.769, 0.116),
    "flatness":         (0.007, 0.012),
    "tempo":            (120.0, 30.0),
    "beat":             (2.29,  0.81),
}


def _norm(val, mean, std):
    """Z-score through sigmoid — corpus-relative 0-1."""
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(results, prepared):
    """Predict contemplative/restless from classifier results + prepared features.

    Args:
        results: dict with classifier outputs (sad, relaxed, etc.)
        prepared: dict from _features.prepare()

    Returns dict with contemplative (0-1), restless (0-1), and component scores.
    """
    # 1. Spaciousness — few events, room to breathe
    sparse = 1 - _norm(prepared.get("onset_rate", 0), *_STATS["onset_rate"])
    low_flux = 1 - _norm(prepared.get("flux", 0), *_STATS["flux"])
    spacious = sparse * 0.6 + low_flux * 0.4

    # 2. Emotional depth — serene (relaxed) or melancholic (sad+relaxed)
    sad = results.get("sad", 0.5)
    relaxed = results.get("relaxed", 0.5)
    emotional = relaxed * 0.6 + sad * 0.4

    # 3. Tonal richness — sustained harmonic content, not noise
    harmonic = _norm(prepared.get("harm_fraction", 0), *_STATS["harm_fraction"])
    low_flat = 1 - _norm(prepared.get("flatness", 0), *_STATS["flatness"])
    tonal_depth = harmonic * 0.5 + low_flat * 0.5

    # 4. Unhurried — slow tempo, gentle beat
    slow = 1 - _norm(prepared.get("tempo", 120), *_STATS["tempo"])
    gentle_beat = 1 - _norm(prepared.get("beat", 0), *_STATS["beat"])
    unhurried = slow * 0.5 + gentle_beat * 0.5

    # Combine
    raw = (
        spacious * 0.30
        + emotional * 0.25
        + tonal_depth * 0.20
        + unhurried * 0.25
    )

    # Stretch distribution for better separation
    contemplative = 1 / (1 + math.exp(-6 * (raw - 0.5)))

    contemplative = round(max(0.0, min(1.0, contemplative)), 4)
    restless = round(1 - contemplative, 4)

    return {
        "contemplative": contemplative,
        "restless": restless,
        "spacious": round(spacious, 4),
        "emotional": round(emotional, 4),
        "tonal_depth": round(tonal_depth, 4),
        "unhurried": round(unhurried, 4),
    }
