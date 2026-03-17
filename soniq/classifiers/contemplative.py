"""Classifier: contemplative/restless — perceived reflective depth of music.

Formula-based, feature-only (no inter-classifier dependencies).

Distinct from energy/still — contemplative requires emotional resonance
(serene or melancholic) AND spaciousness, not just absence of kinetic drive.
A cold, mechanical quiet track is still but not contemplative.

Components: spaciousness (sparse events, low flux), emotional depth
(quiet + harmonic + minor coloring), tonal richness (harmonic, not noisy),
unhurried pace (slow tempo, gentle beat).

0 = restless (urgent, busy, surface-level)
1 = contemplative (reflective, spacious, emotionally deep)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict contemplative/restless from prepared features dict.

    Returns dict with contemplative (0-1), restless (0-1), and component scores.
    """
    # 1. Spaciousness — few events, room to breathe
    sparse = 1 - _norm(prepared.get("onset_rate", 0), "onset_rate")
    low_flux = 1 - _norm(prepared.get("flux", 0), "flux")
    spacious = sparse * 0.6 + low_flux * 0.4

    # 2. Emotional depth — replaces Phase 2 (relaxed * 0.6 + sad * 0.4)
    # Quiet + harmonic + low flux = serene; add minor coloring for melancholy
    quiet = 1 - _norm(prepared.get("rms_mean", 0), "rms_mean")
    harmonic = _norm(prepared.get("harm_fraction", 0), "harm_fraction")
    calm_flux = 1 - _norm(prepared.get("flux", 0), "flux")
    minor_presence = 1.0 - prepared.get("mode", 0.5)
    emotional = quiet * 0.30 + harmonic * 0.25 + calm_flux * 0.25 + minor_presence * 0.20

    # 3. Tonal richness — sustained harmonic content, not noise
    low_flat = 1 - _norm(prepared.get("flatness", 0), "flatness")
    tonal_depth = harmonic * 0.5 + low_flat * 0.5

    # 4. Unhurried — slow tempo, gentle beat
    slow = 1 - _norm(prepared.get("tempo", 120), "tempo")
    gentle_beat = 1 - _norm(prepared.get("beat", 0), "beat")
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
