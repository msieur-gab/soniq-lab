"""Classifier: hypnotic/varied — perceived repetitive trance quality of music.

Formula-based. Uses shared corpus stats for z-score normalization.
Two-path approach:
  - Rhythmic hypnotic: locked pulse + consistent energy (techno loops, process music)
  - Timbral hypnotic: stable timbre + minimal spectral change (drones, ambient)
The stronger path dominates via soft-max blending. When both paths are
strong and close, reports "both" (e.g. Glass arpeggios = rhythmic + timbral).

0 = varied (dynamic, evolving, surprising)
1 = hypnotic (repetitive, trance-inducing, locked-in)
"""

import math
from ._corpus_stats import STATS


def _norm(val, key):
    """Z-score through sigmoid — corpus-relative 0-1."""
    mean, std = STATS[key]
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict hypnotic/varied from prepared features dict.

    Returns dict with hypnotic (0-1), varied (0-1), dominant path,
    and path scores.
    """
    # === PATH 1: RHYTHMIC HYPNOTIC ===
    # Locked pulse + consistent energy = trance-inducing loop
    beat_reg = _norm(prepared.get("beat_regularity", 0), "beat_regularity")
    plp_stab = _norm(prepared.get("plp_stability", 0), "plp_stability")
    rhy_simple = 1 - _norm(prepared.get("rhythm_complexity", 0), "rhythm_complexity")

    # Energy consistency: low rms_var = stable energy envelope
    rms_v = prepared.get("rms_var", 1)
    log_rms_var = math.log(max(rms_v, 0.001))
    # Use rms_var stats for normalization, apply to log scale
    mean, std = STATS["rms_var"]
    log_mean = math.log(max(mean, 0.001))
    z = (log_rms_var - log_mean) / (std + 1e-8)
    energy_c = 1 - (1 / (1 + math.exp(-z)))

    rhythmic_h = (
        beat_reg * 0.25
        + plp_stab * 0.40
        + rhy_simple * 0.15
        + energy_c * 0.20
    )

    # === PATH 2: TIMBRAL HYPNOTIC ===
    # Consistent texture + minimal change = droning/meditative
    centroid_c = 1 - _norm(prepared.get("centroid_std", 0), "centroid_std")
    mfcc_d_c = 1 - _norm(prepared.get("mfcc_delta_var", 0), "mfcc_delta_var")
    flux_std_c = 1 - _norm(prepared.get("flux_std", 0), "flux_std")
    bw_c = 1 - _norm(prepared.get("bandwidth_std", 0), "bandwidth_std")

    timbral_h = (
        centroid_c * 0.35
        + mfcc_d_c * 0.30
        + flux_std_c * 0.20
        + bw_c * 0.15
    )

    # === COMBINE: stronger path dominates ===
    strong = max(rhythmic_h, timbral_h)
    weak = min(rhythmic_h, timbral_h)
    raw = strong * 0.75 + weak * 0.25

    # Stretch distribution away from center
    hypnotic = 1 / (1 + math.exp(-6 * (raw - 0.5)))

    hypnotic = round(max(0.0, min(1.0, hypnotic)), 4)
    varied = round(1 - hypnotic, 4)

    # Path label: "both" when both paths contribute significantly
    ratio = weak / strong if strong > 0 else 0
    if ratio >= 0.60 and strong >= 0.55:
        path = "both"
    elif rhythmic_h > timbral_h:
        path = "rhythmic"
    else:
        path = "timbral"

    return {
        "hypnotic": hypnotic,
        "varied": varied,
        "hypnotic_path": path,
        "rhythmic_h": round(rhythmic_h, 4),
        "timbral_h": round(timbral_h, 4),
    }
