"""Classifier: hypnotic/varied — perceived repetitive trance quality of music.

Formula-based (no ground truth). Two-path approach:
  - Rhythmic hypnotic: locked pulse + consistent energy (techno loops, process music)
  - Timbral hypnotic: stable timbre + minimal spectral change (drones, ambient)
The stronger path dominates via soft-max blending. When both paths are
strong and close, reports "both" (e.g. Glass arpeggios = rhythmic + timbral).

0 = varied (dynamic, evolving, surprising)
1 = hypnotic (repetitive, trance-inducing, locked-in)
"""

import math

# Corpus statistics (N=1359, soniq_0.5.db 2026-03-12)
_STATS = {
    # Rhythmic path
    "beat_regularity":   (6.67,  2.29),
    "plp_stability":     (0.724, 0.066),
    "rhythm_complexity": (7.591, 0.091),
    "rms_var_log_mean":  0.5,    # log(rms_var) center
    "rms_var_log_std":   2.0,    # log(rms_var) spread
    # Timbral path
    "centroid_std":      (472.0, 288.6),
    "mfcc_delta_var":    (2.03,  0.78),
    "flux_std":          (27.1,  18.5),
    "bandwidth_std":     (371.4, 161.6),
}


def _norm(val, mean, std):
    """Z-score through sigmoid — corpus-relative 0-1."""
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict hypnotic/varied from prepared features dict.

    Returns dict with hypnotic (0-1), varied (0-1), dominant path,
    and path scores.
    """
    # === PATH 1: RHYTHMIC HYPNOTIC ===
    # Locked pulse + consistent energy = trance-inducing loop
    # PLP stability weighted higher than beat regularity — captures
    # arpeggiated process music (Glass) where pulse is locked but
    # beat tracking picks up sub-beat patterns.
    beat_reg   = _norm(prepared.get("beat_regularity", 0),   *_STATS["beat_regularity"])
    plp_stab   = _norm(prepared.get("plp_stability", 0),     *_STATS["plp_stability"])
    rhy_simple = 1 - _norm(prepared.get("rhythm_complexity", 0), *_STATS["rhythm_complexity"])

    rms_v = prepared.get("rms_var", 1)
    energy_c = 1 - _norm(
        math.log(max(rms_v, 0.001)),
        _STATS["rms_var_log_mean"],
        _STATS["rms_var_log_std"],
    )

    rhythmic_h = (
        beat_reg   * 0.25
        + plp_stab * 0.40
        + rhy_simple * 0.15
        + energy_c * 0.20
    )

    # === PATH 2: TIMBRAL HYPNOTIC ===
    # Consistent texture + minimal change = droning/meditative
    centroid_c = 1 - _norm(prepared.get("centroid_std", 0), *_STATS["centroid_std"])
    mfcc_d_c   = 1 - _norm(prepared.get("mfcc_delta_var", 0), *_STATS["mfcc_delta_var"])
    flux_std_c = 1 - _norm(prepared.get("flux_std", 0),     *_STATS["flux_std"])
    bw_c       = 1 - _norm(prepared.get("bandwidth_std", 0), *_STATS["bandwidth_std"])

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
    # (e.g. Glass arpeggios = rhythmic pulse + stable piano timbre)
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
