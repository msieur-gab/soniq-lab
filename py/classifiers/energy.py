"""Classifier: energetic/still — perceived kinetic energy of music.

Formula-based (no ground truth). Uses corpus-calibrated z-score normalization.
Components: pulse (rhythmic drive), impact (percussive force), activity (event
density), groove (repetitive lock-in). Loudness acts as a multiplier, not an
additive term — a loud drone stays still, a quiet locked groove still has energy.

0 = still (quiet, sparse, motionless)
1 = energetic (pulsing, driving, want-to-move)
"""

import math

# Corpus statistics (N=1359, soniq_0.5.db 2026-03-12)
_STATS = {
    "beat":             (2.3,   0.8),
    "tempo":            (120.0, 30.0),
    "perc_energy":      (0.037, 0.029),
    "onset_rate":       (3.9,   1.7),
    "beat_regularity":  (6.67,  2.29),
    "plp_stability":    (0.724, 0.066),
    "rhythm_complexity": (7.591, 0.091),
    "rms_mean":         (9.8,   5.7),
}


def _norm(val, mean, std):
    """Z-score through sigmoid — corpus-relative 0-1."""
    z = (val - mean) / (std + 1e-8)
    return 1 / (1 + math.exp(-z))


def predict(prepared):
    """Predict energetic/still from prepared features dict.

    Returns dict with energetic (0-1), still (0-1), and component
    scores: pulse, impact, activity, groove, loudness.
    """
    def f(key):
        mean, std = _STATS[key]
        return _norm(prepared.get(key, mean), mean, std)

    # --- Components ---

    # Pulse: rhythmic drive (beat strength + tempo)
    pulse = f("beat") * 0.5 + f("tempo") * 0.5

    # Impact: percussive transients — physical hit sensation
    impact = f("perc_energy")

    # Activity: onset density — urgency
    activity = f("onset_rate")

    # Groove: repetitive lock-in (regularity + pulse stability + simplicity)
    groove = (
        f("beat_regularity") * 0.5
        + f("plp_stability") * 0.3
        + (1 - f("rhythm_complexity")) * 0.2
    )

    # --- Combine ---
    core = pulse * 0.30 + impact * 0.20 + activity * 0.20 + groove * 0.30

    # Loudness as multiplier: scales core from 75% to 100%
    # A quiet locked groove keeps most of its energy; loudness amplifies, not creates
    loudness = f("rms_mean")
    energetic = core * (0.75 + 0.25 * loudness)

    energetic = round(max(0.0, min(1.0, energetic)), 4)
    still = round(1 - energetic, 4)

    return {
        "energetic": energetic,
        "still": still,
        # Components for UI visualization
        "pulse": round(pulse, 4),
        "impact": round(impact, 4),
        "activity": round(activity, 4),
        "groove": round(groove, 4),
        "loudness": round(loudness, 4),
    }
