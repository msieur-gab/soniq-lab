"""Timbre color — spectral centroid z-score against fixed corpus stats.

Returns (brilliant, warm) summing to 1.
Brilliant = rich upper harmonics, high centroid.
Warm = muted, energy in low frequencies, low centroid.

Based on Schubert & Wolfe 2006, Peeters 2011 Timbre Toolbox.
"""

import numpy as np

# Fixed reference from 681 tracks in music-player DB.
CENTROID_MEAN = 1374.4
CENTROID_STD = 528.7


def compute_timbre(features):
    """Compute timbral color from spectral centroid.

    Returns (brilliant, warm) as floats summing to 1.
    """
    centroid = features.get("centroid_mean", 0)
    z = (centroid - CENTROID_MEAN) / CENTROID_STD if CENTROID_STD > 0 else 0
    brilliant = round(float(1 / (1 + np.exp(-z))), 4)
    warm = round(1 - brilliant, 4)
    return brilliant, warm
