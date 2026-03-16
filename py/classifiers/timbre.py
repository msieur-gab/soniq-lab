"""Classifier: timbre — spectral centroid z-score through sigmoid.

Based on Schubert & Wolfe 2006, Peeters 2011 Timbre Toolbox.
Fixed reference from 1708-track corpus.
"""

import math

CENTROID_MEAN = 1395.4
CENTROID_STD = 609.3


def predict(features):
    """Predict timbral color from prepared features dict.

    Returns dict {"brilliant": float, "warm": float} summing to ~1.
    """
    centroid = features.get("centroid", 0)
    z = (centroid - CENTROID_MEAN) / CENTROID_STD if CENTROID_STD > 0 else 0
    brilliant = round(1 / (1 + math.exp(-z)), 4)
    warm = round(1 - brilliant, 4)
    return {"brilliant": brilliant, "warm": warm}
