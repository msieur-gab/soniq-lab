"""Classifier: timbre — spectral centroid z-score through sigmoid.

Based on Schubert & Wolfe 2006, Peeters 2011 Timbre Toolbox.
Fixed reference from 681-track corpus.
"""

import numpy as np

CENTROID_MEAN = 1374.4
CENTROID_STD = 528.7


def predict(features):
    """Predict timbral color from prepared features dict.

    Returns dict {"brilliant": float, "warm": float} summing to ~1.
    """
    centroid = features.get("centroid", 0)
    z = (centroid - CENTROID_MEAN) / CENTROID_STD if CENTROID_STD > 0 else 0
    brilliant = round(float(1 / (1 + np.exp(-z))), 4)
    warm = round(1 - brilliant, 4)
    return {"brilliant": brilliant, "warm": warm}
