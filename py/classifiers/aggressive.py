"""Classifier: aggressive — logistic regression (numpy only).

CV accuracy: 0.990 (+/- 0.011)
Trained on 682 tracks (13 pos, 669 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['plp_mean', 'rms_x_flux', 'chroma6', 'flux', 'mfcc2', 'contrast3', 'harm_energy', 'mfcc_s2', 'contrast4', 'mfcc_d5', 'chroma7', 'tonnetz3', 'mfcc_d2_4', 'flatness', 'mode']

WEIGHTS = np.array([
    -33.6739962926, 0.0024371876, 4.9908909581, 0.0418373990, -0.0567888450,
    -0.2344938134, 0.8235175688, 0.0544959268, -0.2122622451, 3.4446623353,
    2.3305065880, -5.0779928795, -3.3759223425, 112.5908770486, -0.6722889531,
])

BIAS = -1.6711397842


def predict(features):
    """Predict aggressive probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
