"""Classifier: aggressive — ridge regression (numpy only).

CV R²: 0.027 (+/- 0.429)
Output range: 0.0 - 0.9
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['perc_energy', 'perc_x_beat_reg', 'harm_x_bass', 'centroid', 'mfcc_d2_0', 'bass_ratio', 'flatness', 'mode_x_mfcc1', 'mid_ratio', 'mode', 'zcr', 'mfcc_delta2_var', 'mfcc_d2_11', 'mfcc0', 'mfcc_d11']

WEIGHTS = np.array([
    5.6194700307, -0.4552909504, 0.0954189792, -0.0000027285, -0.0102257191,
    -0.4919046901, 5.3990891264, 0.0005579206, -0.3785154638, -0.0703104754,
    -0.4596374174, 0.0431859809, 0.3521391056, -0.0002100618, -0.2398308950,
])

BIAS = 0.3199531669
CLIP_MIN = 0.0
CLIP_MAX = 0.9


def predict(features):
    """Predict aggressive value from prepared features dict.

    Returns float clipped to [0.0, 0.9].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
