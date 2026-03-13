"""Classifier: acoustic — ridge regression (numpy only).

CV R²: 0.330 (+/- 0.318)
Output range: 0.0 - 1.0
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['bandwidth', 'mfcc0', 'rolloff', 'mfcc_d2_0', 'mfcc_delta_var', 'mfcc_delta2_var', 'onset', 'centroid_std', 'mod_flatness', 'centroid_x_flatness', 'delta_x_flux', 'mod_centroid', 'tempo', 'plp_x_tempo', 'centroid', 'chroma8', 'mfcc_d2_8', 'centroid_var', 'mod_crest', 'chroma11', 'contrast2', 'flatness', 'mfcc_d2_1', 'flux', 'mfcc_d10', 'mfcc_d0', 'rolloff_std', 'mfcc_d8', 'mfcc_d2_11', 'tonnetz4', 'zcr', 'mfcc_d1', 'chroma5', 'mfcc_d2_12', 'chroma4', 'treble_ratio', 'dyn_range', 'beat', 'rhythm_complexity', 'mfcc_d12']

WEIGHTS = np.array([
    0.0005628775, 0.0011310887, -0.0000826664, 0.0624411931, -0.1512331577,
    0.0813094280, -0.2000009563, -0.0003236697, 2.4638306744, 0.0027660950,
    -0.0020415088, -0.0196504602, -0.0018396787, 0.4933035024, -0.0002823772,
    -0.4305905095, -1.1178352470, 0.0000001935, -0.0031212725, -0.2961914854,
    0.0372407646, -7.3083782866, -0.0255448599, -0.0005388059, 0.0950166428,
    0.0243190269, -0.0001138470, 0.7225171111, -0.2588210190, -1.0522401156,
    -0.9681014520, 0.0640635167, -0.3137884957, -0.6793981842, 0.1896425780,
    1.8370461088, 0.0003650677, 0.1060289471, 0.8997453763, 0.3392861651,
])

BIAS = -6.6134991458
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def predict(features):
    """Predict acoustic value from prepared features dict.

    Returns float clipped to [0.0, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
