"""Classifier: instrumental — ridge regression (numpy only).

CV R²: -0.137 (+/- 0.953)
Output range: 0.0 - 1.0
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_delta_var', 'mfcc_delta2_var', 'mfcc_d0', 'flux', 'beat_regularity', 'centroid_std', 'delta_x_flux', 'harm_x_bass', 'rms_x_flux', 'tempo', 'rolloff', 'mfcc_d2_1', 'mfcc_d2_0', 'mfcc0', 'mid_ratio', 'mfcc_d10', 'mod_flatness', 'bass_ratio', 'flatness', 'plp_x_tempo', 'rolloff_std', 'mfcc1', 'rhythm_complexity', 'chroma8', 'mfcc_s10', 'mfcc_d2_6', 'mfcc_s2', 'mfcc_d2_2', 'mfcc_d2_3', 'mfcc_d1', 'bandwidth', 'harm_energy', 'mfcc_d6', 'chroma5', 'bass_mid_ratio', 'mfcc_s7', 'chroma10', 'mfcc_d2_7', 'bandwidth_std', 'flux_std']

WEIGHTS = np.array([
    -0.4574108357, -0.2262922577, 0.0444281635, 0.0032084004, 0.0301818108,
    -0.0003911735, -0.0015614923, 0.1959750565, -0.0000403682, 0.0000978358,
    0.0000494980, 0.0549703193, 0.0165195091, -0.0004399786, 0.3525339917,
    0.2289887738, 0.3335905422, 0.0470302697, 1.4468904530, -0.0053034890,
    0.0000866285, 0.0003569636, 0.3434937067, -0.0315206072, -0.0017300565,
    0.2196636107, 0.0031691822, 0.0416775575, 0.0942918568, 0.0349159004,
    -0.0000660151, 0.4511283166, -0.1040849024, 0.0080283418, 0.0000539820,
    -0.0221065818, 0.0081758277, 0.2449720027, 0.0000864665, 0.0031585386,
])

BIAS = -2.2648045372
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def predict(features):
    """Predict instrumental value from prepared features dict.

    Returns float clipped to [0.0, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
