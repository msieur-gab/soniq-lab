"""Classifier: arousal — ridge regression (numpy only).

CV R²: 0.544 (+/- 0.118)
Output range: 3.4 - 6.2
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_delta2_var', 'tempo_x_beat', 'beat', 'rms_x_flux', 'mfcc_d0', 'flux', 'harm_fraction', 'vocal', 'centroid', 'beat_regularity', 'centroid_x_flatness', 'mfcc_d2_1', 'tempo', 'mfcc_d5', 'harm_x_bass', 'mod_centroid', 'tempo_x_onset', 'mfcc_d2_6', 'mfcc_d2_3', 'centroid_std', 'mfcc_d2_9', 'onset_rate_x_rms', 'flatness', 'mode_x_mfcc1', 'mfcc0', 'chroma1', 'chroma8', 'mode', 'onset_rate', 'mfcc_d2_2', 'chroma5', 'mfcc_delta_var', 'energy_skew', 'mod_flatness', 'mfcc_d11', 'contrast3', 'mfcc_d1', 'mfcc_s7', 'tonnetz4', 'mfcc_d2']

WEIGHTS = np.array([
    -0.8204809804, -0.0011595052, 0.2339085048, -0.0006384082, 0.0720913352,
    0.0125441594, -2.5702771871, 1.2529522334, 0.0003808592, 0.0554949122,
    -0.0047214663, 0.1759319486, 0.0029990444, -0.1291196188, 0.3999954051,
    -0.0113211760, -0.0001484790, 0.3624063928, 0.1859027591, -0.0004946118,
    0.5003312834, 0.0029761872, 8.3024529835, -0.0006350087, -0.0002779199,
    -0.1060048392, 0.0994325925, 0.0768476486, 0.0358648713, 0.1449714175,
    0.1234298463, -0.0798384133, -0.0485608864, 0.8886781088, 0.2375953520,
    0.0214542939, 0.0222564416, -0.0370693878, 0.0224617773, -0.0843727414,
])

BIAS = 3.4078394923
CLIP_MIN = 3.4
CLIP_MAX = 6.2


def predict(features):
    """Predict arousal value from prepared features dict.

    Returns float clipped to [3.4, 6.2].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
