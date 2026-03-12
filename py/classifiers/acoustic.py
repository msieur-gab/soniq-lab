"""Classifier: acoustic — logistic regression (numpy only).

CV accuracy: 0.832 (+/- 0.070)
Trained on 682 tracks (308 pos, 374 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_d2_0', 'low_energy', 'mod_flatness', 'mfcc_d2_11', 'contrast2', 'mfcc0', 'centroid_std', 'mod_centroid', 'mod_crest', 'rolloff', 'rhythm_complexity', 'rolloff_std', 'mfcc_d12', 'contrast4', 'dyn_range', 'mfcc1', 'mfcc_delta2_var', 'spectral_crest', 'mfcc_s5', 'zcr', 'centroid_x_flatness', 'mfcc_d2_8', 'onset', 'bandwidth', 'chroma10', 'mfcc_d8', 'mfcc6', 'rms_var', 'chroma_std', 'delta_x_flux']

WEIGHTS = np.array([
    0.6862809540, -2543.3645641028, 16.7957451575, -7.9856939644, 0.4146128817,
    0.0055018643, -0.0019760288, -0.1354178528, -0.0526151853, -0.0012313826,
    10.8921817861, -0.0008940086, 4.2322476505, 0.2071990975, 0.0156427598,
    -0.0238821535, 1.0022622743, -0.0320997062, -0.2311138075, -57.2530196508,
    0.0423155636, -12.7184603585, -1.1997912869, 0.0018575263, -4.8591764135,
    7.9575820992, 0.0110783647, -0.0345620474, -11.1284812449, -0.0198038050,
])

BIAS = -73.8352339979


def predict(features):
    """Predict acoustic probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
