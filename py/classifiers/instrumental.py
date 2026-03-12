"""Classifier: instrumental — logistic regression (numpy only).

CV accuracy: 0.956 (+/- 0.024)
Trained on 682 tracks (635 pos, 47 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_d2_1', 'beat_regularity', 'harm_perc_ratio', 'delta_x_flux', 'mfcc12', 'contrast0', 'mfcc_s3', 'mfcc_s5', 'onset_rate', 'mfcc8', 'chroma3', 'zcr', 'harm_fraction', 'mfcc_delta_var', 'contrast4', 'mfcc_d0', 'mfcc_s4', 'mfcc_s12', 'contrast_range', 'mfcc_s11', 'harm_energy', 'mfcc_d2_2', 'duration', 'mfcc_s2', 'mfcc_d2_4', 'mfcc_d2', 'bass_mid_ratio', 'plp_x_tempo', 'rolloff', 'contrast1']

WEIGHTS = np.array([
    1.1178604599, 0.6508920551, 0.4242190720, -0.0145486403, 0.1409838186,
    -0.2477403739, -0.1025156322, -0.5061585828, 0.3988833983, 0.0729666286,
    -5.2436207390, -18.8329734156, 5.4310523244, -2.3255903678, -0.0088387617,
    -0.3148502117, -0.3074103822, 0.3766529881, -0.0106319973, 0.1079583319,
    -1.1095626833, 1.2466470734, 0.0034692115, 0.0550506204, 4.4416990982,
    -0.8828136435, 0.0012568212, 7.0505884610, 0.0000677520, -0.1278417492,
])

BIAS = 7.7362123877


def predict(features):
    """Predict instrumental probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
