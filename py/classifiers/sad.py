"""Classifier: sad — logistic regression (numpy only).

CV accuracy: 0.820 (+/- 0.082)
Trained on 682 tracks (320 pos, 362 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_d2_0', 'mfcc_d2_8', 'contrast4', 'beat_regularity', 'onset_rate', 'mfcc_d2_2', 'rolloff', 'mod_crest', 'mfcc_s2', 'low_energy', 'vocal', 'mfcc_s12', 'bandwidth', 'contrast5', 'mfcc_d2_12', 'mfcc_d9', 'rms_max', 'tempo_x_onset', 'centroid', 'mfcc_s1']

WEIGHTS = np.array([
    -0.1237569647, -4.4006760159, 0.4155201432, -0.1120674130, -0.5647945740,
    1.2018809163, -0.0009061603, -0.0568953041, -0.1009430520, -442.4693943039,
    1.3655116049, 0.3418404324, 0.0034689660, -0.1991592772, -7.4018011692,
    3.8587928098, -0.2393976737, -0.0033179215, -0.0014728816, -0.0027366030,
])

BIAS = 5.8517317085


def predict(features):
    """Predict sad probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
