"""Classifier: danceable — logistic regression (numpy only).

CV accuracy: 0.855 (+/- 0.051)
Trained on 682 tracks (349 pos, 333 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_d2_8', 'centroid_std', 'mfcc_s1', 'beat_regularity', 'mod_flatness', 'mfcc_d8', 'mfcc0', 'mfcc_d10', 'flux_std', 'bandwidth', 'harm_energy', 'chroma_std', 'mfcc_d12', 'mfcc_d2_9', 'mfcc_s5', 'harm_perc_ratio', 'mfcc_d2_12', 'rhythm_complexity', 'chroma10', 'mode_x_mfcc1', 'mfcc_s7', 'chroma8', 'mode', 'mfcc9', 'contrast5', 'harm_x_bass', 'mfcc_d2_0', 'mfcc_s9', 'rms_max', 'centroid', 'chroma0', 'mfcc_d9', 'mfcc_s0', 'perc_x_beat_reg', 'flux', 'chroma2', 'mfcc_s10', 'mfcc_s12', 'contrast4', 'vocal']

WEIGHTS = np.array([
    15.2824034627, 0.0067108236, -0.1240553012, 0.4073026418, -11.3333017092,
    -7.0236995509, -0.0077772029, -3.4129482683, 0.0494651531, -0.0011918648,
    16.5429199462, 20.5112137435, -5.5420184274, 3.3892172754, 0.1867749744,
    -0.2991814802, 11.5031526328, -9.5099091432, 6.4536281054, 0.0098023024,
    0.2078650474, 4.8744914188, -1.3027807068, 0.0803676005, 0.1298110547,
    -6.2273716406, -0.3635510925, -0.2080543346, 0.2379984652, 0.0008588787,
    3.6475007416, -2.2875713469, -0.0272212929, 2.8028855797, -0.0149484199,
    0.6973533531, 0.3551045638, -0.2194447656, -0.1892238894, 4.3730774568,
])

BIAS = 60.2227360652


def predict(features):
    """Predict danceable probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
