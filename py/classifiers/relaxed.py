"""Classifier: relaxed — logistic regression (numpy only).

CV accuracy: 0.949 (+/- 0.016)
Trained on 682 tracks (633 pos, 49 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['vocal', 'mfcc_d0', 'contrast6', 'mfcc_d12', 'mfcc_s5', 'mfcc_d2_2', 'mfcc_s7', 'harm_fraction', 'onset_rate', 'rhythm_complexity', 'contrast0', 'flatness', 'mfcc2', 'mfcc_d2_12', 'mfcc_d2_3', 'mfcc_s9', 'mod_crest', 'rms_range', 'chroma_std', 'harm_energy', 'contrast2', 'mfcc_d2_0', 'plp_stability', 'mfcc_d2_5', 'mfcc_d8', 'mfcc_d5', 'mfcc_s8', 'low_energy_rate', 'dyn_range', 'rms_var']

WEIGHTS = np.array([
    -5.5453588480, -0.4934311397, -0.1110093654, 6.4856055912, -0.3439792468,
    1.0959807705, 0.6290210153, 18.0229765616, -0.4750421296, -7.0443142533,
    -0.0904293446, -124.7558411152, 0.0295398749, 1.8566667459, -1.1092571915,
    -0.4750455037, -0.0163698705, -0.5343437532, -13.2215120565, -21.5675052225,
    0.3485056130, 0.2921572464, 4.9470013779, -3.8360163837, -2.7766663200,
    3.0351602331, 0.1913229205, 11.5929986478, 0.1456557485, 0.0418169966,
])

BIAS = 44.8726881718


def predict(features):
    """Predict relaxed probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
