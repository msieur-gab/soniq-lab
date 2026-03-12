"""Classifier: tonal — logistic regression (numpy only).

CV accuracy: 0.862 (+/- 0.055)
Trained on 682 tracks (510 pos, 172 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['perc_x_beat_reg', 'mfcc_d8', 'harm_energy', 'flux', 'mfcc4', 'onset_rate_x_rms', 'mfcc_d9', 'mfcc_d0', 'beat_regularity', 'mfcc_s12', 'mfcc1', 'contrast3', 'mfcc_d2_7', 'mfcc_d2_0', 'mfcc0', 'chroma_std', 'mfcc9', 'tonnetz_energy', 'chroma9', 'mod_crest', 'mod_flatness', 'chroma10', 'rolloff_std', 'rolloff', 'chroma7', 'tonnetz2', 'chroma8', 'rms_max', 'perc_energy', 'energy_skew', 'mfcc_s8', 'tempo_x_beat', 'mfcc_d1', 'mode', 'mfcc_s7', 'mfcc_d5', 'flatness', 'mfcc_d11', 'mfcc_d2_8', 'mfcc2']

WEIGHTS = np.array([
    9.1074072529, -8.2928627152, 32.0585970184, -0.0414102962, -0.1094163180,
    -0.0228427539, -4.4293638098, -0.4783424385, 0.2754494118, -0.4176192930,
    0.0242619644, -0.2648877357, 2.6868813220, -0.0892454075, -0.0154870622,
    14.1958372638, 0.1029263436, -4.3753319087, 1.1942191643, 0.0304928695,
    -7.5497540309, 1.9372867091, 0.0007310314, 0.0005864595, 3.2198117081,
    -2.0814908094, 3.0190015171, 0.2004966472, 15.0191923677, 0.8522360176,
    0.2774667686, 0.0022948082, 0.3557760299, -0.8049532616, 0.1384541921,
    1.2810717259, 71.5149089293, 2.7337549908, 7.0757416412, -0.0154259727,
])

BIAS = -8.7955712806


def predict(features):
    """Predict tonal probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
