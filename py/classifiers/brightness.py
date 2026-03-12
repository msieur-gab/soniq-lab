"""Classifier: brightness — logistic regression (numpy only).

Consolidates py/brightness.py + py/perceived_brightness.py.
CV accuracy: 0.964 (+/- 0.025)
Trained on 523 tracks (501 pos, 22 neg).
"""

import numpy as np

FEATURES = ['duration', 'rms_var', 'mfcc7', 'mfcc8', 'centroid_x_flatness', 'mfcc_d12', 'beat_regularity', 'contrast5', 'mfcc_d2_9', 'mfcc_s9', 'mfcc_d5', 'mfcc_d2_12', 'delta_x_flux', 'mfcc_d0', 'mfcc_d8', 'tonnetz0', 'mfcc_s5', 'mfcc_s10', 'spectral_entropy', 'rms_max', 'mode', 'bandwidth_std', 'mfcc2', 'mod_centroid', 'mfcc_d7', 'bass_mid_ratio', 'harm_energy', 'rolloff_std', 'rms_mean', 'chroma3']

WEIGHTS = np.array([
    0.0057747575, 0.0813113577, 0.1445058569, -0.1024404956, -0.0481407670,
    6.3849961925, 0.5979163012, -0.1818224866, 3.4148991683, 0.6014370898,
    -2.5212111658, -7.8665326085, -0.0181310878, -0.3397013181, -3.0221158608,
    -6.0035788373, 0.4944293615, -0.4305845803, 0.8446896989, 0.2094003172,
    -1.2759308639, -0.0040979059, -0.0231438942, -0.1003030297, -0.6024505034,
    0.0060317817, -11.3948537281, 0.0006743239, 0.0582985459, 5.0907847336,
])

BIAS = 0.0809795485


def predict(features):
    """Predict brightness from prepared features dict.

    Returns dict {"bright": float, "dark": float} summing to ~1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    bright = float(1 / (1 + np.exp(-logit)))
    dark = round(1 - bright, 4)
    bright = round(bright, 4)
    return {"bright": bright, "dark": dark}
