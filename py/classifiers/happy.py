"""Classifier: happy — ridge regression (numpy only).

CV R²: 0.248 (+/- 0.407)
Output range: 0.0 - 0.8
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['centroid_std', 'mod_centroid', 'zcr', 'perc_energy', 'centroid_var', 'centroid_x_flatness', 'flux', 'mfcc_d0', 'tempo_x_onset', 'rms_x_flux', 'tempo_x_beat', 'onset', 'mod_flatness', 'beat', 'low_energy', 'vocal', 'delta_x_flux', 'rolloff_std', 'bass_ratio', 'tempo', 'bandwidth', 'tonnetz1', 'chroma0', 'mfcc_d2_7', 'mode_x_mfcc1', 'treble_ratio', 'mfcc_d10', 'mid_ratio', 'mode', 'mfcc_d5']

WEIGHTS = np.array([
    0.0005106023, -0.0093385807, 4.2807607456, 5.1225793312, -0.0000002668,
    -0.0024138812, 0.0055489657, 0.0242997446, 0.0008225021, -0.0002860457,
    -0.0004777186, -0.0560385457, 0.7089155306, 0.0644094986, 55.1298202985,
    0.3150114489, -0.0009699141, -0.0001283929, 0.0312097300, -0.0000369317,
    0.0000779648, 0.0661184467, -0.0650602734, -0.0756774558, -0.0000531998,
    -0.7849047082, -0.1406351716, -0.1608375817, 0.0067578483, 0.0648063265,
])

BIAS = -0.6243805741
CLIP_MIN = 0.0
CLIP_MAX = 0.8


def predict(features):
    """Predict happy value from prepared features dict.

    Returns float clipped to [0.0, 0.8].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
