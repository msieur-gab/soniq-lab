"""Classifier: valence — ridge regression (numpy only).

CV R²: 0.605 (+/- 0.173)
Output range: 2.7 - 6.2
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_delta2_var', 'beat', 'perc_energy', 'flux', 'mfcc_d0', 'tempo_x_beat', 'vocal', 'harm_fraction', 'mfcc_d5', 'beat_regularity', 'rms_x_flux', 'mfcc1', 'perc_x_beat_reg', 'harm_x_bass', 'mfcc_d2_1', 'tempo_x_onset', 'mfcc_d2_9', 'mfcc_delta_var', 'chroma8', 'centroid', 'mfcc0', 'mfcc_d2_5', 'mfcc_s7', 'onset_rate', 'chroma11', 'chroma1', 'mfcc_d2_3', 'zcr', 'treble_ratio', 'mfcc2']

WEIGHTS = np.array([
    -0.6306954044, 0.0298943259, 10.2134010320, 0.0127709694, 0.0742241941,
    0.0008648329, 1.8713867220, -3.6649774572, -0.4728209623, 0.0538781892,
    -0.0003365953, -0.0008630949, -1.1986686397, 0.6670735309, 0.0568272210,
    -0.0009388352, 0.8005891675, -0.0801240958, 0.2194056769, 0.0003124864,
    -0.0002045054, 0.6184094463, -0.0440751525, 0.0438860012, 0.1496964063,
    -0.3005538379, 0.3155723264, 2.5181847390, -0.8513441332, -0.0060646876,
])

BIAS = 4.0822217523
CLIP_MIN = 2.7
CLIP_MAX = 6.2


def predict(features):
    """Predict valence value from prepared features dict.

    Returns float clipped to [2.7, 6.2].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
