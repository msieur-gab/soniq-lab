"""Classifier: happy — logistic regression (numpy only).

CV accuracy: 0.947 (+/- 0.019)
Trained on 682 tracks (42 pos, 640 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc_d2_0', 'mfcc_d0', 'mfcc_d2_7', 'zcr', 'mfcc_d2_3', 'mfcc_s7', 'rhythm_complexity', 'vocal', 'low_energy_rate', 'mfcc_d2_6', 'mod_centroid', 'harm_perc_ratio', 'chroma_std', 'harm_fraction', 'mfcc2', 'mfcc_s0', 'mfcc_d2_11', 'low_energy', 'chroma1', 'mfcc1', 'duration', 'centroid_x_flatness', 'mfcc_d8', 'contrast6', 'mfcc_s6', 'chroma6', 'tonnetz1', 'mfcc_d1', 'contrast0', 'contrast1', 'tonnetz2', 'perc_energy', 'contrast5', 'mfcc8', 'mfcc_d5', 'flux', 'chroma9', 'mfcc_d2_4', 'chroma7', 'mfcc6']

WEIGHTS = np.array([
    0.6201234238, 0.1749269351, -6.4613719825, 35.5319873045, 2.5638141829,
    -0.5362194310, 6.9588190816, 6.4809804232, -9.8874001288, 5.4296919861,
    -0.0511341760, -0.4324811126, -6.7058584758, -9.7726026613, -0.0186406814,
    -0.0315435210, -6.1940935622, -2694.9398915702, 2.1203837383, 0.0095905524,
    -0.0021877902, -0.0178480429, 3.9084168964, 0.1354080657, -0.1425289360,
    2.6585402206, 1.4551234357, -0.3423157093, 0.1684064933, -0.0445907593,
    2.2402759142, 23.7175648566, -0.2075327106, -0.0577736057, -0.3643554811,
    0.0092300628, -3.6136827043, -2.5404355171, -1.9859272595, -0.0618018442,
])

BIAS = -48.2355245089


def predict(features):
    """Predict happy probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
