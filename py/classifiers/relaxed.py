"""Classifier: relaxed — ridge regression (numpy only).

CV R²: 0.409 (+/- 0.207)
Output range: 0.1 - 1.0
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['perc_energy', 'centroid', 'flatness', 'mfcc_d2_0', 'perc_x_beat_reg', 'harm_fraction', 'vocal', 'centroid_x_flatness', 'mfcc_d0', 'rolloff', 'harm_x_bass', 'dyn_range', 'mfcc_d6', 'tonnetz2', 'spectral_crest', 'onset_rate', 'mfcc_d2_4', 'flux', 'mfcc1', 'spectral_entropy', 'beat_regularity', 'onset_rate_x_rms', 'mid_ratio', 'onset', 'mod_flatness', 'rms_range', 'low_energy', 'rms_x_flux', 'mfcc_d2_3', 'bass_ratio', 'bandwidth', 'mfcc_d7', 'mfcc_d5', 'energy_skew', 'mfcc_s7', 'mod_centroid', 'harm_perc_ratio', 'centroid_var', 'mfcc_d11', 'mfcc_d4']

WEIGHTS = np.array([
    -7.9635345974, -0.0003446401, -10.5936895601, 0.0358970512, 0.4446437399,
    1.1801712585, -0.6365517112, 0.0025460026, -0.0240166695, 0.0000247007,
    -0.2422160069, 0.0046242303, 0.0972243552, 0.0120454422, -0.0006557763,
    -0.0235996456, 0.0307745545, -0.0010069032, -0.0000122360, -0.0333544110,
    -0.0095066216, 0.0012085123, -0.2989033524, -0.0043729998, 0.7351418657,
    -0.0306522670, -10.6599485830, 0.0000189692, -0.1262969903, -0.2779899153,
    0.0001904813, -0.0329483328, -0.0270193343, 0.0473273622, 0.0053069035,
    -0.0051161454, -0.0134447785, 0.0000000856, 0.0903122459, 0.0171415766,
])

BIAS = 1.4091446744
CLIP_MIN = 0.1
CLIP_MAX = 1.0


def predict(features):
    """Predict relaxed value from prepared features dict.

    Returns float clipped to [0.1, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
