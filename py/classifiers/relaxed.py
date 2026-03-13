"""Classifier: relaxed — ridge regression (numpy only).

CV R²: 0.409 (+/- 0.207)
Output range: 0.0 - 1.0
Weights rescaled from raw Ridge output to spread distribution
(original raw range ~[-0.23, 1.23] mapped to [0, 1] via linear stretch).
"""

import numpy as np

FEATURES = ['perc_energy', 'centroid', 'flatness', 'mfcc_d2_0', 'perc_x_beat_reg', 'harm_fraction', 'vocal', 'centroid_x_flatness', 'mfcc_d0', 'rolloff', 'harm_x_bass', 'dyn_range', 'mfcc_d6', 'tonnetz2', 'spectral_crest', 'onset_rate', 'mfcc_d2_4', 'flux', 'mfcc1', 'spectral_entropy', 'beat_regularity', 'onset_rate_x_rms', 'mid_ratio', 'onset', 'mod_flatness', 'rms_range', 'low_energy', 'rms_x_flux', 'mfcc_d2_3', 'bass_ratio', 'bandwidth', 'mfcc_d7', 'mfcc_d5', 'energy_skew', 'mfcc_s7', 'mod_centroid', 'harm_perc_ratio', 'centroid_var', 'mfcc_d11', 'mfcc_d4']

WEIGHTS = np.array([
    -10.6180461299, -0.0004595201, -14.1249194135, 0.0478627349, 0.5928583199,
    1.5735616780, -0.8487356149, 0.0033946701, -0.0320222260, 0.0000329343,
    -0.3229546759, 0.0061656404, 0.1296324736, 0.0160605896, -0.0008743684,
    -0.0314661941, 0.0410327393, -0.0013425376, -0.0000163147, -0.0444725480,
    -0.0126754955, 0.0016113497, -0.3985378032, -0.0058306664, 0.9801891543,
    -0.0408696893, -14.2132647773, 0.0000252923, -0.1683959871, -0.3706532204,
    0.0002539751, -0.0439311104, -0.0360257791, 0.0631031496, 0.0070758713,
    -0.0068215272, -0.0179263713, 0.0000001141, 0.1204163279, 0.0228554355,
])

BIAS = 1.4121928992
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def predict(features):
    """Predict relaxed value from prepared features dict.

    Returns float clipped to [0.0, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
