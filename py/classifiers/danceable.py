"""Classifier: danceable — ridge regression (numpy only).

CV R²: 0.514 (+/- 0.145)
Output range: 0.0 - 1.0
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['mfcc0', 'centroid_std', 'centroid_var', 'mfcc_delta_var', 'perc_energy', 'bandwidth', 'beat_regularity', 'mfcc_d2_8', 'rms_max', 'centroid_x_flatness', 'mfcc_s1', 'rms_mean', 'plp_x_tempo', 'onset', 'spectral_crest', 'flatness', 'mfcc_d2_0', 'rolloff', 'chroma8', 'mfcc_d8', 'centroid', 'perc_x_beat_reg', 'zcr', 'mod_flatness', 'mfcc_d2_6', 'mfcc_d10', 'spectral_entropy', 'chroma5', 'harm_x_bass', 'plp_stability', 'mfcc_d0', 'mfcc_s0', 'mfcc_d5', 'mfcc_s7', 'flux_std', 'harm_energy', 'vocal', 'rms_x_flux', 'mfcc_d9', 'contrast3']

WEIGHTS = np.array([
    -0.0016772243, 0.0007059583, -0.0000003477, 0.0226428994, 6.2204727672,
    -0.0004573921, 0.0576133706, 1.1155132979, 0.0194392323, -0.0042783510,
    -0.0075769554, 0.0080545948, 0.1147149328, 0.2230706048, -0.0011453379,
    9.1923050309, -0.0112177622, 0.0001254268, 0.3606255369, -0.5694905126,
    -0.0000153726, -0.1861239650, 2.4559856670, -1.0033108720, 0.1274657279,
    -0.1595408099, -0.0766827820, 0.1049686927, -0.0045892383, -0.3637491978,
    -0.0165833579, -0.0026484983, 0.1560725734, 0.0175346460, 0.0013756971,
    1.3302050671, 0.0745396478, -0.0001387459, -0.1725085205, -0.0225032333,
])

BIAS = 0.1829766540
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def predict(features):
    """Predict danceable value from prepared features dict.

    Returns float clipped to [0.0, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
