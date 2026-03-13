"""Classifier: party — ridge regression (numpy only).

CV R²: -0.206 (+/- 1.092)
Output range: 0.0 - 0.9
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['perc_energy', 'delta_x_flux', 'perc_x_beat_reg', 'centroid', 'harm_x_bass', 'harm_fraction', 'flux', 'mod_flatness', 'mfcc1', 'mod_centroid', 'dyn_range', 'flatness', 'mfcc0', 'tempo_x_onset', 'bandwidth']

WEIGHTS = np.array([
    5.8177935567, 0.0014044830, -0.5029216476, 0.0000820901, 0.0473953397,
    -0.0376349720, -0.0026931659, -0.9161174136, -0.0008199373, 0.0066671912,
    -0.0017593819, 5.2314439994, -0.0002121205, -0.0000784221, -0.0001065281,
])

BIAS = 0.2387355598
CLIP_MIN = 0.0
CLIP_MAX = 0.9


def predict(features):
    """Predict party value from prepared features dict.

    Returns float clipped to [0.0, 0.9].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
