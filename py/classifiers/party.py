"""Classifier: party — logistic regression (numpy only).

CV accuracy: 0.979 (+/- 0.012)
Trained on 682 tracks (26 pos, 656 neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['delta_x_flux', 'mfcc_s0', 'plp_mean', 'energy_kurtosis', 'beat_regularity', 'mfcc6', 'chroma_std', 'dyn_range', 'mfcc_d2_5', 'key', 'vocal', 'chroma4', 'mfcc10', 'mfcc_s2', 'mfcc1', 'mfcc_d2_0', 'low_energy_rate', 'mfcc2', 'mfcc_s4', 'onset_rate', 'chroma8', 'mfcc_d2_12', 'contrast0', 'chroma3', 'beat', 'mfcc_s5', 'onset_rate_x_rms', 'rms_range', 'mfcc_d2_7', 'mfcc0', 'contrast6', 'centroid', 'mfcc_d12', 'mfcc_d2_10', 'duration', 'perc_energy', 'contrast3', 'mfcc_delta2_var', 'rolloff_std', 'mfcc_d5']

WEIGHTS = np.array([
    0.0204829656, 0.0640368747, -14.5731991106, -0.3843246507, -0.4237609610,
    0.0562172524, 14.6470484303, -0.0714649030, 5.6714302075, 0.1727179615,
    2.7944495388, 4.4556180202, -0.0423570118, 0.0462029180, -0.0196375641,
    -0.4555225825, 6.5931261278, -0.0302172437, -0.2376575712, 0.2641098084,
    4.3112389804, -2.8798067795, 0.1608931837, 1.8925118347, 1.1083264496,
    0.2307400695, 0.0034971963, 0.3468198763, 2.7053996405, 0.0098870574,
    0.0087397644, 0.0019330565, -3.1045971051, 0.7415709968, -0.0021313497,
    29.5752265134, -0.2935607427, -1.4135351616, 0.0005210772, -0.8288435448,
])

BIAS = -18.6249834326


def predict(features):
    """Predict party probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
