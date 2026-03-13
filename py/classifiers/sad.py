"""Classifier: sad — ridge regression (numpy only).

CV R²: 0.512 (+/- 0.137)
Output range: 0.0 - 1.0
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = ['bandwidth', 'mfcc0', 'mfcc_delta2_var', 'rolloff', 'flux', 'rms_x_flux', 'plp_x_tempo', 'centroid_x_flatness', 'centroid', 'tempo_x_beat', 'onset_rate', 'mfcc_d2_8', 'tempo', 'tempo_x_onset', 'mfcc_d2_0', 'flatness', 'beat_regularity', 'mfcc_delta_var', 'mod_flatness', 'chroma11', 'tonnetz2', 'vocal', 'zcr', 'chroma9', 'treble_ratio', 'contrast2', 'harm_fraction', 'chroma8', 'harm_energy', 'onset', 'mod_centroid', 'rms_max', 'harm_x_bass', 'mfcc_d10', 'mfcc_d2_1', 'chroma3', 'mfcc_d2_6', 'tonnetz0', 'beat', 'dyn_range', 'plp_stability', 'rms_mean', 'mfcc_d0', 'mfcc_d2', 'contrast4', 'onset_rate_x_rms', 'chroma4', 'mfcc3', 'mod_crest', 'mfcc_d2_3', 'mfcc8', 'mfcc_d2_9', 'mfcc_d9', 'mfcc_d5', 'chroma0', 'mfcc_d6', 'mfcc_d2_11', 'perc_energy', 'chroma10', 'contrast3', 'mfcc_d2_10', 'mode', 'chroma1', 'mfcc1', 'mid_ratio', 'tonnetz1', 'mfcc_s5', 'mfcc9', 'spectral_entropy', 'mode_x_mfcc1', 'mfcc_d2_12', 'mfcc_d8', 'mfcc_s12', 'mfcc_d2_7', 'mfcc_d1', 'tonnetz3', 'mfcc_s8', 'tonnetz_energy', 'plp_mean', 'mfcc_s10', 'mfcc_s1', 'contrast5', 'contrast1', 'tonnetz4', 'contrast_range', 'contrast6', 'bandwidth_std', 'spectral_crest', 'rolloff_std', 'duration', 'energy_skew', 'mfcc_d7', 'perc_x_beat_reg', 'harm_perc_ratio', 'mfcc_s0', 'chroma7', 'centroid_var', 'mfcc_s7', 'mfcc_s3', 'tonnetz5', 'mfcc10', 'mfcc4', 'mfcc11', 'bass_ratio', 'mfcc12', 'mfcc_s9', 'low_energy_rate', 'low_energy', 'chroma_std', 'mfcc_d2_5', 'chroma5', 'contrast0', 'mfcc2', 'flux_std', 'mfcc_s2', 'rms_var', 'mfcc_d4', 'mfcc6', 'chroma6', 'mfcc_d3', 'spectral_kurtosis', 'delta_x_flux', 'mfcc_d12', 'mfcc_s4', 'mfcc_d2_2', 'mfcc_s6', 'spectral_skew', 'rhythm_complexity', 'chroma2', 'rms_range', 'mfcc5', 'bass_mid_ratio', 'mfcc7', 'tempo_sq', 'centroid_std', 'mfcc_d11', 'key', 'mfcc_s11', 'energy_kurtosis', 'mfcc_d2_4']

WEIGHTS = np.array([
    0.0004486813, 0.0018770037, 0.3990248490, -0.0001103492, -0.0061879537,
    0.0002587148, -0.7672213974, 0.0032156028, -0.0001632272, 0.0006678001,
    -0.0502199701, -0.5869753031, 0.0024308668, -0.0007574947, 0.0323758034,
    -6.9697802005, -0.0313283668, -0.1099621285, 1.0012023133, -0.4503007620,
    0.5785039090, -0.4414747067, -2.1837190512, 0.3970089460, 1.2290550713,
    0.0192829447, 0.6229803335, -0.3912074127, -1.0170045860, -0.0909949015,
    -0.0068251150, -0.0122085852, -0.3310498278, 0.2838320738, -0.0505693592,
    -0.3416190099, -0.3337917740, -0.4718083863, -0.0611747073, 0.0049663424,
    0.6672438887, -0.0090705409, -0.0153590812, 0.0598252232, 0.0135876808,
    0.0013448876, 0.2382750005, 0.0027590858, -0.0016161923, -0.1163469341,
    -0.0050584746, -0.2892212786, 0.1753507613, 0.1191880389, 0.2200314621,
    0.1392186857, -0.2994084142, -1.3005762325, -0.2026009988, 0.0097742459,
    -0.2685473227, 0.0535451354, 0.1964567303, 0.0008013124, -0.1468232816,
    -0.2610347506, -0.0121486696, 0.0037697256, 0.0422948114, -0.0003109558,
    -0.2214453010, 0.1064786400, 0.0125943785, 0.1559323860, 0.0140619811,
    -0.1818922186, 0.0103000665, -0.2092612364, 0.4240620822, 0.0103461680,
    0.0021631348, -0.0056258551, 0.0068699886, -0.5340143737, -0.0037712343,
    -0.0071815668, -0.0001018189, 0.0005476543, -0.0000264037, -0.0000717849,
    0.0208266057, 0.0625335783, -0.0706868055, 0.0054998063, 0.0007060416,
    -0.0853740776, 0.0000000390, 0.0073170257, -0.0037631334, -0.4363596935,
    -0.0020228362, 0.0011937453, -0.0021285023, 0.0638908583, -0.0022892802,
    -0.0071503322, 0.1606814854, -27.8251693784, -0.2945395749, -0.0605093723,
    -0.0639900588, 0.0024450769, 0.0003807186, 0.0005276127, -0.0014251674,
    -0.0004336395, -0.0181978815, 0.0007628978, -0.0389485239, -0.0106638949,
    -0.0000000000, 0.0000690640, -0.0248612199, -0.0015107592, -0.0077675215,
    0.0018604612, -0.0000002921, -0.0427393118, 0.0188067731, 0.0015529587,
    0.0003159844, -0.0000060392, 0.0002404859, 0.0000001733, -0.0000053047,
    0.0071059384, 0.0002738606, -0.0004761166, 0.0003107935, 0.0015402961,
])

BIAS = 0.7900152719
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def predict(features):
    """Predict sad value from prepared features dict.

    Returns float clipped to [0.0, 1.0].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
