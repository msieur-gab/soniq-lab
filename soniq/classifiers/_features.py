"""Feature preparation for v0.6 classifiers.

Maps long librosa feature names to short names, unpacks vectors to
indexed scalars, and computes derived/interaction features.

Input: raw dict from extract_track_features()
Output: flat dict with ~150 named features ready for classifiers.
"""

import math


def prepare(librosa_features):
    """Prepare features dict for classifiers.

    Takes raw librosa_features dict (with long names + vector arrays)
    and returns flat dict with short names + indexed scalars + derived features.
    """
    f = librosa_features
    out = {}

    # Scalars — map long names to short names
    out["centroid"] = f.get("centroid_mean", 0)
    out["centroid_std"] = f.get("centroid_std", 0)
    out["rolloff"] = f.get("rolloff_mean", 0)
    out["rolloff_std"] = f.get("rolloff_std", 0)
    out["bandwidth"] = f.get("bandwidth_mean", 0)
    out["bandwidth_std"] = f.get("bandwidth_std", 0)
    out["flatness"] = f.get("flatness_mean", 0)
    out["flux"] = f.get("spectral_flux", 0)
    out["flux_std"] = f.get("flux_std", 0)
    out["zcr"] = f.get("zcr_mean", 0)
    out["rms_mean"] = f.get("rms_mean", 0)
    out["rms_max"] = f.get("rms_max", 0)
    out["rms_var"] = f.get("rms_variance", 0)
    out["dyn_range"] = f.get("dynamic_range", 0)
    out["tempo"] = f.get("tempo", 0)
    out["key"] = f.get("key", 0)
    out["mode"] = f.get("mode", 1)
    out["onset"] = f.get("onset_strength", 0)
    out["beat"] = f.get("beat_strength", 0)
    out["vocal"] = f.get("vocal_proxy", 0)
    out["duration"] = f.get("duration", 0)

    # New scalars (v0.5)
    out["low_energy_rate"] = f.get("low_energy_rate", 0)
    out["energy_skew"] = f.get("energy_skew", 0)
    out["energy_kurtosis"] = f.get("energy_kurtosis", 0)
    out["bass_ratio"] = f.get("bass_ratio", 0)
    out["mid_ratio"] = f.get("mid_ratio", 0)
    out["treble_ratio"] = f.get("treble_ratio", 0)
    out["bass_mid_ratio"] = f.get("bass_mid_ratio", 0)
    out["spectral_skew"] = f.get("spectral_skew", 0)
    out["spectral_kurtosis"] = f.get("spectral_kurtosis", 0)
    out["spectral_entropy"] = f.get("spectral_entropy", 0)
    out["spectral_crest"] = f.get("spectral_crest", 0)
    out["mfcc_delta_var"] = f.get("mfcc_delta_var", 0)
    out["mfcc_delta2_var"] = f.get("mfcc_delta2_var", 0)
    out["mod_flatness"] = f.get("mod_flatness", 0)
    out["mod_crest"] = f.get("mod_crest", 0)
    out["mod_centroid"] = f.get("mod_centroid", 0)
    out["harm_energy"] = f.get("harm_energy", 0)
    out["perc_energy"] = f.get("perc_energy", 0)
    out["harm_perc_ratio"] = f.get("harm_perc_ratio", 0)
    out["harm_fraction"] = f.get("harm_fraction", 0)
    out["beat_regularity"] = f.get("beat_regularity", 0)
    out["rhythm_complexity"] = f.get("rhythm_complexity", 0)
    out["plp_mean"] = f.get("plp_mean", 0)
    out["plp_stability"] = f.get("plp_stability", 0)
    out["onset_rate"] = f.get("onset_rate", 0)
    out["voice_band_ratio"] = f.get("voice_band_ratio", 0)

    # pYIN features (v0.6)
    out["voiced_ratio"] = f.get("voiced_ratio", 0)
    out["voiced_confidence"] = f.get("voiced_confidence", 0)
    out["f0_mean"] = f.get("f0_mean", 0)
    out["f0_std"] = f.get("f0_std", 0)

    # Chroma major key correlation (v0.6)
    out["chroma_major_corr"] = f.get("chroma_major_corr", 0)

    # Derived scalars
    out["rms_range"] = out["rms_max"] - out["rms_mean"]
    out["centroid_var"] = out["centroid_std"] ** 2 if out["centroid_std"] else 0
    out["low_energy"] = 1.0 / (1.0 + out["centroid"]) if out["centroid"] else 0
    out["tempo_sq"] = out["tempo"] ** 2

    # Unpack vectors to indexed scalars
    mfcc = f.get("mfcc_mean", [0] * 13)
    mfcc_s = f.get("mfcc_std", [0] * 13)
    mfcc_delta = f.get("mfcc_delta_mean", [0] * 13)
    mfcc_delta2 = f.get("mfcc_delta2_mean", [0] * 13)
    contrast = f.get("contrast_mean", [0] * 7)
    chroma = f.get("chroma_mean", [0] * 12)
    tonnetz = f.get("tonnetz_mean", [0] * 6)

    for i in range(13):
        out[f"mfcc{i}"] = mfcc[i] if i < len(mfcc) else 0
        out[f"mfcc_s{i}"] = mfcc_s[i] if i < len(mfcc_s) else 0
        out[f"mfcc_d{i}"] = mfcc_delta[i] if i < len(mfcc_delta) else 0
        out[f"mfcc_d2_{i}"] = mfcc_delta2[i] if i < len(mfcc_delta2) else 0

    for i in range(7):
        out[f"contrast{i}"] = contrast[i] if i < len(contrast) else 0

    for i in range(12):
        out[f"chroma{i}"] = chroma[i] if i < len(chroma) else 0

    for i in range(6):
        out[f"tonnetz{i}"] = tonnetz[i] if i < len(tonnetz) else 0

    # Cross-feature interactions
    out["tempo_x_beat"] = out["tempo"] * out["beat"]
    out["tempo_x_onset"] = out["tempo"] * out["onset"]
    out["rms_x_flux"] = out["rms_mean"] * out["flux"]
    out["mode_x_mfcc1"] = out["mode"] * out["mfcc1"]
    out["centroid_x_flatness"] = out["centroid"] * out["flatness"]
    out["contrast_range"] = (
        out["contrast6"] - out["contrast0"]
        if out["contrast6"] and out["contrast0"] else 0
    )
    chroma_vals = [out[f"chroma{i}"] for i in range(12)]
    chroma_mean = sum(chroma_vals) / 12
    out["chroma_std"] = math.sqrt(
        sum((v - chroma_mean) ** 2 for v in chroma_vals) / 12
    )
    out["tonnetz_energy"] = math.sqrt(sum(
        out[f"tonnetz{i}"] ** 2 for i in range(6)
    ))
    out["harm_x_bass"] = out["harm_fraction"] * out["bass_ratio"]
    out["perc_x_beat_reg"] = out["perc_energy"] * out["beat_regularity"]
    out["delta_x_flux"] = out["mfcc_delta_var"] * out["flux"]
    out["plp_x_tempo"] = out["plp_stability"] * out["tempo"] / 200.0
    out["onset_rate_x_rms"] = out["onset_rate"] * out["rms_mean"]

    return out
