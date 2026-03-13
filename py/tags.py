"""Tag building and writing — v0.5 schema.

Tag structure:
  {src, v, at, s: {scalars}, vec: {vectors}, cls: {classifications}}

Storage: m4a → ----:com.soniq:features custom atom.
"""

import json
from datetime import datetime, timezone

TAG_VERSION = "0.5"
MP4_ATOM = "----:com.soniq:features"

SCALAR_SHORT = {
    "duration": "duration", "tempo": "tempo", "key": "key", "mode": "mode",
    "rms_mean": "rms_mean", "rms_max": "rms_max", "rms_variance": "rms_var",
    "dynamic_range": "dyn_range", "centroid_mean": "centroid",
    "centroid_std": "centroid_std",
    "rolloff_mean": "rolloff", "rolloff_std": "rolloff_std",
    "bandwidth_mean": "bandwidth", "bandwidth_std": "bandwidth_std",
    "flatness_mean": "flatness", "spectral_flux": "flux", "flux_std": "flux_std",
    "onset_strength": "onset", "beat_strength": "beat",
    "vocal_proxy": "vocal", "zcr_mean": "zcr",
    # v0.5 additions
    "low_energy_rate": "low_energy_rate",
    "energy_skew": "energy_skew",
    "energy_kurtosis": "energy_kurtosis",
    "bass_ratio": "bass_ratio",
    "mid_ratio": "mid_ratio",
    "treble_ratio": "treble_ratio",
    "bass_mid_ratio": "bass_mid_ratio",
    "spectral_skew": "spectral_skew",
    "spectral_kurtosis": "spectral_kurtosis",
    "spectral_entropy": "spectral_entropy",
    "spectral_crest": "spectral_crest",
    "mfcc_delta_var": "mfcc_delta_var",
    "mfcc_delta2_var": "mfcc_delta2_var",
    "mod_flatness": "mod_flatness",
    "mod_crest": "mod_crest",
    "mod_centroid": "mod_centroid",
    "harm_energy": "harm_energy",
    "perc_energy": "perc_energy",
    "harm_perc_ratio": "harm_perc_ratio",
    "harm_fraction": "harm_fraction",
    "beat_regularity": "beat_regularity",
    "rhythm_complexity": "rhythm_complexity",
    "plp_mean": "plp_mean",
    "plp_stability": "plp_stability",
    "onset_rate": "onset_rate",
}


def build_tag(librosa_features, classifications, genre=None):
    """Build v0.5 tag from librosa features + classifier results + genre."""
    def r(v, d=4):
        return round(v, d) if isinstance(v, float) else v

    tag = {
        "src": "soniq",
        "v": TAG_VERSION,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "s": {},
        "vec": {},
        "cls": {},
    }

    # Scalars
    for full_key, short_key in SCALAR_SHORT.items():
        if full_key in librosa_features:
            tag["s"][short_key] = r(librosa_features[full_key])

    # Vectors
    if "mfcc_mean" in librosa_features:
        tag["vec"]["mfcc_m"] = [r(v) for v in librosa_features["mfcc_mean"]]
    if "mfcc_std" in librosa_features:
        tag["vec"]["mfcc_s"] = [r(v) for v in librosa_features["mfcc_std"]]
    if "contrast_mean" in librosa_features:
        tag["vec"]["contrast"] = [r(v) for v in librosa_features["contrast_mean"]]
    if "chroma_mean" in librosa_features:
        tag["vec"]["chroma"] = [r(v) for v in librosa_features["chroma_mean"]]
    if "tonnetz_mean" in librosa_features:
        tag["vec"]["tonnetz"] = [r(v) for v in librosa_features["tonnetz_mean"]]
    # v0.5 delta MFCC vectors
    if "mfcc_delta_mean" in librosa_features:
        tag["vec"]["mfcc_d"] = [r(v) for v in librosa_features["mfcc_delta_mean"]]
    if "mfcc_delta2_mean" in librosa_features:
        tag["vec"]["mfcc_d2"] = [r(v) for v in librosa_features["mfcc_delta2_mean"]]

    # Classifications (all from classifiers module)
    for key in ("happy", "sad", "relaxed", "aggressive", "party", "acoustic",
                "danceable", "instrumental", "vocal", "tonal", "atonal",
                "arousal", "valence",
                "radiant", "somber", "brilliant", "warm",
                "energetic", "contemplative",
                "hypnotic", "varied"):
        if key in classifications:
            tag["cls"][key] = r(classifications[key])

    # Energy components (for UI visualization)
    if "_energy_components" in classifications:
        tag["cls"]["nrg"] = classifications["_energy_components"]

    # Hypnotic path (rhythmic vs timbral)
    if "_hypnotic_path" in classifications:
        tag["cls"]["hypnotic_path"] = classifications["_hypnotic_path"]

    # Genre
    if genre:
        tag["cls"]["genre"] = genre

    return tag


def write_tag(filepath, tag):
    """Write v0.5 tag to m4a file. Returns byte size."""
    from mutagen.mp4 import MP4
    tag_json = json.dumps(tag, separators=(",", ":"))
    audio = MP4(filepath)
    if audio.tags is None:
        audio.add_tags()
    audio.tags[MP4_ATOM] = [tag_json.encode("utf-8")]
    audio.save()
    return len(tag_json)


def read_tag(filepath):
    """Read existing tag from m4a file. Returns dict or None."""
    try:
        from mutagen.mp4 import MP4
        audio = MP4(filepath)
        if audio.tags and MP4_ATOM in audio.tags:
            raw = audio.tags[MP4_ATOM][0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
    except Exception:
        pass
    return None
