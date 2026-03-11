"""Tag building and writing — v0.4 schema.

Tag structure:
  {src, v, at, s: {scalars}, vec: {vectors}, cls: {classifications}}

Storage: m4a → ----:com.soniq:features custom atom.
"""

import json
from datetime import datetime, timezone

from .brightness import compute_brightness

TAG_VERSION = "0.4"
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
}


def build_tag(librosa_features, classifications, genre=None):
    """Build v0.4 tag from librosa features + ONNX classifications + genre."""
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

    # Classifications (MusiCNN via ONNX)
    for key in ("happy", "sad", "relaxed", "aggressive", "party", "acoustic",
                "danceable", "instrumental", "tonal", "arousal", "valence"):
        if key in classifications:
            tag["cls"][key] = r(classifications[key])

    # Brightness
    tag["cls"]["brightness"] = compute_brightness(librosa_features)

    # Genre
    if genre:
        tag["cls"]["genre"] = genre

    return tag


def write_tag(filepath, tag):
    """Write v0.4 tag to m4a file. Returns byte size."""
    from mutagen.mp4 import MP4
    tag_json = json.dumps(tag, separators=(",", ":"))
    audio = MP4(filepath)
    if audio.tags is None:
        audio.add_tags()
    audio.tags[MP4_ATOM] = [tag_json.encode("utf-8")]
    audio.save()
    return len(tag_json)


def read_tag(filepath):
    """Read existing v0.4 tag from m4a file. Returns dict or None."""
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
