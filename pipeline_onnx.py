#!/usr/bin/env python3
"""
Soniq Lab — v0.4 Pipeline

Complete feature extraction in one pass:
  1. librosa: scalars + vectors + timbre (reuses STFT, adds <0.1s for new features)
  2. Essentia + ONNX: MusiCNN mood/dance/voice/arousal/valence classifications

Output: v0.4 tag schema — everything music-player needs, portable in the m4a file.

Dependencies:
  - librosa: audio loading, all spectral features, tempo, key
  - essentia: mel spectrogram for MusiCNN (MonoLoader + MelBands)
  - onnxruntime: MusiCNN backbone + classifier heads (~3.9MB)
  - numpy, mutagen

Usage:
  python pipeline_onnx.py                    # process all tracks
  python pipeline_onnx.py --limit 5          # process 5 tracks
  python pipeline_onnx.py --track "Prayer"   # filter by title
  python pipeline_onnx.py --dry-run          # extract but don't write tags
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MUSIC_ROOT = Path.home() / "music-player" / "music"
MODELS_DIR = Path(__file__).parent / "models" / "onnx"
GENRE_CACHE_PATH = Path(__file__).parent / "output" / "genre_cache.json"

TAG_VERSION = "0.4"
MP4_ATOM = "----:com.soniq:features"

# ─── ONNX model URLs ─────────────────────────────────────────

BASE_URL = "https://essentia.upf.edu/models"
ONNX_MODELS = {
    "msd-musicnn-1.onnx": f"{BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.onnx",
    "mood_happy-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_happy/mood_happy-msd-musicnn-1.onnx",
    "mood_sad-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_sad/mood_sad-msd-musicnn-1.onnx",
    "mood_relaxed-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.onnx",
    "mood_aggressive-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.onnx",
    "mood_party-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_party/mood_party-msd-musicnn-1.onnx",
    "mood_acoustic-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.onnx",
    "danceability-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/danceability/danceability-msd-musicnn-1.onnx",
    "voice_instrumental-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.onnx",
    "tonal_atonal-msd-musicnn-1.onnx": f"{BASE_URL}/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1.onnx",
    "emomusic-msd-musicnn-2.onnx": f"{BASE_URL}/classification-heads/emomusic/emomusic-msd-musicnn-2.onnx",
}

CLASSIFIERS = {
    "mood_happy": ("mood_happy-msd-musicnn-1.onnx", ["happy", "non_happy"]),
    "mood_sad": ("mood_sad-msd-musicnn-1.onnx", ["non_sad", "sad"]),
    "mood_relaxed": ("mood_relaxed-msd-musicnn-1.onnx", ["non_relaxed", "relaxed"]),
    "mood_aggressive": ("mood_aggressive-msd-musicnn-1.onnx", ["aggressive", "not_aggressive"]),
    "mood_party": ("mood_party-msd-musicnn-1.onnx", ["non_party", "party"]),
    "mood_acoustic": ("mood_acoustic-msd-musicnn-1.onnx", ["acoustic", "non_acoustic"]),
    "danceability": ("danceability-msd-musicnn-1.onnx", ["danceable", "not_danceable"]),
    "voice_instrumental": ("voice_instrumental-msd-musicnn-1.onnx", ["instrumental", "voice"]),
    "tonal_atonal": ("tonal_atonal-msd-musicnn-1.onnx", ["tonal", "atonal"]),
    "emomusic": ("emomusic-msd-musicnn-2.onnx", ["arousal", "valence"]),
}

# MusiCNN parameters
PATCH_SIZE = 187
PATCH_HOP = 93
BATCH_SIZE = 2

# Corpus reference stats for brightness z-score (from 681 tracks in music-player DB)
# These are fixed — brightness scores are comparable across runs.
CORPUS_STATS = {
    "centroid":  {"mean": 1374.4, "std": 528.7},
    "flatness":  {"mean": 0.005678, "std": 0.009095},
    "flux":      {"mean": 40.0, "std": 21.6},
    "zcr":       {"mean": 0.0523, "std": 0.0267},
    "mfcc1":     {"mean": 134.4, "std": 31.6},
}

BRIGHTNESS_WEIGHTS = {
    "centroid": 0.35,
    "mfcc1":    0.25,
    "flatness": 0.15,
    "flux":     0.15,
    "zcr":      0.10,
}


# ─── Model download ──────────────────────────────────────────

def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in ONNX_MODELS.items():
        path = MODELS_DIR / filename
        if path.exists():
            continue
        print(f"  Downloading {filename}...", end=" ")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"{path.stat().st_size / 1024:.0f} KB")
        except Exception as e:
            print(f"FAILED: {e}")
            if path.exists():
                path.unlink()
            return False
    return True


# ─── Librosa feature extraction ──────────────────────────────

def extract_librosa_features(filepath):
    """Extract all librosa features — v0.2 scalars/vectors + v0.4 timbre additions.

    Same multi-point sampling as music-player's extractor.py.
    Adds: centroid_std, rolloff_mean, rolloff_std, bandwidth_mean, bandwidth_std, flux_std.
    """
    import librosa

    try:
        duration = librosa.get_duration(path=filepath)
    except Exception:
        return None

    if duration < 3:
        return None

    load_offset = max(0, duration * 0.05)
    load_duration = min(90, duration - load_offset)
    try:
        y_full, sr = librosa.load(filepath, sr=22050, offset=load_offset,
                                  duration=load_duration, mono=True)
    except Exception:
        return None

    if len(y_full) < sr:
        return None

    actual_loaded = len(y_full) / sr
    segments = _slice_segments(y_full, sr, actual_loaded)
    if not segments:
        return None

    seg_feats = []
    for y_seg in segments:
        sf = _segment_features(y_seg, sr)
        if sf:
            seg_feats.append(sf)

    if not seg_feats:
        return None

    result = {"duration": duration}

    # Average scalar features across segments
    for key in ("centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
                "bandwidth_mean", "bandwidth_std", "flatness_mean",
                "spectral_flux", "flux_std",
                "onset_strength", "beat_strength", "vocal_proxy", "zcr_mean"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # RMS features
    rms_linear_vals = [f["rms_linear"] for f in seg_feats]
    avg_rms_linear = float(np.mean(rms_linear_vals))
    result["rms_mean"] = float(20 * np.log10(avg_rms_linear + 1e-10))
    rms_db_vals = [20 * np.log10(v + 1e-10) for v in rms_linear_vals]
    result["rms_variance"] = float(np.var(rms_db_vals)) if len(rms_db_vals) > 1 else 0.0

    all_rms_db = []
    for f in seg_feats:
        all_rms_db.extend(f["_rms_db_frames"])
    if all_rms_db:
        arr = np.array(all_rms_db)
        p95 = float(np.percentile(arr, 95))
        result["dynamic_range"] = p95 - float(np.percentile(arr, 5))
        result["rms_max"] = p95
    else:
        result["dynamic_range"] = 0.0
        result["rms_max"] = result["rms_mean"]

    # Vector features — average across segments
    for key in ("mfcc_mean", "mfcc_std", "contrast_mean", "chroma_mean", "tonnetz_mean"):
        vecs = [np.array(f[key]) for f in seg_feats]
        result[key] = np.mean(vecs, axis=0).tolist()

    # Tempo + key
    try:
        mid = len(y_full) // 2
        half_window = min(30 * sr, mid)
        y_tempo = y_full[mid - half_window:mid + half_window]
        tempo, _ = librosa.beat.beat_track(y=y_tempo, sr=sr)
        result["tempo"] = float(np.atleast_1d(tempo)[0])
        key, mode = _extract_key_mode(y_tempo, sr)
        result["key"] = key
        result["mode"] = mode
    except Exception:
        result["tempo"] = 0.0
        result["key"] = 0
        result["mode"] = 1

    return result


def _slice_segments(y, sr, loaded_dur, seg_dur=10.0):
    seg_samples = int(seg_dur * sr)
    if loaded_dur < 15:
        return [y]
    segments = []
    for pct in (0.15, 0.50, 0.85):
        center = int(loaded_dur * pct * sr)
        start = max(0, center - seg_samples // 2)
        end = start + seg_samples
        if end > len(y):
            start = max(0, len(y) - seg_samples)
            end = len(y)
        seg = y[start:end]
        if len(seg) >= sr:
            segments.append(seg)
    return segments


def _segment_features(y, sr):
    """Extract features from a single segment. One STFT, all features."""
    import librosa

    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_power = S ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # RMS
        rms = np.sqrt(np.mean(S_power, axis=0))
        rms_linear = float(np.mean(rms))
        rms_db_frames = (20 * np.log10(rms + 1e-10)).tolist()

        # --- Existing v0.2 features ---
        centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
        centroid_mean = float(np.mean(centroid))

        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness_mean = float(np.mean(flatness))

        if S.shape[1] > 1:
            diff = np.diff(S, axis=1)
            flux_per_frame = np.sqrt(np.sum(diff ** 2, axis=0))
            spectral_flux = float(np.mean(flux_per_frame))
            flux_std = float(np.std(flux_per_frame))
        else:
            spectral_flux = 0.0
            flux_std = 0.0

        S_db = librosa.power_to_db(S_power)
        onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)
        onset_strength = float(np.mean(onset_env))
        beat_strength = float(np.percentile(onset_env, 75))

        H, _ = librosa.decompose.hpss(S)
        h_energy = float(np.sum(H ** 2))
        total_energy = float(np.sum(S_power))
        harmonic_ratio = h_energy / total_energy if total_energy > 1e-10 else 0.0
        h_flatness = float(np.mean(librosa.feature.spectral_flatness(S=H)[0]))
        vocal_proxy = harmonic_ratio * (1.0 - h_flatness)

        mel_S = librosa.feature.melspectrogram(S=S_power, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast_mean = np.mean(contrast, axis=1).tolist()

        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

        # --- New v0.4 timbre features (from same STFT) ---
        centroid_std = float(np.std(centroid))

        rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))

        return {
            "rms_linear": rms_linear,
            "_rms_db_frames": rms_db_frames,
            # v0.2 scalars
            "centroid_mean": centroid_mean,
            "flatness_mean": flatness_mean,
            "spectral_flux": spectral_flux,
            "onset_strength": onset_strength,
            "beat_strength": beat_strength,
            "vocal_proxy": vocal_proxy,
            "zcr_mean": zcr_mean,
            # v0.2 vectors
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "contrast_mean": contrast_mean,
            "chroma_mean": chroma_mean,
            "tonnetz_mean": tonnetz_mean,
            # v0.4 timbre additions
            "centroid_std": centroid_std,
            "rolloff_mean": rolloff_mean,
            "rolloff_std": rolloff_std,
            "bandwidth_mean": bandwidth_mean,
            "bandwidth_std": bandwidth_std,
            "flux_std": flux_std,
        }
    except Exception:
        return None


def _extract_key_mode(y, sr):
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_corr, best_key, best_mode = -2.0, 0, 1
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        maj_corr = float(np.corrcoef(rotated, major)[0, 1])
        min_corr = float(np.corrcoef(rotated, minor)[0, 1])
        if maj_corr > best_corr:
            best_corr, best_key, best_mode = maj_corr, shift, 1
        if min_corr > best_corr:
            best_corr, best_key, best_mode = min_corr, shift, 0
    return best_key, best_mode


# ─── MusiCNN classification (Essentia mel + ONNX) ────────────

def extract_mel(audio_path):
    """Compute MusiCNN mel spectrogram using Essentia DSP.

    Exact match to TensorflowInputMusiCNN (zero diff) via three key params:
    warpingFormula="slaneyMel", weighting="linear", normalize="unit_tri".
    """
    from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands

    audio = MonoLoader(filename=str(audio_path), sampleRate=16000)()

    windowing = Windowing(type="hann", size=512, zeroPadding=0, normalized=False)
    spectrum = Spectrum(size=512)
    mel_bands = MelBands(
        numberBands=96, sampleRate=16000,
        lowFrequencyBound=0, highFrequencyBound=8000,
        inputSize=257,
        warpingFormula="slaneyMel",
        weighting="linear",
        normalize="unit_tri",
    )

    frames = []
    for start in range(0, len(audio) - 512 + 1, 256):
        spec = spectrum(windowing(audio[start:start + 512]))
        frames.append(mel_bands(spec))

    mel = np.array(frames, dtype=np.float32)
    return np.log10(1 + 10000 * mel).astype(np.float32)


def make_patches(mel):
    patches = []
    for start in range(0, mel.shape[0] - PATCH_SIZE + 1, PATCH_HOP):
        patches.append(mel[start:start + PATCH_SIZE])
    if not patches and mel.shape[0] > 0:
        padded = np.zeros((PATCH_SIZE, 96), dtype=np.float32)
        padded[:mel.shape[0]] = mel
        patches.append(padded)
    return np.array(patches, dtype=np.float32) if patches else np.empty((0, PATCH_SIZE, 96), dtype=np.float32)


def run_onnx_classification(mel, backbone, heads):
    """Backbone → embeddings → all classifier heads."""
    patches = make_patches(mel)
    if len(patches) == 0:
        return {}

    # Backbone in small batches
    all_emb = []
    for start in range(0, len(patches), BATCH_SIZE):
        batch = patches[start:start + BATCH_SIZE]
        result = backbone.run(None, {"melspectrogram": batch})
        all_emb.append(result[1])
    embeddings = np.concatenate(all_emb)

    # Classifier heads
    results = {}
    for name, (model_file, labels) in CLASSIFIERS.items():
        if name not in heads:
            continue
        preds = heads[name].run(None, {"embeddings": embeddings})
        avg = np.mean(preds[0], axis=0)

        if name == "emomusic":
            results["arousal"] = round(float(avg[0]), 2)
            results["valence"] = round(float(avg[1]), 2)
        elif len(labels) == 2:
            pos_idx, pos_label = {
                "mood_happy": (0, "happy"),
                "mood_sad": (1, "sad"),
                "mood_relaxed": (1, "relaxed"),
                "mood_aggressive": (0, "aggressive"),
                "mood_party": (1, "party"),
                "mood_acoustic": (0, "acoustic"),
                "danceability": (0, "danceable"),
                "voice_instrumental": (0, "instrumental"),
                "tonal_atonal": (0, "tonal"),
            }.get(name, (0, labels[0]))
            results[pos_label] = round(float(avg[pos_idx]), 4)

    return results


# ─── MusicBrainz genre lookup ─────────────────────────────────

_genre_cache = {}
_last_mb_request = 0.0
MB_USER_AGENT = "SoniqLab/0.4 (https://github.com/msieur-gab/soniq-lab)"
MB_BASE = "https://musicbrainz.org/ws/2"


def _load_genre_cache():
    global _genre_cache
    if _genre_cache:
        return
    if GENRE_CACHE_PATH.exists():
        try:
            _genre_cache = json.loads(GENRE_CACHE_PATH.read_text())
        except Exception:
            _genre_cache = {}


def _save_genre_cache():
    GENRE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENRE_CACHE_PATH.write_text(json.dumps(_genre_cache, indent=2, ensure_ascii=False))


def _mb_get(url):
    """Rate-limited MusicBrainz API GET (1 req/sec)."""
    global _last_mb_request
    elapsed = time.time() - _last_mb_request
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _last_mb_request = time.time()
            return json.loads(resp.read())
    except Exception:
        _last_mb_request = time.time()
        return None


def fetch_album_genre(artist, album):
    """Fetch genre tags for an album from MusicBrainz release groups.

    Returns list of genre strings sorted by vote count, e.g. ["jazz", "post-bop"].
    Results are cached per artist+album key.
    """
    _load_genre_cache()
    cache_key = f"{artist}|||{album}"
    if cache_key in _genre_cache:
        return _genre_cache[cache_key]

    # Step 1: search release groups by artist + album
    query = urllib.parse.quote(f'releasegroup:"{album}" AND artist:"{artist}"')
    url = f'{MB_BASE}/release-group/?query={query}&fmt=json&limit=5'
    data = _mb_get(url)

    genres = []
    if data and "release-groups" in data:
        for rg in data["release-groups"]:
            # Check artist match (fuzzy — MusicBrainz sometimes returns close matches)
            rg_artists = " ".join(c.get("name", "") for c in rg.get("artist-credit", []))
            if artist.lower().split(",")[0].strip() not in rg_artists.lower():
                continue

            # Get tags from this release group
            rg_id = rg.get("id")
            if not rg_id:
                continue

            # Fetch full release group with tags
            rg_url = f'{MB_BASE}/release-group/{rg_id}?inc=tags&fmt=json'
            rg_data = _mb_get(rg_url)
            if rg_data and "tags" in rg_data:
                tag_list = [(t["name"], t.get("count", 0)) for t in rg_data["tags"]]
                tag_list.sort(key=lambda x: -x[1])
                genres = [t[0] for t in tag_list if t[1] > 0]
                break  # Use first matching release group

    _genre_cache[cache_key] = genres
    _save_genre_cache()
    return genres


# ─── Tag building ─────────────────────────────────────────────

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


def compute_brightness(features):
    """Compute brightness score from librosa features using fixed corpus stats.

    Z-score normalizes against 681-track reference corpus, then weighted
    combination through sigmoid → 0 (dark) to 1 (bright).
    """
    values = {
        "centroid": features.get("centroid_mean", 0),
        "flatness": features.get("flatness_mean", 0),
        "flux": features.get("spectral_flux", 0),
        "zcr": features.get("zcr_mean", 0),
        "mfcc1": features.get("mfcc_mean", [0, 0])[1] if len(features.get("mfcc_mean", [])) > 1 else 0,
    }

    score = 0
    for key, weight in BRIGHTNESS_WEIGHTS.items():
        stats = CORPUS_STATS[key]
        z = (values[key] - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
        score += z * weight

    return round(float(1 / (1 + np.exp(-score))), 4)


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

    # Brightness (z-score against 681-track corpus + sigmoid)
    tag["cls"]["brightness"] = compute_brightness(librosa_features)

    # Genre (MusicBrainz album-level tags)
    if genre:
        tag["cls"]["genre"] = genre

    return tag


def write_tag(filepath, tag):
    """Write v0.4 tag to m4a file."""
    from mutagen.mp4 import MP4
    tag_json = json.dumps(tag, separators=(",", ":"))
    audio = MP4(filepath)
    if audio.tags is None:
        audio.add_tags()
    audio.tags[MP4_ATOM] = [tag_json.encode("utf-8")]
    audio.save()
    return len(tag_json)


# ─── Pipeline orchestration ───────────────────────────────────

def load_onnx_pipeline():
    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    print("Loading ONNX models...")
    t0 = time.time()
    backbone = ort.InferenceSession(
        str(MODELS_DIR / "msd-musicnn-1.onnx"),
        providers=["CPUExecutionProvider"],
    )
    heads = {}
    for name, (model_file, _) in CLASSIFIERS.items():
        path = MODELS_DIR / model_file
        if path.exists():
            heads[name] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"],
            )
    print(f"  Loaded backbone + {len(heads)} heads in {time.time() - t0:.1f}s")
    return backbone, heads


def find_tracks(track_filter=None):
    tracks = []
    for root, dirs, files in os.walk(MUSIC_ROOT):
        for f in sorted(files):
            if not f.endswith(".m4a"):
                continue
            path = Path(root) / f
            rel = path.relative_to(MUSIC_ROOT)
            parts = rel.parts
            artist = parts[0] if len(parts) >= 1 else "Unknown"
            album = parts[1] if len(parts) >= 2 else "Unknown"
            title = f.rsplit(".", 1)[0]
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break
            if track_filter and track_filter.lower() not in title.lower():
                continue
            tracks.append({"path": str(path), "artist": artist, "album": album, "title": title})
    return tracks


def process_track(audio_path, backbone, heads, artist=None, album=None):
    """Full v0.4 pipeline: librosa features + ONNX classifications + genre."""
    # 1. Librosa features (~4-5s)
    t0 = time.time()
    features = extract_librosa_features(audio_path)
    t_librosa = time.time() - t0

    if not features:
        return None, None, {}

    # 2. MusiCNN mel + classification (~15s)
    t0 = time.time()
    mel = extract_mel(audio_path)
    t_mel = time.time() - t0

    t0 = time.time()
    cls = run_onnx_classification(mel, backbone, heads)
    t_cls = time.time() - t0
    del mel

    # 3. Genre from MusicBrainz (cached, ~0s after first lookup per album)
    genre = []
    if artist and album:
        genre = fetch_album_genre(artist, album)

    # 4. Build tag
    tag = build_tag(features, cls, genre=genre)

    timings = {"librosa": t_librosa, "mel": t_mel, "onnx": t_cls}
    return tag, features, timings


def main():
    parser = argparse.ArgumentParser(description="Soniq Lab — v0.4 Pipeline")
    parser.add_argument("--track", type=str, help="Filter by track title")
    parser.add_argument("--limit", type=int, default=0, help="Max tracks (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't write tags")
    parser.add_argument("--download-only", action="store_true", help="Just download models")
    args = parser.parse_args()

    print("=" * 60)
    print("Soniq Lab — v0.4 Pipeline")
    print("  librosa features + ONNX MusiCNN classifications")
    print("=" * 60)

    if not download_models():
        print("ERROR: Failed to download models")
        sys.exit(1)

    if args.download_only:
        print("Models ready.")
        return

    tracks = find_tracks(args.track)
    if not tracks:
        print(f"No tracks found{' matching ' + args.track if args.track else ''}")
        sys.exit(1)

    if args.limit:
        tracks = tracks[:args.limit]

    print(f"\nFound {len(tracks)} track{'s' if len(tracks) != 1 else ''}")

    backbone, heads = load_onnx_pipeline()

    total_time = 0
    count = 0
    tag_sizes = []

    for i, track in enumerate(tracks):
        print(f"\n[{i+1}/{len(tracks)}] {track['artist']} — {track['title']}")
        t0 = time.time()

        try:
            tag, features, timings = process_track(
                track["path"], backbone, heads,
                artist=track["artist"], album=track["album"],
            )
            elapsed = time.time() - t0

            if not tag:
                print("  SKIP — extraction failed")
                continue

            total_time += elapsed
            count += 1

            # Print summary
            cls = tag.get("cls", {})
            s = tag.get("s", {})
            print(f"  {elapsed:.1f}s (librosa {timings['librosa']:.1f}s + mel {timings['mel']:.1f}s + onnx {timings['onnx']:.1f}s)")
            print(f"  cls: happy={cls.get('happy',0):.2f} sad={cls.get('sad',0):.2f} "
                  f"relaxed={cls.get('relaxed',0):.2f} dance={cls.get('danceable',0):.2f} "
                  f"instr={cls.get('instrumental',0):.2f} tonal={cls.get('tonal',0):.2f}")
            print(f"  s:   centroid={s.get('centroid',0):.0f}Hz "
                  f"centroid_std={s.get('centroid_std',0):.0f}Hz "
                  f"tempo={s.get('tempo',0):.0f}bpm "
                  f"key={s.get('key',0)} mode={s.get('mode',0)}")
            genre = cls.get("genre", [])
            if genre:
                print(f"  genre: {', '.join(genre)}")
            print(f"  brightness: {cls.get('brightness', 0):.3f}")

            # Write tag
            if not args.dry_run:
                tag_bytes = write_tag(track["path"], tag)
                tag_sizes.append(tag_bytes)
                print(f"  tag: {tag_bytes} bytes written")
            else:
                tag_json = json.dumps(tag, separators=(",", ":"))
                tag_sizes.append(len(tag_json))

        except Exception as e:
            print(f"  ERROR: {e}")

    if count:
        avg = total_time / count
        print(f"\n{'='*60}")
        print(f"Processed {count} tracks in {total_time:.1f}s ({avg:.1f}s/track)")
        if tag_sizes:
            print(f"Tag size: {min(tag_sizes)}–{max(tag_sizes)} bytes (avg {sum(tag_sizes)//len(tag_sizes)})")
        if args.dry_run:
            print("(dry run — no tags written)")


if __name__ == "__main__":
    main()
