"""MusiCNN classification — Essentia mel spectrogram + ONNX Runtime inference.

The mel preprocessing replicates TensorflowInputMusiCNN exactly (zero diff)
via three critical MelBands parameters: slaneyMel, linear, unit_tri.
"""

import time
import urllib.request
from pathlib import Path

import numpy as np

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
BATCH_SIZE = 16


def download_models(models_dir):
    """Download ONNX models if not present. Returns True on success."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ONNX_MODELS.items():
        path = models_dir / filename
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


def load_onnx_pipeline(models_dir):
    """Load ONNX backbone + all classifier heads. Returns (backbone, heads)."""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    models_dir = Path(models_dir)
    t0 = time.time()
    backbone = ort.InferenceSession(
        str(models_dir / "msd-musicnn-1.onnx"),
        providers=["CPUExecutionProvider"],
    )
    heads = {}
    for name, (model_file, _) in CLASSIFIERS.items():
        path = models_dir / model_file
        if path.exists():
            heads[name] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"],
            )
    print(f"  Loaded backbone + {len(heads)} heads in {time.time() - t0:.1f}s")
    return backbone, heads


def extract_mel(audio_path, max_seconds=300):
    """Compute MusiCNN mel spectrogram using Essentia DSP.

    Exact match to TensorflowInputMusiCNN (zero diff) via three key params:
    warpingFormula="slaneyMel", weighting="linear", normalize="unit_tri".

    Returns mel array or None on error.
    """
    try:
        import gc
        from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands

        audio = MonoLoader(filename=str(audio_path), sampleRate=16000)()

        # Duration cap — truncate after loading to avoid OOM on mel patching
        if max_seconds:
            max_samples = int(max_seconds * 16000)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

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

        # Pre-allocate array — avoids doubling memory with list append + np.array
        n_frames = (len(audio) - 512) // 256 + 1
        mel = np.zeros((n_frames, 96), dtype=np.float32)
        for i in range(n_frames):
            start = i * 256
            mel[i] = mel_bands(spectrum(windowing(audio[start:start + 512])))

        # Free raw audio before log transform
        del audio
        gc.collect()

        # In-place log compression — no second array allocation
        np.log10(1 + 10000 * mel, out=mel)
        return mel
    except Exception:
        return None


def make_patches(mel):
    """Split mel spectrogram into overlapping patches for MusiCNN."""
    patches = []
    for start in range(0, mel.shape[0] - PATCH_SIZE + 1, PATCH_HOP):
        patches.append(mel[start:start + PATCH_SIZE])
    if not patches and mel.shape[0] > 0:
        padded = np.zeros((PATCH_SIZE, 96), dtype=np.float32)
        padded[:mel.shape[0]] = mel
        patches.append(padded)
    return np.array(patches, dtype=np.float32) if patches else np.empty((0, PATCH_SIZE, 96), dtype=np.float32)


def run_onnx_classification(mel, backbone, heads):
    """Run backbone → embeddings → all classifier heads. Returns dict of results."""
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
    del patches, all_emb

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
