#!/usr/bin/env python3
"""
Soniq Lab — V2 classification pipeline (ONNX + librosa).

Single MusiCNN backbone via ONNX Runtime — no TensorFlow dependency.
Timbre via librosa DSP feature vector instead of EffNet.
Genre via Discogs API (not implemented yet, placeholder).

Dependencies: essentia (base, no TF), onnxruntime, librosa, numpy
Total model size: ~3.9MB (vs ~21MB for TF models + 500MB TF runtime)

Architecture:
  1. Essentia DSP: audio → mel-spectrogram (16kHz, 96 bands)
  2. ONNX MusiCNN backbone: mel patches → 200-dim embeddings
  3. ONNX classifier heads: embeddings → predictions
  4. Librosa: audio → 35-dim timbre vector → bright/dark score
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

MUSIC_ROOT = Path(__file__).parent / "music"
MODELS_DIR = Path(__file__).parent / "models-onnx"
RESULTS_DIR = Path(__file__).parent / "results-v2"

# ONNX model URLs — MusiCNN backbone + all classifier heads
MODEL_URLS = {
    "msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.onnx",
    "mood_happy-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1.onnx",
    "mood_sad-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1.onnx",
    "mood_relaxed-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.onnx",
    "mood_aggressive-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.onnx",
    "mood_party-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-msd-musicnn-1.onnx",
    "mood_acoustic-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.onnx",
    "danceability-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/danceability/danceability-msd-musicnn-1.onnx",
    "voice_instrumental-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.onnx",
    "tonal_atonal-msd-musicnn-1.onnx":
        "https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1.onnx",
    "emomusic-msd-musicnn-2.onnx":
        "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.onnx",
}

# Classifier heads: name → (onnx_file, labels)
# Label ordering verified from official Essentia metadata JSONs
CLASSIFIERS = {
    "mood_happy":       ("mood_happy-msd-musicnn-1.onnx",       ["happy", "non_happy"]),
    "mood_sad":         ("mood_sad-msd-musicnn-1.onnx",         ["non_sad", "sad"]),
    "mood_relaxed":     ("mood_relaxed-msd-musicnn-1.onnx",     ["non_relaxed", "relaxed"]),
    "mood_aggressive":  ("mood_aggressive-msd-musicnn-1.onnx",  ["aggressive", "not_aggressive"]),
    "mood_party":       ("mood_party-msd-musicnn-1.onnx",       ["non_party", "party"]),
    "mood_acoustic":    ("mood_acoustic-msd-musicnn-1.onnx",    ["acoustic", "non_acoustic"]),
    "danceability":     ("danceability-msd-musicnn-1.onnx",     ["danceable", "not_danceable"]),
    "voice_instrumental": ("voice_instrumental-msd-musicnn-1.onnx", ["instrumental", "voice"]),
    "tonal_atonal":     ("tonal_atonal-msd-musicnn-1.onnx",     ["tonal", "atonal"]),
    "arousal_valence":  ("emomusic-msd-musicnn-2.onnx",         "regression_av"),
}

# MusiCNN mel-spectrogram parameters
SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_SIZE = 256
N_MELS = 96
FMIN = 0
FMAX = 8000
PATCH_SIZE = 187     # frames per patch (~3s)
PATCH_HOP = 93       # 50% overlap
MAX_DURATION = 120   # seconds — cap to limit memory usage


def download_models():
    """Download ONNX models if not present."""
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODEL_URLS.items():
        path = MODELS_DIR / filename
        if path.exists():
            continue
        print(f"  Downloading {filename}...")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, path)
                size_kb = path.stat().st_size / 1024
                print(f"  → {size_kb:.0f} KB")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  → Retry {attempt + 2}/3 ({e})")
                    time.sleep(2)
                else:
                    print(f"  → FAILED: {e}")
                    if path.exists():
                        path.unlink()
                    return False
    return True


def compute_mel_spectrogram(audio_path):
    """Compute mel-spectrogram matching MusiCNN input format using Essentia DSP.

    MusiCNN expects: log10(1 + 10000 * mel_bands) with 96 bands, 16kHz,
    frame=512, hop=256. Uses Essentia (base, no TF) for exact mel match.
    """
    from essentia.standard import (
        MonoLoader, Windowing, Spectrum, MelBands,
    )

    audio = MonoLoader(filename=str(audio_path), sampleRate=SAMPLE_RATE)()
    max_samples = MAX_DURATION * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    windowing = Windowing(type="hann", size=FRAME_SIZE, zeroPadding=0,
                          normalized=False)
    spectrum = Spectrum(size=FRAME_SIZE)
    mel_bands = MelBands(numberBands=N_MELS, sampleRate=SAMPLE_RATE,
                         lowFrequencyBound=FMIN, highFrequencyBound=FMAX,
                         inputSize=FRAME_SIZE // 2 + 1)

    frames = []
    for start in range(0, len(audio) - FRAME_SIZE + 1, HOP_SIZE):
        frame = audio[start:start + FRAME_SIZE]
        spec = spectrum(windowing(frame))
        mb = mel_bands(spec)
        frames.append(mb)

    mel = np.array(frames, dtype=np.float32)
    mel_norm = np.log10(1 + 10000 * mel).astype(np.float32)

    return mel_norm


def make_patches(mel):
    """Cut mel-spectrogram into overlapping patches for MusiCNN."""
    patches = []
    for start in range(0, mel.shape[0] - PATCH_SIZE + 1, PATCH_HOP):
        patches.append(mel[start:start + PATCH_SIZE])
    if not patches:
        # Track too short — pad to one patch
        padded = np.zeros((PATCH_SIZE, N_MELS), dtype=np.float32)
        padded[:mel.shape[0]] = mel
        patches.append(padded)
    return np.array(patches, dtype=np.float32)


def compute_timbre_vector(audio_path):
    """Compute 35-dim timbre vector using librosa (replaces EffNet timbre)."""
    import librosa
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    y, sr = librosa.load(str(audio_path), sr=22050, duration=MAX_DURATION)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def ms(x):
        return [float(np.mean(x)), float(np.std(x))]

    return np.array(
        ms(centroid) + ms(rolloff) + ms(bandwidth)
        + [float(np.mean(flatness))]
        + ms(flux)
        + [val for mfcc in mfccs for val in ms(mfcc)]
    )


class OnnxPipeline:
    """ONNX-based MusiCNN classification pipeline.

    Memory-efficient: only backbone stays loaded; classifier heads are
    loaded on-demand and released after inference to avoid OOM on
    constrained machines.
    """

    def __init__(self):
        import onnxruntime as ort
        self._ort = ort
        ort.set_default_logger_severity(3)

        print("Loading ONNX backbone...")
        t0 = time.time()

        self.backbone = ort.InferenceSession(
            str(MODELS_DIR / "msd-musicnn-1.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print(f"  Backbone loaded in {time.time() - t0:.1f}s")

    def extract_embeddings(self, mel_patches):
        """Run MusiCNN backbone on mel patches → embeddings."""
        result = self.backbone.run(None, {"melspectrogram": mel_patches})
        return result[1]  # shape (N, 200)

    def classify(self, embeddings):
        """Run all classifier heads on embeddings (one at a time for memory)."""
        results = {}
        for name, (onnx_file, labels) in CLASSIFIERS.items():
            path = MODELS_DIR / onnx_file
            if not path.exists():
                continue
            try:
                session = self._ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                preds = session.run(None, {"embeddings": embeddings})[0]
                avg = np.mean(preds, axis=0)
                results[name] = [round(float(v), 4) for v in avg]
                del session
            except Exception as e:
                results[name] = f"error: {e}"
        return results


def classify_track(audio_path, pipeline):
    """Full classification of a single track."""
    # MusiCNN classification via ONNX
    mel = compute_mel_spectrogram(audio_path)
    patches = make_patches(mel)
    embeddings = pipeline.extract_embeddings(patches)
    preds = pipeline.classify(embeddings)

    # Timbre via librosa DSP
    timbre_vec = compute_timbre_vector(audio_path)
    preds["timbre_vector"] = [round(float(v), 4) for v in timbre_vec]

    return preds


def find_tracks(track_filter=None):
    """Find all m4a files in the music library."""
    tracks = []
    for root, dirs, files in os.walk(MUSIC_ROOT):
        for f in sorted(files):
            if not f.endswith(".m4a"):
                continue
            path = Path(root) / f
            rel = path.relative_to(MUSIC_ROOT)
            parts = rel.parts
            artist = parts[0] if len(parts) >= 3 else "Unknown"
            album = parts[1] if len(parts) >= 3 else "Unknown"
            title = f.rsplit(".", 1)[0]
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break
            if track_filter and track_filter.lower() not in title.lower():
                continue
            tracks.append({"path": str(path), "artist": artist, "album": album, "title": title})
    return tracks


def format_result(name, values, labels):
    """Pretty-print a classifier result."""
    if isinstance(values, str):
        return f"  {name:25s}  {values}"
    if name == "timbre_vector":
        return f"  {name:25s}  [{len(values)} dims]"
    if labels == "regression_av":
        return f"  {name:25s}  arousal={values[0]:.2f}  valence={values[1]:.2f}"
    if isinstance(labels, list) and len(labels) == 2 and len(values) == 2:
        idx = 1 if values[1] > values[0] else 0
        return f"  {name:25s}  {labels[idx]:20s}  ({values[idx]:.1%})"
    return f"  {name:25s}  {values}"


def main():
    parser = argparse.ArgumentParser(description="Soniq Lab — V2 ONNX classification")
    parser.add_argument("--track", type=str, help="Filter by track title")
    parser.add_argument("--limit", type=int, default=0, help="Max tracks (0=all)")
    parser.add_argument("--download-only", action="store_true", help="Just download models")
    args = parser.parse_args()

    print("Soniq Lab V2 — ONNX MusiCNN + librosa timbre")
    print(f"Music root: {MUSIC_ROOT}")

    if not MUSIC_ROOT.exists():
        print("ERROR: Music directory not found")
        sys.exit(1)

    print("\nChecking ONNX models...")
    if not download_models():
        print("ERROR: Failed to download models")
        sys.exit(1)

    if args.download_only:
        return

    tracks = find_tracks(args.track)
    if not tracks:
        print(f"No tracks found{' matching ' + args.track if args.track else ''}")
        sys.exit(1)

    if args.limit:
        tracks = tracks[:args.limit]

    print(f"Found {len(tracks)} track{'s' if len(tracks) != 1 else ''}")

    pipeline = OnnxPipeline()

    RESULTS_DIR.mkdir(exist_ok=True)
    total_time = 0
    count = 0

    for i, track in enumerate(tracks):
        print(f"\n[{i+1}/{len(tracks)}] {track['artist']} — {track['title']}")
        t0 = time.time()

        try:
            preds = classify_track(track["path"], pipeline)
            elapsed = time.time() - t0
            total_time += elapsed
            count += 1
            print(f"  Classified in {elapsed:.1f}s")

            for name in sorted(preds.keys()):
                labels = CLASSIFIERS.get(name, (None, []))[1] if name != "timbre_vector" else []
                print(format_result(name, preds[name], labels))

            result = {
                "artist": track["artist"],
                "album": track["album"],
                "title": track["title"],
                "predictions": preds,
                "time_s": round(elapsed, 2),
            }

            safe_name = f"{track['artist']} - {track['title']}".replace("/", "_")
            out_path = RESULTS_DIR / f"{safe_name}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")

    if count:
        avg = total_time / count
        print(f"\n{'='*60}")
        print(f"Processed {count} tracks in {total_time:.1f}s ({avg:.1f}s/track)")
        print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
