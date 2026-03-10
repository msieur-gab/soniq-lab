#!/usr/bin/env python3
"""
Soniq Lab — Essentia classification pipeline.

Runs pre-trained Essentia TensorFlow models on m4a files and outputs
high-level classifications (mood, genre, timbre, voice/instrumental, etc.)

Points to ~/music-player/music/ — reads audio, writes nothing.
Results saved to results.json for comparison with librosa Soniq features.

Architecture notes (from MTG docs):
  - Two-stage inference: embedding extractor → classifier heads
  - MusiCNN and EffNet are DIFFERENT backbones — don't mix them
  - All models expect 16kHz mono input
  - Models output per-patch scores (~3s windows) — average for track-level
  - Suppress TF GPU warnings with TF_CPP_MIN_LOG_LEVEL=3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress GPU warnings

MUSIC_ROOT = Path.home() / "music-player" / "music"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"

# Two model families — each classifier must match its backbone
# MusiCNN backbone: msd-musicnn-1.pb
# EffNet backbone: discogs-effnet-bs64-1.pb
MODEL_URLS = {
    # === MusiCNN family ===
    "msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb",

    # MusiCNN classifier heads
    "mood_happy-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1.pb",
    "mood_sad-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1.pb",
    "mood_relaxed-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.pb",
    "mood_aggressive-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.pb",
    "mood_party-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-msd-musicnn-1.pb",
    "mood_acoustic-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.pb",
    "danceability-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/danceability/danceability-msd-musicnn-1.pb",
    "voice_instrumental-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.pb",
    "tonal_atonal-msd-musicnn-1.pb":
        "https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1.pb",

    # === EffNet family (for genre, timbre) ===
    "discogs-effnet-bs64-1.pb":
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",

    # EffNet classifier heads
    "genre_discogs400-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json":
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "timbre-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb",
    "emomusic-msd-musicnn-2.pb":
        "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb",
}

# Classifier definitions: name → (model_file, backbone, labels, output_node, input_node)
# input_node=None means use default
CLASSIFIERS = {
    # MusiCNN-based (mood, danceability, voice, tonal)
    # IMPORTANT: label ordering verified from official metadata JSONs at essentia.upf.edu
    "mood_happy":       ("mood_happy-msd-musicnn-1.pb",       "musicnn", ["happy", "non_happy"],             "model/Softmax",  None),
    "mood_sad":         ("mood_sad-msd-musicnn-1.pb",         "musicnn", ["non_sad", "sad"],                 "model/Softmax",  None),
    "mood_relaxed":     ("mood_relaxed-msd-musicnn-1.pb",     "musicnn", ["non_relaxed", "relaxed"],         "model/Softmax",  None),
    "mood_aggressive":  ("mood_aggressive-msd-musicnn-1.pb",  "musicnn", ["aggressive", "not_aggressive"],   "model/Softmax",  None),
    "mood_party":       ("mood_party-msd-musicnn-1.pb",       "musicnn", ["non_party", "party"],             "model/Softmax",  None),
    "mood_acoustic":    ("mood_acoustic-msd-musicnn-1.pb",    "musicnn", ["acoustic", "non_acoustic"],       "model/Softmax",  None),
    "danceability":     ("danceability-msd-musicnn-1.pb",     "musicnn", ["danceable", "not_danceable"],     "model/Softmax",  None),
    "voice_instrumental": ("voice_instrumental-msd-musicnn-1.pb", "musicnn", ["instrumental", "voice"],      "model/Softmax",  None),
    "tonal_atonal":     ("tonal_atonal-msd-musicnn-1.pb",     "musicnn", ["tonal", "atonal"],                "model/Softmax",  None),
    # EffNet-based — each has different node names
    "genre":            ("genre_discogs400-discogs-effnet-1.pb", "effnet", "genre_labels",                   "PartitionedCall", "serving_default_model_Placeholder"),
    "timbre":           ("timbre-discogs-effnet-1.pb",        "effnet",  ["bright", "dark"],                 "model/Softmax",   "model/Placeholder"),
    # MusiCNN-based arousal/valence (regression)
    "arousal_valence":  ("emomusic-msd-musicnn-2.pb",         "musicnn", "regression_av",                   "model/Identity",  None),
}


def check_imports():
    try:
        import essentia
        import essentia.standard as es
        print(f"Essentia version: {essentia.__version__}")
        return True
    except ImportError:
        print("ERROR: essentia-tensorflow not installed")
        print("Run: pip install essentia-tensorflow")
        return False


def download_models():
    """Download required model files if not present."""
    MODELS_DIR.mkdir(exist_ok=True)

    import urllib.request

    for filename, url in MODEL_URLS.items():
        path = MODELS_DIR / filename
        if path.exists():
            continue
        print(f"  Downloading {filename}...")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, path)
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  → {size_mb:.1f} MB")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  → Retry {attempt + 2}/3 ({e})")
                    time.sleep(2)
                else:
                    print(f"  → FAILED after 3 attempts: {e}")
                    if path.exists():
                        path.unlink()
                    return False

    print(f"All models ready in {MODELS_DIR}")
    return True


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
            if len(parts) >= 3:
                artist, album = parts[0], parts[1]
            else:
                artist, album = "Unknown", "Unknown"

            title = f.rsplit(".", 1)[0]
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break

            if track_filter and track_filter.lower() not in title.lower():
                continue

            tracks.append({
                "path": str(path),
                "artist": artist,
                "album": album,
                "title": title,
            })
    return tracks


def build_pipeline():
    """Initialize extractors + classifier heads, properly matched by backbone."""
    from essentia.standard import (
        TensorflowPredictMusiCNN,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredict2D,
    )

    print("Loading models...")
    t0 = time.time()

    # MusiCNN backbone
    musicnn_path = MODELS_DIR / "msd-musicnn-1.pb"
    musicnn_extractor = None
    if musicnn_path.exists():
        musicnn_extractor = TensorflowPredictMusiCNN(
            graphFilename=str(musicnn_path),
            output="model/dense/BiasAdd",
        )

    # EffNet backbone
    effnet_path = MODELS_DIR / "discogs-effnet-bs64-1.pb"
    effnet_extractor = None
    if effnet_path.exists():
        effnet_extractor = TensorflowPredictEffnetDiscogs(
            graphFilename=str(effnet_path),
            output="PartitionedCall:1",
        )

    extractors = {"musicnn": musicnn_extractor, "effnet": effnet_extractor}

    # Load classifier heads
    classifiers = {}
    labels = {}
    for name, (model_file, backbone, label_info, output_node, input_node) in CLASSIFIERS.items():
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            continue
        if extractors.get(backbone) is None:
            print(f"  Skipping {name} — {backbone} backbone not loaded")
            continue

        kwargs = {"graphFilename": str(model_path), "output": output_node}
        if input_node:
            kwargs["input"] = input_node

        classifiers[name] = {
            "model": TensorflowPredict2D(**kwargs),
            "backbone": backbone,
        }

        if label_info == "genre_labels":
            meta_path = MODELS_DIR / "genre_discogs400-discogs-effnet-1.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                labels[name] = meta.get("classes", [f"class_{i}" for i in range(400)])
            else:
                labels[name] = [f"class_{i}" for i in range(400)]
        elif label_info == "regression_av":
            labels[name] = "regression_av"
        else:
            labels[name] = label_info

    print(f"  Loaded {len(classifiers)} classifiers in {time.time() - t0:.1f}s")
    return extractors, classifiers, labels


def classify_track(audio_path, extractors, classifiers):
    """Run all classifiers on a single track."""
    from essentia.standard import MonoLoader
    import numpy as np

    audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

    # Extract embeddings per backbone (only compute what's needed)
    needed_backbones = set(c["backbone"] for c in classifiers.values())
    embeddings = {}
    for bb in needed_backbones:
        ext = extractors.get(bb)
        if ext:
            embeddings[bb] = ext(audio)

    # Run classifiers
    results = {}
    for name, cfg in classifiers.items():
        emb = embeddings.get(cfg["backbone"])
        if emb is None:
            continue
        try:
            preds = cfg["model"](emb)
            avg = np.mean(preds, axis=0)
            results[name] = [round(float(v), 4) for v in avg]
        except Exception as e:
            results[name] = f"error: {e}"

    return results


def summarize_predictions(preds, labels):
    """Build a human-readable summary dict: label → 'verdict (confidence%)'."""
    summary = {}
    for name, values in preds.items():
        if isinstance(values, str):
            summary[name] = values
            continue

        label_info = labels.get(name, [])

        if label_info == "regression_av":
            arousal = values[0] if len(values) > 0 else 0
            valence = values[1] if len(values) > 1 else 0
            summary[name] = f"arousal={arousal:.2f} valence={valence:.2f}"
        elif name == "genre" and isinstance(label_info, list) and len(label_info) == len(values):
            pairs = sorted(zip(label_info, values), key=lambda x: -x[1])[:3]
            summary[name] = ", ".join(f"{l} ({v:.0%})" for l, v in pairs)
        elif isinstance(label_info, list) and len(label_info) == 2 and len(values) == 2:
            idx = 1 if values[1] > values[0] else 0
            summary[name] = f"{label_info[idx]} ({values[idx]:.0%})"
        else:
            summary[name] = str(values)

    return summary


def format_result(name, values, label_info):
    """Pretty-print a classifier result."""
    if isinstance(values, str):
        return f"  {name:25s}  {values}"

    if label_info == "regression_av":
        arousal = values[0] if len(values) > 0 else 0
        valence = values[1] if len(values) > 1 else 0
        return f"  {name:25s}  arousal={arousal:.3f}  valence={valence:.3f}"

    if name == "genre" and isinstance(label_info, list) and len(label_info) == len(values):
        pairs = sorted(zip(label_info, values), key=lambda x: -x[1])[:5]
        lines = [f"  {name:25s}  Top genres:"]
        for label, prob in pairs:
            bar = "█" * int(prob * 50)
            lines.append(f"    {label:35s}  {prob:.3f}  {bar}")
        return "\n".join(lines)

    if isinstance(label_info, list) and len(label_info) == 2 and len(values) == 2:
        idx = 1 if values[1] > values[0] else 0
        return f"  {name:25s}  {label_info[idx]:20s}  ({values[idx]:.1%})"

    return f"  {name:25s}  {values}"


def main():
    parser = argparse.ArgumentParser(description="Soniq Lab — Essentia classification")
    parser.add_argument("--track", type=str, help="Filter by track title (substring)")
    parser.add_argument("--limit", type=int, default=0, help="Max tracks (0=all)")
    parser.add_argument("--download-only", action="store_true", help="Just download models")
    args = parser.parse_args()

    if not check_imports():
        sys.exit(1)

    print(f"\nMusic root: {MUSIC_ROOT}")
    if not MUSIC_ROOT.exists():
        print("ERROR: Music directory not found")
        sys.exit(1)

    print("\nChecking models...")
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

    print(f"\nFound {len(tracks)} track{'s' if len(tracks) != 1 else ''}")

    extractors, classifiers, labels = build_pipeline()

    RESULTS_DIR.mkdir(exist_ok=True)
    total_time = 0
    count = 0
    for i, track in enumerate(tracks):
        print(f"\n[{i+1}/{len(tracks)}] {track['artist']} — {track['title']}")
        t0 = time.time()

        try:
            preds = classify_track(track["path"], extractors, classifiers)
            elapsed = time.time() - t0
            total_time += elapsed
            count += 1
            print(f"  Classified in {elapsed:.1f}s")

            for name in sorted(preds.keys()):
                print(format_result(name, preds[name], labels.get(name, [])))

            summary = summarize_predictions(preds, labels)
            result = {
                "artist": track["artist"],
                "album": track["album"],
                "title": track["title"],
                "summary": summary,
                "predictions": preds,
                "time_s": round(elapsed, 2),
            }

            # Save individual JSON per track
            safe_name = f"{track['artist']} - {track['title']}".replace("/", "_")
            out_path = RESULTS_DIR / f"{safe_name}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  → {out_path.name}")

        except Exception as e:
            print(f"  ERROR: {e}")

    if count:
        avg = total_time / count
        print(f"\n{'='*60}")
        print(f"Processed {count} tracks in {total_time:.1f}s ({avg:.1f}s/track)")
        print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
