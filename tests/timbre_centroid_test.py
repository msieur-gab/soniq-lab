#!/usr/bin/env python3
"""
Compare neural timbre (bright/dark) predictions against DSP spectral centroid.

Reads results.json for neural timbre scores, computes spectral centroid
via Essentia, and outputs per-track JSON files to tests/timbre-dsp/.
"""

import json
import os
import sys
from pathlib import Path

import essentia.standard as es

MUSIC_DIR = Path(__file__).parent.parent / "music"
RESULTS_FILE = Path(__file__).parent.parent / "results.json"
OUTPUT_DIR = Path(__file__).parent / "timbre-dsp"


def find_audio_file(artist, album, title):
    """Find the .m4a file matching artist/album/title."""
    album_dir = MUSIC_DIR / artist / album
    if not album_dir.exists():
        return None
    for f in album_dir.iterdir():
        if f.suffix == ".m4a" and title in f.stem:
            return f
    return None


def compute_centroid(audio_path):
    """Compute mean spectral centroid using Essentia."""
    loader = es.MonoLoader(filename=str(audio_path), sampleRate=44100)
    audio = loader()

    w = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    centroid = es.Centroid(range=22050)

    centroids = []
    for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
        spec = spectrum(w(frame))
        c = centroid(spec)
        centroids.append(c)  # already in Hz (range param handles scaling)

    if not centroids:
        return 0.0
    return round(sum(centroids) / len(centroids), 4)


def main():
    if not RESULTS_FILE.exists():
        print("No results.json found. Run classify.py first.")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for key, val in results.items():
        timbre = val.get("predictions", {}).get("timbre")
        if not timbre:
            continue

        artist = val["artist"]
        album = val["album"]
        title = val["title"]

        audio_path = find_audio_file(artist, album, title)
        if not audio_path:
            print(f"  SKIP (file not found): {artist}/{album}/{title}")
            skipped += 1
            continue

        print(f"  Processing: {audio_path.name}")
        centroid_hz = compute_centroid(audio_path)

        out = {
            "file": audio_path.name,
            "path": str(audio_path),
            "neural_bright": round(timbre[0], 4),
            "neural_dark": round(timbre[1], 4),
            "dsp_centroid": centroid_hz,
        }

        out_file = OUTPUT_DIR / f"{audio_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")

        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped")
    print(f"Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
