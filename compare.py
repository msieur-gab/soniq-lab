#!/usr/bin/env python3
"""
Compare Essentia classifications with Soniq librosa features.

Reads results.json (from classify.py) and queries the music-player API
to get the corresponding Soniq features, then shows a side-by-side comparison.
"""

import json
import sys
import urllib.request
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"
API_URL = "http://localhost:8000/api/tracks?per_page=2000"

SCALAR_KEYS = [
    "tempo", "rms_mean", "rms_max", "rms_variance", "dynamic_range",
    "centroid_mean", "flatness_mean", "spectral_flux", "onset_strength",
    "beat_strength", "vocal_proxy", "zcr_mean", "duration",
]


def norm(v, lo, hi):
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def compute_composites(t, ranges):
    """Compute Soniq composites for a track given library-wide ranges."""
    n = {k: norm(t.get(k, 0), *ranges[k]) for k in SCALAR_KEYS}

    return {
        "energy": (n["rms_mean"] * 1.0 + n["onset_strength"] * 0.8
                   + n["beat_strength"] * 0.6 + n["dynamic_range"] * 0.4) / 2.8,
        "acousticness": ((1 - n["flatness_mean"]) * 1.0
                         + (1 - n["centroid_mean"]) * 0.8
                         + (1 - n["zcr_mean"]) * 0.6) / 2.4,
        "danceability": (n["beat_strength"] * 1.2 + n["tempo"] * 0.6
                         + n["onset_strength"] * 0.4) / 2.2,
        "valence": (n["centroid_mean"] * 0.6 + n["rms_mean"] * 0.4) / 1.0,
        "instrumentalness": (1 - n["vocal_proxy"]) / 1.0,
        "texture": (n["flatness_mean"] * 1.0 + n["zcr_mean"] * 0.8
                    + n["spectral_flux"] * 0.6) / 2.4,
        "dynamics": (n["dynamic_range"] * 1.2 + n["rms_variance"] * 0.8
                     + n["rms_max"] * 0.4) / 2.4,
        "stillness": ((1 - n["onset_strength"]) * 1.0
                      + (1 - n["beat_strength"]) * 1.0
                      + (1 - n["spectral_flux"]) * 0.8
                      + n["duration"] * 0.4) / 3.2,
    }


def main():
    # Load Essentia results
    if not RESULTS_FILE.exists():
        print("No results.json found. Run classify.py first.")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        essentia_results = json.load(f)

    # Load Soniq features from API
    print("Fetching Soniq features from API...")
    try:
        with urllib.request.urlopen(API_URL, timeout=10) as resp:
            data = json.loads(resp.read())
        tracks = data.get("tracks", [])
    except Exception as e:
        print(f"API error: {e}")
        print("Make sure the music-player server is running on port 8000")
        sys.exit(1)

    # Compute ranges for normalization
    ranges = {}
    for k in SCALAR_KEYS:
        vals = [t.get(k, 0) for t in tracks if t.get(k) is not None]
        ranges[k] = (min(vals), max(vals)) if vals else (0, 1)

    # Index Soniq tracks by artist/title for matching
    soniq_index = {}
    for t in tracks:
        key = (t.get("artist", "").lower(), t.get("title", "").lower())
        soniq_index[key] = t

    # Compare
    print(f"\nComparing {len(essentia_results)} tracks...\n")
    print("=" * 100)

    for track_key, result in essentia_results.items():
        artist = result["artist"]
        title = result["title"]
        preds = result["predictions"]

        # Find matching Soniq track
        lookup = (artist.lower(), title.lower())
        soniq = soniq_index.get(lookup)
        if not soniq:
            # Fuzzy match: try partial title
            for (a, t), s in soniq_index.items():
                if artist.lower() in a and title.lower() in t:
                    soniq = s
                    break

        print(f"\n{artist} — {title}")
        print("-" * 60)

        if not soniq:
            print("  (no matching Soniq track found)")
            for name, vals in sorted(preds.items()):
                if name == "genre":
                    continue
                if isinstance(vals, list) and len(vals) == 2:
                    print(f"  Essentia {name:20s}: {vals[1]:.2f}")
                elif isinstance(vals, list) and len(vals) == 1:
                    print(f"  Essentia {name:20s}: {vals[0]:.2f}")
            continue

        composites = compute_composites(soniq, ranges)

        # Map Essentia predictions to comparable Soniq composites
        comparisons = [
            # (label, soniq_value, essentia_value, description)
            ("Acousticness",
             composites["acousticness"],
             preds.get("mood_acoustic", [0, 0])[1] if isinstance(preds.get("mood_acoustic"), list) else None,
             "acoustic feel"),
            ("Danceability",
             composites["danceability"],
             preds.get("danceability", [0, 0])[1] if isinstance(preds.get("danceability"), list) else None,
             "danceable"),
            ("Instrumentalness",
             composites["instrumentalness"],
             preds.get("voice_instrumental", [0, 0])[1] if isinstance(preds.get("voice_instrumental"), list) else None,
             "instrumental"),
            ("Stillness/Relaxed",
             composites["stillness"],
             preds.get("mood_relaxed", [0, 0])[1] if isinstance(preds.get("mood_relaxed"), list) else None,
             "relaxed"),
            ("Valence/Happy",
             composites["valence"],
             preds.get("mood_happy", [0, 0])[1] if isinstance(preds.get("mood_happy"), list) else None,
             "happy"),
        ]

        print(f"  {'Dimension':20s}  {'Soniq':>6s}  {'Essentia':>8s}  {'Δ':>6s}  Match")
        print(f"  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*6}  -----")

        matches = 0
        total = 0
        for label, soniq_val, ess_val, desc in comparisons:
            if ess_val is None:
                continue
            delta = abs(soniq_val - ess_val)
            match = "✓" if delta < 0.2 else "✗"
            if delta < 0.2:
                matches += 1
            total += 1
            print(f"  {label:20s}  {soniq_val:6.2f}  {ess_val:8.2f}  {delta:6.2f}  {match}")

        # Essentia-only features (no Soniq equivalent)
        print(f"\n  Essentia-only:")
        for name in ["mood_sad", "mood_aggressive", "mood_party", "timbre",
                      "tonal_atonal", "arousal", "valence"]:
            val = preds.get(name)
            if val is None:
                continue
            if isinstance(val, list) and len(val) == 2:
                labels_map = {
                    "mood_sad": ["not_sad", "sad"],
                    "mood_aggressive": ["not_aggressive", "aggressive"],
                    "mood_party": ["not_party", "party"],
                    "timbre": ["bright", "dark"],
                    "tonal_atonal": ["tonal", "atonal"],
                }
                lbl = labels_map.get(name, ["class0", "class1"])
                idx = 1 if val[1] > val[0] else 0
                print(f"    {name:20s}  {lbl[idx]:15s} ({val[idx]:.1%})")
            elif isinstance(val, list):
                avg = sum(val) / len(val) if val else 0
                print(f"    {name:20s}  {avg:.3f}")

        if total:
            print(f"\n  Score: {matches}/{total} dimensions match (within 0.2)")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
