#!/usr/bin/env python3
"""Validate V2 ONNX pipeline against TF reference from pipeline.db.

Runs one track per artist through V2, compares with stored TF predictions.
Outputs a summary table with direction agreement and value deltas.
"""

import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from classify_v2 import (
    CLASSIFIERS, MODELS_DIR, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE,
    N_MELS, FMIN, FMAX, PATCH_SIZE, PATCH_HOP, MAX_DURATION,
    compute_mel_spectrogram, make_patches, compute_timbre_vector,
    download_models,
)

DB_PATH = Path(__file__).parent.parent / "pipeline.db"

# Map V2 classifier names → TF reference keys
V2_TO_TF = {
    "mood_happy": ("happy", 0),         # happy is index 0
    "mood_sad": ("sad", 1),             # sad is index 1
    "mood_relaxed": ("relaxed", 1),     # relaxed is index 1
    "mood_aggressive": ("aggressive", 0),  # aggressive is index 0
    "mood_party": ("party", 1),         # party is index 1
    "mood_acoustic": ("acoustic", 0),   # acoustic is index 0
    "danceability": ("danceable", 0),   # danceable is index 0
    "voice_instrumental": ("instrumental", 0),  # instrumental is index 0
    "tonal_atonal": ("tonal", 0),       # tonal is index 0
}


def main():
    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    conn = sqlite3.connect(str(DB_PATH))
    tracks = conn.execute("""
        SELECT artist, title, path, cls_json FROM tracks
        WHERE status='done' AND cls_json IS NOT NULL
        GROUP BY artist ORDER BY artist
    """).fetchall()

    print(f"Validating V2 on {len(tracks)} tracks (1 per artist)")
    print(f"Loading backbone...")

    backbone = ort.InferenceSession(
        str(MODELS_DIR / "msd-musicnn-1.onnx"),
        providers=["CPUExecutionProvider"]
    )

    # Collect results
    rows = []
    total_time = 0

    for i, (artist, title, path, cls_json) in enumerate(tracks):
        ref = json.loads(cls_json)
        label = f"{artist} — {title}"
        print(f"\n[{i+1}/{len(tracks)}] {label}")

        t0 = time.time()
        try:
            mel = compute_mel_spectrogram(path)
            patches = make_patches(mel)
            del mel
            emb = backbone.run(None, {"melspectrogram": patches})[1]
            del patches

            v2_preds = {}
            for name, (onnx_file, labels) in CLASSIFIERS.items():
                onnx_path = MODELS_DIR / onnx_file
                if not onnx_path.exists():
                    continue
                session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                preds = session.run(None, {"embeddings": emb})[0]
                avg = np.mean(preds, axis=0)
                v2_preds[name] = [float(v) for v in avg]
                del session

            elapsed = time.time() - t0
            total_time += elapsed

            # Compare each classifier
            row = {"artist": artist, "title": title, "time": elapsed, "comparisons": {}}
            for v2_name, (tf_key, pos_idx) in V2_TO_TF.items():
                if v2_name not in v2_preds or tf_key not in ref:
                    continue
                tf_val = ref[tf_key]
                v2_val = v2_preds[v2_name][pos_idx]
                same_dir = (tf_val > 0.5 and v2_val > 0.5) or (tf_val <= 0.5 and v2_val <= 0.5)
                row["comparisons"][tf_key] = {
                    "tf": round(tf_val, 3),
                    "v2": round(v2_val, 3),
                    "delta": round(v2_val - tf_val, 3),
                    "match": same_dir,
                }

            # Arousal/valence
            if "arousal_valence" in v2_preds and "arousal" in ref:
                row["comparisons"]["arousal"] = {
                    "tf": round(ref["arousal"], 2),
                    "v2": round(v2_preds["arousal_valence"][0], 2),
                    "delta": round(v2_preds["arousal_valence"][0] - ref["arousal"], 2),
                    "match": abs(v2_preds["arousal_valence"][0] - ref["arousal"]) < 1.5,
                }
                row["comparisons"]["valence"] = {
                    "tf": round(ref["valence"], 2),
                    "v2": round(v2_preds["arousal_valence"][1], 2),
                    "delta": round(v2_preds["arousal_valence"][1] - ref["valence"], 2),
                    "match": abs(v2_preds["arousal_valence"][1] - ref["valence"]) < 1.5,
                }

            rows.append(row)

            # Print per-track summary
            matches = sum(1 for c in row["comparisons"].values() if c["match"])
            total = len(row["comparisons"])
            print(f"  {elapsed:.1f}s — {matches}/{total} match")
            for k, c in row["comparisons"].items():
                flag = "  " if c["match"] else "!!"
                print(f"  {flag} {k:15s}  TF={c['tf']:6.3f}  V2={c['v2']:6.3f}  Δ={c['delta']:+.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"artist": artist, "title": title, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print(f"Processed {len(rows)} tracks in {total_time:.0f}s ({total_time/len(rows):.1f}s/track)\n")

    # Per-classifier agreement
    classifiers = ["happy", "sad", "relaxed", "aggressive", "danceable",
                    "instrumental", "acoustic", "party", "tonal", "arousal", "valence"]
    print(f"{'Classifier':15s}  {'Agree':>5s}  {'Total':>5s}  {'Rate':>6s}  {'Mean Δ':>7s}")
    print("-" * 50)
    for cls in classifiers:
        matches = [r["comparisons"][cls]["match"] for r in rows
                   if "comparisons" in r and cls in r["comparisons"]]
        deltas = [r["comparisons"][cls]["delta"] for r in rows
                  if "comparisons" in r and cls in r["comparisons"]]
        if matches:
            agree = sum(matches)
            total = len(matches)
            mean_d = np.mean(deltas)
            print(f"{cls:15s}  {agree:5d}  {total:5d}  {agree/total:6.1%}  {mean_d:+7.3f}")

    overall_match = sum(
        c["match"]
        for r in rows if "comparisons" in r
        for c in r["comparisons"].values()
    )
    overall_total = sum(
        1 for r in rows if "comparisons" in r
        for c in r["comparisons"].values()
    )
    print(f"\nOverall direction agreement: {overall_match}/{overall_total} ({overall_match/overall_total:.1%})")

    # Save full results
    out_path = Path(__file__).parent / "v2_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
