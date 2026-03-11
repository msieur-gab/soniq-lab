#!/usr/bin/env python3
"""Side-by-side comparison: V1 (TF, pipeline.db) vs V2 (ONNX + librosa + MusicBrainz).

Runs V2 on one track per artist, compares all fields against SQL reference.
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
    CLASSIFIERS, MODELS_DIR, BACKBONE_BATCH,
    compute_mel_spectrogram, make_patches, compute_timbre_vector,
    predict_bright_dark, lookup_genre,
)

DB_PATH = Path(__file__).parent.parent / "pipeline.db"

# V2 index for the "positive" class of each binary classifier
POS_IDX = {
    "mood_happy": ("happy", 0),
    "mood_sad": ("sad", 1),
    "mood_relaxed": ("relaxed", 1),
    "mood_aggressive": ("aggressive", 0),
    "mood_party": ("party", 1),
    "mood_acoustic": ("acoustic", 0),
    "danceability": ("danceable", 0),
    "voice_instrumental": ("instrumental", 0),
    "tonal_atonal": ("tonal", 0),
}


def main():
    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    conn = sqlite3.connect(str(DB_PATH))
    tracks = conn.execute("""
        SELECT artist, album, title, path, cls_json FROM tracks
        WHERE status='done' AND cls_json IS NOT NULL
        GROUP BY artist ORDER BY artist
    """).fetchall()[:limit]

    print(f"Comparing V1 vs V2 on {len(tracks)} tracks\n")

    backbone = ort.InferenceSession(
        str(MODELS_DIR / "msd-musicnn-1.onnx"), providers=["CPUExecutionProvider"]
    )

    all_rows = []

    for i, (artist, album, title, path, cj) in enumerate(tracks):
        ref = json.loads(cj)
        print(f"{'='*70}")
        print(f"[{i+1}/{len(tracks)}] {artist} — {title}")
        print(f"{'='*70}")

        t0 = time.time()

        # ONNX classification
        mel = compute_mel_spectrogram(path)
        patches = make_patches(mel)
        del mel
        batches = []
        for b in range(0, len(patches), BACKBONE_BATCH):
            batches.append(backbone.run(None, {"melspectrogram": patches[b:b + BACKBONE_BATCH]})[1])
        emb = np.concatenate(batches)
        del patches, batches

        v2 = {}
        for name, (onnx_file, labels) in CLASSIFIERS.items():
            onnx_path = MODELS_DIR / onnx_file
            if not onnx_path.exists():
                continue
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            preds = session.run(None, {"embeddings": emb})[0]
            avg = np.mean(preds, axis=0)
            v2[name] = [float(v) for v in avg]
            del session

        # Timbre + bright/dark
        timbre_vec = compute_timbre_vector(path)
        bright, dark = predict_bright_dark(timbre_vec)

        # Genre
        genre_tags = lookup_genre(artist, album)

        elapsed = time.time() - t0

        # Print comparison
        print(f"\n  {'Field':20s}  {'V1 (TF)':>12s}  {'V2 (ONNX)':>12s}  {'Δ':>8s}  Match")
        print(f"  {'-'*66}")

        row = {"artist": artist, "title": title, "time": elapsed, "matches": 0, "total": 0}

        # Binary classifiers
        for v2_name, (tf_key, idx) in POS_IDX.items():
            if v2_name not in v2 or tf_key not in ref:
                continue
            tf_val = ref[tf_key]
            v2_val = v2[v2_name][idx]
            delta = v2_val - tf_val
            same = (tf_val > 0.5 and v2_val > 0.5) or (tf_val <= 0.5 and v2_val <= 0.5)
            row["total"] += 1
            if same:
                row["matches"] += 1
            mark = "  " if same else "!!"
            print(f"  {tf_key:20s}  {tf_val:12.3f}  {v2_val:12.3f}  {delta:+8.3f}  {mark}")

        # Arousal/valence
        if "arousal_valence" in v2 and "arousal" in ref:
            for j, key in enumerate(["arousal", "valence"]):
                tf_val = ref[key]
                v2_val = v2["arousal_valence"][j]
                delta = v2_val - tf_val
                same = abs(delta) < 1.0
                row["total"] += 1
                if same:
                    row["matches"] += 1
                mark = "  " if same else "!!"
                print(f"  {key:20s}  {tf_val:12.2f}  {v2_val:12.2f}  {delta:+8.2f}  {mark}")

        # Bright/dark
        if bright is not None and "bright" in ref:
            tf_b = ref["bright"]
            delta = bright - tf_b
            same = (tf_b > 0.5 and bright > 0.5) or (tf_b <= 0.5 and bright <= 0.5)
            row["total"] += 1
            if same:
                row["matches"] += 1
            mark = "  " if same else "!!"
            print(f"  {'bright':20s}  {tf_b:12.3f}  {bright:12.3f}  {delta:+8.3f}  {mark}")

        # Genre
        tf_genre = ref.get("genre", [])
        v2_genre = [t[0] for t in genre_tags[:5]] if genre_tags else []
        print(f"\n  {'genre (V1)':20s}  {', '.join(tf_genre)}")
        print(f"  {'genre (V2)':20s}  {', '.join(v2_genre) if v2_genre else '(no data)'}")

        pct = f"{row['matches']}/{row['total']}" if row["total"] else "N/A"
        print(f"\n  Score: {pct}  |  Time: {elapsed:.1f}s\n")

        all_rows.append(row)

    # Summary
    total_matches = sum(r["matches"] for r in all_rows)
    total_fields = sum(r["total"] for r in all_rows)
    total_time = sum(r["time"] for r in all_rows)
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(all_rows)} tracks in {total_time:.0f}s ({total_time/len(all_rows):.1f}s/track)")
    print(f"Overall agreement: {total_matches}/{total_fields} ({total_matches/total_fields:.1%})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
