#!/usr/bin/env python3
"""Calibrate perceptual bright/dark from librosa features using V1 EffNet as ground truth.

Reads:
  - pipeline.db  → V1 EffNet bright/dark values (cls_json)
  - soniq.db     → V0.4 librosa features (scalars_json, vectors_json)

Outputs:
  - Prints the logistic regression weights to paste into pipeline_onnx.py
  - Validates on the full dataset with cross-validation
  - Shows per-artist spot checks
"""

import json
import sqlite3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Features to use (from Cohen's d analysis) ────────────────────

FEATURE_KEYS = [
    "contrast2", "contrast3", "contrast4",
    "flux", "mfcc1", "mfcc2",
    "tonnetz0", "tonnetz1", "tonnetz2", "tonnetz3", "tonnetz4",
    "chroma0", "chroma6", "chroma7", "chroma8", "chroma11",
]


def extract_row(scalars, vectors):
    """Pull the feature values we need from soniq.db JSON fields."""
    s = json.loads(scalars) if scalars else {}
    v = json.loads(vectors) if vectors else {}
    contrast = v.get("contrast", [0] * 7)
    mfcc_m = v.get("mfcc_m", [0] * 13)
    tonnetz = v.get("tonnetz", [0] * 6)
    chroma = v.get("chroma", [0] * 12)
    return {
        "contrast1": contrast[1] if len(contrast) > 1 else 0,
        "contrast2": contrast[2] if len(contrast) > 2 else 0,
        "contrast3": contrast[3] if len(contrast) > 3 else 0,
        "contrast4": contrast[4] if len(contrast) > 4 else 0,
        "contrast5": contrast[5] if len(contrast) > 5 else 0,
        "flux": s.get("flux", 0),
        "mfcc1": mfcc_m[1] if len(mfcc_m) > 1 else 0,
        "mfcc2": mfcc_m[2] if len(mfcc_m) > 2 else 0,
        "tonnetz0": tonnetz[0] if len(tonnetz) > 0 else 0,
        "tonnetz1": tonnetz[1] if len(tonnetz) > 1 else 0,
        "tonnetz2": tonnetz[2] if len(tonnetz) > 2 else 0,
        "tonnetz3": tonnetz[3] if len(tonnetz) > 3 else 0,
        "tonnetz4": tonnetz[4] if len(tonnetz) > 4 else 0,
        "tonnetz5": tonnetz[5] if len(tonnetz) > 5 else 0,
        "chroma0": chroma[0] if len(chroma) > 0 else 0,
        "chroma1": chroma[1] if len(chroma) > 1 else 0,
        "chroma2": chroma[2] if len(chroma) > 2 else 0,
        "chroma3": chroma[3] if len(chroma) > 3 else 0,
        "chroma4": chroma[4] if len(chroma) > 4 else 0,
        "chroma5": chroma[5] if len(chroma) > 5 else 0,
        "chroma6": chroma[6] if len(chroma) > 6 else 0,
        "chroma7": chroma[7] if len(chroma) > 7 else 0,
        "chroma8": chroma[8] if len(chroma) > 8 else 0,
        "chroma9": chroma[9] if len(chroma) > 9 else 0,
        "chroma10": chroma[10] if len(chroma) > 10 else 0,
        "chroma11": chroma[11] if len(chroma) > 11 else 0,
        "key": s.get("key", 0),
        "mode": s.get("mode", 0),
    }


# ── Load data ─────────────────────────────────────────────────────

def load_data():
    # V1 EffNet bright/dark
    conn1 = sqlite3.connect("pipeline.db")
    v1_rows = conn1.execute(
        "SELECT path, cls_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn1.close()

    v1_by_path = {}
    for path, cj in v1_rows:
        cls = json.loads(cj) if cj else {}
        b = cls.get("bright", 0.5)
        d = cls.get("dark", 0.5)
        ratio = b / (b + d) if (b + d) > 0 else 0.5
        # Rescale from compressed 0.37-0.54 range to 0-1
        rescaled = max(0.0, min(1.0, (ratio - 0.37) / (0.54 - 0.37)))
        v1_by_path[path] = rescaled

    # V0.4 librosa features
    conn2 = sqlite3.connect("soniq.db")
    v04_rows = conn2.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn2.close()

    # Match and build dataset
    X_rows = []
    y_vals = []
    paths = []

    for path, sj, vj in v04_rows:
        if path not in v1_by_path:
            continue
        rescaled = v1_by_path[path]
        # Use only clearly bright/dark tracks for training (skip ambiguous middle)
        if rescaled > 0.6:
            label = 1  # bright
        elif rescaled < 0.4:
            label = 0  # dark
        else:
            continue

        row = extract_row(sj, vj)
        X_rows.append([row[k] for k in FEATURE_KEYS])
        y_vals.append(label)
        paths.append(path)

    return np.array(X_rows), np.array(y_vals), paths, v1_by_path


# ── Train and evaluate ────────────────────────────────────────────

def main():
    X, y, paths, v1_by_path = load_data()
    n_bright = int(y.sum())
    n_dark = len(y) - n_bright
    print(f"Dataset: {len(y)} tracks ({n_bright} bright, {n_dark} dark)")
    print(f"Features: {FEATURE_KEYS}")
    print()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validate
    lr = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(lr, X_scaled, y, cv=10, scoring="accuracy")
    print(f"10-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print()

    # Fit on all data
    lr.fit(X_scaled, y)

    # Print coefficients in standardized space
    print("=== Standardized coefficients ===")
    for feat, coef in zip(FEATURE_KEYS, lr.coef_[0]):
        direction = "-> BRIGHT" if coef > 0 else "-> DARK"
        print(f"  {feat:<12} coef={coef:>7.4f} {direction}")
    print(f"  {'bias':<12} {lr.intercept_[0]:>7.4f}")
    print()

    # Convert to raw-feature weights (no scaler needed at inference)
    # logit = sum(coef_i * (x_i - mean_i) / std_i) + bias
    #       = sum((coef_i / std_i) * x_i) + (bias - sum(coef_i * mean_i / std_i))
    raw_weights = lr.coef_[0] / scaler.scale_
    raw_bias = lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_)

    print("=== Raw-feature weights (paste into pipeline_onnx.py) ===")
    print()
    print("BRIGHTNESS_WEIGHTS = {")
    for feat, w in zip(FEATURE_KEYS, raw_weights):
        print(f'    "{feat}": {w:.6f},')
    print("}")
    print(f'BRIGHTNESS_BIAS = {raw_bias:.6f}')
    print()

    # ── Validation: score ALL tracks (not just bright/dark extremes) ──
    conn2 = sqlite3.connect("soniq.db")
    all_rows = conn2.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn2.close()

    print("=== Per-artist spot check ===")
    print()

    # Collect predictions for all tracks
    predictions = {}
    for path, sj, vj in all_rows:
        row = extract_row(sj, vj)
        x = np.array([row[k] for k in FEATURE_KEYS])
        logit = np.dot(raw_weights, x) + raw_bias
        bright = 1 / (1 + np.exp(-logit))
        dark = 1 - bright
        predictions[path] = (bright, dark)

    # Spot-check artists
    spot_check = [
        "Plastikman", "Daft Punk", "John Coltrane", "Miles Davis",
        "Radiohead", "Massive Attack", "Boards of Canada",
    ]

    for artist_query in spot_check:
        matches = [
            (p, predictions[p]) for p in predictions
            if artist_query.lower() in p.lower()
        ]
        if not matches:
            continue
        matches.sort(key=lambda x: x[1][0])  # sort by bright score
        print(f"  {artist_query}:")
        for path, (bright, dark) in matches[:8]:
            name = path.rsplit("/", 1)[-1][:50]
            v1 = v1_by_path.get(path, -1)
            label = "BRIGHT" if bright > 0.5 else "DARK"
            v1_label = "BRIGHT" if v1 > 0.5 else "DARK" if v1 >= 0 else "?"
            match = "ok" if label == v1_label else "MISMATCH"
            print(f"    b={bright:.2f} d={dark:.2f}  v1={v1:.2f}  {match:<8}  {name}")
        print()

    # Overall accuracy on all matched tracks (not just extremes)
    correct = 0
    total = 0
    for path in predictions:
        if path in v1_by_path:
            v1 = v1_by_path[path]
            pred_bright = predictions[path][0] > 0.5
            v1_bright = v1 > 0.5
            if pred_bright == v1_bright:
                correct += 1
            total += 1
    print(f"Overall accuracy (all {total} tracks, >0.5 threshold): {correct/total:.3f}")


if __name__ == "__main__":
    main()
