#!/usr/bin/env python3
"""Calibrate ALL classifiers from librosa features using V1 as ground truth.

Tests whether we can replace MusiCNN ONNX entirely with librosa-derived
logistic regressions.
"""

import json
import sqlite3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ── Feature extraction ────────────────────────────────────────────

def extract_all_features(scalars, vectors):
    """Extract every available librosa feature as a flat dict."""
    s = json.loads(scalars) if scalars else {}
    v = json.loads(vectors) if vectors else {}

    mfcc = v.get("mfcc_m", [0] * 13)
    mfcc_s = v.get("mfcc_s", [0] * 13)
    contrast = v.get("contrast", [0] * 7)
    chroma = v.get("chroma", [0] * 12)
    tonnetz = v.get("tonnetz", [0] * 6)

    features = {}

    # Scalars
    for key in ("centroid", "centroid_std", "rolloff", "rolloff_std",
                "bandwidth", "bandwidth_std", "flatness", "flux", "flux_std",
                "zcr", "rms_mean", "rms_max", "rms_var", "dyn_range",
                "tempo", "key", "mode", "onset", "beat", "vocal", "duration"):
        features[key] = s.get(key, 0)

    # Derived scalars
    features["rms_range"] = features["rms_max"] - features["rms_mean"]
    features["centroid_var"] = features["centroid_std"] ** 2 if features["centroid_std"] else 0
    features["low_energy"] = 1.0 / (1.0 + features["centroid"]) if features["centroid"] else 0
    features["tempo_sq"] = features["tempo"] ** 2

    # MFCCs (mean + std)
    for i in range(13):
        features[f"mfcc{i}"] = mfcc[i] if i < len(mfcc) else 0
        features[f"mfcc_s{i}"] = mfcc_s[i] if i < len(mfcc_s) else 0

    # Spectral contrast (7 bands)
    for i in range(7):
        features[f"contrast{i}"] = contrast[i] if i < len(contrast) else 0

    # Chroma (12 pitch classes)
    for i in range(12):
        features[f"chroma{i}"] = chroma[i] if i < len(chroma) else 0

    # Tonnetz (6 dimensions)
    for i in range(6):
        features[f"tonnetz{i}"] = tonnetz[i] if i < len(tonnetz) else 0

    # Cross-feature interactions
    features["tempo_x_beat"] = features["tempo"] * features["beat"]
    features["tempo_x_onset"] = features["tempo"] * features["onset"]
    features["rms_x_flux"] = features["rms_mean"] * features["flux"]
    features["mode_x_mfcc1"] = features["mode"] * features["mfcc1"]
    features["centroid_x_flatness"] = features["centroid"] * features["flatness"]
    features["contrast_range"] = (
        features["contrast6"] - features["contrast0"]
        if features["contrast6"] and features["contrast0"] else 0
    )
    features["chroma_std"] = float(np.std([features[f"chroma{i}"] for i in range(12)]))
    features["tonnetz_energy"] = float(np.sqrt(sum(
        features[f"tonnetz{i}"] ** 2 for i in range(6)
    )))

    return features


# ── Load data ─────────────────────────────────────────────────────

def load_data():
    conn1 = sqlite3.connect("pipeline.db")
    v1_rows = conn1.execute(
        "SELECT path, cls_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn1.close()

    v1_by_path = {}
    for path, cj in v1_rows:
        cls = json.loads(cj) if cj else {}
        v1_by_path[path] = cls

    conn2 = sqlite3.connect("soniq.db")
    v04_rows = conn2.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn2.close()

    # Match tracks
    matched = []
    for path, sj, vj in v04_rows:
        if path not in v1_by_path:
            continue
        features = extract_all_features(sj, vj)
        matched.append((path, features, v1_by_path[path]))

    return matched


# ── Per-classifier calibration ────────────────────────────────────

CLASSIFIERS = [
    "happy", "sad", "relaxed", "aggressive", "party",
    "acoustic", "danceable", "instrumental", "tonal",
    "arousal", "valence",
]

# Feature groups to try per classifier
FEATURE_GROUPS = {
    "all_scalars": [
        "centroid", "centroid_std", "rolloff", "rolloff_std",
        "bandwidth", "bandwidth_std", "flatness", "flux", "flux_std",
        "zcr", "rms_mean", "rms_max", "rms_var", "dyn_range",
        "tempo", "key", "mode", "onset", "beat", "vocal",
        "rms_range", "centroid_var", "low_energy", "tempo_sq",
    ],
    "mfcc": [f"mfcc{i}" for i in range(13)],
    "mfcc_std": [f"mfcc_s{i}" for i in range(13)],
    "contrast": [f"contrast{i}" for i in range(7)],
    "chroma": [f"chroma{i}" for i in range(12)],
    "tonnetz": [f"tonnetz{i}" for i in range(6)],
    "interactions": [
        "tempo_x_beat", "tempo_x_onset", "rms_x_flux",
        "mode_x_mfcc1", "centroid_x_flatness",
        "contrast_range", "chroma_std", "tonnetz_energy",
    ],
}

ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group)
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))  # dedupe, preserve order


def calibrate_classifier(matched, cls_name, feature_keys):
    """Fit logistic regression for one classifier. Returns (accuracy, model, scaler, feature_keys)."""
    X_rows = []
    y_vals = []

    for path, features, v1_cls in matched:
        val = v1_cls.get(cls_name)
        if val is None:
            continue
        label = 1 if val > 0.5 else 0
        X_rows.append([features[k] for k in feature_keys])
        y_vals.append(label)

    X = np.array(X_rows)
    y = np.array(y_vals)

    if len(np.unique(y)) < 2:
        return 0, 0, None, None, feature_keys, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(lr, X_scaled, y, cv=10, scoring="accuracy")

    lr.fit(X_scaled, y)

    return scores.mean(), scores.std(), lr, scaler, feature_keys, y


def find_best_features(matched, cls_name):
    """Try feature groups individually and combined to find best set."""
    # First try all features
    best_acc, best_std, best_lr, best_scaler, best_keys, best_y = calibrate_classifier(
        matched, cls_name, ALL_FEATURES
    )

    # Try progressive feature selection: start with all, drop low-impact ones
    lr_full = best_lr
    if lr_full is None:
        return best_acc, best_std, best_lr, best_scaler, best_keys, best_y

    # Get feature importances
    importances = np.abs(lr_full.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]

    # Try top N features
    for n in [40, 30, 20, 15, 10]:
        if n >= len(ALL_FEATURES):
            continue
        top_keys = [ALL_FEATURES[i] for i in sorted_idx[:n]]
        acc, std, lr, scaler, keys, y_data = calibrate_classifier(matched, cls_name, top_keys)
        if acc > best_acc - 0.002:  # within 0.2% is fine with fewer features
            best_acc, best_std = acc, std
            best_lr, best_scaler, best_keys, best_y = lr, scaler, top_keys, y_data

    return best_acc, best_std, best_lr, best_scaler, best_keys, best_y


# ── Main ──────────────────────────────────────────────────────────

def main():
    matched = load_data()
    print(f"Matched tracks: {len(matched)}")
    print(f"Total features: {len(ALL_FEATURES)}")
    print()

    results = []

    for cls_name in CLASSIFIERS:
        acc, std, lr, scaler, keys, y_data = find_best_features(matched, cls_name)
        n_pos = int(y_data.sum()) if y_data is not None else 0
        n_neg = len(y_data) - n_pos if y_data is not None else 0
        results.append((cls_name, acc, std, len(keys), n_pos, n_neg, lr, scaler, keys))

    # Summary table
    print(f"{'Classifier':<15} {'CV Acc':>8} {'Std':>7} {'Feats':>5} {'Pos':>5} {'Neg':>5} {'Status':>10}")
    print("-" * 60)
    for cls_name, acc, std, n_feat, n_pos, n_neg, _, _, _ in sorted(results, key=lambda x: x[1], reverse=True):
        status = "GREAT" if acc >= 0.93 else "GOOD" if acc >= 0.88 else "OK" if acc >= 0.80 else "WEAK"
        print(f"{cls_name:<15} {acc:>7.3f} {std:>7.3f} {n_feat:>5} {n_pos:>5} {n_neg:>5} {status:>10}")

    # Detailed output for each classifier
    print()
    print("=" * 70)
    for cls_name, acc, std, n_feat, n_pos, n_neg, lr, scaler, keys in results:
        if lr is None:
            continue
        print(f"\n### {cls_name} — {acc:.3f} ({n_feat} features)")

        # Top 10 features by coefficient magnitude
        raw_weights = lr.coef_[0] / scaler.scale_
        raw_bias = lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_)

        coefs = sorted(zip(keys, lr.coef_[0], raw_weights), key=lambda x: abs(x[1]), reverse=True)
        print(f"  Top features (standardized coef):")
        for feat, std_coef, raw_w in coefs[:10]:
            direction = "+" if std_coef > 0 else "-"
            print(f"    {direction} {feat:<25} std={std_coef:>7.3f}  raw={raw_w:>10.6f}")
        print(f"  Bias: {raw_bias:.6f}")


if __name__ == "__main__":
    main()
