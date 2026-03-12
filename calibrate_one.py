#!/usr/bin/env python3
"""Calibrate a single classifier with creative feature engineering.

Usage: python calibrate_one.py <classifier_name>
"""

import json
import sqlite3
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def extract_features(scalars, vectors):
    s = json.loads(scalars) if scalars else {}
    v = json.loads(vectors) if vectors else {}

    mfcc = v.get("mfcc_m", [0] * 13)
    mfcc_s = v.get("mfcc_s", [0] * 13)
    contrast = v.get("contrast", [0] * 7)
    chroma = v.get("chroma", [0] * 12)
    tonnetz = v.get("tonnetz", [0] * 6)

    f = {}

    # Raw scalars
    for key in ("centroid", "centroid_std", "rolloff", "rolloff_std",
                "bandwidth", "bandwidth_std", "flatness", "flux", "flux_std",
                "zcr", "rms_mean", "rms_max", "rms_var", "dyn_range",
                "tempo", "key", "mode", "onset", "beat", "vocal", "duration"):
        f[key] = s.get(key, 0)

    # MFCCs
    for i in range(13):
        f[f"mfcc{i}"] = mfcc[i] if i < len(mfcc) else 0
        f[f"mfcc_s{i}"] = mfcc_s[i] if i < len(mfcc_s) else 0

    # Contrast
    for i in range(7):
        f[f"contrast{i}"] = contrast[i] if i < len(contrast) else 0

    # Chroma
    for i in range(12):
        f[f"chroma{i}"] = chroma[i] if i < len(chroma) else 0

    # Tonnetz
    for i in range(6):
        f[f"tonnetz{i}"] = tonnetz[i] if i < len(tonnetz) else 0

    # --- Creative derived features ---

    # Rhythm features
    f["tempo_norm"] = f["tempo"] / 200.0  # normalize to ~0-1
    f["tempo_sq"] = (f["tempo"] / 200.0) ** 2
    f["beat_x_tempo"] = f["beat"] * f["tempo"] / 200.0
    f["onset_x_tempo"] = f["onset"] * f["tempo"] / 200.0
    f["rhythm_energy"] = f["beat"] * f["onset"]

    # Energy features
    f["rms_range"] = f["rms_max"] - f["rms_mean"]
    f["rms_x_flux"] = f["rms_mean"] * f["flux"]
    f["energy_density"] = f["rms_mean"] * f["onset"]
    f["loudness_var"] = f["rms_var"] / (f["rms_mean"] + 1e-6)

    # Spectral shape
    f["centroid_norm"] = f["centroid"] / 8000.0
    f["centroid_var"] = f["centroid_std"] ** 2
    f["spectral_width"] = f["rolloff"] - f["centroid"] if f["rolloff"] > f["centroid"] else 0
    f["centroid_x_flatness"] = f["centroid"] * f["flatness"]
    f["brightness_proxy"] = f["centroid"] / (f["rolloff"] + 1e-6)

    # Tonal features
    f["major_key"] = f["mode"]  # 1=major, 0=minor
    f["mode_x_mfcc1"] = f["mode"] * f["mfcc1"]
    f["chroma_std"] = float(np.std([f[f"chroma{i}"] for i in range(12)]))
    f["chroma_max"] = float(np.max([f[f"chroma{i}"] for i in range(12)]))
    f["chroma_range"] = f["chroma_max"] - float(np.min([f[f"chroma{i}"] for i in range(12)]))
    f["tonnetz_energy"] = float(np.sqrt(sum(f[f"tonnetz{i}"] ** 2 for i in range(6))))

    # Contrast shape
    f["contrast_range"] = f["contrast6"] - f["contrast0"]
    f["contrast_low"] = (f["contrast0"] + f["contrast1"] + f["contrast2"]) / 3
    f["contrast_high"] = (f["contrast4"] + f["contrast5"] + f["contrast6"]) / 3
    f["contrast_slope"] = f["contrast_high"] - f["contrast_low"]

    # Key as circular features (avoid treating key 0 and 11 as far apart)
    f["key_sin"] = float(np.sin(2 * np.pi * f["key"] / 12))
    f["key_cos"] = float(np.cos(2 * np.pi * f["key"] / 12))

    # Vocal presence interactions
    f["vocal_x_mfcc1"] = f["vocal"] * f["mfcc1"]
    f["vocal_x_flux"] = f["vocal"] * f["flux"]

    return f


def load_data(cls_name, use_v04=False):
    """Load matched data. use_v04=True for arousal/valence (1-10 scale in both DBs)."""
    if use_v04:
        conn = sqlite3.connect("soniq.db")
        rows = conn.execute(
            "SELECT path, scalars_json, vectors_json, cls_json FROM tracks WHERE status='done'"
        ).fetchall()
        conn.close()
        matched = []
        for path, sj, vj, cj in rows:
            cls = json.loads(cj) if cj else {}
            val = cls.get(cls_name)
            if val is None:
                continue
            features = extract_features(sj, vj)
            matched.append((path, features, val))
        return matched
    else:
        conn1 = sqlite3.connect("pipeline.db")
        v1_rows = conn1.execute("SELECT path, cls_json FROM tracks WHERE status='done'").fetchall()
        conn1.close()
        v1_by_path = {p: json.loads(cj) if cj else {} for p, cj in v1_rows}

        conn2 = sqlite3.connect("soniq.db")
        v04_rows = conn2.execute(
            "SELECT path, scalars_json, vectors_json FROM tracks WHERE status='done'"
        ).fetchall()
        conn2.close()

        matched = []
        for path, sj, vj in v04_rows:
            if path not in v1_by_path:
                continue
            val = v1_by_path[path].get(cls_name)
            if val is None:
                continue
            features = extract_features(sj, vj)
            matched.append((path, features, val))
        return matched


def try_feature_set(matched, feature_keys, cls_name, is_regression=False, threshold=0.5):
    """Try a specific feature set. Returns accuracy and model details."""
    X_rows = []
    y_vals = []

    for path, features, val in matched:
        X_rows.append([features[k] for k in feature_keys])
        if is_regression:
            y_vals.append(val)
        else:
            y_vals.append(1 if val > threshold else 0)

    X = np.array(X_rows)
    y = np.array(y_vals)

    if not is_regression and len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if is_regression:
        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X_scaled, y, cv=10, scoring="r2")
        model.fit(X_scaled, y)
        metric_name = "R²"
    else:
        model = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(model, X_scaled, y, cv=10, scoring="accuracy")
        model.fit(X_scaled, y)
        metric_name = "Acc"

    return {
        "metric": metric_name,
        "mean": scores.mean(),
        "std": scores.std(),
        "model": model,
        "scaler": scaler,
        "keys": feature_keys,
        "n_pos": int((y > (threshold if not is_regression else np.median(y))).sum()),
        "n_neg": int((y <= (threshold if not is_regression else np.median(y))).sum()),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_one.py <classifier>")
        print("Available: sad, tonal, acoustic, danceable, arousal, valence")
        return

    cls_name = sys.argv[1]
    is_regression = cls_name in ("arousal", "valence")
    use_v04 = is_regression  # arousal/valence on 1-10 scale

    matched = load_data(cls_name, use_v04=use_v04)
    print(f"=== {cls_name.upper()} === ({len(matched)} tracks)")

    all_keys = sorted(extract_features("{}", "{}").keys())
    print(f"Total features available: {len(all_keys)}")
    print()

    # Define feature groups to test
    groups = {
        "rhythm": ["tempo", "tempo_norm", "tempo_sq", "beat", "onset",
                    "beat_x_tempo", "onset_x_tempo", "rhythm_energy"],
        "energy": ["rms_mean", "rms_max", "rms_var", "rms_range", "dyn_range",
                    "rms_x_flux", "energy_density", "loudness_var"],
        "spectral": ["centroid", "centroid_std", "centroid_norm", "centroid_var",
                      "rolloff", "rolloff_std", "bandwidth", "bandwidth_std",
                      "flatness", "flux", "flux_std", "zcr",
                      "spectral_width", "centroid_x_flatness", "brightness_proxy"],
        "tonal": ["key", "key_sin", "key_cos", "mode", "major_key",
                  "mode_x_mfcc1", "chroma_std", "chroma_max", "chroma_range",
                  "tonnetz_energy"] + [f"chroma{i}" for i in range(12)] + [f"tonnetz{i}" for i in range(6)],
        "mfcc": [f"mfcc{i}" for i in range(13)] + [f"mfcc_s{i}" for i in range(13)],
        "contrast": [f"contrast{i}" for i in range(7)] + ["contrast_range",
                     "contrast_low", "contrast_high", "contrast_slope"],
        "vocal": ["vocal", "vocal_x_mfcc1", "vocal_x_flux"],
    }

    # Test each group individually
    print("--- Individual feature groups ---")
    group_scores = {}
    for name, keys in groups.items():
        result = try_feature_set(matched, keys, cls_name, is_regression)
        if result:
            group_scores[name] = result["mean"]
            print(f"  {name:<12} {result['metric']}={result['mean']:.3f} (+/-{result['std']:.3f}) [{len(keys)} feats]")
        else:
            print(f"  {name:<12} FAILED (no class split)")

    # Test all features
    print()
    result_all = try_feature_set(matched, all_keys, cls_name, is_regression)
    if result_all:
        print(f"  ALL          {result_all['metric']}={result_all['mean']:.3f} (+/-{result_all['std']:.3f}) [{len(all_keys)} feats]")

    # Combine top groups progressively
    print()
    print("--- Progressive combination (adding best groups) ---")
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)

    combined_keys = []
    best_result = None
    best_score = 0

    for name, score in sorted_groups:
        combined_keys.extend(groups[name])
        combined_keys = list(dict.fromkeys(combined_keys))  # dedupe
        result = try_feature_set(matched, combined_keys, cls_name, is_regression)
        if result:
            marker = " ***" if result["mean"] > best_score else ""
            print(f"  +{name:<11} {result['metric']}={result['mean']:.3f} (+/-{result['std']:.3f}) [{len(combined_keys)} feats]{marker}")
            if result["mean"] > best_score:
                best_score = result["mean"]
                best_result = result

    # Feature pruning on best combination
    if best_result:
        print()
        print("--- Feature pruning (dropping low-impact features) ---")
        model = best_result["model"]
        keys = best_result["keys"]

        if is_regression:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_[0])

        sorted_idx = np.argsort(importances)[::-1]

        for n in [50, 40, 30, 25, 20, 15, 12, 10, 8]:
            if n >= len(keys):
                continue
            top_keys = [keys[i] for i in sorted_idx[:n]]
            result = try_feature_set(matched, top_keys, cls_name, is_regression)
            if result:
                marker = " ***" if result["mean"] >= best_score - 0.001 else ""
                print(f"  top-{n:<3}      {result['metric']}={result['mean']:.3f} (+/-{result['std']:.3f}){marker}")
                if result["mean"] >= best_score - 0.001:
                    best_result = result
                    best_score = result["mean"]

    # Final report
    if best_result:
        print()
        print(f"=== BEST: {best_result['metric']}={best_score:.3f} with {len(best_result['keys'])} features ===")
        model = best_result["model"]
        scaler = best_result["scaler"]
        keys = best_result["keys"]

        if is_regression:
            coefs = model.coef_
            raw_weights = coefs / scaler.scale_
            raw_bias = model.intercept_ - np.sum(coefs * scaler.mean_ / scaler.scale_)
        else:
            coefs = model.coef_[0]
            raw_weights = coefs / scaler.scale_
            raw_bias = model.intercept_[0] - np.sum(coefs * scaler.mean_ / scaler.scale_)

        ranked = sorted(zip(keys, coefs, raw_weights), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop features:")
        for feat, std_c, raw_w in ranked[:15]:
            d = "+" if std_c > 0 else "-"
            print(f"  {d} {feat:<25} std={std_c:>7.3f}  raw={raw_w:>12.6f}")
        print(f"  Bias: {raw_bias:.6f}")


if __name__ == "__main__":
    main()
