#!/usr/bin/env python3
"""Train all v0.5 classifiers and export weights to py/classifiers/*.py.

Reads libro-soniq.db (expanded librosa features) + pipeline.db (V1 ground truth).
Trains optimal classifier per target, exports raw-space weights (scaler baked in)
to self-contained Python files.

Usage:
    python3 train_classifiers.py                    # train + export
    python3 train_classifiers.py --report-only      # just show CV scores
"""

import argparse
import json
import sqlite3
import textwrap
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
FEATURES_DB = ROOT / "libro-soniq.db"
PIPELINE_DB = ROOT / "pipeline.db"
OUT_DIR = ROOT / "py" / "classifiers"


# ── Feature extraction ────────────────────────────────────────────

def extract_all_features(scalars_json, vectors_json):
    """Extract all features as a flat dict (matches _features.prepare)."""
    s = json.loads(scalars_json) if scalars_json else {}
    v = json.loads(vectors_json) if vectors_json else {}

    mfcc = v.get("mfcc_m", [0] * 13)
    mfcc_s = v.get("mfcc_s", [0] * 13)
    contrast = v.get("contrast", [0] * 7)
    chroma = v.get("chroma", [0] * 12)
    tonnetz = v.get("tonnetz", [0] * 6)
    mfcc_delta = v.get("mfcc_delta", [0] * 13)
    mfcc_delta2 = v.get("mfcc_delta2", [0] * 13)

    features = {}

    # Scalars
    for key in ("centroid", "centroid_std", "rolloff", "rolloff_std",
                "bandwidth", "bandwidth_std", "flatness", "flux", "flux_std",
                "zcr", "rms_mean", "rms_max", "rms_var", "dyn_range",
                "tempo", "key", "mode", "onset", "beat", "vocal", "duration"):
        features[key] = s.get(key, 0)

    # New scalars
    for key in ("low_energy_rate", "energy_skew", "energy_kurtosis",
                "bass_ratio", "mid_ratio", "treble_ratio", "bass_mid_ratio",
                "spectral_skew", "spectral_kurtosis", "spectral_entropy", "spectral_crest",
                "mfcc_delta_var", "mfcc_delta2_var",
                "mod_flatness", "mod_crest", "mod_centroid",
                "harm_energy", "perc_energy", "harm_perc_ratio", "harm_fraction",
                "beat_regularity", "rhythm_complexity", "plp_mean", "plp_stability",
                "onset_rate"):
        features[key] = s.get(key, 0)

    # Derived scalars
    features["rms_range"] = features["rms_max"] - features["rms_mean"]
    features["centroid_var"] = features["centroid_std"] ** 2 if features["centroid_std"] else 0
    features["low_energy"] = 1.0 / (1.0 + features["centroid"]) if features["centroid"] else 0
    features["tempo_sq"] = features["tempo"] ** 2

    # MFCCs
    for i in range(13):
        features[f"mfcc{i}"] = mfcc[i] if i < len(mfcc) else 0
        features[f"mfcc_s{i}"] = mfcc_s[i] if i < len(mfcc_s) else 0

    # MFCC deltas
    for i in range(13):
        features[f"mfcc_d{i}"] = mfcc_delta[i] if i < len(mfcc_delta) else 0
        features[f"mfcc_d2_{i}"] = mfcc_delta2[i] if i < len(mfcc_delta2) else 0

    # Contrast
    for i in range(7):
        features[f"contrast{i}"] = contrast[i] if i < len(contrast) else 0

    # Chroma
    for i in range(12):
        features[f"chroma{i}"] = chroma[i] if i < len(chroma) else 0

    # Tonnetz
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
    features["harm_x_bass"] = features["harm_fraction"] * features["bass_ratio"]
    features["perc_x_beat_reg"] = features["perc_energy"] * features["beat_regularity"]
    features["delta_x_flux"] = features["mfcc_delta_var"] * features["flux"]
    features["plp_x_tempo"] = features["plp_stability"] * features["tempo"] / 200.0
    features["onset_rate_x_rms"] = features["onset_rate"] * features["rms_mean"]

    return features


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    """Load matched tracks from libro-soniq.db + pipeline.db (v0.3).

    All ground truth comes from pipeline.db — the v0.3 MusiCNN + EffNet
    pipeline output. This includes moods, arousal/valence, and bright/dark.
    """
    conn1 = sqlite3.connect(str(PIPELINE_DB))
    v1_rows = conn1.execute(
        "SELECT path, cls_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn1.close()

    v1_by_path = {}
    for path, cj in v1_rows:
        cls = json.loads(cj) if cj else {}
        v1_by_path[path] = cls

    conn2 = sqlite3.connect(str(FEATURES_DB))
    v04_rows = conn2.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks WHERE status='done'"
    ).fetchall()
    conn2.close()

    matched = []
    for path, sj, vj in v04_rows:
        if path not in v1_by_path:
            continue
        features = extract_all_features(sj, vj)
        v1_cls = v1_by_path[path]
        matched.append((path, features, v1_cls))

    return matched


ALL_FEATURE_KEYS = sorted(extract_all_features("{}", "{}").keys())


# ── Classifier configs ────────────────────────────────────────────

# LR classifiers: trained as LogisticRegression, exported as raw weights
LR_CLASSIFIERS = [
    "happy", "relaxed", "aggressive", "party",
    "acoustic", "danceable", "instrumental",
    "sad", "tonal",
]

# Ridge classifiers: trained as Ridge regression, exported as raw weights
RIDGE_CLASSIFIERS = ["arousal", "valence"]

# Special: brightness uses LR, timbre uses centroid z-score
SPECIAL_CLASSIFIERS = ["brightness", "timbre"]


def train_lr(matched, cls_name, threshold=0.5):
    """Train logistic regression for one binary classifier."""
    X_rows, y_vals = [], []
    for path, features, v1_cls in matched:
        val = v1_cls.get(cls_name)
        if val is None:
            continue
        X_rows.append([features[k] for k in ALL_FEATURE_KEYS])
        y_vals.append(1 if val > threshold else 0)

    X = np.array(X_rows)
    y = np.array(y_vals)

    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with all features first
    lr_full = LogisticRegression(max_iter=2000, C=1.0)
    lr_full.fit(X_scaled, y)

    # Feature selection: prune to top N
    importances = np.abs(lr_full.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]

    best_acc, best_n, best_keys = 0, len(ALL_FEATURE_KEYS), ALL_FEATURE_KEYS

    for n in [len(ALL_FEATURE_KEYS), 40, 30, 20, 15, 10]:
        if n > len(ALL_FEATURE_KEYS):
            continue
        top_keys = [ALL_FEATURE_KEYS[i] for i in sorted_idx[:n]]

        X_sub_rows, y_sub = [], []
        for path, features, v1_cls in matched:
            val = v1_cls.get(cls_name)
            if val is None:
                continue
            X_sub_rows.append([features[k] for k in top_keys])
            y_sub.append(1 if val > threshold else 0)

        X_sub = np.array(X_sub_rows)
        y_sub = np.array(y_sub)

        scaler_sub = StandardScaler()
        X_sub_scaled = scaler_sub.fit_transform(X_sub)

        lr_sub = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(lr_sub, X_sub_scaled, y_sub, cv=10, scoring="accuracy")
        acc = scores.mean()

        if acc > best_acc - 0.002:
            if n < best_n or acc > best_acc:
                best_acc = acc
                best_n = n
                best_keys = top_keys

    # Final train with best features
    X_final_rows, y_final = [], []
    for path, features, v1_cls in matched:
        val = v1_cls.get(cls_name)
        if val is None:
            continue
        X_final_rows.append([features[k] for k in best_keys])
        y_final.append(1 if val > threshold else 0)

    X_final = np.array(X_final_rows)
    y_final = np.array(y_final)

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    lr_final = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(lr_final, X_final_scaled, y_final, cv=10, scoring="accuracy")
    lr_final.fit(X_final_scaled, y_final)

    # Compute raw-space weights (scaler baked in)
    raw_weights = lr_final.coef_[0] / scaler_final.scale_
    raw_bias = lr_final.intercept_[0] - np.sum(lr_final.coef_[0] * scaler_final.mean_ / scaler_final.scale_)

    return {
        "type": "lr",
        "keys": best_keys,
        "weights": raw_weights,
        "bias": raw_bias,
        "cv_acc": scores.mean(),
        "cv_std": scores.std(),
        "n_pos": int(y_final.sum()),
        "n_neg": int(len(y_final) - y_final.sum()),
    }


def train_ridge(matched, cls_name):
    """Train Ridge regression for continuous output (arousal/valence).

    Ground truth from pipeline.db (v0.3 MusiCNN emomusic output).
    """
    X_rows, y_vals = [], []
    for path, features, v1_cls in matched:
        val = v1_cls.get(cls_name)
        if val is None:
            continue
        X_rows.append([features[k] for k in ALL_FEATURE_KEYS])
        y_vals.append(val)

    X = np.array(X_rows)
    y = np.array(y_vals)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_scaled, y)
    importances = np.abs(ridge_full.coef_)
    sorted_idx = np.argsort(importances)[::-1]

    best_r2, best_n, best_keys = -999, len(ALL_FEATURE_KEYS), ALL_FEATURE_KEYS

    for n in [len(ALL_FEATURE_KEYS), 40, 30, 20, 15]:
        if n > len(ALL_FEATURE_KEYS):
            continue
        top_keys = [ALL_FEATURE_KEYS[i] for i in sorted_idx[:n]]
        X_sub_rows, y_sub = [], []
        for path, features, v1_cls in matched:
            val = v1_cls.get(cls_name)
            if val is None:
                continue
            X_sub_rows.append([features[k] for k in top_keys])
            y_sub.append(val)

        X_sub = np.array(X_sub_rows)
        y_sub = np.array(y_sub)
        scaler_sub = StandardScaler()
        X_sub_scaled = scaler_sub.fit_transform(X_sub)

        ridge_sub = Ridge(alpha=1.0)
        scores = cross_val_score(ridge_sub, X_sub_scaled, y_sub, cv=10, scoring="r2")
        r2 = scores.mean()

        if r2 > best_r2 - 0.01:
            if n < best_n or r2 > best_r2:
                best_r2 = r2
                best_n = n
                best_keys = top_keys

    # Final train
    X_final_rows, y_final = [], []
    for path, features, v1_cls in matched:
        val = v1_cls.get(cls_name)
        if val is None:
            continue
        X_final_rows.append([features[k] for k in best_keys])
        y_final.append(val)

    X_final = np.array(X_final_rows)
    y_final = np.array(y_final)

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    ridge_final = Ridge(alpha=1.0)
    scores = cross_val_score(ridge_final, X_final_scaled, y_final, cv=10, scoring="r2")
    ridge_final.fit(X_final_scaled, y_final)

    raw_weights = ridge_final.coef_ / scaler_final.scale_
    raw_bias = ridge_final.intercept_ - np.sum(ridge_final.coef_ * scaler_final.mean_ / scaler_final.scale_)

    return {
        "type": "ridge",
        "keys": best_keys,
        "weights": raw_weights,
        "bias": float(raw_bias),
        "cv_r2": scores.mean(),
        "cv_std": scores.std(),
        "y_min": float(y_final.min()),
        "y_max": float(y_final.max()),
    }


def train_brightness(matched):
    """Train LR brightness using rescaling approach from calibrate_brightness.py.

    EffNet bright/dark values in pipeline.db are compressed to 0.42-0.54 range
    (std=0.021). Direct binarization at 0.5 trains on noise. Instead:
    1. Compute bright/(bright+dark) ratio
    2. Rescale from compressed 0.37-0.54 range to 0-1
    3. Filter ambiguous tracks (skip 0.4-0.6)
    4. Train LR on clear bright/dark extremes
    """
    X_rows, y_vals = [], []
    n_skipped = 0
    for path, features, v1_cls in matched:
        bright_val = v1_cls.get("bright")
        dark_val = v1_cls.get("dark")
        if bright_val is None or dark_val is None:
            continue

        # Rescale from compressed EffNet range
        total = bright_val + dark_val
        ratio = bright_val / total if total > 0 else 0.5
        rescaled = max(0.0, min(1.0, (ratio - 0.37) / (0.54 - 0.37)))

        # Filter ambiguous middle — only train on clear extremes
        if rescaled > 0.6:
            label = 1  # bright
        elif rescaled < 0.4:
            label = 0  # dark
        else:
            n_skipped += 1
            continue

        X_rows.append([features[k] for k in ALL_FEATURE_KEYS])
        y_vals.append(label)

    X = np.array(X_rows)
    y = np.array(y_vals)
    print(f"    Brightness: {len(y)} tracks after filtering ({n_skipped} ambiguous skipped)")

    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X_scaled, y)

    importances = np.abs(lr.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]

    best_acc, best_n, best_keys = 0, len(ALL_FEATURE_KEYS), ALL_FEATURE_KEYS

    for n in [len(ALL_FEATURE_KEYS), 40, 30, 20, 15, 10]:
        if n > len(ALL_FEATURE_KEYS):
            continue
        top_keys = [ALL_FEATURE_KEYS[i] for i in sorted_idx[:n]]
        X_sub_rows, y_sub = [], []
        for path, features, v1_cls in matched:
            bright_val = v1_cls.get("bright")
            dark_val = v1_cls.get("dark")
            if bright_val is None or dark_val is None:
                continue
            total = bright_val + dark_val
            ratio = bright_val / total if total > 0 else 0.5
            rescaled = max(0.0, min(1.0, (ratio - 0.37) / (0.54 - 0.37)))
            if rescaled > 0.6:
                label = 1
            elif rescaled < 0.4:
                label = 0
            else:
                continue
            X_sub_rows.append([features[k] for k in top_keys])
            y_sub.append(label)

        X_sub = np.array(X_sub_rows)
        y_sub = np.array(y_sub)
        scaler_sub = StandardScaler()
        X_sub_scaled = scaler_sub.fit_transform(X_sub)

        lr_sub = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(lr_sub, X_sub_scaled, y_sub, cv=10, scoring="accuracy")
        acc = scores.mean()

        if acc > best_acc - 0.002:
            if n < best_n or acc > best_acc:
                best_acc = acc
                best_n = n
                best_keys = top_keys

    # Final train with best features
    X_final_rows, y_final = [], []
    for path, features, v1_cls in matched:
        bright_val = v1_cls.get("bright")
        dark_val = v1_cls.get("dark")
        if bright_val is None or dark_val is None:
            continue
        total = bright_val + dark_val
        ratio = bright_val / total if total > 0 else 0.5
        rescaled = max(0.0, min(1.0, (ratio - 0.37) / (0.54 - 0.37)))
        if rescaled > 0.6:
            label = 1
        elif rescaled < 0.4:
            label = 0
        else:
            continue
        X_final_rows.append([features[k] for k in best_keys])
        y_final.append(label)

    X_final = np.array(X_final_rows)
    y_final = np.array(y_final)
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    lr_final = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(lr_final, X_final_scaled, y_final, cv=10, scoring="accuracy")
    lr_final.fit(X_final_scaled, y_final)

    raw_weights = lr_final.coef_[0] / scaler_final.scale_
    raw_bias = lr_final.intercept_[0] - np.sum(lr_final.coef_[0] * scaler_final.mean_ / scaler_final.scale_)

    return {
        "type": "lr",
        "keys": best_keys,
        "weights": raw_weights,
        "bias": raw_bias,
        "cv_acc": scores.mean(),
        "cv_std": scores.std(),
        "n_pos": int(y_final.sum()),
        "n_neg": int(len(y_final) - y_final.sum()),
    }


# ── Code generation ───────────────────────────────────────────────

def format_array(arr, indent=4):
    """Format numpy array as Python literal."""
    prefix = " " * indent
    vals = [f"{v:.10f}" for v in arr]
    lines = []
    for i in range(0, len(vals), 5):
        chunk = ", ".join(vals[i:i+5])
        lines.append(f"{prefix}{chunk},")
    return "\n".join(lines)


def generate_lr_file(cls_name, result):
    """Generate a self-contained LR classifier file."""
    features_list = repr(result["keys"])
    weights_str = format_array(result["weights"])
    bias = result["bias"]

    return f'''"""Classifier: {cls_name} — logistic regression (numpy only).

CV accuracy: {result["cv_acc"]:.3f} (+/- {result["cv_std"]:.3f})
Trained on {result["n_pos"] + result["n_neg"]} tracks ({result["n_pos"]} pos, {result["n_neg"]} neg).
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = {features_list}

WEIGHTS = np.array([
{weights_str}
])

BIAS = {bias:.10f}


def predict(features):
    """Predict {cls_name} probability from prepared features dict.

    Returns float 0-1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    return float(1 / (1 + np.exp(-logit)))
'''


def generate_ridge_file(cls_name, result):
    """Generate a self-contained Ridge classifier file."""
    features_list = repr(result["keys"])
    weights_str = format_array(result["weights"])
    bias = result["bias"]
    y_min = result["y_min"]
    y_max = result["y_max"]

    return f'''"""Classifier: {cls_name} — ridge regression (numpy only).

CV R²: {result["cv_r2"]:.3f} (+/- {result["cv_std"]:.3f})
Output range: {y_min:.1f} - {y_max:.1f}
Weights are in raw feature space (scaler baked in).
"""

import numpy as np

FEATURES = {features_list}

WEIGHTS = np.array([
{weights_str}
])

BIAS = {bias:.10f}
CLIP_MIN = {y_min:.1f}
CLIP_MAX = {y_max:.1f}


def predict(features):
    """Predict {cls_name} value from prepared features dict.

    Returns float clipped to [{y_min:.1f}, {y_max:.1f}].
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    val = np.dot(WEIGHTS, x) + BIAS
    return float(np.clip(val, CLIP_MIN, CLIP_MAX))
'''


def generate_brightness_file(result):
    """Generate brightness classifier that outputs bright+dark."""
    features_list = repr(result["keys"])
    weights_str = format_array(result["weights"])
    bias = result["bias"]

    return f'''"""Classifier: brightness — logistic regression (numpy only).

Consolidates py/brightness.py + py/perceived_brightness.py.
CV accuracy: {result["cv_acc"]:.3f} (+/- {result["cv_std"]:.3f})
Trained on {result["n_pos"] + result["n_neg"]} tracks ({result["n_pos"]} pos, {result["n_neg"]} neg).
"""

import numpy as np

FEATURES = {features_list}

WEIGHTS = np.array([
{weights_str}
])

BIAS = {bias:.10f}


def predict(features):
    """Predict brightness from prepared features dict.

    Returns dict {{"bright": float, "dark": float}} summing to ~1.
    """
    x = np.array([features.get(k, 0) for k in FEATURES])
    logit = np.dot(WEIGHTS, x) + BIAS
    bright = float(1 / (1 + np.exp(-logit)))
    dark = round(1 - bright, 4)
    bright = round(bright, 4)
    return {{"bright": bright, "dark": dark}}
'''


def generate_timbre_file():
    """Generate timbre classifier (centroid z-score)."""
    return '''"""Classifier: timbre — spectral centroid z-score through sigmoid.

Based on Schubert & Wolfe 2006, Peeters 2011 Timbre Toolbox.
Fixed reference from 681-track corpus.
"""

import numpy as np

CENTROID_MEAN = 1374.4
CENTROID_STD = 528.7


def predict(features):
    """Predict timbral color from prepared features dict.

    Returns dict {"brilliant": float, "warm": float} summing to ~1.
    """
    centroid = features.get("centroid", 0)
    z = (centroid - CENTROID_MEAN) / CENTROID_STD if CENTROID_STD > 0 else 0
    brilliant = round(float(1 / (1 + np.exp(-z))), 4)
    warm = round(1 - brilliant, 4)
    return {"brilliant": brilliant, "warm": warm}
'''


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true", help="Just show CV scores, don't export")
    args = parser.parse_args()

    print("=" * 60)
    print("  Train v0.5 classifiers")
    print("=" * 60)

    matched = load_data()
    print(f"  Matched tracks: {len(matched)}")
    print(f"  Total features: {len(ALL_FEATURE_KEYS)}")
    print()

    results = {}

    # Train LR classifiers
    print("--- Logistic Regression classifiers ---")
    for cls_name in LR_CLASSIFIERS:
        result = train_lr(matched, cls_name)
        if result:
            results[cls_name] = result
            print(f"  {cls_name:<15} acc={result['cv_acc']:.3f} +/-{result['cv_std']:.3f}  ({len(result['keys'])} features)")
        else:
            print(f"  {cls_name:<15} FAILED")

    # Train Ridge classifiers
    print("\n--- Ridge Regression classifiers ---")
    for cls_name in RIDGE_CLASSIFIERS:
        result = train_ridge(matched, cls_name)
        if result:
            results[cls_name] = result
            print(f"  {cls_name:<15} R²={result['cv_r2']:.3f} +/-{result['cv_std']:.3f}  ({len(result['keys'])} features)")
        else:
            print(f"  {cls_name:<15} FAILED")

    # Train brightness
    print("\n--- Special classifiers ---")
    result = train_brightness(matched)
    if result:
        results["brightness"] = result
        print(f"  brightness       acc={result['cv_acc']:.3f} +/-{result['cv_std']:.3f}  ({len(result['keys'])} features)")

    print(f"  timbre           (fixed z-score — no training needed)")

    if args.report_only:
        return

    # Export
    print()
    print("=" * 60)
    print("  Exporting classifiers")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls_name, result in results.items():
        if cls_name == "brightness":
            code = generate_brightness_file(result)
        elif result["type"] == "lr":
            code = generate_lr_file(cls_name, result)
        elif result["type"] == "ridge":
            code = generate_ridge_file(cls_name, result)

        filepath = OUT_DIR / f"{cls_name}.py"
        with open(filepath, "w") as f:
            f.write(code)
        print(f"  Wrote {filepath.relative_to(ROOT)}")

    # Timbre (fixed, no training)
    filepath = OUT_DIR / "timbre.py"
    with open(filepath, "w") as f:
        f.write(generate_timbre_file())
    print(f"  Wrote {filepath.relative_to(ROOT)}")

    print("\n  Done. Run the pipeline to verify.")


if __name__ == "__main__":
    main()
