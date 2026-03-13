#!/usr/bin/env python3
"""Calibration — libro-soniq.db (123 features) vs pipeline.db (MusiCNN ground truth).

Tests how well the full v0.5 librosa feature set predicts each MusiCNN cls dimension.
No hand-crafted feature engineering — raw 123 features + standard ML regressors.

Reports:
  1. Per-target R² with GBR and Ridge
  2. Top feature importances per target
  3. Spearman correlations of new features vs cls targets
"""

import json
import sqlite3
import numpy as np
from scipy import stats as sp_stats

LIBRO_DB = "libro-soniq.db"
PIPELINE_DB = "pipeline.db"

CLS_KEYS = [
    "happy", "sad", "relaxed", "aggressive", "party",
    "acoustic", "danceable", "instrumental", "vocal",
    "tonal", "atonal", "arousal", "valence",
]

# New features added in libro_extract.py (not present in soniq.db v0.4)
NEW_FEATURES = [
    "onset_rate", "spectral_entropy", "spectral_crest",
    "spectral_skew", "spectral_kurtosis",
    "bass_ratio", "mid_ratio", "treble_ratio", "bass_mid_ratio",
    "low_energy_rate", "energy_skew", "energy_kurtosis",
    "harm_energy", "perc_energy", "harm_perc_ratio", "harm_fraction",
    "beat_regularity", "rhythm_complexity", "plp_mean", "plp_stability",
    "mfcc_delta_var", "mfcc_delta2_var",
    "mod_flatness", "mod_crest", "mod_centroid",
    # vectors (flattened)
    "mfcc_delta_0", "mfcc_delta_1", "mfcc_delta_2", "mfcc_delta_3",
    "mfcc_delta_4", "mfcc_delta_5", "mfcc_delta_6", "mfcc_delta_7",
    "mfcc_delta_8", "mfcc_delta_9", "mfcc_delta_10", "mfcc_delta_11", "mfcc_delta_12",
    "mfcc_delta2_0", "mfcc_delta2_1", "mfcc_delta2_2", "mfcc_delta2_3",
    "mfcc_delta2_4", "mfcc_delta2_5", "mfcc_delta2_6", "mfcc_delta2_7",
    "mfcc_delta2_8", "mfcc_delta2_9", "mfcc_delta2_10", "mfcc_delta2_11", "mfcc_delta2_12",
]


def load_data():
    """Load matched tracks from libro-soniq.db + pipeline.db."""
    lconn = sqlite3.connect(LIBRO_DB)
    lconn.row_factory = sqlite3.Row
    lrows = lconn.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks "
        "WHERE status='done' AND scalars_json IS NOT NULL"
    ).fetchall()
    lconn.close()

    libro_by_path = {}
    for r in lrows:
        scalars = json.loads(r["scalars_json"])
        vectors = json.loads(r["vectors_json"])
        libro_by_path[r["path"]] = (scalars, vectors)

    pconn = sqlite3.connect(PIPELINE_DB)
    pconn.row_factory = sqlite3.Row
    prows = pconn.execute(
        "SELECT path, cls_json FROM tracks "
        "WHERE status='done' AND cls_json IS NOT NULL"
    ).fetchall()
    pconn.close()

    cls_by_path = {}
    for r in prows:
        cls_by_path[r["path"]] = json.loads(r["cls_json"])

    matched = []
    for path in libro_by_path:
        if path in cls_by_path:
            scalars, vectors = libro_by_path[path]
            cls = cls_by_path[path]
            matched.append((scalars, vectors, cls))

    print(f"Matched {len(matched)} tracks")
    print(f"  libro-soniq.db: {len(libro_by_path)} tracks")
    print(f"  pipeline.db:    {len(cls_by_path)} tracks")
    return matched


def flatten(scalars, vectors):
    feat = dict(scalars)
    for key, vals in vectors.items():
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                feat[f"{key}_{i}"] = v
        else:
            feat[key] = vals
    return feat


def run_calibration(data):
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Build feature matrix from raw features only (no engineering)
    X_rows = []
    y_dict = {t: [] for t in CLS_KEYS}

    for scalars, vectors, cls in data:
        feat = flatten(scalars, vectors)
        X_rows.append(feat)
        for t in CLS_KEYS:
            y_dict[t].append(cls.get(t, 0))

    feat_names = sorted(X_rows[0].keys())
    X = np.array([[row.get(f, 0) for f in feat_names] for row in X_rows], dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n{'='*95}")
    print(f"CALIBRATION — {X.shape[1]} raw features (no engineering)")
    print(f"  libro-soniq.db v0.5 features vs pipeline.db MusiCNN ground truth")
    print(f"{'='*95}")

    # Previous best from soniq.db (v0.4, 69 raw features)
    prev_raw = {
        "happy": 0.209, "sad": 0.525, "relaxed": 0.35, "aggressive": 0.241,
        "party": 0.35, "acoustic": 0.40, "danceable": 0.530,
        "instrumental": -0.197, "vocal": -0.188, "tonal": 0.30, "atonal": 0.30,
        "arousal": 0.314, "valence": 0.562,
    }

    # Round 4 best (v0.4 + engineered features)
    prev_eng = {
        "happy": 0.307, "sad": 0.578, "relaxed": 0.489, "aggressive": 0.215,
        "party": 0.421, "acoustic": 0.519, "danceable": 0.581,
        "instrumental": -0.029, "vocal": -0.030, "tonal": 0.384, "atonal": 0.374,
        "arousal": 0.364, "valence": 0.578,
    }

    print(f"\n{'TARGET':<14} | {'v0.4 raw':>9} | {'v0.4+eng':>9} | {'v0.5 Ridge':>10} | {'v0.5 GBR':>9} | {'v0.5 best':>9} | {'Δ vs eng':>9} | Status")
    print("-" * 105)

    results = {}
    for target in CLS_KEYS:
        y = np.array(y_dict[target], dtype=float)
        if np.std(y) < 1e-6:
            continue

        ridge_scores = cross_val_score(
            Ridge(alpha=1.0), X_scaled, y, cv=5, scoring="r2"
        )

        gbr_scores = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=10, random_state=42
            ),
            X_scaled, y, cv=5, scoring="r2"
        )

        ridge_r2 = ridge_scores.mean()
        gbr_r2 = gbr_scores.mean()
        best = max(ridge_r2, gbr_r2)
        pr = prev_raw.get(target, 0)
        pe = prev_eng.get(target, 0)
        delta = best - pe

        results[target] = {"ridge": ridge_r2, "gbr": gbr_r2, "best": best}

        if best > 0.5:
            status = "GOOD"
        elif best > 0.3:
            status = "usable"
        elif best > 0.1:
            status = "weak"
        else:
            status = "BROKEN"

        if delta > 0.05:
            status += " ++"
        elif delta > 0.02:
            status += " +"
        elif delta < -0.02:
            status += " --"

        print(f"{target:<14} | {pr:>9.3f} | {pe:>9.3f} | {ridge_r2:>10.3f} | {gbr_r2:>9.3f} | {best:>9.3f} | {delta:>+9.3f} | {status}")

    # Feature importance for all targets
    print(f"\n{'='*95}")
    print("TOP FEATURES per target (GBR importance)")
    print(f"{'='*95}")

    for target in CLS_KEYS:
        y = np.array(y_dict[target], dtype=float)
        if np.std(y) < 1e-6:
            continue

        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        gbr.fit(X_scaled, y)
        imp = sorted(zip(feat_names, gbr.feature_importances_), key=lambda x: -x[1])

        # Mark new features
        top = imp[:10]
        print(f"\n{target} (R²={results[target]['best']:.3f}):")
        for name, score in top:
            marker = " NEW" if name in NEW_FEATURES else ""
            bar = "#" * int(score * 200)
            print(f"  {name:<35} {score:.4f} {bar}{marker}")

    # New features impact analysis
    print(f"\n{'='*95}")
    print("NEW FEATURE CORRELATIONS with cls targets (Spearman)")
    print(f"{'='*95}")

    # Only check scalar new features (not flattened vectors)
    scalar_new = [f for f in NEW_FEATURES if not f.startswith("mfcc_delta_") and not f.startswith("mfcc_delta2_")]

    print(f"\n{'Feature':<25}", end="")
    for t in CLS_KEYS:
        print(f" | {t:>6}", end="")
    print()
    print("-" * (25 + len(CLS_KEYS) * 9))

    for feat_name in sorted(scalar_new):
        vals = []
        for scalars, vectors, cls in data:
            f = flatten(scalars, vectors)
            vals.append(f.get(feat_name, 0))
        vals = np.array(vals)
        if np.std(vals) < 1e-10:
            continue

        print(f"{feat_name:<25}", end="")
        for target in CLS_KEYS:
            y = np.array([c.get(target, 0) for _, _, c in data])
            rho, _ = sp_stats.spearmanr(vals, y)
            marker = "*" if abs(rho) > 0.3 else " "
            print(f" | {rho:>+5.2f}{marker}", end="")
        print()

    # Summary
    print(f"\n{'='*95}")
    print("SUMMARY — v0.5 raw features vs v0.4 + engineering")
    print(f"{'='*95}")

    improved = 0
    regressed = 0
    for target in CLS_KEYS:
        if target in results:
            best = results[target]["best"]
            prev = prev_eng.get(target, 0)
            delta = best - prev
            direction = ">>>" if delta > 0.05 else ">>" if delta > 0.02 else ">" if delta > 0 else "=" if delta > -0.02 else "<"
            print(f"  {target:<14} {prev:.3f} {direction} {best:.3f}  ({delta:+.3f})")
            if delta > 0.02:
                improved += 1
            elif delta < -0.02:
                regressed += 1

    print(f"\n  Improved (>+0.02): {improved}")
    print(f"  Regressed (<-0.02): {regressed}")
    print(f"  Stable: {len(results) - improved - regressed}")


if __name__ == "__main__":
    data = load_data()
    run_calibration(data)
