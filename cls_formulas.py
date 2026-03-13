#!/usr/bin/env python3
"""Build brightness.py-style hand-crafted formulas for all cls dimensions.

For each dimension, try both:
1. Weighted z-score + sigmoid (like brightness.py)
2. Best ML regression

Pick whichever works better.
"""

import json
import sqlite3
import numpy as np
from scipy import stats as sp_stats

SONIQ_DB = "soniq.db"
PIPELINE_DB = "pipeline.db"

CLS_KEYS = [
    "happy", "sad", "relaxed", "aggressive", "party",
    "acoustic", "danceable", "instrumental", "vocal",
    "tonal", "atonal", "arousal", "valence",
]


def load_data():
    sconn = sqlite3.connect(SONIQ_DB)
    sconn.row_factory = sqlite3.Row
    srows = sconn.execute(
        "SELECT path, scalars_json, vectors_json FROM tracks "
        "WHERE scalars_json IS NOT NULL AND vectors_json IS NOT NULL"
    ).fetchall()
    sconn.close()

    librosa_by_path = {}
    for r in srows:
        scalars = json.loads(r["scalars_json"])
        vectors = json.loads(r["vectors_json"])
        librosa_by_path[r["path"]] = (scalars, vectors)

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
    for path in librosa_by_path:
        if path in cls_by_path:
            scalars, vectors = librosa_by_path[path]
            cls = cls_by_path[path]
            matched.append((scalars, vectors, cls))

    print(f"Matched {len(matched)} tracks\n")
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


def extract_features(scalars, vectors):
    """Extract all candidate features for hand-crafted formulas."""
    feat = flatten(scalars, vectors)

    tempo = feat.get("tempo", 120)
    rms = feat.get("rms_mean", 0.1)
    rms_max = feat.get("rms_max", 0.2)
    rms_var = feat.get("rms_var", 0.01)
    centroid = feat.get("centroid", 1500)
    centroid_std = feat.get("centroid_std", 500)
    rolloff = feat.get("rolloff", 3000)
    rolloff_std = feat.get("rolloff_std", 500)
    bandwidth = feat.get("bandwidth", 1500)
    bandwidth_std = feat.get("bandwidth_std", 300)
    flatness = feat.get("flatness", 0.01)
    flux = feat.get("flux", 30)
    flux_std = feat.get("flux_std", 15)
    onset = feat.get("onset", 2)
    beat = feat.get("beat", 0.3)
    zcr = feat.get("zcr", 0.05)
    vocal = feat.get("vocal", 0.5)
    mode = feat.get("mode", 1)
    dyn_range = feat.get("dyn_range", 5)

    mfcc_m = [feat.get(f"mfcc_m_{i}", 0) for i in range(13)]
    mfcc_s = [feat.get(f"mfcc_s_{i}", 0) for i in range(13)]
    contrast = [feat.get(f"contrast_{i}", 0) for i in range(7)]
    chroma = [feat.get(f"chroma_{i}", 0) for i in range(12)]
    tonnetz = [feat.get(f"tonnetz_{i}", 0) for i in range(6)]

    # Build candidate features dict for z-score approach
    candidates = {
        # Raw scalars
        "tempo": tempo,
        "rms": rms,
        "rms_max": rms_max,
        "rms_var": rms_var,
        "centroid": centroid,
        "centroid_std": centroid_std,
        "rolloff": rolloff,
        "rolloff_std": rolloff_std,
        "bandwidth": bandwidth,
        "bandwidth_std": bandwidth_std,
        "flatness": flatness,
        "flux": flux,
        "flux_std": flux_std,
        "onset": onset,
        "beat": beat,
        "zcr": zcr,
        "vocal": vocal,
        "mode": float(mode),
        "dyn_range": dyn_range,

        # MFCC means
        "mfcc_m_0": mfcc_m[0],

        # MFCC std aggregates
        "mfcc_s_sum": sum(mfcc_s),
        "mfcc_s_high": sum(mfcc_s[3:9]),
        "mfcc_s_low": sum(mfcc_s[0:3]),

        # Contrast aggregates
        "contrast_mean": float(np.mean(contrast)),
        "contrast_std": float(np.std(contrast)),
        "contrast_low": sum(contrast[0:3]) / 3,
        "contrast_mid": sum(contrast[2:5]) / 3,

        # Creative combinations
        "beat_x_onset": beat * onset,
        "beat_x_onset_x_flux": beat * onset * flux / 30,
        "rms_x_flux": rms * flux,
        "rms_x_beat": rms * beat,
        "rms_x_centroid": rms * centroid / 3000,
        "inv_contrast": 1 / (float(np.mean(contrast)) + 20),
        "wall_of_sound": rms / (float(np.mean(contrast)) + 20),
        "tonal_clarity": 1 - min(1, flatness * 100),
        "beat_x_tempo": beat * tempo / 120,
        "onset_x_tempo": onset * tempo / 120,
        "mode_x_beat": mode * beat,
        "mode_x_centroid": mode * centroid / 3000,
        "mode_x_rms": mode * rms,
    }

    return candidates


def optimize_formula(data, target_key, candidate_names, sign_hints=None):
    """Find optimal z-score weights for a target dimension.

    Uses correlation-driven weight assignment:
    1. Compute correlation of each candidate with target
    2. Use abs(correlation) as weight, sign from correlation direction
    3. Test via Spearman rank correlation
    """
    # Extract all candidate values and target
    all_candidates = []
    targets = []
    for scalars, vectors, cls in data:
        cands = extract_features(scalars, vectors)
        all_candidates.append(cands)
        targets.append(cls.get(target_key, 0))

    targets = np.array(targets)

    # Compute correlation of each candidate with target
    corrs = {}
    for name in candidate_names:
        vals = np.array([c[name] for c in all_candidates])
        if np.std(vals) < 1e-10:
            continue
        rho, _ = sp_stats.spearmanr(vals, targets)
        if not np.isnan(rho):
            corrs[name] = rho

    # Sort by absolute correlation
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Use top N features, weighted by correlation strength
    top_n = min(6, len(sorted_corrs))
    selected = sorted_corrs[:top_n]

    # Compute corpus stats for selected features
    corpus_stats = {}
    for name, _ in selected:
        vals = [c[name] for c in all_candidates]
        corpus_stats[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Compute z-score combination
    total_weight = sum(abs(c) for _, c in selected)
    weights = {name: corr / total_weight for name, corr in selected}

    # Compute scores
    scores = []
    for cands in all_candidates:
        z_sum = 0
        for name, w in weights.items():
            stats = corpus_stats[name]
            z = (cands[name] - stats["mean"]) / (stats["std"] + 1e-6)
            z_sum += z * w
        scores.append(float(1 / (1 + np.exp(-z_sum))))

    # Test correlation
    rho, _ = sp_stats.spearmanr(scores, targets)

    return {
        "target": target_key,
        "features": [(name, round(w, 4)) for name, w in weights.items()],
        "corpus_stats": corpus_stats,
        "rho": rho,
    }


def main():
    data = load_data()

    # Define which candidates to try for each target
    # Based on correlation analysis insights
    formulas = {}

    # All candidate features
    all_cands = list(extract_features(data[0][0], data[0][1]).keys())

    print(f"{'='*80}")
    print(f"HAND-CRAFTED FORMULAS (brightness.py style)")
    print(f"{'='*80}")
    print(f"\n{'TARGET':<14} | {'Spearman ρ':>11} | Features + weights")
    print("-" * 80)

    for target in CLS_KEYS:
        result = optimize_formula(data, target, all_cands)
        formulas[target] = result

        feats = ", ".join(f"{n}={w:+.3f}" for n, w in result["features"])
        print(f"{target:<14} | {result['rho']:>+11.3f} | {feats}")

    # Compare with ML regression
    print(f"\n{'='*80}")
    print(f"COMPARISON: Hand-crafted vs ML Regression")
    print(f"{'='*80}")

    ml_best = {
        "happy": 0.291, "sad": 0.565, "relaxed": 0.508, "aggressive": 0.211,
        "party": 0.436, "acoustic": 0.500, "danceable": 0.602,
        "instrumental": 0.046, "vocal": 0.046, "tonal": 0.412, "atonal": 0.418,
        "arousal": 0.372, "valence": 0.611,
    }

    print(f"\n{'TARGET':<14} | {'Formula ρ':>10} | {'ML R²':>10} | {'Winner':>10} | Notes")
    print("-" * 80)

    for target in CLS_KEYS:
        rho = formulas[target]["rho"]
        ml = ml_best.get(target, 0)
        # Note: ρ² ≈ R² for comparison (rough)
        rho_sq = rho ** 2
        winner = "formula" if rho_sq > ml else "ML"
        note = ""
        if rho_sq > ml * 1.2:
            note = "formula clearly better"
        elif ml > rho_sq * 1.2:
            note = "ML clearly better"
        else:
            note = "close"

        print(f"{target:<14} | {rho:>+10.3f} | {ml:>10.3f} | {winner:>10} | {note}")

    # Output the formulas in brightness.py format
    print(f"\n{'='*80}")
    print(f"GENERATED FORMULAS (brightness.py format)")
    print(f"{'='*80}")

    for target in CLS_KEYS:
        f = formulas[target]
        print(f"\n# --- {target.upper()} (ρ={f['rho']:+.3f}) ---")
        print(f"CORPUS_STATS_{target.upper()} = {{")
        for name, w in f["features"]:
            stats = f["corpus_stats"][name]
            print(f'    "{name}": {{"mean": {stats["mean"]:.4f}, "std": {stats["std"]:.4f}}},')
        print("}")
        print(f"WEIGHTS_{target.upper()} = {{")
        for name, w in f["features"]:
            print(f'    "{name}": {w:.4f},')
        print("}")


if __name__ == "__main__":
    main()
