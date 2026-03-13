#!/usr/bin/env python3
"""Explore correlations between librosa features and MusiCNN cls targets.

Identifies which librosa features best predict each cls dimension,
then tests hand-crafted weighted formulas (brightness.py style).
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
# bright/dark excluded — already solved in brightness.py


def load_data():
    """Load matched tracks from both databases."""
    # Get librosa features from soniq.db
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

    # Get cls targets from pipeline.db
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

    # Match on path
    matched = []
    for path in librosa_by_path:
        if path in cls_by_path:
            scalars, vectors = librosa_by_path[path]
            cls = cls_by_path[path]
            matched.append((scalars, vectors, cls))

    print(f"Matched {len(matched)} tracks with both librosa + cls data\n")
    return matched


def flatten_features(scalars, vectors):
    """Flatten scalars + vectors into a single feature dict."""
    feat = dict(scalars)

    # Flatten vector features
    for key, vals in vectors.items():
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                feat[f"{key}_{i}"] = v
        else:
            feat[key] = vals

    return feat


def compute_correlations(data):
    """Compute Pearson + Spearman correlations for each cls target."""
    # Build feature matrix
    all_feats = []
    all_cls = []
    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        all_feats.append(feat)
        all_cls.append(cls)

    # Get all feature names
    feat_names = sorted(all_feats[0].keys())
    n = len(data)

    print(f"{'TARGET':<14} | {'TOP CORRELATED FEATURES (Spearman r)'}")
    print("-" * 100)

    correlations = {}

    for target in CLS_KEYS:
        y = np.array([c.get(target, 0) for c in all_cls], dtype=float)

        # Skip if constant
        if np.std(y) < 1e-6:
            print(f"{target:<14} | CONSTANT — skip")
            continue

        corrs = []
        for fname in feat_names:
            x = np.array([f.get(fname, 0) for f in all_feats], dtype=float)
            if np.std(x) < 1e-6:
                continue
            # Spearman is more robust to nonlinear relationships
            rho, pval = sp_stats.spearmanr(x, y)
            if not np.isnan(rho):
                corrs.append((fname, rho, pval))

        # Sort by absolute correlation
        corrs.sort(key=lambda c: abs(c[1]), reverse=True)
        correlations[target] = corrs

        top = corrs[:8]
        parts = [f"{name}={rho:+.3f}" for name, rho, _ in top]
        print(f"{target:<14} | {', '.join(parts)}")

    return correlations, all_feats, all_cls


def engineer_features(scalars, vectors):
    """Creative feature engineering — round 2, bolder combinations."""
    feat = flatten_features(scalars, vectors)
    eng = {}

    # --- Raw scalars we'll use ---
    tempo = feat.get("tempo", 120)
    rms = feat.get("rms_mean", 0.1)
    centroid = feat.get("centroid_mean", 1500)
    rolloff = feat.get("rolloff_mean", 3000)
    bandwidth = feat.get("bandwidth_mean", 1500)
    flatness = feat.get("flatness_mean", 0.01)
    flux = feat.get("spectral_flux", 30)
    onsets = feat.get("onset_rate", 2)
    beats = feat.get("beat_strength", 0.3)
    zcr = feat.get("zcr_mean", 0.05)
    vocal = feat.get("vocal_probability", 0.5)
    duration = feat.get("duration_s", 200)

    # MFCC means
    mfcc = [feat.get(f"mfcc_mean_{i}", 0) for i in range(13)]
    mfcc_std = [feat.get(f"mfcc_std_{i}", 0) for i in range(13)]

    # Spectral contrast
    contrast = [feat.get(f"spectral_contrast_{i}", 0) for i in range(7)]

    # Chroma
    chroma = [feat.get(f"chroma_mean_{i}", 0) for i in range(12)]

    # Tonnetz
    tonnetz = [feat.get(f"tonnetz_{i}", 0) for i in range(6)]

    # ===== AGGRESSIVE =====
    # High energy + harsh spectrum + fast onsets
    eng["aggr_drive"] = rms * flux * onsets  # raw power × attack rate
    eng["aggr_harsh"] = zcr * flatness * 1000  # noise-like texture
    eng["aggr_loud_fast"] = rms * tempo / 120  # loud + fast
    eng["aggr_low_centroid_loud"] = rms / (centroid / 5000 + 0.1)  # heavy bass + loud
    eng["aggr_onset_density"] = onsets * rms  # percussive hits × volume
    eng["aggr_bandwidth_push"] = bandwidth * rms / 1000  # wide spectrum + loud
    eng["aggr_mfcc_bite"] = abs(mfcc[3]) * rms if len(mfcc) > 3 else 0  # mid-freq edge
    eng["aggr_contrast_range"] = (max(contrast) - min(contrast)) * rms if contrast else 0
    eng["aggr_flux_squared"] = flux ** 2 / 1000  # emphasize high-flux tracks
    eng["aggr_no_vocal_drive"] = (1 - vocal) * rms * flux  # instrumental aggression

    # ===== HAPPY =====
    # Major key + bright + upbeat + moderate tempo
    eng["happy_bright_bounce"] = (centroid / 3000) * beats * (tempo / 120)
    eng["happy_major"] = sum(chroma[i] for i in [0, 2, 4, 5, 7, 9, 11]) / 7 if chroma else 0  # major scale degrees
    eng["happy_minor"] = sum(chroma[i] for i in [0, 2, 3, 5, 7, 8, 10]) / 7 if chroma else 0  # minor scale degrees
    eng["happy_major_minus_minor"] = eng["happy_major"] - eng["happy_minor"]  # major key tendency
    eng["happy_bright_tempo"] = centroid * tempo / (3000 * 120)
    eng["happy_chroma_var"] = float(np.std(chroma)) if chroma else 0  # harmonic variety
    eng["happy_tonnetz_pos"] = sum(max(0, t) for t in tonnetz) if tonnetz else 0  # positive tonal intervals
    eng["happy_mfcc1_high"] = max(0, mfcc[0]) / 200 if mfcc else 0  # overall energy level
    eng["happy_bounce_vocal"] = beats * vocal * (tempo / 120)  # singalong factor
    eng["happy_rolloff_tempo"] = (rolloff / 5000) * (tempo / 120)
    eng["happy_not_flat"] = (1 - min(1, flatness * 100)) * beats  # tonal + rhythmic

    # ===== RELAXED =====
    # Low energy + slow + smooth spectrum + tonal
    eng["relax_calm"] = 1 / (rms * 10 + 0.1) * (1 / (tempo / 60 + 0.1))  # inverse energy × inverse speed
    eng["relax_smooth"] = (1 - min(1, flatness * 100)) * (1 - min(1, zcr * 10))  # tonal + smooth
    eng["relax_low_onset"] = 1 / (onsets + 0.5)  # few sudden changes
    eng["relax_mfcc_warm"] = max(0, mfcc[0]) / 200 * (1 - min(1, zcr * 10)) if mfcc else 0  # warm tone
    eng["relax_low_flux"] = 1 / (flux + 1)  # spectral stability
    eng["relax_slow_quiet"] = (1 - min(1, rms * 5)) * max(0, 1 - tempo / 200)
    eng["relax_acoustic_sig"] = (1 - min(1, flatness * 100)) * (1 - vocal)  # acoustic instrumental

    # ===== AROUSAL (1-9 scale) =====
    # Overall activation: loud + fast + dense onsets + spectral energy
    eng["arousal_power"] = rms * tempo * onsets / 100  # triple-power indicator
    eng["arousal_spectral_mass"] = centroid * bandwidth * rms / 1e7  # spectral footprint × volume
    eng["arousal_onset_tempo"] = onsets * tempo / 120
    eng["arousal_flux_rms"] = flux * rms  # spectral change × loudness
    eng["arousal_beat_drive"] = beats * rms * tempo / 120
    eng["arousal_dynamic_range"] = mfcc_std[0] * rms if mfcc_std else 0  # loudness variation
    eng["arousal_zcr_energy"] = zcr * rms * 100  # noisy energy

    # ===== VALENCE (1-9 scale) =====
    # Positive = major, bright, moderate tempo; negative = minor, dark, extreme
    eng["valence_tonal_bright"] = (centroid / 3000) * (1 - min(1, flatness * 100))
    eng["valence_major_key"] = eng["happy_major_minus_minor"]  # reuse major tendency
    eng["valence_moderate_tempo"] = 1 - abs(tempo - 110) / 110  # peak at ~110 BPM
    eng["valence_harmonic_rich"] = float(np.mean(contrast)) if contrast else 0
    eng["valence_tonnetz_sum"] = sum(tonnetz) if tonnetz else 0
    eng["valence_vocal_bright"] = vocal * centroid / 3000

    # ===== DANCEABLE =====
    eng["dance_groove"] = beats * tempo / 120 * rms  # beat + tempo + energy
    eng["dance_regular_beat"] = beats * (1 - min(1, abs(tempo - 120) / 60))  # strongest near 120bpm
    eng["dance_low_end"] = max(0, contrast[0]) * beats if contrast else 0  # bass presence + beat
    eng["dance_four_on_floor"] = beats ** 2 * rms  # emphasize strong regular beats
    eng["dance_onset_regular"] = onsets * beats  # percussive + rhythmic

    # ===== INSTRUMENTAL / VOCAL =====
    eng["instr_pure"] = (1 - vocal) ** 2  # squared to emphasize low-vocal
    eng["instr_spectral"] = (1 - vocal) * flatness * 1000  # electronic/textural
    eng["instr_complex"] = (1 - vocal) * float(np.std(contrast)) if contrast else 0
    eng["vocal_pure"] = vocal ** 2  # squared to emphasize high-vocal
    eng["vocal_mid_centroid"] = vocal * max(0, 1 - abs(centroid - 2000) / 2000)  # voice freq range
    eng["vocal_mfcc_shape"] = vocal * abs(mfcc[1]) / 50 if len(mfcc) > 1 else 0  # vocal timbre

    # ===== SAD =====
    eng["sad_slow_dark"] = (1 - min(1, rms * 5)) * max(0, 1 - centroid / 3000) * max(0, 1 - tempo / 120)
    eng["sad_minor_key"] = -eng["happy_major_minus_minor"]  # minor tendency
    eng["sad_low_beats"] = (1 - beats) * (1 - min(1, rms * 5))
    eng["sad_tonnetz_neg"] = sum(min(0, t) for t in tonnetz) if tonnetz else 0

    # ===== ACOUSTIC =====
    eng["acoustic_tonal"] = (1 - min(1, flatness * 100)) * (1 - min(1, zcr * 10))
    eng["acoustic_low_flux"] = (1 - min(1, flux / 80)) * (1 - min(1, flatness * 100))
    eng["acoustic_natural"] = (1 - min(1, flatness * 100)) * max(0, 1 - rms * 3)  # quiet + tonal
    eng["acoustic_contrast_low"] = min(contrast) if contrast else 0  # low-band presence

    # ===== TONAL / ATONAL =====
    eng["tonal_clarity"] = (1 - min(1, flatness * 100))
    eng["tonal_chroma_peak"] = max(chroma) - float(np.mean(chroma)) if chroma else 0  # dominant pitch
    eng["tonal_harmonic"] = float(np.mean([abs(t) for t in tonnetz])) if tonnetz else 0
    eng["atonal_noise"] = flatness * zcr * 1000  # noise-like
    eng["atonal_chroma_flat"] = 1 / (float(np.std(chroma)) + 0.01) if chroma else 0  # no dominant pitch

    # ===== PARTY =====
    eng["party_energy"] = rms * beats * (tempo / 120)
    eng["party_loud_fast"] = rms * tempo / 100
    eng["party_bass_beat"] = max(0, contrast[0]) * beats * rms if contrast else 0

    # ===== CROSS-DIMENSION INTERACTIONS (2nd order) =====
    eng["energy_x_brightness"] = rms * centroid / 3000
    eng["rhythm_x_tone"] = beats * (1 - min(1, flatness * 100))
    eng["speed_x_noise"] = tempo * zcr / 10
    eng["vocal_x_energy"] = vocal * rms
    eng["onset_x_spectral_width"] = onsets * bandwidth / 2000

    return eng


def test_handcrafted(data, correlations):
    """Test hand-crafted formulas against ground truth."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    # Build full feature matrix: raw + engineered
    X_rows = []
    y_dict = {t: [] for t in CLS_KEYS}

    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        eng = engineer_features(scalars, vectors)
        feat.update(eng)
        X_rows.append(feat)
        for t in CLS_KEYS:
            y_dict[t].append(cls.get(t, 0))

    # Consistent feature order
    feat_names = sorted(X_rows[0].keys())
    X = np.array([[row.get(f, 0) for f in feat_names] for row in X_rows], dtype=float)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n{'='*80}")
    print(f"REGRESSION RESULTS — {X.shape[1]} features ({len(feat_names) - len(flatten_features(data[0][0], data[0][1]))} engineered)")
    print(f"{'='*80}")
    print(f"\n{'TARGET':<14} | {'Ridge R²':>10} | {'GBR R²':>10} | {'Δ vs baseline':>14} | Notes")
    print("-" * 80)

    # Baseline R² from summary (raw 69 features only, GBR)
    baseline = {
        "happy": 0.209, "sad": 0.525, "relaxed": 0.35, "aggressive": 0.241,
        "party": 0.35, "acoustic": 0.40, "danceable": 0.530,
        "instrumental": -0.197, "vocal": -0.188, "tonal": 0.30, "atonal": 0.30,
        "arousal": 0.314, "valence": 0.562,
    }

    for target in CLS_KEYS:
        y = np.array(y_dict[target], dtype=float)
        if np.std(y) < 1e-6:
            continue

        ridge_scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=5, scoring="r2")
        gbr_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=42),
            X_scaled, y, cv=5, scoring="r2"
        )

        ridge_r2 = ridge_scores.mean()
        gbr_r2 = gbr_scores.mean()
        base = baseline.get(target, 0)
        delta = gbr_r2 - base

        note = ""
        if gbr_r2 > 0.5:
            note = "GOOD"
        elif gbr_r2 > 0.3:
            note = "usable"
        elif gbr_r2 > 0.1:
            note = "weak"
        else:
            note = "BROKEN"

        if delta > 0.05:
            note += " ↑↑"
        elif delta > 0.02:
            note += " ↑"

        print(f"{target:<14} | {ridge_r2:>10.3f} | {gbr_r2:>10.3f} | {delta:>+14.3f} | {note}")

    # Feature importance for worst targets
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE — worst targets")
    print(f"{'='*80}")

    for target in ["happy", "aggressive", "arousal", "instrumental", "vocal"]:
        y = np.array(y_dict[target], dtype=float)
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        gbr.fit(X_scaled, y)
        imp = list(zip(feat_names, gbr.feature_importances_))
        imp.sort(key=lambda x: x[1], reverse=True)
        top = imp[:10]
        print(f"\n{target}:")
        for name, score in top:
            print(f"  {name:<35} {score:.4f}")


if __name__ == "__main__":
    data = load_data()
    correlations, feats, cls_data = compute_correlations(data)
    test_handcrafted(data, correlations)
