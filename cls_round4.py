#!/usr/bin/env python3
"""Round 4 — fixed feature names + key/mode + sharper combinations.

Bug fix: previous rounds used wrong feature names (mfcc_mean_* instead of mfcc_m_*).
New: key and mode (major/minor) should help happy/sad/valence significantly.
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


def flatten_features(scalars, vectors):
    feat = dict(scalars)
    for key, vals in vectors.items():
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                feat[f"{key}_{i}"] = v
        else:
            feat[key] = vals
    return feat


def engineer_v4(scalars, vectors):
    """Round 4 — correct feature names, key/mode, sharper targeting."""
    feat = flatten_features(scalars, vectors)
    eng = {}

    # --- CORRECT scalar names ---
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
    onsets = feat.get("onset", 2)
    beats = feat.get("beat", 0.3)
    zcr = feat.get("zcr", 0.05)
    vocal = feat.get("vocal", 0.5)
    duration = feat.get("duration", 200)
    key = feat.get("key", 0)       # 0-11 pitch class
    mode = feat.get("mode", 1)     # 1=major, 0=minor — GOLD
    dyn_range = feat.get("dyn_range", 5)

    # --- CORRECT vector names ---
    mfcc_m = [feat.get(f"mfcc_m_{i}", 0) for i in range(13)]
    mfcc_s = [feat.get(f"mfcc_s_{i}", 0) for i in range(13)]
    contrast = [feat.get(f"contrast_{i}", 0) for i in range(7)]
    chroma = [feat.get(f"chroma_{i}", 0) for i in range(12)]
    tonnetz = [feat.get(f"tonnetz_{i}", 0) for i in range(6)]

    # =====================================================
    # KEY & MODE features — direct emotional indicators
    # =====================================================
    eng["is_major"] = float(mode)  # 1=major, 0=minor
    eng["is_minor"] = 1.0 - float(mode)

    # Major + bright + fast = happy
    eng["major_bright"] = mode * centroid / 3000
    eng["major_fast"] = mode * tempo / 120
    eng["major_energetic"] = mode * rms * beats
    eng["major_upbeat"] = mode * beats * (tempo / 120)

    # Minor + dark + slow = sad
    eng["minor_dark"] = (1 - mode) * max(0, 1 - centroid / 2500)
    eng["minor_slow"] = (1 - mode) * max(0, 1 - tempo / 120)
    eng["minor_quiet"] = (1 - mode) * max(0, 1 - rms * 5)
    eng["minor_sad_combo"] = (1 - mode) * max(0, 1 - centroid / 2500) * max(0, 1 - tempo / 120)

    # =====================================================
    # INSTRUMENTAL / VOCAL — MFCC std is the signal
    # =====================================================
    mfcc_s_sum = sum(mfcc_s)
    mfcc_s_high = sum(mfcc_s[3:9])  # bands 3-8 are most discriminative
    mfcc_s_low = sum(mfcc_s[0:3])

    eng["voc_mfcc_s_sum"] = mfcc_s_sum
    eng["voc_mfcc_s_high"] = mfcc_s_high
    eng["voc_mfcc_s_ratio"] = mfcc_s_high / (mfcc_s_low + 0.01)
    eng["voc_mfcc_s_mean"] = mfcc_s_sum / 13
    eng["voc_mfcc_s_max"] = max(mfcc_s)
    eng["voc_mfcc_s_geomean"] = float(np.exp(np.mean(np.log(np.array(mfcc_s) + 1e-6))))
    eng["voc_mfcc_s_product_high"] = float(np.prod(np.array(mfcc_s[3:7]) + 0.01))
    eng["voc_mfcc_s_5678"] = mfcc_s[5] + mfcc_s[6] + mfcc_s[7] + mfcc_s[8] if len(mfcc_s) > 8 else 0

    # Spectral variability
    eng["voc_centroid_var"] = centroid_std / (centroid + 1)
    eng["voc_bandwidth_var"] = bandwidth_std / (bandwidth + 1)
    eng["voc_rolloff_var"] = rolloff_std / (rolloff + 1)

    # Voice formant region (1-4kHz)
    eng["voc_formant_zone"] = max(0, 1 - abs(centroid - 2500) / 2500)

    # Inverse for instrumental
    eng["ins_mfcc_s_stable"] = 1 / (mfcc_s_sum + 1)
    eng["ins_mfcc_s_high_stable"] = 1 / (mfcc_s_high + 1)
    eng["ins_centroid_stable"] = 1 / (centroid_std + 1)

    # =====================================================
    # AGGRESSIVE — spectral contrast + loudness
    # =====================================================
    contrast_mean = float(np.mean(contrast))
    contrast_std_val = float(np.std(contrast))

    eng["aggr_contrast_inv"] = 1 / (contrast_mean + 20)
    eng["aggr_contrast_low"] = -(contrast[0] + contrast[1] + contrast[2]) / 3
    eng["aggr_contrast_flat"] = 1 / (contrast_std_val + 1)
    eng["aggr_wall"] = rms / (contrast_mean + 20)
    eng["aggr_crush"] = rms * rms_max / (contrast_std_val + 1)
    eng["aggr_distort"] = rms * flatness * 100 / (contrast_mean + 20)
    eng["aggr_rms_sq"] = rms ** 2 * 10
    eng["aggr_loud_harsh"] = rms * zcr * 100
    eng["aggr_flux_onset"] = flux * onsets / 10
    eng["aggr_dense"] = flatness * 100 / (contrast_std_val + 1)
    eng["aggr_low_dyn"] = rms / (dyn_range + 1)  # compressed = low dynamic range per loudness

    # =====================================================
    # HAPPY — energy WITHOUT harshness + major key
    # =====================================================
    eng["happy_clean_energy"] = mfcc_m[0] / 200 * (1 - min(1, zcr * 10)) if mfcc_m[0] > 0 else 0
    eng["happy_bright_tonal"] = centroid / 3000 * (1 - min(1, flatness * 100))
    eng["happy_upbeat"] = beats * (tempo / 120) * mode  # major + rhythmic
    eng["happy_major_energy"] = mode * rms * beats * centroid / 3000
    eng["happy_singalong"] = vocal * beats * mode * (tempo / 120)

    # Chroma-based major interval detection
    eng["happy_major_3rd"] = sum(chroma[i] * chroma[(i+4) % 12] for i in range(12))
    eng["happy_minor_3rd"] = sum(chroma[i] * chroma[(i+3) % 12] for i in range(12))
    eng["happy_interval_ratio"] = eng["happy_major_3rd"] / (eng["happy_minor_3rd"] + 0.01)
    eng["happy_perfect_5th"] = sum(chroma[i] * chroma[(i+7) % 12] for i in range(12))

    eng["happy_tempo_sweet"] = max(0, 1 - abs(tempo - 120) / 60)  # peaks 100-140
    eng["happy_tempo_sweet_beat"] = eng["happy_tempo_sweet"] * beats
    eng["happy_bounce"] = beats * onsets / (zcr * 100 + 1)
    eng["happy_tonnetz_major"] = (tonnetz[0] + tonnetz[2] + tonnetz[4]) if len(tonnetz) >= 5 else 0

    # =====================================================
    # SAD — minor + dark + slow + quiet
    # =====================================================
    eng["sad_minor_dark"] = (1 - mode) * max(0, 1 - centroid / 2500) * max(0, 1 - rms * 5)
    eng["sad_minor_slow"] = (1 - mode) * max(0, 1 - tempo / 120)
    eng["sad_minor_3rd_dom"] = eng["happy_minor_3rd"] - eng["happy_major_3rd"]
    eng["sad_quiet_dark_slow"] = max(0, 1-rms*5) * max(0, 1-centroid/3000) * max(0, 1-tempo/120)
    eng["sad_low_beats"] = (1 - beats) * max(0, 1 - rms * 5)
    eng["sad_tonnetz_minor"] = (tonnetz[1] + tonnetz[3] + tonnetz[5]) if len(tonnetz) >= 6 else 0

    # =====================================================
    # AROUSAL — rhythm + energy (beat/onset dominate)
    # =====================================================
    eng["ar_beat_onset"] = beats * onsets
    eng["ar_beat_onset_flux"] = beats * onsets * flux / 30
    eng["ar_total"] = rms * flux * onsets * beats
    eng["ar_rhythm_energy"] = (beats + onsets / 5) * rms * 10
    eng["ar_beat_sq"] = beats ** 2 * (tempo / 120)
    eng["ar_onset_sq"] = onsets ** 2 / 10
    eng["ar_dynamic"] = rms_var * 100 * onsets
    eng["ar_vocal_rhythm"] = vocal * beats * onsets  # vocal tracks can have high arousal
    eng["ar_spectral_mass"] = centroid * bandwidth * rms / 1e7

    # =====================================================
    # RELAXED — inverse arousal
    # =====================================================
    eng["rel_inv_arousal"] = 1 / (beats * onsets + 0.1)
    eng["rel_spectral_calm"] = 1 / (flux + 1) / (flux_std + 1)
    eng["rel_slow_smooth"] = max(0, 1-tempo/160) * (1-min(1, zcr*10)) * (1-min(1, flatness*50))
    eng["rel_no_attacks"] = 1 / (onsets + 0.5) / (flux_std + 1)
    eng["rel_sustained"] = max(0, 1 - rms_var * 10) * max(0, 1 - onsets / 5)
    eng["rel_warm_bass"] = max(0, contrast[0]) / 30 * max(0, 1-zcr*10) if contrast else 0

    # =====================================================
    # PARTY — loud + fast + bass + danceable
    # =====================================================
    eng["par_power"] = rms * beats * tempo / 120
    eng["par_electronic"] = flatness * 100 * beats * rms
    eng["par_loud_fast"] = rms * tempo / 100
    eng["par_bass_beat"] = max(0, -contrast[0]) * beats * rms if contrast else 0

    # =====================================================
    # ACOUSTIC
    # =====================================================
    eng["acou_rich_contrast"] = contrast_std_val * (1 - min(1, flatness * 100))
    eng["acou_organic"] = (1 - min(1, flatness * 100)) * contrast_mean / 20
    eng["acou_quiet_tonal"] = max(0, 1 - rms * 3) * (1 - min(1, flatness * 50))
    eng["acou_natural_dyn"] = dyn_range / 20 * (1 - min(1, flatness * 100))

    # =====================================================
    # DANCEABLE
    # =====================================================
    eng["dan_groove"] = beats * tempo / 120 * rms
    eng["dan_regular"] = beats * max(0, 1 - abs(tempo - 120) / 60)
    eng["dan_beat_sq"] = beats ** 2 * rms * 10
    eng["dan_consistency"] = beats * max(0, 1 - rms_var * 5)

    # =====================================================
    # TONAL / ATONAL
    # =====================================================
    eng["ton_clarity"] = 1 - min(1, flatness * 100)
    eng["ton_chroma_peak"] = float(np.max(chroma) / (np.mean(chroma) + 0.01)) if chroma else 0
    eng["ton_contrast_energy"] = contrast_mean * rms
    eng["aton_noise"] = flatness * zcr * 1000
    eng["aton_chroma_flat"] = 1 / (float(np.std(chroma)) + 0.01) if chroma else 0

    # =====================================================
    # VALENCE
    # =====================================================
    eng["val_major"] = mode * centroid / 3000 * beats
    eng["val_happy_e"] = mode * rms * beats * (tempo / 120)
    eng["val_not_sad"] = 1 - eng["sad_quiet_dark_slow"]
    eng["val_interval"] = eng["happy_interval_ratio"]

    # =====================================================
    # CROSS-DIMENSION 2nd order
    # =====================================================
    eng["x_energy_bright"] = rms * centroid / 3000
    eng["x_rhythm_tone"] = beats * (1 - min(1, flatness * 100))
    eng["x_speed_noise"] = tempo * zcr / 10
    eng["x_vocal_energy"] = vocal * rms
    eng["x_onset_width"] = onsets * bandwidth / 2000
    eng["x_dyn_flux"] = dyn_range * flux / 100
    eng["x_mode_energy"] = mode * rms * beats  # major + energy
    eng["x_mode_brightness"] = mode * centroid / 3000  # major + bright

    return eng


def run(data):
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X_rows = []
    y_dict = {t: [] for t in CLS_KEYS}

    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        eng = engineer_v4(scalars, vectors)
        feat.update(eng)
        X_rows.append(feat)
        for t in CLS_KEYS:
            y_dict[t].append(cls.get(t, 0))

    feat_names = sorted(X_rows[0].keys())
    X = np.array([[row.get(f, 0) for f in feat_names] for row in X_rows], dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_eng = len(engineer_v4(data[0][0], data[0][1]))
    n_raw = len(flatten_features(data[0][0], data[0][1]))
    print(f"{'='*95}")
    print(f"ROUND 4 — {X.shape[1]} total features ({n_raw} raw + {n_eng} engineered)")
    print(f"NOTE: Previous rounds had WRONG feature names — mfcc_mean→mfcc_m, etc.")
    print(f"{'='*95}")

    prev = {
        "happy": 0.307, "sad": 0.578, "relaxed": 0.489, "aggressive": 0.215,
        "party": 0.421, "acoustic": 0.519, "danceable": 0.581,
        "instrumental": -0.029, "vocal": -0.030, "tonal": 0.384, "atonal": 0.374,
        "arousal": 0.364, "valence": 0.578,
    }

    print(f"\n{'TARGET':<14} | {'R3 best':>9} | {'R4 GBR':>9} | {'R4 RF':>9} | {'R4 best':>9} | {'Δ R3→R4':>9} | Status")
    print("-" * 95)

    results = {}
    for target in CLS_KEYS:
        y = np.array(y_dict[target], dtype=float)
        if np.std(y) < 1e-6:
            continue

        gbr_scores = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=10, random_state=42
            ),
            X_scaled, y, cv=5, scoring="r2"
        )

        rf_scores = cross_val_score(
            RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            ),
            X_scaled, y, cv=5, scoring="r2"
        )

        gbr_r2 = gbr_scores.mean()
        rf_r2 = rf_scores.mean()
        best = max(gbr_r2, rf_r2)
        p = prev.get(target, 0)
        delta = best - p

        results[target] = {"gbr": gbr_r2, "rf": rf_r2, "best": best}

        if best > 0.5:
            status = "★ GOOD"
        elif best > 0.3:
            status = "● usable"
        elif best > 0.1:
            status = "○ weak"
        else:
            status = "✗ broken"

        if delta > 0.05:
            status += " ▲▲"
        elif delta > 0.02:
            status += " ▲"
        elif delta < -0.02:
            status += " ▼"

        print(f"{target:<14} | {p:>9.3f} | {gbr_r2:>9.3f} | {rf_r2:>9.3f} | {best:>9.3f} | {delta:>+9.3f} | {status}")

    # Feature importance for key targets
    print(f"\n{'='*95}")
    print("KEY FEATURE IMPORTANCES (GBR)")
    print(f"{'='*95}")

    for target in ["happy", "aggressive", "instrumental", "vocal", "arousal"]:
        y = np.array(y_dict[target], dtype=float)
        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        gbr.fit(X_scaled, y)
        imp = sorted(zip(feat_names, gbr.feature_importances_), key=lambda x: -x[1])
        top = imp[:10]
        print(f"\n{target}:")
        for name, score in top:
            bar = "█" * int(score * 200)
            print(f"  {name:<40} {score:.4f} {bar}")

    # --- Hand-crafted vocal score with CORRECT features ---
    print(f"\n{'='*95}")
    print("HAND-CRAFTED VOCAL SCORE (corrected)")
    print(f"{'='*95}")

    # Gather corpus stats for MFCC std bands 3-8
    mfcc_s_high_vals = []
    centroid_std_vals = []
    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        mfcc_s = [feat.get(f"mfcc_s_{i}", 0) for i in range(13)]
        mfcc_s_high_vals.append(sum(mfcc_s[3:9]))
        centroid_std_vals.append(feat.get("centroid_std", 0))

    ms_high_mean = float(np.mean(mfcc_s_high_vals))
    ms_high_std = float(np.std(mfcc_s_high_vals))
    cs_mean = float(np.mean(centroid_std_vals))
    cs_std = float(np.std(centroid_std_vals))

    print(f"  mfcc_s_high: mean={ms_high_mean:.4f}, std={ms_high_std:.4f}")
    print(f"  centroid_std: mean={cs_mean:.4f}, std={cs_std:.4f}")

    # Compute scores
    true_vocal = [cls.get("vocal", 0) for _, _, cls in data]
    true_instr = [cls.get("instrumental", 0) for _, _, cls in data]

    hand_scores = []
    for i in range(len(data)):
        z_high = (mfcc_s_high_vals[i] - ms_high_mean) / (ms_high_std + 1e-6)
        z_cent = (centroid_std_vals[i] - cs_mean) / (cs_std + 1e-6)
        score = z_high * 0.7 + z_cent * 0.3
        hand_scores.append(float(1 / (1 + np.exp(-score))))

    rho_v, _ = sp_stats.spearmanr(hand_scores, true_vocal)
    rho_i, _ = sp_stats.spearmanr(hand_scores, true_instr)
    print(f"\n  Hand-crafted vocal score:")
    print(f"    vs vocal:        rho={rho_v:+.3f}")
    print(f"    vs instrumental: rho={rho_i:+.3f}")

    # Summary
    print(f"\n{'='*95}")
    print("SUMMARY — can we replace MusiCNN?")
    print(f"{'='*95}")
    for target in CLS_KEYS:
        if target in results:
            r2 = results[target]["best"]
            if r2 > 0.5:
                verdict = "YES — regression is viable"
            elif r2 > 0.3:
                verdict = "MAYBE — usable with caveats"
            elif r2 > 0.1:
                verdict = "NEEDS WORK — weak signal"
            else:
                verdict = "NO — consider brightness.py approach or drop"
            print(f"  {target:<14} R²={r2:.3f}  {verdict}")


if __name__ == "__main__":
    data = load_data()
    run(data)
