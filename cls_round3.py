#!/usr/bin/env python3
"""Round 3 — targeted attack on weak cls dimensions.

Key insights from correlation analysis:
- instrumental/vocal: MFCC stds (spectral variability) are the ONLY signal
- aggressive: spectral contrast (negative = "wall of sound" compression)
- happy: mfcc_m_0 dominates = overall energy, need to isolate "happy energy"
- arousal: beat + onset dominate, need more rhythm features

Strategy: fewer, sharper features + try brightness.py-style hand-crafted formulas.
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


def engineer_v3(scalars, vectors):
    """Round 3 features — targeted at weak dimensions."""
    feat = flatten_features(scalars, vectors)
    eng = {}

    # --- Base features ---
    tempo = feat.get("tempo", 120)
    rms = feat.get("rms_mean", 0.1)
    rms_max = feat.get("rms_max", 0.2)
    rms_var = feat.get("rms_var", 0.01)
    centroid = feat.get("centroid_mean", 1500)
    centroid_std = feat.get("centroid_std", 500)
    rolloff = feat.get("rolloff_mean", 3000)
    bandwidth = feat.get("bandwidth_mean", 1500)
    bandwidth_std = feat.get("bandwidth_std", 300)
    flatness = feat.get("flatness_mean", 0.01)
    flux = feat.get("spectral_flux", 30)
    flux_std = feat.get("flux_std", 15)
    onsets = feat.get("onset_rate", 2)
    beats = feat.get("beat_strength", 0.3)
    zcr = feat.get("zcr_mean", 0.05)
    vocal = feat.get("vocal_probability", 0.5)

    mfcc_m = [feat.get(f"mfcc_mean_{i}", 0) for i in range(13)]
    mfcc_s = [feat.get(f"mfcc_std_{i}", 0) for i in range(13)]
    contrast = [feat.get(f"spectral_contrast_{i}", 0) for i in range(7)]
    chroma = [feat.get(f"chroma_mean_{i}", 0) for i in range(12)]
    tonnetz = [feat.get(f"tonnetz_{i}", 0) for i in range(6)]

    # =====================================================
    # INSTRUMENTAL / VOCAL — the hardest problem
    # Signal: MFCC stds (spectral variability over time)
    # Vocals → more spectral variation → higher MFCC stds
    # =====================================================

    # MFCC std aggregates — the core signal
    mfcc_std_sum = sum(mfcc_s)
    mfcc_std_high = sum(mfcc_s[3:9])  # mid-high MFCC bands (most discriminative)
    mfcc_std_low = sum(mfcc_s[0:3])   # low MFCC bands (less useful)
    eng["vocal_mfcc_std_sum"] = mfcc_std_sum
    eng["vocal_mfcc_std_high"] = mfcc_std_high
    eng["vocal_mfcc_std_ratio"] = mfcc_std_high / (mfcc_std_low + 0.01)  # high-to-low ratio
    eng["vocal_mfcc_std_mean"] = mfcc_std_sum / 13
    eng["vocal_mfcc_std_max"] = max(mfcc_s)
    eng["vocal_mfcc_std_geomean"] = float(np.exp(np.mean(np.log(np.array(mfcc_s) + 1e-6))))

    # Spectral variability over time
    eng["vocal_centroid_var"] = centroid_std / (centroid + 1)  # relative centroid movement
    eng["vocal_bandwidth_var"] = bandwidth_std / (bandwidth + 1)  # relative bandwidth movement
    eng["vocal_flux_var"] = flux_std / (flux + 1)  # spectral flux variability

    # Voice formant region (1-4 kHz centroid)
    eng["vocal_formant_zone"] = max(0, 1 - abs(centroid - 2500) / 2500)  # peaks at 2.5kHz
    eng["vocal_formant_std"] = eng["vocal_formant_zone"] * centroid_std / 500

    # Harmonic structure (vocals have strong harmonics)
    eng["vocal_contrast_mid"] = sum(contrast[2:5]) / 3  # mid-band contrast
    eng["vocal_tonnetz_activity"] = sum(abs(t) for t in tonnetz)  # tonal movement

    # Inverse features for instrumental
    eng["instr_mfcc_stable"] = 1 / (mfcc_std_sum + 1)  # spectral stability
    eng["instr_mfcc_high_stable"] = 1 / (mfcc_std_high + 1)
    eng["instr_centroid_stable"] = 1 / (centroid_std + 1)
    eng["instr_low_std_product"] = 1 / (mfcc_s[5] * mfcc_s[6] * mfcc_s[7] + 1e-6)

    # =====================================================
    # AGGRESSIVE — "wall of sound" detection
    # Signal: negative spectral contrast + energy
    # =====================================================

    # Spectral contrast features (low contrast = compressed/aggressive)
    contrast_mean = float(np.mean(contrast))
    contrast_std = float(np.std(contrast))
    eng["aggr_contrast_inv"] = 1 / (contrast_mean + 20)  # inverse mean contrast
    eng["aggr_contrast_low_bands"] = -sum(contrast[0:3]) / 3  # negated low-band contrast
    eng["aggr_contrast_flat"] = 1 / (contrast_std + 1)  # flat = uniformly compressed
    eng["aggr_contrast_mid_inv"] = -sum(contrast[2:5]) / 3  # inverted mid contrast
    eng["aggr_wall_of_sound"] = rms / (contrast_mean + 20)  # loud + compressed
    eng["aggr_crush"] = rms * rms_max / (contrast_std + 1)  # sustained loudness + flat contrast
    eng["aggr_dense_spectrum"] = flatness * 100 / (contrast_std + 1)  # noisy + compressed

    # Energy-based aggression
    eng["aggr_rms_squared"] = rms ** 2 * 100  # emphasize loud tracks
    eng["aggr_peak_ratio"] = rms_max / (rms + 0.01)  # dynamic range (low = compressed)
    eng["aggr_loud_harsh"] = rms * zcr * 100  # volume × harshness
    eng["aggr_flux_onset"] = flux * onsets / 10  # spectral attack rate

    # Distortion proxy: high RMS + high flatness + low contrast
    eng["aggr_distortion"] = rms * flatness * 1000 / (contrast_mean + 20)

    # =====================================================
    # HAPPY — isolate "happy energy" from "aggressive energy"
    # Happy = energy + brightness + rhythm + tonality (NOT harsh)
    # =====================================================

    # Happy = energetic but NOT harsh
    eng["happy_energy_clean"] = mfcc_m[0] / 200 * (1 - min(1, zcr * 10))
    eng["happy_bright_clean"] = centroid / 3000 * (1 - min(1, flatness * 100))
    eng["happy_rhythmic_tonal"] = beats * (1 - min(1, flatness * 100))
    eng["happy_upbeat"] = max(0, mfcc_m[0]) * beats * (tempo / 120) / 200

    # Major key indicators (weighted chroma)
    # C major: C E G → chroma 0, 4, 7
    # Major 3rd interval is the "happy" interval
    eng["happy_major_thirds"] = sum(chroma[i] * chroma[(i+4) % 12] for i in range(12))
    eng["happy_minor_thirds"] = sum(chroma[i] * chroma[(i+3) % 12] for i in range(12))
    eng["happy_major_over_minor"] = eng["happy_major_thirds"] / (eng["happy_minor_thirds"] + 0.01)

    # Tempo zone — happy clusters around 100-140 BPM
    eng["happy_tempo_zone"] = max(0, 1 - abs(tempo - 120) / 60) * beats
    eng["happy_tempo_zone_energy"] = eng["happy_tempo_zone"] * rms

    # Brightness + energy WITHOUT aggression
    eng["happy_bright_not_harsh"] = (centroid / 3000) * rms * contrast_mean / 20
    eng["happy_bounce_ratio"] = beats * onsets / (zcr * 100 + 1)  # rhythmic / noisy

    # Positive tonnetz (major intervals)
    eng["happy_tonnetz_major"] = tonnetz[0] + tonnetz[2] + tonnetz[4] if len(tonnetz) >= 5 else 0
    eng["happy_tonnetz_minor"] = tonnetz[1] + tonnetz[3] + tonnetz[5] if len(tonnetz) >= 6 else 0
    eng["happy_tonnetz_diff"] = eng["happy_tonnetz_major"] - eng["happy_tonnetz_minor"]

    # =====================================================
    # AROUSAL — rhythm + energy + spectral activity
    # =====================================================

    eng["arousal_beat_onset"] = beats * onsets  # core rhythm
    eng["arousal_beat_onset_flux"] = beats * onsets * flux / 30
    eng["arousal_total_energy"] = rms * flux * onsets * beats  # everything multiplied
    eng["arousal_spectral_activity"] = flux * flux_std * centroid / 1e6
    eng["arousal_rhythm_energy"] = (beats + onsets / 5) * rms * 10
    eng["arousal_no_vocal"] = (1 - vocal) * beats * onsets  # instrumental energy
    eng["arousal_vocal_energy"] = vocal * beats * onsets  # vocal energy
    eng["arousal_dynamic"] = rms_var * 100 * onsets  # dynamic variation
    eng["arousal_tempo_beats_sq"] = (tempo / 120) * beats ** 2

    # =====================================================
    # RELAXED — inverse arousal with nuance
    # =====================================================
    eng["relax_inverse_arousal"] = 1 / (beats * onsets + 0.1)
    eng["relax_spectral_calm"] = 1 / (flux + 1) * 1 / (flux_std + 1)
    eng["relax_slow_smooth"] = max(0, 1 - tempo / 160) * (1 - min(1, zcr * 10)) * (1 - min(1, flatness * 50))
    eng["relax_warm_tone"] = max(0, mfcc_m[0]) / 200 * max(0, contrast[0]) / 20  # warm bass + some energy
    eng["relax_no_attacks"] = 1 / (onsets + 0.5) * 1 / (flux_std + 1)
    eng["relax_sustained"] = (1 - rms_var * 10) * (1 - min(1, onsets / 5))  # steady + few onsets

    # =====================================================
    # SAD — dark + slow + minor + low energy
    # =====================================================
    eng["sad_dark_slow"] = max(0, 1 - centroid / 2500) * max(0, 1 - tempo / 120)
    eng["sad_minor_thirds"] = eng["happy_minor_thirds"] - eng["happy_major_thirds"]
    eng["sad_low_energy_dark"] = max(0, 1 - rms * 5) * max(0, 1 - centroid / 2500)
    eng["sad_tonnetz_minor"] = eng["happy_tonnetz_minor"] - eng["happy_tonnetz_major"]
    eng["sad_quiet_slow_dark"] = max(0, 1 - rms * 5) * max(0, 1 - tempo / 120) * max(0, 1 - centroid / 3000)

    # =====================================================
    # PARTY — loud + fast + danceable + not acoustic
    # =====================================================
    eng["party_power"] = rms * beats * tempo / 120
    eng["party_bass_energy"] = max(0, -contrast[0]) * rms if contrast else 0  # bass-heavy
    eng["party_electronic"] = flatness * 100 * beats * rms  # synthetic + rhythmic
    eng["party_loud_bright_fast"] = rms * centroid / 3000 * tempo / 120

    # =====================================================
    # ACOUSTIC vs ELECTRONIC texture
    # =====================================================
    eng["acoustic_rich_contrast"] = contrast_std * (1 - min(1, flatness * 100))
    eng["acoustic_organic"] = (1 - min(1, flatness * 100)) * contrast_mean / 20
    eng["acoustic_quiet_tonal"] = max(0, 1 - rms * 3) * (1 - min(1, flatness * 50))

    # =====================================================
    # TONAL / ATONAL
    # =====================================================
    eng["tonal_chroma_concentration"] = float(np.max(chroma) / (np.mean(chroma) + 0.01))
    eng["tonal_contrast_energy"] = contrast_mean * rms
    eng["atonal_flat_spectrum"] = flatness * 100 * zcr * 10

    # =====================================================
    # DANCEABLE — rhythm precision
    # =====================================================
    eng["dance_beat_tempo"] = beats * (tempo / 120)
    eng["dance_beat_rms"] = beats ** 2 * rms * 10
    eng["dance_groove_consistency"] = beats * (1 - rms_var * 5)  # steady beat
    eng["dance_bass_beat"] = max(0, -contrast[0]) * beats * rms if contrast else 0

    # =====================================================
    # VALENCE — reuse some happy/sad features
    # =====================================================
    eng["valence_happy_energy"] = eng["happy_upbeat"]
    eng["valence_not_sad"] = 1 - eng["sad_dark_slow"]
    eng["valence_major_interval"] = eng["happy_major_over_minor"]
    eng["valence_bright_rhythmic"] = centroid / 3000 * beats

    return eng


def run_regression(data):
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X_rows = []
    y_dict = {t: [] for t in CLS_KEYS}

    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        eng = engineer_v3(scalars, vectors)
        feat.update(eng)
        X_rows.append(feat)
        for t in CLS_KEYS:
            y_dict[t].append(cls.get(t, 0))

    feat_names = sorted(X_rows[0].keys())
    X = np.array([[row.get(f, 0) for f in feat_names] for row in X_rows], dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_eng = len(engineer_v3(data[0][0], data[0][1]))
    print(f"{'='*90}")
    print(f"ROUND 3 — {X.shape[1]} features ({n_eng} engineered)")
    print(f"{'='*90}")

    # Baselines from previous rounds
    baseline_r1 = {  # raw 69 features
        "happy": 0.209, "sad": 0.525, "relaxed": 0.35, "aggressive": 0.241,
        "party": 0.35, "acoustic": 0.40, "danceable": 0.530,
        "instrumental": -0.197, "vocal": -0.188, "tonal": 0.30, "atonal": 0.30,
        "arousal": 0.314, "valence": 0.562,
    }
    baseline_r2 = {  # round 2
        "happy": 0.284, "sad": 0.569, "relaxed": 0.477, "aggressive": 0.184,
        "party": 0.410, "acoustic": 0.512, "danceable": 0.581,
        "instrumental": -0.141, "vocal": -0.170, "tonal": 0.355, "atonal": 0.352,
        "arousal": 0.337, "valence": 0.570,
    }

    print(f"\n{'TARGET':<14} | {'R1 (raw)':>9} | {'R2':>9} | {'R3 GBR':>9} | {'R3 RF':>9} | {'Δ R2→R3':>9} | Notes")
    print("-" * 90)

    for target in CLS_KEYS:
        y = np.array(y_dict[target], dtype=float)
        if np.std(y) < 1e-6:
            continue

        # GBR with tuned params
        gbr_scores = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=10, random_state=42
            ),
            X_scaled, y, cv=5, scoring="r2"
        )

        # Random Forest as alternative (sometimes better for noisy targets)
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
        r1 = baseline_r1.get(target, 0)
        r2 = baseline_r2.get(target, 0)
        delta = best - r2

        note = ""
        if best > 0.5:
            note = "GOOD"
        elif best > 0.3:
            note = "usable"
        elif best > 0.1:
            note = "weak"
        else:
            note = "BROKEN"

        if delta > 0.05:
            note += " ++"
        elif delta > 0.02:
            note += " +"
        elif delta < -0.02:
            note += " --"

        print(f"{target:<14} | {r1:>9.3f} | {r2:>9.3f} | {gbr_r2:>9.3f} | {rf_r2:>9.3f} | {delta:>+9.3f} | {note}")

    # Feature importance for the 5 hardest targets
    print(f"\n{'='*90}")
    print("TOP FEATURES — hard targets (GBR importance)")
    print(f"{'='*90}")

    for target in ["happy", "aggressive", "instrumental", "vocal", "arousal"]:
        y = np.array(y_dict[target], dtype=float)
        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        gbr.fit(X_scaled, y)
        imp = sorted(zip(feat_names, gbr.feature_importances_), key=lambda x: -x[1])
        top = imp[:12]
        print(f"\n{target}:")
        for name, score in top:
            bar = "█" * int(score * 200)
            print(f"  {name:<40} {score:.4f} {bar}")

    # --- Brightness.py-style hand-crafted score for vocal/instrumental ---
    print(f"\n{'='*90}")
    print("HAND-CRAFTED VOCAL SCORE (brightness.py style)")
    print(f"{'='*90}")

    # Compute corpus stats for key vocal features
    vocal_features = []
    for scalars, vectors, cls in data:
        feat = flatten_features(scalars, vectors)
        mfcc_s = [feat.get(f"mfcc_std_{i}", 0) for i in range(13)]
        vf = {
            "mfcc_std_high": sum(mfcc_s[3:9]),
            "centroid_std": feat.get("centroid_std", 0),
            "mfcc_std_sum": sum(mfcc_s),
        }
        vocal_features.append(vf)

    # Compute corpus stats
    for key in ["mfcc_std_high", "centroid_std", "mfcc_std_sum"]:
        vals = [vf[key] for vf in vocal_features]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"  {key}: mean={mean:.4f}, std={std:.4f}")

    # Compute hand-crafted vocal score and test correlation
    true_vocal = [cls.get("vocal", 0) for _, _, cls in data]
    true_instr = [cls.get("instrumental", 0) for _, _, cls in data]

    # Try weighted z-score approach
    hand_scores = []
    for vf in vocal_features:
        # z-score each feature
        z_high = (vf["mfcc_std_high"] - np.mean([v["mfcc_std_high"] for v in vocal_features])) / (np.std([v["mfcc_std_high"] for v in vocal_features]) + 1e-6)
        z_cent = (vf["centroid_std"] - np.mean([v["centroid_std"] for v in vocal_features])) / (np.std([v["centroid_std"] for v in vocal_features]) + 1e-6)

        score = z_high * 0.7 + z_cent * 0.3
        hand_scores.append(float(1 / (1 + np.exp(-score))))

    rho_v, _ = sp_stats.spearmanr(hand_scores, true_vocal)
    rho_i, _ = sp_stats.spearmanr(hand_scores, true_instr)
    print(f"\n  Hand-crafted vocal score correlation:")
    print(f"    vs vocal:       rho={rho_v:+.3f}")
    print(f"    vs instrumental: rho={rho_i:+.3f}")


if __name__ == "__main__":
    data = load_data()
    run_regression(data)
