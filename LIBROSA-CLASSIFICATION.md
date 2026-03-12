# Librosa-Derived Music Classification — Complete Specification

**Date:** 2026-03-12
**Version:** 0.5 (validated — replaces MusiCNN ONNX pipeline)
**Objective:** Derive all classification values from librosa features alone, no neural network

---

## Summary

13 classifiers derived from 124 librosa audio features using logistic regression
(binary) or ridge/GBR regression (continuous). Calibrated against V1 EffNet/MusiCNN
ground truth on 682 tracks. Extracted via `libro_extract.py` → `libro-soniq.db`.

**v0.5 approximates MusiCNN/EffNet as well or better than v0.4 on every binary target and 10/12 R² targets.**
All accuracy numbers measure agreement with MusiCNN ground truth — not absolute classification quality.
ONNX/MusiCNN dependency can be dropped. See `~/working_memory/research/SONIQ_V05_RESULTS.md` for full analysis.

| Classifier | Type | Output | v0.4 Acc | v0.5 Acc | Status |
|------------|------|--------|----------|----------|--------|
| aggressive | binary | 0-1 | 98.4% | **99.1%** | READY |
| party | binary | 0-1 | 97.2% | **97.9%** | READY |
| bright/dark | binary | pair sums to 1 | 93.0% | **95.8%** | READY |
| brilliant/warm | formula | pair sums to 1 | n/a | n/a (centroid) | READY |
| instrumental | binary | 0-1 | 95.6% | **95.7%** | READY |
| relaxed | binary | 0-1 | 94.1% | **95.2%** | READY |
| happy | binary | 0-1 | 93.7% | **95.0%** | READY |
| danceable | binary | 0-1 | 84.9% | **86.1%** | READY |
| tonal | binary | 0-1 | 85.9% | **86.2%** | READY |
| sad | binary | 0-1 | 81.7% | **85.2%** | READY (near ceiling) |
| acoustic | binary | 0-1 | 82.1% | **84.9%** | READY |
| valence | regression | 1-10 | R²=0.606 | **R²=0.676** | GOOD |
| arousal | regression | 1-10 | R²=0.388 | **R²=0.571** | GOOD (+0.183) |

---

## Required Librosa Extractions

Every librosa function needed, what it provides, and which classifiers use it.

### Currently Extracted (in py/librosa_features.py)

| # | Librosa Function | Provides | Used By | Count |
|---|-----------------|----------|---------|-------|
| 1 | `librosa.onset.onset_strength()` | flux, flux_std, onset | ALL 11 + arousal, valence | 13 |
| 2 | `librosa.feature.spectral_bandwidth()` | bandwidth, bandwidth_std | ALL 11 + arousal, valence | 12 |
| 3 | `librosa.feature.spectral_centroid()` | centroid, centroid_std | 9 binary + brilliant/warm | 10 |
| 4 | `librosa.feature.spectral_rolloff()` | rolloff, rolloff_std | ALL 9 binary | 9 |
| 5 | `librosa.feature.mfcc(n_mfcc=13)` | mfcc[0-12] mean+std | ALL 9 binary | 9 |
| 6 | `librosa.feature.spectral_contrast(n_bands=6)` | contrast[0-6] | ALL 9 binary | 9 |
| 7 | `librosa.feature.chroma_cqt()` | chroma[0-11] | ALL 9 binary | 9 |
| 8 | `librosa.feature.tonnetz()` | tonnetz[0-5] | ALL 9 binary | 9 |
| 9 | `librosa.feature.spectral_flatness()` | flatness | 7 classifiers | 7 |
| 10 | `librosa.feature.zero_crossing_rate()` | zcr | 7 classifiers | 7 |
| 11 | `librosa.feature.rms()` | rms_mean, rms_max, rms_var, dyn_range | 7 classifiers | 7 |
| 12 | `librosa.beat.beat_track()` | tempo, beat | 8 classifiers | 8 |
| 13 | Key/mode detection (from chroma) | key, mode | 6 classifiers | 6 |
| 14 | Vocal proxy (contrast ratio) | vocal | 9 classifiers | 9 |

### New Extractions (added in v0.5 — validated)

All extracted via `libro_extract.py` into `libro-soniq.db`. 52 new features total.

| # | Librosa Function | Provides | Actual Impact | Cost |
|---|-----------------|----------|--------------|------|
| 15 | `librosa.effects.hpss()` | harm_energy, perc_energy, harm_perc_ratio, harm_fraction | **MVP**: perc_energy |rho|>0.4 on 8 targets | ~2s |
| 16 | `librosa.feature.delta(mfcc)` | 13 delta + 13 delta2 means + delta_var + delta2_var | arousal R²+0.183 | ~0s |
| 17 | RMS statistics | low_energy_rate, energy_skew, energy_kurtosis | **Dead weight** (max |rho|=0.15) | ~0s |
| 18 | Sub-band energy ratios | bass_ratio, mid_ratio, treble_ratio, bass_mid_ratio | treble_ratio party +0.58, relaxed -0.52 | ~0s |
| 19 | `librosa.feature.tempogram()` | beat_regularity, rhythm_complexity | rhythm_complexity arousal -0.51 | ~0.5s |
| 20 | `librosa.beat.plp()` | plp_mean, plp_stability | modest contribution | ~0s |
| 21 | Modulation spectrum | mod_flatness, mod_crest, mod_centroid | modest contribution | ~0s |
| 22 | Spectral moments | spectral_skew, spectral_kurtosis | modest contribution | ~0s |
| 23 | `librosa.onset.onset_detect()` | onset_rate | **MVP**: arousal +0.62, valence +0.54 | ~0s |
| 24 | Spectral entropy | spectral_entropy | modest contribution | ~0s |
| 25 | Spectral crest factor | spectral_crest | happy -0.50, relaxed +0.47 | ~0s |

---

## Derived Features (computed from raw extractions)

These are calculated from the raw librosa outputs above. No additional
librosa calls needed.

| Feature | Formula | Used By |
|---------|---------|---------|
| `centroid_norm` | centroid / 8000 | aggressive, danceable, party, acoustic, relaxed |
| `centroid_var` | centroid_std² | happy, sad, danceable, instrumental, party |
| `spectral_width` | rolloff - centroid | sad, acoustic, danceable, instrumental, tonal, valence |
| `centroid_x_flatness` | centroid * flatness | happy, sad, relaxed, acoustic, danceable, tonal |
| `brightness_proxy` | centroid / rolloff | happy, sad, relaxed, aggressive, acoustic, instrumental, tonal |
| `rms_range` | rms_max - rms_mean | sad, acoustic, party |
| `rms_x_flux` | rms_mean * flux | happy, acoustic, danceable, tonal, arousal, valence |
| `energy_density` | rms_mean * onset | sad, relaxed, aggressive, party, danceable, valence |
| `loudness_var` | rms_var / rms_mean | aggressive, tonal |
| `tempo_norm` | tempo / 200 | acoustic, party |
| `tempo_sq` | (tempo/200)² | acoustic, danceable, instrumental, party |
| `beat_x_tempo` | beat * tempo/200 | acoustic, tonal |
| `onset_x_tempo` | onset * tempo/200 | happy, sad, party, instrumental, tonal |
| `rhythm_energy` | beat * onset | happy, party, relaxed, instrumental, tonal |
| `mode_x_mfcc1` | mode * mfcc[1] | acoustic, aggressive |
| `vocal_x_mfcc1` | vocal * mfcc[1] | sad, aggressive, party |
| `vocal_x_flux` | vocal * flux | sad, acoustic, relaxed, tonal |
| `chroma_std` | std(chroma[0..11]) | happy, aggressive, danceable, instrumental, party, tonal |
| `chroma_max` | max(chroma[0..11]) | sad, aggressive, party, relaxed |
| `chroma_range` | max - min of chroma | aggressive, acoustic, instrumental, relaxed, tonal |
| `tonnetz_energy` | sqrt(sum(tonnetz²)) | sad, danceable, party, relaxed, tonal |
| `contrast_range` | contrast[6] - contrast[0] | (available, low usage) |
| `contrast_low` | mean(contrast[0:3]) | (available, low usage) |
| `contrast_high` | mean(contrast[4:7]) | (available, low usage) |
| `key_sin` | sin(2π * key / 12) | aggressive, relaxed |
| `key_cos` | cos(2π * key / 12) | aggressive, sad |

---

## Per-Classifier Specifications

### BRILLIANT / WARM (timbre color)

**File:** `py/timbre.py`
**Type:** Centroid z-score through sigmoid

```
z = (centroid - 1374.4) / 528.7
brilliant = sigmoid(z)
warm = 1 - brilliant
```

**Librosa inputs:** centroid

---

### BRIGHT / DARK (perceived emotional brightness)

**File:** `py/perceived_brightness.py`
**Type:** Logistic regression, 16 features, 93.0% CV

| Feature | Weight |
|---------|--------|
| contrast2 | +0.183279 |
| contrast3 | +0.262677 |
| contrast4 | -0.223765 |
| flux | -0.013672 |
| mfcc1 | +0.009763 |
| mfcc2 | -0.034074 |
| tonnetz0 | -4.922302 |
| tonnetz1 | -11.381657 |
| tonnetz2 | +4.873693 |
| tonnetz3 | +3.380292 |
| tonnetz4 | -10.426049 |
| chroma0 | -0.655038 |
| chroma6 | -1.770176 |
| chroma7 | +5.267907 |
| chroma8 | -5.212868 |
| chroma11 | -3.355041 |
| **bias** | **+0.735607** |

```
logit = bias + sum(weight * feature)
bright = sigmoid(logit)
dark = 1 - bright
```

**Librosa inputs:** contrast[2-4], flux, mfcc[1-2], tonnetz[0-4], chroma[0,6,7,8,11]

---

### HAPPY — 90.7% CV

**Type:** Logistic regression, 34 features

**Top drivers:**
- (+) flux, zcr, onset, beat, mfcc0, mfcc1 → energetic, loud
- (-) rolloff_std, vocal, brightness_proxy, rhythm_energy → not harsh, not heavy
- (-) chroma7, chroma3 → specific pitch class avoidance
- (+) tonnetz1, tonnetz2 → fifth/minor-third harmonic warmth

**Librosa inputs:** flux, zcr, onset, beat, rolloff_std, vocal, duration, mfcc[0,1,7,9,10], mfcc_s[0,3,7,10], contrast6, chroma[1,2,3,4,7,10], tonnetz[1,2,3,4], + derived features

---

### SAD — 85.2% CV (v0.5, near literature ceiling ~R²0.50-0.55)

**Type:** GBC top-20 features (GBC outperforms LR here due to nonlinear interactions)

**v0.5 result:** 81.7% → 85.2% (+3.5%). Best method: GBC with top-20 features.

**Key new features:** perc_energy (rho -0.61), harm_fraction (+0.51), onset_rate (-0.49)

**Note:** Research predicted low_energy_rate would help sad (+2-4%). Actual: near-zero signal (max |rho|=0.03). Dead weight.

**Librosa inputs:** bandwidth, flux, rms_mean, rms_max, onset, rolloff, flux_std, duration, key, zcr, rms_var, mfcc[0,3,4,8], mfcc_s[10,12], contrast[1,4,5], chroma[1,6,7,8,10], tonnetz[0], + derived features

---

### RELAXED — 91.4% CV

**Type:** Logistic regression, 45 features

**Top drivers:**
- (-) flux, zcr, rms_max, onset, rolloff → quiet, smooth, slow
- (+) bandwidth, centroid_std, dyn_range, beat → wide but dynamic
- (-) mfcc_s0, mfcc_s5 → low spectral variability
- (+) contrast2, contrast4 → harmonic richness
- (-) chroma8, chroma_range → tonal stability

**Librosa inputs:** flux, zcr, bandwidth, centroid_std, dyn_range, beat, rms_max, rolloff, centroid, onset, rms_mean, vocal, rolloff_std, mfcc[4,5,8,9], mfcc_s[0,1,5,7,9,10,12], contrast[2,4], chroma[4,6,7,8,10], tonnetz[1], + derived features

---

### AGGRESSIVE — 94.5% CV

**Type:** Logistic regression, 59 features

**Top drivers:**
- (+) centroid, flux, zcr, onset, rolloff → harsh, bright, loud
- (-) bandwidth, flatness, centroid_std → focused spectrum
- (+) mfcc1, mfcc_s6, mfcc_s2 → specific spectral envelope shape
- (-) contrast4, contrast3, contrast2 → less harmonic smoothness
- (-) tonnetz3 → less harmonic warmth
- (+) key, chroma9, chroma5 → pitch class preferences

**Librosa inputs:** centroid, flux, zcr, onset, bandwidth, flatness, rolloff, rolloff_std, centroid_std, key, mode, duration, mfcc[1,2,5,9,10,12], mfcc_s[0-9,12], contrast[1-6], chroma[0,1,4,5,8,9,11], tonnetz[0,3,4], + derived features

---

### PARTY — 94.6% CV

**Type:** Logistic regression, 48 features

**Top drivers:**
- (+) flux, tempo, centroid, rolloff → fast, bright, energetic
- (-) bandwidth, onset → narrow effective spectrum
- (+) rhythm_energy, rms_var → rhythmic energy variation
- (+) chroma8 → specific pitch energy
- (-) tonnetz_energy, tonnetz3 → less tonal complexity

**Librosa inputs:** flux, bandwidth, key, bandwidth_std, onset, dyn_range, tempo, centroid, vocal, rms_var, rolloff, duration, mfcc[1,4,5,6,9,10], mfcc_s[0,1,5,6,9,10], contrast[1,2,4,6], chroma[0,1,4,8,9], tonnetz[2,3], + derived features

---

### ACOUSTIC — 84.9% CV (v0.5)

**Type:** Logistic regression, top-50 features

**v0.5 result:** 82.1% → 84.9% (+2.8%). Ridge R² jumped from 0.497 → 0.603 (+0.106).

**Key new features:** harm_fraction, perc_energy (confirmed: acoustic = high harmonic ratio)

**Note:** Sub-band boundary optimization (research suggests [20-150], [150-800], [800-4000], [4000-20000] Hz) could push further.

**Librosa inputs:** bandwidth, zcr, rolloff, flux, rolloff_std, bandwidth_std, onset, rms_var, rms_mean, vocal, rms_max, tempo, dyn_range, centroid, mfcc[0,4,6,8,9,10], mfcc_s[0,3,5,8,12], contrast[1-5], chroma[0,1,3,6,7,10], tonnetz[2,3], + derived features

---

### DANCEABLE — 86.1% CV (v0.5)

**Type:** Logistic regression, top-40 features

**v0.5 result:** 84.9% → 86.1% (+1.2%). Ridge R² from 0.615 → 0.635.

**Key new features:** perc_energy (+0.72 rho — strongest single correlation in the dataset), harm_fraction (-0.57), onset_rate (+0.39)

**Librosa inputs:** bandwidth, centroid_std, rms_max, flux_std, rms_mean, onset, rolloff, centroid, duration, zcr, vocal, mfcc[0,2,3,4,7], mfcc_s[1,5,7,11,12], contrast[1-5], chroma[0,2,3,4,8,9,11], tonnetz[0,2], + derived features

---

### INSTRUMENTAL — 91.6% CV

**Type:** Logistic regression, 35 features

**Top drivers:**
- (-) centroid_std, flux, bandwidth → stable, focused spectrum (no vocal variation)
- (+) rms_mean, rms_max, vocal, dyn_range → uses vocal proxy inversely
- (+) contrast3, mfcc8, mfcc12 → specific timbre signature
- (+) chroma_std, centroid_var → spectral variety without vocal instability

**Librosa inputs:** centroid_std, flux, bandwidth, rms_mean, vocal, rms_max, duration, key, dyn_range, rolloff_std, rolloff, mfcc[0,8,11,12], mfcc_s[0,1,5,7], contrast3, chroma[3,4,6,7,11], tonnetz[0,5], + derived features

---

### TONAL — 86.2% CV (v0.5)

**Type:** Logistic regression, top-40 features

**v0.5 result:** 85.9% → 86.2% (+0.3%). GBR R² from 0.384 → 0.467 (+0.083).

**Key new features:** perc_energy (+0.58 rho), harm_fraction (-0.41)

**Librosa inputs:** flux, rms_max, flatness, rms_mean, flux_std, vocal, rolloff_std, rms_var, centroid_std, zcr, beat, rolloff, bandwidth, dyn_range, bandwidth_std, mfcc[0,1,2,4,7,9,11], mfcc_s[0,1,7,12], contrast[3,6], chroma[0,7,8,9,10,11], tonnetz[0,2,5], + derived features

---

### AROUSAL — R²=0.571 (v0.5, biggest improvement)

**Type:** Ridge regression, continuous 1-10 scale

**v0.5 result:** R² 0.388 → 0.571 (+0.183). The single largest R² improvement across all targets.

**Key new features:** onset_rate (+0.62 rho — "the arousal king"), perc_energy (+0.61), harm_fraction (-0.59), rhythm_complexity (-0.51)

**Note:** Literature ceiling for pre-CNN arousal is ~R²0.55-0.65. We're inside it.

---

### VALENCE — R²=0.676 (v0.5)

**Type:** Ridge regression, continuous 1-10 scale

**v0.5 result:** R² 0.606 → 0.676 (+0.070).

**Key new features:** perc_energy (+0.70), harm_fraction (-0.65), onset_rate (+0.54)

---

## New Librosa Extractions (implemented in v0.5)

All features below are implemented in `libro_extract.py` and validated in `libro-soniq.db`.
Reference implementations for integration into `py/librosa_features.py`:

### Priority 1: HPSS (Harmonic-Percussive Separation) — ~2s/track

```python
y_harm, y_perc = librosa.effects.hpss(y)
rms_harm = np.mean(librosa.feature.rms(y=y_harm)[0])
rms_perc = np.mean(librosa.feature.rms(y=y_perc)[0])
harm_perc_ratio = rms_harm / (rms_perc + 1e-8)
harm_fraction = rms_harm / (rms_harm + rms_perc + 1e-8)
```

**Targets:** acoustic (+6%), danceable (+4%), sad (+3%), tonal (+2%)

### Priority 2: MFCC Delta features — ~0s (uses existing MFCC matrix)

```python
mfcc_delta = librosa.feature.delta(mfcc_matrix)
mfcc_delta2 = librosa.feature.delta(mfcc_matrix, order=2)
mfcc_delta_var = np.mean(np.std(mfcc_delta, axis=1))
mfcc_delta2_var = np.mean(np.std(mfcc_delta2, axis=1))
```

**Targets:** sad (+3%), arousal (R²+0.10), valence (R²+0.05)

### Priority 3: RMS Statistics — ~0s (uses existing RMS vector)

```python
rms = librosa.feature.rms(y=y)[0]
low_energy_rate = np.mean(rms < np.mean(rms))
energy_skew = scipy.stats.skew(rms)
energy_kurtosis = scipy.stats.kurtosis(rms)
```

**Targets:** sad (+3%), acoustic (+2%), arousal (R²+0.05)

### Priority 4: Sub-band Energy Ratios — ~0s (uses existing STFT)

```python
S = np.abs(librosa.stft(y))**2
freqs = librosa.fft_frequencies(sr=sr)
bass = S[freqs < 300].sum(axis=0)
mid = S[(freqs >= 300) & (freqs < 2000)].sum(axis=0)
treble = S[freqs >= 2000].sum(axis=0)
total = bass + mid + treble + 1e-8
bass_ratio = np.mean(bass / total)
mid_ratio = np.mean(mid / total)
treble_ratio = np.mean(treble / total)
```

**Targets:** acoustic (+3%), danceable (+2%), sad (+1%)

### Priority 5: Tempogram / Rhythm Regularity — ~0.5s/track

```python
oenv = librosa.onset.onset_strength(y=y, sr=sr)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
beat_regularity = np.max(tempogram.mean(axis=1)) / (np.mean(tempogram.mean(axis=1)) + 1e-8)

pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr)
plp_stability = np.mean(pulse) / (np.std(pulse) + 1e-8)
```

**Targets:** danceable (+5%), party (+1%)

### Priority 6: Modulation Spectrum — ~0s

```python
centroid_trace = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
mod_spectrum = np.abs(np.fft.rfft(centroid_trace))
mod_flatness = scipy.stats.gmean(mod_spectrum + 1e-8) / (np.mean(mod_spectrum) + 1e-8)
mod_crest = np.max(mod_spectrum) / (np.mean(mod_spectrum) + 1e-8)
```

**Targets:** arousal (R²+0.05), valence (R²+0.03)

### Priority 7: Spectral Higher-Order Moments — ~1s/track

```python
S = np.abs(librosa.stft(y))**2
spectral_skew = np.mean([scipy.stats.skew(S[:, i]) for i in range(S.shape[1])])
spectral_kurtosis = np.mean([scipy.stats.kurtosis(S[:, i]) for i in range(S.shape[1])])
```

**Targets:** acoustic (+2%), tonal (+2%)

---

## Tag Schema (cls section — complete)

```json
{
  "cls": {
    "happy": 0.82,
    "sad": 0.18,
    "relaxed": 0.45,
    "aggressive": 0.12,
    "party": 0.67,
    "acoustic": 0.30,
    "danceable": 0.78,
    "instrumental": 0.95,
    "tonal": 0.88,
    "arousal": 4.7,
    "valence": 5.2,
    "brilliant": 0.85,
    "warm": 0.15,
    "bright": 0.72,
    "dark": 0.28,
    "genre": "electronic"
  }
}
```

---

## Compute Budget (validated)

| Stage | v0.2 (TF, MusiCNN+EffNet) | v0.4 (ONNX hybrid) | v0.5 (librosa-only) |
|-------|---------------------------|---------------------|---------------------|
| MusiCNN backbone | ~20-30s | ~15-20s (ONNX) | **REMOVED** |
| EffNet backbone | ~20-30s | **REMOVED** | **REMOVED** |
| Librosa feature extraction | — | ~5-6s | ~4.4s (124 features) |
| Classification heads | ~0s | ~0s | ~0s (just multiply-add) |
| **Total per track** | **~60-77s** | **~22s** | **~4.4s** |

**Net result: ~7x faster than ONNX hybrid, ~15x faster than original TF pipeline. No model dependencies.**

---

## Implementation Steps

1. ~~Add new feature extractions~~ → Done in `libro_extract.py` (v0.5)
2. ~~Re-run extraction~~ → Done: 682 tracks in `libro-soniq.db`
3. ~~Re-calibrate all classifiers~~ → Done: all improve or match
4. Create `py/classifiers.py` with all logistic/ridge weights
5. Update `py/tags.py` to use classifiers instead of MusiCNN
6. Validate on key tracks (Plastikman, Daft Punk, etc.)
7. Remove ONNX dependency from pipeline
8. Consider pYIN for instrumental/vocal (R² ~0.05, binary 95.7% acceptable)

---

## References

- Schubert & Wolfe (2006). Timbral brightness and spectral centroid. Acta Acustica.
- Peeters et al. (2011). The Timbre Toolbox. JASA 130.
- Eerola et al. (2009). Prediction of multidimensional emotional ratings. ISMIR.
- Yang & Chen (2012). Machine recognition of music emotion. ACM TIST.
- Nature Scientific Reports (2025). Spectral/temporal drivers of musical emotion.
- librosa 0.10+ docs: tempogram_ratio, plp, hpss, delta.
