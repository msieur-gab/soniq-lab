# Librosa-Derived Music Classification — Complete Specification

**Date:** 2026-03-12
**Version:** 0.5 (proposed — replaces MusiCNN ONNX pipeline)
**Objective:** Derive all classification values from librosa features alone, no neural network

---

## Summary

13 classifiers derived from librosa audio features using logistic regression
(binary) or ridge regression (continuous). Calibrated against V1 EffNet/MusiCNN
ground truth on 1081 tracks.

| Classifier | Type | Output | CV Accuracy | Status |
|------------|------|--------|------------|--------|
| party | binary | 0-1 | 94.6% | READY |
| aggressive | binary | 0-1 | 94.5% | READY |
| bright/dark | binary | pair sums to 1 | 93.0% | READY |
| brilliant/warm | formula | pair sums to 1 | n/a (centroid) | READY |
| instrumental | binary | 0-1 | 91.6% | READY |
| relaxed | binary | 0-1 | 91.4% | READY |
| happy | binary | 0-1 | 90.7% | READY |
| tonal | binary | 0-1 | 87.4% | NEEDS IMPROVEMENT |
| danceable | binary | 0-1 | 85.1% | NEEDS IMPROVEMENT |
| acoustic | binary | 0-1 | 84.4% | NEEDS IMPROVEMENT |
| sad | binary | 0-1 | 83.9% | NEEDS IMPROVEMENT |
| valence | regression | 1-10 | R²=0.625 | NEEDS IMPROVEMENT |
| arousal | regression | 1-10 | R²=0.456 | NEEDS IMPROVEMENT |

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

### Proposed New Extractions (to improve weak classifiers)

| # | Librosa Function | Provides | Target Classifiers | Expected Gain | Cost |
|---|-----------------|----------|-------------------|---------------|------|
| 15 | `librosa.effects.hpss()` | harm_perc_ratio, harm_energy, perc_energy | acoustic, danceable, sad | +4-6% | ~2s |
| 16 | `librosa.feature.delta(mfcc)` | mfcc_delta_var, mfcc_delta2_var | sad, arousal, valence | +2-3% / R²+0.1 | ~0s |
| 17 | RMS statistics | low_energy_rate, energy_skew, energy_kurtosis | sad, acoustic, aggressive | +2-4% | ~0s |
| 18 | Sub-band energy ratios | bass_ratio, mid_ratio, treble_ratio | acoustic, danceable | +2-3% | ~0s |
| 19 | `librosa.feature.tempogram()` | beat_regularity, rhythm_complexity | danceable | +5-7% | ~0.5s |
| 20 | `librosa.beat.plp()` | plp_stability | danceable | +2-3% | ~0s |
| 21 | Modulation spectrum | mod_flatness, mod_crest | arousal, valence | R²+0.05 | ~0s |
| 22 | Spectral moments | spectral_skew, spectral_kurtosis | acoustic, tonal | +2-3% | ~1s |

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

### SAD — 83.9% CV (needs improvement)

**Type:** Logistic regression, 39 features

**Top drivers:**
- (+) bandwidth, mfcc0, mfcc3, mfcc4, zcr → wide spectrum, specific timbre
- (-) flux, rms_mean, onset, rms_max, rolloff → low energy, quiet, slow
- (-) spectral_width, key → narrow effective range, key preference
- (+) contrast4, contrast1 → mid-high harmonic presence

**Missing features that could help:**
- `low_energy_rate` — fraction of quiet frames (sad = more silence)
- `mfcc_delta_var` — slow spectral change = contemplative
- `harm_perc_ratio` — sad tends more harmonic

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

### ACOUSTIC — 84.4% CV (needs improvement)

**Type:** Logistic regression, 52 features

**Top drivers:**
- (+) bandwidth, mfcc0, tempo → wide, warm, steady
- (-) zcr, rolloff, flux, onset, rms_mean → not harsh, not loud, not attacking
- (-) spectral_width, rms_x_flux, brightness_proxy → not bright, not energetic
- (+) centroid_x_flatness, vocal_x_flux → tonal detail interaction

**Missing features that could help:**
- `harm_perc_ratio` — acoustic instruments = high harmonic ratio (THE key feature)
- `spectral_kurtosis` — acoustic = peaked harmonics
- `bass_ratio`, `mid_ratio` — acoustic = mid-heavy

**Librosa inputs:** bandwidth, zcr, rolloff, flux, rolloff_std, bandwidth_std, onset, rms_var, rms_mean, vocal, rms_max, tempo, dyn_range, centroid, mfcc[0,4,6,8,9,10], mfcc_s[0,3,5,8,12], contrast[1-5], chroma[0,1,3,6,7,10], tonnetz[2,3], + derived features

---

### DANCEABLE — 85.1% CV (needs improvement)

**Type:** Logistic regression, 45 features

**Top drivers:**
- (-) bandwidth → narrow spectrum focus
- (+) centroid_std, rms_max, flux_std → dynamic, punchy
- (+) onset, rms_mean, rolloff → loud, attacking
- (-) mfcc0, mfcc3, mfcc4 → specific spectral absence
- (+) chroma_std, rms_x_flux → harmonic variety, energy-flux interaction

**Missing features that could help:**
- `beat_regularity` — from tempogram, THE missing feature for dance
- `plp_stability` — steady pulse = danceable
- `perc_energy` — percussive component from HPSS
- `bass_ratio` — dance = bass-heavy

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

### TONAL — 87.4% CV (needs improvement)

**Type:** Logistic regression, 49 features

**Top drivers:**
- (-) flux → low spectral change (stable tonality)
- (+) rms_max, flatness, rms_mean → present signal
- (+) rms_x_flux → energy-flux interaction
- (-) vocal_x_flux → vocal instability hurts tonality
- (+) chroma_std, chroma[7,8,10,11] → pitch class richness
- (-) chroma0, chroma9 → specific pitch avoidance

**Missing features that could help:**
- `spectral_kurtosis` — tonal = peaked spectrum
- `harm_perc_ratio` — tonal = high harmonic content

**Librosa inputs:** flux, rms_max, flatness, rms_mean, flux_std, vocal, rolloff_std, rms_var, centroid_std, zcr, beat, rolloff, bandwidth, dyn_range, bandwidth_std, mfcc[0,1,2,4,7,9,11], mfcc_s[0,1,7,12], contrast[3,6], chroma[0,7,8,9,10,11], tonnetz[0,2,5], + derived features

---

### AROUSAL — R²=0.456 (needs improvement)

**Type:** Ridge regression, continuous 1-10 scale
**Current features:** Only 3 features pass threshold (flux, bandwidth, rms_x_flux)

**Top drivers:**
- (+) flux → spectral energy change
- (-) bandwidth → spectral width
- (-) rms_x_flux → energy interaction

**Missing features that could help:**
- `mfcc_delta_var` — rate of spectral change = energy
- `mod_crest` — modulation regularity
- `low_energy_rate` — fewer quiet moments = higher arousal
- `perc_energy` — percussive energy from HPSS
- `tempo` — faster = higher arousal (currently not selected but should help with new features)

**Librosa inputs (current):** flux, bandwidth, rms_mean

---

### VALENCE — R²=0.625 (needs improvement)

**Type:** Ridge regression, continuous 1-10 scale
**Current features:** Only 4 features pass threshold

**Top drivers:**
- (+) flux, spectral_width → spectral energy and range
- (-) bandwidth, rms_x_flux → interactions

**Missing features that could help:**
- `mfcc_delta_var` — temporal dynamics
- `mod_flatness` — modulation patterns
- `harm_perc_ratio` — harmonic balance
- `key`, `mode` — major/minor (should correlate with valence)
- `chroma` features from harmonic component via HPSS

**Librosa inputs (current):** flux, bandwidth, rms_mean

---

## Proposed New Librosa Extractions

To push weak classifiers above 90%, these new features should be added to
`py/librosa_features.py`:

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

## Compute Budget

| Stage | Current | With New Features |
|-------|---------|------------------|
| Librosa feature extraction | ~5-6s | ~8-9s (+HPSS, tempogram, moments) |
| Classification (11 logistic + 2 ridge) | ~0s | ~0s (just multiply-add) |
| MusiCNN ONNX inference | ~15-20s | **REMOVED** |
| **Total per track** | **~22s** | **~9s** |

**Net result: 2.5x faster pipeline, no ONNX dependency, no model downloads.**

---

## Implementation Steps

1. Add new feature extractions to `py/librosa_features.py` (HPSS, delta, etc.)
2. Re-run extraction on 1081 tracks to populate soniq.db with new features
3. Re-calibrate weak classifiers (sad, acoustic, danceable, tonal, arousal, valence)
4. Create `py/classifiers.py` with all logistic/ridge weights
5. Update `py/tags.py` to use classifiers instead of MusiCNN
6. Validate on key tracks (Plastikman, Daft Punk, etc.)
7. Remove ONNX dependency from pipeline

---

## References

- Schubert & Wolfe (2006). Timbral brightness and spectral centroid. Acta Acustica.
- Peeters et al. (2011). The Timbre Toolbox. JASA 130.
- Eerola et al. (2009). Prediction of multidimensional emotional ratings. ISMIR.
- Yang & Chen (2012). Machine recognition of music emotion. ACM TIST.
- Nature Scientific Reports (2025). Spectral/temporal drivers of musical emotion.
- librosa 0.10+ docs: tempogram_ratio, plp, hpss, delta.
