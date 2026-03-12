# Advanced Librosa Features for Music Classification

**Date:** 2026-03-12
**Purpose:** Document novel librosa features to improve weak classifiers (sad, acoustic, danceable, arousal, valence)
**Sources:** MIR research papers, MCT Blog, Nature Scientific Reports, IEEE, librosa docs

---

## Current Accuracy Baseline

| Classifier | Current CV | Target | Gap |
|------------|-----------|--------|-----|
| sad | 83.9% | 90%+ | ~6% |
| acoustic | 84.4% | 90%+ | ~6% |
| danceable | 85.1% | 90%+ | ~5% |
| arousal | R²=0.456 | R²>0.65 | ~0.2 |
| valence | R²=0.625 | R²>0.75 | ~0.13 |

---

## Feature 1: Harmonic-Percussive Source Separation (HPSS)

### Concept

Decompose audio into harmonic (sustained tones, melody, chords) and percussive
(transient, drums, attacks) components. Features extracted from each component
separately carry more signal than from the mix.

### Librosa Implementation

```python
y_harm, y_perc = librosa.effects.hpss(y)

rms_harm = np.mean(librosa.feature.rms(y=y_harm)[0])
rms_perc = np.mean(librosa.feature.rms(y=y_perc)[0])
harm_perc_ratio = rms_harm / (rms_perc + 1e-8)
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `harm_perc_ratio` | rms_harm / rms_perc | Harmonic vs percussive balance |
| `harm_energy` | mean(rms(y_harm)) | Pure harmonic loudness |
| `perc_energy` | mean(rms(y_perc)) | Pure percussive loudness |
| `harm_fraction` | harm / (harm + perc) | Harmonic dominance 0-1 |
| `chroma_harm` | chroma_cqt(y_harm) | Clean tonal content (no drum bleed) |
| `mfcc_perc` | mfcc(y_perc) | Percussive timbre profile |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **acoustic** | Acoustic instruments = high harmonic ratio. Electronic/synth = lower. Guitars, pianos, strings have strong sustained harmonics. | +4-6% |
| **danceable** | Dance music = strong percussive energy. Ambient = almost none. | +3-5% |
| **sad** | Sad music often more harmonic (sustained pads, strings). Percussive energy low. | +2-3% |
| **instrumental** | Vocals appear in harmonic component. High harm energy with specific MFCC shape = vocals present. | +1-2% |

### Compute Cost

- `librosa.effects.hpss()`: median filtering on spectrogram, O(n log n)
- Adds ~1-2s per track at sr=22050
- Memory: same as STFT (already computed)

---

## Feature 2: Tempogram Ratio / Rhythm Regularity

### Concept

Measure how strongly the rhythm energy aligns with regular metric subdivisions
(half notes, quarter notes, eighth notes). Danceable music has strong energy at
exact tempo multiples. Free-form jazz or ambient does not.

### Librosa Implementation

```python
oenv = librosa.onset.onset_strength(y=y, sr=sr)

# Tempogram: autocorrelation of onset envelope
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)

# Tempogram ratio: energy at metric subdivisions
# Ratios: [1, 2, 4, 1/3, 2/3, 3, 5, 8] — multiples of detected tempo
tempo_ratio = librosa.feature.tempogram_ratio(tg=tempogram, sr=sr)
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `tempo_ratio_half` | ratio at 2x tempo | Eighth note energy (groove) |
| `tempo_ratio_quarter` | ratio at 1x tempo | Quarter note energy (beat) |
| `tempo_ratio_third` | ratio at 3x tempo | Triplet feel |
| `beat_regularity` | max(tempogram) / mean(tempogram) | How dominant the main beat is |
| `rhythm_complexity` | entropy of tempo_ratio | Regular vs complex rhythm |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **danceable** | THE missing feature. Danceable = strong regular beat subdivisions. | +5-7% |
| **party** | Already at 94.6% but rhythm regularity would add confidence. | +1% |
| **arousal** | Steady driving rhythm = higher arousal. | R² +0.05 |

### Compute Cost

- `librosa.feature.tempogram()`: autocorrelation of onset envelope, fast
- `tempogram_ratio()`: just resampling the tempogram, negligible
- Adds <0.5s per track

---

## Feature 3: Predominant Local Pulse (PLP) Stability

### Concept

PLP finds the locally dominant pulse (beat) at each time frame. The stability
(low variance) of PLP across a track measures how consistent the beat is.
Dance music has very stable PLP. Rubato classical or free jazz does not.

### Librosa Implementation

```python
oenv = librosa.onset.onset_strength(y=y, sr=sr)
pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr)

plp_mean = np.mean(pulse)
plp_std = np.std(pulse)
plp_stability = plp_mean / (plp_std + 1e-8)  # high = stable beat
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `plp_mean` | mean(plp) | Overall pulse strength |
| `plp_std` | std(plp) | Pulse variation |
| `plp_stability` | mean / std | Beat consistency |
| `plp_entropy` | entropy(plp) | Rhythmic predictability |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **danceable** | Stable pulse = danceable. Complements tempogram ratio. | +2-3% |
| **relaxed** | Very stable low-energy pulse or no pulse = relaxed. | +1% |

### Compute Cost

- Negligible (reuses onset envelope already computed)

---

## Feature 4: Low Energy Rate

### Concept

Fraction of audio frames with below-average RMS energy. Captures the
distribution of energy over time — not just how loud, but how much silence
or quietness exists.

### Librosa Implementation

```python
rms = librosa.feature.rms(y=y)[0]
low_energy_rate = np.mean(rms < np.mean(rms))
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `low_energy_rate` | mean(rms < mean(rms)) | Fraction of quiet frames |
| `silence_ratio` | mean(rms < 0.01) | Near-silent frames |
| `energy_skew` | skew(rms) | Energy distribution shape |
| `energy_kurtosis` | kurtosis(rms) | Energy peakedness |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **sad** | Sad music has more quiet passages, pauses, space. High low_energy_rate. | +2-4% |
| **aggressive** | Aggressive = consistently loud. Low low_energy_rate. | +1% |
| **acoustic** | Acoustic music has more dynamic variation (high low_energy_rate). | +1-2% |

### Compute Cost

- Negligible (RMS already computed)

---

## Feature 5: Delta Features (Temporal Dynamics)

### Concept

First derivative (delta) and second derivative (delta-delta) of spectral
features capture how fast the sound changes over time. Sad music changes
slowly. Aggressive music changes rapidly.

### Librosa Implementation

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

mfcc_delta = librosa.feature.delta(mfcc)       # velocity
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  # acceleration

# Summary statistics
delta_mean = np.mean(np.abs(mfcc_delta), axis=1)   # 13 values
delta_std = np.std(mfcc_delta, axis=1)              # 13 values
delta_var = np.mean(delta_std)                      # single value: overall change rate
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `mfcc_delta_var` | mean(std(delta)) | Overall spectral change rate |
| `mfcc_delta_mean[0:3]` | mean(abs(delta[i])) | Per-coefficient change rate |
| `mfcc_delta2_var` | mean(std(delta2)) | Spectral acceleration |
| `chroma_delta_var` | mean(std(delta(chroma))) | Harmonic change rate |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **sad** | Slow spectral change = contemplative, melancholic. | +2-3% |
| **aggressive** | Fast spectral change = intensity, chaos. | +1% |
| **arousal** | Change rate directly correlates with perceived energy. | R² +0.05-0.10 |
| **valence** | Slow harmonic changes + minor key = negative valence. | R² +0.03-0.05 |

### Compute Cost

- `librosa.feature.delta()`: simple finite differences, negligible
- Uses MFCC matrix already computed

---

## Feature 6: Sub-band Energy Ratios

### Concept

Divide the spectrum into frequency bands (bass / mid / treble) and compute
energy ratios. More informative than centroid alone because it captures the
shape of energy distribution.

### Librosa Implementation

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
bass_mid_ratio = np.mean(bass / (mid + 1e-8))
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `bass_ratio` | bass_energy / total | Bass dominance (sub, kick) |
| `mid_ratio` | mid_energy / total | Mid presence (vocals, guitars) |
| `treble_ratio` | treble_energy / total | High frequency content |
| `bass_mid_ratio` | bass / mid | Bass vs mid balance |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **danceable** | Dance music = strong bass ratio. | +1-2% |
| **acoustic** | Acoustic = mid-heavy (instrument fundamentals). Low bass ratio. | +2-3% |
| **sad** | Sad music often has more mid/low energy, less treble. | +1% |

### Compute Cost

- Uses STFT already computed
- Negligible additional cost

---

## Feature 7: Modulation Spectrum

### Concept

Compute the FFT of temporal feature trajectories (e.g., spectral centroid
over time). This captures rhythmic modulation patterns — how features
oscillate at different rates.

### Librosa Implementation

```python
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
mod_spectrum = np.abs(np.fft.rfft(centroid))

mod_flatness = scipy.stats.gmean(mod_spectrum + 1e-8) / (np.mean(mod_spectrum) + 1e-8)
mod_crest = np.max(mod_spectrum) / (np.mean(mod_spectrum) + 1e-8)
mod_centroid = np.sum(np.arange(len(mod_spectrum)) * mod_spectrum) / (np.sum(mod_spectrum) + 1e-8)
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `mod_flatness` | gmean(mod) / mean(mod) | Regular vs irregular modulation |
| `mod_crest` | max(mod) / mean(mod) | Dominant modulation strength |
| `mod_centroid` | weighted mean of mod freqs | Average modulation rate |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **arousal** | High modulation rate = high arousal. Strong periodic modulation = driving. | R² +0.05 |
| **valence** | Modulation patterns encode musical "flow" and tension-release. | R² +0.03 |

### Compute Cost

- Single FFT of a 1D vector, negligible

---

## Feature 8: Spectral Higher-Order Moments

### Concept

Beyond centroid (1st moment) and bandwidth (2nd moment), skewness (3rd)
and kurtosis (4th) of the spectrum capture shape details.

### Librosa Implementation

```python
S = np.abs(librosa.stft(y))**2
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

# Per-frame spectral moments
centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

# Skewness: asymmetry of spectrum
# Kurtosis: peakiness of spectrum
from scipy.stats import skew, kurtosis
spec_skew = np.mean([skew(S[:, i]) for i in range(S.shape[1])])
spec_kurtosis = np.mean([kurtosis(S[:, i]) for i in range(S.shape[1])])
```

### Derived Features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `spectral_skew` | 3rd moment of spectrum | Harmonic asymmetry |
| `spectral_kurtosis` | 4th moment of spectrum | Energy concentration vs spread |

### Expected Impact

| Classifier | Why it helps | Expected gain |
|------------|-------------|---------------|
| **acoustic** | Acoustic instruments = high kurtosis (peaked harmonics). Synths = low. | +2-3% |
| **tonal** | Tonal music = high kurtosis, low skew. Atonal = flat, high skew. | +1-2% |

### Compute Cost

- Per-frame scipy.stats calls on spectrum columns
- ~1s per track

---

## Implementation Priority

Based on expected impact vs compute cost:

| Priority | Feature | Compute Cost | Classifiers Helped | Expected Total Gain |
|----------|---------|-------------|-------------------|-------------------|
| 1 | HPSS + harm_perc_ratio | ~1-2s | acoustic, danceable, sad | +3-6% each |
| 2 | Delta features | negligible | sad, arousal, valence | +2-3% / R²+0.05 |
| 3 | Low energy rate | negligible | sad, acoustic, aggressive | +2-4% |
| 4 | Sub-band energy ratios | negligible | danceable, acoustic, sad | +1-3% |
| 5 | Tempogram ratio + PLP | ~0.5s | danceable | +5-7% |
| 6 | Modulation spectrum | negligible | arousal, valence | R²+0.03-0.05 |
| 7 | Spectral moments | ~1s | acoustic, tonal | +2-3% |

### Total additional compute per track: ~3-4 seconds
### Current extraction time: ~5-6 seconds (librosa features at sr=22050)
### Projected new total: ~8-10 seconds per track

---

## References

- Schubert, E. (2004). "Modeling Perceived Emotion with Continuous Musical Features." Music Perception.
- Peeters, G. (2004). "A large set of audio features for sound description." IRCAM Technical Report.
- Lartillot & Toiviainen (2007). "MIR in Matlab." ISMIR.
- MCT Blog (2020). "Music Mood Classifier Using Spectral Features."
- Eerola, T. et al. (2009). "Prediction of Multidimensional Emotional Ratings in Music." ISMIR.
- Yang, Y-H. & Chen, H-H. (2012). "Machine Recognition of Music Emotion." ACM TIST.
- Nature Scientific Reports (2025). "Distinct Spectral/Temporal Drivers of Musical Emotion."
- librosa 0.10+ docs: tempogram_ratio, plp, hpss, delta.
