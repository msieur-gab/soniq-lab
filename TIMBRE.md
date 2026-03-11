# Soniq Lab — Timbre Derivation & Validation

Research and validation by **Gab** and **Claude Code** (Anthropic CLI, Opus 4.6).
March 2026.

---

## The Problem

EffNet's timbre classifier (bright/dark) is useless on this library.
682 tracks processed: 93% score between 0.45–0.55. Zero tracks above 0.55
bright. The model can't distinguish acoustic kora from electronic synths.

EffNet costs 10s/track (38% of pipeline time) for data that carries no signal.

---

## The Solution: Librosa Timbre Vector

Replace EffNet's binary classifier with a multi-dimensional timbre vector
extracted from librosa. Validated against Essentia (the same engine
AcousticBrainz used) and MusicBrainz metadata.

---

## How Timbre Is Derived

### What Timbre Actually Measures

Timbre is *where* energy sits in the frequency spectrum, *how spread* it is,
*how it moves* over time, and *what shape* the spectral envelope has. It's
what makes a kora sound different from a piano playing the same note.

One number (spectral centroid) captures only the first dimension. A proper
timbre representation needs all four.

### The Feature Vector (35 dimensions)

For each track, librosa computes per-frame spectral features across the
full duration, then aggregates as mean + standard deviation:

```
timbre_vector = [
  centroid_mean, centroid_std,       # brightness + its variability
  rolloff_mean, rolloff_std,         # where 85% of energy lives
  bandwidth_mean, bandwidth_std,     # tonal spread (narrow=pure, wide=noisy)
  flatness_mean,                     # tone vs noise ratio
  flux_mean, flux_std,               # spectral change rate
  mfcc_1_mean, mfcc_1_std,          # spectral envelope shape (13 pairs)
  mfcc_2_mean, mfcc_2_std,
  ...
  mfcc_13_mean, mfcc_13_std
]
```

### What Each Feature Captures

**Spectral Centroid** — center of mass of the spectrum.

```
C = Σ(fk · |Xk|) / Σ|Xk|
```

where fk = frequency of bin k, Xk = magnitude at bin k.
High centroid = bright (cymbals, synths). Low = dark (bass, warm pads).
Mean tells you the average brightness. Std tells you whether brightness is
stable (synth pad) or dynamic (trumpet solo). Two tracks can share the same
centroid_mean but sound completely different — std is what separates them.

**Spectral Rolloff** — the frequency below which 85% of spectral energy lives.

```
Σ(k=0 to R) |Xk| = 0.85 · Σ(k=0 to N) |Xk|
```

Distinguishes harmonic tails from noisy tails. A warm acoustic instrument
rolls off early; a bright electronic texture extends further.

**Spectral Bandwidth** — spread around the centroid.

```
B = sqrt( Σ(fk - C)² · |Xk| / Σ|Xk| )
```

Narrow = pure tone (sine wave, flute). Wide = complex/noisy texture
(distorted guitar, layered electronics).

**Spectral Flatness** — geometric mean / arithmetic mean of the spectrum.

```
F = (∏|Xk|)^(1/N) / (1/N · Σ|Xk|)
```

Near 1 = white noise. Near 0 = tonal/pitched. Captures whether the sound
has harmonic structure or is noise-like.

**Spectral Flux** — frame-to-frame change in spectrum.

```
Flux(t) = Σ(|Xk(t)| - |Xk(t-1)|)²
```

High = percussive, dynamic. Low = sustained, pad-like. Mean captures
overall activity; std captures whether the dynamics are steady or bursty.

**MFCCs (1–13)** — Mel-Frequency Cepstral Coefficients. Compress the
mel-scaled log spectrum through a DCT. The resulting 13 coefficients
capture the shape of the spectral envelope — the primary perceptual
timbre carrier. MFCC[1] in particular captures the spectral slope
(overall tilt from low to high frequencies), which is a direct
correlate of perceived brightness.

### Why std Matters — A Concrete Example

```
Track X: centroid_mean=1500 Hz, centroid_std=200   → uniformly bright (synth pad)
Track Y: centroid_mean=1500 Hz, centroid_std=1200  → wildly varying (trumpet solo)
```

Same mean, completely different timbre experience. Without std, these
tracks are identical. The previous librosa extraction stored centroid_mean
but not centroid_std — a gap this work addresses.

---

## How Normalisation Works

### The Problem with Raw Values

Raw centroid values depend on the extraction engine:

| Engine   | Grandbrothers 1202 | Jay-Jay Johanson Anywhere Anytime | Ratio |
|----------|-------------------:|----------------------------------:|------:|
| Librosa  |            589 Hz  |                         2597 Hz   | 4.4x  |
| Essentia |            296 Hz  |                         1309 Hz   | 4.4x  |

The absolute Hz values differ (librosa runs ~2.2x higher than Essentia
on average), but the *relative ordering* is the same. For classification,
only the ordering matters.

### Z-Score Normalisation

Each feature dimension is normalised independently across the full corpus:

```
x_norm = (x - μ) / σ
```

where μ = mean across all tracks, σ = standard deviation across all tracks.

This ensures:
- Every feature has mean 0 and std 1
- No single feature dominates due to scale differences
- The 2.2x offset between librosa and Essentia washes out
- A centroid of 589 Hz becomes "1.5 std below average" regardless of engine

### From Vector to Brightness Score

After z-normalisation, a weighted combination produces a single brightness
score:

```python
weights = {
    centroid_mean:  0.30,   # where spectral mass sits
    mfcc_slope:     0.25,   # spectral envelope tilt
    contrast_ratio: 0.20,   # high-band vs low-band energy
    rolloff_mean:   0.10,   # energy distribution tail
    zcr:            0.08,   # high frequency content
    flatness:       0.05,   # noise vs tone
    flux:           0.02,   # spectral dynamics
}
```

The weighted sum is passed through a sigmoid to map to 0–1:

```
brightness = 1 / (1 + exp(-weighted_sum))
```

This gives a continuous score where 0 = dark, 1 = bright, with natural
clustering around the extremes.

---

## Validation Results

### Cross-Engine Validation: Librosa vs Essentia

39 tracks across 13 artists. Essentia uses the same algorithms as
AcousticBrainz (the largest public audio feature database, 29.4M tracks).

**Centroid correlation:**

| Metric | Value |
|--------|------:|
| Pearson correlation (linear) | 0.897 |
| Spearman rank correlation | 0.896 |
| p-value | < 0.000001 |
| Mean ratio (librosa/essentia) | 2.21x |
| Ratio std | 0.36 |

Both engines rank tracks in the same order. The ~2x scale difference is
a systematic offset from different algorithm definitions
(`spectral_centroid` vs `SpectralCentroidTime`), not measurement error.

### EffNet vs Full Vector — Spread Comparison

| Approach | Min | Max | Spread |
|----------|----:|----:|-------:|
| EffNet bright | 0.42 | 0.53 | **0.11** |
| Full vector brightness | 0.10 | 0.82 | **0.72** |

The librosa vector gives 6.5x more discrimination. EffNet's 0.11 spread
means every track scores roughly the same — no usable signal.

### Sanity Check: Known Dark vs Bright

| Track | Brightness | Expected |
|-------|--------:|----------|
| Grandbrothers — 1202 | 0.18 | dark (prepared piano, minimal) |
| Esbjörn Svensson Trio — Beggar's Blanket | 0.18 | dark (quiet jazz piano) |
| Ballaké Sissoko — An Badidjo | 0.31 | dark (acoustic kora) |
| GoGo Penguin — Akasthesia | 0.36 | dark (moody, bass-driven) |
| Four Tet — 31 Bloom | 0.59 | bright (electronic textures) |
| Portico Quartet — A.O.E | 0.67 | bright (layered, energetic) |
| Four Tet — As Serious As Your Life | 0.79 | bright (electronic, active) |
| Jay-Jay Johanson — Anywhere Anytime | 0.82 | bright (vocals, production) |

Results match musical intuition across the full range.

### MusicBrainz Genre Tags

Genre tags from MusicBrainz community voting correlate with timbre:

| Artist | Top Tags | Centroid Range |
|--------|----------|---------------|
| Ballaké Sissoko | jazz | 977–1259 Hz (dark) |
| Grandbrothers | electronic, experimental piano | 590–1102 Hz (dark) |
| Four Tet | electronic, downtempo, idm | 1531–2140 Hz (bright) |
| Jay-Jay Johanson | trip hop, electronic, synthpop | 1947–2597 Hz (bright) |
| GoGo Penguin | jazz, electronica, post-rock | 1137–1630 Hz (spans) |

Artists tagged as jazz/acoustic cluster dark. Electronic/produced cluster bright.

---

## What This Replaces

| Before (v0.3) | After (v0.4) |
|---------------|-------------|
| EffNet backbone: 10s/track | Dropped |
| EffNet timbre: 0.11 spread, useless | Librosa 35-dim vector: 0.72 spread |
| EffNet genre: accurate but expensive | MusicBrainz API: free, cached |
| MusiCNN backbone: 16s/track | Kept — mood, dance, voice, arousal/valence |
| Librosa extraction: ~4s/track | Extended with rolloff, bandwidth, centroid_std |

**Pipeline time per track: ~26s → ~20s** (MusiCNN 16s + extended librosa 4s).
EffNet's 10s eliminated entirely.

---

## Features Missing from Current Extraction

The music-player's librosa DB (v0.2) stores `centroid_mean` but is missing:

| Feature | Status | Impact |
|---------|--------|--------|
| centroid_std | **missing** | Can't distinguish stable vs dynamic brightness |
| rolloff_mean, rolloff_std | **missing** | Can't measure energy distribution tail |
| bandwidth_mean, bandwidth_std | **missing** | Can't measure tonal vs noisy spread |

These must be added to the extractor for v0.4 tags. The extraction cost
is negligible — librosa computes them from the same STFT already computed
for centroid, adding <0.1s per track.

---

## Files

```
~/soniq-lab/
├── timbre_extract.py          # Full 35-dim vector extraction (librosa)
├── timbre_test.py             # Initial proxy comparison (5 approaches)
├── timbre_validate.py         # AcousticBrainz API validation (spotty)
├── validation_pipeline.py     # Full validation: MB + Essentia + librosa
├── output/
│   ├── timbre_comparison.json # Proxy approach results
│   ├── timbre_vectors.json    # Full librosa vectors (39 tracks)
│   ├── reference_dataset.json # Validation dataset (MB + Essentia + librosa)
│   ├── musicbrainz_cache.json # Cached MB artist tags
│   └── validation.json        # AcousticBrainz API comparison
├── TIMBRE.md                  # This document
├── FINDINGS.md                # Essentia classification findings
└── README.md                  # Pipeline setup and usage
```

---

## References

- Schubert & Wolfe (2006) — "Does Timbral Brightness Scale with Frequency
  and Spectral Centroid." Validates centroid as a proxy for perceived brightness.
- McGill MPCL — "The Perceptual Representation of Timbre." Timbre perception
  is multidimensional; centroid alone is insufficient.
- AcousticBrainz (archived) — 29.4M submissions using Essentia extraction.
  CC0 license. API partially functional as of March 2026.
- Essentia (MTG, UPF) — Open-source audio analysis library. Same extraction
  engine as AcousticBrainz.
- librosa (McFee et al.) — Python audio analysis library. Different algorithm
  implementations produce systematically different absolute values but
  correlated rankings (ρ = 0.896).
