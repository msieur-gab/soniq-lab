# Soniq Lab — ONNX Migration: From TensorFlow to Pure DSP + ONNX Runtime

Research and implementation by **Gab** and **Claude Code** (Anthropic CLI, Opus 4.6).
March 2026.

---

## The Problem

The v0.3 pipeline depended on `essentia-tensorflow`, which bundles a full
TensorFlow 2.x runtime (~574MB) just to run inference on small classification
models. Combined with librosa's dependency on `audioread` (deprecated, will
break), the dependency stack was fragile and bloated.

| v0.3 dependency | Size | Risk |
|---|---|---|
| essentia-tensorflow | ~574MB (TF runtime) | Bundles TF 2.5, pins numpy |
| librosa + audioread | ~30MB | audioread deprecated, will break |
| TF model files (.pb) | ~24MB | TF-specific format |

**Goal:** Replace TF inference with ONNX Runtime, replace librosa DSP with
Essentia DSP, and achieve identical classification results.

---

## The Breakthrough: Replicating TensorflowInputMusiCNN

### The Problem

`TensorflowInputMusiCNN` is a C++ algorithm inside `essentia-tensorflow` that
converts raw audio frames into mel-spectrogram bands matching MusiCNN's
training format. It's the critical preprocessing step — without an exact match,
classification results drift.

The V2 branch (`pipeline-v2-onnx`) attempted to replace it using Essentia's
generic `MelBands` algorithm with default parameters. This produced a mean
difference of **0.76** and correlation of **0.856** against the reference,
resulting in only **79.4% direction agreement** with TF predictions.

### The Root Cause

`MelBands` has several parameters that default to values different from what
MusiCNN was trained on. The V2 branch used:

```python
# V2 approach — WRONG defaults
mel_bands = MelBands(numberBands=96, sampleRate=16000,
                     lowFrequencyBound=0, highFrequencyBound=8000,
                     inputSize=257)
# Uses default warpingFormula, weighting, normalize → DIFFERENT mel output
```

### The Solution

By reading the [Essentia C++ source](https://github.com/MTG/essentia/blob/master/src/algorithms/spectral/tensorflowinputmusicnn.cpp)
for `TensorflowInputMusiCNN`, we found three critical parameters:

```python
mel_bands = MelBands(
    numberBands=96,
    sampleRate=16000,
    lowFrequencyBound=0,
    highFrequencyBound=8000,
    inputSize=257,
    warpingFormula="slaneyMel",    # ← not the default
    weighting="linear",            # ← not the default
    normalize="unit_tri",          # ← not the default
)
```

Combined with the log compression formula `log10(1 + 10000 * mel)`, this
produces **zero difference** against `TensorflowInputMusiCNN`:

```
Mean diff: 0.00000000
Max diff:  0.00000000
Correlation: 1.00000000
```

**This is a perfect, bit-for-bit match.** Not approximate — identical.

### Why This Matters

With these three parameters, the entire `TensorflowInputMusiCNN` algorithm
is replicated using only base Essentia DSP components:

```python
# Complete MusiCNN mel preprocessing — NO TF-specific algorithms needed
from essentia.standard import Windowing, Spectrum, MelBands

windowing = Windowing(type="hann", size=512, zeroPadding=0, normalized=False)
spectrum = Spectrum(size=512)
mel_bands = MelBands(
    numberBands=96, sampleRate=16000,
    lowFrequencyBound=0, highFrequencyBound=8000,
    inputSize=257,
    warpingFormula="slaneyMel",
    weighting="linear",
    normalize="unit_tri",
)

for frame in frames:
    spec = spectrum(windowing(frame))
    mel = mel_bands(spec)
    output = np.log10(1 + 10000 * mel)
```

`Windowing`, `Spectrum`, and `MelBands` are all standard DSP algorithms
available in the base `essentia` package. No `TensorflowInputMusiCNN`,
no TensorFlow runtime.

---

## What Each Parameter Does

| Parameter | Default | Required | Effect |
|---|---|---|---|
| `warpingFormula` | `"htkMel"` | `"slaneyMel"` | Different mel-to-Hz conversion. Slaney uses a linear region below 1kHz + log above. HTK is purely logarithmic. Changes band center frequencies. |
| `weighting` | `"warping"` | `"linear"` | How mel bands are weighted. "warping" applies the derivative of the warping function; "linear" applies uniform weights. Changes relative band amplitudes. |
| `normalize` | `"unit_sum"` | `"unit_tri"` | How triangular filters are normalized. "unit_sum" = area=1; "unit_tri" = peak=1. Changes absolute magnitude of each band. |

All three must be set correctly. Missing any one produces classification drift.

---

## ONNX Model Architecture

```
audio (16kHz mono)
  → mel spectrogram: Windowing + Spectrum + MelBands + log10(1+10000*mel)
  → patches: 187 frames (~3s) with 93-frame hop (50% overlap)
  → MusiCNN backbone (msd-musicnn-1.onnx, 3.1MB)
    input: "melspectrogram" (batch, 187, 96)
    output: "embeddings" (batch, 200)
  → classifier heads (10 × ~81KB each)
    input: "embeddings" (batch, 200)
    output: "activations" (batch, 2) or (batch, 2) for regression
```

### Models

| Model | File | Size |
|---|---|---|
| MusiCNN backbone | `msd-musicnn-1.onnx` | 3.1 MB |
| mood_happy head | `mood_happy-msd-musicnn-1.onnx` | 81 KB |
| mood_sad head | `mood_sad-msd-musicnn-1.onnx` | 81 KB |
| mood_relaxed head | `mood_relaxed-msd-musicnn-1.onnx` | 81 KB |
| mood_aggressive head | `mood_aggressive-msd-musicnn-1.onnx` | 81 KB |
| mood_party head | `mood_party-msd-musicnn-1.onnx` | 81 KB |
| mood_acoustic head | `mood_acoustic-msd-musicnn-1.onnx` | 81 KB |
| danceability head | `danceability-msd-musicnn-1.onnx` | 81 KB |
| voice_instrumental head | `voice_instrumental-msd-musicnn-1.onnx` | 81 KB |
| tonal_atonal head | `tonal_atonal-msd-musicnn-1.onnx` | 81 KB |
| emomusic (arousal/valence) | `emomusic-msd-musicnn-2.onnx` | 81 KB |
| **Total** | | **~3.9 MB** |

vs TF models: ~24MB (.pb format) + 574MB TF runtime.

---

## Timbre: Essentia DSP Replaces Both EffNet AND Librosa

### EffNet Timbre Was Useless (from earlier research)

682 tracks: 93% scored between 0.45–0.55 bright/dark. Spread of 0.11.
Zero discriminating signal. Costs 10s/track for nothing.

### Librosa Timbre Worked But Had Dependency Risk

35-dim vector with 0.72 spread — 6.5x more than EffNet. But depends on
`audioread` (deprecated) and `librosa` (heavy).

### Essentia Spectral DSP: Same Features, No Extra Dependencies

The same 35-dim timbre vector can be extracted using Essentia's standard
DSP algorithms, which are already loaded for mel spectrogram computation:

```python
from essentia.standard import (
    SpectralCentroidTime, RollOff, Flatness, Flux,
    MFCC, CentralMoments, DistributionShape,
)
```

Validated against librosa: Pearson r=0.897, Spearman ρ=0.896.

The timbre vector layout:

```
centroid_mean, centroid_std,       # 2 dims — brightness + variability
rolloff_mean, rolloff_std,         # 2 dims — energy distribution tail
bandwidth_mean, bandwidth_std,     # 2 dims — tonal spread
flatness_mean,                     # 1 dim  — tone vs noise
flux_mean, flux_std,               # 2 dims — spectral dynamics
mfcc1_mean, mfcc1_std,            # 26 dims — spectral envelope (13 pairs)
...
mfcc13_mean, mfcc13_std
```

Brightness score: z-score normalize across corpus → weighted combination → sigmoid → 0–1.

---

## v0.3 vs v0.4 Pipeline Comparison

| | v0.3 (TF) | v0.4 (ONNX) |
|---|---|---|
| **Backbone** | TensorFlow `.pb` via `TensorflowPredictMusiCNN` | ONNX `.onnx` via `onnxruntime` |
| **Mel preprocessing** | `TensorflowInputMusiCNN` (C++, essentia-tf only) | `Windowing + Spectrum + MelBands` (base Essentia DSP) |
| **Classifier heads** | TF `.pb` via `TensorflowPredict2D` | ONNX `.onnx` via `onnxruntime` |
| **Timbre** | EffNet (useless, 0.11 spread) | Essentia spectral DSP (0.50+ spread) |
| **Genre** | EffNet Discogs (10s/track) | MusicBrainz API (free, cached) |
| **Audio loading** | `essentia.standard.MonoLoader` | `essentia.standard.MonoLoader` (same) |
| **TF runtime** | Required (~574MB) | Not needed |
| **librosa** | Required (audioread risk) | Not needed |
| **Model size** | ~24MB (.pb) | ~3.9MB (.onnx) |
| **Time/track** | ~26s (MusiCNN 16s + EffNet 10s) | ~19s (mel+backbone+heads+timbre) |
| **Classification match** | Reference | Identical (zero mel diff) |

### Dependency Reduction

```
v0.3: essentia-tensorflow (~574MB TF) + librosa + audioread + onnxruntime
v0.4: essentia + onnxruntime + numpy
```

The key question remaining is whether the base `essentia` pip package provides
working C++ bindings for `MonoLoader`, `MelBands`, etc. without the TF bundle.
The `pipeline-v2-onnx` branch claims it does, but our testing found the plain
`essentia` pip package was metadata-only (no C++ bindings). This needs
resolution — possibly a different essentia build or version.

---

## Known Issues

### Duration Cap

Tracks longer than ~5 minutes can cause OOM on constrained machines (4GB RAM)
when all mel patches are held in memory. Current mitigation: `MAX_DURATION = 300`
(5 minutes). The V2 branch used 120s which introduced bias — valence was
+1.06 higher than full-track TF reference.

### Essentia Base Package

The pip package `essentia` (without `-tensorflow`) may not include C++
bindings in all builds. The `essentia-tensorflow` package bundles both DSP
and TF runtime. We currently still use `essentia-tensorflow` for `MonoLoader`
and spectral features, even though we no longer use any TF-specific algorithms.
Eliminating this dependency requires finding a working `essentia` build or
alternative audio loading (e.g., `soundfile` + `soxr`).

---

## Files

```
~/soniq-lab/
├── pipeline_onnx.py       # v0.4 pipeline (this work)
├── classify.py            # v0.3 TF pipeline (reference)
├── classify_v2.py         # V2 attempt (pipeline-v2-onnx branch, 79.4% agreement)
├── models/onnx/           # ONNX models (~3.9MB total)
│   ├── msd-musicnn-1.onnx
│   └── *-msd-musicnn-1.onnx (10 classifier heads)
├── ONNX-MIGRATION.md      # This document
├── FINDINGS.md            # v0.3 Essentia classification findings
└── TIMBRE.md              # Timbre vector derivation and validation
```
