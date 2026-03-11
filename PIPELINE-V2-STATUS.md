# Pipeline V2 — Status & Architecture

Single-backbone ONNX pipeline replacing the V1 dual-backbone TensorFlow setup.

---

## Why V2

V1 ran two CNN backbones per track (MusiCNN + EffNet) via TensorFlow:
- ~44s/track on i5-7Y57
- ~500MB TensorFlow runtime dependency
- ~21MB model files
- EffNet accounted for ~55% of processing time but only provided 2/12 classifiers (genre + timbre)

V2 drops EffNet entirely and replaces TensorFlow with ONNX Runtime:
- ~14–28s/track (depends on duration cap)
- ~50MB ONNX Runtime (no TensorFlow)
- ~3.9MB model files
- Genre → Discogs API lookup (not yet implemented)
- Timbre → 35-dim DSP feature vector via librosa (R²=0.91 vs neural)

---

## Current Stack

```
essentia (base, no TF)  →  mel-spectrogram extraction (16kHz, 96 bands)
onnxruntime             →  MusiCNN backbone + 10 classifier heads
librosa                 →  35-dim timbre feature vector
```

No TensorFlow anywhere. The base `essentia` package was installed replacing
`essentia-tensorflow` to avoid TF auto-loading on import (was causing OOM).

---

## Architecture

```
audio file
  │
  ├─→ Essentia DSP: mel-spectrogram (16kHz, 512 frame, 256 hop, 96 bands)
  │     │
  │     └─→ log10(1 + 10000 * mel) normalization
  │           │
  │           └─→ 187-frame patches (50% overlap)
  │                 │
  │                 └─→ ONNX MusiCNN backbone → 200-dim embeddings
  │                       │
  │                       └─→ 10 ONNX classifier heads → predictions
  │
  └─→ librosa: spectral features + MFCCs → 35-dim timbre vector
```

### Classifier heads (all MusiCNN-based via ONNX)

| Classifier | Labels | Type |
|---|---|---|
| mood_happy | happy / non_happy | binary |
| mood_sad | non_sad / sad | binary |
| mood_relaxed | non_relaxed / relaxed | binary |
| mood_aggressive | aggressive / not_aggressive | binary |
| mood_party | non_party / party | binary |
| mood_acoustic | acoustic / non_acoustic | binary |
| danceability | danceable / not_danceable | binary |
| voice_instrumental | instrumental / voice | binary |
| tonal_atonal | tonal / atonal | binary |
| arousal_valence | arousal + valence (1–9) | regression |

### Timbre vector (35 dimensions)

Replaces EffNet bright/dark classifier. Computed via librosa at 22050Hz:

| Features | Dims | Description |
|---|---|---|
| Spectral centroid | 2 | mean, std — frequency center of mass |
| Spectral rolloff | 2 | mean, std — frequency below which 85% of energy |
| Spectral bandwidth | 2 | mean, std — spread around centroid |
| Spectral flatness | 1 | mean — noise vs tonal (0=tonal, 1=noise) |
| Onset flux | 2 | mean, std — spectral change rate |
| MFCCs 1–13 | 26 | mean + std each — timbral texture |

Validated against neural timbre: **R²=0.91** on 42 tracks (1 per artist).
See `tests/BACKBONE-ANALYSIS.md` on `pipeline-v2-investigation` branch for full analysis.

---

## Validation Results (42 tracks, 1 per artist)

Compared V2 ONNX predictions against V1 TF reference stored in `pipeline.db`.

### Per-classifier direction agreement

| Classifier | Agree/Total | Rate | Mean Δ | Notes |
|---|---|---|---|---|
| arousal | 42/42 | **100%** | -0.30 | Rock solid |
| aggressive | 38/42 | **90.5%** | +0.05 | Very reliable |
| acoustic | 36/42 | **85.7%** | -0.05 | Good |
| party | 36/42 | **85.7%** | +0.17 | Good |
| danceable | 34/42 | **81.0%** | +0.16 | Good |
| instrumental | 34/42 | **81.0%** | -0.07 | Good |
| tonal | 32/42 | 76.2% | +0.27 | V2 leans more tonal |
| relaxed | 32/42 | 76.2% | -0.21 | V2 slightly less relaxed |
| sad | 29/42 | 69.0% | -0.18 | V2 less sad |
| valence | 28/42 | 66.7% | **+1.06** | Systematic positive bias |
| happy | 26/42 | 61.9% | +0.27 | V2 happier overall |

**Overall: 367/462 = 79.4% direction agreement**

### Known biases

- **Valence** has a +1.06 systematic shift — V2 consistently scores higher valence
- **Happy** shifts positive (+0.27) — V2 rates tracks happier
- **Tonal** shifts positive (+0.27) — V2 rates tracks more tonal

Root causes (suspected):
1. **120s duration cap** — V2 processes first 120s, V1 processed full track. Intros are often calmer/simpler than full pieces
2. **Essentia base vs essentia-tensorflow mel differences** — minor numerical differences in mel band computation between the two packages

Full per-track results: `tests/v2_validation.json`

---

## Files

```
classify_v2.py              # V2 pipeline script
models-onnx/                # Downloaded ONNX models (~3.9MB total)
  msd-musicnn-1.onnx        #   backbone (3.1MB)
  mood_happy-*.onnx          #   classifier heads (~81KB each)
  ...
results-v2/                 # Per-track JSON results
tests/validate_v2.py        # Validation script (V2 vs TF reference)
tests/v2_validation.json    # Full validation results (42 tracks)
```

---

## Known Issues / Bugs

### 1. OOM on full-length tracks
**Status:** open — workaround in place (120s cap)

The machine has 6.5GB RAM. Processing full tracks (some 400s+) causes OOM kills.
The frame-by-frame Python loop in `compute_mel_spectrogram()` builds ~25,000
numpy arrays in a list before converting, which peaks memory usage.

Current workaround: `MAX_DURATION = 120` caps audio at 120 seconds.

Possible fixes:
- Pre-allocate numpy array instead of appending to list
- Process mel in chunks and discard intermediate data
- Use Essentia's streaming API instead of standard frame loop
- Increase swap space on the machine

This is likely the main cause of the validation discrepancies — V1 processed
full tracks, V2 only sees the first 2 minutes.

### 2. Valence systematic bias (+1.06)
**Status:** open — needs investigation

V2 valence is consistently ~1 point higher than V1 across all tracks.
Could be caused by the 120s cap (intros may have different emotional tone)
or a subtle difference in mel computation between essentia base and
essentia-tensorflow.

### 3. Happy/tonal positive shift
**Status:** open — likely related to duration cap

V2 trends happier (+0.27) and more tonal (+0.27). Probably same root cause
as the valence bias — truncated audio misses darker/more complex later sections.

---

## TODO

### Must fix
- [ ] **Remove 120s duration cap** — fix OOM to process full tracks and improve validation accuracy
- [ ] **Re-validate after OOM fix** — re-run `tests/validate_v2.py` with full tracks

### Should do
- [ ] **Genre via Discogs API** — replace EffNet genre classifier with API lookup (artist + album → genre tags). Fallback chain: Discogs → MusicBrainz → file metadata → skip
- [ ] **Integrate timbre vector into tag schema** — decide how to store 35-dim vector in m4a tags, or derive bright/dark score from it
- [ ] **Update pipeline.py for V2** — adapt the batch pipeline (SQLite tracking, web monitor, tag writing) to use the new ONNX backend

### Nice to have
- [ ] **Vectorize mel computation** — replace frame-by-frame Python loop with batch Essentia or numpy-based mel extraction for speed
- [ ] **Compare full-track vs 120s predictions** — quantify how much the cap actually affects results (once OOM is fixed)
- [ ] **Benchmark against V1** — time comparison on same tracks with full processing

---

## Setup

```bash
cd ~/soniq-lab
source venv/bin/activate

# Dependencies (essentia base, NOT essentia-tensorflow)
pip install essentia onnxruntime librosa numpy

# Download ONNX models (auto on first run, or manually)
python3 classify_v2.py --download-only

# Classify tracks
python3 classify_v2.py --limit 5          # test on 5 tracks
python3 classify_v2.py --track "Prayer"   # single track by name
python3 classify_v2.py                    # full library

# Validate against TF reference
python3 tests/validate_v2.py
```

---

## Branch History

| Branch | Purpose | Status |
|---|---|---|
| `main` | V1 pipeline (TF, dual backbone) | complete — 1081 tracks processed |
| `pipeline-v2-investigation` | Timbre proxy tests, backbone analysis | pushed, research complete |
| `pipeline-v2-onnx` | **V2 pipeline (this branch)** | active |
