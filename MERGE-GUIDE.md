# Merge Guide — onnx-speedup branch improvements

The `pipeline-v2-onnx` branch found and solved several issues that
`onnx-speedup` should adopt. This document describes what to change and why.

---

## 1. Fix OOM on long tracks — pre-allocate mel array

**Problem:** `extract_mel()` builds a Python list of ~25,000 numpy arrays
via `frames.append()`, then converts with `np.array(frames)`. This doubles
memory and causes OOM kills on machines with <8GB RAM (e.g. 6.5GB Chromebook).

**Fix:** Pre-allocate the output array and write directly into it.

```python
def extract_mel(audio_path):
    import gc
    from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands

    audio = MonoLoader(filename=str(audio_path), sampleRate=16000)()

    n_frames = (len(audio) - 512) // 256 + 1
    mel = np.zeros((n_frames, 96), dtype=np.float32)

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

    for i in range(n_frames):
        start = i * 256
        mel[i] = mel_bands(spectrum(windowing(audio[start:start + 512])))

    del audio; gc.collect()
    np.log10(1 + 10000 * mel, out=mel)  # in-place, no copy

    return mel
```

Key changes:
- `np.zeros` pre-allocation instead of list append
- `del audio; gc.collect()` frees the raw audio before patches are built
- `np.log10(..., out=mel)` avoids creating a second array

---

## 2. Batch backbone inference

**Problem:** Sending all patches (e.g. 268 for a 400s track) to MusiCNN
in a single `backbone.run()` call allocates ~1.1GB of internal buffers.
Combined with mel + patches + librosa, this can OOM.

**Fix:** Process patches in small batches (32 at a time).

```python
BACKBONE_BATCH = 32

def run_onnx_classification(mel, backbone, heads):
    patches = make_patches(mel)

    # Batched backbone inference
    batches = []
    for start in range(0, len(patches), BACKBONE_BATCH):
        batch = patches[start:start + BACKBONE_BATCH]
        batches.append(backbone.run(None, {"melspectrogram": batch})[1])
    embeddings = np.concatenate(batches)
    del patches, batches

    # ... rest of classification unchanged
```

The current `BATCH_SIZE = 2` is too conservative — 32 works fine and is
much faster. Peak memory stays under 1.8GB even for 400s tracks.

---

## 3. Validation — run the full 42-track comparison

We validated against the TF reference in `pipeline.db` on all 42 artists.
Final results with the correct MelBands params + full tracks + batched
inference:

```
Classifier       Agree  Total    Rate   Mean Δ
--------------------------------------------------
happy               41     42   97.6%   -0.013
sad                 36     42   85.7%   +0.069
relaxed             40     42   95.2%   +0.021
aggressive          41     42   97.6%   -0.002
danceable           33     42   78.6%   -0.061
instrumental        37     42   88.1%   -0.046
acoustic            41     42   97.6%   +0.024
party               42     42  100.0%   +0.010
tonal               40     42   95.2%   -0.047
arousal             42     42  100.0%   -0.007
valence             42     42  100.0%   +0.052

Overall direction agreement: 435/462 (94.2%)
```

The validation script is at `tests/validate_v2.py` on `pipeline-v2-onnx`.
It compares every classifier output against `pipeline.db` TF reference
and reports direction agreement + mean deltas. Worth running after any
changes to mel extraction or classification to catch regressions.

---

## 4. Keep the 35-dim timbre vector in the tag

The brightness z-score approach in `onnx-speedup` is clean and portable,
but consider also storing the full 35-dim timbre vector in the tag.
It enables future use cases without re-extraction:

- Timbre-based similarity search (cosine distance between vectors)
- Clustering tracks by timbral characteristics
- Retraining the brightness model with more data

The vector is 35 floats = ~280 bytes. Fits easily in m4a padding.

```json
"cls": {
    "bright": 0.49,
    "dark": 0.51,
    "timbre_vec": [35 floats]
}
```

---

## Summary of changes to make on onnx-speedup

| # | What | Where | Priority |
|---|---|---|---|
| 1 | Pre-allocate mel array | `extract_mel()` | Must — prevents OOM |
| 2 | Batch backbone to 32 | `run_onnx_classification()` | Must — prevents OOM |
| 3 | Free audio before patches | `extract_mel()` | Must — memory |
| 4 | Increase BATCH_SIZE 2→32 | constant | Should — 2 is too slow |
| 5 | Run 42-track validation | use `tests/validate_v2.py` | Should — verify accuracy |
| 6 | Store timbre vector in tag | `build_tag()` | Nice to have |
