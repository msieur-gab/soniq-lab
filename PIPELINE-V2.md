# Pipeline V2 — Single-backbone architecture

## Problem

Current pipeline runs **two CNN backbones** per track (~42s average):
- MusiCNN (3.1MB) — mood, danceability, voice/instrumental, tonal/atonal, arousal/valence
- EffNet (18MB) — genre (discogs400), timbre (bright/dark)

Each backbone processes the full audio through sliding windows. The classifier heads on top are near-instant — the bottleneck is running two full-audio CNN passes.

## Insight

The two things keeping EffNet in the pipeline — **genre** and **timbre** — don't actually need a neural network:

- **Genre**: The discogs400 model predicts what Discogs *would* tag a track. We can just look up the actual Discogs tag via API — ground truth instead of a prediction.
- **Timbre (bright/dark)**: Brightness is spectral centroid, a basic DSP calculation. No model needed.

## Proposed architecture

```
1. MonoLoader + resample to 16kHz          (~2-3s)
2. MusiCNN backbone — single pass           (~15-18s)
3. Classifier heads on MusiCNN embeddings   (~0.5s)
     → mood_happy, mood_sad, mood_relaxed
     → mood_aggressive, mood_party, mood_acoustic
     → danceability
     → voice_instrumental
     → tonal_atonal
     → emomusic (arousal/valence)
4. Spectral centroid → brightness score     (~0.1s)
5. Genre lookup via API                     (~0.3s)
                                       ─────────────
                                       Total: ~20s
```

**~50% faster**, single backbone, genre is more accurate.

## Genre: API strategy

### Primary: Discogs API
- Free, 60 requests/min with auth token
- Query by artist + album → get styles and genres
- Returns the exact taxonomy the discogs400 model was trained to predict
- Hierarchical: genre (electronic, jazz, rock...) + style (ambient, house, techno...)
- Endpoint: `GET /database/search?artist=X&release_title=Y&type=release`

### Fallback: MusicBrainz
- Fully open, no auth needed (polite rate limit: 1 req/s)
- Genre tags via release groups or recordings
- Less granular than Discogs but good coverage
- Endpoint: `https://musicbrainz.org/ws/2/release?query=artist:X+release:Y&fmt=json`

### Fallback: existing metadata
- m4a files may already contain genre tags from purchase/rip source
- Read from ID3/MP4 atoms before hitting any API

### Last resort: neural model
- Keep genre_discogs400-discogs-effnet as optional fallback
- Only load EffNet backbone when API lookups fail
- For well-tagged libraries this should rarely trigger

## Timbre: spectral centroid

Brightness/darkness is perceptually tied to spectral centroid — where the "center of mass" of the frequency spectrum sits.

```python
import librosa
import numpy as np

def compute_brightness(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Normalize to 0-1 range (typical centroid: 500-8000 Hz)
    mean_centroid = centroid.mean()
    brightness = np.clip((mean_centroid - 500) / 7500, 0, 1)
    darkness = 1 - brightness
    return round(float(brightness), 4), round(float(darkness), 4)
```

- Milliseconds to compute, no model, no GPU
- Could also add spectral rolloff and spectral contrast for richer timbre profile
- Calibration: compare against current neural timbre scores on existing 947 tracks to tune normalization

## Migration plan

### Phase 1: Validate
- Pick 25 test tracks (extremes + mid-range from current data)
- Run spectral centroid, compare against neural bright/dark scores
- Run Discogs API lookup, compare against discogs400 predictions
- Quantify accuracy vs current pipeline

### Phase 2: Build
- New `classify_v2.py` with MusiCNN-only backbone
- `genre_lookup.py` — Discogs API with MusicBrainz + metadata fallback
- `timbre_dsp.py` — spectral centroid + optional extras
- Update `pipeline.py` to use V2 path

### Phase 3: Backfill
- Re-tag the 1081 tracks with V2 pipeline
- Genre: API lookup for all (most should hit Discogs easily)
- Timbre: spectral centroid for all (fast, minutes not hours)
- Keep neural cls scores alongside for comparison

### Phase 4: Cleanup
- Remove EffNet backbone and heads if V2 validated
- Drop EffNet model files (~20MB)
- Update server.py and prototypes if field names change

## What stays the same

- Tag schema structure (v0.3 cls fields)
- All prototype code — field names don't change
- SQLite database schema
- Librosa-derived fields (tempo, key, mode, duration, chroma, tonnetz) — kept as-is

## Risks

| Risk | Mitigation |
|---|---|
| API rate limits (Discogs 60/min) | Batch with delays, cache results, MusicBrainz fallback |
| Artist/album not found in Discogs | MusicBrainz fallback → file metadata fallback → neural fallback |
| Spectral centroid doesn't match neural timbre | Calibrate normalization against existing scores; add rolloff/contrast if needed |
| Obscure / self-released music has no API tags | Keep EffNet as optional fallback, don't delete the models |

## Estimated improvement

| | V1 (current) | V2 (proposed) |
|---|---|---|
| Backbones | 2 (MusiCNN + EffNet) | 1 (MusiCNN) |
| Time per track | ~42s | ~20s |
| 134 new tracks | ~94 min | ~45 min |
| Genre source | Neural prediction | Ground truth (API) |
| Timbre source | Neural prediction | DSP (spectral centroid) |
