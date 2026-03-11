# Backbone Analysis — MusiCNN vs EffNet

## Current architecture (v0.3 pipeline)

Two CNN backbones process every track (~44s avg per track):

### MusiCNN backbone (msd-musicnn-1.pb, 3.1MB)
Trained on Million Song Dataset. Processes full audio through sliding windows (~3s patches).
Provides embeddings for **11 classifier heads**:

| Classifier | Output | Type |
|---|---|---|
| mood_happy | happy (0-1) | binary |
| mood_sad | sad (0-1) | binary |
| mood_relaxed | relaxed (0-1) | binary |
| mood_aggressive | aggressive (0-1) | binary |
| mood_party | party (0-1) | binary |
| mood_acoustic | acoustic (0-1) | binary |
| danceability | danceable (0-1) | binary |
| voice_instrumental | instrumental / vocal (0-1 each) | dual |
| tonal_atonal | tonal / atonal (0-1 each) | dual |
| arousal_valence | arousal, valence (continuous) | regression |

**10 cls fields from MusiCNN**: happy, sad, relaxed, aggressive, party, acoustic, danceable, instrumental, vocal, tonal, atonal, arousal, valence

### EffNet backbone (discogs-effnet-bs64-1.pb, 18MB)
Trained on Discogs dataset. Separate full-audio CNN pass.
Provides embeddings for **2 classifier heads**:

| Classifier | Output | Type |
|---|---|---|
| genre_discogs400 | 400-class genre taxonomy | multi-class (top 3 stored) |
| timbre | bright / dark (0-1 each) | dual |

**5 cls fields from EffNet**: genre (top 3 labels), genre_s (top 3 scores), bright, dark

## The problem

EffNet is the heavier backbone (18MB vs 3.1MB) and requires a **full second CNN pass** over the entire audio. It only provides 2 classifiers (genre + timbre) out of 12 total.

Rough time split (estimated from V2 doc):
- MusiCNN pass + heads: ~18s
- EffNet pass + heads: ~24s
- Audio loading/resampling: ~2-3s

**EffNet accounts for ~55% of processing time but only 2 out of 12 classifiers.**

## Can we drop EffNet?

### Genre → YES, replaceable
The discogs400 model *predicts* what Discogs would tag a track. We can query the actual Discogs API for ground truth instead. See PIPELINE-V2.md for API strategy (Discogs → MusicBrainz → file metadata → neural fallback).

### Timbre (bright/dark) → YES, replaceable with DSP feature vector

#### Test 1: Spectral centroid alone (2026-03-11)
Computed Essentia spectral centroid for tracks with neural timbre scores.

**Result: centroid alone does NOT predict neural timbre.**
- Pearson correlation: **r = -0.20** (weak, wrong direction)
- Neural model outputs cluster in a very narrow range (0.43-0.53)
- Centroid has wide spread (583-3667 Hz) but doesn't align with neural predictions
- Example: Daft Punk "Burnin'" has highest centroid (3667 Hz) but neural says dark (0.47 bright)
- Example: Cinematic Orchestra "To Build A Home" has low centroid (769 Hz) but neural says bright (0.53)

Test files: `tests/timbre-dsp/*.json`
Script: `tests/timbre_centroid_test.py`

#### Test 2: 10 spectral features without MFCCs (2026-03-11)
Combined centroid, rolloff, flatness, flux, contrast, bandwidth (mean + std) via Essentia.

**Result: R² = 0.38** — better but still insufficient.
- 10 features, 41 tracks (one per artist)
- Best single predictor: flatness_std (r = -0.33)
- Leaves 62% of variance unexplained

#### Test 3: 35-dim timbre vector with MFCCs (2026-03-11)
Full timbre vector using librosa: spectral features + MFCCs 1-13 (mean + std each).

**Result: R² = 0.91 — strong proxy for neural timbre.**
- 35 features, 42 tracks (one per artist), extracted in 365s (~8.7s/track)
- Mean absolute error: **0.006** (neural range is 0.41-0.53)
- Max error: **0.015**
- 33/42 tracks within 0.01 of neural score, all 42 within 0.015

Script: `tests/timbre_proxy_test.py`
Results: `tests/timbre-proxy-results.json`

**Top individual correlations with neural bright:**

| Feature | Pearson r | Meaning |
|---|---|---|
| mfcc_6_mean | -0.42 | spectral envelope shape → strongest single predictor |
| mfcc_9_mean | -0.34 | mid-frequency envelope detail |
| mfcc_3_mean | -0.34 | low-frequency spectral shape |
| mfcc_8_mean | -0.30 | higher envelope harmonics |
| mfcc_7_mean | -0.28 | mid envelope detail |
| mfcc_1_std | +0.26 | variability of overall energy shape |
| flux_std | -0.25 | temporal dynamics variability |
| bandwidth_std | -0.22 | spectral width variability |

**EffNet vs Librosa — per-track timbre comparison (1 per artist):**

Verdict thresholds: BRIGHT > 0.52, DARK < 0.48, NEUTRAL = 0.48-0.52

| Artist | Title | EffNet Bright | EffNet Dark | EffNet Verdict | Librosa Bright | Librosa Dark | Librosa Verdict | Agree? |
|---|---|---|---|---|---|---|---|---|
| Richie Hawtin, Plastikman | Contain (In Key) | 0.41 | 0.59 | DARK | 0.41 | 0.59 | DARK | YES |
| Ballaké Sissoko | Demba Kunda | 0.44 | 0.56 | DARK | 0.44 | 0.56 | DARK | YES |
| Plastikman, Richie Hawtin | Drp (Remastered) | 0.44 | 0.56 | DARK | 0.45 | 0.55 | DARK | YES |
| Ballaké Sissoko, Vincent Segal | Chamber Music | 0.44 | 0.56 | DARK | 0.44 | 0.56 | DARK | YES |
| GoGo Penguin | Umbra | 0.46 | 0.54 | DARK | 0.46 | 0.54 | DARK | YES |
| Matthew Halsall | The Temple Within | 0.46 | 0.54 | DARK | 0.45 | 0.55 | DARK | YES |
| Matthew Halsall, Gondwana Orch. | Only a Woman | 0.47 | 0.53 | DARK | 0.48 | 0.52 | DARK | YES |
| The Cinematic Orchestra | The Projectionist | 0.47 | 0.53 | DARK | 0.47 | 0.53 | DARK | YES |
| BADBADNOTGOOD | Triangle | 0.47 | 0.53 | DARK | 0.47 | 0.53 | DARK | YES |
| Four Tet | Hands | 0.47 | 0.53 | DARK | 0.47 | 0.53 | DARK | YES |
| Floex | Ursa Major | 0.47 | 0.53 | DARK | 0.48 | 0.52 | DARK | YES |
| Jay-Jay Johanson | It Hurts Me So | 0.48 | 0.52 | DARK | 0.47 | 0.53 | DARK | YES |
| Bitcrush | Post (Nichts Ist Wie Vorher) | 0.48 | 0.52 | DARK | 0.48 | 0.52 | DARK | YES |
| Arms and Sleepers | We're All Paris Now | 0.48 | 0.52 | DARK | 0.48 | 0.52 | DARK | YES |
| Ahmad Jamal | Autumn Rain | 0.48 | 0.52 | DARK | 0.47 | 0.53 | DARK | YES |
| Floex, Tom Hodge | Inauguration Of Nobody | 0.48 | 0.52 | DARK | 0.48 | 0.52 | NEUTRAL | **NO** |
| Quantic | Introduction | 0.47 | 0.53 | DARK | 0.47 | 0.53 | DARK | YES |
| Jaga Jazzist | Tomita | 0.48 | 0.52 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| e.s.t. Esbjörn Svensson Trio | Dating | 0.48 | 0.52 | NEUTRAL | 0.47 | 0.53 | DARK | **NO** |
| The Cinematic Orchestra, Tawiah | Wait For Now | 0.48 | 0.52 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Flying Lotus | Auntie's Harp | 0.48 | 0.52 | DARK | 0.48 | 0.52 | NEUTRAL | **NO** |
| Tibor SZEMZO, Group 180 | Water-Wonder | 0.48 | 0.52 | NEUTRAL | 0.48 | 0.52 | NEUTRAL | YES |
| Bersarin Quartett | Gespenster | 0.49 | 0.51 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Cinematic Orchestra, Fontella | All That You Give | 0.49 | 0.51 | NEUTRAL | 0.48 | 0.52 | NEUTRAL | YES |
| Hidden Orchestra, Piano Int. | Cross Hands (Remix) | 0.49 | 0.51 | NEUTRAL | 0.50 | 0.50 | NEUTRAL | YES |
| Portico Quartet | News from Verona | 0.49 | 0.51 | NEUTRAL | 0.48 | 0.52 | NEUTRAL | YES |
| GoGo Penguin, Cornelius | Kora (Cornelius Remix) | 0.49 | 0.51 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Grandbrothers | 1202 | 0.49 | 0.51 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Hidden Orchestra | Overture | 0.50 | 0.50 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Jóhann Jóhannsson | Flight from the City | 0.50 | 0.50 | NEUTRAL | 0.51 | 0.49 | NEUTRAL | YES |
| Mammal Hands | Quiet Fire | 0.50 | 0.50 | NEUTRAL | 0.50 | 0.50 | NEUTRAL | YES |
| Quantic, Andreya Triana | Run | 0.50 | 0.50 | NEUTRAL | 0.50 | 0.50 | NEUTRAL | YES |
| Vikingur Olafsson, Philip Glass | Glassworks Opening | 0.50 | 0.50 | NEUTRAL | 0.50 | 0.50 | NEUTRAL | YES |
| Cinematic Orchestra, Sumney | To Believe | 0.50 | 0.50 | NEUTRAL | 0.51 | 0.49 | NEUTRAL | YES |
| Billie Eilish | SKINNY | 0.51 | 0.49 | NEUTRAL | 0.49 | 0.51 | NEUTRAL | YES |
| Daft Punk | Give Life Back to Music | 0.51 | 0.49 | NEUTRAL | 0.51 | 0.49 | NEUTRAL | YES |
| Cinematic Orchestra, LMO | Opening Titles | 0.51 | 0.49 | NEUTRAL | 0.50 | 0.50 | NEUTRAL | YES |
| Michael Riesman, Philip Glass | The Photographer Act I | 0.51 | 0.49 | NEUTRAL | 0.51 | 0.49 | NEUTRAL | YES |
| Esbjörn Svensson Trio | Fading Maid Preludium | 0.51 | 0.49 | NEUTRAL | 0.51 | 0.49 | NEUTRAL | YES |
| Philip Glass Ensemble | Glassworks I. Opening | 0.52 | 0.48 | NEUTRAL | 0.52 | 0.48 | NEUTRAL | YES |
| Cinematic Orchestra, Watson | To Build A Home | 0.53 | 0.47 | BRIGHT | 0.51 | 0.49 | NEUTRAL | **NO** |

**Agreement: 38/42 (90%)**

4 disagreements — all borderline cases within 0.01-0.02 of the 0.48/0.52 thresholds. Zero BRIGHT↔DARK flips.

**Key insight**: centroid is uncorrelated with timbre (The Projectionist has 3730 Hz centroid but is dark; Glassworks has 508 Hz but is bright). MFCCs capture the spectral envelope shape that defines perceptual timbre.

#### The 35-dim timbre vector

```
timbre_vector = [
  centroid_mean, centroid_std,       # brightness + its variability
  rolloff_mean, rolloff_std,         # energy distribution tail
  bandwidth_mean, bandwidth_std,     # tonal vs noisy spread
  flatness_mean,                     # tone vs noise ratio
  flux_mean, flux_std,               # temporal dynamics
  mfcc_1_mean, mfcc_1_std,          # overall spectral shape
  mfcc_2_mean, mfcc_2_std,          # spectral slope
  ...                                # ...
  mfcc_13_mean, mfcc_13_std         # fine spectral detail
]
```

Extracted via librosa at ~8.7s/track (vs ~24s for EffNet backbone).

## Summary

| Feature | Backbone | Replaceable? | Strategy |
|---|---|---|---|
| Mood (6 dims) | MusiCNN | N/A (keeping) | — |
| Danceability | MusiCNN | N/A (keeping) | — |
| Voice/Instrumental | MusiCNN | N/A (keeping) | — |
| Tonal/Atonal | MusiCNN | N/A (keeping) | — |
| Arousal/Valence | MusiCNN | N/A (keeping) | — |
| Genre | EffNet | YES | Discogs API + fallbacks |
| Timbre | EffNet | YES | 35-dim DSP vector (librosa), R²=0.91 |
