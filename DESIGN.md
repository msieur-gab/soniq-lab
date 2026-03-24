# Design & Research

How the classifiers were built, what research informed them, and what
the current results look like. Validated on a 1942-track corpus spanning
jazz, electronic, ambient, acoustic, hip-hop, and classical.

## Philosophy

**Formulas over ML models.** The starting point was Essentia's neural pipeline:
MusiCNN and EffNet backbones extracting embeddings, then classifier heads for
each dimension. It works — but costs ~20s/track for embedding alone, requires
two separate neural backends, and produces scores you can't explain or debug.

The question: can direct audio analysis, grounded in music theory and
perception research, match those results? The answer is yes. Every classifier
is now a hand-crafted formula operating on librosa features — no trained
weights, no model files, no opaque inference.

Why this matters:
- **Transparent** — you can read the code and understand what drives each score
- **Fast** — ~3ms total for all 14 classifiers (vs ~20s for neural embedding + inference)
- **Lightweight** — only librosa + numpy + scipy, no TensorFlow or model downloads
- **No training data bias** — formulas express acoustic principles, not dataset artifacts
- **Tunable** — change a weight, re-run on stored features, hear the difference immediately
- **Portable** — features + classifications embedded in the audio file itself

The v0.7 formulas achieve strong Spearman correlations against MusiCNN-trained
reference scores. The tonal classifiers (happy, sad, valence) became
effectively genre-independent after the chroma_major_corr breakthrough.
The entire pipeline was built iteratively: extract features, write formula,
validate against reference DB, listen to outliers, refine. Each classifier
went through multiple rounds of this cycle.

## Research foundations

The classifier designs draw from these key papers:

**Friberg 2014** — 9 perceptual features (speed, rhythmic clarity, dynamics,
modality, brightness, pitch). Key findings:
- Modality (major/minor) is the **strongest predictor of valence** (R²=0.87 with 4 features)
- Brightness and pitch have r=0.9 correlation — listeners perceive them as one feature
- Energy predicted by speed + dynamics alone (R²=0.93)
- Simple linear regression is sufficient — no need for nonlinear models

**Grekow 2018** — Feature sets for arousal vs valence:
- Arousal: spectral energy, entropy, flux, rolloff, skewness + rhythm (danceability, onset rate)
- Valence: tonal features are critical — chords strength, HPCP entropy, key strength, key scale
- "Tonal features help a lot for detecting valence"

**Yang 2008** — Regression approach to music emotion recognition:
- R²=0.583 arousal, R²=0.281 valence
- Arousal is much easier than valence with audio-only features
- Valence ceiling with audio is r≈0.67 — lyrics carry the rest

**Eerola & Vuoskoski 2011** — 110 film music excerpts, 116 participants:
- 3D model (valence, energy arousal, tension arousal)
- Two dimensions sufficient for most variance

**Griffiths 2021** — Linear regressors for arousal/valence:
- Arousal from energy features alone achieves R²=0.92
- Valence from spectral spread + flatness: R²=0.96 on small set

## Key breakthroughs

### chroma_major_corr (v0.6)

The single biggest quality improvement. Computes the correlation between a
track's chroma vector and the Krumhansl-Schmuckler major key profile — a
continuous measure of how "major key" the music sounds (0 to 1).

Before: valence used binary `mode` (0=minor, 1=major) from key detection.
This was noisy and lost the strength of the tonal signal.

After: `chroma_major_corr` provides a continuous, genre-independent tonal
signal. Impact on Spearman correlation vs reference:
- Valence: 0.486 → 0.772
- Sad: 0.310 → 0.657

Zero audio cost — computed from existing chroma vectors already being extracted.

### Centered audio sampling (v0.7)

Changed from loading 90s from the start to loading 60s centered on the track.
Avoids intros/outros, captures the musical heart.

Fixed false positives like Kiwanuka (instrumental score 0.80 → 0.36) where
long vocal intros were being sampled as representative.

### ffmpeg direct decode (v0.7)

Replaced `librosa.load()` with direct ffmpeg subprocess calls. No audioread
dependency (deprecated in librosa 0.10, removed in 1.0). ffmpeg handles
m4a/AAC natively. Future-proof.

## Feature extraction

**Multi-point sampling:** 3 × 10s segments at 15%, 50%, 85% of the loaded
audio. Single STFT per segment, reused for all spectral features. Results
averaged across segments for stability.

**Full-audio features:** HPSS (harmonic/percussive separation), tempogram,
PLP (predominant local pulse), and onset detection run on the full 60s window
because they need temporal context.

**Key feature groups:**
- Spectral: centroid, bandwidth, flatness, flux (mean + std)
- Rhythmic: tempo, beat strength, onset rate, beat regularity, rhythm complexity, PLP stability
- Harmonic: HPSS energy split, harm_fraction, chroma, tonnetz, chroma_major_corr
- Timbral: MFCCs (13-dim mean), treble ratio, modulation crest
- Dynamic: RMS (mean in dB, variance from frame-level data), delta MFCC variance
- Metadata: duration, key, mode

## The 14 classifiers

All classifiers output 0–1 scores. Each uses z-score normalization through
sigmoid (corpus-relative scaling) to ensure full range usage.

### Foundation pair
- **arousal** (0=calm, 1=intense) — 4 components: urgency (onset rate + percussive energy), drive (tempo + flux), loudness (RMS), simplicity (inverse rhythm complexity). Weights: 0.35 / 0.30 / 0.20 / 0.15.
- **valence** (0=negative, 1=positive) — chroma_major_corr as dominant signal (0.45 weight), spectral warmth (centroid + treble, 0.30), vitality (tempo + beat, 0.25). Literature confirms tonal features are the key to valence; energy features support but don't override.

### Emotion
- **happy** — tonal positivity from chroma_major_corr (0.35), spectral brightness (0.25), rhythmic joy (0.20), onset liveliness (0.20). The continuous major key measure correctly separates ambiguous tracks (Billie Eilish "bad guy" chroma_corr=0.24) from clearly happy ones (Feist "1234" chroma_corr=0.74).
- **sad** — inverted tonal signal: low/ambiguous chroma_major_corr (0.35), stillness (low percussive energy + sparse onsets, 0.25), spectral darkness (0.20), harmonic weight from sustained harmonics (0.20). Captures tonal ambiguity that makes melancholic music feel sad.
- **relaxed** — calm (low onset rate + perc energy + flux, 0.40), gentle (low centroid + flatness, 0.25), quiet (low RMS, 0.20), harmonic richness (0.15).
- **aggressive** — harshness (high centroid + flatness + treble, 0.35), intensity (flux + onset rate + percussive energy, 0.35), loudness (0.30).

### Rhythm / movement
- **danceable** — percussive energy as MVP signal (strongest single correlate, rho +0.72). Components: percussive (0.30), beat lock (regularity + PLP stability, 0.30), tempo zone (gaussian at 120 BPM, 0.20), groove (simplicity + beat strength, 0.20).
- **party** — feature-only, no inter-classifier dependencies. Percussive drive (0.25), rhythmic lock (regularity + PLP + beat, 0.25), activity (0.15), loudness (0.15), spectral brightness (0.10), tempo zone (0.10).

### Energy (composite)
- **energetic / still** — decomposed into 4 components:
  - *pulse* — beat strength + tempo (0.30 weight)
  - *impact* — percussive energy (0.20 weight)
  - *activity* — onset rate (0.20 weight)
  - *groove* — beat regularity + PLP stability + rhythm simplicity (0.30 weight)
  - Loudness is a **multiplier** (scales core 75%–100%), not additive — a loud drone stays still, a quiet locked groove still has energy.

### Character
- **hypnotic / varied** — two-path approach with soft-max blending:
  - *Rhythmic hypnotic*: locked pulse (PLP stability 0.40) + beat regularity + rhythm simplicity + energy consistency (low RMS variance)
  - *Timbral hypnotic*: stable timbre (low centroid_std 0.35) + low MFCC delta variance + low flux_std + low bandwidth_std
  - Stronger path dominates (0.75/0.25 blend). Reports path label: "rhythmic", "timbral", or "both" when paths converge.
- **instrumental / vocal** — three signal families grounded in VAD research and MFCC design:
  - *Stability* (0.45): voice modulates spectrum dynamically — measures centroid_std (d=1.60), MFCC delta var (d=1.29), flux_std (d=1.21), bandwidth_std (d=1.14)
  - *Modulation regularity* (0.25): instrumental has peaked modulation pattern (d=1.24)
  - *Spectral shape* (0.30): MFCC1 (spectral slope, d=1.19) + MFCC3 (formant absence, d=0.89)
  - Effect sizes (Cohen's d) validated on 145 vocal vs 397 instrumental tracks from the corpus.

### Timbre
- **brilliant / warm** — spectral centroid z-score through sigmoid, calibrated from 1942-track corpus (mean=1395 Hz, std=609 Hz). Based on Schubert & Wolfe 2006 and Peeters 2011 Timbre Toolbox. Pure acoustic color — NOT perceptual atmosphere (that's radiant/somber).

### Atmosphere
- **radiant / somber** — perceived atmosphere: euphoric, inviting vs scary, ominous, void-like. This is explicitly **not** acoustic brightness (brilliant/warm) and **not** emotional valence (happy/sad). Components: fullness (mfcc0 acoustic body), melodic richness (harm_fraction × centroid interaction), atmosphere dimming (stillness + spectral darkness + minor key pull). Key test: acid techno has high centroid (brilliant) but is perceptually somber.
- **contemplative / restless** — reflective depth, distinct from mere stillness. A cold mechanical quiet track is still but not contemplative. Components: spaciousness (sparse events + low flux, 0.30), emotional depth (quiet + harmonic + minor coloring, 0.25), tonal richness (0.20), unhurried pace (0.25).

## Inactive classifiers

**acoustic** (`acoustic.py`) — Dropped in v0.6. Attempted to measure
"recorded with real instruments" vs "electronic/synthetic." The harmonic/flatness
signals it relies on don't reliably separate organic synthesis from acoustic
recording. A synthesizer imitating strings scores acoustic; a close-miked drum
kit scores electronic. Needs a fundamentally different approach.

**tonal** (`tonal.py`) — Dropped in v0.6. Music theory concept (clear pitch
and harmonic structure vs noise/atonal) that doesn't map to listener experience.
The useful signals — key clarity, harmonic purity — are already captured by
valence, sad, and happy through `chroma_major_corr`. May return if a distinct
use case emerges.

Both files are preserved for reference and potential reactivation.

## Results and known boundaries

### What works well

The v0.7 classifiers produce musically coherent results across 1942 tracks.
The tonal classifiers (happy, sad, valence) are effectively genre-independent
thanks to chroma_major_corr — the same acoustic feature correctly identifies
happiness in jazz, electronic, and acoustic music. The energy/danceable/party
cluster reliably separates driving tracks from ambient ones. Hypnotic correctly
identifies locked grooves (techno, process music) and drones (ambient) via
its two-path approach.

Each classifier was iteratively validated: write formula, run against
reference DB, listen to outliers, refine weights. The reference DB preserves
MusiCNN-trained scores from the Essentia pipeline (v0.3–v0.5) as ground truth
for correlation testing.

### Genre bias (measured, understood)

Classifiers that rely on spectral/rhythmic features have measurable genre
sensitivity — a jazz track and an electronic track that "feel" equally
relaxed score differently because their acoustic profiles are genre-specific.

**Genre-independent** (spread < 0.15 across 5 genre groups, N=1180 tracks):
- happy (0.062), valence (0.087), sad (0.129)
- Solved by chroma_major_corr — tonal features are genre-invariant

**Genre-sensitive** (spread > 0.20):
- instrumental (0.409), relaxed (0.297), arousal (0.270)
- radiant (0.256), danceable (0.226), contemplative (0.207)

This is a known property of spectral/rhythmic features in the MIR literature,
not a bug in the formulas. Possible improvement paths: finding more
genre-invariant signals (like chroma_major_corr was for valence), or
per-genre normalization using MusicBrainz tags already available in the DB.

### Audio-only boundaries

These are fundamental limits of audio analysis, not formula limitations:
- **Valence** ceiling is r≈0.67 without lyrics (Yang 2008). Lyrics carry
  the emotional meaning that audio can't access.
- **Instrumental** — sustained warm vocals (Kiwanuka) are acoustically
  similar to instruments. Dynamic electronic music (acid techno) has
  spectral instability that resembles vocal modulation. Centered 60s
  sampling (v0.7) significantly improved this by avoiding misleading
  intros/outros.

### Corpus stats

Z-score normalization uses corpus statistics (`_corpus_stats.py`) computed
from 1942 tracks (2026-03-17). A library with very different genre
distribution may benefit from recomputed stats for optimal range usage.
The `rebuild_corpus_stats.py` utility (available on the `formula-no-weights`
branch) regenerates these from any soniq DB.

## Evolution history

```
v0.3  Essentia neural models (MusiCNN + EffNet). 26s/track. EffNet timbre useless.
v0.4  Hybrid: MusiCNN + extended librosa. ONNX migration attempted.
v0.5  Librosa-only. 124 features, Ridge regression classifiers. 4.4s/track.
v0.6  Formula-only classifiers. chroma_major_corr breakthrough. Dropped acoustic/tonal.
v0.7  Centered 60s sampling. ffmpeg decode. Trimmed extraction (removed pYIN, rolloff,
      zcr, vocal_proxy). 14 classifiers, ~5.5s/track extraction, ~3ms classification.
```
