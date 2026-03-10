# Soniq Lab — Essentia Classification Findings

Research and experiments by **Gab**, creator of Soniq, with **Claude Code**
(Anthropic CLI, Opus 4.6).

Gab designed the Soniq audio feature system (librosa-based, 61 dimensions,
8 composite scores), built the music player and its sphere visualization,
and initiated this investigation to evaluate whether Essentia's pre-trained
neural classifiers could complement the existing librosa pipeline. He selected
the test tracks, identified the critical label ordering bug by ear (see below),
and defined the hybrid architecture goal: librosa for geometry, Essentia for
semantic labels, both embedded in portable m4a tags.

Claude Code handled the Essentia pipeline implementation — model downloads,
backbone/classifier wiring, TensorFlow node configuration, batch processing,
and documentation. The two worked as a pair: Gab steering direction and
validating results against musical knowledge, Claude writing and running code.

March 2026.

---

## Architecture

### Two-Stage Inference Pipeline

```
audio (16kHz mono) → backbone extractor → embeddings → classifier head → predictions
```

- **Backbone extractors** produce dense embeddings from raw audio
- **Classifier heads** are lightweight models trained on specific tasks
- Each head was trained on a specific backbone — **never mix them**

### Backbones Used

| Backbone | Model File | Size | Output |
|---|---|---|---|
| MusiCNN | `msd-musicnn-1.pb` | 3 MB | 200-dim embeddings per ~3s patch |
| EffNet (Discogs) | `discogs-effnet-bs64-1.pb` | 17.5 MB | 1280-dim embeddings |

### Classifier Heads (12 models)

**MusiCNN-based (9 classifiers):**

| Classifier | Labels | Type |
|---|---|---|
| mood_happy | happy, non_happy | binary |
| mood_sad | non_sad, sad | binary |
| mood_relaxed | non_relaxed, relaxed | binary |
| mood_aggressive | aggressive, not_aggressive | binary |
| mood_party | non_party, party | binary |
| mood_acoustic | acoustic, non_acoustic | binary |
| danceability | danceable, not_danceable | binary |
| voice_instrumental | instrumental, voice | binary |
| tonal_atonal | tonal, atonal | binary |

**EffNet-based (2 classifiers):**

| Classifier | Labels | Type |
|---|---|---|
| genre | 400 Discogs styles | multi-class |
| timbre | bright, dark | binary |

**MusiCNN-based regression (1 model):**

| Classifier | Output | Type |
|---|---|---|
| arousal_valence | arousal + valence (1–9 scale) | regression |

---

## Critical Lesson: Label Ordering

The biggest pitfall we encountered — and one that was caught by Gab through
manual inspection of the raw JSON output. Looking at Prayer's `mood_happy: [0.008, 0.992]`,
he knew from listening that this melancholic piano piece is not happy. So if index 1 = happy,
something is wrong. That observation led to the discovery that label ordering is
**not consistent** across models.

Initial naive assumption: index 0 = negative, index 1 = positive.
**This is wrong for most models.** The only way to know is to download
the metadata JSON that ships alongside each .pb model file. Without it,
you are guessing — and you will guess wrong for 5 out of 9 classifiers.

Correct ordering (verified from official metadata JSONs at essentia.upf.edu):

```python
"mood_happy":       ["happy", "non_happy"]           # 0=happy, 1=non_happy
"mood_sad":         ["non_sad", "sad"]                # 0=non_sad, 1=sad
"mood_relaxed":     ["non_relaxed", "relaxed"]        # 0=non_relaxed, 1=relaxed
"mood_aggressive":  ["aggressive", "not_aggressive"]  # 0=aggressive, 1=not_aggressive
"mood_party":       ["non_party", "party"]            # 0=non_party, 1=party
"mood_acoustic":    ["acoustic", "non_acoustic"]      # 0=acoustic, 1=non_acoustic
"danceability":     ["danceable", "not_danceable"]    # 0=danceable, 1=not_danceable
"voice_instrumental": ["instrumental", "voice"]       # 0=instrumental, 1=voice
"tonal_atonal":     ["tonal", "atonal"]               # 0=tonal, 1=atonal
```

**Rule: always download the metadata JSON alongside each model .pb file.**
The JSONs contain the authoritative `"classes"` array. No JSON = no reliable
predictions. This is not documented prominently in the Essentia docs and is
easy to miss — treat it as mandatory.

---

## EffNet Node Names

EffNet classifier heads use different TensorFlow input/output node names
depending on the model. This must be configured per classifier:

```python
# Genre model
TensorflowPredict2D(
    graphFilename="genre_discogs400-discogs-effnet-1.pb",
    input="serving_default_model_Placeholder",
    output="PartitionedCall"
)

# Timbre model
TensorflowPredict2D(
    graphFilename="timbre-discogs-effnet-1.pb",
    input="model/Placeholder",
    output="model/Softmax"
)
```

MusiCNN heads all use the same default nodes (`model/Softmax` output, default input).

---

## Results: 14-Track Diverse Batch

### Daft Punk — Get Lucky

```
mood_happy:       happy (90%)
mood_sad:         non_sad (89%)
mood_relaxed:     non_relaxed (86%)
mood_aggressive:  not_aggressive (93%)
mood_party:       non_party (52%)     ← borderline, expected higher
danceability:     danceable (98%)     ← spot on
mood_acoustic:    non_acoustic (95%)
voice_instrumental: voice (79%)       ← correct, has Pharrell vocals
tonal_atonal:     tonal (77%)
genre:            African 12%, Nu-Disco 7%, Reggae-Pop 6%, House 6%
arousal/valence:  5.95 / 5.96         ← high energy, positive mood
timbre:           dark (51%)          ← borderline
```

### GoGo Penguin — Prayer

```
mood_happy:       non_happy (99%)
mood_sad:         sad (61%)
mood_relaxed:     relaxed (100%)
mood_aggressive:  not_aggressive (97%)
mood_party:       non_party (100%)
danceability:     not_danceable (83%)
mood_acoustic:    non_acoustic (62%)  ← surprising, possibly due to production/reverb
voice_instrumental: instrumental (92%)
tonal_atonal:     tonal (88%)
genre:            Ambient 24%, Experimental 13%, Drone 8%
arousal/valence:  4.07 / 3.71         ← calm, slightly melancholic
timbre:           dark (51%)
```

### Ballaké Sissoko — Djourou (kora)

```
mood_happy:       non_happy (97%)
mood_sad:         sad (65%)
mood_relaxed:     relaxed (99%)
mood_aggressive:  not_aggressive (99%)
mood_party:       non_party (100%)
danceability:     not_danceable (82%)
mood_acoustic:    acoustic (97%)      ← kora is purely acoustic
voice_instrumental: instrumental (88%)
tonal_atonal:     atonal (94%)        ← kora tuning is non-Western
genre:            Folk 31%, Flamenco 15%, Celtic 9%, Bluegrass 8%, African 7%
arousal/valence:  5.00 / 4.19
timbre:           dark (56%)
```

Genre classifier correctly identifies folk/world music. The "atonal" label
reflects non-Western tuning rather than actual atonality — an important nuance.

### Hidden Orchestra — Dust

```
mood_happy:       happy (62%)
mood_sad:         non_sad (83%)
mood_relaxed:     non_relaxed (58%)   ← borderline — Dust has both calm and energetic sections
mood_aggressive:  not_aggressive (86%)
mood_party:       non_party (90%)
danceability:     danceable (74%)     ← correct, strong rhythmic drive
mood_acoustic:    non_acoustic (76%)
voice_instrumental: instrumental (90%)
tonal_atonal:     tonal (91%)
genre:            Experimental 12%, Ambient 9%, Downtempo 7%, Indie Rock 6%
arousal/valence:  5.15 / 5.05         ← moderate-high energy, neutral-positive
timbre:           dark (50%)
```

### Mammal Hands — Summary (10 tracks)

| Track | Happy | Relaxed | Danceable | Top Genre | Instrumental |
|---|---|---|---|---|---|
| Eyes That Saw the Mountain | non_happy 72% | relaxed 99% | not danceable 84% | Jazz-Contemporary 13% | 85% |
| Hillum | non_happy 77% | relaxed 93% | not danceable 71% | Jazz-Contemporary 10% | 90% |
| Hourglass | non_happy 74% | relaxed 91% | not danceable 60% | Jazz-Contemporary 13% | 85% |
| In the Treetops | non_happy 76% | relaxed 99% | not danceable 83% | Jazz-Contemporary 12% | 78% |
| Kudu | happy 64% | relaxed 82% | danceable 78% | Indie Rock 9% | 95% |
| Quiet Fire | non_happy 62% | relaxed 91% | danceable 56% | Jazz-Contemporary 12% | 90% |
| Shift | non_happy 62% | relaxed 83% | not danceable 70% | Jazz-Contemporary 12% | 78% |
| The Falling Dream | non_happy 76% | relaxed 99% | not danceable 76% | Jazz-Contemporary 12% | 78% |
| Think Anything | happy 68% | relaxed 72% | danceable 71% | Jazz-Contemporary 11% | 96% |
| Window to Your World | happy 64% | relaxed 63% | danceable 78% | Indie Rock 9% | 95% |

The models capture the spectrum within a single band: from contemplative
("In the Treetops" — non_happy, very relaxed, not danceable) to upbeat
("Window to Your World" — happy, danceable, less relaxed). This is exactly
the kind of differentiation needed for intelligent playlist generation.

---

## Performance

| Metric | Value |
|---|---|
| Average time per track | ~60–77s (CPU, i5-7Y57) |
| Model loading | ~0.5s |
| MusiCNN extraction | ~20–30s per track |
| EffNet extraction | ~20–30s per track |
| Classifier heads | <100ms each |
| 14-track batch | ~13 minutes |
| Estimated full library (861 tracks) | ~15 hours |

One-time cost — results get embedded in m4a tags and never recomputed.

---

## Librosa vs Essentia: Complementary, Not Competing

### What Essentia replaces

| Dimension | Librosa (heuristic) | Essentia (neural) | Winner |
|---|---|---|---|
| Mood (happy/sad/relaxed) | Weighted formula on spectral features | Trained classifier | **Essentia** |
| Danceability | beat_strength + tempo formula | Trained classifier | **Essentia** |
| Voice/instrumental | vocal_proxy (mid-freq energy ratio) | Trained classifier | **Essentia** (fixes piano false positives) |
| Genre | none | 400 Discogs classes | **Essentia** (no librosa equivalent) |
| Arousal/Valence | none | Regression model (1–9 scale) | **Essentia** (no librosa equivalent) |

### What librosa still provides

| Feature | Why Essentia can't replace it |
|---|---|
| Continuous feature vectors (61-dim) | Powers sphere positioning and cosine similarity |
| Tempo, key, chroma, tonnetz | Structural music theory — Essentia doesn't extract these |
| Spectral shape (centroid, flatness, flux) | Fine-grained similarity within genres |
| Speed (~15s/track vs ~60s/track) | 4x faster extraction |
| Embedded in m4a tags | Already tagged, instant on startup |

### The hybrid model

```
librosa  →  geometry (where tracks sit in feature space)
essentia →  labels (what those positions mean in human terms)
```

Both get embedded in the m4a `----:com.soniq:features` tag.
On startup, read tags → full feature set available instantly.

---

## Integration Plan

### Phase 1: Tag schema v0.3

Add Essentia outputs to the existing Soniq tag. Key design decisions:

- **Drop `s.vocal`** (vocal_proxy) — replaced by neural `instrumental`/`vocal`
- **Self-describing keys** — `danceable: 0.17` reads as "17% danceable." No mapping needed.
- **Single-float for negations** — `happy`, `sad`, `relaxed`, etc. Opposite is `1 - value`.
- **Both sides for real concepts** — `instrumental`/`vocal`, `tonal`/`atonal`, `bright`/`dark`
  are kept as pairs because both carry distinct musical meaning. Someone searches
  "dark instrumental atonal" — those are positive descriptors, not negations.
- **No derived moods in the tag** — raw data stays honest, interpretation lives in the API.
  "Melancholy" = low happy + mild sad + high relaxed — computed at query time, not stored.

```json
{
  "v": "0.3",
  "s": { "...14 librosa scalars (vocal dropped)..." },
  "vec": { "...51 floats (mfcc, contrast, chroma, tonnetz)..." },
  "e": {
    "happy": 0.01,
    "sad": 0.61,
    "relaxed": 1.00,
    "aggressive": 0.03,
    "party": 0.00,
    "acoustic": 0.38,
    "danceable": 0.17,
    "instrumental": 0.92,
    "vocal": 0.08,
    "tonal": 0.88,
    "atonal": 0.12,
    "bright": 0.49,
    "dark": 0.51,
    "arousal": 4.07,
    "valence": 3.71,
    "genre": ["Electronic---Ambient", "Electronic---Experimental", "Electronic---Drone"],
    "genre_s": [0.243, 0.133, 0.083]
  }
}
```

17 keys in `nrn`, ~1KB total per track. Fits in m4a padding, no file size increase.
Full schema spec: see `tag-schema.md`.

### Phase 2: Use in Soniq player

- **Sphere**: Use genre labels for cluster naming, mood for color coding
- **Sonic Mix**: Use neural danceability/mood instead of heuristic composites
- **Filters**: Enable "show me all danceable tracks" using Essentia labels
- **Similarity**: Experiment with EffNet 1280-dim embeddings vs librosa 61-dim

### Phase 3: Batch processing

- Run Essentia on full library (~15 hours, one-time)
- Could run on a more powerful machine and copy tags back
- Progressive: process on-demand when tracks are played, batch overnight

---

## Known Limitations

1. **mood_party underestimates party tracks** — Get Lucky scored only 52% party.
   The model may be biased toward EDM/club music rather than funk/disco.

2. **Timbre classifier is weak** — Most tracks score near 50/50 bright/dark.
   Our librosa centroid_mean is more discriminating.

3. **"Atonal" for non-Western music** — Ballaké Sissoko scored 94% atonal.
   The model interprets non-Western tuning as atonality. This is a bias in
   the training data (mostly Western music).

4. **mood_acoustic misses Prayer** — Scored non_acoustic 62% despite being
   pure acoustic piano. Possibly confused by production (reverb, mastering).
   AcousticBrainz correctly identified it as acoustic (88%).

5. **Processing time** — 60–77s per track on i5-7Y57 is acceptable for
   batch processing but too slow for real-time.

---

## Files

```
~/soniq-lab/
├── classify.py          # Main pipeline — downloads models, runs inference
├── compare.py           # Compare Essentia vs librosa features
├── results/             # Individual JSON per track (summary + raw)
│   ├── Daft Punk - Get Lucky (...).json
│   ├── GoGo Penguin - Prayer.json
│   ├── Ballaké Sissoko - Djourou.json
│   ├── Hidden Orchestra - Dust.json
│   └── Mammal Hands - *.json (10 tracks)
├── models/              # Downloaded .pb model files + metadata JSONs
├── FINDINGS.md          # This document
└── README.md            # Setup instructions
```

---

## References

- Essentia model zoo: https://essentia.upf.edu/models.html
- Discogs genre taxonomy: 400 styles across Blues, Brass, Children's, Classical,
  Electronic, Folk/World/Country, Funk/Soul, Hip Hop, Jazz, Latin, Non-Music,
  Pop, Reggae, Rock, Stage & Screen
- MusiCNN paper: Pons & Serra (2019) — "musicnn: Pre-trained convolutional
  neural networks for music audio tagging"
- AcousticBrainz (archived): https://acousticbrainz.org
