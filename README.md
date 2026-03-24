# Soniq Lab

The research side of [Soniq Player](https://github.com/msieur-gab/soniq-player) — a music player that understands how music _feels_.

By **Gab** and **Claude Code** (Anthropic CLI, Opus 4.6).

## Why this exists

Industry music classification typically relies on deep learning models like
MusiCNN and EffNet — two-stage pipelines that extract embeddings then run
inference through classifier heads. These work, but they're heavy (two
separate neural backends), slow (~20s/track for embedding alone), opaque
(you can't explain _why_ a track scored 0.7 on "happy"), and the labels
they produce often reflect obscure musicological terms rather than how
listeners actually perceive music.

We started there. Ran the full Essentia pipeline on 1942 tracks. Then asked:
what if we could match or beat those results using plain audio analysis —
no embeddings, no inference, no model files?

By diving into music theory and foundational MIR research (Friberg 2014,
Grekow 2018, Yang 2008), we explored what's possible with direct audio
feature analysis. The result: **14 formula-based classifiers** built on
librosa DSP features, grounded in acoustic principles, producing human-readable
scores that address musical perception and emotion.

The formulas achieve comparable accuracy to the neural approach — with the
entire classification running in ~3ms (vs ~20s for embedding + inference),
in a fully explainable and tunable package. Each classifier is a readable
formula you can open, understand, and adjust.

**Portable by design.** Both the raw audio features and the derived
classifications are embedded directly as tags inside each audio file.
This means:
- Repopulating any audio backend is instant — just read the tags, no
  re-extraction needed
- Changing classification formulas is fast — re-run classifiers on the
  stored features without re-analyzing audio (~3ms/track vs ~5.5s/track)
- The music carries its own intelligence — no server, no database, no
  network dependency

See [DESIGN.md](DESIGN.md) for the full research story, formula details, and
validation results.

## The pipeline

1. **Feature extraction** — librosa DSP (STFT, HPSS, MFCC, chroma, tempogram, etc.) via ffmpeg audio decode (~5.5s/track)
2. **Classification** — 14 formula-based classifiers produce 0–1 scores (~3ms total)
3. **Genre lookup** — MusicBrainz artist+album tag blending with caching
4. **Tag writing** — JSON embedded in m4a custom atoms (`----:com.soniq:features`)
5. **Progress tracking** — SQLite database with scan, resume, and error recovery

## What it classifies

| Dimension | Scale | What it captures |
|-----------|-------|-----------------|
| **arousal** | calm → intense | Urgency, drive, loudness |
| **valence** | negative → positive | Tonal positivity (major key strength) |
| **happy / sad** | 0–1 | Emotion from tonal + spectral + rhythmic signals |
| **relaxed / aggressive** | 0–1 | Calm smoothness vs harsh intensity |
| **danceable** | 0–1 | Percussive drive, beat lock, tempo zone |
| **party** | 0–1 | Dance energy + brightness + loudness |
| **energetic / still** | 0–1 | Kinetic energy (pulse, impact, activity, groove) |
| **hypnotic / varied** | 0–1 | Repetitive trance vs dynamic evolution |
| **instrumental / vocal** | 0–1 | Spectral stability + MFCC shape |
| **brilliant / warm** | 0–1 | Acoustic spectral color (centroid-based) |
| **radiant / somber** | 0–1 | Perceived atmosphere (euphoric vs ominous) |
| **contemplative / restless** | 0–1 | Reflective depth vs nervous motion |

## Setup

```bash
python3 soniq.py          # auto-creates venv, installs deps, launches web UI
python3 soniq.py --port 9000
python3 soniq.py --dry-run # extract without writing tags to files
```

Dependencies (auto-installed): `librosa`, `mutagen`, `numpy`, `scipy`.
Requires `ffmpeg` on the system for audio decoding.

The web UI starts at `http://localhost:8877` with folder selection, start/stop
controls, progress tracking, and track detail modals.

## Tag schema (v0.7)

Each m4a gets a `----:com.soniq:features` atom containing:

```json
{
  "src": "soniq",
  "v": "0.7",
  "at": "2026-03-24T12:00:00Z",
  "s": { "duration": 174.2, "tempo": 66.3, "key": 8, "mode": 0, "..." : "..." },
  "vec": { "mfcc_m": [13], "chroma": [12], "tonnetz": [6] },
  "cls": {
    "happy": 0.01, "sad": 0.61, "relaxed": 1.00, "aggressive": 0.03,
    "danceable": 0.17, "party": 0.05,
    "energetic": 0.12, "still": 0.88,
    "hypnotic": 0.74, "varied": 0.26,
    "instrumental": 0.92, "vocal": 0.08,
    "brilliant": 0.35, "warm": 0.65,
    "radiant": 0.42, "somber": 0.58,
    "contemplative": 0.81, "restless": 0.19,
    "arousal": 0.25, "valence": 0.38,
    "genre": ["electronic", "ambient", "experimental"]
  }
}
```

The player reads these tags at library scan time — no server, no API,
no network needed. The classification lives with the music.

## File structure

```
soniq.py                          # Entry point — venv setup + server launch
DESIGN.md                         # Research, formulas, validation results
soniq/
├── server.py                     # HTTP server, web UI, processing thread
├── librosa_features.py           # Feature extraction (ffmpeg + librosa DSP)
├── db.py                         # SQLite progress tracking + results
├── genre.py                      # MusicBrainz genre lookup with caching
├── tags.py                       # Tag building + m4a writing (v0.7 schema)
├── validate_formulas.py          # Validation tool against reference DB
└── classifiers/
    ├── __init__.py               # predict_all() — orchestrates all classifiers
    ├── _features.py              # Feature preparation + derived features
    ├── _corpus_stats.py          # Z-score normalization stats (N=1942)
    ├── arousal.py                # Foundation: arousal
    ├── valence.py                # Foundation: valence
    ├── happy.py, sad.py          # Emotion pair
    ├── relaxed.py, aggressive.py # Emotion pair
    ├── danceable.py, party.py    # Rhythm / movement
    ├── energy.py                 # Composite: energetic/still + components
    ├── hypnotic.py               # Two-path: rhythmic vs timbral hypnotic
    ├── instrumental.py           # Stability + MFCC shape + modulation
    ├── timbre.py                 # Brilliant/warm (spectral centroid)
    ├── brightness.py             # Radiant/somber (perceived atmosphere)
    ├── contemplative.py          # Contemplative/restless (reflective depth)
    ├── acoustic.py               # [inactive] See DESIGN.md
    └── tonal.py                  # [inactive] See DESIGN.md
```
