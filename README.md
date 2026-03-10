# Soniq Lab

Essentia neural classification pipeline for the Soniq music library.
Runs pre-trained TensorFlow models (MusiCNN + EffNet) and writes
classification tags directly into m4a files alongside existing librosa data.

By **Gab** and **Claude Code** (Anthropic CLI, Opus 4.6).

## Setup

```bash
cd ~/soniq-lab
python3 -m venv venv
source venv/bin/activate
pip install essentia-tensorflow mutagen
```

Models are downloaded automatically on first run (~25MB total).

## Music Library

Clone or symlink your music into `~/soniq-lab/music/`.
Expected structure: `music/Artist/Album/01 - Title.m4a`

Tracks should already have v0.2 Soniq tags (librosa features).
The pipeline reads existing tags, keeps structural data (tempo, key, chroma,
tonnetz), and appends neural classifications (`cls` section).

## Pipeline

The main tool. Scans the library, runs Essentia on each track, writes v0.3
tags, and tracks everything in SQLite.

```bash
source venv/bin/activate

# Full run (resumes automatically if interrupted)
python3 pipeline.py

# Test run — process only N tracks
python3 pipeline.py --limit 5

# Reset all tracks to pending and start over
python3 pipeline.py --reset

# Reset + limit (good for re-testing)
python3 pipeline.py --reset --limit 10
```

**Performance:** ~60-77s per track on i5-7Y57 (CPU). Full library of 947
tracks takes ~16-20 hours. Safe to interrupt and resume anytime.

## Web Monitor

Starts automatically with the pipeline on port **8877**.

```
http://localhost:8877
```

Shows live stats: total/done/pending/errors, progress bar, ETA,
current track being processed, and recent results. Auto-refreshes every 3s.

### API Endpoint

```
GET http://localhost:8877/api/status
```

Returns JSON:

```json
{
  "stats": {
    "total": 947,
    "done": 124,
    "errors": 0,
    "pending": 823,
    "avg_time": 68.5
  },
  "current": {
    "artist": "GoGo Penguin",
    "album": "A Humdrum Star",
    "title": "Prayer"
  },
  "recent": [
    {
      "artist": "Daft Punk",
      "title": "Get Lucky",
      "album": "Random Access Memories",
      "status": "done",
      "duration_s": 47.3
    }
  ]
}
```

## SQLite Database

All progress and results are stored in `pipeline.db`. Query it directly:

```bash
# Summary
sqlite3 pipeline.db "SELECT status, COUNT(*) FROM tracks GROUP BY status"

# All classifications for an artist
sqlite3 pipeline.db -json "SELECT title, cls_json FROM tracks WHERE artist='GoGo Penguin' AND status='done'"

# Errors
sqlite3 pipeline.db "SELECT artist, title, error FROM tracks WHERE status='error'"

# Average processing time
sqlite3 pipeline.db "SELECT AVG(duration_s) FROM tracks WHERE status='done'"

# Find all danceable tracks (cls stored as JSON)
sqlite3 pipeline.db "SELECT artist, title, json_extract(cls_json, '$.danceable') as dance FROM tracks WHERE status='done' AND json_extract(cls_json, '$.danceable') > 0.5 ORDER BY dance DESC"

# Find dark instrumental tracks
sqlite3 pipeline.db "SELECT artist, title FROM tracks WHERE status='done' AND json_extract(cls_json, '$.dark') > 0.6 AND json_extract(cls_json, '$.instrumental') > 0.7"

# Run log (start/stop/reset events)
sqlite3 pipeline.db "SELECT * FROM run_log ORDER BY at DESC LIMIT 10"
```

## Other Tools

```bash
# Single-track classification (outputs individual JSON files to results/)
python3 classify.py --track "Prayer"
python3 classify.py --limit 10

# Compare Essentia vs librosa composites (needs music-player API running)
python3 compare.py
```

## Tag Schema v0.3

Written into each m4a as MP4 atom `----:com.soniq:features`:

```json
{
  "src": "soniq",
  "v": "0.3",
  "at": "2026-03-10T14:00:00Z",
  "s": {
    "duration": 174.2,
    "tempo": 66.3,
    "key": 8,
    "mode": 0
  },
  "vec": {
    "chroma": [12 floats],
    "tonnetz": [6 floats]
  },
  "cls": {
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

See `FINDINGS.md` for full research documentation and design rationale.

## Files

```
~/soniq-lab/
├── pipeline.py      # Main pipeline — scan, classify, tag, monitor
├── pipeline.db      # SQLite progress + results
├── classify.py      # Standalone classifier (individual JSON output)
├── compare.py       # Essentia vs librosa comparison
├── models/          # Downloaded .pb model files + metadata JSONs
├── music/           # Cloned music library (m4a files with tags)
├── results/         # Individual JSON results from classify.py
├── FINDINGS.md      # Full research documentation
└── README.md        # This file
```
