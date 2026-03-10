#!/usr/bin/env python3
"""
Soniq Lab — Full classification pipeline with SQLite progress tracking.

Scans music/ for m4a files, reads existing v0.2 librosa tags,
runs Essentia classification, writes v0.3 tags (structural + harmonic + cls).
Serves a small web UI on port 8877 to monitor progress.
"""

import json
import os
import sqlite3
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress Essentia's noisy "No network created" warnings
import logging
logging.getLogger("essentia").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*No network created.*")

MUSIC_ROOT = Path(__file__).parent / "music"
MODELS_DIR = Path(__file__).parent / "models"
DB_PATH = Path(__file__).parent / "pipeline.db"
SCHEMA_VERSION = "0.3"

# --- SQLite setup ---

def init_db():
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            artist TEXT,
            album TEXT,
            title TEXT,
            status TEXT DEFAULT 'pending',
            tag_v TEXT,
            cls_json TEXT,
            error TEXT,
            duration_s REAL,
            created_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            detail TEXT,
            at TEXT DEFAULT (datetime('now'))
        )
    """)
    db.commit()
    return db


def scan_tracks(db):
    """Find all m4a files and insert into DB if not already present."""
    count_new = 0
    for root, dirs, files in os.walk(MUSIC_ROOT):
        for f in sorted(files):
            if not f.endswith(".m4a"):
                continue
            path = str(Path(root) / f)
            rel = Path(path).relative_to(MUSIC_ROOT)
            parts = rel.parts
            artist = parts[0] if len(parts) >= 3 else "Unknown"
            album = parts[1] if len(parts) >= 3 else "Unknown"
            title = f.rsplit(".", 1)[0]
            # Strip track numbers
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break

            try:
                db.execute(
                    "INSERT OR IGNORE INTO tracks (path, artist, album, title) VALUES (?, ?, ?, ?)",
                    (path, artist, album, title)
                )
                count_new += db.total_changes
            except sqlite3.IntegrityError:
                pass

    db.commit()
    return count_new


def read_existing_tag(path):
    """Read the existing soniq tag from an m4a file."""
    try:
        from mutagen.mp4 import MP4
        audio = MP4(path)
        atom = "----:com.soniq:features"
        if atom not in audio.tags:
            return None
        val = audio.tags[atom][0]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return json.loads(val)
    except Exception:
        return None


def write_tag(path, tag_data):
    """Write the v0.3 tag to an m4a file."""
    from mutagen.mp4 import MP4
    audio = MP4(path)
    atom = "----:com.soniq:features"
    tag_json = json.dumps(tag_data, separators=(",", ":"))
    audio.tags[atom] = [tag_json.encode("utf-8")]
    audio.save()


def build_v03_tag(existing_tag, cls_results):
    """Build a v0.3 tag from existing librosa data + Essentia classifications."""
    tag = {
        "src": "soniq",
        "v": SCHEMA_VERSION,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Keep structural scalars from existing tag
    if existing_tag and "s" in existing_tag:
        old_s = existing_tag["s"]
        tag["s"] = {
            "duration": old_s.get("duration", 0),
            "tempo": old_s.get("tempo", 0),
            "key": old_s.get("key", 0),
            "mode": old_s.get("mode", 0),
        }
    else:
        tag["s"] = {"duration": 0, "tempo": 0, "key": 0, "mode": 0}

    # Keep harmonic vectors from existing tag
    if existing_tag and "vec" in existing_tag:
        old_vec = existing_tag["vec"]
        tag["vec"] = {}
        if "chroma" in old_vec:
            tag["vec"]["chroma"] = old_vec["chroma"]
        if "tonnetz" in old_vec:
            tag["vec"]["tonnetz"] = old_vec["tonnetz"]
    else:
        tag["vec"] = {}

    # Add classifications
    preds = cls_results
    tag["cls"] = {}

    # Binary classifiers — store positive class probability
    # Labels verified from official Essentia metadata JSONs
    binary_map = {
        "mood_happy":       ("happy", 0),       # index 0 = happy
        "mood_sad":         ("sad", 1),          # index 1 = sad
        "mood_relaxed":     ("relaxed", 1),      # index 1 = relaxed
        "mood_aggressive":  ("aggressive", 0),   # index 0 = aggressive
        "mood_party":       ("party", 1),        # index 1 = party
        "mood_acoustic":    ("acoustic", 0),     # index 0 = acoustic
        "danceability":     ("danceable", 0),     # index 0 = danceable
    }

    for pred_key, (cls_key, pos_idx) in binary_map.items():
        vals = preds.get(pred_key)
        if isinstance(vals, list) and len(vals) == 2:
            tag["cls"][cls_key] = round(vals[pos_idx], 4)

    # Dual-concept pairs — store both sides
    dual_map = {
        "voice_instrumental": [("instrumental", 0), ("vocal", 1)],
        "tonal_atonal":       [("tonal", 0), ("atonal", 1)],
        "timbre":             [("bright", 0), ("dark", 1)],
    }

    for pred_key, pairs in dual_map.items():
        vals = preds.get(pred_key)
        if isinstance(vals, list) and len(vals) == 2:
            for cls_key, idx in pairs:
                tag["cls"][cls_key] = round(vals[idx], 4)

    # Arousal / valence
    av = preds.get("arousal_valence")
    if isinstance(av, list) and len(av) >= 2:
        tag["cls"]["arousal"] = round(av[0], 2)
        tag["cls"]["valence"] = round(av[1], 2)

    # Genre top 3
    genre_vals = preds.get("genre")
    if isinstance(genre_vals, list):
        # Load genre labels
        meta_path = MODELS_DIR / "genre_discogs400-discogs-effnet-1.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            labels = meta.get("classes", [])
            if len(labels) == len(genre_vals):
                pairs = sorted(zip(labels, genre_vals), key=lambda x: -x[1])[:3]
                tag["cls"]["genre"] = [p[0] for p in pairs]
                tag["cls"]["genre_s"] = [round(p[1], 4) for p in pairs]

    return tag


# --- Essentia pipeline ---

_pipeline = None

def get_pipeline():
    """Lazy-load Essentia models (once)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from classify import build_pipeline, download_models, check_imports
    if not check_imports():
        sys.exit(1)
    download_models()
    extractors, classifiers, labels = build_pipeline()
    _pipeline = (extractors, classifiers, labels)
    return _pipeline


def classify_single(audio_path):
    """Run all classifiers on a single track."""
    from classify import classify_track
    extractors, classifiers, labels = get_pipeline()
    return classify_track(audio_path, extractors, classifiers)


# --- Web monitor ---





class MonitorHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                html = (Path(__file__).parent / "monitor.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            except FileNotFoundError:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<pre>monitor.html not found - use the API at /api/status</pre>")
        elif self.path == "/api/status":
            data = json.dumps(get_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/api/track?"):
            data = json.dumps(get_track(self.path)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def get_track(path):
    from urllib.parse import urlparse, parse_qs
    params = parse_qs(urlparse(path).query)
    artist = params.get("artist", [""])[0]
    title = params.get("title", [""])[0]

    db = sqlite3.connect(str(DB_PATH), timeout=5)
    db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT artist, album, title, cls_json, duration_s, error FROM tracks WHERE artist=? AND title=? AND status='done' LIMIT 1",
        (artist, title)
    ).fetchone()
    db.close()

    if not row:
        return {"error": "not found"}

    result = dict(row)
    if result.get("cls_json"):
        result["cls"] = json.loads(result["cls_json"])
    del result["cls_json"]
    return result


_status_cache = {"data": None, "lock": threading.Lock()}

def _refresh_status_cache():
    """Background thread: refresh cached status every 4s so API never hits DB."""
    while True:
        try:
            db = sqlite3.connect(str(DB_PATH), timeout=5)
            db.execute("PRAGMA journal_mode=WAL")
            db.row_factory = sqlite3.Row

            stats = {}
            stats["total"] = db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            stats["done"] = db.execute("SELECT COUNT(*) FROM tracks WHERE status='done'").fetchone()[0]
            stats["errors"] = db.execute("SELECT COUNT(*) FROM tracks WHERE status='error'").fetchone()[0]
            stats["pending"] = db.execute("SELECT COUNT(*) FROM tracks WHERE status='pending'").fetchone()[0]
            stats["avg_time"] = db.execute("SELECT AVG(duration_s) FROM tracks WHERE status='done' AND duration_s > 0").fetchone()[0]

            current = db.execute("SELECT artist, album, title FROM tracks WHERE status='processing' LIMIT 1").fetchone()
            current = dict(current) if current else None

            recent = []
            if current:
                recent.append({**current, "status": "processing", "duration_s": None})
            for r in db.execute("SELECT artist, album, title, status, duration_s, cls_json FROM tracks WHERE status IN ('done','error') ORDER BY processed_at DESC LIMIT 10"):
                row = dict(r)
                if row.get("cls_json"):
                    row["cls"] = json.loads(row.pop("cls_json"))
                else:
                    row.pop("cls_json", None)
                recent.append(row)

            db.close()

            with _status_cache["lock"]:
                _status_cache["data"] = {"stats": stats, "current": current, "recent": recent}
        except Exception:
            pass
        time.sleep(4)

def get_status():
    with _status_cache["lock"]:
        return _status_cache["data"] or {"stats": {"total": 0, "done": 0, "errors": 0, "pending": 0, "avg_time": None}, "current": None, "recent": []}


def start_monitor(port=8877):
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("", port), MonitorHandler)
    threading.Thread(target=_refresh_status_cache, daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"  Monitor: http://localhost:{port}")
    return server


# --- Main processing loop ---

def process_all(limit=0, reset=False):
    db = init_db()

    if reset:
        db.execute("UPDATE tracks SET status='pending', cls_json=NULL, error=NULL, duration_s=NULL, processed_at=NULL")
        db.execute("INSERT INTO run_log (event, detail) VALUES ('reset', 'all tracks reset to pending')")
        db.commit()
        print("Reset all tracks to pending.")

    print("Scanning music library...")
    scan_tracks(db)

    total = db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM tracks WHERE status='pending'").fetchone()[0]
    done = db.execute("SELECT COUNT(*) FROM tracks WHERE status='done'").fetchone()[0]

    # Fix any stuck "processing" tracks from interrupted runs
    stuck = db.execute("SELECT COUNT(*) FROM tracks WHERE status='processing'").fetchone()[0]
    if stuck:
        db.execute("UPDATE tracks SET status='pending' WHERE status='processing'")
        db.commit()
        pending += stuck
        print(f"  Recovered {stuck} stuck tracks from previous interrupted run")

    print(f"  {total} tracks total, {done} already done, {pending} pending")

    if pending == 0:
        print("Nothing to process!")
        return

    if limit:
        pending = min(pending, limit)
        print(f"  Limiting to {limit} tracks")

    print("\nLoading Essentia models...")
    get_pipeline()

    print(f"\nStarting classification of {pending} tracks...")
    db.execute("INSERT INTO run_log (event, detail) VALUES ('start', ?)",
               (f"{pending} tracks pending",))
    db.commit()

    server = start_monitor()

    processed = 0
    total_time = 0

    while True:
        # Get next pending track
        row = db.execute(
            "SELECT id, path, artist, album, title FROM tracks WHERE status='pending' ORDER BY artist, album, title LIMIT 1"
        ).fetchone()

        if not row:
            break

        if limit and processed >= limit:
            break

        track_id, path, artist, album, title = row
        processed += 1

        print(f"\n[{done + processed}/{total}] {artist} — {title}")

        # Mark as processing
        db.execute("UPDATE tracks SET status='processing' WHERE id=?", (track_id,))
        db.commit()

        t0 = time.time()
        try:
            # Read existing tag
            existing = read_existing_tag(path)
            tag_v = existing.get("v", "?") if existing else "none"

            # Run Essentia
            preds = classify_single(path)
            elapsed = time.time() - t0
            total_time += elapsed

            # Build v0.3 tag
            new_tag = build_v03_tag(existing, preds)

            # Write tag to file
            write_tag(path, new_tag)

            # Store in DB
            cls_json = json.dumps(new_tag.get("cls", {}))
            db.execute("""
                UPDATE tracks SET
                    status='done', tag_v=?, cls_json=?, duration_s=?,
                    processed_at=datetime('now')
                WHERE id=?
            """, (SCHEMA_VERSION, cls_json, round(elapsed, 2), track_id))
            db.commit()

            avg = total_time / processed
            remaining = pending - processed
            eta_min = int(avg * remaining / 60)
            print(f"  Done in {elapsed:.1f}s (avg {avg:.1f}s, ETA ~{eta_min}m)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            db.execute("""
                UPDATE tracks SET status='error', error=?, duration_s=?,
                    processed_at=datetime('now')
                WHERE id=?
            """, (str(e), round(elapsed, 2), track_id))
            db.commit()

    # Done
    db.execute("INSERT INTO run_log (event, detail) VALUES ('complete', ?)",
               (f"{processed} processed in {total_time:.0f}s",))
    db.commit()

    avg = total_time / processed if processed else 0
    print(f"\n{'='*60}")
    print(f"Complete! {processed} tracks in {total_time:.0f}s ({avg:.1f}s/track)")
    print(f"Database: {DB_PATH}")
    print(f"Monitor still running at http://localhost:8877")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Soniq Lab — Classification Pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N tracks (0=all)")
    parser.add_argument("--reset", action="store_true", help="Reset all tracks to pending")
    args = parser.parse_args()
    process_all(limit=args.limit, reset=args.reset)
