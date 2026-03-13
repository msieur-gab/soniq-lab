"""SQLite database — progress tracking, results storage, resume capability.

Database: soniq.db (separate from pipeline.db which is preserved for comparison).
WAL mode for concurrent reads from the web UI thread.
"""

import json
import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "0.5"


def init_db(db_path):
    """Initialize database and return connection."""
    db = sqlite3.connect(str(db_path), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")

    db.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            artist TEXT,
            album TEXT,
            title TEXT,
            status TEXT DEFAULT 'pending',
            tag_version TEXT,
            scalars_json TEXT,
            vectors_json TEXT,
            cls_json TEXT,
            genre_json TEXT,
            radiant REAL,
            error TEXT,
            duration_s REAL,
            created_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
        CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);

        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            detail TEXT,
            at TEXT DEFAULT (datetime('now'))
        );
    """)
    db.commit()
    return db


def scan_folder(db, music_root):
    """Walk music directory and insert new tracks. Returns count of new tracks."""
    music_root = Path(music_root)
    count = 0
    for root, dirs, files in os.walk(music_root):
        for f in sorted(files):
            if not f.endswith(".m4a"):
                continue
            path = Path(root) / f
            rel = path.relative_to(music_root)
            parts = rel.parts
            artist = parts[0] if len(parts) >= 1 else "Unknown"
            album = parts[1] if len(parts) >= 2 else "Unknown"
            title = f.rsplit(".", 1)[0]
            # Strip track number prefix
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break
            try:
                db.execute(
                    "INSERT OR IGNORE INTO tracks (path, artist, album, title) VALUES (?, ?, ?, ?)",
                    (str(path), artist, album, title),
                )
                count += db.total_changes  # approximate
            except sqlite3.IntegrityError:
                pass
    db.commit()
    return count


def get_next_pending(db):
    """Get next track to process. Returns dict or None."""
    row = db.execute(
        "SELECT * FROM tracks WHERE status='pending' ORDER BY artist, album, title LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def mark_processing(db, track_id):
    db.execute("UPDATE tracks SET status='processing' WHERE id=?", (track_id,))
    db.commit()


def mark_done(db, track_id, tag, duration_s):
    """Mark track as done and store results from the tag dict."""
    db.execute("""
        UPDATE tracks SET
            status='done',
            tag_version=?,
            scalars_json=?,
            vectors_json=?,
            cls_json=?,
            genre_json=?,
            radiant=?,
            duration_s=?,
            processed_at=datetime('now')
        WHERE id=?
    """, (
        tag.get("v"),
        json.dumps(tag.get("s", {})),
        json.dumps(tag.get("vec", {})),
        json.dumps(tag.get("cls", {})),
        json.dumps(tag.get("cls", {}).get("genre", [])),
        tag.get("cls", {}).get("radiant"),
        duration_s,
        track_id,
    ))
    db.commit()


def mark_error(db, track_id, error_msg, duration_s=0):
    db.execute("""
        UPDATE tracks SET
            status='error',
            error=?,
            duration_s=?,
            processed_at=datetime('now')
        WHERE id=?
    """, (str(error_msg)[:500], duration_s, track_id))
    db.commit()


def recover_stuck(db):
    """Reset any tracks stuck in 'processing' state back to 'pending'."""
    cursor = db.execute("UPDATE tracks SET status='pending' WHERE status='processing'")
    db.commit()
    return cursor.rowcount


def get_stats(db):
    """Get processing statistics."""
    row = db.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) as processing,
            AVG(CASE WHEN status='done' THEN duration_s END) as avg_time
        FROM tracks
    """).fetchone()
    if not row:
        return {"total": 0, "done": 0, "errors": 0, "pending": 0, "processing": 0, "avg_time": None}
    d = dict(row)
    # SQLite SUM returns None when no rows match — normalize to 0
    for k in ("total", "done", "errors", "pending", "processing"):
        if d[k] is None:
            d[k] = 0
    return d


def get_current(db):
    """Get currently processing track."""
    row = db.execute(
        "SELECT * FROM tracks WHERE status='processing' LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_recent(db, limit=15):
    """Get recently processed tracks."""
    rows = db.execute(
        "SELECT * FROM tracks WHERE status IN ('done', 'error') ORDER BY processed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def log_event(db, event, detail=""):
    db.execute("INSERT INTO run_log (event, detail) VALUES (?, ?)", (event, detail))
    db.commit()
