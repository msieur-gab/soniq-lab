#!/usr/bin/env python3
"""
Minimal API server for soniq-lab prototypes.
Serves cls data from pipeline.db on port 8878.
"""

import json
import sqlite3
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DB_PATH = Path(__file__).parent / "pipeline.db"
ROOT = Path(__file__).parent


def get_db():
    db = sqlite3.connect(str(DB_PATH), timeout=5)
    db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    return db


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/tracks":
            self._json(get_tracks())
        elif path == "/api/artists":
            self._json(get_artists())
        elif path.startswith("/music/"):
            self._serve_audio(path)
        elif path.startswith("/prototypes/") or path == "/":
            self._serve_file(path)
        else:
            self.send_error(404)

    def _serve_file(self, path):
        if path == "/":
            path = "/prototypes/index.html"
        fpath = ROOT / path.lstrip("/")
        if not fpath.exists():
            self.send_error(404)
            return
        ct = "text/html" if fpath.suffix == ".html" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(fpath.read_bytes())


    def _serve_audio(self, path):
        from urllib.parse import unquote
        rel = unquote(path[7:])  # strip /music/
        fpath = ROOT / "music" / rel
        if not fpath.exists():
            self.send_error(404)
            return
        size = fpath.stat().st_size
        # Support Range requests for seeking
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            end = size - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
        else:
            start = 0
            length = size
            self.send_response(200)
            self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", "audio/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(fpath, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))


def get_tracks():
    db = get_db()
    rows = db.execute("""
        SELECT artist, album, title, path, cls_json
        FROM tracks WHERE status='done' AND cls_json IS NOT NULL
        ORDER BY artist, album, title
    """).fetchall()
    db.close()

    tracks = []
    for r in rows:
        cls = json.loads(r["cls_json"])
        tracks.append({
            "artist": r["artist"],
            "album": r["album"],
            "title": r["title"],
            "file": r["path"],
            # cls fields flat for easy access
            **cls
        })
    return tracks


def get_artists():
    db = get_db()
    rows = db.execute("""
        SELECT DISTINCT artist FROM tracks
        WHERE status='done' ORDER BY artist
    """).fetchall()
    db.close()
    return [r["artist"] for r in rows]


if __name__ == "__main__":
    port = 8878
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("", port), Handler)
    print(f"Soniq Lab prototypes: http://localhost:{port}")
    print(f"API: http://localhost:{port}/api/tracks")
    server.serve_forever()
