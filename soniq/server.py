"""HTTP server — web UI + API + processing thread.

Serves a single-page monitor with folder selection, start/stop controls,
progress tracking, and track detail modals. Processing runs in a
background thread with graceful stop support.
"""

import json
import sqlite3
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from . import db as dbmod
from .librosa_features import extract_track_features
from .classifiers import predict_all
from .genre import init as genre_init, fetch_album_genre
from .tags import build_tag, write_tag

DEFAULT_PORT = 8877

# Shared state between server and processing thread
_state = {
    "running": False,
    "stop_requested": False,
    "folder": "",
    "db_path": "",
    "genre_cache_path": "",
    "dry_run": False,
    "default_folder": "",
    "current": None,
    "error": None,
    "started_at": None,
}
_state_lock = threading.Lock()

# Status cache — refreshed from DB periodically, served to the UI
_status_cache = {"data": None, "lock": threading.Lock()}


def _refresh_status_loop(db_path):
    """Background thread: refresh cached status every 4s."""
    while True:
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")

            stats = dbmod.get_stats(conn)
            current = dbmod.get_current(conn)
            recent = dbmod.get_recent(conn, limit=15)

            # Enrich recent with parsed classifications
            for r in recent:
                if r.get("cls_json"):
                    try:
                        r["cls"] = json.loads(r["cls_json"])
                    except Exception:
                        r["cls"] = {}

            conn.close()

            with _state_lock:
                started_at = _state.get("started_at")
                running = _state["running"]

            with _status_cache["lock"]:
                _status_cache["data"] = {
                    "stats": stats,
                    "current": current,
                    "recent": recent,
                    "running": running,
                    "started_at": started_at,
                }
        except Exception:
            pass
        time.sleep(4)


def _processing_loop():
    """Background thread: process tracks one by one until done or stopped."""
    with _state_lock:
        db_path = _state["db_path"]
        folder = _state["folder"]
        genre_cache_path = _state["genre_cache_path"]
        dry_run = _state["dry_run"]

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # Scan folder for new tracks
    new_count = dbmod.scan_folder(conn, folder)
    dbmod.log_event(conn, "scan", f"Found {new_count} new tracks in {folder}")

    # Recover any stuck tracks from previous run
    recovered = dbmod.recover_stuck(conn)
    if recovered:
        dbmod.log_event(conn, "recover", f"Reset {recovered} stuck tracks")

    # Init genre
    genre_init(genre_cache_path)

    dbmod.log_event(conn, "start", f"Processing started — folder: {folder}")

    while True:
        with _state_lock:
            if _state["stop_requested"]:
                break

        track = dbmod.get_next_pending(conn)
        if not track:
            break  # All done

        track_id = track["id"]
        dbmod.mark_processing(conn, track_id)

        with _state_lock:
            _state["current"] = {
                "artist": track["artist"],
                "album": track["album"],
                "title": track["title"],
            }

        t0 = time.time()
        try:
            # 1. Librosa features (expanded v0.5)
            features = extract_track_features(track["path"], max_duration=300)
            if not features:
                raise ValueError("Librosa extraction returned None")

            # 2. Classifications (numpy dot products, ~3ms total)
            cls = predict_all(features)

            # 3. Genre (never blocks — returns [] on error)
            genre = fetch_album_genre(track["artist"], track["album"])

            # 4. Build tag
            tag = build_tag(features, cls, genre=genre)

            elapsed = time.time() - t0

            # 5. Write tag to file
            if not dry_run:
                write_tag(track["path"], tag)

            dbmod.mark_done(conn, track_id, tag, elapsed)

            print(f"  [{track['artist']}] {track['title']} — {elapsed:.1f}s "
                  f"bright={tag['cls'].get('bright', 0):.2f}")

        except Exception as e:
            elapsed = time.time() - t0
            dbmod.mark_error(conn, track_id, str(e), elapsed)
            print(f"  [{track['artist']}] {track['title']} — ERROR: {e}")

    with _state_lock:
        _state["running"] = False
        _state["current"] = None
        _state["stop_requested"] = False

    dbmod.log_event(conn, "stop", "Processing finished")
    conn.close()
    print("Processing complete.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence request logs

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/status":
            with _status_cache["lock"]:
                data = _status_cache["data"] or {
                    "stats": {}, "current": None, "recent": [], "running": False,
                }
            with _state_lock:
                data["default_folder"] = _state["default_folder"]
            self._json(data)

        elif self.path.startswith("/api/browse"):
            # Parse ?path= parameter
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            browse_path = qs.get("path", [str(Path.home())])[0]
            p = Path(browse_path)
            if not p.is_dir():
                p = Path.home()
            try:
                dirs = sorted([
                    d.name for d in p.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ])
            except PermissionError:
                dirs = []
            self._json({
                "path": str(p),
                "parent": str(p.parent) if p != p.parent else None,
                "dirs": dirs,
            })

        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/start":
            body = self._read_body()
            folder = body.get("folder", "")

            if not folder or not Path(folder).is_dir():
                self._json({"error": "Invalid folder path"}, 400)
                return

            with _state_lock:
                if _state["running"]:
                    self._json({"error": "Already running"}, 409)
                    return
                _state["running"] = True
                _state["stop_requested"] = False
                _state["folder"] = folder
                _state["error"] = None
                _state["started_at"] = time.time()

            t = threading.Thread(target=_processing_loop, daemon=True)
            t.start()
            self._json({"ok": True})

        elif self.path == "/api/stop":
            with _state_lock:
                if _state["running"]:
                    _state["stop_requested"] = True
            self._json({"ok": True})

        else:
            self.send_error(404)


def start(db_path, genre_cache_path, port=DEFAULT_PORT, dry_run=False, default_folder=""):
    """Start the HTTP server and status refresh thread."""
    with _state_lock:
        _state["db_path"] = str(db_path)
        _state["genre_cache_path"] = str(genre_cache_path)
        _state["dry_run"] = dry_run
        _state["default_folder"] = default_folder

    # Init DB
    conn = dbmod.init_db(db_path)
    conn.close()

    # Start status refresh thread
    t = threading.Thread(target=_refresh_status_loop, args=(db_path,), daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  Soniq Lab running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with _state_lock:
            if _state["running"]:
                _state["stop_requested"] = True
                print("\nStopping after current track...")
                time.sleep(2)
        server.shutdown()
        print("Server stopped.")


# ─── Inline HTML ──────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Soniq Lab</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0f; color: #e0e0e0; padding: 24px; max-width: 900px; margin: 0 auto; }
  h1 { font-size: 18px; color: #8888ff; margin-bottom: 20px; }

  /* Controls */
  .ctrl { display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }
  .ctrl input { flex: 1; background: #14141f; border: 1px solid #2a2a3a; border-radius: 6px; padding: 8px 12px; color: #e0e0e0; font-family: inherit; font-size: 13px; }
  .ctrl input:focus { outline: none; border-color: #4444aa; }
  .ctrl button { padding: 8px 16px; border: none; border-radius: 6px; font-family: inherit; font-size: 13px; cursor: pointer; }
  #btn-start { background: #2a4a2a; color: #44aa44; }
  #btn-start:hover { background: #3a5a3a; }
  #btn-start.running { background: #4a2a2a; color: #ff6666; }
  #btn-start:disabled { opacity: 0.4; cursor: default; }

  /* Stats */
  .stats { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .s { background: #14141f; border: 1px solid #2a2a3a; border-radius: 8px; padding: 14px 20px; }
  .s b { font-size: 24px; color: #aaaaff; display: block; }
  .s span { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  .bar { background: #14141f; border-radius: 8px; height: 20px; overflow: hidden; margin-bottom: 8px; border: 1px solid #2a2a3a; }
  .bar div { height: 100%; background: linear-gradient(90deg, #4444aa, #8888ff); transition: width 0.5s; border-radius: 8px; }
  #eta { font-size: 12px; color: #666; margin-bottom: 20px; }
  #now { margin-bottom: 20px; padding: 10px 14px; background: #14141f; border: 1px solid #4444aa; border-radius: 8px; font-size: 14px; color: #aaaaff; display: none; }
  #now em { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px; font-style: normal; }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #555; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; padding: 6px 10px; border-bottom: 1px solid #2a2a3a; }
  td { padding: 5px 10px; border-bottom: 1px solid #1a1a2a; }
  .done { cursor: pointer; }
  .done:hover td { color: #aaaaff !important; }
  .done td { color: #5a5a7a; }
  .processing td { color: #aaaaff; }
  .error td { color: #ff6666; }
  .b { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
  .b-done { background: #1a2a1a; color: #44aa44; }
  .b-processing { background: #1a1a2a; color: #8888ff; }
  .b-error { background: #2a1a1a; color: #ff6666; }
  #err { color: #ff6666; font-size: 12px; margin-top: 12px; display: none; }

  /* Modal */
  .overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 100; justify-content: center; align-items: center; }
  .overlay.open { display: flex; }
  .modal { background: #14141f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 24px; max-width: 480px; width: 90%; max-height: 80vh; overflow-y: auto; }
  .modal h2 { font-size: 15px; color: #aaaaff; margin-bottom: 2px; }
  .modal h3 { font-size: 12px; color: #666; margin-bottom: 16px; font-weight: normal; }
  .modal .x { float: right; background: none; border: none; color: #666; font-size: 18px; cursor: pointer; }
  .modal .x:hover { color: #fff; }
  .row { display: flex; align-items: center; margin-bottom: 5px; }
  .row .lbl { width: 100px; font-size: 11px; color: #888; }
  .row .bg { flex: 1; height: 14px; background: #1a1a2a; border-radius: 3px; overflow: hidden; }
  .row .fg { height: 100%; border-radius: 3px; }
  .row .val { width: 40px; text-align: right; font-size: 11px; color: #888; margin-left: 6px; }
  .fg.hi { background: linear-gradient(90deg, #3a5a3a, #44aa44); }
  .fg.md { background: linear-gradient(90deg, #3a3a5a, #6666aa); }
  .fg.lo { background: linear-gradient(90deg, #2a2a2a, #444); }
  .tag { display: inline-block; padding: 2px 7px; margin: 2px; border-radius: 3px; font-size: 10px; background: #1a1a2a; color: #8888ff; }

  /* Folder browser */
  .browse-btn { padding: 8px 12px; background: #1a1a2a; border: 1px solid #2a2a3a; border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 13px; color: #8888ff; }
  .browse-btn:hover { border-color: #4444aa; background: #1f1f2f; }
  #fbrowser { background: #14141f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; max-width: 520px; width: 90%; max-height: 70vh; display: flex; flex-direction: column; }
  #fbrowser h2 { font-size: 14px; color: #aaaaff; margin-bottom: 12px; }
  #fpath { font-size: 11px; color: #666; margin-bottom: 10px; word-break: break-all; padding: 6px 10px; background: #0a0a0f; border-radius: 4px; }
  #fdirs { flex: 1; overflow-y: auto; max-height: 50vh; }
  #fdirs div { padding: 7px 12px; cursor: pointer; border-radius: 4px; font-size: 13px; color: #ccc; }
  #fdirs div:hover { background: #1a1a2a; color: #aaaaff; }
  #fdirs div::before { content: ''; display: inline-block; width: 8px; height: 8px; margin-right: 8px; border: 1px solid #555; border-radius: 2px; }
  #fdirs .up { color: #888; font-style: italic; }
  #fdirs .up::before { content: none; }
  .fbtn-row { display: flex; gap: 8px; margin-top: 12px; justify-content: flex-end; }
  .fbtn-row button { padding: 7px 16px; border: none; border-radius: 6px; font-family: inherit; font-size: 12px; cursor: pointer; }
  .fbtn-select { background: #2a4a2a; color: #44aa44; }
  .fbtn-select:hover { background: #3a5a3a; }
  .fbtn-cancel { background: #1a1a2a; color: #888; }
  .fbtn-cancel:hover { background: #2a2a3a; }
</style>
</head>
<body>
<h1>soniq lab</h1>

<div class="ctrl">
  <input type="text" id="folder" placeholder="/path/to/music/folder" />
  <button class="browse-btn" onclick="openBrowser()">Browse</button>
  <button id="btn-start" onclick="toggle()">Start</button>
</div>

<div class="stats" id="stats"></div>
<div class="bar"><div id="fill"></div></div>
<div id="eta"></div>
<div id="now"><em>Now processing</em><span id="track"></span></div>
<table>
  <thead><tr><th>Artist</th><th>Title</th><th>Status</th><th>Time</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<div id="err"></div>

<div class="overlay" id="fov" onclick="if(event.target===this)closeBrowser()">
  <div id="fbrowser">
    <button class="x" onclick="closeBrowser()" style="float:right;background:none;border:none;color:#666;font-size:18px;cursor:pointer">&times;</button>
    <h2>Select music folder</h2>
    <div id="fpath"></div>
    <div id="fdirs"></div>
    <div class="fbtn-row">
      <button class="fbtn-cancel" onclick="closeBrowser()">Cancel</button>
      <button class="fbtn-select" onclick="selectFolder()">Select this folder</button>
    </div>
  </div>
</div>

<div class="overlay" id="ov" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="x" onclick="closeModal()">&times;</button>
    <h2 id="mt"></h2>
    <h3 id="ms"></h3>
    <div id="mb"></div>
  </div>
</div>

<script>
const API = location.origin;
let fails = 0;
let isRunning = false;
let folderSet = false;

async function poll() {
  try {
    const r = await fetch(API + '/api/status');
    const d = await r.json();
    fails = 0;
    document.getElementById('err').style.display = 'none';
    render(d);
  } catch(e) {
    if (++fails > 2) {
      document.getElementById('err').style.display = 'block';
      document.getElementById('err').textContent = 'API unreachable';
    }
  }
}

function render(d) {
  // Set default folder on first load
  if (d.default_folder && !document.getElementById('folder').value && !folderSet) {
    document.getElementById('folder').value = d.default_folder;
    folderSet = true;
  }
  isRunning = d.running;
  const btn = document.getElementById('btn-start');
  btn.textContent = isRunning ? 'Stop' : 'Start';
  btn.className = isRunning ? 'running' : '';
  document.getElementById('folder').disabled = isRunning;

  const s = d.stats || {};
  document.getElementById('stats').innerHTML =
    [['Total',s.total||0],['Done',s.done||0],['Errors',s.errors||0],['Pending',s.pending||0],['Avg',s.avg_time?(s.avg_time|0)+'s':'-']]
    .map(([l,v])=>`<div class="s"><b>${v}</b><span>${l}</span></div>`).join('');

  const pct = s.total ? ((s.done||0) / s.total * 100) : 0;
  document.getElementById('fill').style.width = pct.toFixed(1) + '%';

  if (s.avg_time && s.pending > 0) {
    const sec = Math.round(s.avg_time * s.pending);
    const h = (sec / 3600) | 0, m = ((sec % 3600) / 60) | 0;
    document.getElementById('eta').textContent = pct.toFixed(1) + '% — ~' + h + 'h ' + m + 'm left';
  } else if ((s.pending||0) === 0 && (s.done||0) > 0) {
    document.getElementById('eta').textContent = '100% — Complete!';
  } else {
    document.getElementById('eta').textContent = '';
  }

  const now = document.getElementById('now');
  if (d.current) {
    now.style.display = 'block';
    document.getElementById('track').textContent = d.current.artist + ' — ' + d.current.title;
  } else {
    now.style.display = 'none';
  }

  const rows = d.recent || [];
  for (let t of rows) {
    if (t.cls) trackCache[t.artist + '|' + t.title] = t;
  }
  document.getElementById('rows').innerHTML = rows.map(t => {
    const cls = t.status === 'done' ? ' onclick="detail(\'' + esc(t.artist) + '\',\'' + esc(t.title) + '\',\'' + esc(t.album||'') + '\')"' : '';
    return '<tr class="' + t.status + '"' + cls + '><td>' + t.artist + '</td><td>' + t.title + '</td><td><span class="b b-' + t.status + '">' + t.status + '</span></td><td>' + (t.duration_s ? (t.duration_s|0) + 's' : '') + '</td></tr>';
  }).join('');
}

function esc(s) { return (s||'').replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

async function toggle() {
  if (isRunning) {
    await fetch(API + '/api/stop', {method:'POST'});
  } else {
    const folder = document.getElementById('folder').value.trim();
    if (!folder) { alert('Enter a music folder path'); return; }
    const r = await fetch(API + '/api/start', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({folder})
    });
    const d = await r.json();
    if (d.error) alert(d.error);
  }
  setTimeout(poll, 500);
}

const trackCache = {};

function detail(artist, title, album) {
  const t = trackCache[artist + '|' + title];
  if (!t || !t.cls) return;

  document.getElementById('mt').textContent = artist + ' — ' + title;
  document.getElementById('ms').textContent = album;

  const c = t.cls;
  let h = '';

  // Mood bars
  const keys = [['happy','Happy'],['sad','Sad'],['relaxed','Relaxed'],['aggressive','Aggressive'],['party','Party'],['acoustic','Acoustic'],['danceable','Danceable']];
  for (const [k, label] of keys) {
    if (c[k] === undefined) continue;
    const v = c[k], p = (v * 100)|0;
    const lvl = v > 0.6 ? 'hi' : v > 0.3 ? 'md' : 'lo';
    h += '<div class="row"><span class="lbl">' + label + '</span><div class="bg"><div class="fg ' + lvl + '" style="width:' + p + '%"></div></div><span class="val">' + p + '%</span></div>';
  }

  // Instrumental / Tonal / Brightness
  if (c.instrumental !== undefined) {
    const p = (c.instrumental * 100)|0;
    const lvl = c.instrumental > 0.6 ? 'hi' : c.instrumental > 0.3 ? 'md' : 'lo';
    h += '<div class="row"><span class="lbl">Instrumental</span><div class="bg"><div class="fg ' + lvl + '" style="width:' + p + '%"></div></div><span class="val">' + p + '%</span></div>';
  }
  if (c.tonal !== undefined) {
    const p = (c.tonal * 100)|0;
    const lvl = c.tonal > 0.6 ? 'hi' : c.tonal > 0.3 ? 'md' : 'lo';
    h += '<div class="row"><span class="lbl">Tonal</span><div class="bg"><div class="fg ' + lvl + '" style="width:' + p + '%"></div></div><span class="val">' + p + '%</span></div>';
  }
  if (c.bright !== undefined) {
    const p = (c.bright * 100)|0;
    const lvl = c.bright > 0.6 ? 'hi' : c.bright < 0.4 ? 'lo' : 'md';
    h += '<div class="row"><span class="lbl">Brightness</span><div class="bg"><div class="fg ' + lvl + '" style="width:' + p + '%"></div></div><span class="val">' + p + '%</span></div>';
  }

  // Genre tags
  if (c.genre && c.genre.length) {
    h += '<div style="margin-top:10px">';
    for (const g of c.genre) {
      h += '<span class="tag">' + g + '</span>';
    }
    h += '</div>';
  }

  // Arousal / Valence
  if (c.arousal !== undefined) {
    h += '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1a1a2a">';
    for (const [k, label] of [['arousal','Arousal'],['valence','Valence']]) {
      const v = c[k], p = (((v - 1) / 8) * 100)|0;
      h += '<div class="row"><span class="lbl">' + label + '</span><div class="bg"><div class="fg md" style="width:' + p + '%"></div></div><span class="val">' + v.toFixed(1) + '</span></div>';
    }
    h += '</div>';
  }

  document.getElementById('mb').innerHTML = h;
  document.getElementById('ov').classList.add('open');
}

function closeModal() { document.getElementById('ov').classList.remove('open'); }

// Folder browser
let browsePath = '';

async function openBrowser() {
  const current = document.getElementById('folder').value.trim();
  await browse(current || '');
  document.getElementById('fov').classList.add('open');
}

function closeBrowser() { document.getElementById('fov').classList.remove('open'); }

async function browse(path) {
  const url = path ? API + '/api/browse?path=' + encodeURIComponent(path) : API + '/api/browse';
  const r = await fetch(url);
  const d = await r.json();
  browsePath = d.path;
  document.getElementById('fpath').textContent = d.path;
  let h = '';
  if (d.parent) {
    h += '<div class="up" onclick="browse(\'' + esc(d.parent) + '\')">..</div>';
  }
  for (const name of d.dirs) {
    const full = d.path + '/' + name;
    h += '<div onclick="browse(\'' + esc(full) + '\')">' + name + '</div>';
  }
  if (!d.dirs.length && !d.parent) h = '<div style="color:#555">No folders found</div>';
  document.getElementById('fdirs').innerHTML = h;
}

function selectFolder() {
  document.getElementById('folder').value = browsePath;
  closeBrowser();
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>"""
