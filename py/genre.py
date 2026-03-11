"""MusicBrainz genre lookup — album + artist tag blending with noise filtering.

Strategy: album tags weighted 2x + artist tags 1x, with fallback
for collaborative releases (retry with primary artist name).
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

MB_USER_AGENT = "SoniqLab/0.4 (https://github.com/msieur-gab/soniq-lab)"
MB_BASE = "https://musicbrainz.org/ws/2"

_genre_cache = {}
_last_mb_request = 0.0

# Noise patterns to reject — instruments, ratings, magazines, cities, etc.
_GENRE_NOISE = re.compile(
    r"^\d|"
    r"stars?$|"
    r"^ph_|"
    r"thing \d|"
    r"^(piano|guitar|bass|drums?|saxophone|trumpet|violin|cello|flute|"
    r"double bass|harp|organ|synthesizer|vocals?)$|"
    r"^(manchester|london|berlin|new york|chicago|detroit|los angeles|bristol|"
    r"uk|usa|american|british|french|german|japanese|norwegian|swedish)$",
    re.IGNORECASE,
)


def init(cache_path):
    """Set the cache file path. Call once at startup."""
    global _cache_path
    _cache_path = Path(cache_path)
    _load_cache()


def _load_cache():
    global _genre_cache
    if _cache_path.exists():
        try:
            _genre_cache = json.loads(_cache_path.read_text())
        except Exception:
            _genre_cache = {}


def _save_cache():
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    _cache_path.write_text(json.dumps(_genre_cache, indent=2, ensure_ascii=False))


def _filter_genres(tags):
    return [t for t in tags if not _GENRE_NOISE.search(t)]


def _mb_get(url):
    """Rate-limited MusicBrainz API GET (1 req/sec)."""
    global _last_mb_request
    elapsed = time.time() - _last_mb_request
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _last_mb_request = time.time()
            return json.loads(resp.read())
    except Exception:
        _last_mb_request = time.time()
        return None


def fetch_album_genre(artist, album):
    """Fetch genre tags by blending album + artist tags from MusicBrainz.

    Strategy (from pipeline-v2-onnx):
      1. Search release-group by artist + album → get album tags + artist MBID
      2. Fetch artist tags (usually richer than album)
      3. Blend: album tags weighted 2x, artist tags 1x
      4. Fallback: retry with primary artist name for collaborative releases

    Returns list of genre strings sorted by blended score.
    Results are cached per artist+album key. Never raises — returns [] on error.
    """
    cache_key = f"{artist}|||{album}"
    if cache_key in _genre_cache:
        return _genre_cache[cache_key]

    try:
        genres = _filter_genres(_search_and_blend(artist, album))

        # Fallback: try primary artist name for collaborative releases
        if not genres and "," in artist:
            primary = artist.split(",")[0].strip()
            genres = _filter_genres(_search_and_blend(primary, album))
    except Exception:
        genres = []

    _genre_cache[cache_key] = genres
    _save_cache()
    return genres


def _search_and_blend(artist_q, album_q):
    query = urllib.parse.quote(f'artist:"{artist_q}" AND releasegroup:"{album_q}"')
    url = f'{MB_BASE}/release-group/?query={query}&fmt=json&limit=1'
    data = _mb_get(url)

    if not data or not data.get("release-groups"):
        return []

    rg = data["release-groups"][0]
    rg_id = rg.get("id")
    artist_id = None
    if rg.get("artist-credit"):
        artist_id = rg["artist-credit"][0]["artist"]["id"]

    # Fetch full release-group tags
    rg_tags = []
    if rg_id:
        rg_data = _mb_get(f'{MB_BASE}/release-group/{rg_id}?inc=tags&fmt=json')
        if rg_data and "tags" in rg_data:
            rg_tags = rg_data["tags"]

    # Fetch artist tags
    artist_tags = []
    if artist_id:
        a_data = _mb_get(f'{MB_BASE}/artist/{artist_id}?inc=tags&fmt=json')
        if a_data and "tags" in a_data:
            artist_tags = a_data["tags"]

    # Blend: album 2x, artist 1x
    scores = {}
    for t in rg_tags:
        count = t.get("count", 0)
        if count > 0:
            scores[t["name"]] = scores.get(t["name"], 0) + count * 2
    for t in artist_tags:
        count = t.get("count", 0)
        if count > 0:
            scores[t["name"]] = scores.get(t["name"], 0) + count

    sorted_tags = sorted(scores.items(), key=lambda x: -x[1])
    return [t[0] for t in sorted_tags]
