#!/usr/bin/env python3
"""
Soniq Lab — Validation Pipeline

Builds a reference dataset for comparing timbre extraction approaches.

Sources:
  1. MusicBrainz — genre tags per artist (cached)
  2. Essentia standard — low-level spectral features (same engine as AcousticBrainz)
  3. Librosa — our timbre vectors (loaded from timbre_vectors.json)

Output: output/reference_dataset.json
  Reusable baseline for comparing v0.3, v0.4, and beyond.

Read-only: no DB or m4a files modified.
"""

import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MUSIC_DB = Path.home() / "music-player" / ".data" / ".audio_features.db"
TIMBRE_VECTORS = Path(__file__).parent / "output" / "timbre_vectors.json"
OUTPUT = Path(__file__).parent / "output" / "reference_dataset.json"
MB_CACHE = Path(__file__).parent / "output" / "musicbrainz_cache.json"

USER_AGENT = "SoniqLab/0.1 (timbre research, contact: baude.gabriel@gmail.com)"

TEST_ARTISTS = [
    "Ballaké Sissoko",
    "Four Tet",
    "GoGo Penguin",
    "Jay-Jay Johanson",
    "Hidden Orchestra",
    "Mammal Hands",
    "Grandbrothers",
    "Flying Lotus, S.Ellison",
    "Esbjörn Svensson Trio",
    "The Cinematic Orchestra",
    "Portico Quartet",
    "Jaga Jazzist",
    "Matthew Halsall",
]


# ─── MusicBrainz ───────────────────────────────────────────────

def mb_api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def load_mb_cache():
    if MB_CACHE.exists():
        with open(MB_CACHE) as f:
            return json.load(f)
    return {}


def save_mb_cache(cache):
    MB_CACHE.parent.mkdir(exist_ok=True)
    with open(MB_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_artist_tags(artist_name, cache):
    """Get genre tags for an artist from MusicBrainz."""
    # Strip collaborator names for lookup
    lookup_name = artist_name.split(",")[0].strip()

    if lookup_name in cache:
        return cache[lookup_name]

    query = urllib.parse.urlencode({
        "query": f'artist:"{lookup_name}"',
        "fmt": "json",
        "limit": 1,
    })
    url = f"https://musicbrainz.org/ws/2/artist/?{query}"
    data = mb_api_get(url)
    time.sleep(1.1)

    result = {"name": lookup_name, "tags": [], "country": None, "mbid": None}

    if "error" in data:
        result["error"] = data["error"]
        cache[lookup_name] = result
        save_mb_cache(cache)
        return result

    artists = data.get("artists", [])
    if not artists:
        result["error"] = "not found"
        cache[lookup_name] = result
        save_mb_cache(cache)
        return result

    a = artists[0]
    result["mbid"] = a.get("id")
    result["country"] = a.get("country")

    # Get tags with vote counts
    tags = a.get("tags", [])
    result["tags"] = sorted(
        [{"name": t["name"], "count": t.get("count", 0)} for t in tags],
        key=lambda x: -x["count"]
    )

    cache[lookup_name] = result
    save_mb_cache(cache)
    return result


# ─── Essentia low-level extraction ──────────────────────────────

def essentia_extract(audio_path):
    """
    Extract low-level spectral features using Essentia standard algorithms.
    Same engine as AcousticBrainz — this IS the reference.
    """
    from essentia.standard import (
        MonoLoader,
        Windowing,
        Spectrum,
        SpectralCentroidTime,
        RollOff,
        Flatness,
        Flux,
        MFCC,
        FrameGenerator,
    )

    audio = MonoLoader(filename=audio_path, sampleRate=22050)()

    window = Windowing(type="hann")
    spectrum = Spectrum()
    centroid_algo = SpectralCentroidTime(sampleRate=22050)
    rolloff_algo = RollOff(sampleRate=22050)
    flatness_algo = Flatness()
    flux_algo = Flux()
    mfcc_algo = MFCC(numberCoefficients=13, sampleRate=22050)

    centroids = []
    rolloffs = []
    flatnesses = []
    fluxes = []
    mfcc_frames = []
    prev_spectrum = None

    for frame in FrameGenerator(audio, frameSize=2048, hopSize=512):
        windowed = window(frame)
        spec = spectrum(windowed)

        centroids.append(float(centroid_algo(windowed)))
        rolloffs.append(float(rolloff_algo(spec)))
        flatnesses.append(float(flatness_algo(spec)))

        if prev_spectrum is not None:
            fluxes.append(float(flux_algo(spec)))
        prev_spectrum = spec

        _, mfcc_coeffs = mfcc_algo(spec)
        mfcc_frames.append([float(c) for c in mfcc_coeffs])

    mfcc_array = np.array(mfcc_frames)

    return {
        "centroid_mean": round(float(np.mean(centroids)), 2),
        "centroid_std": round(float(np.std(centroids)), 2),
        "rolloff_mean": round(float(np.mean(rolloffs)), 2),
        "rolloff_std": round(float(np.std(rolloffs)), 2),
        "flatness_mean": round(float(np.mean(flatnesses)), 6),
        "flux_mean": round(float(np.mean(fluxes)), 4) if fluxes else 0,
        "flux_std": round(float(np.std(fluxes)), 4) if fluxes else 0,
        "mfcc_mean": [round(float(v), 4) for v in np.mean(mfcc_array, axis=0)],
        "mfcc_std": [round(float(v), 4) for v in np.std(mfcc_array, axis=0)],
    }


# ─── Load our librosa results ──────────────────────────────────

def load_librosa_timbre():
    """Load our timbre vectors from earlier extraction."""
    if not TIMBRE_VECTORS.exists():
        return {}
    with open(TIMBRE_VECTORS) as f:
        results = json.load(f)
    # Index by artist+title for fast lookup
    index = {}
    for r in results:
        key = f"{r['artist']}|{r['title']}".lower()
        index[key] = {
            "centroid_mean": r["breakdown"]["centroid_mean"],
            "centroid_std": r["breakdown"]["centroid_std"],
            "rolloff_mean": r["breakdown"]["rolloff_mean"],
            "rolloff_std": r["breakdown"]["rolloff_std"],
            "flatness_mean": r["breakdown"]["flatness_mean"],
            "flux_mean": r["breakdown"]["flux_mean"],
            "flux_std": r["breakdown"]["flux_std"],
            "mfcc_mean": [r["breakdown"].get(f"mfcc{i}_mean") for i in range(1, 14)],
            "mfcc_std": [r["breakdown"].get(f"mfcc{i}_std") for i in range(1, 14)],
            "brightness_score": r.get("brightness_score"),
            "brightness_label": r.get("brightness_label"),
        }
    return index


# ─── Get test tracks ────────────────────────────────────────────

def get_test_tracks(limit_per_artist=3):
    db = sqlite3.connect(str(MUSIC_DB))
    tracks = []
    for artist in TEST_ARTISTS:
        rows = db.execute("""
            SELECT artist, title, album, file
            FROM tracks WHERE artist LIKE ?
            ORDER BY title LIMIT ?
        """, (f"{artist}%", limit_per_artist)).fetchall()
        tracks.extend(rows)
    db.close()
    return tracks


# ─── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Soniq Lab — Validation Pipeline")
    print("=" * 60)

    # Step 1: MusicBrainz genre tags
    print("\n[1/3] Fetching MusicBrainz genre tags...")
    mb_cache = load_mb_cache()
    artist_tags = {}
    for artist in TEST_ARTISTS:
        lookup = artist.split(",")[0].strip()
        if lookup in mb_cache:
            print(f"  {lookup:35s} (cached)")
            artist_tags[lookup] = mb_cache[lookup]
        else:
            info = fetch_artist_tags(artist, mb_cache)
            top_tags = [t["name"] for t in info.get("tags", [])[:5]]
            print(f"  {lookup:35s} → {', '.join(top_tags) or 'no tags'}")
            artist_tags[lookup] = info

    # Step 2: Essentia extraction
    print("\n[2/3] Extracting Essentia reference features...")
    test_tracks = get_test_tracks(limit_per_artist=3)
    print(f"  {len(test_tracks)} tracks to process\n")

    essentia_results = {}
    for i, (artist, title, album, filepath) in enumerate(test_tracks):
        audio_path = filepath
        if not Path(audio_path).exists():
            audio_path = str(Path.home() / "music-player" / "music" / filepath)
        if not Path(audio_path).exists():
            print(f"  [{i+1}/{len(test_tracks)}] SKIP — {filepath}")
            continue

        t0 = time.time()
        try:
            features = essentia_extract(audio_path)
            elapsed = time.time() - t0
            key = f"{artist}|{title}".lower()
            essentia_results[key] = features
            print(f"  [{i+1}/{len(test_tracks)}] {artist:35s} {title:25s} centroid={features['centroid_mean']:7.1f} Hz  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(test_tracks)}] ERROR: {artist} — {title}: {e}")

    # Step 3: Load librosa
    print("\n[3/3] Loading librosa timbre vectors...")
    librosa_index = load_librosa_timbre()
    print(f"  {len(librosa_index)} tracks loaded")

    # Build reference dataset
    print("\n" + "=" * 60)
    print("Building reference dataset...")

    dataset = []
    for artist, title, album, filepath in test_tracks:
        key = f"{artist}|{title}".lower()
        lookup_artist = artist.split(",")[0].strip()

        entry = {
            "artist": artist,
            "title": title,
            "album": album,
            "musicbrainz": artist_tags.get(lookup_artist),
            "essentia": essentia_results.get(key),
            "librosa": librosa_index.get(key),
        }
        dataset.append(entry)

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  → {OUTPUT}")

    # ─── Analysis ───────────────────────────────────────────────

    # Centroid comparison
    print(f"\n{'='*60}")
    print("CENTROID COMPARISON: Librosa vs Essentia (Hz)")
    print(f"{'='*60}")
    print(f"  {'Artist':35s} {'Title':25s} {'Librosa':>9s} {'Essentia':>9s} {'Ratio':>7s}")
    print(f"  {'-'*35} {'-'*25} {'-'*9} {'-'*9} {'-'*7}")

    ratios = []
    pairs = []
    for e in dataset:
        if e["essentia"] and e["librosa"]:
            lc = e["librosa"]["centroid_mean"]
            ec = e["essentia"]["centroid_mean"]
            ratio = lc / ec if ec > 0 else 0
            ratios.append(ratio)
            pairs.append((lc, ec))
            print(f"  {e['artist']:35s} {e['title']:25s} {lc:9.1f} {ec:9.1f} {ratio:7.2f}x")

    if ratios:
        print(f"\n  Mean ratio (librosa/essentia): {np.mean(ratios):.2f}x")
        print(f"  Std ratio:                     {np.std(ratios):.2f}")

        # Pearson correlation
        if len(pairs) >= 3:
            lvals = [p[0] for p in pairs]
            evals = [p[1] for p in pairs]
            corr = np.corrcoef(lvals, evals)[0, 1]
            print(f"  Pearson correlation:            {corr:.4f}")

    # Rank comparison
    print(f"\n{'='*60}")
    print("RANK ORDER: darkest → brightest")
    print(f"{'='*60}")

    ranked_librosa = sorted(
        [(e["artist"], e["title"], e["librosa"]["centroid_mean"])
         for e in dataset if e["librosa"]],
        key=lambda x: x[2]
    )
    ranked_essentia = sorted(
        [(e["artist"], e["title"], e["essentia"]["centroid_mean"])
         for e in dataset if e["essentia"]],
        key=lambda x: x[2]
    )

    print(f"\n  {'Rank':>4s}  {'Librosa':65s}  {'Essentia':65s}")
    print(f"  {'-'*4}  {'-'*65}  {'-'*65}")
    maxlen = max(len(ranked_librosa), len(ranked_essentia))
    for i in range(maxlen):
        l_str = ""
        e_str = ""
        if i < len(ranked_librosa):
            a, t, c = ranked_librosa[i]
            l_str = f"{a[:25]:25s} {t[:25]:25s} {c:7.1f} Hz"
        if i < len(ranked_essentia):
            a, t, c = ranked_essentia[i]
            e_str = f"{a[:25]:25s} {t[:25]:25s} {c:7.1f} Hz"
        print(f"  {i+1:4d}  {l_str:65s}  {e_str:65s}")

    # Spearman rank correlation
    if len(pairs) >= 5:
        from scipy.stats import spearmanr
        lvals = [p[0] for p in pairs]
        evals = [p[1] for p in pairs]
        rho, pval = spearmanr(lvals, evals)
        print(f"\n  Spearman rank correlation: ρ = {rho:.4f} (p = {pval:.6f})")

    # MFCC comparison
    print(f"\n{'='*60}")
    print("MFCC[1] COMPARISON (spectral slope)")
    print(f"{'='*60}")
    for e in dataset:
        if e["essentia"] and e["librosa"]:
            l_mfcc = e["librosa"]["mfcc_mean"]
            e_mfcc = e["essentia"]["mfcc_mean"]
            if l_mfcc and e_mfcc and len(l_mfcc) >= 2 and len(e_mfcc) >= 2:
                print(f"  {e['artist']:35s} librosa={l_mfcc[1]:8.2f}  essentia={e_mfcc[1]:8.2f}")

    print(f"\n{'='*60}")
    print(f"Reference dataset saved: {OUTPUT}")
    print(f"MusicBrainz cache saved: {MB_CACHE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
