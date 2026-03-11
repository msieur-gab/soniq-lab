#!/usr/bin/env python3
"""
Validate our timbre extraction against AcousticBrainz reference data.

Flow:
  1. Look up MusicBrainz recording IDs for test tracks
  2. Query AcousticBrainz low-level API for spectral features
  3. Compare their centroid/MFCCs against ours
  4. Output side-by-side in output/validation.json

Read-only: touches no DB or m4a files.
Writes: output/validation.json only.

Rate limits: MusicBrainz = 1 req/sec, AcousticBrainz = 1 req/sec
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

OUTPUT = Path(__file__).parent / "output" / "validation.json"
TIMBRE_VECTORS = Path(__file__).parent / "output" / "timbre_vectors.json"

USER_AGENT = "SoniqLab/0.1 (timbre research, contact: baude.gabriel@gmail.com)"

# Same test tracks — use a subset to be respectful of rate limits
TEST_QUERIES = [
    ("Ballaké Sissoko", "Akilimaya"),
    ("Ballaké Sissoko", "An Badidjo"),
    ("Four Tet", "And They All Look Broken Hearted"),
    ("Four Tet", "As Serious As Your Life"),
    ("GoGo Penguin", "A Hundred Moons"),
    ("GoGo Penguin", "Akasthesia"),
    ("Jay-Jay Johanson", "Alone Again"),
    ("Hidden Orchestra", "Antiphon"),
    ("Mammal Hands", "Alia's Abandon"),
    ("Mammal Hands", "A Thread in the Dark"),
    ("Grandbrothers", "1202"),
    ("Grandbrothers", "Alice"),
    ("Esbjörn Svensson Trio", "Beggar's Blanket"),
    ("Portico Quartet", "A Luminous Beam"),
    ("Portico Quartet", "A.O.E"),
    ("Jaga Jazzist", "Apex"),
    ("The Cinematic Orchestra", "A Caged Bird"),
    ("Matthew Halsall", "A Japanese Garden in Ethiopia"),
    ("Flying Lotus", "Aht Uh Mi Hed"),
    ("Flying Lotus", "Behold the Day"),
]


def api_get(url):
    """GET with User-Agent header and rate limiting."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def search_musicbrainz(artist, title):
    """Search MusicBrainz for a recording MBID."""
    query = f'recording:"{title}" AND artist:"{artist}"'
    url = (
        "https://musicbrainz.org/ws/2/recording?"
        + urllib.parse.urlencode({"query": query, "fmt": "json", "limit": 3})
    )
    data = api_get(url)
    time.sleep(1.1)  # rate limit

    if "error" in data:
        return None, data["error"]

    recordings = data.get("recordings", [])
    if not recordings:
        return None, "no results"

    # Pick best match
    for rec in recordings:
        mbid = rec.get("id")
        rec_title = rec.get("title", "")
        rec_artist = ""
        for ac in rec.get("artist-credit", []):
            rec_artist += ac.get("name", "")
        return mbid, f"{rec_artist} — {rec_title}"

    return None, "no match"


def get_acousticbrainz_lowlevel(mbid):
    """Get low-level features from AcousticBrainz."""
    url = f"https://acousticbrainz.org/api/v1/{mbid}/low-level"
    data = api_get(url)
    time.sleep(1.1)  # rate limit
    return data


def extract_ab_timbre(lowlevel):
    """Extract timbre-relevant features from AcousticBrainz response."""
    if "error" in lowlevel:
        return {"error": lowlevel["error"]}

    ll = lowlevel.get("lowlevel", {})
    result = {}

    # Spectral centroid
    sc = ll.get("spectral_centroid", {})
    if sc:
        result["centroid_mean"] = sc.get("mean")
        result["centroid_var"] = sc.get("var")

    # Spectral rolloff
    sr = ll.get("spectral_rolloff", {})
    if sr:
        result["rolloff_mean"] = sr.get("mean")

    # Spectral flatness (dissonance is related)
    for key in ["spectral_flatness_db", "dissonance"]:
        feat = ll.get(key, {})
        if feat:
            result[f"{key}_mean"] = feat.get("mean")

    # Spectral flux
    sf = ll.get("spectral_flux", {})
    if sf:
        result["flux_mean"] = sf.get("mean")

    # MFCCs
    mfcc = ll.get("mfcc", {})
    if mfcc:
        mfcc_mean = mfcc.get("mean", [])
        if mfcc_mean:
            result["mfcc_mean"] = [round(v, 4) for v in mfcc_mean[:13]]

    # Spectral contrast
    sc_bands = ll.get("spectral_contrast_coeffs", {})
    if sc_bands:
        result["contrast_mean"] = sc_bands.get("mean", [])

    return result


def load_our_timbre(artist, title):
    """Load our timbre vector results for comparison."""
    if not TIMBRE_VECTORS.exists():
        return None

    with open(TIMBRE_VECTORS) as f:
        results = json.load(f)

    for r in results:
        if (artist.lower() in r["artist"].lower() and
                title.lower() in r["title"].lower()):
            return {
                "centroid_mean": r["breakdown"]["centroid_mean"],
                "centroid_std": r["breakdown"]["centroid_std"],
                "rolloff_mean": r["breakdown"]["rolloff_mean"],
                "flatness_mean": r["breakdown"]["flatness_mean"],
                "flux_mean": r["breakdown"]["flux_mean"],
                "mfcc_mean": [r["breakdown"].get(f"mfcc{i}_mean") for i in range(1, 14)],
                "brightness_score": r.get("brightness_score"),
                "brightness_label": r.get("brightness_label"),
            }
    return None


def main():
    print("Timbre validation: our librosa vs AcousticBrainz\n")
    print(f"Querying {len(TEST_QUERIES)} tracks (rate limited, ~2s per track)...\n")

    results = []
    found = 0
    not_found = 0

    for i, (artist, title) in enumerate(TEST_QUERIES):
        print(f"  [{i+1}/{len(TEST_QUERIES)}] {artist} — {title}")

        # Step 1: Find MBID
        mbid, match_info = search_musicbrainz(artist, title)
        if not mbid:
            print(f"    MusicBrainz: not found ({match_info})")
            not_found += 1
            results.append({
                "artist": artist,
                "title": title,
                "mbid": None,
                "match": match_info,
                "acousticbrainz": None,
                "ours": load_our_timbre(artist, title),
            })
            continue

        print(f"    MBID: {mbid} ({match_info})")

        # Step 2: Get AcousticBrainz features
        ab_raw = get_acousticbrainz_lowlevel(mbid)
        ab_timbre = extract_ab_timbre(ab_raw)

        if "error" in ab_timbre:
            print(f"    AcousticBrainz: {ab_timbre['error']}")
            not_found += 1
        else:
            ab_centroid = ab_timbre.get("centroid_mean", "?")
            print(f"    AB centroid: {ab_centroid}")
            found += 1

        # Step 3: Load our data
        ours = load_our_timbre(artist, title)
        our_centroid = ours["centroid_mean"] if ours else "?"
        print(f"    Our centroid: {our_centroid}")

        results.append({
            "artist": artist,
            "title": title,
            "mbid": mbid,
            "match": match_info,
            "acousticbrainz": ab_timbre if "error" not in ab_timbre else None,
            "ours": ours,
        })

    # Save
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{len(results)} tracks queried, {found} with AB data, {not_found} missing")
    print(f"Results → {OUTPUT}")

    # === Side-by-side comparison ===
    print(f"\n=== Centroid comparison (Hz) ===")
    print(f"  {'Artist':35s} {'Title':30s} {'Ours':>8s} {'AB':>8s} {'Diff%':>7s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*8} {'-'*7}")

    diffs = []
    for r in results:
        if not r["acousticbrainz"] or not r["ours"]:
            continue
        our_c = r["ours"]["centroid_mean"]
        ab_c = r["acousticbrainz"].get("centroid_mean")
        if our_c and ab_c:
            diff_pct = ((our_c - ab_c) / ab_c) * 100
            diffs.append(abs(diff_pct))
            print(f"  {r['artist']:35s} {r['title']:30s} {our_c:8.1f} {ab_c:8.1f} {diff_pct:+6.1f}%")

    if diffs:
        print(f"\n  Mean absolute difference: {sum(diffs)/len(diffs):.1f}%")
        print(f"  Max absolute difference:  {max(diffs):.1f}%")

    # === MFCC comparison ===
    print(f"\n=== MFCC[1] comparison (spectral slope) ===")
    for r in results:
        if not r["acousticbrainz"] or not r["ours"]:
            continue
        our_mfcc = r["ours"].get("mfcc_mean", [])
        ab_mfcc = r["acousticbrainz"].get("mfcc_mean", [])
        if len(our_mfcc) >= 2 and len(ab_mfcc) >= 2:
            print(f"  {r['artist']:35s} ours={our_mfcc[1]:8.2f}  AB={ab_mfcc[1]:8.2f}")


if __name__ == "__main__":
    main()
