#!/usr/bin/env python3
"""
Timbre proxy comparison: librosa features vs EffNet bright/dark.

Tests multiple approaches to deriving a timbre (bright/dark) signal
from librosa features, and compares against EffNet's classification.

Key insight from Gab: centroid_mean alone is weak. A proper timbre
proxy needs a feature vector combining centroid with flatness, flux,
ZCR, and MFCCs (1-13). Centroid without variance is especially
limited — two tracks can share the same mean but sound completely
different (dynamically bright vs uniformly bright).

Available librosa features:
  - centroid_mean (but NO centroid_std — gap)
  - flatness_mean
  - spectral_flux
  - zcr_mean
  - mfcc_mean (13 dims) + mfcc_std (13 dims)
  - contrast_mean (7 bands)

Output: output/timbre_comparison.json
"""

import json
import math
import sqlite3
from pathlib import Path

MUSIC_DB = Path.home() / "music-player" / ".data" / ".audio_features.db"
PIPELINE_DB = Path(__file__).parent / "pipeline.db"
OUTPUT = Path(__file__).parent / "output" / "timbre_comparison.json"

TEST_ARTISTS = [
    "Ballaké Sissoko",           # acoustic kora — should be dark/warm
    "Four Tet",                  # electronic — should be brighter
    "GoGo Penguin",              # acoustic trio + electronic edge
    "Jay-Jay Johanson",          # vocals, production
    "Hidden Orchestra",          # cinematic, layered
    "Mammal Hands",              # sax/piano/drums, acoustic jazz
    "Grandbrothers",             # prepared piano + electronics
    "Flying Lotus, S.Ellison",   # electronic, bass-heavy
    "Esbjörn Svensson Trio",     # jazz piano trio
    "The Cinematic Orchestra",   # orchestral + electronic
    "Portico Quartet",           # hang drum, sax, electronic
    "Jaga Jazzist",              # large ensemble, layered
    "Matthew Halsall",           # spiritual jazz, trumpet
]


def get_librosa_features(db, artist, limit=3):
    """Get all timbre-relevant features for an artist's tracks."""
    return db.execute("""
        SELECT artist, title, album,
               centroid_mean, flatness_mean, spectral_flux,
               zcr_mean, rms_mean, dynamic_range,
               mfcc_mean_json, mfcc_std_json, contrast_mean_json
        FROM tracks
        WHERE artist LIKE ?
        ORDER BY title
        LIMIT ?
    """, (f"{artist}%", limit)).fetchall()


def get_effnet_timbre(db, artist, title):
    """Get EffNet bright/dark from pipeline DB."""
    row = db.execute("""
        SELECT cls_json FROM tracks
        WHERE artist LIKE ? AND title LIKE ?
        AND status='done'
        LIMIT 1
    """, (f"{artist}%", f"%{title}%")).fetchone()
    if row and row[0]:
        cls = json.loads(row[0])
        return {"bright": cls.get("bright"), "dark": cls.get("dark")}
    return None


def compute_timbre_proxies(centroid, flatness, flux, zcr, rms, dyn_range,
                           mfcc_json, mfcc_std_json, contrast_json):
    """
    Compute multiple timbre proxy candidates.

    Approaches:
    1. Single-feature (centroid only) — baseline, known to be weak
    2. Scalar blend — weighted combination of scalars
    3. MFCC-derived — MFCC[1] captures spectral slope (brightness)
    4. Spectral contrast ratio — high-band vs low-band energy
    5. Full vector approach — combine everything into one score
    """
    mfcc_mean = json.loads(mfcc_json) if mfcc_json else []
    mfcc_std = json.loads(mfcc_std_json) if mfcc_std_json else []
    contrast = json.loads(contrast_json) if contrast_json else []

    proxies = {}

    # --- Raw features for inspection ---
    proxies["centroid_raw"] = round(centroid, 2)
    proxies["flatness_raw"] = round(flatness, 6)
    proxies["zcr_raw"] = round(zcr, 6)
    proxies["flux_raw"] = round(flux, 4)

    # --- Normalizations ---
    # These ranges are derived from typical music values
    centroid_norm = max(0, min(1, (centroid - 400) / 3000))
    flatness_norm = min(1, flatness / 0.015)
    zcr_norm = min(1, zcr / 0.12)
    flux_norm = min(1, flux / 120)

    # --- Approach 1: Centroid only (baseline) ---
    proxies["A1_centroid_only"] = round(centroid_norm, 4)

    # --- Approach 2: Scalar blend ---
    # centroid = where the spectral mass sits
    # flatness = noise vs tone (noisy = harsher brightness)
    # zcr = high frequency content / percussiveness
    # flux = spectral change rate (bright tracks often have more flux)
    a2 = (centroid_norm * 0.45 +
          zcr_norm * 0.25 +
          flatness_norm * 0.15 +
          flux_norm * 0.15)
    proxies["A2_scalar_blend"] = round(a2, 4)

    # --- Approach 3: MFCC spectral slope ---
    # MFCC[1] captures the overall spectral slope
    # Negative = energy concentrated in low freqs (dark)
    # Positive/less negative = more high-freq energy (bright)
    # MFCC[2] adds spectral shape nuance
    if len(mfcc_mean) >= 4:
        proxies["mfcc1_mean"] = round(mfcc_mean[1], 2)
        proxies["mfcc2_mean"] = round(mfcc_mean[2], 2)
        # MFCC[1] typical range roughly -30 to +30
        mfcc1_norm = max(0, min(1, (mfcc_mean[1] + 30) / 60))
        proxies["A3_mfcc_slope"] = round(mfcc1_norm, 4)

        # With MFCC std — captures whether brightness is stable or dynamic
        if len(mfcc_std) >= 2:
            proxies["mfcc1_std"] = round(mfcc_std[1], 2)
            # High std = timbre varies a lot (dynamically bright vs uniformly bright)
            mfcc1_std_norm = min(1, mfcc_std[1] / 20)
            proxies["mfcc1_stability"] = round(1 - mfcc1_std_norm, 4)

    # --- Approach 4: Spectral contrast ratio ---
    # Compare energy in high frequency bands vs low frequency bands
    # 7 bands span the spectrum — ratio of upper to lower = brightness indicator
    if len(contrast) >= 7:
        lo_bands = sum(contrast[:3]) / 3      # low frequency bands
        hi_bands = sum(contrast[4:]) / 3      # high frequency bands
        mid_band = contrast[3]                 # midrange
        proxies["contrast_lo_avg"] = round(lo_bands, 2)
        proxies["contrast_hi_avg"] = round(hi_bands, 2)
        proxies["contrast_mid"] = round(mid_band, 2)
        # Ratio: >1 = more high-freq contrast = brighter
        ratio = hi_bands / max(lo_bands, 0.01)
        proxies["A4_contrast_ratio"] = round(ratio, 4)
        # Normalized version
        ratio_norm = max(0, min(1, (ratio - 0.3) / 1.4))
        proxies["A4_contrast_ratio_norm"] = round(ratio_norm, 4)

    # --- Approach 5: Full vector composite ---
    # Combines the best signals with weights informed by MIR research
    components = {
        "centroid": centroid_norm,
        "zcr": zcr_norm,
        "flatness": flatness_norm,
        "flux": flux_norm,
    }
    # Add MFCC slope if available
    if len(mfcc_mean) >= 2:
        mfcc1_norm = max(0, min(1, (mfcc_mean[1] + 30) / 60))
        components["mfcc_slope"] = mfcc1_norm
    # Add contrast ratio if available
    if len(contrast) >= 7:
        lo_bands = sum(contrast[:3]) / 3
        hi_bands = sum(contrast[4:]) / 3
        ratio = hi_bands / max(lo_bands, 0.01)
        ratio_norm = max(0, min(1, (ratio - 0.3) / 1.4))
        components["contrast_ratio"] = ratio_norm

    # Weights: centroid and MFCC slope are strongest signals
    weights = {
        "centroid": 0.30,
        "mfcc_slope": 0.25,
        "contrast_ratio": 0.20,
        "zcr": 0.12,
        "flatness": 0.08,
        "flux": 0.05,
    }
    total_w = sum(weights.get(k, 0) for k in components)
    a5 = sum(components[k] * weights.get(k, 0) for k in components) / max(total_w, 0.01)
    proxies["A5_full_composite"] = round(a5, 4)

    # --- Component breakdown for A5 ---
    proxies["A5_components"] = {k: round(v, 4) for k, v in components.items()}

    return proxies


def main():
    librosa_db = sqlite3.connect(str(MUSIC_DB))
    pipeline_db = sqlite3.connect(str(PIPELINE_DB))

    results = []

    for artist in TEST_ARTISTS:
        rows = get_librosa_features(librosa_db, artist, limit=3)
        if not rows:
            print(f"  No librosa data for {artist}, skipping")
            continue

        for row in rows:
            artist_name, title, album = row[0], row[1], row[2]
            centroid, flatness, flux, zcr, rms, dyn_range = row[3:9]
            mfcc_json, mfcc_std_json, contrast_json = row[9], row[10], row[11]

            proxies = compute_timbre_proxies(
                centroid, flatness, flux, zcr, rms, dyn_range,
                mfcc_json, mfcc_std_json, contrast_json
            )

            effnet = get_effnet_timbre(pipeline_db, artist_name, title)

            entry = {
                "artist": artist_name,
                "title": title,
                "album": album,
                "effnet": effnet,
                "librosa_proxies": proxies,
            }
            results.append(entry)

            a5 = proxies["A5_full_composite"]
            label = "BRIGHT" if a5 > 0.5 else "DARK"
            effnet_str = ""
            if effnet:
                effnet_str = f"  effnet={effnet['bright']:.4f}"

            print(f"  {artist_name:40s} {title:30s} A5={a5:.3f} ({label:6s}){effnet_str}")

    # Save
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} tracks → {OUTPUT}")

    # === Distribution comparison ===
    print("\n=== Spread comparison (wider = more useful) ===")
    print(f"  {'Approach':30s} {'min':>8s} {'max':>8s} {'spread':>8s} {'median':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    approaches = [
        ("EffNet bright",    [r["effnet"]["bright"] for r in results if r["effnet"]]),
        ("A1: Centroid only", [r["librosa_proxies"]["A1_centroid_only"] for r in results]),
        ("A2: Scalar blend", [r["librosa_proxies"]["A2_scalar_blend"] for r in results]),
        ("A3: MFCC slope",   [r["librosa_proxies"]["A3_mfcc_slope"] for r in results if "A3_mfcc_slope" in r["librosa_proxies"]]),
        ("A4: Contrast ratio",[r["librosa_proxies"]["A4_contrast_ratio_norm"] for r in results if "A4_contrast_ratio_norm" in r["librosa_proxies"]]),
        ("A5: Full composite",[r["librosa_proxies"]["A5_full_composite"] for r in results]),
    ]

    for name, vals in approaches:
        if not vals:
            continue
        vals_s = sorted(vals)
        spread = vals_s[-1] - vals_s[0]
        median = vals_s[len(vals_s) // 2]
        print(f"  {name:30s} {vals_s[0]:8.4f} {vals_s[-1]:8.4f} {spread:8.4f} {median:8.4f}")

    # === Sanity check: do known bright/dark tracks sort correctly? ===
    print("\n=== Sanity check: expected bright vs dark ===")
    print("  (Ballaké = dark/warm kora, Four Tet = bright electronics)")
    for r in results:
        if "Ballaké" in r["artist"] or "Four Tet" in r["artist"]:
            a5 = r["librosa_proxies"]["A5_full_composite"]
            label = "BRIGHT" if a5 > 0.5 else "DARK"
            print(f"  {r['artist']:40s} {r['title']:30s} A5={a5:.3f} ({label})")


if __name__ == "__main__":
    main()
