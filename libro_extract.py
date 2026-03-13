#!/usr/bin/env python3
"""Standalone librosa feature extraction → libro-soniq.db

Extracts ALL features (current 72 + ~45 new) into a standalone database.
No imports from py/ modules. Self-contained.

Usage:
    python3 libro_extract.py              # extract all tracks
    python3 libro_extract.py --reset      # drop and re-extract everything
    python3 libro_extract.py --limit 10   # process only N tracks (for testing)
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "libro-soniq.db"

# Music folder — same resolution as soniq.py
MUSIC_CANDIDATES = [
    ROOT / "music",                          # local symlink
    Path.home() / "music-player" / "music",  # MusiCast
    Path.home() / "Music",                   # standard
]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path, reset=False):
    if reset and Path(db_path).exists():
        Path(db_path).unlink()
        print("  Deleted existing database.")

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
            scalars_json TEXT,
            vectors_json TEXT,
            duration_s REAL,
            created_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
    """)
    db.commit()
    return db


def scan_folder(db, music_root):
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
            for sep in [" - ", "- ", " "]:
                if sep in title and title.split(sep, 1)[0].strip().isdigit():
                    title = title.split(sep, 1)[1]
                    break
            try:
                db.execute(
                    "INSERT OR IGNORE INTO tracks (path, artist, album, title) VALUES (?, ?, ?, ?)",
                    (str(path), artist, album, title),
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
    db.commit()
    return count


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(filepath, max_duration=300):
    """Extract current 72 + ~45 new features from an audio file.

    Returns (scalars_dict, vectors_dict, duration) or None on error.
    """
    import librosa
    import numpy as np
    from scipy.stats import skew, kurtosis, gmean

    # --- Load audio ---
    try:
        duration = librosa.get_duration(path=filepath)
    except Exception:
        return None

    if duration < 3:
        return None

    load_offset = max(0, duration * 0.05)
    load_duration = min(90, duration - load_offset)
    if max_duration and duration > max_duration:
        load_duration = min(load_duration, max_duration * 0.3)

    try:
        y_full, sr = librosa.load(filepath, sr=22050, offset=load_offset,
                                  duration=load_duration, mono=True)
    except Exception:
        return None

    if len(y_full) < sr:
        return None

    actual_loaded = len(y_full) / sr

    # --- HPSS on full loaded audio (needs context) ---
    try:
        y_harm, y_perc = librosa.effects.hpss(y_full)
        rms_harm = float(np.mean(librosa.feature.rms(y=y_harm)[0]))
        rms_perc = float(np.mean(librosa.feature.rms(y=y_perc)[0]))
        harm_energy = rms_harm
        perc_energy = rms_perc
        harm_perc_ratio = rms_harm / (rms_perc + 1e-8)
        harm_fraction = rms_harm / (rms_harm + rms_perc + 1e-8)
    except Exception:
        harm_energy = 0.0
        perc_energy = 0.0
        harm_perc_ratio = 1.0
        harm_fraction = 0.5

    # --- Tempogram / PLP on full audio ---
    try:
        oenv_full = librosa.onset.onset_strength(y=y_full, sr=sr)

        tempogram = librosa.feature.tempogram(onset_envelope=oenv_full, sr=sr)
        tg_mean = tempogram.mean(axis=1)
        beat_regularity = float(np.max(tg_mean) / (np.mean(tg_mean) + 1e-8))
        # Rhythm complexity: entropy of tempo profile
        tg_norm = tg_mean / (np.sum(tg_mean) + 1e-8)
        tg_norm = tg_norm[tg_norm > 0]
        rhythm_complexity = float(-np.sum(tg_norm * np.log2(tg_norm + 1e-12)))

        pulse = librosa.beat.plp(onset_envelope=oenv_full, sr=sr)
        plp_mean = float(np.mean(pulse))
        plp_stability = float(np.mean(pulse) / (np.std(pulse) + 1e-8))

        # Onset rate: actual count of detected onsets per second
        onset_frames = librosa.onset.onset_detect(onset_envelope=oenv_full, sr=sr)
        onset_rate = float(len(onset_frames) / (actual_loaded + 1e-8))
    except Exception:
        beat_regularity = 1.0
        rhythm_complexity = 0.0
        plp_mean = 0.0
        plp_stability = 1.0
        onset_rate = 0.0

    # --- Segments ---
    segments = _slice_segments(y_full, sr, actual_loaded)
    if not segments:
        return None

    seg_feats = []
    for y_seg in segments:
        sf = _segment_features(y_seg, sr)
        if sf:
            seg_feats.append(sf)

    if not seg_feats:
        return None

    # --- Aggregate segment features into scalars + vectors ---
    scalars = {"duration": duration}

    # Current scalar features — average across segments
    for key in ("centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
                "bandwidth_mean", "bandwidth_std", "flatness_mean",
                "spectral_flux", "flux_std",
                "onset_strength", "beat_strength", "vocal_proxy", "zcr_mean"):
        scalars[key] = float(np.mean([f[key] for f in seg_feats]))

    # RMS features
    rms_linear_vals = [f["rms_linear"] for f in seg_feats]
    avg_rms_linear = float(np.mean(rms_linear_vals))
    scalars["rms_mean"] = float(20 * np.log10(avg_rms_linear + 1e-10))
    rms_db_vals = [20 * np.log10(v + 1e-10) for v in rms_linear_vals]
    scalars["rms_variance"] = float(np.var(rms_db_vals)) if len(rms_db_vals) > 1 else 0.0

    all_rms_db = []
    for f in seg_feats:
        all_rms_db.extend(f["_rms_db_frames"])
    if all_rms_db:
        arr = np.array(all_rms_db)
        p95 = float(np.percentile(arr, 95))
        scalars["dynamic_range"] = p95 - float(np.percentile(arr, 5))
        scalars["rms_max"] = p95
    else:
        scalars["dynamic_range"] = 0.0
        scalars["rms_max"] = scalars["rms_mean"]

    # --- NEW: RMS statistics (from all segment RMS frames) ---
    if all_rms_db:
        rms_arr = np.array(all_rms_db)
        scalars["low_energy_rate"] = float(np.mean(rms_arr < np.mean(rms_arr)))
        scalars["energy_skew"] = float(skew(rms_arr))
        scalars["energy_kurtosis"] = float(kurtosis(rms_arr))
    else:
        scalars["low_energy_rate"] = 0.5
        scalars["energy_skew"] = 0.0
        scalars["energy_kurtosis"] = 0.0

    # --- NEW: Sub-band energy ratios (averaged across segments) ---
    for key in ("bass_ratio", "mid_ratio", "treble_ratio", "bass_mid_ratio"):
        scalars[key] = float(np.mean([f[key] for f in seg_feats]))

    # --- NEW: Spectral moments + shape (averaged across segments) ---
    for key in ("spectral_skew", "spectral_kurtosis", "spectral_entropy", "spectral_crest"):
        scalars[key] = float(np.mean([f[key] for f in seg_feats]))

    # --- NEW: Delta MFCC summary scalars ---
    for key in ("mfcc_delta_var", "mfcc_delta2_var"):
        scalars[key] = float(np.mean([f[key] for f in seg_feats]))

    # --- NEW: Modulation spectrum (averaged across segments) ---
    for key in ("mod_flatness", "mod_crest", "mod_centroid"):
        scalars[key] = float(np.mean([f[key] for f in seg_feats]))

    # --- NEW: HPSS scalars (from full audio, already computed) ---
    scalars["harm_energy"] = harm_energy
    scalars["perc_energy"] = perc_energy
    scalars["harm_perc_ratio"] = harm_perc_ratio
    scalars["harm_fraction"] = harm_fraction

    # --- NEW: Tempogram / PLP / onset rate scalars ---
    scalars["beat_regularity"] = beat_regularity
    scalars["rhythm_complexity"] = rhythm_complexity
    scalars["plp_mean"] = plp_mean
    scalars["plp_stability"] = plp_stability
    scalars["onset_rate"] = onset_rate

    # --- Vector features — average across segments ---
    vectors = {}
    for key in ("mfcc_mean", "mfcc_std", "contrast_mean", "chroma_mean", "tonnetz_mean"):
        vecs = [np.array(f[key]) for f in seg_feats]
        vectors[key] = np.mean(vecs, axis=0).tolist()

    # --- NEW: Delta MFCC vectors ---
    for key in ("mfcc_delta_mean", "mfcc_delta2_mean"):
        vecs = [np.array(f[key]) for f in seg_feats]
        vectors[key] = np.mean(vecs, axis=0).tolist()

    # --- Tempo + key (from center of full audio) ---
    import librosa
    try:
        mid = len(y_full) // 2
        half_window = min(30 * sr, mid)
        y_tempo = y_full[mid - half_window:mid + half_window]
        tempo, _ = librosa.beat.beat_track(y=y_tempo, sr=sr)
        scalars["tempo"] = float(np.atleast_1d(tempo)[0])
        key, mode = _extract_key_mode(y_tempo, sr)
        scalars["key"] = key
        scalars["mode"] = mode
    except Exception:
        scalars["tempo"] = 0.0
        scalars["key"] = 0
        scalars["mode"] = 1

    # --- Rename to match soniq.db conventions ---
    # The calibration scripts expect these specific key names
    scalar_out = {
        "centroid": scalars["centroid_mean"],
        "centroid_std": scalars["centroid_std"],
        "rolloff": scalars["rolloff_mean"],
        "rolloff_std": scalars["rolloff_std"],
        "bandwidth": scalars["bandwidth_mean"],
        "bandwidth_std": scalars["bandwidth_std"],
        "flatness": scalars["flatness_mean"],
        "flux": scalars["spectral_flux"],
        "flux_std": scalars["flux_std"],
        "onset": scalars["onset_strength"],
        "beat": scalars["beat_strength"],
        "vocal": scalars["vocal_proxy"],
        "zcr": scalars["zcr_mean"],
        "rms_mean": scalars["rms_mean"],
        "rms_max": scalars["rms_max"],
        "rms_var": scalars["rms_variance"],
        "dyn_range": scalars["dynamic_range"],
        "tempo": scalars["tempo"],
        "key": scalars["key"],
        "mode": scalars["mode"],
        "duration": scalars["duration"],
        # New features
        "low_energy_rate": scalars["low_energy_rate"],
        "energy_skew": scalars["energy_skew"],
        "energy_kurtosis": scalars["energy_kurtosis"],
        "bass_ratio": scalars["bass_ratio"],
        "mid_ratio": scalars["mid_ratio"],
        "treble_ratio": scalars["treble_ratio"],
        "bass_mid_ratio": scalars["bass_mid_ratio"],
        "spectral_skew": scalars["spectral_skew"],
        "spectral_kurtosis": scalars["spectral_kurtosis"],
        "spectral_entropy": scalars["spectral_entropy"],
        "spectral_crest": scalars["spectral_crest"],
        "mfcc_delta_var": scalars["mfcc_delta_var"],
        "mfcc_delta2_var": scalars["mfcc_delta2_var"],
        "mod_flatness": scalars["mod_flatness"],
        "mod_crest": scalars["mod_crest"],
        "mod_centroid": scalars["mod_centroid"],
        "harm_energy": scalars["harm_energy"],
        "perc_energy": scalars["perc_energy"],
        "harm_perc_ratio": scalars["harm_perc_ratio"],
        "harm_fraction": scalars["harm_fraction"],
        "beat_regularity": scalars["beat_regularity"],
        "rhythm_complexity": scalars["rhythm_complexity"],
        "plp_mean": scalars["plp_mean"],
        "plp_stability": scalars["plp_stability"],
        "onset_rate": scalars["onset_rate"],
    }

    vector_out = {
        "mfcc_m": vectors["mfcc_mean"],
        "mfcc_s": vectors["mfcc_std"],
        "contrast": vectors["contrast_mean"],
        "chroma": vectors["chroma_mean"],
        "tonnetz": vectors["tonnetz_mean"],
        # New vectors
        "mfcc_delta": vectors["mfcc_delta_mean"],
        "mfcc_delta2": vectors["mfcc_delta2_mean"],
    }

    return scalar_out, vector_out, duration


def _slice_segments(y, sr, loaded_dur, seg_dur=10.0):
    seg_samples = int(seg_dur * sr)
    if loaded_dur < 15:
        return [y]
    segments = []
    for pct in (0.15, 0.50, 0.85):
        center = int(loaded_dur * pct * sr)
        start = max(0, center - seg_samples // 2)
        end = start + seg_samples
        if end > len(y):
            start = max(0, len(y) - seg_samples)
            end = len(y)
        seg = y[start:end]
        if len(seg) >= sr:
            segments.append(seg)
    return segments


def _segment_features(y, sr):
    """Extract all features from a single segment. One STFT, reused everywhere."""
    import librosa
    import numpy as np
    from scipy.stats import skew, kurtosis, gmean

    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_power = S ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # --- RMS ---
        rms = np.sqrt(np.mean(S_power, axis=0))
        rms_linear = float(np.mean(rms))
        rms_db_frames = (20 * np.log10(rms + 1e-10)).tolist()

        # --- Spectral features ---
        centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))

        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness_mean = float(np.mean(flatness))

        if S.shape[1] > 1:
            diff = np.diff(S, axis=1)
            flux_per_frame = np.sqrt(np.sum(diff ** 2, axis=0))
            spectral_flux = float(np.mean(flux_per_frame))
            flux_std = float(np.std(flux_per_frame))
        else:
            spectral_flux = 0.0
            flux_std = 0.0

        S_db = librosa.power_to_db(S_power)
        onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)
        onset_strength = float(np.mean(onset_env))
        beat_strength = float(np.percentile(onset_env, 75))

        # Vocal proxy (same as existing)
        H, _ = librosa.decompose.hpss(S)
        h_energy = float(np.sum(H ** 2))
        total_energy = float(np.sum(S_power))
        harmonic_ratio = h_energy / total_energy if total_energy > 1e-10 else 0.0
        h_flatness = float(np.mean(librosa.feature.spectral_flatness(S=H)[0]))
        vocal_proxy = harmonic_ratio * (1.0 - h_flatness)

        # MFCC
        mel_S = librosa.feature.melspectrogram(S=S_power, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()

        # Contrast, chroma, tonnetz
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast_mean = np.mean(contrast, axis=1).tolist()

        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

        # Rolloff + bandwidth
        rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))

        # ---------------------------------------------------------------
        # NEW FEATURES (from same STFT / computed data)
        # ---------------------------------------------------------------

        # --- Sub-band energy ratios ---
        bass_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs < 2000)
        treble_mask = freqs >= 2000

        bass = S_power[bass_mask].sum(axis=0)
        mid = S_power[mid_mask].sum(axis=0)
        treble = S_power[treble_mask].sum(axis=0)
        total_band = bass + mid + treble + 1e-8

        bass_ratio = float(np.mean(bass / total_band))
        mid_ratio = float(np.mean(mid / total_band))
        treble_ratio = float(np.mean(treble / total_band))
        bass_mid_ratio = float(np.mean(bass / (mid + 1e-8)))

        # --- Spectral higher-order moments ---
        # Vectorized: compute skew/kurtosis across all frames at once
        S_norm = S_power / (S_power.sum(axis=0, keepdims=True) + 1e-12)
        freqs_col = freqs[:, np.newaxis]
        mu = np.sum(freqs_col * S_norm, axis=0)
        sigma = np.sqrt(np.sum(S_norm * (freqs_col - mu) ** 2, axis=0) + 1e-12)
        z = (freqs_col - mu) / (sigma + 1e-12)
        spec_skew = float(np.mean(np.sum(S_norm * z ** 3, axis=0)))
        spec_kurt = float(np.mean(np.sum(S_norm * z ** 4, axis=0) - 3.0))

        # --- Spectral entropy: how random/organized the spectrum is ---
        S_prob = S_norm + 1e-12  # already normalized per frame
        spec_entropy = float(np.mean(-np.sum(S_prob * np.log2(S_prob), axis=0)))

        # --- Spectral crest: peak-to-mean ratio (tonal vs noise) ---
        spec_crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-8)))

        # --- Delta MFCCs ---
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta_mean = np.mean(np.abs(mfcc_delta), axis=1).tolist()  # 13 values
        mfcc_delta2_mean = np.mean(np.abs(mfcc_delta2), axis=1).tolist()  # 13 values
        mfcc_delta_var = float(np.mean(np.std(mfcc_delta, axis=1)))
        mfcc_delta2_var = float(np.mean(np.std(mfcc_delta2, axis=1)))

        # --- Modulation spectrum (FFT of centroid trace) ---
        if len(centroid) > 4:
            mod_spectrum = np.abs(np.fft.rfft(centroid))
            mod_spectrum_pos = mod_spectrum[mod_spectrum > 0]
            if len(mod_spectrum_pos) > 0:
                mod_flat = float(gmean(mod_spectrum + 1e-8) / (np.mean(mod_spectrum) + 1e-8))
                mod_cr = float(np.max(mod_spectrum) / (np.mean(mod_spectrum) + 1e-8))
                freqs_mod = np.arange(len(mod_spectrum))
                mod_cent = float(np.sum(freqs_mod * mod_spectrum) / (np.sum(mod_spectrum) + 1e-8))
            else:
                mod_flat, mod_cr, mod_cent = 0.0, 1.0, 0.0
        else:
            mod_flat, mod_cr, mod_cent = 0.0, 1.0, 0.0

        return {
            # --- Current features ---
            "rms_linear": rms_linear,
            "_rms_db_frames": rms_db_frames,
            "centroid_mean": centroid_mean,
            "centroid_std": centroid_std,
            "rolloff_mean": rolloff_mean,
            "rolloff_std": rolloff_std,
            "bandwidth_mean": bandwidth_mean,
            "bandwidth_std": bandwidth_std,
            "flatness_mean": flatness_mean,
            "spectral_flux": spectral_flux,
            "flux_std": flux_std,
            "onset_strength": onset_strength,
            "beat_strength": beat_strength,
            "vocal_proxy": vocal_proxy,
            "zcr_mean": zcr_mean,
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "contrast_mean": contrast_mean,
            "chroma_mean": chroma_mean,
            "tonnetz_mean": tonnetz_mean,
            # --- New features ---
            "bass_ratio": bass_ratio,
            "mid_ratio": mid_ratio,
            "treble_ratio": treble_ratio,
            "bass_mid_ratio": bass_mid_ratio,
            "spectral_skew": spec_skew,
            "spectral_kurtosis": spec_kurt,
            "spectral_entropy": spec_entropy,
            "spectral_crest": spec_crest,
            "mfcc_delta_mean": mfcc_delta_mean,
            "mfcc_delta2_mean": mfcc_delta2_mean,
            "mfcc_delta_var": mfcc_delta_var,
            "mfcc_delta2_var": mfcc_delta2_var,
            "mod_flatness": mod_flat,
            "mod_crest": mod_cr,
            "mod_centroid": mod_cent,
        }
    except Exception as e:
        return None


def _extract_key_mode(y, sr):
    import librosa
    import numpy as np
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_corr, best_key, best_mode = -2.0, 0, 1
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        maj_corr = float(np.corrcoef(rotated, major)[0, 1])
        min_corr = float(np.corrcoef(rotated, minor)[0, 1])
        if maj_corr > best_corr:
            best_corr, best_key, best_mode = maj_corr, shift, 1
        if min_corr > best_corr:
            best_corr, best_key, best_mode = min_corr, shift, 0
    return best_key, best_mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Libro-Soniq — Standalone librosa extraction")
    parser.add_argument("--reset", action="store_true", help="Drop DB and re-extract everything")
    parser.add_argument("--limit", type=int, default=0, help="Process only N tracks (0 = all)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Libro-Soniq — Standalone librosa feature extraction")
    print("=" * 55)

    # Find music folder
    music_root = None
    for candidate in MUSIC_CANDIDATES:
        if candidate.is_dir():
            music_root = candidate
            break

    if not music_root:
        print("ERROR: No music folder found. Checked:")
        for c in MUSIC_CANDIDATES:
            print(f"  - {c}")
        sys.exit(1)

    print(f"\n  Music folder: {music_root}")
    print(f"  Database: {DB_PATH}")

    # Init DB
    db = init_db(DB_PATH, reset=args.reset)

    # Scan for tracks
    new_count = scan_folder(db, music_root)
    total = db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    done = db.execute("SELECT COUNT(*) FROM tracks WHERE status='done'").fetchone()[0]
    pending = total - done

    print(f"  Tracks found: {total} ({done} already done, {pending} pending)")

    if pending == 0:
        print("\n  All tracks already processed. Use --reset to re-extract.")
        db.close()
        return

    # Process
    limit_str = f" (limit: {args.limit})" if args.limit else ""
    print(f"\n  Processing {pending} tracks{limit_str}...\n")

    processed = 0
    errors = 0
    start_time = time.time()

    while True:
        if args.limit and processed >= args.limit:
            break

        row = db.execute(
            "SELECT id, path, artist, title FROM tracks WHERE status='pending' ORDER BY artist, title LIMIT 1"
        ).fetchone()

        if not row:
            break

        track_id = row["id"]
        filepath = row["path"]
        artist = row["artist"]
        title = row["title"]

        db.execute("UPDATE tracks SET status='processing' WHERE id=?", (track_id,))
        db.commit()

        processed += 1
        t0 = time.time()

        try:
            result = extract_all_features(filepath)
            elapsed = time.time() - t0

            if result is None:
                db.execute(
                    "UPDATE tracks SET status='error', processed_at=datetime('now') WHERE id=?",
                    (track_id,)
                )
                db.commit()
                errors += 1
                print(f"  [{processed:4d}/{pending}] ERROR  {artist} — {title} ({elapsed:.1f}s)")
                continue

            scalar_out, vector_out, duration = result

            db.execute("""
                UPDATE tracks SET
                    status='done',
                    scalars_json=?,
                    vectors_json=?,
                    duration_s=?,
                    processed_at=datetime('now')
                WHERE id=?
            """, (
                json.dumps(scalar_out),
                json.dumps(vector_out),
                duration,
                track_id,
            ))
            db.commit()

            n_scalars = len(scalar_out)
            n_vectors = sum(len(v) if isinstance(v, list) else 1 for v in vector_out.values())
            print(f"  [{processed:4d}/{pending}] OK     {artist} — {title}  ({elapsed:.1f}s, {n_scalars}s+{n_vectors}v)")

        except Exception as e:
            db.execute(
                "UPDATE tracks SET status='error', processed_at=datetime('now') WHERE id=?",
                (track_id,)
            )
            db.commit()
            errors += 1
            print(f"  [{processed:4d}/{pending}] CRASH  {artist} — {title}: {e}")

    # Summary
    total_time = time.time() - start_time
    done_now = db.execute("SELECT COUNT(*) FROM tracks WHERE status='done'").fetchone()[0]
    errors_now = db.execute("SELECT COUNT(*) FROM tracks WHERE status='error'").fetchone()[0]

    print()
    print("=" * 55)
    print(f"  Done: {done_now} tracks")
    print(f"  Errors: {errors_now}")
    print(f"  Time: {total_time:.1f}s ({total_time/max(processed,1):.1f}s/track)")
    if done_now > 0:
        row = db.execute("SELECT scalars_json, vectors_json FROM tracks WHERE status='done' LIMIT 1").fetchone()
        s = json.loads(row["scalars_json"])
        v = json.loads(row["vectors_json"])
        n_s = len(s)
        n_v = sum(len(val) if isinstance(val, list) else 1 for val in v.values())
        print(f"  Features: {n_s} scalars + {n_v} vector values = {n_s + n_v} total")
    print("=" * 55)

    db.close()


if __name__ == "__main__":
    main()
