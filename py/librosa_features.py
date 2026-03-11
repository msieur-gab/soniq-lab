"""Librosa feature extraction — scalars, vectors, and timbre.

Multi-point sampling (3 × 10s segments at 15%, 50%, 85% of track).
Single STFT per segment, reused for all spectral features.
"""

import numpy as np


def extract_librosa_features(filepath, max_duration=300):
    """Extract all librosa features from an audio file.

    Returns dict with scalars (centroid, tempo, key, etc.),
    vectors (mfcc, chroma, tonnetz, etc.), and timbre additions
    (centroid_std, rolloff, bandwidth, flux_std).

    Returns None on any error or if track is too short.
    """
    import librosa

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

    result = {"duration": duration}

    # Average scalar features across segments
    for key in ("centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
                "bandwidth_mean", "bandwidth_std", "flatness_mean",
                "spectral_flux", "flux_std",
                "onset_strength", "beat_strength", "vocal_proxy", "zcr_mean"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # RMS features
    rms_linear_vals = [f["rms_linear"] for f in seg_feats]
    avg_rms_linear = float(np.mean(rms_linear_vals))
    result["rms_mean"] = float(20 * np.log10(avg_rms_linear + 1e-10))
    rms_db_vals = [20 * np.log10(v + 1e-10) for v in rms_linear_vals]
    result["rms_variance"] = float(np.var(rms_db_vals)) if len(rms_db_vals) > 1 else 0.0

    all_rms_db = []
    for f in seg_feats:
        all_rms_db.extend(f["_rms_db_frames"])
    if all_rms_db:
        arr = np.array(all_rms_db)
        p95 = float(np.percentile(arr, 95))
        result["dynamic_range"] = p95 - float(np.percentile(arr, 5))
        result["rms_max"] = p95
    else:
        result["dynamic_range"] = 0.0
        result["rms_max"] = result["rms_mean"]

    # Vector features — average across segments
    for key in ("mfcc_mean", "mfcc_std", "contrast_mean", "chroma_mean", "tonnetz_mean"):
        vecs = [np.array(f[key]) for f in seg_feats]
        result[key] = np.mean(vecs, axis=0).tolist()

    # Tempo + key
    try:
        mid = len(y_full) // 2
        half_window = min(30 * sr, mid)
        y_tempo = y_full[mid - half_window:mid + half_window]
        tempo, _ = librosa.beat.beat_track(y=y_tempo, sr=sr)
        result["tempo"] = float(np.atleast_1d(tempo)[0])
        key, mode = _extract_key_mode(y_tempo, sr)
        result["key"] = key
        result["mode"] = mode
    except Exception:
        result["tempo"] = 0.0
        result["key"] = 0
        result["mode"] = 1

    return result


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
    """Extract features from a single segment. One STFT, all features."""
    import librosa

    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_power = S ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # RMS
        rms = np.sqrt(np.mean(S_power, axis=0))
        rms_linear = float(np.mean(rms))
        rms_db_frames = (20 * np.log10(rms + 1e-10)).tolist()

        # Spectral features
        centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
        centroid_mean = float(np.mean(centroid))

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

        H, _ = librosa.decompose.hpss(S)
        h_energy = float(np.sum(H ** 2))
        total_energy = float(np.sum(S_power))
        harmonic_ratio = h_energy / total_energy if total_energy > 1e-10 else 0.0
        h_flatness = float(np.mean(librosa.feature.spectral_flatness(S=H)[0]))
        vocal_proxy = harmonic_ratio * (1.0 - h_flatness)

        mel_S = librosa.feature.melspectrogram(S=S_power, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast_mean = np.mean(contrast, axis=1).tolist()

        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

        # v0.4 timbre additions (from same STFT)
        centroid_std = float(np.std(centroid))

        rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))

        return {
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
        }
    except Exception:
        return None


def _extract_key_mode(y, sr):
    import librosa
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
