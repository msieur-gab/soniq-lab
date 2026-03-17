"""Librosa feature extraction — v0.6 trimmed.

Multi-point sampling (3 × 10s segments at 15%, 50%, 85% of track).
Single STFT per segment, reused for all spectral features.

v0.6: removed unused extractions (rolloff, zcr, vocal_proxy, spectral
higher-order moments, delta2 MFCCs, contrast, pYIN). Saves ~2s/track.
Tonnetz kept for future time-series harmonic analysis.
"""

import warnings
import numpy as np

# Suppress librosa PySoundFile warnings — expected for m4a/AAC files,
# audioread fallback via ffmpeg works correctly
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


def extract_librosa_features(filepath, max_duration=300):
    """Extract librosa features from an audio file.

    Returns dict with scalars, vectors, and derived features.
    Returns None on any error or if track is too short.
    """
    import librosa

    try:
        duration = librosa.get_duration(path=filepath)
    except Exception:
        return None

    if duration < 3:
        return None

    # Load 60s centered on the track — avoids intros/outros,
    # samples the heart of the music, cuts HPSS cost by ~33%
    load_duration = min(60, duration * 0.9)
    load_offset = max(0, (duration - load_duration) / 2)

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
        harm_fraction = rms_harm / (rms_harm + rms_perc + 1e-8)
    except Exception:
        harm_energy = 0.0
        perc_energy = 0.0
        harm_fraction = 0.5

    # --- Tempogram / PLP on full audio ---
    try:
        oenv_full = librosa.onset.onset_strength(y=y_full, sr=sr)

        tempogram = librosa.feature.tempogram(onset_envelope=oenv_full, sr=sr)
        tg_mean = tempogram.mean(axis=1)
        beat_regularity = float(np.max(tg_mean) / (np.mean(tg_mean) + 1e-8))
        tg_norm = tg_mean / (np.sum(tg_mean) + 1e-8)
        tg_norm = tg_norm[tg_norm > 0]
        rhythm_complexity = float(-np.sum(tg_norm * np.log2(tg_norm + 1e-12)))

        pulse = librosa.beat.plp(onset_envelope=oenv_full, sr=sr)
        plp_stability = float(np.mean(pulse) / (np.std(pulse) + 1e-8))

        onset_frames = librosa.onset.onset_detect(onset_envelope=oenv_full, sr=sr)
        onset_rate = float(len(onset_frames) / (actual_loaded + 1e-8))
    except Exception:
        beat_regularity = 1.0
        rhythm_complexity = 0.0
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

    result = {"duration": duration}

    # Average scalar features across segments
    for key in ("centroid_mean", "centroid_std",
                "bandwidth_std", "flatness_mean",
                "spectral_flux", "flux_std",
                "onset_strength", "beat_strength",
                "rms_linear"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # RMS: convert averaged linear to dB
    result["rms_mean"] = float(20 * np.log10(result["rms_linear"] + 1e-10))
    del result["rms_linear"]

    # RMS variance across segments
    rms_db_vals = [20 * np.log10(f["rms_linear"] + 1e-10) for f in seg_feats]
    result["rms_variance"] = float(np.var(rms_db_vals)) if len(rms_db_vals) > 1 else 0.0

    # Treble ratio (used by 5 classifiers)
    result["treble_ratio"] = float(np.mean([f["treble_ratio"] for f in seg_feats]))

    # Delta MFCC variance (used by 2 classifiers)
    result["mfcc_delta_var"] = float(np.mean([f["mfcc_delta_var"] for f in seg_feats]))

    # Modulation crest (used by instrumental)
    result["mod_crest"] = float(np.mean([f["mod_crest"] for f in seg_feats]))

    # HPSS scalars (from full audio)
    result["harm_energy"] = harm_energy
    result["perc_energy"] = perc_energy
    result["harm_fraction"] = harm_fraction

    # Tempogram / PLP / onset rate (from full audio)
    result["beat_regularity"] = beat_regularity
    result["rhythm_complexity"] = rhythm_complexity
    result["plp_stability"] = plp_stability
    result["onset_rate"] = onset_rate

    # Vector features — average across segments
    for key in ("mfcc_mean", "chroma_mean", "tonnetz_mean"):
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

    # Chroma major key correlation (Krumhansl-Schmuckler)
    try:
        chroma_avg = np.array(result.get("chroma_mean", [0] * 12))
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        best_key = result.get("key", 0)
        rotated = np.roll(chroma_avg, -best_key)
        corr = float(np.corrcoef(rotated, major_profile)[0, 1])
        result["chroma_major_corr"] = max(-1.0, min(1.0, corr))
    except Exception:
        result["chroma_major_corr"] = 0.0

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

        # Spectral features (USED)
        centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0]
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))

        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness_mean = float(np.mean(flatness))

        # Flux
        if S.shape[1] > 1:
            diff = np.diff(S, axis=1)
            flux_per_frame = np.sqrt(np.sum(diff ** 2, axis=0))
            spectral_flux = float(np.mean(flux_per_frame))
            flux_std = float(np.std(flux_per_frame))
        else:
            spectral_flux = 0.0
            flux_std = 0.0

        # Onset/beat from spectrogram
        S_db = librosa.power_to_db(S_power)
        onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)
        onset_strength = float(np.mean(onset_env))
        beat_strength = float(np.percentile(onset_env, 75))

        # Bandwidth (only std is used, but need full array to compute std)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
        bandwidth_std = float(np.std(bandwidth))

        # MFCCs (mean values used by instrumental for mfcc1/mfcc3 shape)
        mel_S = librosa.feature.melspectrogram(S=S_power, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        # Delta MFCCs (only order 1 variance used)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_var = float(np.mean(np.std(mfcc_delta, axis=1)))

        # Chroma (used for chroma_major_corr and future tonnetz)
        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        # Tonnetz (kept for future time-series harmonic analysis)
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

        # Treble ratio (used by 5 classifiers)
        treble_mask = freqs >= 2000
        treble = S_power[treble_mask].sum(axis=0)
        total_band = S_power.sum(axis=0) + 1e-8
        treble_ratio = float(np.mean(treble / total_band))

        # Modulation spectrum — only mod_crest used (by instrumental)
        if len(centroid) > 4:
            mod_spectrum = np.abs(np.fft.rfft(centroid))
            if np.any(mod_spectrum > 0):
                mod_cr = float(np.max(mod_spectrum) / (np.mean(mod_spectrum) + 1e-8))
            else:
                mod_cr = 1.0
        else:
            mod_cr = 1.0

        return {
            "rms_linear": rms_linear,
            "centroid_mean": centroid_mean,
            "centroid_std": centroid_std,
            "bandwidth_std": bandwidth_std,
            "flatness_mean": flatness_mean,
            "spectral_flux": spectral_flux,
            "flux_std": flux_std,
            "onset_strength": onset_strength,
            "beat_strength": beat_strength,
            "mfcc_mean": mfcc_mean,
            "mfcc_delta_var": mfcc_delta_var,
            "chroma_mean": chroma_mean,
            "tonnetz_mean": tonnetz_mean,
            "treble_ratio": treble_ratio,
            "mod_crest": mod_cr,
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
