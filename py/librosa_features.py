"""Librosa feature extraction — scalars, vectors, and v0.5 additions.

Multi-point sampling (3 × 10s segments at 15%, 50%, 85% of track).
Single STFT per segment, reused for all spectral features.

v0.5: adds HPSS, delta MFCCs, tempogram/PLP, onset_rate,
sub-band ratios, spectral entropy/crest/skew/kurtosis,
modulation spectrum, energy shape stats.
"""

import warnings
import numpy as np

# Suppress librosa PySoundFile warnings — expected for m4a/AAC files,
# audioread fallback via ffmpeg works correctly
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


def extract_librosa_features(filepath, max_duration=300):
    """Extract all librosa features from an audio file.

    Returns dict with scalars (centroid, tempo, key, etc.),
    vectors (mfcc, chroma, tonnetz, etc.), and v0.5 additions.

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

    # --- pYIN pitch tracking (vocal/instrument detection) ---
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_full, fmin=80, fmax=800, sr=sr
        )
        voiced_ratio = float(np.mean(voiced_flag))
        voiced_confidence = float(np.mean(voiced_prob[voiced_flag])) if np.any(voiced_flag) else 0.0
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
    except Exception:
        voiced_ratio = 0.0
        voiced_confidence = 0.0
        f0_mean = 0.0
        f0_std = 0.0

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
        plp_mean = float(np.mean(pulse))
        plp_stability = float(np.mean(pulse) / (np.std(pulse) + 1e-8))

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

    # RMS statistics
    if all_rms_db:
        from scipy.stats import skew, kurtosis
        rms_arr = np.array(all_rms_db)
        result["low_energy_rate"] = float(np.mean(rms_arr < np.mean(rms_arr)))
        result["energy_skew"] = float(skew(rms_arr))
        result["energy_kurtosis"] = float(kurtosis(rms_arr))
    else:
        result["low_energy_rate"] = 0.5
        result["energy_skew"] = 0.0
        result["energy_kurtosis"] = 0.0

    # Sub-band energy ratios
    for key in ("bass_ratio", "mid_ratio", "treble_ratio", "bass_mid_ratio",
                "voice_band_ratio"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # Spectral shape
    for key in ("spectral_skew", "spectral_kurtosis", "spectral_entropy", "spectral_crest"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # Delta MFCC summary scalars
    for key in ("mfcc_delta_var", "mfcc_delta2_var"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # Modulation spectrum
    for key in ("mod_flatness", "mod_crest", "mod_centroid"):
        result[key] = float(np.mean([f[key] for f in seg_feats]))

    # HPSS scalars
    result["harm_energy"] = harm_energy
    result["perc_energy"] = perc_energy
    result["harm_perc_ratio"] = harm_perc_ratio
    result["harm_fraction"] = harm_fraction

    # Tempogram / PLP / onset rate
    result["beat_regularity"] = beat_regularity
    result["rhythm_complexity"] = rhythm_complexity
    result["plp_mean"] = plp_mean
    result["plp_stability"] = plp_stability
    result["onset_rate"] = onset_rate

    # pYIN pitch tracking
    result["voiced_ratio"] = voiced_ratio
    result["voiced_confidence"] = voiced_confidence
    result["f0_mean"] = f0_mean
    result["f0_std"] = f0_std

    # Vector features — average across segments
    for key in ("mfcc_mean", "mfcc_std", "contrast_mean", "chroma_mean", "tonnetz_mean"):
        vecs = [np.array(f[key]) for f in seg_feats]
        result[key] = np.mean(vecs, axis=0).tolist()

    # Delta MFCC vectors
    for key in ("mfcc_delta_mean", "mfcc_delta2_mean"):
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
    from scipy.stats import gmean

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

        # v0.4 timbre additions
        centroid_std = float(np.std(centroid))

        rolloff = librosa.feature.spectral_rolloff(S=S, freq=freqs)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0]
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))

        # --- v0.5 additions ---

        # Sub-band energy ratios
        bass_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs < 2000)
        treble_mask = freqs >= 2000

        bass = S_power[bass_mask].sum(axis=0)
        mid = S_power[mid_mask].sum(axis=0)
        treble = S_power[treble_mask].sum(axis=0)
        total_band = bass + mid + treble + 1e-8

        bass_ratio = float(np.mean(bass / total_band))
        mid_ratio = float(np.mean(mid / total_band))

        # Voice band energy ratio (300-3000 Hz) — VAD feature
        voice_mask = (freqs >= 300) & (freqs < 3000)
        voice_band = S_power[voice_mask].sum(axis=0)
        voice_band_ratio = float(np.mean(voice_band / total_band))
        treble_ratio = float(np.mean(treble / total_band))
        bass_mid_ratio = float(np.mean(bass / (mid + 1e-8)))

        # Spectral higher-order moments
        S_norm = S_power / (S_power.sum(axis=0, keepdims=True) + 1e-12)
        freqs_col = freqs[:, np.newaxis]
        mu = np.sum(freqs_col * S_norm, axis=0)
        sigma = np.sqrt(np.sum(S_norm * (freqs_col - mu) ** 2, axis=0) + 1e-12)
        z = (freqs_col - mu) / (sigma + 1e-12)
        spec_skew = float(np.mean(np.sum(S_norm * z ** 3, axis=0)))
        spec_kurt = float(np.mean(np.sum(S_norm * z ** 4, axis=0) - 3.0))

        # Spectral entropy
        S_prob = S_norm + 1e-12
        spec_entropy = float(np.mean(-np.sum(S_prob * np.log2(S_prob), axis=0)))

        # Spectral crest
        spec_crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-8)))

        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta_mean = np.mean(np.abs(mfcc_delta), axis=1).tolist()
        mfcc_delta2_mean = np.mean(np.abs(mfcc_delta2), axis=1).tolist()
        mfcc_delta_var = float(np.mean(np.std(mfcc_delta, axis=1)))
        mfcc_delta2_var = float(np.mean(np.std(mfcc_delta2, axis=1)))

        # Modulation spectrum (FFT of centroid trace)
        if len(centroid) > 4:
            mod_spectrum = np.abs(np.fft.rfft(centroid))
            if np.any(mod_spectrum > 0):
                mod_flat = float(gmean(mod_spectrum + 1e-8) / (np.mean(mod_spectrum) + 1e-8))
                mod_cr = float(np.max(mod_spectrum) / (np.mean(mod_spectrum) + 1e-8))
                freqs_mod = np.arange(len(mod_spectrum))
                mod_cent = float(np.sum(freqs_mod * mod_spectrum) / (np.sum(mod_spectrum) + 1e-8))
            else:
                mod_flat, mod_cr, mod_cent = 0.0, 1.0, 0.0
        else:
            mod_flat, mod_cr, mod_cent = 0.0, 1.0, 0.0

        return {
            # Current features
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
            # v0.5 additions
            "bass_ratio": bass_ratio,
            "mid_ratio": mid_ratio,
            "treble_ratio": treble_ratio,
            "bass_mid_ratio": bass_mid_ratio,
            "voice_band_ratio": voice_band_ratio,
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
