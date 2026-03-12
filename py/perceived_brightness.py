"""Perceptual bright/dark classification from librosa features.

Logistic regression calibrated against V1 EffNet timbre classifier
on 1081 tracks (93% 10-fold CV accuracy). Uses spectral contrast,
flux, MFCCs, tonnetz, and chroma — no neural network needed.
"""

import numpy as np

WEIGHTS = {
    "contrast2":  0.183279,
    "contrast3":  0.262677,
    "contrast4": -0.223765,
    "flux":      -0.013672,
    "mfcc1":      0.009763,
    "mfcc2":     -0.034074,
    "tonnetz0":  -4.922302,
    "tonnetz1": -11.381657,
    "tonnetz2":   4.873693,
    "tonnetz3":   3.380292,
    "tonnetz4": -10.426049,
    "chroma0":   -0.655038,
    "chroma6":   -1.770176,
    "chroma7":    5.267907,
    "chroma8":   -5.212868,
    "chroma11":  -3.355041,
}
BIAS = 0.735607


def compute_perceived_brightness(features):
    """Compute perceptual bright/dark from librosa features.

    Returns (bright, dark) as floats summing to 1.
    """
    mfcc = features.get("mfcc_mean", [0] * 13)
    contrast = features.get("contrast_mean", [0] * 7)
    tonnetz = features.get("tonnetz_mean", [0] * 6)
    chroma = features.get("chroma_mean", [0] * 12)

    values = {
        "contrast2": contrast[2] if len(contrast) > 2 else 0,
        "contrast3": contrast[3] if len(contrast) > 3 else 0,
        "contrast4": contrast[4] if len(contrast) > 4 else 0,
        "flux":      features.get("spectral_flux", 0),
        "mfcc1":     mfcc[1] if len(mfcc) > 1 else 0,
        "mfcc2":     mfcc[2] if len(mfcc) > 2 else 0,
        "tonnetz0":  tonnetz[0] if len(tonnetz) > 0 else 0,
        "tonnetz1":  tonnetz[1] if len(tonnetz) > 1 else 0,
        "tonnetz2":  tonnetz[2] if len(tonnetz) > 2 else 0,
        "tonnetz3":  tonnetz[3] if len(tonnetz) > 3 else 0,
        "tonnetz4":  tonnetz[4] if len(tonnetz) > 4 else 0,
        "chroma0":   chroma[0] if len(chroma) > 0 else 0,
        "chroma6":   chroma[6] if len(chroma) > 6 else 0,
        "chroma7":   chroma[7] if len(chroma) > 7 else 0,
        "chroma8":   chroma[8] if len(chroma) > 8 else 0,
        "chroma11":  chroma[11] if len(chroma) > 11 else 0,
    }

    logit = BIAS
    for key, weight in WEIGHTS.items():
        logit += values[key] * weight

    bright = round(float(1 / (1 + np.exp(-logit))), 4)
    dark = round(1 - bright, 4)
    return bright, dark
