"""Classifier: radiant/somber — perceived brightness/darkness of music.

Uses acoustic features (mfcc0 fullness, harmonic content, spectral centroid)
combined with sadness as a penalty.
0 = somber (dark, sparse, ominous), 1 = radiant (bright, full, euphoric).
"""

import math


def predict(results, prepared=None):
    sad = results.get("sad", 0.5)

    if prepared is not None:
        mfcc0 = prepared.get("mfcc0", -200)
        harm_fraction = prepared.get("harm_fraction", 0.5)
        centroid = prepared.get("centroid", 1500)

        fullness = 1 / (1 + math.exp(-0.015 * (mfcc0 + 150)))
        centroid_factor = min(1.0, centroid / 1500)
        melodic = harm_fraction * centroid_factor * min(1.0, fullness * 2.5)
        acoustic = fullness * 0.6 + melodic * 0.4
        radiant = acoustic * (1 - 0.4 * sad) + 0.08
    else:
        # Fallback: classifier-only (less accurate)
        valence_norm = (results.get("valence", 5.0) - 1) / 8
        brilliant = results.get("brilliant", 0.5)
        relaxed = results.get("relaxed", 0.5)
        radiant = (
            0.45 * valence_norm
            + 0.30 * (1 - sad)
            + 0.15 * brilliant
            + 0.10 * (1 - relaxed)
        )

    radiant = round(max(0.0, min(1.0, radiant)), 4)
    somber = round(1 - radiant, 4)
    return {"radiant": radiant, "somber": somber}
