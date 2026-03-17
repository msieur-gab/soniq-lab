"""v0.6 classifiers — 14 formula-based, all feature-only.

Single-phase execution: every classifier operates on prepared features only.
No inter-classifier dependencies. No circular dependency risk.

Dropped: acoustic (confuses production method with instrument type),
         tonal/atonal (music theory jargon, useful signals already in valence/sad/happy).

Usage:
    from py.classifiers import predict_all
    cls = predict_all(librosa_features)
"""

from . import _features
from . import arousal, valence
from . import sad, relaxed, happy
from . import aggressive, danceable, instrumental
from . import energy, hypnotic, timbre
from . import brightness, contemplative, party


def predict_all(librosa_features):
    """Run all classifiers on raw librosa features.

    Args:
        librosa_features: dict from extract_track_features()

    Returns:
        dict with all classification results (0-1 scale).
    """
    prepared = _features.prepare(librosa_features)
    results = {}

    # Foundation: arousal / valence
    arousal_r = arousal.predict(prepared)
    results["arousal"] = arousal_r["arousal"]

    valence_r = valence.predict(prepared)
    results["valence"] = valence_r["valence"]

    # Emotion
    sad_r = sad.predict(prepared)
    results["sad"] = sad_r["sad"]

    relaxed_r = relaxed.predict(prepared)
    results["relaxed"] = relaxed_r["relaxed"]

    happy_r = happy.predict(prepared)
    results["happy"] = happy_r["happy"]

    aggressive_r = aggressive.predict(prepared)
    results["aggressive"] = aggressive_r["aggressive"]

    # Rhythm / movement
    danceable_r = danceable.predict(prepared)
    results["danceable"] = danceable_r["danceable"]

    party_r = party.predict(prepared)
    results["party"] = party_r["party"]

    # Energy
    energy_r = energy.predict(prepared)
    results["energetic"] = energy_r["energetic"]
    results["still"] = energy_r["still"]
    results["_energy_components"] = {
        "pulse": energy_r["pulse"],
        "impact": energy_r["impact"],
        "activity": energy_r["activity"],
        "groove": energy_r["groove"],
        "loudness": energy_r["loudness"],
    }

    # Character
    hypnotic_r = hypnotic.predict(prepared)
    results["hypnotic"] = hypnotic_r["hypnotic"]
    results["varied"] = hypnotic_r["varied"]
    results["_hypnotic_path"] = hypnotic_r["hypnotic_path"]

    instrumental_r = instrumental.predict(prepared)
    results["instrumental"] = instrumental_r["instrumental"]
    results["vocal"] = round(1 - results["instrumental"], 4)

    # Timbre
    timbre_r = timbre.predict(prepared)
    results["brilliant"] = timbre_r["brilliant"]
    results["warm"] = timbre_r["warm"]

    # Atmosphere
    brightness_r = brightness.predict(prepared)
    results["radiant"] = brightness_r["radiant"]
    results["somber"] = brightness_r["somber"]

    contemplative_r = contemplative.predict(prepared)
    results["contemplative"] = contemplative_r["contemplative"]
    results["restless"] = contemplative_r["restless"]

    return results
