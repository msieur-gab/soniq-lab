"""v0.5 classifiers — librosa features → classifications via numpy dot products.

Usage:
    from py.classifiers import predict_all
    cls = predict_all(librosa_features)
    # {"happy": 0.82, "sad": 0.18, ..., "radiant": 0.7, "somber": 0.3, ...}
"""

from . import _features
from . import happy, sad, relaxed, aggressive, party
from . import acoustic, danceable, instrumental, tonal
from . import arousal, valence
from . import brightness, timbre, energy, hypnotic

_SINGLE_CLASSIFIERS = {
    "happy": happy,
    "sad": sad,
    "relaxed": relaxed,
    "aggressive": aggressive,
    "party": party,
    "acoustic": acoustic,
    "danceable": danceable,
    "instrumental": instrumental,
    "tonal": tonal,
    "arousal": arousal,
    "valence": valence,
}


def predict_all(librosa_features):
    """Run all classifiers on raw librosa features.

    Args:
        librosa_features: dict from extract_librosa_features()

    Returns:
        dict with all classification results:
        - happy, sad, relaxed, aggressive, party: 0-1
        - acoustic, danceable, instrumental, tonal: 0-1
        - arousal, valence: continuous (1-9 range)
        - radiant, somber: 0-1 (sum to ~1) — acoustic features + sadness
        - brilliant, warm: 0-1 (sum to ~1)
    """
    prepared = _features.prepare(librosa_features)

    results = {}
    for name, module in _SINGLE_CLASSIFIERS.items():
        results[name] = round(module.predict(prepared), 4)

    # Timbre first (brightness depends on brilliant)
    timbre_result = timbre.predict(prepared)
    results["brilliant"] = timbre_result["brilliant"]
    results["warm"] = timbre_result["warm"]

    # Radiant/somber — acoustic features + sadness penalty
    radiant_somber = brightness.predict(results, prepared)
    results["radiant"] = radiant_somber["radiant"]
    results["somber"] = radiant_somber["somber"]

    # Complements for instrumental/tonal
    results["vocal"] = round(1 - results["instrumental"], 4)
    results["atonal"] = round(1 - results["tonal"], 4)

    # Energy — kinetic/physical drive from rhythmic core + loudness multiplier
    energy_result = energy.predict(prepared)
    results["energetic"] = energy_result["energetic"]
    results["contemplative"] = energy_result["contemplative"]
    # Store components for UI visualization
    results["_energy_components"] = {
        "pulse": energy_result["pulse"],
        "impact": energy_result["impact"],
        "activity": energy_result["activity"],
        "groove": energy_result["groove"],
        "loudness": energy_result["loudness"],
    }

    # Hypnotic — two-path: rhythmic lock vs timbral consistency
    hypnotic_result = hypnotic.predict(prepared)
    results["hypnotic"] = hypnotic_result["hypnotic"]
    results["varied"] = hypnotic_result["varied"]
    results["_hypnotic_path"] = hypnotic_result["hypnotic_path"]

    return results
