"""v0.5 classifiers — librosa features → classifications via numpy dot products.

Usage:
    from py.classifiers import predict_all
    cls = predict_all(librosa_features)
    # {"happy": 0.82, "sad": 0.18, ..., "bright": 0.7, "dark": 0.3, ...}
"""

from . import _features
from . import happy, sad, relaxed, aggressive, party
from . import acoustic, danceable, instrumental, tonal
from . import arousal, valence
from . import brightness, timbre

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
        - bright, dark: 0-1 (sum to ~1)
        - brilliant, warm: 0-1 (sum to ~1)
    """
    prepared = _features.prepare(librosa_features)

    results = {}
    for name, module in _SINGLE_CLASSIFIERS.items():
        results[name] = round(module.predict(prepared), 4)

    # Multi-output classifiers
    bright_dark = brightness.predict(prepared)
    results["bright"] = bright_dark["bright"]
    results["dark"] = bright_dark["dark"]

    timbre_result = timbre.predict(prepared)
    results["brilliant"] = timbre_result["brilliant"]
    results["warm"] = timbre_result["warm"]

    return results
