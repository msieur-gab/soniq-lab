# Derived Classifiers — Formula Lab

These classifiers are computed from existing classifier outputs. Zero extraction cost.

---

## 1. Bright/Dark (Perceived Emotional Quality)

**What it measures:** Does the music feel light/bright or dark/somber when listening?
Not spectral brightness — emotional quality.

**Why it exists:** The EffNet bright/dark ground truth was useless (0.166 spread across 1081 tracks).
This formula replaces it using signals that actually work.

**Formula (v2 — current best):**
```
bright_dark = 0.45 * valence_01 + 0.30 * (1 - sad) + 0.15 * brilliant + 0.10 * (1 - relaxed)
```
Where `valence_01 = (valence - 1) / 8` (normalize from 1-9 to 0-1 scale).

**Scale:** 0 = dark/somber, 1 = light/bright

**Tested on:** 662 tracks. Distribution: 1.5% < 0.2, 28% 0.2-0.4, 50% 0.4-0.6, 20% 0.6-0.8, 0.4% > 0.8

**Sanity checks (artist means):**
- Ballaké Sissoko: 0.297 (darkest) — contemplative kora
- Jóhann Jóhannsson: 0.331 — dark cinematic
- Billie Eilish: 0.396 — dark pop
- Bersarin Quartett: 0.401 — melancholic ambient
- GoGo Penguin: 0.462 — moody jazz
- Bonobo: 0.521 — warm electronic
- Four Tet: 0.553 — upbeat electronic
- Daft Punk: 0.647 (brightest) — party/energy

**Known limitation:** Relies on `sad` for darkness, so cold/void darkness (Plastikman Consumed)
may not score dark enough. Needs testing once Plastikman is processed. May need an additional
term for emptiness/tension (low tonal, low arousal, atonal).

**Track-level validation:** Portico Quartet Terrain I→II→III shows correct darkening arc
(0.477 → 0.464 → 0.384), matching both rising sadness and falling timbral brightness.

---

## 2. Melancholic/Nostalgic

**What it measures:** Bittersweet beauty. Music you want to keep listening to despite
(because of) its sadness. The "beautiful sadness" zone that no MIR system classifies.

**Why it exists:** Every system flags "sad" or "not happy" but misses the distinction
between painful sadness (Billie "listen before i go") and nostalgic melancholy
(Sakamoto "Merry Christmas Mr. Lawrence", Bersarin Quartett, EST).

**Formula (v1 — prototype):**
```python
import math

# Sadness sweet spot: peaks around 0.5-0.7, drops at extremes
sad_sweet = math.exp(-((sad - 0.6) ** 2) / (2 * 0.25 ** 2))

# Valence sweet spot: mid-low (bittersweet, not rock-bottom despair)
val_sweet = math.exp(-((valence_01 - 0.4) ** 2) / (2 * 0.15 ** 2))

melancholy = (
    0.30 * sad_sweet +       # sweet spot of sadness
    0.20 * tonal +           # melodic beauty
    0.20 * relaxed +         # contemplative
    0.10 * (1 - brilliant) + # warm timbre helps
    0.10 * val_sweet +       # bittersweet valence zone
    0.10 * (1 - aggressive)  # gentle
)
```

**Scale:** 0 = not melancholic, 1 = deeply melancholic/nostalgic

**Tested on:** 662 tracks. Distribution skewed high (80% > 0.6) because Gab's library
IS heavily curated toward this aesthetic.

**Known issues:**
- Catches "contemplative" too broadly — Matthew Halsall scores high but his music is
  warm/positive, not really melancholic.
- Need to tighten sad sweet spot or penalize high valence to separate contemplative
  from melancholic.
- Ahmad Jamal ranks #1 most melancholic — debatable.

**Top tracks (correct):** GoGo Penguin "The Letter", Mammal Hands "Living Frost",
Bersarin Quartett, Jóhannsson "Flight from the City"

**Bottom tracks (correct):** Daft Punk bangers, Four Tet "She Moves She"

---

## 3. Contemplative (discovered accidentally)

**What it measures:** Beautiful, calm, immersive music. The "lean back and get lost" quality.
Not necessarily sad — can be positive (Matthew Halsall) or neutral (Steve Reich).

**Formula:** Not yet written. The melancholy v1 formula effectively measures this already.
To separate contemplative from melancholic:

```
contemplative = tonal + relaxed + warm + gentle (broadly)
melancholic   = contemplative + bittersweet sadness (narrower)
```

**Possible formula:**
```
contemplative = 0.30 * tonal + 0.30 * relaxed + 0.20 * (1 - brilliant) + 0.20 * (1 - aggressive)
```

**Not yet tested.** Wait for full 1359 tracks.

**Expected artist ranking:**
- High: Sakamoto, Ballaké Sissoko, Bersarin Quartett, Glass, Matthew Halsall
- Medium: GoGo Penguin, Bonobo, EST
- Low: Daft Punk, Plastikman

**Key test case:** Steve Reich "Music for 18 Musicians" should be tonal + NOT relaxed
(propulsive) = moderate contemplative. Plastikman Consumed should be NOT tonal +
NOT relaxed = low contemplative. These edge cases will validate the formula.

---

## Inputs (all already computed)

| Input | Source | Range |
|---|---|---|
| `sad` | Ridge classifier on librosa features | 0-1 |
| `valence` | Ridge classifier, raw scale | 1-9 (normalize to 0-1) |
| `brilliant` | z-score timbre (spectral centroid) | 0-1 |
| `relaxed` | Ridge classifier | 0-1 |
| `tonal` | Ridge classifier | 0-1 |
| `aggressive` | Ridge classifier | 0-1 |

---

## Next Steps

1. Wait for full 1359 tracks to finish processing
2. Test bright/dark v2 on Plastikman (cold darkness test case)
3. Tune melancholy formula to separate from contemplative
4. Prototype contemplative formula
5. Compare all three against validation set (output/validation_set.json)
6. If formulas hold, wire into py/classifiers/ as derived modules
