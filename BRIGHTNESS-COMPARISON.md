# Brightness Method Comparison

**Date:** 2026-03-12
**Dataset:** 1,081 tracks from soniq.db + pipeline.db
**All values:** bright/dark probabilities summing to 1.0

## Methods Compared

| Method | Description | Source |
|--------|-------------|--------|
| **V2 (ONNX)** | NSynth-trained MusiCNN classifier, bright/dark binary | pipeline.db |
| **v0.4 (composite)** | Weighted z-scores of 5 features through sigmoid | soniq.db |
| **Centroid-only** | Spectral centroid z-score through sigmoid | computed from soniq.db scalars |

### v0.4 Composite Formula

Z-score normalize against 681-track corpus, weighted sum through sigmoid:

| Feature | Weight | What it measures |
|---------|--------|-----------------|
| Spectral centroid | 0.35 | Center of mass of spectrum (brightness) |
| MFCC[1] | 0.25 | Spectral slope/tilt |
| Spectral flatness | 0.15 | Noise-like vs tonal |
| Spectral flux | 0.15 | Rate of spectral change |
| Zero-crossing rate | 0.10 | High-frequency content proxy |

### Centroid-only Formula

```
z = (centroid_hz - 1374.4) / 528.7
bright = sigmoid(z)
dark = 1 - bright
```

Backed by psychoacoustic literature (Schubert & Wolfe 2006, Peeters 2011, TOR/McGill).

## Distribution

| Metric | V2 (ONNX) | v0.4 (composite) | Centroid-only |
|--------|-----------|-------------------|---------------|
| min | 0.3745 | 0.2836 | 0.0935 |
| max | 0.5407 | 0.9620 | 0.9985 |
| mean | 0.4797 | 0.5148 | 0.5006 |
| std | 0.0243 | 0.1203 | 0.2247 |
| median | 0.4823 | 0.4877 | 0.4674 |
| spread | 0.1662 | 0.6784 | 0.9050 |

## Correlations

| Pair | Pearson r |
|------|----------|
| V2 vs v0.4 | -0.1601 |
| V2 vs Centroid | -0.0996 |
| v0.4 vs Centroid | 0.8571 |

## Bright/Dark Classification Agreement

Threshold: bright > 0.5

| Pair | Agreement |
|------|-----------|
| V2 vs v0.4 | 537/1081 (49.7%) |
| V2 vs Centroid | 546/1081 (50.5%) |
| v0.4 vs Centroid | 898/1081 (83.1%) |
| All three | 450/1081 (41.6%) |

## Key Findings

### 1. V2 (ONNX classifier) has no discriminative power

- Spread of only 0.17 (0.37-0.54) across 1,081 tracks
- Negative correlation with both other methods
- Agrees with v0.4 at coin-flip rates (49.7%)
- Cannot distinguish 173 Hz sub-bass from 4802 Hz acid

### 2. v0.4 composite is distorted by non-brightness features

Flatness and flux contaminate the brightness score:

- **Flatness inflation:** Noisy low-frequency tracks score as "bright"
  (e.g., Four Tet "Chia" -- 773 Hz centroid but v0.4 says bright at 0.63)
- **Flux inflation:** Rhythmically active bass tracks score as "bright"
  (e.g., Plastikman "Marbles" -- 615 Hz but v0.4 says 0.51 due to flux of 101)
- **Range compression:** Spread of 0.68 vs centroid's 0.91

### 3. Centroid-only matches the literature

- Spectral centroid is the established psychoacoustic measure of brightness
  (Schubert & Wolfe 2006, Peeters 2011 Timbre Toolbox, TOR/McGill)
- NSynth dataset defines bright as "large amount of high frequency content
  and strong upper harmonics" -- this is what centroid measures
- Widest spread (0.91), most proportional to actual spectral content

## Top 30 Biggest Disagreements

| Artist | Title | V2 B | V2 D | v0.4 B | v0.4 D | Cent B | Cent D | Hz |
|--------|-------|------|------|--------|--------|--------|--------|----|
| Plastikman | Rekall | 0.421 | 0.579 | 0.951 | 0.049 | 0.999 | 0.002 | 4802 |
| Jay-Jay Johanson | Extended Beats | 0.431 | 0.569 | 0.962 | 0.038 | 0.995 | 0.005 | 4202 |
| Plastikman | Kriket | 0.435 | 0.565 | 0.811 | 0.189 | 0.992 | 0.008 | 3896 |
| Daft Punk | Indo Silver Club | 0.419 | 0.581 | 0.917 | 0.083 | 0.974 | 0.026 | 3299 |
| GoGo Penguin | Signal In The Noise (808 State) | 0.432 | 0.568 | 0.921 | 0.079 | 0.987 | 0.013 | 3672 |
| Quantic | The Picture Inside | 0.441 | 0.559 | 0.928 | 0.072 | 0.982 | 0.018 | 3481 |
| The Cinematic Orchestra | Ode To The Big Sea | 0.438 | 0.562 | 0.842 | 0.158 | 0.973 | 0.027 | 3263 |
| Daft Punk | Steam Machine | 0.430 | 0.570 | 0.895 | 0.105 | 0.965 | 0.035 | 3125 |
| Daft Punk | Burnin' (Ian Pooley Cut up Mix) | 0.433 | 0.567 | 0.929 | 0.071 | 0.967 | 0.033 | 3162 |
| Daft Punk | One More Time / Aerodynamic | 0.456 | 0.544 | 0.925 | 0.075 | 0.988 | 0.013 | 3686 |
| Daft Punk | Burnin' (DJ Sneak Main Mix) | 0.451 | 0.549 | 0.948 | 0.052 | 0.982 | 0.018 | 3474 |
| Daft Punk | Burnin' (DJ Sneak Mongowarrier) | 0.450 | 0.550 | 0.956 | 0.044 | 0.978 | 0.022 | 3373 |
| Plastikman | Plasmatik | 0.447 | 0.553 | 0.833 | 0.167 | 0.969 | 0.031 | 3191 |
| Daft Punk | Technologic | 0.445 | 0.555 | 0.842 | 0.158 | 0.966 | 0.034 | 3138 |
| The Cinematic Orchestra | The Projectionist | 0.466 | 0.534 | 0.815 | 0.185 | 0.987 | 0.013 | 3650 |
| Plastikman | Koma (Remastered) | 0.421 | 0.579 | 0.760 | 0.240 | 0.940 | 0.060 | 2833 |
| Plastikman | Lasttrak | 0.420 | 0.580 | 0.607 | 0.393 | 0.938 | 0.062 | 2806 |
| Daft Punk | Burnin' / Too Long | 0.451 | 0.549 | 0.866 | 0.134 | 0.968 | 0.033 | 3169 |
| Daft Punk | Around the World (Motorbass) | 0.456 | 0.544 | 0.864 | 0.137 | 0.972 | 0.028 | 3245 |
| Jay-Jay Johanson | She's Mine but I'm Not Hers | 0.436 | 0.564 | 0.749 | 0.251 | 0.950 | 0.050 | 2926 |
| Daft Punk | Aerodynamic | 0.465 | 0.535 | 0.863 | 0.137 | 0.978 | 0.022 | 3384 |
| Quantic | Life in the Rain | 0.466 | 0.534 | 0.892 | 0.108 | 0.976 | 0.024 | 3333 |
| Jay-Jay Johanson | Keep It a Secret | 0.468 | 0.532 | 0.879 | 0.121 | 0.972 | 0.028 | 3252 |
| Daft Punk | Alive | 0.466 | 0.534 | 0.803 | 0.197 | 0.962 | 0.038 | 3089 |
| Jay-Jay Johanson | Changed | 0.440 | 0.560 | 0.779 | 0.221 | 0.934 | 0.066 | 2779 |
| Daft Punk | Rollin' & Scratchin' | 0.441 | 0.559 | 0.798 | 0.202 | 0.936 | 0.064 | 2789 |
| The Cinematic Orchestra | Zero One / This Fantasy | 0.469 | 0.531 | 0.775 | 0.225 | 0.960 | 0.040 | 3059 |
| Flying Lotus | Living Beats | 0.475 | 0.525 | 0.835 | 0.165 | 0.965 | 0.035 | 3126 |
| Daft Punk | Superheroes / Human After All | 0.432 | 0.568 | 0.739 | 0.261 | 0.920 | 0.080 | 2667 |
| Four Tet | She Moves She | 0.435 | 0.565 | 0.840 | 0.160 | 0.922 | 0.078 | 2683 |

## Composite Distortion Cases

V2 + Centroid agree, but v0.4 disagrees: **96 tracks**

These are cases where flatness/flux/zcr/mfcc1 push v0.4 in the wrong direction.

| Artist | Title | V2 B | V2 D | v0.4 B | v0.4 D | Cent B | Cent D | Hz |
|--------|-------|------|------|--------|--------|--------|--------|----|
| Four Tet | Chia | 0.473 | 0.527 | 0.626 | 0.374 | 0.243 | 0.757 | 773 |
| GoGo Penguin | Don't Go (Portico Quartet Remix) | 0.481 | 0.519 | 0.556 | 0.444 | 0.183 | 0.817 | 582 |
| Bersarin Quartett | Im Glanze des Kometen | 0.457 | 0.543 | 0.563 | 0.437 | 0.198 | 0.802 | 634 |
| Four Tet | Gliding Through Everything | 0.460 | 0.540 | 0.551 | 0.449 | 0.211 | 0.789 | 678 |
| Plastikman | Marbles | 0.374 | 0.625 | 0.506 | 0.494 | 0.192 | 0.808 | 615 |
| Bersarin Quartett | Bedingungslos | 0.477 | 0.523 | 0.529 | 0.470 | 0.218 | 0.781 | 701 |
| Plastikman | Goo | 0.432 | 0.568 | 0.565 | 0.435 | 0.278 | 0.722 | 871 |
| The Cinematic Orchestra | Lessons (Dorian Concept Remix) | 0.490 | 0.510 | 0.530 | 0.470 | 0.262 | 0.738 | 827 |
| Bitcrush | Of Days (Widescreen Mix) | 0.473 | 0.527 | 0.548 | 0.452 | 0.283 | 0.717 | 882 |
| Bersarin Quartett | Schwarzer Regen faellt | 0.441 | 0.559 | 0.542 | 0.458 | 0.300 | 0.700 | 926 |

## Case Study: Plastikman

Richie Hawtin's catalog spans sub-bass minimal to harsh acid -- ideal stress test.

| Title | V2 B | V2 D | v0.4 B | v0.4 D | Cent B | Cent D | Hz |
|-------|------|------|--------|--------|--------|--------|----|
| Are Friends Electrik | 0.430 | 0.570 | 0.463 | 0.537 | 0.566 | 0.434 | 1514 |
| Ask Yourself | 0.449 | 0.551 | 0.385 | 0.615 | 0.254 | 0.746 | 804 |
| Consume | 0.383 | 0.617 | 0.342 | 0.658 | 0.252 | 0.748 | 798 |
| Contain | 0.390 | 0.610 | 0.352 | 0.648 | 0.118 | 0.882 | 311 |
| Converge | 0.449 | 0.551 | 0.449 | 0.551 | 0.123 | 0.877 | 335 |
| Ekko | 0.428 | 0.572 | 0.470 | 0.530 | 0.121 | 0.879 | 328 |
| Goo | 0.432 | 0.568 | 0.565 | 0.435 | 0.278 | 0.722 | 871 |
| Headcase | 0.404 | 0.596 | 0.297 | 0.703 | 0.100 | 0.900 | 211 |
| Koma (Remastered) | 0.421 | 0.579 | 0.760 | 0.240 | 0.940 | 0.060 | 2833 |
| Kriket | 0.435 | 0.565 | 0.811 | 0.189 | 0.992 | 0.008 | 3896 |
| Lasttrak | 0.420 | 0.580 | 0.607 | 0.393 | 0.938 | 0.062 | 2806 |
| Marbles | 0.374 | 0.625 | 0.506 | 0.494 | 0.192 | 0.808 | 615 |
| Ping Pong | 0.448 | 0.552 | 0.284 | 0.716 | 0.093 | 0.906 | 173 |
| Plasmatik | 0.447 | 0.553 | 0.833 | 0.167 | 0.969 | 0.031 | 3191 |
| Plastique | 0.442 | 0.558 | 0.372 | 0.628 | 0.766 | 0.234 | 2001 |
| Rekall | 0.421 | 0.579 | 0.951 | 0.049 | 0.999 | 0.002 | 4802 |

Notable observations:
- V2 is flatlined (0.37-0.51) across all tracks -- zero discrimination
- "Plastique" (2001 Hz acid): centroid says bright (0.77), v0.4 says dark (0.37)
- "Marbles" (615 Hz, flux=101): v0.4 says bright (0.51), centroid correctly says dark (0.19)
- "Rekall" (4802 Hz): centroid 0.999 vs v0.4 0.951 -- both agree but centroid is more proportional
- "Ping Pong" (173 Hz): centroid 0.093 vs v0.4 0.284 -- centroid captures the extreme darkness better

## Conclusion

| Criterion | V2 (ONNX) | v0.4 (composite) | Centroid-only |
|-----------|-----------|-------------------|---------------|
| Spread | 0.17 (useless) | 0.68 (compressed) | 0.91 (full range) |
| Literature backing | NSynth labels, opaque NN | Heuristic, no source | Schubert 2006, Peeters 2011 |
| Bright/dark accuracy | Coin flip | Distorted by flux/flatness | Matches spectral content |
| Interpretability | Black box | 5 mixed features | Single feature, clear meaning |

**Recommendation:** Replace the v0.4 composite with centroid-only brightness.
The extra features (flatness, flux, zcr, mfcc1) measure different timbral
dimensions and contaminate the brightness score. Spectral centroid alone is
what the psychoacoustic literature defines as brightness.

## References

- Schubert & Wolfe (2006). "Does Timbral Brightness Scale with Frequency and Spectral Centroid." Acta Acustica.
- Peeters et al. (2011). "The Timbre Toolbox: Extracting audio descriptors from musical signals." JASA 130, 2902-2916.
- NSynth Dataset (Google Magenta). Bright: "large amount of high frequency content and strong upper harmonics."
- TOR / McGill. Spectral Centroid article by Julie Delisle.
- comma-lab/timbre-resources (Queen Mary University of London, C4DM).
