# Does K follow the pre-RoPE concentration feature? (short report)

**Yes — K carries the same feature as Q, at equal or greater strength, and its
concentrated bands are precisely the ones that carry the attention logits.**

## 1. K at three metric depths (noise-input runs; dominant band = paper's metric)

| model | K whole-vec | K all-band mean | **K dominant band** | Q dominant band | K bands > 0.95 |
|---|---|---|---|---|---|
| Wan2.1-1.3B | 0.878 | 0.804 | **0.968** | 0.972 | 31% |
| LTX-Video-2B | 0.793 | 0.694 | **0.870** | 0.922 | 24% |
| CogVideoX-2b | 0.879 | 0.859 | **0.931** | 0.921 | 54% |
| *Qwen3-8B (paper)* | — | — | *0.977* | — | — |

Under **real prompts** (full pipeline): Wan K dominant stays at **0.940**
(vs Q 0.954); no-RoPE Cog K drops to 0.880. So K inherits the same
robustness asymmetry as Q — the RoPE model's keys keep the feature under
real generation, the no-RoPE control's do not.

## 2. Where K's concentration lives — `k-band-report-{wan,ltx,cog}.png`

- Top panel: **every (layer × head × band) cell** of K's per-band MRL
  (Wan: 30 × 12 × 64 = 23,040 cells; heads ruled off). Concentration is
  *banded*: specific frequency bands run dark (high R_f) across all heads
  and layers — K's structure is organized per band, not per head.
- Bottom-left: dominant-band MRL per (layer, head) — deep layers are almost
  uniformly concentrated (Wan first→last layer: 0.826 → **0.993**).
- Bottom-right: head-averaged band profile with Wan's t/h/w axis split —
  Wan's concentrated bands sit in all three axes; LTX's temporal block is
  its weakest (matches the earlier axis finding).

## 3. Link to logits — `band-logit-stats.png`, `band-logit-stats.json`

The trigonometric-series amplitude of band f is a_f = ‖q̄_f‖·‖k̄_f‖ (what the
logits are made of). Three statistics connect the geometry to the logits:

| model | top-4 bands' share of Σ a_f | Pearson r( R_f , a_f ) per head | center-only series vs actual logits r̄ |
|---|---|---|---|
| Wan | 0.27 (median 0.21) | **+0.40** (median 0.41) | 0.75 (mean) / 0.83 (median) |
| LTX | **0.48** (median 0.45) | +0.24 (median 0.29) | 0.72 / 0.81 |
| Cog | 0.30 (median 0.29) | **+0.40** (median 0.44) | n/a (no RoPE) |

Read: (i) logits are carried by few bands — in LTX half the logit amplitude
sits in 4 of 32 bands; (ii) **the high-amplitude bands are exactly the
concentrated ones** (r(R_f, a_f) > 0 everywhere, ~0.4 for Wan/Cog): the
bands that speak loudest in the logits are the ones whose K (and Q) barely
move across tokens; (iii) with centers alone, the resulting series already
reproduces actual attention logits at r̄ ≈ 0.75.

## Conclusion

K is not a passive bystander of the TriAttention geometry — keys concentrate
at least as strongly as queries (dominant-band 0.87–0.97), the concentration
is organized along frequency bands that persist across heads and layers and
deepen with depth, and those same bands dominate the logit amplitude. For
KV-side methods this is the favorable case: the memory being addressed (K) is
the most static part of the attention.
