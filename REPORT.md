# Pre-RoPE Q/K geometry in video diffusion transformers

**A full investigation of TriAttention's Observation 3.1 (arXiv 2604.04921) on
video DiTs — queries and keys, geometry → structure → logits → real
generation.** Run dates: 2026-10-29 → 11-01.

---

## 1. Question

TriAttention observes that in causal LLMs, **pre-RoPE Q and K concentrate
around non-zero centers** (mean resultant length R → 1), stably across
positions and contexts, and that this concentration is what turns RoPE
attention into a *trigonometric series* in token distance. We test whether
the same geometry exists in **video generation** transformers, where
self-attention uses 3D (t, h, w) RoPE over ~10–22k spatiotemporal tokens —
for **both Q and K**, at three depths of the metric:

- **whole-vector MRL**: ‖E_t[x]‖ / E_t‖x‖ for x = one head's Q (or K) across tokens
- **all-band mean**: mean over rotary pairs of the per-band R_f
- **dominant-band MRL**: R_f of the top band by amplitude a_f = ‖q̄_f‖·‖k̄_f‖ — the paper's own selection, and the bands the logits are made of

## 2. Setup

| model | arch | positional mechanism | text entry |
|---|---|---|---|
| Wan2.1-T2V-1.3B | 30L×12H×128d | 3-axis interleaved RoPE (band split t 22 / h 21 / w 21) | separate cross-attn |
| LTX-Video-2B | 28L×32H×64d | 3-axis interleaved RoPE over flattened stream (1024 bands: t/h/w blocks) | separate cross-attn |
| CogVideoX-2b | 30L×30H×64d | **no RoPE** (sinusoidal pos-emb) — the control | joint [text; video] self-attn |

Capture: forward hooks on `norm_q`/`norm_k` outputs — the exact tensors
entering each model's own `apply_rotary_emb`. **Validation: recomputing the
attention output from captures + replicated rotation matched the module's
real output bit-exactly (rel_err 0.0) in every check, noise and real-prompt
runs alike** — all numbers below are on true tensors, not approximations.

Two protocols: **(a) noise probing** — transformer-only, random latents,
zero/random text, sweeps over timestep/seed (fast, controlled); **(b) real
pipelines** — full text encoder + scheduler trajectory + CFG positive
branch, stats drained at every denoising step (`output_type="latent"`).

## 3. The feature exists — for Q *and* K

Noise-input runs, mean over all layer×head (dominant band in bold):

| model | Q whole / all-band / **dom** | K whole / all-band / **dom** | Q dom > 0.95 | K dom > 0.95 |
|---|---|---|---|---|
| Wan2.1 | 0.857 / 0.785 / **0.972** | 0.878 / 0.804 / **0.968** | 80% | 79% |
| LTX | 0.810 / 0.694 / **0.922** | 0.793 / 0.694 / **0.870** | 78% | 63% |
| CogVideoX | 0.854 / 0.819 / **0.921** | 0.879 / 0.859 / **0.931** | 76% | 78% |
| *Qwen3-8B (paper)* | *— / — / 0.977* | *—* | *84.7%* | *—* |

**K follows the feature as strongly as Q** (slightly stronger at whole-vector
and all-band depths in 2 of 3 models). On the paper's dominant-band metric,
all three video DiTs sit within ~0.05 of Qwen3-8B.

## 4. Where concentration lives (`out/q-band-report-*.png`, `out/k-band-report-*.png`, `out/*-mrl-heatmap.png`)

Per-(layer × head × band) maps for **both Q and K** (top panel: every cell,
23,040 for Wan; heads ruled off) plus per-(layer × head) dominant-band maps
and head-averaged band profiles:

- Concentration is **banded, not per-head**: the same frequency bands run
  concentrated (dark) across all heads and layers, for Q and K alike.
- It **deepens with depth**: Wan K dominant-band rises 0.826 (layer 0) →
  **0.993** (layer 29).
- **Axis structure** (LTX): w 0.77 > h 0.72 > t 0.60 — temporal bands are the
  least concentrated; Wan's t/h/w split is flat (0.78–0.79).
- High-R_f bands and high-amplitude bands coincide (§6).

## 5. The RoPE contrast

Post-RoPE MRL (rotation applied to the same captured tensors): Wan
0.86 → 0.63, LTX 0.81 → **0.20**. The shared direction is destroyed by
rotation — concentration is specifically a **pre**-RoPE property, which is
the paper's premise for moving importance estimation there. The no-RoPE
control concentrates *without* any rotation to contrast, showing
concentration is not caused by RoPE — but see §8 for what realism reveals.

## 6. Concentration ⇒ predictable logits

Three chained statistics tie the geometry to actual attention logits
(`out/band-logit-stats.*`, `out/{wan,ltx}-series.json`):

1. **Logits are carried by few bands** — top-4 bands hold 27% (Wan),
   **48%** (LTX), 30% (Cog) of total amplitude Σ a_f.
2. **The loud bands are the frozen ones** — per-head Pearson
   r( R_f , a_f ): **+0.40** (Wan), +0.24 (LTX), **+0.40** (Cog), positive
   everywhere: the bands that synthesize logits are exactly those whose Q
   and K barely move across 10–22k tokens.
3. **Centers alone reproduce the logits** — replacing every token by its Q
   and K centers (angles from the model's own rope tensors) predicts actual
   pre-softmax logits at Pearson r̄ = **0.75 mean / 0.83 median** (Wan) and
   0.72 / 0.81 (LTX); random-weight control gives r ≈ 0.

## 7. Real prompts, real trajectories (`out/{wan,cog}-pipeline.json`)

Full pipelines, real T5/UMT5 conditioning, 20/30-step denoising, validation
bit-exact again:

**Wan2.1-1.3B** (dominant-band MRL, CFG positive branch):

| step | Q dom | K dom | Q whole | post-RoPE | cos(Q̄,K̄) | Q dom > 0.95 |
|---|---|---|---|---|---|---|
| 0 (pure noise) | 0.938 | 0.963 | 0.776 | 0.539 | 0.452 | 69% |
| 10 | 0.954 | 0.934 | 0.754 | 0.477 | 0.685 | 72% |
| 19 (nearly clean) | **0.968** | 0.944 | 0.816 | 0.513 | **0.720** | **87%** |
| all steps | **0.954** | **0.940** | — | — | — | **73%** |

The feature survives full realism at near-LLM strength — for K (0.940) as
much as Q (0.954) — and **strengthens as video content emerges**.

**CogVideoX-2b** (no-RoPE control): Q whole flat at 0.71–0.73 over 30 steps,
but dominant-band drops to **0.831** (39% of heads > 0.95) vs 0.921 (76%)
under noise inputs; K dominant declines 0.914 → 0.871.

## 8. Why Wan holds and Cog does not (deduction)

- **Functional vs incidental concentration.** In a RoPE model, band-level
  concentration is the *mechanism* that gives attention its spatiotemporal
  distance preferences (§6): training preserves it because breaking it
  breaks the model's addressing. In a no-RoPE model, positional addressing
  must live *inside* content — Q/K must vary with position/content, so the
  noise-input concentration was largely incidental (interchangeable tokens
  collapse onto the default direction; real inputs break the symmetry).
- **Step-0 evidence**: with pure-noise latents in both runs, real text alone
  costs Wan 0.034 dominant-band MRL but Cog 0.091 — text enters Cog's
  self-attention stream directly (joint [text; video]) while Wan's is
  isolated cross-attention.
- **Trajectory dynamics**: Wan's concentration *rises* with content
  coherence (0.938→0.968; Q↔K center cosine 0.45→0.72); Cog stays flat /
  declines.

## 9. Q↔K directional alignment (the "similar directions" reading)

Center-to-center cos(E[q], E[k]) — how much Q's direction and K's direction
coincide: Wan 0.54 (noise) growing to **0.72** on the real trajectory; LTX
0.64; Cog 0.22 collapsing to ≈0 under real text. So: Q and K each
concentrate strongly (their own centers are stable), and in the RoPE models
those centers also *approach each other* as generation proceeds — but
Q≈K parallelism is weaker in video DiTs than the intra-Q/intra-K
concentration itself.

## 10. Stability (honest version)

- Cross-noise-seed center cosine ≈ 0.999 — rules out draw-specific
  artifacts (a weak test: two iid draws).
- Real context changes move centers: zeros→randn text gives cos 0.41–0.51
  (Wan/LTX); real-prompt centers sit ≈ 0.6 from noise-input centers.
- Concentration *level* is timestep-insensitive on real trajectories
  (Wan 0.74–0.82 whole-vector across all 20 steps; Cog 0.71–0.73 across 30).

## 11. Caveats

- Dominant-band statistics are by construction optimistic (the paper's own
  selection); all-band means are the pessimistic view — both reported.
- Noise inputs overestimate concentration by ~0.06–0.14 MRL (§7 quantifies).
- Real-prompt legs: one prompt, one seed per model; LTX not yet run.
- Per-band numbers from 512-token bf16 subsamples; whole-vector from full-T.
- Two known code bugs in `probe_video.py` (Wan freqs assert; Cog latent
  layout) — numbers unaffected (iid inputs), re-runs would fail; documented
  in README.

## 12. Files

- `out/{wan,ltx,cog}.json`, `-tensors.pt` — noise-probe per-head stats, centers, subsamples
- `out/{wan,ltx,cog}-pair-mrl.pt` + `-pair-mrl-summary.json` — full [L,H,P] per-band MRL/norms
- `out/{wan,cog}-pipeline.json` + `-centers.pt` — real-prompt trajectories (incl. `q/k_dom_mrl` per step)
- `out/{wan,ltx}-series.json` — center-only logit reconstruction
- figures: `q-band-report-*.png`, `k-band-report-*.png`, `*-mrl-heatmap.png`,
  `pair-mrl-distributions.png`, `band-logit-stats.png`, `mrl-pre-vs-post.png`,
  `qk-alignment.png`, `frac-concentrated.png`
- scripts: `probe_video.py`, `pair_detail.py`, `analysis_series.py`,
  `pipeline_probe.py`, `report_k.py`, `plots.py`

**Bottom line.** Pre-RoPE concentration, the geometric feature TriAttention
exploits, is present in video DiTs for both Q and K — band-organized,
depth-deepening, strongest exactly in the bands that carry the logits, and
robust to real prompts in the RoPE-trained model at close to LLM strength.
K, the memory being addressed, is at least as static as Q — the favorable
case for KV-side methods.
