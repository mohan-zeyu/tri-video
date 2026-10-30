# Pre-RoPE Q/K geometry in video diffusion transformers

Verification of the TriAttention paper's Observation 3.1 (arXiv 2604.04921:
pre-RoPE Q/K vectors concentrate around non-zero centers — "point in similar
directions" — stable across positions and contexts) on **video generation**
DiTs, where self-attention uses 3D (t, h, w) RoPE instead of 1D positional
RoPE. Run date: 2026-10-29.

## Verdict

**The feature is present in video DiTs, at moderate strength, and it carries
the paper's causal consequence.**

| # | paper claim (LLMs) | video DiT result |
|---|---|---|
| 1 | pre-RoPE Q/K concentrate (MRL → 1) | **Yes** — mean MRL 0.79–0.88 across all heads of all 3 models (LLMs: 0.98) |
| 2 | centers stable across contexts | **Yes** — cross-noise-seed center cosine ≈ 0.999; MRL timestep-insensitive |
| 3 | rotation destroys the shared direction | **Yes** — post-RoPE MRL drops to 0.63 (Wan) and 0.20 (LTX) |
| 4 | Q and K point in similar directions | **Partially** — center-to-center cos 0.54 (Wan) / 0.64 (LTX), positive but weaker than LLMs |
| 5 | centers predict attention logits (trig series, §3.3) | **Yes, strongly** — Pearson r of center-only prediction vs actual logits: **0.75 mean / 0.83 median (Wan), 0.72 / 0.81 (LTX)**; 69–70% of heads with r > 0.7 (paper's LLM range: mean > 0.5, peaks 0.6–0.9) |

## Models

| key | model | arch | positional mechanism | capture point |
|---|---|---|---|---|
| wan | Wan2.1-T2V-1.3B | 30L × 12H × 128d | 3-axis interleaved RoPE (t/h/w subspaces of head dim) | norm_q/norm_k (RMSNorm) |
| ltx | LTX-Video (2B) | 28L × 32H × 64d | 3-axis interleaved RoPE over flattened stream | norm_q/norm_k (RMSNorm across heads) |
| cog | CogVideoX-2b | 30L × 30H × 64d | **no RoPE** (sinusoidal pos-emb in patch embed; joint [text; video] stream) | norm_q/norm_k (LayerNorm), video tail only |

Probing setup: transformer-only forward passes (no text encoder / VAE), bf16,
random-N(0,1) latents at native-resolution-class grids (T = 10,800 / 13,728 /
21,600 video tokens), zero text conditioning (= the CFG-uncond branch), sweeps
over timestep {high, mid, low}, text {zeros, randn}, noise seeds {0, 1}.

**Capture validation**: for 3 layers per model the attention output was
recomputed from the captured pre-RoPE tensors + replicated rotation and
compared with the module's real output — max relative error **0.0 (bit-exact)
in all 9 checks**, so the statistics below are computed on exactly the tensors
that enter the model's own `apply_rotary_emb`.

## Headline numbers (mid timestep, zeros text)

| model | q MRL pre | k MRL pre | q MRL post | k MRL post | cos(Q̄,K̄) | E cos(q_t,k_s) | % heads q-MRL>0.95 |
|---|---|---|---|---|---|---|---|
| Wan2.1-1.3B | 0.857 / 0.873 | 0.878 / 0.889 | 0.631 | 0.607 | 0.542 / 0.585 | 0.412 | 16% |
| LTX-Video-2B | 0.810 / 0.834 | 0.793 / 0.822 | 0.204 | 0.167 | 0.642 / 0.758 | 0.444 | 4% |
| CogVideoX-2b (no RoPE) | 0.854 / 0.914 | 0.879 / 0.940 | n/a | n/a | 0.224 / 0.393 | 0.136 | 32% |
| *(paper, Qwen3-8B)* | *0.977–0.98* | *—* | *→~0* | *—* | *high* | *—* | *~90%* |

(cells are mean / median over all layer×head entries)

## Notable observations

- **Concentration is weaker than in LLMs but unambiguous and universal.** Every
  model shows mean MRL 0.79–0.88 with split-half center stability 0.82–0.99;
  Wan and Cog have >25% of heads above 0.95. Video tokens carry far more
  content diversity than autoregressive text, which plausibly costs MRL.
- **The no-RoPE control also concentrates** (CogVideoX, 0.85/0.88). So
  concentration is not *caused* by RoPE; RoPE just rotates the concentrated
  mass apart (claim 3). Cog's Q↔K alignment is however much weaker (0.22) —
  the models that *do* use RoPE align Q and K more.
- **Text conditioning moves the centers** in the RoPE models (zeros→randn
  center cosine 0.41–0.51) while CogVideoX is text-insensitive (0.99). With
  out-of-distribution randn text, Wan/LTX Q/K collapse onto a single direction
  (MRL → 0.99): real conditioning adds diversity around the center, garbage
  conditioning reveals the bare "default direction" of the weights.
- **The trigonometric-series mechanism transfers unchanged** — replacing the
  paper's 1D distance with the 3D rotation-angle difference (angles read
  directly from the model's own rope tensors) yields logit predictions that
  correlate at LLM-level strength (r̄ ≈ 0.72–0.75 mean, 0.81–0.83 median).
  Attention in video DiTs is, to first order, a function of spatiotemporal
  offset encoded in the Q/K centers.

## Files

- `scripts/probe_video.py` — main probe (hooks, MRL/cosine stats pre/post RoPE,
  capture validation, sweeps). `--model {wan,ltx,cog} --out ../out/<m>.json`
- `scripts/analysis_series.py` — paper §3.3 causal check adapted to 3D RoPE
  (center-only logit prediction, per-head Pearson r, offset marginals).
- `scripts/inspect_model.py` — structure/signature dump per adapter.
- `scripts/plots.py` — figures + `out/summary.md` (dataviz palette).
- `scripts/multidl.py` — multi-range HF mirror downloader used here.
- `out/` — `<m>.json` (per-head stats), `<m>-tensors.pt` (centers + 512-token
  subsamples per layer), `<m>-series.json` (causal check), figures, summary.

## Reproducibility

- env: torch 2.11.0+cu128, diffusers 0.42.1 (venv in /dev/shm/trivenv —
  volatile; recreate with `uv pip install torch --index-url
  https://download.pytorch.org/whl/cu128` + `diffusers transformers
  accelerate safetensors matplotlib`)
- weights: HF cache `/mnt/data/liuzeyu/hf-cache` (transformer subfolders only,
  sha256-verified)
- seeds fixed (torch.manual_seed(0), per-input generators); every run's config
  recorded in the JSON headers.
