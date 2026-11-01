# Pre-RoPE Q/K geometry in video diffusion transformers

Verification of the TriAttention paper's Observation 3.1 (arXiv 2604.04921:
pre-RoPE Q/K vectors concentrate around non-zero centers, stable across
positions and contexts) on **video generation** DiTs, where self-attention uses
3D (t, h, w) RoPE instead of 1D positional RoPE. Run date: 2026-10-29;
per-pair analysis added 2026-10-31.

## Verdict

**The feature is present in video DiTs, and on the paper's own metric
(dominant-band MRL) it is close to LLM strength.**

| # | claim | video DiT result |
|---|---|---|
| 1 | pre-RoPE Q/K concentrate (paper metric: per-band MRL) | **Yes** — dominant-pair MRL mean 0.92–0.97, median 0.98–0.99; 76–80% of heads > 0.95 (paper, Qwen3-8B: mean 0.977, 84.7% > 0.95 per Table G). Over *all* bands, mean R_f = 0.69–0.82 |
| 2 | centers stable | **Yes, with caveats** — split-half 0.85–0.99; cross-noise-seed ≈ 0.999 (weak test: two iid draws); real input change (text swap) moves Wan/LTX centers to cos 0.41–0.51 |
| 3 | rotation destroys the shared direction *(our extension — not quantified in the paper)* | **Yes** — whole-vector MRL drops 0.86→0.63 (Wan), 0.81→0.20 (LTX) |
| 4 | Q and K point in similar directions *(our extension — not claimed in the paper)* | **Partially** — center-to-center cos 0.54 (Wan) / 0.64 (LTX) / 0.22 (Cog), positive but moderate |
| 5 | centers predict attention logits (trig series, §3.3) | **Yes** — r̄ = 0.75 mean / 0.83 median (Wan), 0.72 / 0.81 (LTX) over 4 layers. *Protocol differs from the paper* (all keys vs their log-spaced distance grid), so not directly comparable to their LLM numbers |

## Models

| key | model | arch | positional mechanism | capture point |
|---|---|---|---|---|
| wan | Wan2.1-T2V-1.3B | 30L × 12H × 128d | 3-axis interleaved RoPE; pair split t 22 / h 21 / w 21 | norm_q/norm_k (RMSNorm) |
| ltx | LTX-Video (2B) | 28L × 32H × 64d | 3-axis interleaved RoPE over flattened stream (1024 pairs: t/h/w blocks of 341 + 1 pad); each head owns a contiguous 32-pair slice | norm_q/norm_k (RMSNorm across heads) |
| cog | CogVideoX-2b | 30L × 30H × 64d | **no RoPE** (sinusoidal pos-emb; joint [text; video] stream) | norm_q/norm_k (LayerNorm), video tail only |

Probing setup: transformer-only forwards (no text encoder / VAE), bf16,
random-N(0,1) latents (T = 10,800 / 13,728 / 21,600 video tokens), zeros text
embedding (= the true CFG-uncond input for **Wan only**; LTX masks padding and
Cog's padding is not zero, so zeros are merely a fixed null-ish input there),
sweeps over timestep {high, mid, low}, text {zeros, randn}, seeds {0, 1}.

**Capture validation**: for 3 layers per model the attention output was
recomputed from captured pre-RoPE tensors + replicated rotation and matched
the module's real output **bit-exactly (rel_err 0.0, 9/9 checks)** — the stats
below are on exactly the tensors entering the model's `apply_rotary_emb`.

## Headline numbers (mid timestep, zeros text; mean / median over layer×head)

Whole-vector MRL and cosines are from `out/<m>.json` (full-token, fp32):

| model | q MRL pre | k MRL pre | q MRL post | cos(Q̄,K̄) | E cos(q_t,k_s) |
|---|---|---|---|---|---|
| Wan2.1-1.3B | 0.857 / 0.873 | 0.878 / 0.889 | 0.631 | 0.542 / 0.585 | 0.412 |
| LTX-Video-2B | 0.810 / 0.834 | 0.793 / 0.822 | 0.204 | 0.642 / 0.758 | 0.444 |
| CogVideoX-2b (no RoPE) | 0.854 / 0.914 | 0.879 / 0.940 | n/a | 0.224 / 0.393 | 0.136 |

Per-rotary-pair MRL (paper's metric; from 512-token bf16 subsamples,
`out/<m>-pair-mrl.pt` via `scripts/pair_detail.py`):

| model | mean R_f, all pairs | (head,pair) cells > 0.95 | dominant-pair MRL mean / median | heads dom > 0.95 |
|---|---|---|---|---|
| Wan2.1-1.3B | 0.785 | 27% | **0.972 / 0.992** | **80%** |
| LTX-Video-2B | 0.694 | 24% | **0.922 / 0.987** | **78%** |
| CogVideoX-2b | 0.819 | 40% | **0.921 / 0.984** | **76%** |
| *(paper, Qwen3-8B)* | *—* | *—* | *0.977 mean* | *84.7% (Table G)* |

Dominant pair = top band by ‖q_f‖·‖k_f‖ (the paper's selection rule); the
statistic is by construction optimistic — the all-pairs row is the honest
full distribution, the dominant row is the paper-comparable one.

## Real-prompt pipeline verification (2026-10-31)

Full pipelines (real T5/UMT5 text encoding, real scheduler trajectories, CFG
positive branch, validation again bit-exact 0.0 in all checks):

**Wan2.1-1.3B** ("A stylish woman walks down a Tokyo street…", 45f 480², 20
steps, `out/wan-pipeline.json`):

| step | q MRL | k MRL | q MRL post | cos(Q̄,K̄) | E cos(q,k) |
|---|---|---|---|---|---|
| 0 (pure noise) | 0.776 | 0.861 | 0.539 | 0.452 | 0.315 |
| 10 | 0.754 | 0.747 | 0.477 | 0.685 | 0.397 |
| 19 (nearly clean) | 0.816 | 0.778 | 0.513 | 0.720 | 0.467 |

- Concentration **survives real generation** — q MRL stays 0.74–0.82 across
  the whole trajectory (noise-input run said 0.857: noise inputs overestimate
  concentration by ~0.08, but the phenomenon is not an artifact).
- Post-RoPE collapse intact (0.48–0.54 vs pre 0.74–0.82 at every step).
- **Q↔K alignment grows as video content emerges**: cos(Q̄,K̄) rises
  0.45→0.72 and all-pairs cos 0.32→0.47 along the trajectory — the "Q and K
  point in similar directions" aspect strengthens with real content.
- Centers transfer moderately from noise inputs to real prompts
  (cos ≈ 0.59–0.65 per head).
- **Dominant-band MRL (paper metric) on the real trajectory**: mean 0.954
  over all steps, **73% of heads > 0.95, rising to 0.968 / 87% at the last
  step** — close to Qwen3-8B (0.977, 84.7% > 0.95) at full realism.

**CogVideoX-2b** (panda-dancing prompt, 49f 480×720, 30 steps,
`out/cog-pipeline.json`): q MRL flat at 0.71–0.73 across all 30 steps, k
0.79–0.84, per-pair 0.67–0.69 — the no-RoPE control keeps its concentration
under real inputs too, while its weak Q↔K alignment mostly disappears with
real text (all-pairs cos ≈ 0). Its dominant-band MRL however drops to 0.831
(39% of heads > 0.95) versus 0.921 (76%) under noise inputs — unlike Wan,
the no-RoPE model does **not** maintain dominant-band concentration under
real conditioning, which sharpens the association between strong band-level
concentration and RoPE-trained models.

Conclusions the real-prompt runs change: the earlier noise-only caveat is now
resolved directionally (everything holds, with modestly smaller magnitudes);
and the honest statement about "stable across contexts" becomes: centers are
*the same kind of object* across inputs (cos 0.6 transfer) but not identical —
input content rotates/refines them, concentration level stays within a band.

## Notable observations

- **On the paper's dominant-band metric, video DiTs are near LLM strength**
  (0.92–0.97 mean, 76–80% of heads > 0.95 vs Qwen3-8B's 84.7%). Concentration
  lives primarily in the high-amplitude bands; low-amplitude bands spread.
- **Axis structure (LTX)**: per-axis mean R_f is **w 0.77 > h 0.72 > t 0.60** —
  temporal frequency bands are the least concentrated. Wan's t/h/w splits are
  flat (0.78–0.79).
- **The no-RoPE control also concentrates** (Cog 0.85/0.88 whole-vector,
  0.92 dominant) — concentration is not *caused* by RoPE; RoPE just rotates
  the concentrated mass apart. Cog's Q↔K alignment is weaker (0.22).
- **Text conditioning moves the centers** in the RoPE models (zeros→randn cos
  0.41–0.51); with OOD randn text, Wan/LTX Q/K collapse to MRL ≈ 0.99 — real
  conditioning adds diversity around the center, garbage conditioning reveals
  the weights' default direction.
- **Trig-series prediction works** (r̄ ≈ 0.72–0.75 mean over all keys, 4
  layers, noise inputs): to first order, video-DiT attention logits are a
  function of spatiotemporal offset encoded in the Q/K centers.

## Caveats & known issues

- Noise-input results overestimate concentration by roughly 0.06–0.14 MRL
  relative to real prompts (see the real-prompt section above); direction of
  every conclusion is unchanged.
- The real-prompt legs use one prompt per model, one seed, CFG positive
  branch, and skip VAE decode (`output_type="latent"`); single-prompt scope.
- **Two code bugs in `probe_video.py`** (found in review; numbers unaffected
  since inputs are iid noise, but re-runs would fail):
  the Wan freqs assert should be `freqs[0].shape[1] == T` (Wan freqs are
  `[1,T,1,2D]`), and Cog latents must be laid out `[B,F,C,H,W]` (masked by
  C=F=16 at 61 frames). `wan.json` was produced by the pre-final script
  version with the correct assert. (`pipeline_probe.py` feeds latents via the
  real pipeline and is unaffected.)
- Per-pair numbers come from 512-token bf16 subsamples, not full-T fp32.
- Cog's `t_high=990` actually executes as 992 in bf16 (cosmetic).

## Files

- `scripts/probe_video.py` — main probe (hooks, MRL/cosine stats pre/post
  RoPE, capture validation, sweeps). `--model {wan,ltx,cog} --out ../out/<m>.json`
- `scripts/pair_detail.py` — full per-pair MRL / dominant-pair / per-axis
  analysis from stored subsamples → `out/<m>-pair-mrl.pt` + summary JSON.
- `scripts/analysis_series.py` — §3.3 causal check adapted to 3D RoPE.
- `scripts/pipeline_probe.py` — real-prompt full-pipeline probe (T5/UMT5
  encoding, denoising-trajectory stats via callback, latent output) →
  `out/<m>-pipeline.json` + `-centers.pt`.
- `scripts/fetch_small.py` — git-blob-accurate small-file fetcher (hub's
  snapshot_download hangs on this network).
- `scripts/inspect_model.py`, `scripts/plots.py`, `scripts/multidl.py` —
  structure dump, figures, mirror downloader.
- `out/` — `<m>.json` (per-head stats), `<m>-tensors.pt` (centers + 512-token
  subsamples), `<m>-pair-mrl.pt` (full [L,H,P] R_f + norms + dominant idx),
  `<m>-pair-mrl-summary.json`, `<m>-series.json`, `<m>-pipeline.json`
  (real-prompt trajectories), figures, `summary.md`.

## Reproducibility

- env: torch 2.11.0+cu128, diffusers 0.42.1 (venv at `/dev/shm/trivenv` is
  volatile; recreate via the uv cache on /mnt/data — wheels fully cached)
- weights: HF cache `/mnt/data/liuzeyu/hf-cache` (transformer subfolders,
  sha256-verified)
- seeds fixed; run configs recorded in each JSON header.
