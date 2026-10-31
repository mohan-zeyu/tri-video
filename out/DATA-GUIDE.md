# DATA-GUIDE — what every file and number in `out/` is

Written 2026-10-31. Companion to `README.md` (findings) — this file explains
**the data itself**: every file, every JSON field, every array shape, and one
example number traced end-to-end.

---

## 0. Model parameters — every shape below is one of these four numbers

All shapes in the result files are dimensions of the model they came from.
Memorize this table and every array in `out/` becomes readable.

| symbol | meaning | Wan2.1-1.3B | LTX-Video-2B | CogVideoX-2b |
|---|---|---|---|---|
| repo | HF id | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `Lightricks/LTX-Video` | `zai-org/CogVideoX-2b` |
| size | transformer params | 1.3 B | ~2 B | ~2 B (the "2b") |
| **L** | self-attn blocks (layers) | **30** | **28** | **30** |
| **H** | attention heads | **12** | **32** | **30** |
| **D** | head dim | **128** | **64** | **64** |
| H×D | hidden width | 1536 | 2048 | 1920 |
| **T** | video tokens probed | **10,800** | **13,728** | **21,600** |
| P = D/2 | rotary pairs per head | 64 | 32 | 32 (no RoPE, see §5) |
| text tokens | cross-attn context | 512 | 128 | 226 (joint stream) |
| token grid | how T arises | latent 12f×60×60, patch (1,2,2) → 12×30×30 | latent 33f×16×26, patch (1,1,1) → 33×16×26 | latent 16f×60×90, spatial patch 2 → 16×30×45 |
| RoPE | 3-axis, interleaved pairs | yes (t/h/w split 44/42/42 of the 128 dims) | yes (10/10/10 pairs + 2 pad dims, per head) | **none** (sinusoidal pos-emb added in patch embed) |
| QK-norm | at capture point | RMSNorm | RMSNorm (across heads) | LayerNorm |

So e.g. "a list of 12 floats in `wan.json`" = one value per head (H=12);
"a list of 32 in `ltx.json`" = same thing (H=32). Total heads in each model:
L×H = **360** (wan), **896** (ltx), **900** (cog).

What was run: the **transformer only** (no VAE, no text encoder) on 5 input
conditions ("runs" below), bf16, on one GPU. Inputs: random N(0,1) latents at
the native token grids above; text embedding = zeros (or randn in one run).
Q/K are captured **after QK-norm, before RoPE** — exactly the tensors the
model feeds into its own `apply_rotary_emb` (verified: recomputing the
attention output from the captures is bit-exact, `validation` field below).

---

## 1. File map

| file | what it is | produced by |
|---|---|---|
| `wan/ltx/cog.json` (1.1–2.5 MB) | **main results**: per-head statistics for 5 runs | `scripts/probe_video.py` |
| `<m>-tensors.pt` (473–591 MB) | raw captured tensors: centers + 512-token subsamples per layer/run | `scripts/probe_video.py` |
| `<m>-series.json` (28–37 KB) | trigonometric-series causal check (paper §3.3 analog) | `scripts/analysis_series.py` |
| `<m>-pair-mrl.pt` (~1 MB) | per-rotary-pair MRL, full [L][H][P] cubes | recompute from `-tensors.pt` (added 10-31) |
| `<m>-pair-mrl-summary.json` (~0.5 KB) | headline stats of the above, incl. dominant-band | same |
| `summary.md` | one table, all models, mid-timestep run | `scripts/plots.py` |
| `wan/ltx/cog-mrl-heatmap.png` | layer×head heatmap of whole-vector Q/K MRL | `scripts/plots.py` |
| `mrl-pre-vs-post.png` | pre- vs post-RoPE MRL histograms (pooled models) | `scripts/plots.py` |
| `qk-alignment.png` | histograms of the three Q↔K cosine stats | `scripts/plots.py` |
| `frac-concentrated.png` | % heads with MRL>0.95, pre vs post, per model | `scripts/plots.py` |
| `pair-mrl-distributions.png` | per-pair MRL distributions (added 10-31) | — |
| `inspect-cog.txt` | one-off structure dump (contains a stale traceback; ignore) | `scripts/inspect_model.py` |

---

## 2. `<model>.json` — the main probe output

```
{
  ...header...                          # scalars
  "runs": [                             # list of 5 — one per input condition
    {
      "tag": "t_mid-zeros-s0",          # which condition (see below)
      "timestep": 500, "text": "zeros", "seed": 0,
      "tokens": T,
      "by_layer": {                     # dict: "0".."L-1"  (30/28/30 entries)
        "0": {
          "tokens": T,
          "q_mrl":              [...],  # list of H floats  (12/32/30)
          "k_mrl":              [...],
          ... every stat below, one float per head ...
          "q_mrl_post":         [...],  # same stats AFTER RoPE (absent for cog)
          ...
          "q_pair_mrl_mean":    [...],  # per-pair MRL averaged over pairs
        },
        "1": {...}, ...
      }
    },
    ... 4 more runs ...
  ],
  "validation": {...},                  # capture correctness proof
  "capture_stage": {...},               # which module was hooked, per layer
  "center_stability_vs_seed": {...},    # list of L floats each
  "center_stability_vs_text": {...},
}
```

### 2a. The 5 runs (`runs[*].tag`)

| tag | timestep | text | latent seed | what it varies |
|---|---|---|---|---|
| `t_mid-zeros-s0` | 500 / 500 / 500 | zeros | 0 | **the reference run — quote this one** |
| `t_high-zeros-s0` | 1000/1000/990 | zeros | 0 | timestep (early denoising) |
| `t_low-zeros-s0` | 100/100/100 | zeros | 0 | timestep (late denoising) |
| `t_mid-randn-s0` | 500 | **randn** | 0 | text conditioning content |
| `t_mid-zeros-s1` | 500 | zeros | **1** | noise realization |

### 2b. Field glossary (inside `by_layer["0"]`)

Every field is `list[H]`: **element h = the statistic for head h of that
layer**, a float in [-1, 1] for cosines, [0, 1] for MRLs, positive reals for
norms. Let `q` ∈ R^{T×D} be that head's captured pre-RoPE queries (one row per
video token), `k` likewise; `q̄ = mean_t q[t]` the center; `q̂ = q/‖q‖` unit rows.

| field | formula | in words | example (`wan` L0 h0) |
|---|---|---|---|
| `q_mrl` | ‖q̄‖ / mean_t‖q[t]‖ | fraction of average query length that survives averaging — 1 = all queries parallel ("concentration") | 0.971 |
| `k_mrl` | same on k | key concentration | 0.839 |
| `q_mean_norm`, `k_mean_norm` | mean_t‖q‖, mean_t‖k‖ | raw scale of Q/K (in bf16 units) | 9.03 / 4.80 |
| `q_center_norm`, `k_center_norm` | ‖q̄‖, ‖k̄‖ | length of the center vector itself | 8.77 / 4.02 |
| `qk_center_cos` | cos(q̄, k̄) | do the Q center and K center point the same way (the "similar directions" test) | 0.833 |
| `qk_cross_cos` | (Σq̂)·(Σk̂)/T² | exact mean cosine over **all T² query–key pairs** | 0.680 |
| `qk_same_pos_cos` | mean_t cos(q[t], k[t]) | diagonal of the pair-cosine matrix | 0.689 |
| `q_split_cos`, `k_split_cos` | cos(mean first T/2, mean second T/2) | does the center from half the video predict the other half (within-run stability) | 0.99999 |
| `q_mrl_post`, `k_mrl_post` | q_mrl after applying the model's own RoPE to q | the contrast: rotation should destroy concentration | 0.638 |
| `*_post` (all) | same stat on rotated tensors | | |
| `q_pair_mrl_mean` | mean over the P rotary pairs of ‖E x^(i)‖/E‖x^(i)‖ | the paper's per-frequency-band MRL, averaged over bands (stricter than `q_mrl`) | 0.505 |
| `k_pair_mrl_mean` | same on k | | 0.414 |

Note `q_mean_norm_post == q_mean_norm` (rotation preserves norms) — that's a
sanity check, not a bug.

**Cog exception:** cog.json has **no `_post` fields** — this checkpoint runs
without RoPE, so there is no rotation to apply.

### 2c. `validation` — why you can trust the capture

```
"layer0": {"max_abs_err": 0.0, "output_scale": 15.5, "rel_err": 0.0}
```
For layers {0, L/2, L−1} the whole attention module output was recomputed
from the captured q, k, v + replicated rotation and compared to the module's
real output. `rel_err = max_abs_err / output_scale`. **0.0 = bit-exact** in
all 9 checks (3 layers × 3 models) ⇒ the stats are computed on the true
pre/post-RoPE tensors, not approximations.

### 2d. `center_stability_vs_seed` / `_vs_text`

`{"q": [30 floats], "k": [30 floats]}` — one float per **layer** (mean cosine
over that layer's heads) between the reference run's centers and the rerun's
centers (seed-1 run / randn-text run respectively). ≈1 ⇒ same center
direction.

### 2e. Worked example — reading one number end-to-end

`wan.json → runs[0].by_layer["3"].q_mrl[7] = 0.93` means:
in Wan2.1-1.3B (30 layers × 12 heads), run the mid-timestep zeros-text forward,
take **layer 3, head 7**, gather its 10,800 pre-RoPE query vectors (each
128-d), average them, and 0.93 is ‖average‖ ÷ average-norm — i.e. replacing
every query of that head by one fixed vector keeps 93% of the norm.

---

## 3. `<m>-tensors.pt` — the raw data behind the stats

```
{run_tag: {layer_int: {
    "_center_q":    [H, D]     bf16     # E_t[q]  per head (the centers)
    "_center_k":    [H, D]     bf16
    "_subsample_q": [H, 512, D] bf16    # q at 512 random tokens (histograms, recompute)
    "_subsample_k": [H, 512, D] bf16
}}}
```
- 5 runs × L layers each; subsample = 512 uniformly random token positions
  (drawn with `torch.manual_seed(0)`).
- Sizes: wan 473 MB (30L×12H×128d), ltx 589 MB (28×32×64), cog 591 MB
  (30×30×64 — biggest because H×L is largest).
- This is the file to load if you want to recompute anything not already in
  the JSONs (the per-pair cubes in §5 were derived from these subsamples).
- Load with `torch.load(..., mmap=True)` — the files are big.

---

## 4. `<m>-series.json` — the causal check ("do centers predict logits?")

PAPER-CLAIM TEST: if Q/K concentrate, substituting every token by its center
should reproduce real attention logits via rotation-angle differences — a
trigonometric series in the token offset. Produced by
`scripts/analysis_series.py` from a fresh t=500 zeros forward, 4 layers per
model (wan: 0,5,15,29 / ltx: 0,5,14,27).

```
"layers": [0, 5, 15, 29]              # which layers were analyzed
"queries": 96                          # sampled query tokens per head
"heads": [                             # list of 4×H dicts (48 for wan, 128 for ltx)
   {"layer": 0, "head": 0, "recon_r": 0.8303, "q_mrl": 0.9709}, ...
]
"summary": {"mean_r": 0.7522, "median_r": 0.8303,
            "frac_r>0.5": 0.7917, "frac_r>0.7": 0.6875}
"marginals": {                          # for the most-concentrated head of each layer
   "L0H4": {
     "dt": [[-10, -1.5496, 900], ...],  # rows: [Δ in frames, mean actual logit, #keys]
     "dh": [[...], ...],                # Δ in latent rows  (12 dt / 30 dh / 30 dw rows for wan)
     "dw": [[...], ...]
   }, ...
}
```

- `recon_r` per head: Pearson correlation, for 96 sampled queries, between the
  query's true logit row (over all T keys) and the center-only prediction
  Σ_f Re(conj(q̄_f) e^{i(θ_s−θ_t)} k̄_f)/√D, averaged over queries.
- `marginals`: actual logit of the best head's first sampled query, averaged
  over keys binned by signed frame offset (dt), row offset (dh), column offset
  (dw) — the "attention vs distance" curves of the paper's Fig 2(D), one
  query's worth.
- Caveat (kept from the review): correlates over all keys, not the paper's
  log-spaced distance grid; 4 layers, one noise input.

## 5. `<m>-pair-mrl.pt` + `<m>-pair-mrl-summary.json` — per-pair MRL cubes

The paper's MRL is defined **per rotary frequency band**. These files (added
2026-10-31, computed from the 512-token subsamples of §3) hold it explicitly:

```
"layers": [0..L-1]                # list of L ints
"heads": H                        # 12 / 32 / 30
"pairs": P                        # 64 (wan) / 32 (ltx) / 32 (cog)
"q_pair_mrl":  list[L][H][P]      # MRL of pair p of head h, layer l ∈ [0,1]
"k_pair_mrl":  list[L][H][P]
"q_pair_norm": list[L][H][P]      # mean 2D norm of the pair (for dominant-band selection)
```
Rotary pair p of head h = coordinates (x_{2p}, x_{2p+1}) of that head's
128/64-dim vectors, treated as one 2D arrow per token; its MRL is
‖E_t x^(p)‖ / E_t‖x^(p)‖. For wan the first pairs belong to the time axis,
then height, then width (t/h/w = 44/42/42 dims → 22/21/21 pairs); for ltx
10 t + 10 h + 10 w pairs + 2 padded dims; cog has no rope axes, hence its
summary's single `"none"` axis entry.

`-pair-mrl-summary.json` fields — **mind the three denominators**:

| field | denominator | wan | ltx | cog |
|---|---|---|---|---|
| `q_pair_mrl_mean_over_all` | all L×H×P cells | 0.785 | 0.694 | 0.819 |
| `q_frac>0.95_allpairs` | all L×H×P cells | 26.9% | 24.4% | 40.4% |
| `dom_q_mrl_mean` | L×H heads (each represented by its dominant band) | 0.972 | 0.922 | 0.921 |
| `dom_q_frac>0.95` | L×H heads | 79.7% | 77.6% | 75.9% |
| `per_axis_mean` | cells, grouped by rope axis | t/h/w per model | t .60, h .72, w .77 | — |

(Compare with the JSONs' whole-vector `q_mrl` mean: 0.857/0.810/0.854 and its
">0.95" head-fraction 16%/4%/32% — **four different denominators give four
different "fractions above 0.95"; never mix them**. The paper's headline
"~90% of heads R>0.95" is closest in spirit to `dom_*_frac>0.95`.)

---

## 6. Figures

- `<m>-mrl-heatmap.png` — 30×12 (28×32, 30×30) grid of `q_mrl`/`k_mrl`, color
  = whole-vector MRL 0→1 (blue ramp). Rows = layers, cols = heads.
- `mrl-pre-vs-post.png` — histograms of per-head MRL pooled over models, blue
  pre-RoPE / orange post-RoPE (cog appears only in blue: no RoPE).
- `qk-alignment.png` — histograms of `qk_center_cos` / `qk_same_pos_cos` /
  `qk_cross_cos` (pooled).
- `frac-concentrated.png` — % of heads with whole-vector q-MRL>0.95, pre vs
  post per model; dashed line = the paper's Qwen3-8B reference.
- `pair-mrl-distributions.png` — per-pair MRL distribution (all cells).

## 7. Reading recipes

```python
import json, torch
r = json.load(open("out/wan.json"))
# layer 3, head 7 MRL, reference run:
print(r["runs"][0]["by_layer"]["3"]["q_mrl"][7])
# mean q-MRL over all 360 heads, reference run:
import statistics as st
print(st.mean(v for l in r["runs"][0]["by_layer"].values() for v in l["q_mrl"]))

t = torch.load("out/wan-tensors.pt", mmap=True)
q = t["t_mid-zeros-s0"][3]["_subsample_q"]        # [12, 512, 128] bf16
centers = torch.stack([t["t_mid-zeros-s0"][i]["_center_q"] for i in range(30)])  # [30,12,128]
```
