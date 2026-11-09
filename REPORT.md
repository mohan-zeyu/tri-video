# Complete Wan 3D RoPE experiment

[PDF report](output/pdf/wan-full-sweep-report.pdf) · [Browse every figure](http://127.0.0.1:17841/index.html) · [Per-head CSV](out/full-sweep-wan-20261101/per-head-results.csv)

Four real prompts · 20 denoising steps per prompt · every layer/head/axis · all tokens

## Scope and provenance

Wan2.1-T2V-1.3B, 45 frames at 480×480, CFG 5.0 and seed 0. Four prompts share the same initial noise, isolating prompt changes at this seed. Each step has a 12×30×30 grid of 10,800 video tokens. Video time and diffusion timestep are distinct coordinates. Every conditional transformer input was matched exactly to its supplied positive text embedding; unconditional CFG forwards were excluded from captures.

## Measurement definitions

Each primary figure belongs to one prompt, one denoising step, one layer, one head and one axis. No step/prompt pooling occurs there. The 2D scatter uses every token in the strongest frequency pair belonging to that axis, selected by mean_token(||q_f||) × mean_token(||k_f||). Pre/post plots use the identical pair and tokens. Pair R = ||mean_token(x_f)|| / mean_token(||x_f||). Distributions contain the 12 heads of this one layer/step/prompt/axis, separately for Q/K and pre/post.

Logit reconstruction uses the entire 128D head, summing all 64 pairs. Measured logits are recomputed with fp32 dot products from the captured Q/K after native bf16 RoPE rounding. The prediction rotates fp32 centers without bf16 rounding. Those Q/K centers are fp32 averages over all 10,800 video tokens at this same prompt/step/layer/head. Every token acts as a query; keys range over its whole axis line, fixing the other two coordinates. This evaluates 129,600 temporal and 324,000 height or width logits per head/layer/step/prompt. Both signed offset directions and self are included. These are pre-softmax logits, not attention probabilities.

Primary reconstruction curves average the measured and predicted logits over all query–key pairs at each signed offset. Shading is the population standard deviation of individual measured logits at that offset. The head reconstruction score first correlates each individual query's measured and predicted logit vectors over its same-axis keys, then averages the valid query correlations. This differs from correlating the two already-averaged curves; both statistics and valid-query counts are retained. The single origin-query curves are also retained separately.

## Coverage and validation

Completed: 2,400 layer captures, 7,200 all-head sheets, 86,400 individual head figures and 7,200 concentration distributions. Each individual head figure has pre/post scatter plus logit reconstruction. Full bf16 Q/K are retained for every layer/step/prompt (159.3 GB). All outputs are indexed by prompt, step, layer, axis and head.

Attention-output recomputation checks: 144, covering all 30 layers at the first step of each prompt and layers 0/15/29 at steps 10/20. 144 checks are bit-exact; maximum relative error is 0. First-step, first-layer Q centers are identical across all prompts, as expected before cross-attention when initial noise and timestep are identical. Numerical tests independently compare axis logit matrices, signed-offset counts and per-query correlations against explicit calculations.

## Results

Across the six Q/K-by-axis summaries, mean selected-2D-pair concentration ranges from 0.850 to 0.927. Mean full-128D-head per-query reconstruction correlation is 0.569 / 0.728 / 0.612 for t / h / w. Each axis summary pools 4 prompts × 20 steps × 30 layers × 12 heads; every underlying pair concentration or query evaluation uses all 10,800 video tokens at its individual prompt/step/layer/head.

The following table pools **4 prompts × 20 steps × 30 layers × 12 heads = 28,800 cells per axis**. R columns average selected-2D-pair MRL values. The r column averages each cell's mean over its valid token-query correlations, using full-head logits. It does not use the correlation of the plotted mean curves. These are descriptive pooled summaries; primary per-step/per-head data remain separate.

| Axis | Q pair R, pre | K pair R, pre | Q pair R, post | K pair R, post | Mean query r |
|---|---:|---:|---:|---:|---:|
| t | 0.8950 | 0.8495 | 0.5334 | 0.4815 | 0.5689 |
| h | 0.9271 | 0.9070 | 0.3562 | 0.3259 | 0.7284 |
| w | 0.9114 | 0.8915 | 0.5278 | 0.5001 | 0.6117 |

![Concentration trajectories](out/full-sweep-wan-20261101/summary/concentration-trajectories.png)

Each colored line is one prompt. Each point averages selected-pair R over 30 layers and 12 heads at one denoising step; Q and K and the three axes are separated. Every underlying pair R uses all 10,800 tokens. The dominant pair is reselected separately at each prompt/step/layer/head/axis; these curves do not track one fixed band through the trajectory.

![Reconstruction trajectories](out/full-sweep-wan-20261101/summary/reconstruction-trajectories.png)

Each colored line is one prompt. Each point averages the full-head per-query reconstruction score over 30 layers and 12 heads, at one step and one axis. Its underlying query correlations use every token as a query and all same-axis-line keys.

## 4. Complete A/B/C/D examples

Following TriAttention Figure 2: **A = pre-RoPE pairs, B = post-RoPE pairs, C = across-head concentration, D = logit prediction**. Each example combines all four panels at one prompt, denoising step, layer and axis.

A/B plot every token's Q/K in one axis-dominant 2D pair, with the same pair, token IDs and scale. C shows the pre-RoPE pair-MRL distribution over the 12 heads at this one prompt, step, layer and axis: **each head chooses its own dominant pair by C_f**; each R uses all 10,800 tokens. Q and K each contribute 12 values, so one head is 8.33%. C does not pool layers, steps or prompts. The archive also retains post-RoPE distributions.

D uses the **full 128D head, all 64 pairs**, with Q/K centers averaged over all 10,800 tokens at this same prompt/step/layer/head. Curves average all same-axis-line query-key logits per signed offset; shading is measured population standard deviation. The displayed r first correlates each query over its axis-line keys and then averages all 10,800 valid queries. It is not the correlation between the two plotted mean curves.

### 4.1 Fixed examples

Retained from the previous report: p03, denoising step 20, layer 15, head 0, separately for t/h/w. Layer, head and pair indices are zero-based; denoising steps are numbered 1-20.

![Fixed t example with A/B/C/D](output/full-sweep-wan-20261101/report-figures/fixed-t-abcd.png)

t: pair 12; pair R_Q=0.791980, R_K=0.802809; full-head mean query r=0.695480.

![Fixed h example with A/B/C/D](output/full-sweep-wan-20261101/report-figures/fixed-h-abcd.png)

h: pair 39; pair R_Q=0.938053, R_K=0.025757; full-head mean query r=0.725007.

![Fixed w example with A/B/C/D](output/full-sweep-wan-20261101/report-figures/fixed-w-abcd.png)

w: pair 53; pair R_Q=0.721710, R_K=0.615983; full-head mean query r=0.492262.

The fixed examples underestimate the zero-offset peak. Pearson r measures shape agreement, not matching logit magnitudes. Concentration in one selected 2D pair and reconstruction using the full head are different measurements.

### 4.2 Additional best reconstruction examples

For 'best performance', rank **full-head mean per-query r**, separately for each axis, over 4 prompts x 20 steps x 30 layers x 12 heads = **28,800 cells per axis**, requiring all 10,800 queries to have defined correlations. The displayed pair is this winning head's C_f-selected axis-dominant pair; r evaluates the full head, not that isolated pair. These are maxima selected after the sweep, not representative performance.

| Axis | Prompt | Step | Layer | Head | Selected pair | Pair R_Q | Pair R_K | Full-head mean query r |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t | p01 | 17 | 0 | 5 | 0 | 0.999739 | 0.999935 | 0.998670 |
| h | p01 | 12 | 29 | 9 | 24 | 0.999989 | 0.999990 | 0.999360 |
| w | p03 | 2 | 29 | 9 | 47 | 0.999999 | 0.999989 | 0.999973 |

![Best full-head reconstruction along t](output/full-sweep-wan-20261101/report-figures/best_reconstruction-t-abcd.png)

![Best full-head reconstruction along h](output/full-sweep-wan-20261101/report-figures/best_reconstruction-h-abcd.png)

![Best full-head reconstruction along w](output/full-sweep-wan-20261101/report-figures/best_reconstruction-w-abcd.png)

### 4.3 Best concentrated pair, ranked separately

If 'best pair' means both 2D Q and K are concentrated, maximize min(R_Q,R_K) among the same 28,800 **axis-dominant selected pairs** per axis. This does not search every non-dominant band. Each R is pre-RoPE and averages all tokens within one condition. The winners differ from the best full-head predictions:

| Axis | Prompt | Step | Layer | Head | Pair | Pair R_Q | Pair R_K |
|---|---|---:|---:|---:|---:|---:|---:|
| t | p01 | 20 | 0 | 5 | 0 | 0.999806 | 0.999908 |
| h | p01 | 4 | 29 | 9 | 24 | 0.999992 | 0.999991 |
| w | p03 | 3 | 28 | 11 | 48 | 1.000000 | 0.999998 |

[All nine A/B/C/D figures, including the strongest-pair examples](output/full-sweep-wan-20261101/report-figures/index.html). [Exact selection definitions, source hashes and values](output/full-sweep-wan-20261101/report-figures/selection.json). No denoising-step or prompt averaging enters any selected example.

## 5. Layer-by-head MRL across frequency bands

Each heatmap has layer on the y-axis and head on the x-axis. Q and K have separate panels. Each underlying pair MRL uses all 10,800 tokens at one prompt, denoising step, layer and head. Band cells are arithmetic means of these 2D-pair MRL values over the explicitly listed pair IDs.

High, middle and low frequencies are nearly equal-count thirds sorted by the actual saved RoPE angular frequency, separately within t/h/w. Temporal groups contain 8/7/7 pairs, and height/width groups contain 7/7/7 pairs. The color scale is fixed at 0-1 across all views. The website also exposes individual-pair values, pre/post stages and click-to-select layer/head cells for all 80 prompt/step conditions.

[Complete heatmap appendix](output/full-sweep-wan-20261101/wan-mrl-heatmaps-complete.pdf). All 160 landscape A3 pages are indexed by prompt, step and stage. Each page contains all three axes, both Q/K, all 30 layers and 12 heads, and the three frequency groups.

The following figures retain the fixed p03, step 20 example. Each cell preserves layer/head identity and averages only its stated frequency pairs. The accompanying tables pool 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 band-mean cells per entry, separately by axis, Q/K and stage.

![t-axis layer-by-head pair MRL](output/full-sweep-wan-20261101/mrl-heatmaps/figures/p03-step-020-t-pre.png)

| Axis | Band | Pair IDs | Q pre | K pre | Q post | K post |
|---|---|---|---:|---:|---:|---:|
| t | high | 0-7 | 0.7187 | 0.7476 | 0.3910 | 0.4061 |
| t | middle | 8-14 | 0.5448 | 0.5361 | 0.5443 | 0.5353 |
| t | low | 15-21 | 0.5684 | 0.5436 | 0.5684 | 0.5436 |

![h-axis layer-by-head pair MRL](output/full-sweep-wan-20261101/mrl-heatmaps/figures/p03-step-020-h-pre.png)

| Axis | Band | Pair IDs | Q pre | K pre | Q post | K post |
|---|---|---|---:|---:|---:|---:|
| h | high | 22-28 | 0.8587 | 0.8587 | 0.2251 | 0.2194 |
| h | middle | 29-35 | 0.5417 | 0.5342 | 0.5284 | 0.5181 |
| h | low | 36-42 | 0.5725 | 0.5478 | 0.5724 | 0.5478 |

![w-axis layer-by-head pair MRL](output/full-sweep-wan-20261101/mrl-heatmaps/figures/p03-step-020-w-pre.png)

| Axis | Band | Pair IDs | Q pre | K pre | Q post | K post |
|---|---|---|---:|---:|---:|---:|
| w | high | 43-49 | 0.8513 | 0.8474 | 0.2459 | 0.2432 |
| w | middle | 50-56 | 0.5145 | 0.5193 | 0.5017 | 0.5047 |
| w | low | 57-63 | 0.5678 | 0.5443 | 0.5678 | 0.5442 |

[Exact frequency groups, summary definitions and source hashes](output/full-sweep-wan-20261101/mrl-heatmaps/manifest.json). Every prompt/step's full 30 x 12 x 64 pair-MRL array is retained under `output/full-sweep-wan-20261101/mrl-heatmaps/data/`.

## 5.4 Heatmap patterns and averaging scopes

Each pair R_f uses all 10,800 conditional video-token Q or K vectors in one 2D frequency pair at one prompt/step/layer/head. R_f = ||mean_tokens(x_f)|| / mean_tokens(||x_f||). A band cell equally averages its listed pair R_f values. Q/K and t/h/w remain separate. Layer/head IDs are zero-based and denoising steps are numbered 1-20.

| Axis | High pair IDs | Middle pair IDs | Low pair IDs |
| --- | --- | --- | --- |
| t | 0-7 | 8-14 | 15-21 |
| h | 22-28 | 29-35 | 36-42 |
| w | 43-49 | 50-56 | 57-63 |

**Frequency selectivity**

High frequencies have the greatest pre-RoPE mean in all 80 prompt/step conditions for Q and K on every axis. Pooled high-band Q/K means are 0.719/0.748 (t), 0.859/0.859 (h), 0.851/0.847 (w).

*Rank bands after averaging 30 layers x 12 heads within each prompt/step. Pooled means average 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 cells per axis/side.*

**Positional dispersion**

After RoPE, high-band Q/K means fall to 0.391/0.406 (t), 0.225/0.219 (h), 0.246/0.243 (w). Low-band mean absolute changes stay below 0.00043. The low-band phase span reaches only 0.021 radians temporally and 0.062 spatially, supporting weak positional dispersion there.

*Compare pre/post at matched cells, then average 28,800 cells per axis/side. Phase spans use the highest low-band frequency times 11 temporal or 29 spatial coordinate increments.*

**Depth**

Low-band Q rises from 0.52-0.53 in layers 1-9 to 0.619-0.624 in layers 20-29 across axes. High bands show a middle-layer dip. Height high-Q means are 0.847, 0.829, 0.887 for layers 1-9, 10-19, 20-29.

*Average all 4 prompts, all 20 steps, all 12 heads and the stated layer block, separately by axis, band and Q/K, before RoPE.*

**Head heterogeneity and Q/K**

Mean within-layer head ranges are 0.19-0.33. Head-index means across layers span 0.03-0.06. One illustrative asymmetry is layer 0/head 3, whose width high-band Q/K means are 0.960/0.307.

*First average 80 prompt/step conditions at each layer/head. Within-layer max-minus-min spans 12 heads, then averages 30 layers. Head-index means also average layers. The example holds layer 0/head 3 and width pairs 43-49 fixed. All values are pre-RoPE.*

**Denoising**

Middle/low K means decrease from step 1 to 20 for every prompt. Pooled middle/low Q dips around steps 8-9 and recovers late. Spatial high bands strengthen, while temporal high-K falls from 0.822 to 0.750.

| Step | High Q | High K | Low Q | Low K |
| --- | --- | --- | --- | --- |
| 1 | 0.843 | 0.841 | 0.611 | 0.736 |
| 10 | 0.854 | 0.854 | 0.558 | 0.524 |
| 20 | 0.894 | 0.893 | 0.613 | 0.512 |

*The table gives pre-RoPE height-axis means over 4 prompts x 30 layers x 12 heads = 1,440 cells at each stated step, using high pairs 22-28 and low pairs 36-42. Per-prompt decline checks average 30 layers x 12 heads at steps 1 and 20.*

**First-layer exception**

Layer 0 low-band K grows from 0.46-0.52 at step 1 to 0.68-0.70 at step 20 across axes. Later layers drive the pooled decline.

*Hold layer 0 and each endpoint step fixed. Average 4 prompts x 12 heads = 48 cells, separately by axis, before RoPE.*

**Gradual map reorganization**

Neighboring steps have median MRL-map correlations of 0.996-0.998. First/last low-Q maps correlate only 0.13-0.23.

*At each step, average 4 prompts, then correlate the 360 layer/head MRL cells. Neighbor values are medians of 19 adjacent-step comparisons, separately for 18 axis/band/QK views. Endpoint values span the three low-Q axis views. All are pre-RoPE.*

[Computed profiles and source definitions](output/full-sweep-wan-20261101/mrl-heatmaps/pattern-analysis.json).

## Interpretation limits

These predictions use centers estimated from the same input being reconstructed. They establish within-input approximation quality, not cross-prompt calibration transfer. A selected strong band does not imply that every band or the whole head is equally concentrated. Axis-line evaluation covers all three directions but not arbitrary mixed-axis query–key offsets. Four prompts and one shared seed do not establish robustness across seeds or a broad prompt population. No KV pruning, generated-video quality comparison, memory saving or inference speedup is claimed.

## Navigation and reproducibility

Open `index.html` to select prompt, denoising step, layer, axis and individual head, or view all heads together and the corresponding concentration distribution. `per-head-results.csv` has one row per prompt/step/layer/axis/head. `coverage.json` records coverage, validation and hashes of every numeric layer record. Full Q/K tensors and full pair-level metrics are stored under `pXX/step-XXX/layer-XXX.{pt,json}`. `rope.pt` stores the exact model rotary angles and grid. Per-prompt metadata includes the literal prompt, seed, model snapshot, environment, GPU and code hashes. Step metadata records both the sequential denoising index and actual scheduler timestep.

Figure subjects and pair-ranking/MRL definitions follow [TriAttention §3 and Appendix B](https://arxiv.org/abs/2604.04921). The original LLM experiment uses positive, logarithmically spaced 1D text distances and calibration centers. Our video experiment uses three coordinate axes, same-input centers, both signed directions and self, evaluating every same-axis-line query/key. Its correlations are not a matched LLM benchmark.

The old Wan `pipeline_probe.py` trajectory was incorrectly labelled conditional: its hooks retained the final, unconditional CFG forward. This verified conditional sweep supersedes that interpretation. The earlier synthetic-input plots are separate measurements and are not relabelled by this correction.
