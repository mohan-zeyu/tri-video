# TriAttention video research handoff

Updated 9 November 2026, Asia/Singapore. Sections 11-14 cover the completed generation benchmarks and latest interpretation. Sections 4-7 define the geometry and correlation experiments.

## 1. Current position

The project investigates pre-RoPE Q/K concentration in video diffusion, its frequency/layer/head/denoising-step structure, and its usefulness for attention approximation and pruning.

- Full conditional Wan sweep, paper-style A/B/C/D figures, all-condition heatmaps and correlation analyses are complete.
- A 100-prompt paired Q/K mean-replacement study is complete, with 300 videos and seven VBench dimensions. It strongly degrades quality. See section 11.
- A separate 100-prompt paired TriAttention-style original-K/V pruning study is complete at 50% and 75% per-bank retention. See section 13.
- Performance profiling was started, then stopped at the user's request. No speed optimization was implemented or validated. See section 14.

## 2. Measurement requirements

For each result, identify tokens, 2D pairs versus full Q/K vectors, prompts, denoising steps, layers, heads, CFG branches and pre/post-RoPE stage. State the averaging order and correlation sample population. MRL pattern similarity, Q/K center directions, logit reconstruction and generation quality are distinct measurements. See `AGENTS.md`.

## 3. Project layout and reproduction

All paths below are relative to the repository root unless marked as external runtime locations.

| Location | Purpose |
|---|---|
| `REPORT.md` | Full conditional Wan geometry report |
| `configs/full_sweep_wan.json` | Four prompts and full geometry-sweep settings |
| `configs/center_benchmark_wan.json` | 100-prompt direct mean-replacement study |
| `configs/triattention_pruning_wan_queryblocks.json` | Authoritative 100-prompt pruning study |
| `out/full-sweep-wan-20261101/` | Geometry captures, metrics and atlas on the data host |
| `out/center-benchmark-wan-20261109/` | Replacement outputs on the data host |
| `out/triattention-pruning-wan-queryblocks-20261109/` | Authoritative pruning outputs on the data host |
| `output/` | Local review mirrors and generated PDFs, excluded from this commit |
| `scripts/` | Generation, measurement, validation, report and site tools |

Generated result archives are external to the source checkout. Configure model caches through `HF_HOME`, evaluator caches through `VBENCH_CACHE_DIR`, and cached Wan weights through `WAN_SNAPSHOT`. Server addresses and SSH connection details are deployment configuration.

To recover an SSH-forwarded atlas, supply the deployment's host and checkout explicitly.

```sh
python3 scripts/open_full_sweep_atlas.py --host "$TRI_REMOTE_HOST" --repo "$TRI_REMOTE_REPO"
```

To start a protected atlas on its data host, supply the desired interface explicitly.

```sh
python3 scripts/start_private_atlas.py --bind "$TRI_ATLAS_BIND"
```

Access credentials, shared URLs, runtime viewer metadata and machine-specific connection settings are excluded from version control.

## 4. The experiment and exact measurements

### Complete sweep

- Model is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- Four real prompts, 20 denoising steps each, 30 layers, 12 heads, 128 dimensions per head.
- Video configuration is 45 output frames, 480 x 480, CFG 5.0, seed 0 shared across prompts.
- Latent token grid is 12 temporal x 30 height x 30 width = **10,800 video tokens**.
- Prompt IDs are p00, Tokyo woman; p01, waterfall; p02, dancing panda; p03, red toy car. Literal texts are in `config.json`.
- Sequential denoising indices are 1-20. Layer/head/pair indices are zero-based. Latent temporal coordinates are separate from denoising-step indices.
- Captures are normalized Q/K immediately before RoPE on the conditional branch. Every actual text embedding input was checked against its supplied conditional or unconditional embedding. Wan runs the two CFG branches sequentially.

The capture run used GPUs 2 and 5 and parallel CPU rendering. Later work encountered a shared-filesystem bottleneck, and late trajectories used RAM staging before verified persistence. Capture/metrics code hashes are retained in `coverage.json` and checked by the summarizer. Report-only work belongs in the reporting/analysis scripts.

### Pair MRL and dominant pairs

For one prompt, denoising step, layer, head and 2D pair `f`, compute separately for Q and K:

```text
R_f = ||mean_over_all_10800_tokens(x_f)|| / mean_over_all_10800_tokens(||x_f||)
```

Source Q/K are native bf16 and statistics use fp32. Post-RoPE MRL uses the captured vectors rotated with the saved angles and rounded to native bf16.

Original scatter plots choose the axis-dominant pair by:

```text
C_f = mean_tokens(||q_f||) * mean_tokens(||k_f||)
dominant_pair = argmax C_f within the requested axis
```

Selection is repeated per prompt/step/layer/head/axis. The 2D scatter contains every token's Q and K in that pair. The selected pair and plot scale match before/after RoPE.

### Frequency thirds used by heatmaps

The heatmap groups use actual angular frequency. Sort descending separately within each axis, then split into nearly equal-count thirds.

| Axis | All pair IDs | High | Middle | Low |
|---|---|---|---|---|
| t | 0-21 | 0-7 | 8-14 | 15-21 |
| h | 22-42 | 22-28 | 29-35 | 36-42 |
| w | 43-63 | 43-49 | 50-56 | 57-63 |

Within an axis, larger pair index means lower angular frequency. Temporal groups contain 8/7/7 pairs. Spatial groups each contain seven pairs.

Actual frequencies were recovered from `rope.pt` using `atan2(sin,cos)` at a unit coordinate increment. Flattened unit positions are 900, 30 and 1 for t/h/w. Frequencies decrease strictly from 1 rad/token to approximately 0.000152-0.000155 rad/token.

A band heatmap cell is the arithmetic mean of its individual pair MRL values. Each cell retains one prompt, step, layer, head, side and stage. The individual-pair view uses one saved `R_f` directly. All maps share a 0-1 color scale.

### Full-head attention-logit reconstruction

This metric uses all 128 dimensions, summing all 64 pairs. Q/K centers are fp32 means of the full normalized pre-RoPE vectors over all 10,800 tokens, within the same prompt/step/layer/head.

- Measured logits use native bf16 rotated token Q/K with fp32 dots, divided by `sqrt(128)`.
- Predicted logits rotate the fp32 centers at the actual positions.
- Every token is a query. Keys span its complete axis line, fixing the other two coordinates. Both signed directions and self are included.
- Per head/condition, this gives 129,600 temporal logits or 324,000 logits for each spatial axis.
- `per_query_r_mean` first computes Pearson across a query's same-axis keys, then averages the valid query correlations.
- `curve_r` correlates the two offset-mean curves and is retained separately.
- Curve points average all query/key logits at each signed offset. Shading is population standard deviation of measured logits.

## 5. Keep the correlation analyses distinct

### Paper reconstruction correlation

Primary reference is https://arxiv.org/html/2604.04921v1, section 3.3 and Appendix B.6.

TriAttention computes Pearson between actual and center-predicted full-head logits for each query over logarithmically spaced distances `1,2,4,8,...`, then averages over queries within a head. It derives centers from calibration data.

Our video reconstruction follows the same mathematical principle with same-input centers and exhaustive signed t/h/w axis-line distances. Our MRL-pattern correlations below are additional analyses introduced for the video experiment. They answer concentration-pattern stability questions. Explain their data and aggregation explicitly when relating them to the paper.

### Analysis A, denoising changes across the layer/head map

For one axis/band/QK side and one step:

1. Compute each pair MRL from all tokens.
2. Average its band's pair MRLs within each prompt/layer/head.
3. Average four prompts at each layer/head.
4. Flatten the resulting 30 x 12 map into 360 matching layer/head cells.
5. Pearson compares those 360 MRL values between two steps.

Neighbor summary uses the median of 19 adjacent-step correlations. Its 0.996-0.998 range spans 18 pre-RoPE axis/band/QK views, including high/middle/low bands.

Low-Q endpoint results are:

| Axis | Mean at step 1 | Mean at step 10 | Mean at step 20 | Map r, step 1 vs 20 |
|---|---|---|---|---|
| t | 0.6117 | 0.5530 | 0.6066 | 0.15757 |
| h | 0.6111 | 0.5575 | 0.6131 | 0.13470 |
| w | 0.6115 | 0.5523 | 0.6051 | 0.23124 |

The scalar means average four prompts x 30 layers x 12 heads at each stated step. A recovered scalar mean can accompany a changed pattern across layer/head cells.

### Analysis B, denoising changes inside one fixed layer/head

Hold one prompt, numerical layer/head, axis, band and Q/K side fixed. At each step, form a vector of the 7-8 **individual pair MRL values**. Pearson is computed across frequency-pair entries.

```text
Input per axis/band/side       [4 prompts, 30 layers, 12 heads, 20 steps, F pairs]
Neighbor correlations         [4, 30, 12, 19]
Median over 19 transitions     [4, 30, 12]
Median over 1440 trajectories  one scalar
```

Prompts and pairs remain separate before correlation. Each input MRL already aggregates all 10,800 token vectors within one 2D pair. These vectors are frequency profiles of MRL scalars.

The following pre-RoPE results are medians over 1,440 trajectories per view:

| Axis | Band | Q r, step 1 vs 20 | K r, step 1 vs 20 | Q neighboring summary | K neighboring summary |
|---|---|---|---|---|---|
| t | High | 0.598 | 0.551 | 0.998163 | 0.998390 |
| t | Low | 0.240 | 0.187 | 0.995860 | 0.994669 |
| h | High | 0.398 | 0.346 | 0.998220 | 0.998235 |
| h | Low | 0.271 | 0.161 | 0.996046 | 0.994368 |
| w | High | 0.326 | 0.364 | 0.997703 | 0.997918 |
| w | Low | 0.277 | 0.175 | 0.996063 | 0.994496 |

The 0.994-0.998 range spans these 12 summaries, with only high/low bands. All 27,360 adjacent-step correlations per view were defined in the checked data.

**First-transition correction.** The earlier headline that neighboring steps remain highly similar was too broad. The medians describe typical transitions and hide a larger initial change. For height low-Q, the median across 1,440 trajectories is 0.6963 for steps 1->2, 0.9981 for 10->11 and 0.9801 for 19->20. Across axes, low-Q first-transition medians are approximately 0.694-0.726, and low-K medians are 0.734-0.742.

Late-stage frequency profiles retain more similarity. Step 10->20 median correlations span approximately 0.805-0.892 for high bands and 0.619-0.754 for low bands across the Q/K and axis views.

A separate ordering test averages pairs within each high/low band at each step, then checks high mean > low mean throughout all 20 steps. Percentages of trajectories satisfying this condition are:

| Axis | Q | K |
|---|---|---|
| t | 61.7% | 67.2% |
| h | 88.8% | 73.1% |
| w | 81.4% | 68.5% |

Fixed illustration uses p03, layer 15, head 0, pre-RoPE. Individual-pair endpoints include height K pair 39, 0.5982->0.0258, height K pair 41, 0.1731->0.3262, width Q pair 60, 0.5477->0.9096, and width Q pair 58, 0.8126->0.4986. These retain one prompt/layer/head/pair and compare steps 1 and 20.

### Analysis C, changes across layers

Hold one prompt, one denoising step, one numerical head index, axis, band and Q/K side fixed. Compare individual-pair MRL profiles between layers using Pearson across the 7-8 pair entries.

```text
Input per view                 [4 prompts, 20 steps, 12 heads, 30 layers, F pairs]
Adjacent-layer correlations    [4, 20, 12, 29]
Median over 29 transitions     [4, 20, 12]
Median over 960 trajectories   one scalar
```

The same head index is matched between different learned attention blocks. This is a numerical head-index comparison.

| Axis | High-Q neighbor r | High-K neighbor r | Low-Q neighbor r | Low-K neighbor r |
|---|---|---|---|---|
| t | 0.553 | 0.536 | 0.015 | 0.008 |
| h | 0.230 | 0.231 | -0.035 | 0.007 |
| w | 0.035 | 0.076 | 0.019 | -0.023 |

These are two-stage medians, over 29 adjacent-layer correlations and then over 960 prompt/step/head trajectories. Low-frequency profiles have almost zero neighboring-layer correlation. High-frequency similarity varies strongly by axis. Neighboring-layer similarity is much weaker than typical neighboring-denoising-step similarity.

Endpoint and layer-1-to-29 distributions are retained in the Analysis C JSON. Layer distance alone is insufficient to describe the observed profile similarity.

## 6. Other established findings

### Frequency and RoPE means

Each entry below pools 28,800 band-mean cells, separately for the stated axis/stage/side. The cell averages only its frequency pairs, with each pair using all tokens.

| Axis | Pre high Q/K | Pre middle Q/K | Pre low Q/K | Post high Q/K |
|---|---|---|---|---|
| t | 0.7187/0.7476 | 0.5448/0.5361 | 0.5684/0.5436 | 0.3910/0.4061 |
| h | 0.8587/0.8587 | 0.5417/0.5342 | 0.5725/0.5478 | 0.2251/0.2194 |
| w | 0.8513/0.8474 | 0.5145/0.5193 | 0.5678/0.5443 | 0.2459/0.2432 |

High bands rank first after layer/head averaging in all 80 conditions for both sides and all axes. Low-band mean absolute pre/post changes are below 0.00043. Their maximum phase span across the grid is about 0.0206 radians for t or 0.0625 for h/w. High frequencies experience much larger position-dependent rotation.

### Depth and Q/K asymmetry

These pre-RoPE block means average four prompts, 20 steps, all 12 heads and the stated layers, with pair averaging within each band.

| Axis | Low-Q L1-9 | Low-Q L10-19 | Low-Q L20-29 |
|---|---|---|---|
| t | 0.5264 | 0.5589 | 0.6216 |
| h | 0.5278 | 0.5685 | 0.6243 |
| w | 0.5227 | 0.5637 | 0.6189 |

Height high-Q block means are 0.8467, 0.8290 and 0.8867 for the same blocks. Layer 0 behaves differently over denoising. Its low-K means increase from 0.46-0.52 at step 1 to 0.68-0.70 at step 20 across axes, averaging four prompts and 12 heads at that one layer/step.

After averaging 80 conditions at each layer/head, mean within-layer head ranges span roughly 0.19-0.33 across the pre-RoPE axis/band/QK views. Head-index means additionally averaged across layers span roughly 0.03-0.06. At layer 0/head 3, width high-Q/K means over 80 conditions are 0.9597/0.3072.

Middle/low K means decline from step 1 to 20 for each of the four prompts on all axes after averaging layers/heads. Q middle/low pooled curves dip around steps 8-9 and recover. Spatial high Q/K groups strengthen late, while temporal high-K means fall from roughly 0.822 to 0.750.

### Original selected-pair and reconstruction results

Across all 28,800 cells per axis, mean pre-RoPE dominant-pair Q/K MRL is t 0.8950/0.8495, h 0.9271/0.9070, w 0.9114/0.8915. Mean full-head per-query reconstruction r is t 0.5689, h 0.7284, w 0.6117. Frequency-third means and dominant-pair means have different pair populations.

Best full-head reconstruction examples, selected after inspecting all 28,800 cells per axis, requiring all 10,800 valid queries:

| Axis | Prompt | Step | Layer | Head | Displayed dominant pair | Mean query r |
|---|---|---|---|---|---|---|
| t | p01 | 17 | 0 | 5 | 0 | 0.9986695 |
| h | p01 | 12 | 29 | 9 | 24 | 0.9993596 |
| w | p03 | 2 | 29 | 9 | 47 | 0.9999729 |

Best concentrated selected pairs maximize `min(R_Q,R_K)` over the same axis-specific selected-pair population. Winners are t p01/s20/L0/H5/pair0, h p01/s4/L29/H9/pair24, and w p03/s3/L28/H11/pair48. These rankings are separate from full-head reconstruction performance. Exact values, rules and source hashes are in `report-figures/selection.json`.

## 7. Research and implementation file map

### Core experiment and report scripts

| Files under `scripts/` | Role |
|---|---|
| `full_sweep.py`, `full_sweep_metrics.py` | Conditional capture and exact token/pair/head measurements |
| `launch_full_sweep.py`, `render_full_sweep.py` | Capture/render orchestration |
| `summarize_full_sweep.py` | Coverage, CSV, summaries, Markdown and HTML index |
| `make_paper_figures.py` | Nine A/B/C/D composites and best-case selection |
| `export_mrl_heatmaps.py` | Standard-library export of all 2400 hash-verified layer records |
| `build_mrl_heatmaps.py` | Saved-angle grouping, data packaging, six example figures and 160-page appendix |
| `mrl-heatmaps.js`, `full_sweep_atlas.html` | Interactive heatmaps and main atlas template |
| `mrl_pattern_summary.py` | Shared scope-labelled Analysis A prose for HTML, Markdown and PDF |
| `build_full_sweep_pdf.py` | Main nine-page PDF |
| `finalize_full_sweep_delivery.py` | Source/artifact verification and delivery manifest, preserving private-sharing metadata |
| `serve_full_sweep.py`, `start_private_atlas.py`, `open_full_sweep_atlas.py` | Authenticated serving and recovery |
| `test_atlas_access.py`, `test_full_sweep.py`, `verify_full_sweep_samples.py` | Access, math and real-tensor checks |

### Latest local analysis files

These are real files under the local repository. The `tmp/` directory persists through conversation compaction but is gitignored.

| Path | Contents |
|---|---|
| `tmp/analyze-mrl-patterns.py` | Computes Analysis A profiles, prompt/layer/head/step summaries and targeted checks |
| `tmp/mrl-pattern-analysis.json` | Analysis A numerical output before the published source-manifest hash was attached |
| `output/full-sweep-wan-20261101/mrl-heatmaps/pattern-analysis.json` | Published Analysis A source, with `source_heatmap_manifest_sha256` |
| `tmp/analyze-fixed-head-frequency.py` | Computes Analysis B for pre/post, ordering tests and fixed-head illustration |
| `tmp/fixed-head-frequency-analysis.json` | Full Analysis B outputs, example pair values, ordering proportions and scopes |
| `tmp/across-layer-frequency-analysis.json` | Analysis C outputs, scopes, endpoint distributions and per-transition medians |
| `output/full-sweep-wan-20261101/mrl-heatmaps/figures/fixed-head-p03-L15-H00-pre.png` | Local Analysis B illustration, each pair x step displayed individually |
| `tmp/check-mrl-ui.cjs` | DOM integration check for the deployed HTML and interactive heatmaps |
| `tmp/heatmap-check/node_modules/jsdom` | Temporary DOM test dependency |

The Analysis B illustration and Analysis B/C JSON files currently have local/chat publication state. The last published delivery manifest describes the earlier HTML/PDF bundle and Analysis A. A future request to publish B/C should update the site/report, source links and delivery gate accordingly.

To reproduce Analysis C, use the loading and `correlation` helpers in `tmp/analyze-fixed-head-frequency.py`. `raw` has shape `[stage,side,prompt,step,layer,head,pair]`. For pre-RoPE:

```python
profiles = np.moveaxis(np.take(raw[0, side_index], pair_ids, axis=-1), 2, -2)
# Shape [4 prompts,20 steps,12 heads,30 layers,F pairs]
r = correlation(profiles[..., :-1, :], profiles[..., 1:, :])
per_trajectory = np.nanmedian(r, axis=-1)
summary = np.nanmedian(per_trajectory)
```

For Analysis B, move axis 1 to -2 instead, giving `[4,30,12,20,F]` before adjacent-step comparisons. Correlation centers both input profiles over their final pair dimension, then divides the centered dot product by the product of centered norms. The helper treats negligible/zero-variance cases as undefined. All Analysis B pre-RoPE adjacent correlations were defined when checked.

## 8. Runtime, regeneration and verification

### Environments

- Local `.report-venv/bin/python` is Python 3.14. It contains numpy, matplotlib, reportlab, Pillow, pypdf and torch 2.15.0.
- Remote system `python3` is Python 3.12.3. It has the standard library needed for serving and JSON export.
- The original `/dev/shm/trivenv/bin/python` environment disappeared before 8 November. Remote torch/numpy/matplotlib/reportlab were absent when checked then.
- Routine analysis and report work uses the saved data and local environment. Fresh model inference needs the appropriate remote model environment rebuilt first.

For local matplotlib, use a writable cache directory:

```sh
MPLCONFIGDIR=tmp/matplotlib-mrl \
  .report-venv/bin/python tmp/analyze-fixed-head-frequency.py
```

### Regeneration

The saved per-condition heatmap arrays provide all pairs and layers for later statistics. Reading full raw Q/K is necessary only for analyses beyond those retained statistics.

```sh
# Main PDF, local
.report-venv/bin/python scripts/build_full_sweep_pdf.py

# Full heatmap/appendix regeneration, local
.report-venv/bin/python scripts/build_mrl_heatmaps.py

# Analysis B, local
MPLCONFIGDIR=tmp/matplotlib-mrl \
  .report-venv/bin/python tmp/analyze-fixed-head-frequency.py
```

The export generator reads remote `pXX/step-NNN/layer-NNN.json`, checks source hashes against coverage, then creates inputs under `mrl-heatmaps/inputs/` and a transfer bundle under remote `tmp/`. The local heatmap builder expects those inputs and `rope.pt` in the review bundle.

`mrl_pattern_summary.load_summary(root)` loads the published Analysis A source, validates its source-manifest hash, matches its group definitions and verifies all 36 pre/post axis/band/QK means against the heatmap manifest. `summarize_full_sweep.write_index(root,config)` inserts the shared analysis into the HTML and copies the heatmap JS asset.

### Verification already completed

- Coverage is 2400 layer captures, 86400 head figures, 7200 all-head sheets and 7200 concentration distributions.
- Full raw bf16 Q/K total 159,257,560,800 bytes.
- All 144 native attention-output recomputation checks are bit-exact.
- Independent CPU real-tensor spot checks and explicit axis-line math checks passed.
- Ten RAM-staged late-step folders were persisted and SHA256-verified, totaling 21,720,321,189 bytes. The stage tree was removed after persistence.
- All 2400 source layer JSON hashes were verified for heatmap export.
- All 80 exported condition arrays were hash-verified. All 2880 band heatmaps passed shape/range and independently indexed sample checks.
- All 160 appendix pages cover the exact 80-condition x two-stage product, with 18 maps per page.
- Main PDF pages were rendered with `pdftoppm` and visually inspected. Analysis A fits on one page in section 5.4.
- Live HTML/DOM checks verified selectors, exact pair/color indexing, hover, cell selection, individual-pair views, stage changes, prompt/step changes and all seven published analysis scope notes.
- Private HTTP checks verified authorized HTML/images/JSON/PDF/tensor routes and anonymous 403 responses. Published PDFs matched local SHA256 hashes.
- Access-control integration tests passed for anonymous requests, valid URL/cookie sessions, invalid/expired sessions, secret/log handling and outside-root paths.

CUA/browser automation repeatedly timed out, including inventory and DOM access attempts. The latest report work used rendered PDF review and live HTTP plus jsdom tests. A live browser screenshot of the newest analysis section has no successful verification record. Check current tool availability before attempting fresh browser QA.

Current important hashes from `delivery.json`:

```text
report.pdf
9c8a146d49720897e41ba30ac342cdca736f087d25af4fc19b0bd0a0755ce886

coverage.json
a02b87628853fb94156aabd0deb27b3323982101be1c929b542861ee379089a8

per-head-results.csv
6b9d07f2708a1018ca250e1fe364ab25bce1d1cd0586b00f2849df121d8b61b7
```

Main PDF currently has nine pages and 2,978,969 bytes. The appendix has 160 pages and 8,146,170 bytes. Re-read the manifest after future edits.

To update a published report, stage the reviewed PDF and data, atomically copy files into the result root and remote `output/pdf/`, regenerate HTML after its dependencies exist, and run:

```sh
python3 scripts/finalize_full_sweep_delivery.py --pdf-sha256 THE_NEW_PDF_SHA256
```

Then retrieve the new `index.html` and `delivery.json` into the local bundle. The finalizer records the new hashes and preserves `private_sharing` from the prior manifest.

For future PDF edits, read the available PDF skill and follow its authoring marker and render/verify workflow. Scientific plots use numpy/matplotlib and saved numerical data.

## 9. Earlier work and corrections to preserve

- The first limited reports covered Wan/LTX/Cog synthetic probes and limited real trajectories. They are historical context. Current full-Wan results are in `REPORT.md` and the nine-page report.
- The old Wan `pipeline_probe.py` hook retained the final unconditional CFG pass while its report called it conditional. The verified full conditional sweep supersedes that interpretation. Earlier synthetic-input measurements have their own scope.
- Legacy report snapshots are under `out/report/`, including `limited-report-superseded.md` and `previous-report.md`. `scripts/build_report.py` was changed to write `LIMITED_REPORT.md` to preserve the current report.
- The paper's concentration illustration is Figure 2. Its panels are A pre-RoPE pairs, B post-RoPE pairs, C across-head concentration, D logit prediction. The user's third numbered experiment instruction corresponds to panel D.
- Original full-sweep head PNGs contain A/B/D. All 7200 C distributions were computed separately. The corrected report and nine-figure gallery combine all A/B/C/D panels. The atlas presents C alongside the selected original figure.
- C for this video study covers the 12 heads at one prompt/step/layer/axis. Each head chooses its own dominant pair. The paper's Figure 2(C) covers its LLM attention-head population.
- The nine report examples comprise three fixed p03/s20/L15/H0 views, three best selected-pair concentration examples and three best full-head reconstructions. Best cases are explicitly selected maxima.
- Paper reconstruction correlations, layer/head-map correlations, fixed-head frequency-profile correlations and cross-layer profile correlations all use Pearson with different samples and aggregation orders. Name the input population every time.
- MRL measures concentration magnitude. Claims about Q/K center directions, center transfer across prompts or performance require the corresponding measurements.
- Four prompts share seed 0. The full experiment retained latents. Video quality, pruning/compression throughput and KV-memory savings have separate experimental requirements.

## 10. Next-agent priorities

1. Start with sections 11-14 for the latest benchmark results and implementation limitations.
2. Reuse saved result arrays for follow-up statistics. Match each number to its token/pair/prompt/layer/head/step scope.
3. Preserve the first-transition correction in neighboring-step correlation summaries.
4. Performance work is stopped at the user's request. Resume profiling or optimization after a subsequent instruction to do so.
5. Committed source is sanitized for portable deployment. Historical generation hashes identify the versions actually executed before path/configuration cleanup.

## 11. Completed Q/K center-replacement benchmark, 9 November

### User decisions and scope

The direct-replacement study uses current-token centers recomputed at every step and a 100-prompt paired subset. Generation, evaluation, audit and report/site verification are complete.

Use the same Wan cached revision `0fad780a534b6463e45facd96134c9f345acfa5b`, 45 frames at 480 x 480, 16 fps, 20 denoising steps, CFG 5, empty negative prompt. There are still 10,800 video tokens on a 12 x 30 x 30 grid. Transformer/text encoder are bf16. VAE is float32 with tiling.

Three arms are fresh baseline, all-pair replacement and high-pair replacement. At `norm_q`/`norm_k` outputs immediately before native RoPE, compute fp32 coordinate-wise means over all 10,800 current tokens, independently per prompt, step, layer, head and CFG branch. Cast centers to bf16. **Replace both Q and K jointly, with separate Q/K means.** All-pair selects all 64 pairs / 128 coordinates. High-pair selects t 0-7, h 22-28 and w 43-49, totaling 22 pairs / 44 coordinates, matching the earlier upper frequency thirds. Other pairs retain their current trajectory's token representations.

All 30 layers, 12 heads, 20 steps and both branches receive the intervention. Native full attention evaluates every query against every key across the complete 3D grid, summing all 64 pairs. These are full generation interventions. Means adapt to each modified trajectory, so the results concern updated centers. Calibration transfer, stale centers and Q-only/K-only interventions have separate experimental requirements.

`configs/center_benchmark_wan.json` freezes 100 distinct prompts sampled before generation from official VBench category lists with sampling seed 20261109. There are 13 each for animal, architecture, food and human, and 12 each for lifestyle, plant, scenery and vehicles. Each prompt has one independent deterministic seed shared across all three arms. Processing interleaves categories.

### VBench results and aggregation

Official source is pinned to `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. The wrapper calls its unmodified compute functions with exact custom prompt/video metadata. Six quality dimensions and the optional official ViCLIP text-video dimension all completed.

| Dimension | Baseline | All pairs | High pairs |
|---|---:|---:|---:|
| Aesthetic quality | 0.58859467 | 0.31698027 | 0.26696126 |
| Imaging quality | 0.66067940 | 0.40813432 | 0.44479362 |
| Text-video consistency | 0.23255504 | 0.03017356 | 0.04265920 |
| Subject consistency | 0.96399870 | 0.90661325 | 0.95657187 |
| Background consistency | 0.97646601 | 0.97089921 | 0.97964999 |
| Motion smoothness | 0.98446269 | 0.97751718 | 0.98433361 |
| Dynamic degree | 0.33000000 | 0.83000000 | 0.04000000 |

Each arm mean averages 100 per-video scores. MUSIQ averages 45 frames and divides by 100. Aesthetic quality averages 45 frame predictor scores and divides by 10. Subject/background use first/previous-frame cosine scores across 44 transitions. AMT reconstructs 22 odd-index frames from even-index neighbors. RAFT subsamples to 23 frames at 8 fps, yielding 22 flow transitions. ViCLIP uses eight segment-midpoint frames and the actual prompt. Full evaluator scope descriptions are in the report and builder's `SCOPES` dictionary.

Confidence intervals resample 100 paired prompt/seed units, retaining all three arms together, 10,000 times with NumPy RNG seed 20261109. Report percentile 95% intervals on mean paired differences. This is a one-video-per-prompt category-sampled custom-input study, with raw dimension scores and its own sampling scope relative to the full official protocol.

Text alignment falls on all 100 prompts for both arms. Aesthetics fall for 99/100 all-pair prompts and 100/100 high-pair prompts. MUSIQ falls for 95/100 prompts in each arm. High-only versus all-pair changes aesthetics by -0.0500190, CI [-0.0588969,-0.0406960], and imaging by +0.0366593, CI [+0.0108599,+0.0627853]. Its tradeoff differs by dimension. Consistency can stay high for degraded similar frames. Dynamic degree records flow-derived motion incidence. Joint appearance/text/video inspection is essential for interpreting it.

### Concentration/error audit

Eight prompts fixed before result inspection are `v000,v013,v026,v039,v052,v064,v076,v088`. NPZ files retain pair diagnostics at steps 1,10,20, all layers/heads and both branches for every arm. Published summaries use the **conditional baseline** branch.

Each pair MRL uses all 10,800 token vectors. Pair replacement error is `(E||x||^2 - ||E[x]||^2) / E||x||^2`, for the fp32 center. Average individual pair fractions equally across that band's pairs, 12 heads, 30 layers and eight prompts. Figures keep steps separate. The PDF table additionally averages the three audited steps equally.

| High group | Mean Q MRL | Mean K MRL | Mean Q error fraction | Mean K error fraction |
|---|---:|---:|---:|---:|
| t, 8 pairs | 0.7190 | 0.7570 | 0.4904 | 0.4362 |
| h, 7 pairs | 0.8615 | 0.8595 | 0.3063 | 0.2985 |
| w, 7 pairs | 0.8692 | 0.8636 | 0.2948 | 0.2920 |

High MRL still accompanies substantial representation change under mean replacement. All-layer/all-step joint Q/K replacement strongly degrades quality even with updated centers. Assigning the effect to individual layers, steps or Q versus K requires corresponding ablations.

### Artifacts, runtime and checks

- Remote source root is `out/center-benchmark-wan-20261109`. Local bundle is `output/center-benchmark-wan-20261109`, including all 300 videos. Raw final latents remain remote.
- Main PDF is `output/pdf/wan-center-replacement-report.pdf`, 10 pages, 13,330,552 bytes, SHA256 `b4eb45862bdd549f0add92b6a0af531cc76f4446033bb77558c56a312252747c`.
- The new bundle contains `index.html`, `REPORT.md`, `summary.json`, `per-prompt-results.csv` (700 rows), `config.json`, `provenance.json`, `requirements-runtime.txt`, `native-pretrained-validation.json`, `site-verification.json` and `delivery.json`.
- The private route is `/center-benchmark/index.html` under the existing atlas and secret. Its source template/main page now link **Q/K replacement benchmark**. Report/videos/records/evaluation files are hard-linked within the served archive's `center-benchmark` directory. Raw final latents are covered by the same gate.
- New scripts are `center_replacement.py`, `run_center_benchmark.py`, `check_center_replacement.py`, `prepare_center_vbench.py`, `evaluate_center_benchmark.py`, `summarize_center_benchmark.py`, `build_center_benchmark_report.py`. Read `CENTER_BENCHMARK_PLAN.md` for the design.
- Generation code hashes are frozen and verified across all 300 outputs. Official VBench is local `tmp/center-benchmark/VBench` and remote `/dev/shm/tri-vbench-source`. Checkpoints persist in the new output's `evaluator-cache`.
- **Use `/dev/shm/tri-center-runtime/bin/python` for current inference.** It has torch 2.11.0+cu128, torchvision 0.26.0+cu128, Diffusers 0.42.1, Transformers 5.22.2, pyiqa 0.1.16, decord 0.6.0, timm 1.0.12, openai-clip 1.0.1, NumPy 2.5.4. Driver is 580.173.02.
- The earlier on-disk runtime had incompatible/stalled imports. The working runtime was rebuilt using pinned torch/torchvision wheels and their declared dependencies. Runtime versions are recorded in output metadata.
- `WAN_SNAPSHOT=/dev/shm/tri-wan-snapshot` points to the full dereferenced cached Wan snapshot, about 27 GB. Runtime/model scratch trees are ephemeral. Durable cached weights, runtime requirements and evaluator checkpoints support rebuilding.
- Generation used GPUs 2-6 after the GPU-0 pilot. GPU 0 evaluated the dimensions concurrently with generation. GPU 1 was left available when another task used it. All experiment processes finished and all seven GPUs were idle at the final check.
- Verified all 300 video hashes/format checks, exactly 2,400 Q/K hooks per video and 720,000 total, both CFG embedding identities and all finite step latents. Native module output checks were bit-exact in all modes for random/pretrained layer-0 weights, 36 synthetic tokens, two batches. Independent fp32 3D identity error is 4.47e-7.
- Independently checked all 700 CSV rows, 21 arm means and 14 paired changes. Rendered/visually inspected all 10 PDF pages. DOM checks passed 100 prompt selectors, 300 video paths, 2,100 score cells and eight category filters.
- Playwright/isolated Chrome verified video metadata, synchronized playback, zero script errors and zero desktop/mobile page overflow. Screenshots and UI check scripts are under `tmp/center-benchmark`. Direct Chrome CLI screenshots were blank, and Playwright provided successful rendering verification.
- Private HTTP checks passed authorized access and anonymous 403 responses for HTML/PDF/JSON/video/raw latents. The live PDF hash matched local. Main atlas navigation and its delivery manifest were refreshed.

## 12. Independent RoPE audit and official-code comparison

The implementation was independently audited against actual activations and the official TriAttention GitHub implementation, including upstream-cache checks. Its finding is normal native 3D RoPE and correct execution of the direct-replacement trial, with a major methodological difference from official center-guided KV pruning.

- `scripts/audit_center_trajectory.py` replays real prompt v000/seed2026110900 through all 20 steps, all 30 layers/12 heads and both CFG branches. Fresh zero-hook baseline is bit-exact to the saved baseline. Independently written flat-coordinate all/high processors are bit-exact to the saved all/high final latents. All maximum and relative L2 errors are zero. Each modified arm runs 1,200 attention and 1,200 RoPE calls.
- Subagent real first-step checks cover that prompt, both branches, layers0/15/29 and every one of 10,800 tokens/12 heads/64 pairs. All36 Q/K tensor checks pass exactly. Native angle tables match independent NumPy float64 analytic t/h/w phases after fp32 conversion. Selected means match float64 token-sum references after bf16 cast. Unselected coordinates are exact. The earlier pretrained36-token synthetic check had narrower coverage, now supplemented by these actual-path checks.
- Official repository is cloned locally to `tmp/center-benchmark/TriAttention`, pinned to `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`. The mean formula agrees in the native video collector, while its use differs. It calibrates Q means/mean pair norms, ranks real individual K with a trig term plus `(E|q_f|-|E q_f|)|k_f|`, then gathers original K/V values. Actual queries and retained token-specific K/V perform attention. Our completed trials directly replace selected live Q/K coordinates at every specified head/layer/step/branch.
- LongLive's calibrated statistics have shape30L×12H×64pairs and pool token/batch positions across hooked invocations, with one denominator per layer and no denoising-step dimension. Its ranking varies temporal offsets and sets spatial offsets to zero while summing all64pairs. Its actual model attention retains its native positional representations.
- Official LongLive video configs use dedicated causal generator/LoRA checkpoints, three latent frames per chunk and four configured denoising steps. This differs from our base full-video Diffusers Wan20-step pipeline.
- Upstream single-prompt config has budget6frames, sink3 and recent3, leaving zero dynamically ranked tokens. Interactive config has sink1/recent3, leaving two ranked frames. An isolated official CPU cache test confirmed scorer calls0 versus1 and exact original retained K/V in both. Test scope is one synthetic layer/head with18,720slots×128D and a mocked deterministic ranking function. This checks budget/gather behavior, with no video generation.

The full evidence/source comparison is `output/center-benchmark-wan-20261109/implementation-audit/README.md`, with JSON records and audit scripts alongside it. `audit-native-actual-first-step.json` also exists at the new experiment root. The HTML report links the audit note and raw checks. The new delivery manifest includes these artifacts. Frozen generation source and benchmark numerical results retain their original values. Direct replay coverage is one prompt/seed. The interpretation is direct Q/K replacement ablations, while official good results concern center-guided selection and retained actual representations.

## 13. Completed TriAttention-style Wan pruning rerun, 9 November

### Request, controls and adaptation

The user asked to rerun with official TriAttention's center-guided selection after the replacement audit. The scope is base Wan, paired retention budgets of 50% and 75%, and the same 100 prompts/seeds. Generation, all seven dimensions, PDF/gallery QA and authenticated publication are complete.

The authoritative final root is `out/triattention-pruning-wan-queryblocks-20261109`, mirrored locally at `output/triattention-pruning-wan-queryblocks-20261109`. Config is `configs/triattention_pruning_wan_queryblocks.json`. The preliminary root `out/triattention-pruning-wan-20261109` is superseded and excluded from final results. It transferred fixed last-frame protection and future probes from the causal implementation directly, allowing middle-frame queries to lose their own keys. Its v000 pilot led to a corrected query-block adaptation. The first final prompt also served as an implementation/protection pilot and is included among the 100 prompts. Disclose this design exposure.

The base model/revision, output format and seeds match section 11. Native Wan Q/K normalization and full 3D RoPE are unmodified. Every layer/head/step/CFG branch uses actual Q and gathered original post-RoPE K and original V. Means only guide ranking. All 64 pairs enter the scorer with the official norm correction `(E||q_f|| - ||E[q_f]||) ||k_f||`.

The full 12 x 30 x 30 token grid is split into four query blocks of three temporal latent frames. Every query is evaluated exactly once, and concatenation restores original token order. Protect all keys in the current live query block plus the first sink frame. Protected unions have 2,700 keys for the first block and 3,600 later. Retained banks have 5,400 keys/head at 50%, with 2,700/1,800 dynamically ranked first/later keys. At 75% retain 8,100 keys/head, with 5,400/4,500 dynamically ranked first/later keys. Separate selections are made per CFG forward/layer/head/block.

Ranking averages the trigonometric proxy over the block's three actual query-frame positions, using `delta_t=query_frame-key_frame`. It follows official LongLive's `delta_h=delta_w=0` ranking approximation. Native attention retains actual full 3D positional coordinates. This is a base-Wan bidirectional query-block adaptation, with a different checkpoint and generation protocol from official causal LongLive. Per-bank retention denotes attention operands. Full projections are materialized and selected-bank unions can cover many keys, so persistent KV-memory savings were not established.

### Calibration and implementation checks

Calibrate on four separate prompts p00-p03 from the original sweep, each seed 0 and all 20 denoising steps. Pool normalized pre-RoPE Q coordinates and each 2D pair norm over 4 x 20 x 10,800 = 864,000 tokens per CFG branch/layer. Keep CFG branches, 30 layers, 12 heads and all 64 pairs separate. Q center shape is [2,30,12,64,2]. The calibration prompts are text-disjoint from evaluation prompts. These statistics are static across evaluation steps. This follows the approved switch to calibrated Q ranking in the pruning rerun, distinct from the earlier current-step Q/K replacement intervention.

An independent audit checked the new production path. No coding defect was found. Its synthetic CPU dispatch audit covers a 12 x 2 x 2 grid, 48 tokens, 12 heads, 128D, one layer/branch and four query blocks. It verifies original Q views, original native K/V coordinate gathering, live/sink protection, exact retention counts and output ordering. This is routing coverage, not generated-video coverage.

`selection-verification.json` compares production factored scoring with the pinned unmodified LongLive scorer on synthetic 10,800-token keys, 12 heads, 64 pairs, all four query blocks and both budgets. Maximum score error is 2.2888e-5, or 2.1340e-6 head-standard-deviation units. All official top-k overlaps are 1.0. Original selected K/V coordinates are exact and every live query key is retained.

`independent-calibration-audit.json` reaggregates source sums in float64, then casts to float32. Both stored Q means and pair mean norms are bit-exact to that reference. The small 2.8e-6 differences reported against unrounded float64 are ordinary storage rounding.

99 native baseline videos are reused with verified hashes/provenance. v000 baseline was regenerated through the full-retention controller and its final latent tensor is bit-exact to the old native baseline. Each of the 200 new pruned videos covers 1,200 module dispatches = 20 steps x 2 branches x 30 layers. Each dispatch makes four query-block attention calls, totaling 4,800 per pruned video. All generation source/calibration hashes agree, all finite-step checks pass, and all 300 videos pass SHA256/45-frame/480x480/16-fps checks. The full local 300-video download was independently hash-checked too.

### Results and aggregation

| Dimension | Baseline | 50% retained | 75% retained |
|---|---:|---:|---:|
| Aesthetic quality | 0.58859467 | 0.53216211 | 0.57829409 |
| Imaging quality | 0.66067940 | 0.57824909 | 0.62033538 |
| Text-video consistency | 0.23255504 | 0.20447362 | 0.22691068 |
| Subject consistency | 0.96399870 | 0.88715346 | 0.94573857 |
| Background consistency | 0.97646601 | 0.93923018 | 0.96903559 |
| Motion smoothness | 0.98446269 | 0.96594978 | 0.98003205 |
| Dynamic degree | 0.33000000 | 0.53000000 | 0.44000000 |

Every cell averages 100 official per-video scores, one seed per prompt. The official frame populations are the same as section 11. Aesthetics/MUSIQ use all 45 frames, ViCLIP uses eight segment midpoints. Prompt-bootstrap 95% intervals use 10,000 paired resamples, seed 20261109, keeping all arms together within each prompt. Tokens, pairs, heads, layers and steps remain within their generated video during metric/bootstrap aggregation.

At 75%, aesthetics minus baseline is -0.0103006, CI [-0.0280579,+0.0067161], text-video consistency -0.00564436, CI [-0.0132250,+0.00160231], and imaging -0.0403440, CI [-0.0645857,-0.0172401]. Aesthetics/text intervals include zero. Imaging and subject/background/smoothness intervals are entirely below zero. At 50%, all six continuous dimensions have intervals below zero. 75% exceeds 50% in all six continuous means, with positive paired intervals. Dynamic degree measures motion incidence and rises while quality/smoothness fall.

Pruning recovers much of the quality lost by mean replacement. This remains an adaptation study on a 100-prompt custom-input VBench subset. It does not establish exact reproduction of official LongLive quality or a full official VBench overall score.

Recorded mean generation plus VAE decode times are baseline 15.4087 s, 50% 20.5654 s, 75% 22.3803 s. Baseline timing mostly comes from the previous run. Ranking, gathering and four SDPA calls make this Python prototype slower. These are descriptive cross-run timings and include full projections.

### Artifacts and publication

- Final local PDF is `output/pdf/wan-triattention-pruning-report.pdf`, 10 pages. All pages were rendered with Poppler and visually inspected. The gallery exposes all 300 videos and 2,100 per-video score cells, eight category filters and synchronized playback.
- The atlas route `/triattention-pruning/index.html` contains all 300 videos and per-prompt scores. The deployment uses authenticated read-only access. Connection settings and access credentials are managed outside git.
- `summary.json`, `per-prompt-results.csv` (700 rows), `REPORT.md`, `index.html`, calibration/manifest, selection checks, independent routing/calibration checks, browser verification and delivery manifest are in the final bundle. Authoritative raw latents/calibration tensors remain remote.
- Numerical independent checks covered 700 CSV rows, 21 means and 14 paired mean differences. DOM checks passed every prompt, video path and score. Playwright verified 480x480 video metadata/playback, zero script errors and zero desktop/mobile horizontal overflow.
- HTTP verification covers HTML, PDF, summary, a final-prompt video and a raw latent. Each denies anonymous HEAD with 403 and permits authenticated requests. The served final PDF hash matches local.
- Generation used all seven GPUs in parallel. Seven dimensions then ran independently, one per GPU. All benchmark/evaluator workers completed. Keep the private atlas server running.
- Key frozen generation files are `scripts/triattention_wan_queryblock_pruning.py` and `scripts/run_triattention_queryblock_benchmark.py`. Other new helpers are calibration, prepare, scorer-check, summarizer and report-builder scripts. Changes after generation concern reporting/publication only.

Final pruning-report SHA256 is `2f9f50e02b06c2ae232a98f0aa8705c57419b5ac71ffdee4b07f1217182987fd`. Verified manifest is `output/triattention-pruning-wan-queryblocks-20261109/delivery.json`. PDF rendering, gallery checks and private HTTP validation passed.

## 14. Latest interpretation and stopped performance investigation

### What the quality results establish

At 75% retention, imaging quality is 6.11% below baseline, with a paired 95% interval for the absolute change entirely below zero. Aesthetics is 1.75% lower and text-video consistency 2.43% lower, with both paired intervals including zero. At 50%, imaging is 12.48% lower and text consistency 12.08% lower. These relative changes divide the difference between 100-video arm means by the baseline mean. The official per-video frame populations and paired bootstrap are specified in sections 11 and 13.

The current Wan adaptation yields a modest quality-retention trade-off. A matched random/uniform-pruning control is missing, so the specific ranking advantage is unresolved. High token-wise concentration within a 2D frequency pair alone does not establish accurate key-importance ranking or negligible generation error.

### Runtime attribution and status

The measured slowdown belongs to this project's Wan adaptation. Its frozen reference code recomputes the full pre-key float32 conversion, all 64 pair norms and norm correction inside each of four query blocks. It also rebuilds selection masks, generates tie-breaking noise on CPU, copies it to GPU, gathers original K/V, and makes four separate attention calls per module. These are identifiable overhead sources. A completed component-level profile is still needed to quantify each cost.

`scripts/profile_wan_pruning.py` was added to measure native/reference trajectories and score/selection/gather/attention stages with CUDA events and wall time. A run was launched and explicitly terminated at the user's request. Partial scratch outputs under `out/pruning-performance-20261109/` are exploratory and incomplete. No optimized pruning module, verified speed fix or new quality benchmark was completed. The unused optimized-module branch in that initial profiling scaffold was removed before publication.

### Autoregressive versus full-video interpretation

Autoregressive generation can reuse historical K/V and amortize compression over later queries. Base Wan updates all video tokens during each denoising step, so the present adaptation repeats selection over changing representations. This creates a structural efficiency difference.

The RoPE relative-position identity also holds for bidirectional attention. At one layer/head/step/CFG branch, all 10,800 queries attending to 5,400 retained keys require half as many query-key interactions as full attention, before counting selection/projection overhead. Applicability to full-video attention remains open.

The current experiments do not isolate causality as the cause of quality loss. Official LongLive uses a different causal generator/LoRA checkpoint and a four-step schedule. Our base Wan uses 20 steps. The adaptation additionally uses static Q calibration pooled across four prompts/all steps, spatial-zero ranking offsets and live-query-block protection. Their separate effects require controlled comparisons. The user stopped profiling and requested interpretation only, followed by this source/handoff publication.

## 15. Source publication and deployment configuration

The source checkout includes the research handoff, geometry report, experiment plans, configurations, generation/evaluation code and report/site builders. Generated bulk archives, deployment metadata, credentials and shared access URLs are excluded from version control.

Private machine paths in documentation and cache defaults were replaced with repository-relative paths or configurable cache locations. Atlas startup now requires an explicit interface through `--bind`; SSH tunnel recovery requires explicit `--host` and `--repo` values. These portability changes follow the completed experiments. Recorded generation hashes continue to describe the original executed source versions.

Python syntax compilation, JavaScript syntax validation, CLI argument checks and all eight existing CPU/access-control tests pass. Publication checks compare every tracked blob against the locally stored access secret, private endpoint/personal-path patterns and common credential formats. Research numerical results were carried forward from the completed verified studies. Performance profiling remains stopped.

## 16. Wan2.1 learned mean-vector distillation

The real-model runtime pilots, two-GPU FSDP training, on-policy rollout sampling, checkpoint resume and one EMA student decode ran on the lab host. See `DISTILL_REPORT.md` for per-measurement scopes and limits. Calibration and both velocity references are complete. Their per-step results are in `distill-results/reference.json`, covering 32 held-out prompts, 20 denoising steps and both CFG branches. Resumed training capture uses the resident text encoder after a three-prompt exact tensor comparison. Worker elapsed time across attempts counts against the 72 GPU-hour cap. The separate static-versus-teacher benchmark reuses the original prompt and seed manifest. The concurrency limit is four GPUs. The static-versus-teacher comparison is complete across all 100 prompts and seven dimensions. Per-prompt scores, paired intervals, and agreement are in `distill-results/static-paired/`. The six-arm comparison and trained controls remain pending. Use the pinned benchmark runtime and `WAN_SNAPSHOT` from §11; the pilot logs and environment are in `out/distill-pilot/`. A node-local path recorded there holds current resumable artifacts; source and compact evidence are kept separately from bulk outputs.
