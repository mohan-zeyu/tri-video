# Task: per-pair attention-score attribution (how much of the attention score each frequency pair contributes, in every cell)

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v` (this is
`claude/magical-goodall-dmq9i1` at `73cabcd` plus the attribution code), lab host with RTX 5090s (32 GB).
This task only reads the existing teacher trajectory cache and writes new files. It changes nothing in the
running or finished distillation, selection and removal runs.

## Why

The selection study ranks rotary pairs by `damage` (replacement error × partner energy), a proxy for the logit
change. Nothing so far measures the attention score itself: for a given head, how much of the logit that
decides where a query attends comes from each of the 64 frequency pairs. This task measures that directly,
for every layer, denoising step, CFG branch and head of every prompt, and then asks how much of that score the
selected pairs (`fixed_high`, `separate`, `shared`, `pooled`) carry.

**Hard rule from the owner: no averaging over cells.** A cell is one prompt × layer × step × branch × head. The
collector keeps every cell and every pair. Reports give distributions over a named set of cells (minimum, 5%,
25%, median, 75%, 95%, maximum, or counts) and full per-cell maps. Do not report a mean over heads, layers,
steps, branches or prompts anywhere, and do not add one to the code. Inside one cell a pair's share is, by
definition, a ratio of sums over that cell's queries and keys; the across-query spread is stored next to it.

Read first: `AGENTS.md`, the module docstring of `scripts/pair_score_attribution.py` (every field and what it
aggregates), `scripts/collect_pair_score_attribution.py`, `scripts/analyze_pair_score_attribution.py`,
`DISTILL_AGENT_PROMPT.md` §3 (ground rules still apply), `DISTILL_SELECTION_PROMPT.md` (definitions of the
selection variants).

## Definitions (use these words in the report)

For one cell, with `qr`, `kr` the normalized pre-RoPE Q/K (`attn1.norm_q` / `norm_k` outputs, all 10,800 video
tokens) rotated by the model's own RoPE angles and rounded to bf16 as the model does:

- Pair contribution `c_f[i,j] = <qr_i[f], kr_j[f]> / sqrt(128)`; the logit is exactly `s[i,j] = Σ_f c_f[i,j]`
  (full 128-dim head, every query i, every key j on all three axes).
- Row-centring: softmax ignores a per-query constant, so contributions are taken relative to the query's mean
  over keys.
- `var_cov_share[f]` = `Σ_g G[f,g] / Σ G`, with `G[f,g] = mean_i Cov_j(c_f[i,·], c_g[i,·])` computed exactly
  over all 10,800 × 10,800 query–key combinations from float64 second moments. Signed, sums to 1.
- `var_own_share[f]` = `G[f,f] / Σ_g G[g,g]`. Non-negative, sums to 1.
- `adv_share[f]` = `Σ_i A_f[i] / Σ_i Σ_g A_g[i]`, `A_f[i] = Σ_j p[i,j] c_f[i,j] − mean_j c_f[i,j]`,
  `p = softmax_j s[i,·]` over all keys: the pair's share of the logit advantage of the attended keys over the
  average key. Sums to 1.
- `top1_share[f]`: the same with `p` replaced by the single arg-max key.
- `pair_cov` = `G / Σ G` (64 × 64 per cell, float16). For a pair set S, `Σ_{f,g∈S} pair_cov` is the exact
  relative logit variance of the part removed by deleting S.
- `var_share_quantiles`, `adv_share_quantiles`: 5/25/50/75/95% quantiles across the cell's 10,800 queries of
  the per-query shares.
- Center model: Q/K replaced by their token means of the same cell, rotated at the real positions.
  `center_pair_r` = pooled row-centred correlation between a pair's actual and center-predicted contribution.

## 0. Before anything

```sh
python scripts/test_pair_score_attribution.py     # CPU, about a minute; must print ALL TESTS OK
python scripts/test_learned_center.py             # unchanged code, about ten minutes; must still pass
```

Use the pinned GPU runtime and `WAN_SNAPSHOT` of `handoff.md` §11/§16. Check `nvidia-smi` first; use idle GPUs
only and do not crowd the removal rerun (`handoff.md` §16, last paragraph).

## A. One-prompt timing pilot (one GPU)

```sh
CUDA_VISIBLE_DEVICES=<idle> python scripts/collect_pair_score_attribution.py --split calib --max-prompts 1
# -> out/distill-score-attribution/d0000.pt  (about 210 MB)
```

Each prompt replays its 20 stored entering states with both CFG branches and analyzes 30 layers × 20 steps × 2
branches = 1,200 layer calls of 12 heads. Record from the pilot: `seconds` in the record, peak GPU memory, and
check
- tensor shapes `[30, 20, 2, 12, 64]` (pair fields), `[30, 20, 2, 12, 64, 64]` (`pair_cov`), `filled` all true;
- `attention_output_validation`: 6 entries (step 0, layers 0/15/29, both branches), every
  `relative_max_err` ≤ 1e-3 (the full sweep was bit-exact, 0.0; report what you get);
- consistency with the existing statistics for the same prompt:
  `energy_q`/`energy_k` here equal `out/distill-pair-stats/d0000.pt` to bf16 replay precision.

Then project the cost of B = pilot seconds × number of prompts. **ASK** before B if the projection for the 64
calibration prompts exceeds 12 GPU-hours. Do not reach for `--query-stride` or `--steps` to make it cheaper
without asking: they reduce the study (the variance fields stay exact, the attention fields would use a subset
of queries).

If the pilot runs out of GPU memory, lower `--chunk` (queries per attention chunk, default 1350); it does not
change any result.

## B. Full collection (shard across idle GPUs)

```sh
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/collect_pair_score_attribution.py --split calib & done; wait
# -> out/distill-score-attribution/<id>.pt, 64 files, about 13.5 GB
# held-out prompts, separate directory (only if the budget allows; say so if skipped):
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/collect_pair_score_attribution.py --split val \
  --out out/distill-score-attribution-val & done; wait
```

The calibration split is the population the selections were computed on, so the selection overlay is in-sample
there and out-of-sample on `val`. Workers are resumable and claim prompts with `.claim` directories; a killed
worker leaves one claim behind, remove only that directory and restart.

Check: 64 (and 32) files, no `.claim` left, every record `filled.all()`, every validation entry ≤ 1e-3.

Optional, CPU or GPU, no model needed: the saved conditional captures of the original sweep (p00–p03, the
prompts of the atlas figures), if `out/full-sweep-wan-20261101/` is still on the data host:

```sh
python scripts/collect_pair_score_attribution.py --sweep-root out/full-sweep-wan-20261101 --out out/distill-score-attribution-sweep
```

Those records have one branch (conditional) and must be analyzed separately (step C with their own `--out`).

## C. Analysis (CPU, about 25 GB RAM for 64 prompts, tens of minutes)

```sh
python scripts/analyze_pair_score_attribution.py --stats out/distill-score-attribution \
    --selection out/distill-selection-v2/selection.pt --out distill-results/score-attribution
python scripts/analyze_pair_score_attribution.py --stats out/distill-score-attribution-val \
    --selection out/distill-selection-v2/selection.pt --out distill-results/score-attribution-val
# any single cell, all 64 pairs: prompt, layer, 0-based step, branch, head
python scripts/analyze_pair_score_attribution.py --stats out/distill-score-attribution --cell d0000 15 9 conditional 0
```

Use `out/distill-selection/selection.pt` if `-v2` (with the `pooled` variant) is not present. Outputs:
`report.json` (every distribution overall and by layer, by step, by branch, each with its scope),
`REPORT.md` (per-pair tables and the selection table), `figures/` (per-pair distribution bands, and for the
first `--atlas-prompts` prompts a full map of every layer × head × pair at all 20 steps, per branch). Needs
matplotlib; `--no-figures` skips them.

Report, each with its scope line (prompts × layers × steps × branches × heads, "every query × every key"),
always as distributions over cells:

1. **Which pairs carry the score**: per pair, the distribution of `var_cov_share` and `adv_share`
   (`pair_share`), overall and how it moves by layer, by step and by branch. State where the high-, middle- and
   low-frequency pairs of t, h and w sit against the uniform 1/64, from the per-pair tables, not from a pooled
   group mean.
2. **How concentrated a head is**: distributions and counts of the number of pairs needed for 50% and 90% of
   `var_own_share`; the largest pair's share; `largest_pair_counts` (which pair is largest in how many cells).
3. **Amplitude versus score**: per-cell Spearman of the amplitude `E‖q_f‖·E‖k_f‖` with the shares, and the share
   carried by the three axis-dominant pairs that the earlier scatter figures show.
4. **Sign and interference**: number of negative `var_cov_share` / `adv_share` entries, and the interference
   ratio (total row variance / sum of own variances).
5. **Center model**: distribution of `center_logit_r` and `center_explained_share`; `center_r_vs_mrl_counts`
   (is a concentrated pair's contribution predicted by the token means?). Note this pools every key of a cell
   and differs from the axis-line per-query Pearson of the full sweep.
6. **Across queries**: for a few named cells (use `--cell`), the across-query quantiles next to the pooled
   share, to show how much a pair's share varies between tokens.
7. **Cross-prompt stability** (`cross_prompt_stability`): per cell, each prompt's shares against the per-pair
   median over prompts.
8. **Selections** (`selection`): per variant, the distribution over prompt × cell of the score carried by the
   selected pairs (`union_var_cov_share`, `union_adv_share`, `union_deleted_logit_variance`), next to the
   smallest value any 22 pairs could reach (`lowest_own_share_set`) and the 22/64 = 0.344 of a share-blind
   choice; and the per-cell Spearman between `damage` and `var_own`. Say plainly whether the damage criterion
   picks the pairs that carry little attention score. The overlay does not apply the layer-0 / step-1
   exemptions; say so, and use the by-layer and by-step arrays if you discuss exempt cells.

Do not interpret synthetic or tiny-model numbers; only the lab records count.

## Budget and rules

About 12 GPU-hours for A+B on the calibration split, 6 more for `val`; about 25 GB disk under `out/`
(gitignored by `out/distill-*/`). Idle GPUs only, at most four concurrently. Preserve everything from earlier
runs; delete only your own outputs. Commit `distill-results/score-attribution*/` (report, tables, figures;
not the per-prompt records) and push after B and after C, with messages listing what ran. **ASK** before:
changing any definition or adding a pooled/averaged statistic, `--query-stride` > 1, a subset of steps or
prompts, exceeding the budget, or any deletion outside your own outputs. If a check fails, stop and report the
numbers instead of working around it.

## Deliverables

1. `out/distill-score-attribution*/` on the lab host (the full per-cell records) and
   `distill-results/score-attribution*/` in git.
2. A section "Per-pair attention-score attribution" in `DISTILL_REPORT.md`: the definitions above, the eight
   items, every number with its scope, no means over cells.
3. `handoff.md` §17 updated with what ran, where the records are and the GPU time used.
4. A final message: what ran, what failed and how it was fixed, the main distributions, open issues.
