# Task: how each head's pair score shares move across denoising steps (figures and step similarity, CPU only)

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. This task reads the
per-prompt attribution records that already exist and writes figures and tables. No GPU, no model, no new
collection, nothing in any other checkout is touched.

## Why

The published attribution analysis (`DISTILL_REPORT.md`, "Per-pair attention-score attribution") gives, for each
denoising step, the distribution of pair shares over cells. That says the steps look alike as populations
(neighbouring steps: per-pair medians correlate at 0.99 or more; step 1 against step 20: 0.74 for `adv_share`,
64 calibration prompts, medians over 46,080 cells per step). It does not say whether the same head keeps the same
shares from step to step. The user asked to see that directly: individual pairs against the step, for
representative heads, layer by layer, and wants the per-head similarity measured.

## Definitions

- Cell: one prompt × layer × denoising step × CFG branch × head. Its 64 pair shares pool every query × every key
  of the full 128-dim post-RoPE logits (`scripts/pair_score_attribution.py`).
- `adv_share[f]`: the pair's share of the attention-weighted logit advantage of the attended keys over the mean
  key. `var_cov_share`, `var_own_share`, `top1_share` as in the earlier brief (`SCORE_ATTRIBUTION_PROMPT.md`).
- Trajectory: one prompt × layer × CFG branch × head followed through the 20 steps.
- **No averaging.** A plotted point is one cell and one pair. A reported number is a quantile over a named set of
  trajectories. Do not add means over heads, layers, steps, branches or prompts.

## Inputs

The records written by the earlier collection, by absolute path (they are in another checkout):

- calibration, 64 prompts: `/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution`
- held-out, 32 prompts: `/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution-val`

If either directory is missing or does not hold 64 and 32 `.pt` files, stop and hand off with
`--status question`.

## 0. Tests

```sh
python scripts/test_pair_score_attribution.py     # CPU, about two minutes; must print ALL TESTS OK
```

Use the pinned runtime of `handoff.md` §11. The scripts need numpy, torch and matplotlib.

## A. Individual pairs against the step, for representative heads (minutes)

```sh
C=/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution
V=/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution-val
python scripts/plot_step_trajectories_pair_score_attribution.py --stats $C \
    --out distill-results/score-attribution/step-trajectories --prompts d0000 d0001
python scripts/plot_step_trajectories_pair_score_attribution.py --stats $V \
    --out distill-results/score-attribution-val/step-trajectories --prompts d0064
```

Per prompt (conditional branch) this writes 30 pages, one per layer, and one CSV with every plotted number. A
page has four panels, one per selected head; each line is one single rotary pair. The script picks four heads
evenly through the layer's order of largest-pair share (median over steps), and per head the three largest
pairs at the first, middle and last step, at most six. That rule only decides what is drawn.

Check: 90 PNG files and 3 CSV files; open three pages (an early, a middle and a late layer) and confirm the
titles, legends and axes are readable. Do not restyle the figures.

## B. Step-to-step similarity of every head (minutes, a few GB of RAM)

```sh
python scripts/step_similarity_pair_score_attribution.py --stats $C --out distill-results/score-attribution
python scripts/step_similarity_pair_score_attribution.py --stats $V --out distill-results/score-attribution-val
```

For every trajectory and every two steps: Pearson and Spearman across the 64 pairs, the share mass that changes
pair, and the number of pairs in common among the 5 and the 22 largest. Outputs `step-similarity.json`,
`STEP-SIMILARITY.md` and `figures/step-similarity-<measure>.png`, all as quantiles over trajectories.

Check: `trajectories` is 46,080 for calibration (64 × 30 × 2 × 12) and 23,040 for held-out; every diagonal entry
of `pearson` is 1 and of `moved_share` is 0.

## Rules

CPU only. Write nothing outside this checkout. Do not rerun the collection or the earlier analysis. **ASK**
(hand off with `--status question`) if a script raises an error, if a check above fails, or before changing any
definition, selection rule or script. No budget beyond CPU minutes is granted.

## Deliverables

1. Commit `distill-results/score-attribution*/step-trajectories/`, `step-similarity.json`, `STEP-SIMILARITY.md`
   and the new figures.
2. One paragraph in `handoff.md` §17: what ran, the file counts and the two trajectory counts. No
   `DISTILL_REPORT.md` section and no interpretation; the research session examines the output.
3. Hand off to `claude` with `scripts/handoff.py send ... --ack <this brief's commit> --status done`.
