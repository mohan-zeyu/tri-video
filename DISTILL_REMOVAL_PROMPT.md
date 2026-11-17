# Task: can the shared22 pairs be removed outright? (pre-norm removal: training-free drop, then matched fine-tune)

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).

## Why

The shared22 deletion control (`DISTILL_REPORT.md`, "Shared22 deletion control") showed that zeros on the selected
pairs **after** Wan's QK-RMSNorm, fine-tuned with the tuned recipe, match the learned means on all seven VBench
dimensions (100 prompts, paired intervals spanning zero) and on validation error (0.0205 vs 0.0207 guided, 8 prompts ×
20 states). So the means are not needed. The next question is whether the pairs can be removed **before** the norm,
which is what deleting rows of `to_q`/`to_k` does: the norm then runs over the remaining coordinates, so every
surviving coordinate is scaled up per token. The only pre-norm evidence so far is the narrow control, which zeroed the
*fixed high third* at *every* step and collapsed (aesthetics 0.31 after 5,000 steps). That is not a test of the
selected pairs. This task runs that test: training-free first, then the same gentle fine-tune if quality drops.

Two removal arms, because a per-step mask and a weight drop are different things:

| arm | masks | steps | what it is |
|---|---|---|---|
| `shared/narrow` | shared22 per-cell masks (layer, step, branch, head) | layer 0 and step 1 exempt, as the deletion control | the matched pre-norm twin of the deletion control: isolates the norm-statistics effect |
| `pooled/narrow` | one set per (layer, head), same at every step and branch | layer 0 exempt, **all 20 steps** | a true weight-level removal: equivalent to deleting 22 of 64 pair rows of `to_q`/`to_k` per head in 29 layers |

Read first: `AGENTS.md`, `DISTILL_REPORT.md` (deletion and narrow sections), `DISTILL_AGENT_PROMPT.md` §3 (ground
rules still apply), and the code changes in this commit: `scripts/learned_center.py` (narrow now takes the per-cell
masks and the step exemption, and `cell_invariant()`), `scripts/select_pairs.py` (new `pooled` variant and report
fields), `scripts/screen_selections.py` (`--modes narrow`, `--all-steps`), `scripts/generate_learned_center.py`
(scope strings). `python scripts/test_learned_center.py` must pass in the GPU runtime before you start (CPU-only,
about ten minutes).

Definitions to use in the report: *pre-norm removal* = zeros on the selected coordinates of the `to_q`/`to_k` output,
before RMSNorm; *post-norm deletion* = zeros after RMSNorm (the existing control). `pooled` selection = per (layer,
head), the 22 pairs with the lowest sum over steps and branches of damage_q + damage_k from the same statistics and
criterion as `shared` (`select_pairs.py` docstring). Report `pooled_vs_shared_intersection` (how many of the 22 pairs
per cell the pooled set shares with the per-cell shared set) and `shared_vs_removal_criterion_intersection` (overlap
between the mean-replacement criterion used for shared22 and the energy × partner-energy criterion that a zero
replacement would call for); both are in `selection-report.json`.

## A. Selection (minutes, one GPU or CPU)

```sh
python scripts/select_pairs.py --stats out/distill-pair-stats --out out/distill-selection-v2 --pairs 22
```

Check: `out/distill-selection-v2/selection.pt` has variants `fixed_high, separate, shared, pooled`; `shared` masks are
`torch.equal` to those in `out/distill-selection/selection.pt` (same statistics, deterministic); `pooled` masks are
identical across steps and branches with exactly 22 pairs per (layer, head). If `shared` differs, stop and **ASK**.
Do not overwrite `out/distill-selection/`.

Static exports for the training-free video arms (narrow checkpoints with the right exemptions):

```python
import torch
from learned_center import LearnedCenter
sel = torch.load("out/distill-selection-v2/selection.pt")
for name, variant, exempt_steps in (("static-shared-narrow", "shared", (0,)), ("static-pooled-narrow", "pooled", ())):
    c = LearnedCenter(mode="narrow", exempt_layers=(0,), exempt_steps=exempt_steps, selection=sel["variants"][variant])
    torch.save({"state_dict": c.state_dict(), "mode": "narrow", "variant": variant}, f"out/distill-selection-v2/{name}/learned_center.pt")
```
(create the directories first; run from `scripts/` on the path, as the other tools do.)

## B. Training-free drop (about 2.5 GPU-hours total; shardable across idle GPUs)

B1. Velocity-error screen on the 32 held-out prompts, same metric and references as the earlier screen
(`distill-results/selection/screen/summary.md`: shared/mean 0.0322, shared/delete 0.0374, fixed_high/delete 0.4146 guided;
noise floor 0.0027):

```sh
# student exemptions (layer 0, step 1): the matched comparison with the deletion control
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/screen_selections.py --val-prompts 32 \
  --selection out/distill-selection-v2/selection.pt --statistics out/distill-selection-v2/pair-statistics.pt \
  --variants shared pooled --modes delete narrow --out out/distill-selection-v2/screen & done; wait
# weight-removal scope (layer 0 exempt, every step touched)
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/screen_selections.py --val-prompts 32 --all-steps \
  --selection out/distill-selection-v2/selection.pt --statistics out/distill-selection-v2/pair-statistics.pt \
  --variants pooled --modes delete narrow --out out/distill-selection-v2/screen-allsteps & done; wait
```

Report the six arms in one table next to the earlier shared/mean and shared/delete rows, each with its scope line
(32 prompts × 20 teacher entering states × both branches; guided w=5; exemptions as in `meta.json`).

B2. Training-free videos, first 30 benchmark prompts, teacher weights, for `shared/narrow` and `pooled/narrow`
(all steps), using the static exports from A (`generate_learned_center.py --center-mode narrow --student
out/distill-selection-v2/static-<...>-narrow --max-prompts 30`, arm names `narrow_shared22_tf` and
`narrow_pooled_tf`, added to the benchmark config's `modes` list; expected intervention calls 2,204 and 2,320).
Score aesthetics, imaging and text–video on the 30 videos with the pinned VBench, paired against the teacher and the
30-prompt training-free shared22-means row (`distill-results/static-shared30/`). Say explicitly that 30 prompts
cannot resolve differences below about 0.02 in aesthetics.

Decision after B (state which branch applies, with the numbers):
- `shared/narrow` guided error ≤ 0.05 (the regime where the gentle recipe worked, from 0.030): go to C1 as written.
- `pooled/narrow` (all steps) guided error ≤ 0.05: go to C2 as written. Between 0.05 and 0.15: run C2 as written but
  say in the report that 2,500 steps may be too few, and **ASK** before extending. Above 0.15: do not train; report
  and **ASK**.
- If `shared/narrow` is within 1.2× of `shared/delete` training-free, the norm-statistics effect is small and C1 can
  be expected to land where the deletion control did; say so.

## C. Matched fine-tune (about 11 GPU-hours per arm, two ranks each; C1 first, then C2)

Exactly the tuned recipe of shared22 and its deletion control (`distill-results/shared22-delete/args.json`), only the
mode, variant and (for C2) the step exemption differ. From teacher weights, same seed:

```sh
# C1: pre-norm removal on the shared22 per-cell masks, matched to the deletion control
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-narrow-shared22 --steps 2500 \
  --accum 2 --lr 1e-5 --lr-center 2e-4 --warmup 250 --lambda-cfg 1.0 --lambda-feat 0.1 --ema 0.999 --clip 1.0 \
  --val-every 250 --val-prompts 8 --val-at-start --save-every 500 --ema-checkpoints --seed 20261114 \
  --selection out/distill-selection-v2/selection.pt --variant shared --calibration out/distill-selection-v2/pair-statistics.pt \
  --center-mode narrow
# C2: weight-level removal (pooled set, every step)
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-narrow-pooled --steps 2500 \
  ... same flags ... --variant pooled --center-mode narrow --include-step1
```
Use the same trajectory and snapshot paths as the deletion run; `--budget-seconds` and `--resume` as you did there.
Record peak memory, median seconds per update, the validation curve (8 prompts × 20 states, guided), and the audit
that the masks in the saved `learned_center.pt` equal the selection (`cell_invariant()` must be True for C2 and
`step_active` all True; for C1 identical to the deletion control's masks).

Then for each trained arm the 100-prompt benchmark (`generate_learned_center.py --center-mode narrow --student
out/distill-narrow-<arm>/student`; arm names `narrow_shared22` and `narrow_pooled`; expected calls 2,204 and 2,320),
all seven VBench dimensions, paired comparisons (10,000 paired prompt resamples, NumPy seed 20261109) against the
teacher, dense, tuned shared22 and shared22 deletion, and same-seed latent L2 / PSNR, in the layout of
`distill-results/shared22-delete/full100/`.

## Budget and rules

About 2.5 GPU-hours for A+B, about 11 per fine-tuned arm (C1 and C2 together about 22). Idle GPUs only, at most four
concurrently; do not interrupt the shared22 5,000-update extension if it is running. Preserve all earlier outputs;
never regenerate the baseline; the 100 evaluation prompts and p00–p03 stay out of training. **ASK** before: changing
`--pairs`, the criterion, exemptions or the recipe; exceeding the budget; any deletion outside your own outputs.
Commit after A+B and after each arm of C, with messages listing what ran. No model identifiers in committed files.

## Deliverables

1. `distill-results/removal/` with `selection-report.json` (v2), the screen `complete.json`/`summary.md` files, the
   30-prompt training-free tables, and per trained arm `args.json`, `history.jsonl`, `validation.jsonl`, mask audit,
   final EMA SHA-256, and `full100/` in the usual format.
2. A new `DISTILL_REPORT.md` section "Pre-norm removal of the selected pairs": definitions above, the screen table,
   the training-free 30-prompt table, the validation curves next to the deletion control's, the 100-prompt table and
   paired intervals; every number with its scope (pairs vs whole Q/K, token average, step average, prompts, branches,
   exemptions). State plainly: (i) does pre-norm removal of the shared22 pairs match post-norm deletion after
   identical fine-tuning, and (ii) does the step-invariant pooled removal, the one that can be cut from the weights,
   hold up, and at what cost in aesthetics/imaging/text–video.
3. `handoff.md` §16 updated.
4. A final message: what ran, what failed and how it was fixed, the headline numbers, open issues. No speed claims:
   nothing here measures inference time.
