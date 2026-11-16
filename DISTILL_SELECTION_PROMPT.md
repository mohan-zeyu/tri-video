# Task: per-pair selection study (A: statistics and selection, B: training-free screen, then C: short training)

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).
Context: the completed study (`DISTILL_REPORT.md`) replaced a *fixed* third of rotary pairs (the 22
highest-frequency pairs, same set for Q and K, every head). The student recovered most of the quality but stayed
below the teacher. The hypothesis now: the fixed set is a poor choice; which pairs are safe to replace differs
by layer, step, branch and head, and differs between Q and K. This task measures that, training-free first, then
with short training runs. Nothing here changes the running or finished controls (narrow, deletion).

Read first: `AGENTS.md`, `DISTILL_PLAN.md` §1–4, `DISTILL_REPORT.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules
still apply), and the new code: `scripts/learned_center.py` (now takes per-cell Q/K pair masks; old 44-wide
checkpoints still load), `scripts/collect_pair_statistics.py`, `scripts/select_pairs.py`,
`scripts/screen_selections.py`, and the `--selection/--variant` options of `scripts/train_learned_center.py`.
`python scripts/test_learned_center.py` must pass in the GPU runtime before you start (CPU-only, a few minutes).

## Definitions (from `select_pairs.py`; use these words in the report)

For one cell (layer, step, CFG branch, head), side x ∈ {Q, K}, rotary pair f, calibration prompt p:
- `mu_f` = pooled static mean = mean over the 64 calibration prompts of the prompt's all-token mean of x_f.
- `err_pool_p,f` = E_p‖x_f − mu_f‖², the replacement error of the pooled mean on prompt p.
- `damage_x,f` = mean_p err_pool_x,p,f × mean_p E_p‖partner_f‖²: expected squared logit perturbation (up to
  1/√d) from replacing that side's pair by the pooled mean. Lower = safer.
- Selections (bool masks per cell and side, `--pairs 22` to match the original budget):
  `fixed_high` (the original third, both sides), `separate` (per cell, each side its own 22 lowest-damage
  pairs), `shared` (per cell, the 22 lowest damage_q + damage_k, same set on both sides).
- Caveat to state in the report: a pair replaced on one side only still costs a dot product at inference; only
  pairs in S_q ∩ S_k are position-only. Report the intersection size; this study measures quality, not speed.

## A. Statistics and selection (about 1 GPU-hour, no training)

```sh
# per-prompt statistics on the 64 calibration prompts (teacher replay of stored states; shardable across GPUs)
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/collect_pair_statistics.py --split calib & done; wait
# -> out/distill-pair-stats/<id>.pt  (64 files, about 22 MB each)
python scripts/select_pairs.py --stats out/distill-pair-stats --out out/distill-selection --pairs 22
# -> out/distill-selection/{selection.pt, pair-statistics.pt, selection-report.json, static-<variant>/learned_center.pt}
```

Check: 64 statistics files, no `.claim` directories left; every variant mask has exactly 22 pairs per cell and side.

Report from `selection-report.json`, each with its scope line:
1. **Cross-prompt stability**: median and p10 Jaccard between each prompt's own selection and the pooled
   selection, and the Spearman correlation of per-prompt damage vs pooled damage, for Q and for K. This is the
   test of the premise that a fixed (prompt-independent) selection is justified. Interpret: Jaccard ≥ 0.8 is
   strong support; ≤ 0.5 means the right selection is input-dependent and a static one will underperform.
2. **How different the selections are from the fixed third**: `fixed_high_in_separate_{q,k}_fraction`,
   composition by frequency third per axis, per-layer damage ratio fixed/separate, total damage by variant.
3. **Q/K disagreement**: `separate_q_k_intersection` (mean, median, p10, p90, per layer). This is the direct
   measurement of the claim that high Q concentration does not imply high K concentration.
4. Pooled RMS-concentration of selected vs unselected pairs (`mrl_*`).

## B. Training-free screen (about 1–1.5 GPU-hours; shardable)

```sh
for g in <idle gpus>; do CUDA_VISIBLE_DEVICES=$g python scripts/screen_selections.py --val-prompts 32 & done; wait
# -> out/distill-selection/screen/{complete.json, summary.md}   arms = {fixed_high, separate, shared} x {mean, delete}
```

Scope: 32 held-out prompts × 20 teacher entering states × both branches; relative squared velocity error vs the
native bf16 teacher (cond, uncond, guided w=5); exemptions as in the student (layer 0, step 1 untouched). Put the
six arms next to the Stage 0 references (noise floor 0.0027, high current means 0.350 guided; note the
reference had no exemptions, so also run `--no-exempt --out out/distill-selection/screen-noexempt` for
`fixed_high/mean` and `separate/mean` only, with `--variants fixed_high separate --modes mean`, to make one
exactly comparable row).

Decision after B:
- If `separate/mean` (or `shared/mean`) is not at least 2× lower in guided error than `fixed_high/mean`,
  the selection idea does not survive the training-free test; report that and **ASK** before training.
- The `*/delete` arms say whether the static means matter at all before training: if `separate/mean` ≈
  `separate/delete`, the means carry nothing (consistent with the deletion-control question).

## C. Short training comparison (only if B passes; about 3 GPU-hours per arm, run two arms in parallel on 4 GPUs)

Fast protocol, identical for every arm so they compare at equal budget:
```sh
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-fast-<arm> --steps 1000 \
    --lr 3e-5 --warmup 100 --lambda-feat 0 --val-every 250 --save-every 500 \
    --selection out/distill-selection/selection.pt --variant <variant> --calibration out/distill-selection/pair-statistics.pt
```
Arms: `fixed_high` (the original choice re-run at the short budget: the baseline for this comparison), the
best of `separate`/`shared` from B, and, if budget allows, the other one. Then for each arm the 100-prompt
benchmark via `generate_learned_center.py --student out/distill-fast-<arm>/student` (the checkpoint carries
its masks), the seven VBench dimensions, and paired comparisons (10,000 paired prompt resamples, seed
20261109) against teacher, the 5,000-step student, and each other. If GPU time is short, use the first 30
prompts of the manifest for all arms, say so, and note that differences below about 0.02 in aesthetics are
not resolvable at 30 prompts.

Record for every arm: peak memory, seconds per step, validation curve, and `mu` norm drift.

## Budget and rules

About 10 GPU-hours for A+B, about 12 GPU-hours for C with three arms (training + benchmarks); 40 GB disk.
Idle GPUs only, at most four concurrently. Preserve everything from the earlier runs; delete only your own
intermediate checkpoints. Commit after A, after B and after C with messages listing what ran. **ASK** before:
changing `--pairs`, the damage criterion, the exemptions or the training recipe; exceeding the budget; or any
deletion outside your own outputs.

## Deliverables

1. `distill-results/selection/` with `selection-report.json`, the screen `complete.json` files and `summary.md`,
   and (after C) the training curves and paired benchmark files in the same format as the other arms.
2. A new section in `DISTILL_REPORT.md`: definitions above, the stability numbers, the Q/K intersection,
   the screen table, and (after C) the paired table; every number with its scope. State plainly whether the
   premise (prompt-stable, side-specific selections) held.
3. `handoff.md` §16 updated.
4. A final message: what ran, what failed and how it was fixed, the headline numbers, open issues.
