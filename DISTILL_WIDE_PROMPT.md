# Task: widened head. Full 128-dim means under RoPE + 32 highest-damage content pairs without RoPE (Q/K = 192 per head)

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).

## The architecture

Per self-attention head, layer l, denoising step s, CFG branch b (`LearnedCenter(wide=True)`, `WideHeadProcessor`):

```
q_wide = [ RoPE( mu_q[l,s,b,h] )  (128 dims, constant over tokens)  |  q[:, selected 32 pairs] (64 dims, NOT rotated) ]
k_wide = [ RoPE( mu_k[l,s,b,h] )  (128)                               |  k[:, selected 32 pairs] (64, not rotated)    ]
logit  = q_wide . k_wide / sqrt(128)         V and to_out unchanged (128)
```

- The first 128 dims are the all-pair mean vectors (as in every earlier arm, but on all 64 pairs) under the normal
  3D RoPE: content-free, so this term is a per-head relative-position bias over all 64 frequencies,
  `sum_f |mu_q,f||mu_k,f| cos(w_f delta + phi_f)` (`position_bias_tables`).
- The 64 appended dims are the real post-QK-norm projection values of the 32 **highest-damage** pairs (the pairs the
  damage ranking says carry the most content × partner energy), without RoPE: a pure content channel. In weights this
  equals appending to `to_q`/`to_k` 64 rows copied from those pairs' rows; the code gathers them at run time so the
  per-cell selection and the FSDP trainer work unchanged.
- Logit scale stays 1/sqrt(128). At deployment the position term is a bias table and attention runs on 64 content
  dims; this study measures quality only, no speed.
- Mode `delete` zeroes the position channel: content-only (position-blind) attention on the same student. That is
  the fine-tune-then-drop test for this architecture.
- Per the user's decision of 20 November, **nothing is exempt**: all 30 layers and all 20 steps
  (`--include-layer0 --include-step1`; 2,400 replacement calls and 1,200 processor calls per video).

Code (this commit): `scripts/learned_center.py` (`wide`, `content_index`, `WideHeadProcessor`),
`scripts/select_pairs.py` (`wide` variant, `--content-pairs 32`), `scripts/generate_learned_center.py` (builds the
center from the checkpoint; records `wide_head` and `content_pairs`), `scripts/train_learned_center.py` (stage 2
only; stage 1 refuses wide). `python scripts/test_learned_center.py` must pass in the GPU runtime first (CPU, about
fifteen minutes; `test_wide` checks the processor against an independent reference, exempt elements, permutation
equivariance of the content-only mode, and checkpoint round trips).

Read first: `AGENTS.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules), the shared22, deletion, swap and NoPE sections
of `DISTILL_REPORT.md`, and the `learned_center.py` docstring. Ground rules still apply: never regenerate the frozen
baseline; never delete earlier outputs; the 100 evaluation prompts and p00–p03 stay out of training; no speed claims;
no model identifiers in committed files; every reported number with its scope.

## A. Selection (minutes, CPU)

```sh
python scripts/select_pairs.py --stats out/distill-pair-stats --out out/distill-selection-v4 --pairs 22 --content-pairs 32 --include-layer0 --include-step1
```
Check: `shared` masks tensor-equal to `out/distill-selection/selection.pt`; `wide` has 64 means pairs and exactly 32
content pairs per cell, `wide: True`; `static-wide/learned_center.pt` loads with `is_wide`, `content_pairs == 32`,
`layer_active` and `step_active` all True. Report `selection-report.json` → `wide`: composition of the content set by
frequency third and `fixed_high_in_content_fraction`.

## B. Training-free screen (about 20 GPU-minutes)

```sh
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 --no-exempt \
  --selection out/distill-selection-v4/selection.pt --statistics out/distill-selection-v4/pair-statistics.pt \
  --variants shared wide --modes mean delete --out out/distill-selection-v4/screen
```
Rows: shared/mean, shared/delete (nothing exempt; references), wide/mean (the architecture with pooled calibration
means, training-free) and wide/delete (content-only). Scope: 32 held-out prompts × 20 teacher entering states × both
branches, guided w = 5, vs the native bf16 teacher, no exemptions. Expect wide/mean far from the teacher (split42
started at 0.44); proceed to C unless non-finite (**ASK**).

## C. Fine-tune (about 25 GPU-hours, two ranks; attention runs on 192-dim Q/K so each update costs about 1.3× the
shared22 recipe)

```sh
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-wide32 --steps 5000 \
  --accum 2 --lr 1e-5 --lr-center 1e-3 --warmup 250 --lambda-cfg 1.0 --lambda-feat 0.1 --ema 0.999 --clip 1.0 \
  --val-every 250 --val-prompts 8 --val-at-start --save-every 500 --ema-checkpoints \
  --resume-state-with-ema --resume-state-every 1000 --seed 20261114 \
  --selection out/distill-selection-v4/selection.pt --variant wide --calibration out/distill-selection-v4/pair-statistics.pt \
  --center-mode mean --include-layer0 --include-step1
```
Same trajectory and snapshot paths, `--budget-seconds` and `--resume` as before; resume from the last full save after
a kill, never restart from teacher weights. Gate at update 1,000: if guided validation error (8 prompts × 20 states)
is above 0.10, stop and **ASK**. Record peak memory (the widened Q/K/V add about 100 MB each per layer call at
10,800 tokens), median seconds per update, the validation curve beside shared22's (0.0301 → 0.0207 with exemptions)
and split42's (0.44 → 0.054), and the mean-norm drift (`mu_norm_q/k`): with lr 1e-3 the means are expected to move.
Audit the saved `learned_center.pt`: `wide` True, 32 content pairs per cell, 64 means pairs, no exemptions.

## D. Benchmark (about 1.3 GPU-hours)

`generate_learned_center.py --center-mode mean --student out/distill-wide32/student`, arm `wide32`, 100 prompts;
expected 2,400 intervention calls and 1,200 NoPE/processor calls per video (`selection.wide_head` true,
`content_pairs` 32). Seven VBench dimensions, paired (10,000 paired prompt resamples, NumPy seed 20261109) against the
teacher, dense, tuned shared22, shared22 deletion and split42; same-seed latent relative L2 / PSNR.

## E. Drop the position channel (about 1.5 GPU-hours, no training)

1. `screen_selections.py --val-prompts 32 --student out/distill-wide32/student --modes mean delete`.
2. `generate_learned_center.py --center-mode delete --student out/distill-wide32/student`, arm `wide32_zeros`,
   100 prompts, paired against `wide32` and the teacher. With the position channel zeroed and nothing exempt, no
   layer has any positional input: this is the clean position-blind test the split study could not give.
3. CPU diagnostic: `position_bias_tables` on the trained vs calibration means for layers {5, 15, 25}, steps
   {2, 10, 18}, both branches, every head (as in `distill-results/nope/bias-tables/`, script
   `scripts/diagnose_nope_bias.py` can be reused), saved under `distill-results/wide/bias-tables/`.

Put `wide32_zeros − wide32` in the swap table with shared22 (−0.005 aesthetics), fixed high (−0.163) and split42
(mixed).

Out of scope unless the user approves (**ASK**): other content widths (16, 48), zeros-trained wide arm, recipe changes.

## Budget and rules

About 28 GPU-hours (B 0.3, C 25, D 1.3, E 1.5). Idle GPUs only, at most two for the fine-tune and two more for
screens/benchmarks; protected outputs untouched. **ASK** before exceeding the budget, changing the recipe or the
content width, or any deletion outside your own outputs. Commit after A+B, after C (with the curve), after D and
after E, with messages listing what ran.

## Deliverables

1. `distill-results/wide/` with `selection-report.json`, the screen files, training `args.json`, `history.jsonl`,
   `validation.jsonl`, mask/width audit, final EMA SHA-256, `full100/` for `wide32` and `wide32_zeros`, `bias-tables/`.
2. A new `DISTILL_REPORT.md` section "Widened head: 128 means under RoPE + 32 unrotated content pairs": the
   architecture as above, the screen rows, the curve, the 100-prompt tables and paired intervals for both arms, the
   swap table, the mean drift and bias figure; every number with its scope (pairs vs whole Q/K, token average, step
   average, prompts, branches, exemptions: none). State plainly whether the widened student reaches shared22 quality
   and what zeroing its position channel does.
3. `handoff.md` §16 updated.
4. A final message: what ran, what failed and how it was fixed, the headline numbers, open issues.
