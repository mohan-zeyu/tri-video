# Task: shared22 with nothing exempt (all 30 layers, all 20 steps), then the fine-tune-then-drop swap

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).

## Why

Every arm so far left layer 0 and denoising step 1 native (1,176 of the 14,400 layer–step–branch–head cells).
That was first-run insurance from the original plan, never revisited. The one no-exemption screen that exists
(`distill-results/selection/screen-noexempt/`) says the cost on the selected pairs is small training-free:
separate/mean 0.0367 → 0.0410 guided with nothing exempt, and step 1's own error (0.083) is in line with step 2's.
The user has decided the exemptions should go. This task repeats the headline design, shared22 pairs with learned
means and the tuned 2,500-step recipe, with **no exempt layer and no exempt step**, benchmarks it, and then runs the
inference-only swap of its means for zeros, so the two results can be set next to the exempted ones.

Read first: `AGENTS.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules), the shared22, deletion-control and swap
sections of `DISTILL_REPORT.md`. No code changes are needed: `train_learned_center.py`, `select_pairs.py` and
`generate_learned_center.py` take `--include-layer0 --include-step1`; `screen_selections.py` takes `--no-exempt`.
The selection masks already cover every cell (`out/distill-selection/selection.pt`, variant `shared`).
`python scripts/test_learned_center.py` must pass in the GPU runtime first (CPU, about fifteen minutes).

Arms (all shared22 masks, 22 pairs per cell on both Q and K, 30 layers × 20 steps × 2 branches × 12 heads active):

| arm | weights | on the selected pairs | exemptions |
|---|---|---|---|
| `s22_all_means` | new 2,500-step student (A) | learned means (`--center-mode mean`) | none |
| `s22_all_zeros` | the same student | zeros at inference (`--center-mode delete`) | none |
| references | tuned shared22 (`distill-results/shared22/full100/`), its swap `s22_zeros` (`distill-results/swap/full100/`), shared22 deletion, dense, teacher | | layer 0, step 1 |

Expected intervention calls per video: 30 layers × 2 sides × 2 branches × 20 steps = **2,400** (the generator
computes this from the checkpoint's `layer_active`/`step_active`; both must be all True). The exempted arms had 2,204.

## A. Training-free screen (about 15 GPU-minutes)

```sh
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 --no-exempt \
  --variants shared --modes mean delete --out out/distill-noexempt/screen
```
Report `shared/mean` and `shared/delete` with nothing exempt next to the exempted rows (0.0322 / 0.0374) and the
earlier no-exempt separate/mean row (0.0410). Scope: 32 held-out prompts × 20 teacher entering states × both
branches, guided w = 5, vs the native bf16 teacher. Proceed to B whatever the numbers are, unless non-finite (**ASK**).

## B. Fine-tune with nothing exempt (about 10 GPU-hours, two ranks)

The tuned shared22 recipe (`distill-results/shared22/args.json`), from teacher weights, same seed, only the two
exemption flags added, plus the resumable saves that exist now:

```sh
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-s22-all --steps 2500 \
  --accum 2 --lr 1e-5 --lr-center 2e-4 --warmup 250 --lambda-cfg 1.0 --lambda-feat 0.1 --ema 0.999 --clip 1.0 \
  --val-every 250 --val-prompts 8 --val-at-start --save-every 500 --ema-checkpoints \
  --resume-state-with-ema --resume-state-every 1000 --seed 20261114 \
  --selection out/distill-selection/selection.pt --variant shared --calibration out/distill-selection/pair-statistics.pt \
  --center-mode mean --include-layer0 --include-step1
```
Same trajectory and snapshot paths, `--budget-seconds` and `--resume` as before. If rank 0 is killed again, resume
from the last full save; do not restart from teacher weights. Record peak memory, median seconds per update, the
validation curve (8 prompts × 20 states, guided; put it next to the exempted shared22 curve, 0.0301 → 0.0207), and
mean-norm drift. Audit the saved `learned_center.pt`: masks equal to the shared selection, `layer_active` and
`step_active` all True.

## C. Benchmark and swap (about 2.3 GPU-hours, no training)

1. `generate_learned_center.py --center-mode mean --student out/distill-s22-all/student`, arm `s22_all_means`,
   100 prompts, expected calls 2,400; seven VBench dimensions; paired (10,000 paired prompt resamples, NumPy seed
   20261109) against the teacher, dense, tuned shared22 (exempted) and shared22 deletion; same-seed latent relative
   L2 / PSNR.
2. Screen the student with its own masks and means, then with zeros:
   `screen_selections.py --val-prompts 32 --student out/distill-s22-all/student --modes mean delete`.
3. `generate_learned_center.py --center-mode delete --student out/distill-s22-all/student`, arm `s22_all_zeros`,
   100 prompts, expected calls 2,400, seven dimensions, paired against `s22_all_means`, the teacher, and the
   exempted swap arm `s22_zeros`.

Put the swap in the swap table with the three earlier rows (shared22 exempted: aesthetics −0.005, subject
consistency −0.008; fixed high: −0.163; split42: mixed).

Not in scope unless the user approves (**ASK** first): a zeros-trained no-exemption arm (`--center-mode delete`,
about 11 GPU-hours), or any recipe change.

## Budget and rules

About 13 GPU-hours (A 0.25, B 10, C 2.3). Idle GPUs only, at most two for the fine-tune and two more for
screens/benchmarks; do not disturb protected outputs. Preserve all earlier outputs; never regenerate the baseline;
the 100 evaluation prompts and p00–p03 stay out of training. **ASK** before exceeding the budget, changing the recipe,
or any deletion outside your own outputs. Commit after A, after B (with the curve) and after C, and hand off to
`claude` with `scripts/handoff.py send` when C is complete, or earlier with `--status question` / `interrupted`.
No model identifiers in committed files; no speed claims.

## Deliverables

1. `distill-results/noexempt/` with the screen files, training `args.json`, `history.jsonl`, `validation.jsonl`,
   mask/exemption audit, final EMA SHA-256, and `full100/` for `s22_all_means` and `s22_all_zeros`.
2. A new `DISTILL_REPORT.md` section "Shared22 with nothing exempt": the screen rows, the curve beside the
   exempted one, the 100-prompt tables and paired intervals for both arms, the swap table; every number with its
   scope (pairs vs whole Q/K, token average, step average, prompts, branches, exemptions: none). State plainly whether
   the no-exemption student is inside the paired intervals of the exempted one.
3. `handoff.md` §16 updated.
4. The hand-off message: what ran, what failed and how it was fixed, the headline numbers, open issues.
