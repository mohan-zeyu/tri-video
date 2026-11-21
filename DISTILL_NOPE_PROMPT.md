# Task: content/position split. Means on the 42 lowest-damage pairs, NoPE on the other 22; then zero the means

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).

## Why

In every experiment so far, replacing the learned means by zeros cost little or nothing once the model was
fine-tuned (`DISTILL_REPORT.md`, deletion control and the swap section). The working explanation: the 42 pairs that
were left alone still carry RoPE, so attention never lost its positional signal; the 22 constant pairs were not
needed for position. This task tests that explanation directly by making the constant pairs the **only** positional
channel:

- per cell (layer, step, CFG branch, head), rank the 64 rotary pairs by the shared damage `damage_q + damage_k`
  (the same criterion and statistics as shared22);
- the **42 lowest-damage** pairs are replaced by learned means on both Q and K (as in shared22, but 42 instead of 22);
- the **22 highest-damage** pairs keep their content but receive **no rotary embedding** (NoPE).

With both Q and K constant on a rotated pair, that pair's logit is `|mu_q||mu_k| cos(w delta + phi)`: a per-head
additive relative-position bias, learnable through the means. The NoPE pairs give a position-blind content term. So
this student's attention is `content(NoPE pairs) + position_bias(means)`. Replacing the means by zeros then removes
every positional signal from the 29 active layers at 19 steps. If the deduction is right, this swap must collapse,
where the earlier swaps did not. Layer 0 and denoising step 1 stay exempt (native RoPE everywhere), as in all arms.

Implementation (this commit): `LearnedCenter` takes a third mask `mask_nope`; `attach()` wraps each active layer's
`attn1.processor` with `NoPEProcessor`, which feeds the processor rotary tensors with cos = 1, sin = 0 on NoPE
coordinates per batch element and head (bit-identical to native on every other pair; tested against a reference
implementation and by token-permutation equivariance). `select_pairs.py` writes the `split` variant
(`--means-pairs 42`). `screen_selections.py`, `train_learned_center.py` and `generate_learned_center.py` need no new
flags: the masks travel in `selection.pt` and in every `learned_center.pt`. `generate_learned_center.py` checks
`nope_calls == 29 layers x 2 branches x 19 steps = 1,102` per video in addition to the 2,204 replacement calls.
`python scripts/test_learned_center.py` must pass in the GPU runtime first (CPU, about fifteen minutes).

Read first: `AGENTS.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules), the shared22, deletion-control and swap
sections of `DISTILL_REPORT.md`, and the docstrings of `scripts/learned_center.py` (NoPE paragraph and
`position_bias_tables`) and `scripts/select_pairs.py` (`split`).

## A. Selection (minutes, CPU)

```sh
python scripts/select_pairs.py --stats out/distill-pair-stats --out out/distill-selection-v3 --pairs 22 --means-pairs 42
```
Check: `shared` masks tensor-equal to `out/distill-selection/selection.pt` (**ASK** if not); `split` has exactly 42
means pairs and 22 NoPE pairs per cell, disjoint, covering all 64; the shared22 set is contained in the split means
set (`shared_in_split_means_fraction == 1`). Report from `selection-report.json` → `split`: the composition by
frequency third of the means set and of the NoPE set, and `fixed_high_in_nope_fraction` (how much of the original
high third ends up NoPE). Do not overwrite `out/distill-selection/` or `-v2/`.

## B. Training-free screen (about 0.5 GPU-hours)

```sh
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 \
  --selection out/distill-selection-v3/selection.pt --statistics out/distill-selection-v3/pair-statistics.pt \
  --variants shared split --modes mean delete --out out/distill-selection-v3/screen
```
Four arms: `shared/mean` and `shared/delete` (should reproduce 0.0322 and 0.0374 guided), `split/mean` (the proposed
student, training-free: 42 pooled means + 22 NoPE pairs) and `split/delete` (position-blind attention). Scope as
always: 32 held-out prompts × 20 teacher entering states × both branches, guided w = 5, vs the native bf16 teacher.
No videos at this stage: the training-free split student is expected to be poor (32 means alone already gave
aesthetics 0.40 training-free on 30 prompts), and the fine-tune is the experiment. Proceed to C whatever the
numbers are, unless they are non-finite (then **ASK**).

## C. Fine-tune (about 21 GPU-hours, two ranks)

The tuned recipe, lengthened to 5,000 updates because the start is far from the teacher, and with a higher
learning rate on the means, which now carry the whole positional channel (at 2e-4 the means of every earlier run
stayed within 0.03% of their initialisation):

```sh
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out out/distill-split42 --steps 5000 \
  --accum 2 --lr 1e-5 --lr-center 1e-3 --warmup 250 --lambda-cfg 1.0 --lambda-feat 0.1 --ema 0.999 --clip 1.0 \
  --val-every 250 --val-prompts 8 --val-at-start --save-every 500 --ema-checkpoints \
  --resume-state-with-ema --resume-state-every 1000 --seed 20261114 \
  --selection out/distill-selection-v3/selection.pt --variant split --calibration out/distill-selection-v3/pair-statistics.pt \
  --center-mode mean
```
Same trajectory and snapshot paths, `--budget-seconds` and `--resume` as in the deletion run. Full optimizer saves
every 1,000 updates so a kill does not cost a rerun (both earlier SIGKILLs hit rank 0; resume from the last full
save, do not restart from scratch). Gate: at update 1,000, if guided validation error (8 prompts × 20 states) is
above 0.08, stop and **ASK** (the fixed-high Stage 2 run was at 0.041 at update 1,000 from a 0.283 start). Record
peak memory, median seconds per update, the validation curve, and the mean-norm drift (`mu_norm_q/k` in
`history.jsonl`, first vs last row): this time the means are expected to move.

Optional warm start, only if `split/mean` in B is above 0.3 guided: Stage 1 (means only, layer-local attention loss)
for 1,000 updates (`--stage 1 --steps 1000 --lr-center 1e-3`, same selection; about 1.5 GPU-hours), then Stage 2 as
above with `--init-student out/distill-split42-s1/student`. Say which path was taken.

## D. Benchmark the split student (about 1 GPU-hour)

`generate_learned_center.py --center-mode mean --student out/distill-split42/student`, arm `split42` (add to the
config's `modes`; expected calls 2,204 replacement and 1,102 NoPE per video; the record's `selection.nope_pairs_per_cell`
must be 22.0). Seven VBench dimensions on 100 prompts, paired (10,000 paired prompt resamples, NumPy seed 20261109)
against the teacher, dense, tuned shared22 and the shared22 deletion student; same-seed latent relative L2 and PSNR.

## E. The ablation: zero the means of the trained split student (about 1.3 GPU-hours, no training)

1. Screen: `screen_selections.py --val-prompts 32 --student out/distill-split42/student --modes mean delete`
   (`student/mean` = as trained; `student/delete` = zeros on the 42 pairs with the 22 NoPE pairs unchanged, i.e.
   position-blind self-attention in 29 layers at 19 steps).
2. Videos: `generate_learned_center.py --center-mode delete --student out/distill-split42/student`, arm
   `split42_zeros`, 100 prompts, seven dimensions, paired against `split42` and the teacher.
3. Diagnostic, CPU, minutes: for layers {5, 15, 25}, steps {2, 10, 18}, both branches, every head, evaluate
   `LearnedCenter.position_bias_tables(layer, step, branch, rope_freqs)` on the trained means and on the
   calibration means (`pair-statistics.pt`), for delta = −11..11 (t), −29..29 (h, w). Save the tables as JSON under
   `distill-results/nope/bias-tables/` and plot a few (trained vs initial) as PNG. `rope_freqs` are the 64 per-pair
   angular frequencies of Wan's rope (theta 10,000; t pairs 0–21, h 22–42, w 43–63; take them from
   `transformer.rope` or recompute from `get_1d_rotary_pos_embed`). This shows what positional prior the means
   learned, and how far they moved from the pooled statistics.

Put the swap next to the two earlier swaps (shared22: −0.005 aesthetics, interval spanning zero; fixed high:
−0.163) in one table.

Not in scope unless asked (**ASK** first): zeros-then-fine-tune for the split (training a position-blind student,
about 21 GPU-hours); other splits (32/32, 52/12); per-side selections.

## Budget and rules

About 25 GPU-hours in total (B 0.5, C 21, D 1, E 1.3, optional Stage 1 1.5). Idle GPUs only, at most two for the
fine-tune and two more for screens/benchmarks; do not disturb the pooled C2 recovery or the shared22 extension.
Preserve all earlier outputs; never regenerate the baseline; the 100 evaluation prompts and p00–p03 stay out of
training. **ASK** before: changing the split sizes, the criterion, exemptions or the recipe beyond what is written;
exceeding the budget; any deletion outside your own outputs. Commit after A+B, after C (with the curve), after D and
after E. No model identifiers in committed files; no speed claims.

## Deliverables

1. `distill-results/nope/` with `selection-report.json`, the screen files, and per run `args.json`,
   `history.jsonl`, `validation.jsonl`, mask audit (42 means / 22 NoPE per cell, exemptions), final EMA SHA-256,
   `full100/` for `split42` and `split42_zeros`, and `bias-tables/`.
2. A new `DISTILL_REPORT.md` section "Content/position split: means on 42 pairs, NoPE on 22": the definitions above,
   the screen table, the validation curve next to shared22's and the deletion control's, the 100-prompt tables and
   paired intervals for the split student and for its zeroed version, the mean-norm drift, the bias-table figure, and
   a plain statement of whether zeroing the means collapses the model once the other pairs are NoPE; every number
   with its scope (pairs vs whole Q/K, token average, step average, prompts, branches, exemptions).
3. `handoff.md` §16 updated.
4. A final message: what ran, what failed and how it was fixed, the headline numbers, open issues.
