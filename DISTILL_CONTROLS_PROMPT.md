# Task: run the two missing controls for the Wan2.1 mean-vector distillation study

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).
The main study is complete: see `DISTILL_REPORT.md`, `distill-results/`, `handoff.md` §16. Its headline is
that the stage-3 student (22 high-frequency rotary pairs of Q and K replaced by learned per-step constants,
rest of the network fine-tuned) recovers most of the quality lost by the training-free replacement, but stays
below the teacher and below the dense control on aesthetic quality (−0.06), text-video consistency (−0.015)
and imaging quality (−0.017), paired 95% intervals below zero (100 prompts, one seed each).

Two controls are missing, and they decide what the deficit means. Run them **in this order**:

1. **Narrow control (capacity-matched)**: the same 44 Q/K coordinates per head are removed outright, with
   no mean vector and no position prior, and the model is fine-tuned with the same recipe. It separates
   "fewer Q/K dimensions" from "a frozen high-frequency position prior".
2. **Deletion control**: the existing `--center-mode delete` (HIGH coordinates zeroed *after* QK-norm, same
   layer-0 and step-1 exemptions as the student). It tests whether the learned means carry any information
   beyond removing the pairs.

Read first: `AGENTS.md` (scope rule for every number), `DISTILL_PLAN.md` §1–4 and §10–11,
`DISTILL_REPORT.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules; they all still apply),
`scripts/learned_center.py`, `scripts/train_learned_center.py`, `scripts/generate_learned_center.py`.

## 0. Before anything else: preserve and make room

- The EMA exports of the student (stage 3), the stage-2 student, the dense control, `calibration.pt`, and
  `distill-results/` sources are irreplaceable. If any of them still live only under `/tmp` on the lab host
  (see `out/distill-pilot/` for the recorded path), copy them to durable storage under `out/` first and record
  the new paths in `handoff.md` §16.
- The previous run stopped at 260 of 300 GB. Free space by deleting the *resumable training checkpoints*
  (`checkpoint/` directories with sharded optimizer state) of completed stages 1, 2, 3 and dense. Keep every
  `student/` EMA export, every `validation.jsonl`/`history.jsonl`, and the 137 GB of teacher trajectories
  (they are needed for training). Do not touch `out/center-benchmark-wan-20261109` or
  `out/triattention-pruning-wan-queryblocks-20261109`.
- Budget for this task: about 32 GPU-hours (two stage-2 trainings at about 13 GPU-hours each on two GPUs,
  two 100-prompt benchmarks at about 1.2 GPU-hours each, plus margin) and about 60 GB of disk. Use idle GPUs
  only, at most four concurrently, as before. **ASK** if either limit will be exceeded.

## 1. Narrow control: definition and implementation

Definition. For every non-exempt layer (layer 0 stays intact, as in the student), in `attn1`, the outputs of
`to_q` and `to_k` have their HIGH coordinates (the 44 per head listed in `learned_center.HIGH_DIMS`, i.e. 528
of the 1,536 output features) set to zero **before** `norm_q`/`norm_k`. Consequences, which are the point:

- The QK-RMSNorm then normalizes over the 84 live coordinates per head (the zeros contribute nothing), so the
  kept coordinates are genuinely an 84-dim Q/K space, not the teacher's 128-dim space with holes.
- Native 3D RoPE still runs on all 128 coordinates; rotating zeros gives zeros, so the 22 high pairs contribute
  exactly zero to every logit. There is no position prior.
- Zeroing the outputs is mathematically identical to deleting those 528 rows of `to_q.weight`/`to_k.weight`
  and their biases; the rows receive zero gradient and are dead. This is the "parameters removed" arm. Report
  the removed count: 2 projections × 528 rows × 1,536 inputs (+ 528 biases) per non-exempt layer.
- Unlike the student, this arm cannot exempt denoising step 1: removed parameters are removed at every step.
  Apply it at all 20 steps and say so in the report. Layer 0 is exempt to match the student.

Implementation: add a mode `"narrow"` to `LearnedCenter` (or a small sibling module) that registers forward
hooks on `block.attn1.to_q` and `block.attn1.to_k` for non-exempt layers and zeroes the HIGH coordinates of
their outputs; in this mode do not hook `norm_q`/`norm_k`, do not use `mu`, and ignore the step context. Wire
`--center-mode narrow` through `train_learned_center.py` and `generate_learned_center.py` (fix
`expected_intervention_calls` for this mode: 29 layers × 2 projections × 2 branches × 20 steps = 2,320 calls
per video). Add tests to `scripts/test_learned_center.py`:

- narrow output: HIGH coordinates of `to_q`/`to_k` outputs are exactly zero; the post-norm kept coordinates
  equal `RMSNorm` applied to the zero-padded vector (compare against a manual computation);
- gradient check: after a backward pass, the gradient of `to_q.weight` on the HIGH rows is exactly zero and
  non-zero on at least one kept row;
- the existing tests still pass.

Then train it with exactly the dense control's recipe (`distill-results/dense/args.json`): stage 2,
5,000 updates, two ranks, `--center-mode narrow`, teacher weights as initialization, same learning rates,
warmup, cosine, EMA, validation every 250. Record the per-step validation curve in
`distill-results/narrow/` as for the other arms.

## 2. Deletion control

Train `--center-mode delete` with the same stage-2 recipe (5,000 updates, two ranks, teacher initialization,
default exemptions: layer 0 and step 1). Record `distill-results/delete/`.

Note for both controls: the student had stage 2 and stage 3 (on-policy); the dense control had stage 2 only,
and stage 3 changed the student's held-out velocity error by nothing measurable (0.0276 → 0.0282). Run stage 2
only for both controls so they match the dense control, and state this limitation. If budget remains after
both benchmarks, **ASK** before adding a stage 3 to the narrow arm.

## 3. Evaluation

For each control, generate the 100 benchmark videos with `generate_learned_center.py` (same prompts, seeds and
protocol; verify 45 frames, 480×480, finite, `intervention_calls == expected_intervention_calls`), run the seven
VBench dimensions with the pinned evaluator, and produce, in `distill-results/narrow-paired/` and
`distill-results/delete-paired/`, the same files as the existing arms: `summary.json`, `per-prompt-results.csv`,
`agreement.json` (same-seed final-latent relative L2 and frame PSNR vs the teacher), and paired comparisons
(10,000 paired prompt resamples, NumPy seed 20261109) against **teacher, student (stage 3), dense control, and
each other**. Reuse the audited per-prompt results of the existing arms; do not regenerate them.

## 4. Interpretation to write (in `DISTILL_REPORT.md`, new section, and `handoff.md` §16)

State every number with its scope. Then answer, with the paired intervals:

- **Narrow vs student.** If the intervals include zero on aesthetics, imaging and text-video consistency, the
  student's deficit is explained by the loss of content-dependent high-frequency Q/K (the "width"), and the
  learned position prior neither helps nor hurts. If narrow is *better* than the student, the frozen position
  prior itself hurts. If narrow is *worse*, the means contribute something.
- **Deletion vs student.** If the intervals include zero, the learned means carry no measurable information
  beyond removing the pairs: the finding becomes "Wan can be fine-tuned to work without its high-frequency
  Q/K pairs", not "learned mean vectors replace them".
- **Deletion vs narrow.** Isolates the norm-statistics and step-1 effects (deletion keeps the teacher's 128-dim
  norm scale and exempts step 1; narrow renormalizes over 84 dims at all steps).
- **Both vs dense.** Confirms the deficit is architectural, as with the student.

Do not claim any speedup; none was measured.

## 5. Deliverables

1. Code: narrow mode with tests, committed with a message that lists what changed.
2. `distill-results/narrow/`, `distill-results/delete/` (training curves, args), `distill-results/narrow-paired/`,
   `distill-results/delete-paired/`.
3. `DISTILL_REPORT.md` section with both controls, their paired tables and the interpretation above;
   `handoff.md` §16 updated with artifacts, GPU-hours used, durable paths.
4. A final message: what ran, what failed and how it was fixed, the four paired comparisons, open issues.

**ASK** before: changing the recipe or exemptions, exceeding the budget, or deleting anything other than
the resumable checkpoints named in section 0.
