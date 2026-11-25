# Task: measure where one distillation update spends its time

Repository `mohan-zeyu/tri-video`, branch `claude/profile-distill-update` (cut from
`claude/magical-goodall-dmq9i1` at `a17b768`), lab host with RTX 5090s (32 GB).

This brief is for the executor the user launches on this branch. It is not a hand-off to `codex`: do not use
`scripts/handoff.py send`. Commit and push to this branch; the draft pull request shows the result.

## Why

The fine-tunes are slow for a few 5090s: 10 GPU-hours for 2,500 updates, 21 to 25 for 5,000. Before choosing what
to build, we need to know where the time goes. Two of three levels are already measured; the third is a guess.

| Level | What is known | Source and scope |
|---|---|---|
| Whole study | training 68%, teacher capture and calibration 22%, benchmarks 8%, margin 1.5% of 71.5 GPU-hours | allocation ledger of the original Stage 0-3 study plus dense control, `DISTILL_REPORT.md` "Resource stop" table; worker elapsed time x GPUs |
| One training run | optimizer updates about 97% of 10.06 GPU-hours | no-exemption shared22 run, `distill-results/noexempt/training/`; mean of the 253 logged `optimizer_step_seconds` (7.00 s) x 2,500 updates x 2 GPUs, against 36,204 allocated GPU-seconds |
| One training sample | **not measured** | see below |

One training sample is one prompt at one denoising step with both CFG branches as a batch of 2, 10,800 tokens.
It costs 3.46 GPU-seconds: the median update time 6.911 s (250 logged updates, mean over the two ranks, validation
and saving excluded) divided by the 2 samples each rank processes per update. The split of those 3.46 s drawn in
`distill-results/profile/estimate-before-profiling.png` is an estimate:

| Part | Estimate | How it was obtained |
|---|---:|---|
| Teacher forward | <= 0.62 s (18%) | upper bound from capture: median 12.48 s per 20-step trajectory over 3,545 training prompts (`distill-results/stage0-timing.json`), including prompt encoding and file writes, divided by 20 |
| Student forward | about 0.62 s (18%) | assumed equal to the teacher forward |
| Student forward repeated for gradient checkpointing | about 0.62 s (18%) | assumed equal to the teacher forward |
| Backward, optimizer, EMA, FSDP traffic | about 1.58 s (46%) | remainder |

The largest block is a remainder, and it mixes things with different remedies: backward compute, FSDP2 all-gather
and reduce-scatter between two cards that have no NVLink, the optimizer over 1.3B fp32 parameters, and the EMA.
This task replaces the estimate with measurements. It trains nothing that is kept and changes no recipe.

Questions the measurements must answer:

1. How much of one update is the teacher forward, the student first forward, the checkpoint recomputation, and the
   rest of the backward pass?
2. How much of it is FSDP2 communication (all-gather, reduce-scatter), and in which phases does it fall?
3. How much is fixed cost per update (gradient clipping, optimizer steps, EMA, bookkeeping) rather than per sample?
4. Is any of it not GPU compute: trajectory loading, CPU-side hooks, host-to-device copies?
5. Are the cards throttling (temperature, power cap) during a run?
6. Inside one forward, how much is the self-attention kernel, the rest of self-attention, cross-attention and the FFN?

Read first: `AGENTS.md` (every number carries its scope), `DISTILL_AGENT_PROMPT.md` section 3 (ground rules),
`AGENT_TAKEOVER.md` (host layout, idle-GPU rule, lock files), the update loop in `scripts/train_learned_center.py`
(`main`, from `for step in range(start, stop_at)`), `distillation_loss` and `shard_student` in the same file.

## The recipe being measured

The no-exemption shared22 Stage 2 recipe, from teacher weights, exactly as in
`distill-results/noexempt/training/args.json`:

```sh
torchrun --nproc_per_node 2 scripts/train_learned_center.py --stage 2 --out <new profile dir> --steps 2500 \
  --accum 2 --lr 1e-5 --lr-center 2e-4 --warmup 250 --lambda-cfg 1.0 --lambda-feat 0.1 --ema 0.999 --clip 1.0 \
  --seed 20261114 --selection <selection.pt> --variant shared --calibration <pair-statistics.pt> \
  --center-mode mean --include-layer0 --include-step1  <profile flags, see below>
```

Use the trajectory directory, snapshot, selection and calibration files of that run (paths in its `args.json` and
in `distill-results/shared32/training/args.json`). If the trajectories or the snapshot are no longer on the host,
**ASK**; do not re-capture (capture cost 15.8 GPU-hours).

Keep `--steps 2500` so the learning-rate schedule is the real one, and stop early with the profile flag. Every arm
runs 5 warm-up updates, which are discarded, then 30 measured updates. With the fixed seed every arm on two ranks
draws the same prompt-and-step sequence.

## Code to add

Recommended: profile flags in `scripts/train_learned_center.py` itself, so the timings come from the real loop.
A separate script that re-implements the loop is acceptable only if it calls the trainer's own functions
(`load_models`, `shard_student`, `distillation_loss`, `ema_parameters`, ...) and the same order of operations.

Requirements:

- With the profile flags absent, the trainer's behaviour is unchanged: same samples, same losses, same files.
  `python scripts/test_learned_center.py` must pass before and after.
- A profile run writes no student, no EMA and no resume state, and runs no validation. Today the loop saves at
  `last_step == stop_at` and `save_every` is used as a modulus, so `--stop-after` alone is not enough; the profile
  mode must skip the save block.
- Output goes to a new directory under `out/distill-profile/`. Never write into an existing run directory.
- Add a CPU test on the tiny random Wan that runs the profile mode for a few updates and checks that the phases
  are all present and sum to the update time.

### Phase timers

Call `torch.cuda.synchronize(device)` at every phase boundary, then read `time.perf_counter()`. Phases, with their
boundaries in the current code:

| Phase | Per | Boundaries |
|---|---|---|
| `data` | sample | `store.sample(rng)` (mmap read and clone of `states.pt`, cached `embeddings.pt`) |
| `teacher_forward` | sample | the `torch.no_grad()` block in `distillation_loss`: host-to-device copies, teacher forward of both branches, feature taps |
| `student_forward` | sample | `forward_both_branches(student, ...)`: first pass through the 30 blocks, FSDP unshard, `LearnedCenter` hooks |
| `loss` | sample | from the end of the student forward to just before `.backward()`: the fp32 `rel_sq` terms, the feature loss on 5 layers, the `.item()` calls |
| `recompute` | sample | time spent inside block forwards that execute during `.backward()` (gradient-checkpoint recomputation) |
| `backward_rest` | sample | `.backward()` minus `recompute`: autograd backward kernels, FSDP pre-backward unshard, post-backward reduce-scatter |
| `center_sync` | update | zero-filling missing `mu` gradients and the `dist.all_reduce` over them |
| `clip` | update | both `clip_grad_norm_` calls and the gradient-norm read-back |
| `optimizer` | update | `opt_center.step()` and `opt_model.step()` |
| `ema` | update | `reshard(student)` and the `lerp_` loop |
| `bookkeeping` | update | everything else between `step_started` and the existing `optimizer_step_seconds` read: learning-rate setting, `zero_grad`, `mu` norms, logging reductions |

Notes for the implementer:

- To separate `recompute` from the first pass, set a flag around `.backward()` and time block forwards with forward
  pre/post hooks on `student.blocks[i]`. Do not rely on `torch.is_grad_enabled()` to tell the two passes apart; check
  how the installed Diffusers calls `torch.utils.checkpoint` (`_gradient_checkpointing_func`) first.
- Record the per-block times as well (30 values per pass), so first pass and recomputation can be compared block by block.
- The phases of one update must sum to its wall time within 1%. If they do not, something is untimed; find it.

### FSDP communication

Communication is reported as an overlapping category, not as an extra phase: seconds inside all-gather and
reduce-scatter, attributed to the phase in which they occur. Use the `FSDP::*` ranges of `torch.profiler` if the
installed torch emits them (look in `torch/distributed/fsdp/_fully_shard/`), otherwise wrap the collectives FSDP2
calls. Verify which functions those are in the installed version instead of assuming names. Report bytes moved per
update as well (bf16 parameters gathered, fp32 gradients reduced).

## Arms

All on idle GPUs only, under the lock rule in `AGENT_TAKEOVER.md`. Two-rank arms use the same two cards throughout.

| Arm | Setup | Purpose |
|---|---|---|
| P0 | recipe, two ranks, no timers | reference update time on today's host |
| P1 | recipe, two ranks, phase timers | the measured version of the third bar |
| P2 | recipe, two ranks, `torch.profiler` on 3 updates after warm-up, rank 0 | FSDP ranges, top 20 CUDA operators by self time |
| P3 | one GPU, no FSDP, student parameters in bf16, same loss and backward, no optimizer step and no EMA | compute floor without sharding; P1 minus P3 per phase is what sharding costs |
| P4a | P1 with `--lambda-feat 0` | cost of the feature taps and feature loss |
| P4b | P1 with `--ema 0` | cost of the EMA |
| P4c | P1 with `--accum 1` | fixed cost per update, from the intercept of accum 1 against accum 2 |
| P4d | P1 with `--no-checkpointing`, tried once | time without recomputation; running out of memory is a valid result, record the peak |
| P5 | teacher only, one GPU, no gradient, 20 forwards of both branches on stored training states | anatomy of one forward |

P3 needs a profile-only path, because fp32 master weights, gradients, Adam state and EMA for 1.3B parameters do not
fit one 32 GB card next to the teacher. Gradients in bf16 are acceptable there: the arm measures time, not loss
values. State in the report that P3's numerics differ from training.

P5 times, with a synchronize at each boundary, per block and then summed over the 30 blocks: `attn1` split into
Q/K/V projections with their norms, RoPE application, the scaled-dot-product kernel and `to_out`; `attn2`; the FFN;
block norms and residual arithmetic; and outside the blocks the patch embedding, condition embedder and output
projection. Hooks or a wrapped attention processor are both fine; the teacher's outputs must be unchanged
(check `torch.equal` against an uninstrumented forward on one state).

During every arm, sample all GPUs twice per second with `nvidia-smi` (utilisation, memory, power draw,
temperature, clock-throttle reasons if the driver reports them) into `gpu-telemetry.csv`. Record once per arm an
`nvidia-smi` snapshot of the whole host before and after, so that other users' load is visible.

Record once in `host.json`: GPU model and driver, PCIe link generation and width per card, CPU model and core
count, NUMA layout, RAM, versions of torch, CUDA, NCCL, Diffusers and Transformers, which scaled-dot-product
backend dispatches on a `[2, 12, 10800, 128]` bf16 input, and the filesystem holding the trajectories.

## Statistics and scopes

For each arm and phase:

1. Per update and rank, sum the per-sample phases over the samples of that update (2 with `--accum 2`).
2. Average the two ranks.
3. Report the median and the interquartile range across the 30 measured updates, and the mean.
4. Shares are the phase median divided by the sum of the phase medians; give the mean-based shares next to them.
5. The per-sample table is the per-update sample phases divided by the samples per update.

Every table carries a scope line: arm, ranks, updates measured and discarded, samples per update, both CFG
branches as a batch of 2, tokens, whether times are per sample or per update, per rank or rank mean, median or
mean. Keep the raw rows (one per arm, rank, update, phase) in `phase-times.jsonl`.

## Checks

1. P0 median update time against 6.911 s (no-exemption run). If it differs by more than 10%, say so, describe the
   host load, and continue.
2. P1 total against P0: timer overhead must stay under 3%. If it does not, drop the per-block timers from P1 and
   run them in a separate arm.
3. Phases sum to the update wall time within 1%.
4. The fixed cost per update from P4c (intercept) agrees with `center_sync + clip + optimizer + ema + bookkeeping`
   from P1 within 0.2 s.
5. P4a and P4b move the total by about what the `loss`/tap and `ema` phases predict. Report the differences
   whether or not they agree.

## What the result decides

Fill this table from the measurements. Each bound is the share of one update that the change could remove at most.

| Candidate change | Upper bound = measured share of |
|---|---|
| Store teacher velocities at capture and skip the teacher forward off-policy | `teacher_forward`; valid only together with dropping the feature loss, whose targets come from the same forward |
| Avoid checkpoint recomputation (partial checkpointing, or a design that needs none) | `recompute` |
| Avoid sharding (a design whose trainable state fits one card) | FSDP communication plus the P1 minus P3 gap |
| Cheaper per-update work (EMA cadence, optimizer state) | `clip + optimizer + ema` |
| Faster data path | `data` |
| Less quadratic attention work (query subsampling, fewer tokens) | P5 kernel share, applied to `student_forward + recompute + backward_rest` |

State the bounds; do not implement or time any of the changes. That is a later, separately approved task.

## Budget and rules

About 3 GPU-hours expected (nine short arms, most of the time is model loading); hard cap 4 GPU-hours, counted as
allocated wall time x GPUs including loading. Idle GPUs only, at most two at a time, no other study's outputs
touched, no baseline regenerated, nothing deleted outside `out/distill-profile/`. **ASK** before exceeding the cap,
if no two idle cards are available, if losses are non-finite, or if anything in the recipe has to change to make an
arm run (P3 and P4d excepted, as described). No model identifiers in committed files. Timings are descriptive
measurements on this host on the day of the run; make no claim about the speed of anything that was not run.

Commit after the code and CPU tests pass, after P0 to P2, and at the end.

## Deliverables

1. Code and the CPU test, with `scripts/test_learned_center.py` passing.
2. `distill-results/profile/`: `host.json`; per arm `args.json`, `phase-times.jsonl`, `summary.json`,
   `gpu-telemetry.csv`, peak memory per rank; `profiler-top-ops.json` for P2; `forward-anatomy.json` for P5;
   `README.md` with the tables and scope lines. Profiler traces stay out of git.
3. A `DISTILL_REPORT.md` section "Where one optimizer update goes": the measured per-sample split next to the
   estimate above, the per-update fixed cost, FSDP communication, the compute floor, the forward anatomy, the
   telemetry summary, the five checks, and the decision table. Every number with its scope.
4. A measured version of the third bar, same layout as `estimate-before-profiling.png`, as
   `distill-results/profile/measured-update-breakdown.png`.
5. A `handoff.md` section 16 paragraph, and a final commit message that says what ran, what failed and how it was
   handled, the headline shares, and open issues.
