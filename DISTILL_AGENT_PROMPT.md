# Task: run the Wan2.1 mean-vector distillation end to end and report the result

You are working in the repository `mohan-zeyu/tri-video` on branch `claude/magical-goodall-dmq9i1`, on a lab
host with 7x NVIDIA RTX 5090 (32 GB each). Your job is to take the implemented but never-run distillation
pipeline on that branch, run it on the real Wan2.1-T2V-1.3B model, fix what breaks, evaluate the result
against the existing 100-prompt benchmark, and write up what happened. Work autonomously through the stages
below; stop and ask only at the decision points marked **ASK**.

## 1. Read first, in this order

1. `AGENTS.md` (one rule, binding for every number you report: state where it comes from — pairs vs whole Q/K,
   one token vs averages, one step vs averages, which prompts, layers, heads, CFG branch, pre/post-RoPE).
2. `DISTILL_PLAN.md` — the full plan: student definition (section 1), data (2), losses (3), stages and
   hyperparameters (4), seeds (5), monitoring and gates (6), ablations (7), evaluation arms (8), what it does
   not deliver (9), decisions already taken (10), implementation and runbook (11).
3. `handoff.md` sections 11-14 — the completed benchmarks you must compare against (the training-free
   high-pair replacement that destroyed quality, and the TriAttention pruning rerun), the runtime that was used
   (`/dev/shm/tri-center-runtime/bin/python`, torch 2.11+cu128, diffusers 0.42.1; it may be gone — rebuild from
   `requirements-runtime.txt` in the benchmark bundle if so), `WAN_SNAPSHOT`, `HF_HOME`, and the evaluator setup
   (`VBENCH_CACHE_DIR`, pinned VBench commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`).
4. The code: `scripts/learned_center.py`, `scripts/dump_teacher_trajectories.py`,
   `scripts/train_learned_center.py`, `scripts/generate_learned_center.py`, `scripts/test_learned_center.py`,
   `configs/distill_prompts.json`, `configs/learned_center_benchmark_wan.json`.

## 2. What the student is (do not change this without asking)

Wan2.1-T2V-1.3B with one change: in every self-attention head, after Wan's QK-RMSNorm and before 3D RoPE,
the 44 coordinates of the 22 high-frequency rotary pairs (t 0-7, h 22-28, w 43-49) of Q and K are replaced by
learned constants `mu_q[l, s, b, h]`, `mu_k[l, s, b, h]`, one per layer l, denoising step s (20-step UniPC
schedule), CFG branch b and head h. Layer 0 and step 1 are exempt by default. All other weights are fine-tuned.
The teacher is the frozen original Wan. Generation protocol everywhere: 45 frames, 480x480 (10,800 tokens),
20 UniPC steps (flow_shift 3), CFG 5, empty negative prompt, bf16 transformer, fp32 tiled VAE.

The expected quality outcome is uncertain ("it may not be so good" is acceptable); the point is a clean,
controlled, well-scoped measurement. The study delivers no speedup by itself (plan section 9); never claim one.

## 3. Ground rules

- Keep the 100 evaluation prompts (`configs/center_benchmark_wan.json`) and p00-p03 out of all training data.
  `configs/distill_prompts.json` is already disjoint; do not add prompts without re-checking disjointness.
- Reuse the existing baseline videos/records from `out/center-benchmark-wan-20261109` (hard-link via
  `--link-baseline`); do not regenerate the baseline. Keep the same prompt seeds so all arms are paired.
- Do not modify the frozen benchmark scripts (`run_center_benchmark.py`, `center_replacement.py`,
  `triattention_*`). You may fix the new scripts freely; record every fix in the commit message.
- Every result is a fresh measurement; never carry a number from a plan or estimate into a result table.
  Compute estimates in `DISTILL_PLAN.md` are assumptions, not results.
- Use atomic writes, claim files and resumable workers as the existing scripts do. Never leave a job running
  unattended without a log you can read later; check `nvidia-smi` before launching and never kill or crowd out
  processes you did not start.
- Commit to `claude/magical-goodall-dmq9i1` and push after each completed stage, with clear messages.
  Generated bulk outputs (`out/distill-*`, `out/learned-center-*`) stay out of git (gitignored patterns
  `out/*benchmark*/` already exist; add `out/distill-*/` and `out/learned-center-*/` to `.gitignore`).

## 4. Stages, with gates

### Stage P — pilots (do all of this before any long run)

1. `python scripts/test_learned_center.py` must pass in the GPU runtime (it is CPU-only and small).
2. `CUDA_VISIBLE_DEVICES=0 python scripts/dump_teacher_trajectories.py --split calib --max-prompts 2`.
   Check `out/distill-traj-wan/<id>/meta.json` (20 timesteps, `states_shape [20,16,12,60,60]`), and that
   `center-means.pt` exists for calib prompts. Record seconds per prompt.
3. One-GPU training pilots on those 2 prompts, `--no-fsdp`, `--steps 20 --val-every 10 --save-every 20`, for
   stage 1, stage 2 and stage 3 (`--rollout-every 10 --rollouts-per-rank 1`). Record peak GPU memory
   (`torch.cuda.max_memory_allocated`, add a print if needed), seconds per optimizer step, and that all losses
   are finite. If stage 2 does not fit in 32 GB without FSDP, that is expected: run the stage-2 pilot with
   `torchrun --nproc_per_node 2` instead and record memory per GPU.
4. Two-GPU FSDP pilot of stage 2 and stage 3 (`torchrun --nproc_per_node 2`, 20 steps each): checkpoint save,
   `--resume`, and a rollout must all work. Then load the saved `student/transformer` with
   `WanTransformer3DModel.from_pretrained` and generate one video with `generate_learned_center.py --max-prompts 1`.
5. Record `torch.cuda.get_device_name`, driver, torch/diffusers versions and the SDPA backend in a
   `out/distill-pilot/environment.json`.

**Gate P:** everything above runs. If the full fine-tuning path cannot be made to fit or run on the 5090s
after reasonable fixes (FSDP2 sharding is already in place; options: gradient checkpointing is on, try
`--accum 1`, smaller `--rollouts-per-rank`, CPU offload of optimizer state), **ASK** before switching to LoRA.

### Stage 0 — data

Dump all trajectories, 7 workers (calib and val first, then train). Expected: 3,641 prompts x 28 MB states
= about 100 GB plus 8 MB embeddings each; check disk first. Then confirm `out/distill-traj-wan/calibration.pt`
and `calibration-spread.json` exist and report the spread numbers (cosine of per-prompt means vs pooled, per
pair group, median and p10) in your notes — this is the first full-scale measurement of center-direction
stability across prompts in this project.

Measure the two reference levels for the validation metric before training (plan section 6): the teacher's
bf16-vs-fp32 velocity error and the training-free high-pair current-mean replacement's velocity error, on the
32 validation prompts, all 20 steps, both branches, guided (w=5). Write a small script for this using
`scripts/center_replacement.py` for the training-free arm; store results in `out/distill-reference/`.

### Stage 1 — means only

`torchrun --nproc_per_node 7 scripts/train_learned_center.py --stage 1 --out out/distill-s1 --steps 2000`.
Report the per-layer attention error before (calibrated means) and after training. **Gate:** record whether
trained means reduce the median per-layer error by at least 20% relative to calibrated means; proceed either
way. Also generate the `static` arm input: a directory with only `learned_center.pt` built from
`calibration.pt` (plan section 11).

### Stage 2 — off-policy full fine-tuning

`torchrun --nproc_per_node 7 ... --stage 2 --out out/distill-s2 --steps 5000 --init-student out/distill-s1/student`.
Watch `out/distill-s2/validation.jsonl` (per-step cond/uncond/guided relative velocity errors on held-out
prompts). **Gate:** guided validation error should fall below half of the training-free reference level from
Stage 0; if it plateaus above that by step 2,500, **ASK** with the curves before continuing (options: raise
`--lr`, drop `--lambda-feat`, include/exclude layer 0, longer training).

Launch the two controls with the same recipe when GPUs are free: `--center-mode off` (dense control) and
`--center-mode delete` (deletion control), 5,000 steps each, to `out/distill-dense` and `out/distill-delete`.

### Stage 3 — on-policy

`torchrun --nproc_per_node 7 ... --stage 3 --out out/distill-s3 --steps 5000 --init-student out/distill-s2/student`.
Confirm rollouts appear in `out/distill-s3/onpolicy/` and that the `onpolicy` fraction in `history.jsonl`
ramps to 0.5. Watch `mu_norm_q/k` in the logs: if they decay toward zero, the student is learning deletion;
note it (the deletion control will tell whether it matters).

### Stage E — evaluation

Generate the six arms with `generate_learned_center.py` as listed in `DISTILL_PLAN.md` section 11
(baseline linked; static; student_s1; student = stage 3 EMA weights; dense_control; deletion), 100 prompts
each, one worker per GPU. Verify per arm: 100 videos, 45 frames, 480x480, finite, `intervention_calls ==
expected_intervention_calls` in every record.

Run the seven VBench dimensions with the repo's pinned evaluator exactly as in `handoff.md` section 11
(`prepare_center_vbench.py`, `evaluate_center_benchmark.py` with the new config and arms). The summarizer
(`summarize_center_benchmark.py`) assumes 3 arms x 100 videos; generalize it to `len(config["modes"])` arms
(keep its bootstrap: 10,000 paired resamples over prompts, NumPy seed 20261109, all arms kept together per
prompt). Report for every arm: the 7 dimension means (100 videos), paired differences vs baseline with 95%
percentile CIs, and additionally same-seed agreement with the teacher (final-latent relative L2 and frame
PSNR between each arm's video and the baseline video of the same prompt/seed).

Also report the training-time validation curves, the Stage 0 reference levels, and wall-clock/memory facts.

## 5. Deliverables

1. `DISTILL_REPORT.md` in the repo root: question, student definition, data, training as actually run
   (steps, batch, LR, wall-clock, hardware), the validation curves (figures under `output/distill/` are
   fine but keep them out of git if large; commit the numbers), the arm table with CIs, the controls'
   verdicts (does the trained student beat static means? beat deletion? how far below the dense control?),
   same-seed agreement, and an honest interpretation. Every number with its scope (AGENTS.md).
2. A new section 16 in `handoff.md` summarizing the study, artifacts, runtime, and what was not done.
3. Updated `.gitignore`, any script fixes, and the generalized summarizer — committed and pushed.
4. A short final message listing: what ran, what failed and how it was fixed, the headline paired results,
   and open issues.

## 6. Decision points (ASK before acting)

- Switching from full fine-tuning to LoRA, or changing the loss definition or the exempt set.
- Any change to the student definition in section 2.
- Spending more than roughly 3 GPU-days total, or more than 300 GB of disk.
- Deleting or overwriting anything under `out/center-benchmark-wan-20261109` or
  `out/triattention-pruning-wan-queryblocks-20261109` (never do this; ask if something seems to require it).

Everything else — fixing bugs in the new scripts, adjusting worker counts, adding logging, re-running a failed
stage from its checkpoint — do on your own and document.
