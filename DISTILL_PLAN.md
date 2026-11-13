# Distilling Wan2.1-T2V-1.3B into a "mean-vector" student

Plan for the senior's proposal: a student with Wan's architecture in which the high-frequency third of the
rotary pairs in every self-attention head is replaced by learned constant ("mean") vectors, one set per
denoising step, trained by distillation from the frozen Wan teacher. All weights may be updated.

Status: plan. Nothing in this document has been run. Compute estimates are analytic and state their
assumptions. Repo numbers carry their scope per `AGENTS.md`.

## 1. The student, stated exactly

Teacher T: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision `0fad780a`, frozen. 30 blocks, 12 heads, 128 dims per
head, 20 UniPC steps (flow_shift 3), CFG 5, empty negative prompt, 45 frames at 480×480 (12×30×30 = 10,800
tokens). This is the configuration of every completed benchmark in the repo, so the student can be compared
pair-wise with the existing baseline/all/high arms on the same 100 prompts and seeds.

Student S: a copy of T with one change in every self-attention layer `l`, head `h`, denoising step `s` and
CFG branch `b`:

```
q = norm_q(W_q x)   reshaped to [tokens, 12 heads, 128]          (normalized, pre-RoPE, as in the teacher)
q[:, h, HIGH] = mu_q[l, s, b, h]                                  (44 coordinates, same for every token)
k[:, h, HIGH] = mu_k[l, s, b, h]
then native 3D RoPE and native attention on all 10,800 tokens, V untouched
```

- `HIGH` = the 44 coordinates of pairs t 0–7, h 22–28, w 43–49 (22 pairs), exactly the set used by the
  repo's high-pair arm (`scripts/center_replacement.py: HIGH_PAIRS`).
- `mu_q, mu_k`: fp32 parameters of shape `[30 layers, 20 steps, 2 branches, 12 heads, 44]` each.
  That is 1,267,200 parameters in total, which is 0.1% of the 1.3B model.
- "One mean per step" is taken literally: the step index `s` (1–20 of the fixed 20-step schedule) selects the
  vector. The CFG branch index `b` is also included because the repo's calibration kept branches separate
  (`handoff.md` §13) and it costs nothing. Sharing across branches is ablation A1.
- Variant for schedule independence (optional, stage 2 onward): replace the step table by a 2-layer MLP from
  Wan's timestep embedding to the 44 coordinates, initialized to reproduce the table. Use it only if the
  student must run at other step counts.
- What the change means mathematically: after RoPE, the 22 constant pairs contribute
  `sum_f |mu_q,f||mu_k,f| cos(omega_f * delta_f + phi_f)` to every logit, a learned relative-position bias
  `b_t(dt) + b_h(dh) + b_w(dw)` per (l, s, b, h). Content in those pairs is gone; the other 42 pairs and the
  rest of the network must carry it. That is why every parameter is allowed to train.
- Training-time implementation: literal substitution in the normalized Q/K, so the standard attention kernel
  and the exact teacher numerics are reused. The Q/K projection columns for `HIGH` are still computed because
  Wan's QK-RMSNorm statistics span all 1,536 dims. Dropping them, and the speed kernel that turns the constant
  pairs into bias tables, is deployment work after the quality question is settled (section 9).

Code: a `LearnedCenter` module attached to each block's `attn1`, with forward hooks on `norm_q`/`norm_k`
outputs (the same hook points as `run_center_benchmark.Intervention`), and the current `(s, b)` set by the
training loop. The existing branch check (`torch.equal` against the supplied embeddings) and the 120-hooks-per-step
count check are kept.

## 2. Data

### Prompts

- Training: at least 2,000 captions, text-disjoint from the 100 evaluation prompts in
  `configs/center_benchmark_wan.json` and from the 4 calibration prompts p00–p03. Suggested sources: the
  VBench prompt suite entries not drawn in the evaluation sample, plus OpenVid-1M or VidProM captions,
  deduplicated by normalized text against the evaluation set.
- Validation: 32 held-out training-distribution prompts (never trained on) for step-wise velocity error and
  quick visual checks.
- Evaluation: the existing 100 prompts and their seeds (section 8).

### Teacher trajectories (off-policy states)

For each training prompt `p` with seed `seed(p)` (deterministic, derived from the prompt id, independent
across prompts):

1. Encode the prompt once with umT5 (bf16); store the conditional and unconditional embeddings
   (512×4096 bf16 each, 8 MB per prompt).
2. Run the teacher's 20-step CFG-5 trajectory and store the latent entering every step, `x_s` for s = 1..20
   (16×12×60×60 bf16 = 1.4 MB each, 28 MB per prompt; 56 GB for 2,000 prompts).
3. Do not store teacher velocities; they are recomputed on the fly (so the same code serves on-policy states).

Cost: 2,000 videos at the measured 15.4 s each (lab host, descriptive timing, `handoff.md` §13) on 7 GPUs
is about 1.3 hours without VAE decode.

### Calibration of the initial means (stage 0)

On 64 training prompts, record at every (l, s, b, h) the fp32 mean over all 10,800 tokens of the normalized
pre-RoPE Q and K restricted to `HIGH`, then average over the 64 prompts. This is the per-step analogue of
`scripts/calibrate_wan_triattention.py`, which pooled all steps. Initialize `mu_q, mu_k` from it. Also store
the per-prompt means to report their spread (cosine between per-prompt and pooled centers, per pair group),
which the repo has never measured at full scale.

## 3. Losses

Notation: `v_T^b(x, s)` and `v_S^b(x, s)` are teacher and student velocity predictions (Wan predicts flow
velocity) at latent `x`, step `s`, branch `b in {c, u}`. Norms are over all latent entries of one sample.
`w = 5` is the guidance scale. Every term is a relative error so that all 20 steps weigh equally despite
different velocity scales.

```
L_vel  = sum_{b in {c,u}} ||v_S^b - v_T^b||^2 / ||v_T^b||^2
L_cfg  = ||vhat_S - vhat_T||^2 / ||vhat_T||^2,   vhat = v^u + w (v^c - v^u)
L_feat = (1/5) sum_{l in {5,10,15,20,25}} ||h_S^l - h_T^l||^2 / ||h_T^l||^2     (residual stream after block l)
L_attn = sum_l ||A_S^l(x_T^l) - A_T^l(x_T^l)||^2 / ||A_T^l(x_T^l)||^2           (stage 1 only; see below)

Stage 2/3 objective:   L = L_vel + lambda_cfg L_cfg + lambda_feat L_feat
                        lambda_cfg = 1.0,  lambda_feat = 0.1
```

Why each term:

- `L_vel`: the direct distillation target for both branches.
- `L_cfg`: the sampler uses `vhat`, which multiplies the branch difference by 5. Both earlier replacement
  arms lost text-video consistency almost completely (0.233 → 0.043 / 0.030, 100-prompt arm means,
  `handoff.md` §11), i.e. the conditional/unconditional *difference* is what broke. This term protects it.
- `L_feat`: intermediate matching speeds up early training and keeps deep layers from drifting. Drop it
  (set to 0) if validation `L_cfg` is not improved by it after stage 2; it is a convenience, not a necessity.
- `L_attn` (stage 1): layer-local. Feed each block's attention module the *teacher's* input for that layer and
  match the teacher's attention output. Every layer is independent, so one teacher forward yields 30 layers ×
  2 branches of training signal, and only `mu` is trained. This is the cheapest direct test of "train the mean
  vectors": if the means alone cannot bring per-layer error down, the rest of the network must do the work.

Not used: losses through the full 20-step rollout (backprop through 40 forwards of a 1.3B model is not
affordable and not needed), adversarial or DMD losses (the student keeps 20 steps; plain regression suffices).

## 4. Training stages

| Stage | Trainable | States | Optimizer steps | Batch (samples) | Purpose |
|---|---|---|---|---|---|
| 0 Calibrate | none | 64 teacher trajectories | – | – | initialize `mu` |
| 1 Means only | `mu_q, mu_k` | teacher states, teacher layer inputs | 2,000 | 8 | test the senior's idea in isolation; good init |
| 2 Off-policy | all weights + `mu` | teacher states | 5,000 | 8 | main distillation |
| 3 On-policy | all weights + `mu` | 50% teacher states, 50% student states | 5,000 | 8 | fix compounding error |
| 4 Deploy (optional) | all | as stage 3 | 2,000 | 8 | norm over kept dims only, dropped projections |

One sample = one (prompt, step, both branches). Steps are sampled uniformly over 1–20 so that every per-step
mean receives equal signal; re-weight toward steps whose validation error lags.

### Stage 3: on-policy distillation

The student will be run with its own sampler, so its errors accumulate across steps; training only on
teacher states never shows it the states it actually visits (the training-free arm's collapse of dynamic
degree to 0.04 is this effect at its worst). Procedure (student rollouts, teacher labels; no gradient through
the sampler):

1. Every 200 optimizer steps, generate 64 rollouts with the EMA student on fresh training prompts and fresh
   seeds, full 20-step CFG-5 UniPC, and store all 20 entering latents per rollout (1,280 states).
2. Keep a FIFO buffer of the most recent 10,000 student states.
3. Each training batch draws half its samples from teacher trajectories and half from the buffer. In both
   cases the target is the teacher's velocity at that state and step, so the loss is identical.
4. Ramp the on-policy fraction from 0 to 0.5 over the first 1,000 steps of stage 3.

Rollout cost: 64 rollouts × 40 forwards is about 15 s each on one GPU at the lab's measured speed, i.e. about
2.5 minutes per refresh on 7 GPUs, 25 refreshes per stage.

### Optimizer and precision

| Item | Setting |
|---|---|
| `mu_q, mu_k` | Adam, lr 1e-3, no weight decay (stage 1); lr 2e-4 in stages 2–3 |
| all other weights | AdamW, lr 1e-5, weight decay 0, betas (0.9, 0.99), 500-step warmup then cosine to 1e-6 |
| precision | bf16 compute, fp32 master weights, fp32 loss |
| gradient clip | 1.0 |
| EMA | 0.999 on all trained parameters; EMA weights are used for rollouts and evaluation |
| memory | gradient checkpointing per block; optimizer state sharded (FSDP/ZeRO-2) across GPUs |
| CFG branches | run as a batch of 2 through the student and the teacher |

Memory per GPU for full fine-tuning: about 2.6 GB bf16 weights + 2.6 GB grads + 15.6 GB fp32 master and
Adam (before sharding), plus a few GB of activations with checkpointing at 10,800 tokens × 2 branches. Fits
an 80 GB GPU unsharded; on 32 GB GPUs shard the optimizer state or use 8-bit Adam. If memory still binds,
fall back to LoRA rank 64 on `to_q, to_k, to_v, to_out` and the FFN (lr 1e-4) and keep `mu` fully trainable.

### Compute estimate

Per sample: teacher forward 2 branches + student forward and backward 2 branches ≈ 400 TFLOP (forward ≈ 50
TFLOP per branch at 10,800 tokens, analytic, 2 FLOPs per MAC, transformer blocks only). At an assumed 150
TFLOP/s sustained per GPU, 7 GPUs process about 2.6 samples/s:

- stage 1: 16,000 samples, but only means train and no backward through the network: under 1 hour;
- stage 2: 40,000 samples, about 4.5 hours;
- stage 3: 40,000 samples plus rollouts, about 5.5 hours;
- data generation 1.3 hours; evaluation (300 videos + 7 VBench dimensions) about 3 hours.

Roughly one day on 7 GPUs. Halve the throughput if the GPUs are consumer cards. The lab host's GPU model
is not recorded in the repo; record it in the run metadata.

## 5. Seeds

- Teacher trajectories: one seed per training prompt, `seed(p) = sha256(prompt_id) mod 2^31`, fixed.
- On-policy rollouts: `seed = base + refresh_index * 1000 + rollout_index`, never reused.
- Training data order: seeded sampler, logged.
- Evaluation: the 100 seeds already frozen in `configs/center_benchmark_wan.json`, so every arm is paired
  with the existing baseline, all-pair and high-pair videos. Add a second seed per prompt for the final claim.

## 6. Monitoring and gates

Validation every 250 optimizer steps on the 32 held-out prompts, teacher states, both branches, all 20 steps:

- `L_vel`, `L_cfg` per step (20 curves each); the per-step curves show whether a particular step's mean is
  failing.
- Same-seed agreement with the teacher: generate 8 validation videos with the EMA student and the teacher
  from identical seeds; report latent relative error at the final step and PSNR/LPIPS on decoded frames.
- `||mu||` per pair group over training. If the means shrink toward zero, the student has learned to
  *delete* the pairs; compare with the deletion control (section 8) before claiming the mean matters.
- Reference levels for `L_cfg` (measure once, before training): bf16-vs-fp32 teacher noise floor (harmless)
  and the training-free high-pair replacement on the same states (known destructive, `handoff.md` §11).
  The student's validation `L_cfg` should end much closer to the first than the second.

Gates:

- After stage 1: if trained means reduce the per-layer attention error by less than 20% relative to the
  calibrated means, record it (the senior's idea in isolation does not suffice) and proceed to stage 2 anyway;
  the rest of the network is allowed to adapt.
- After stage 2: if validation `L_cfg` is not below half of the training-free reference, consider the
  exemptions in section 7 before stage 3.
- After stage 3: run the full evaluation.

## 7. Ablations and exemptions, in priority order

- A1 branch-shared means (`[30, 20, 12, 44]`) vs branch-specific.
- A2 timestep-MLP means vs the step table.
- A3 exempt layer 0 and step 1 (both behave differently in the repo's measurements: layer 0 low-K MRL rises
  from 0.46–0.52 to 0.68–0.70 across denoising, 4 prompts × 12 heads; step 1→2 is the largest profile change).
- A4 means only (stage 1 result carried to generation) vs full fine-tuning.
- A5 replace the *low* third instead of the high third (constant low-frequency pairs are near-constant
  logits, i.e. deletion), as a check that the high-frequency choice matters.

## 8. Evaluation

Arms, all on the 100 evaluation prompts with the frozen seeds, 45×480×480, 20 steps, CFG 5:

1. Teacher baseline (already generated and verified; reuse).
2. Training-free high-pair arm with current-token means (already generated; reuse).
3. Static calibrated means, no training (stage 0 only).
4. Student after stage 1 (means trained, network frozen).
5. Student after stage 3 (full plan).
6. Dense control: the teacher fine-tuned with exactly the stage 2–3 recipe but *without* the replacement.
   This separates the effect of the architecture change from the effect of the distillation data and
   training. Without it, a quality drop cannot be attributed.
7. Deletion control: `HIGH` coordinates set to zero in Q and K, trained with the same recipe. If it matches
   arm 5, the learned "mean" carries no information beyond removing the pairs.

Metrics: the seven VBench dimensions with the repo's pinned evaluator and paired prompt bootstrap (10,000
resamples, seed 20261109), reported with the same scopes as `handoff.md` §11; same-seed teacher agreement
(final-latent relative error, frame PSNR/LPIPS); per-step validation velocity errors. Dynamic degree is
reported but never interpreted alone.

## 9. What this plan does not deliver

- Speed. The student computes the same FLOPs as the teacher during training and in arms 3–7. The saving
  (about 10% of block FLOPs at 10,800 tokens, about 13% at 32,760; analytic) needs a kernel that takes
  84-dim Q/K plus three bias tables, and stage 4 to drop the projections and renormalize over kept dims.
  Measure wall-clock only after that exists.
- A guarantee of quality. The training-free arm's per-head attention error has a heavy tail (40–61 of 360
  heads worse than a constant output at every measured step, one prompt, conditional branch, CPU pilot in
  this project's scratch notes). Full fine-tuning may or may not absorb that. The gates and the dense control
  make the outcome interpretable either way.

## 10. Open decisions for the senior

1. Means per (step, branch) as planned, or per step only?
2. Full fine-tuning (default) or LoRA, given the GPU memory on the lab host?
3. Keep layer 0 and step 1 in scope for the first run, or exempt them from the start?
4. Source of the 2,000 training prompts.
5. Whether a timestep-conditioned mean (A2) is wanted so that the student can run at other step counts.
