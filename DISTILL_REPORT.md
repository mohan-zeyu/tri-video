# Wan2.1 mean-vector distillation

Status: real-model pilots, the calibration pool, and both held-out velocity references are complete. The training trajectory dump resumed on idle GPUs after a worker interruption. The static-versus-teacher benchmark is complete on 100 matched prompts. Full training and the six-arm benchmark are pending. The run is limited to four concurrent GPUs, 72 total GPU-hours and 300 GB of incremental disk. Only idle devices are used.

## Question and setup

Can Wan2.1-T2V-1.3B recover video quality after replacing the high-frequency parts of self-attention Q and K with learned, token-independent vectors? Frozen teacher: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at `0fad780a534b6463e45facd96134c9f345acfa5b`. Generation uses 45 frames at 480×480, 20 UniPC steps (flow shift 3), CFG 5 with an empty negative prompt, a bf16 transformer and a tiled fp32 VAE. Each step on each CFG branch evaluates 10,800 tokens and full native attention across all 64 rotary pairs.

The student replaces 22 pairs (44 coordinates: t 0–7, h 22–28, w 43–49) of **both** Q and K after QK-RMSNorm and before 3D RoPE. A separate 44-coordinate vector applies to every token at a fixed layer, step, CFG branch and head. The 42 other pairs and V stay content-dependent. Layer 0 and denoising step 1 are exempt. There is no attention kernel or inference speedup claim.

The manifests contain 3,641 distinct text prompts: 1,641 VBench prompts and 2,000 OpenVid captions, divided into 64 calibration, 32 validation and 3,545 additional training prompts. Rechecked normalized text uniqueness and disjointness from benchmark prompts and p00–p03; the six-arm evaluation manifest preserves every original benchmark prompt and seed. These are manifest counts, not completed data coverage.

## Real-model pilots

The pinned lab runtime uses NVIDIA GeForce RTX 5090, driver 580.173.02, torch 2.11.0+cu128, Diffusers 0.42.1 and Transformers 5.22.2. A native SDPA probe dispatched `aten::_scaled_dot_product_flash_attention` on a synthetic `[1,12,10800,128]` bf16 dense attention input with no mask. This probe checks the dispatcher's forward kernel for the real sequence and head dimensions.

| Pilot | Measured result |
|---|---|
| Calibration captures, GPU 3 | `d0000` 21.17 s, `d0001` 23.95 s; each includes embedding, the full teacher sampler trajectory and files, without VAE decode. Each retained 20 `[16,12,60,60]` entering latents and separate Q/K `[30,20,2,12,44]` center means. |
| Held-out captures, GPU 6 | `d0064` 21.29 s, `d0065` 24.98 s, same cached-state protocol without center hooks. |
| Stage 1, one GPU, 20 optimizer steps | Median 1.57 s/step (steps 2–20, excluding initialization/checkpoints); 7.39 GiB peak PyTorch allocated, 33.2 s training loop including save. One sampled training prompt and denoise step per optimizer step, both branches; only means train. |
| Stage 2, one GPU | Ran forward/backward and failed in the first Adam state allocation. PyTorch reported 29.18 GiB allocated and 2.81 MiB device free at the failed allocation. The model and loss were kept; switched to FSDP2. |
| Stage 2, GPUs 3 and 6, 20 optimizer steps | Median 3.60 s/step (steps 2–20; excludes validation and export); 19.58/19.57 GiB peak PyTorch allocated by rank. 395.6 s training loop including validation, export and shared-filesystem checkpoint. Per step: one prompt/denoise sample per rank, both branches. |
| Stage 3, GPUs 3 and 6, 20 optimizer steps | Four full rollouts (two refreshes × two ranks), each with 20 entering latents. Peak 19.59/19.59 GiB by rank; 199.6 s elapsed including rollouts, validation and node-local save. |
| Stage 3 resume | Loaded step 20 checkpoint; resumed at optimizer step 21. Forced both rank samples to come from student rollouts for this one smoke-test update; guided relative squared error was 0.2403, the average of those two prompt/step samples, with both branches combined at CFG 5. |
| Export and decode | `WanTransformer3DModel.from_pretrained` loaded the resumed Stage 3 EMA export; `v000` with its frozen seed decoded in 15.96 s (sampler and VAE only). All 20 latents were finite. Verified MP4 hash, 45×480×480 frame shape and 2,204 replacement calls = 29 nonexempt layers × 19 nonexempt steps × 2 branches × Q/K. This is a generation smoke test, not a quality score. |

Validation uses squared velocity error over every latent coordinate, normalized by the corresponding teacher velocity energy. Each number below averages 40 cached teacher states (two held-out prompts × 20 steps); conditional and unconditional are scored separately and guided uses `u + 5(c − u)` before computing the ratio. These checkpoints come from training on only the two calibration prompts, and the validation trajectories are off-policy in both stages.

| Pilot checkpoint | Conditional | Unconditional | Guided |
|---|---:|---:|---:|
| Stage 2 step 10 | 0.09371 | 0.09440 | 0.28456 |
| Stage 2 step 20 | 0.07327 | 0.07407 | 0.21188 |
| Stage 3 step 10 | 0.10204 | 0.10247 | 0.31071 |
| Stage 3 step 20 | 0.10078 | 0.10121 | 0.30655 |

The single GPU means-only pilot predates the fix to use the specified fixed 1e-3 means LR; it used the first 20 points of the original 500-step warmup. It establishes execution and memory, not a Stage 1 gate result. The Stage 3 short run created rollouts; the normal ramp yielded zero sampled rollout states in those 20 updates. The forced resume check verifies the actual buffer-to-loss path. Neither pilot qualifies as a complete training stage or a VBench result.

## Fixes and next evidence

Stage 1 now backpropagates each independent attention loss immediately, preserving the sum without retaining 30 backward graphs. NCCL scalar reductions use CUDA tensors. FSDP leaves fp32 RoPE inputs and schedule timesteps intact, fixing the forward/recompute dtype mismatch. Validation pads distributed work without counting repeated cells, preserving all prompt-step observations. Stage 3 uses EMA for rollouts and final export, including the means. Checkpoint publication is atomic and resume preserves the per-rank data sampler. Frozen prior benchmark source, outputs and model protocol remain intact.

Stage 0 measures precision/current-mean velocity references over the 32 validation prompts, both branches and all 20 denoise steps. Full Stage 1/2/3, dense and deletion controls, and the six-arm paired VBench results are pending. The run will pause at the Stage 2 gate if the guided held-out error plateaus above half the measured training-free reference by step 2,500. The planned final summary uses 10,000 paired prompt resamples (NumPy seed 20261109) and same-seed final latent relative L2 and frame PSNR.

During capture, two workers and the static video generator exited with status 137. The source of the kill was unconfirmed. Capture retained 236 training prompts plus all 64 calibration and 32 validation prompts. Static generation retained 33 of the planned 100 evaluation videos. Workers resumed from the last completed prompt, clearing only empty claims from exited workers.

Keeping the text encoder on GPU gave bit-for-bit identical conditional and unconditional embeddings, all 20 entering latents, and the final latent for `d0096`, `d0097`, and `d0098` with their original prompt seeds. Capture wall time per prompt, including text encoding and file writes with no video decode, was 19.18–19.34 s with offload and 12.42–12.74 s with the resident encoder. The resumed training capture uses the resident mode. Short worker invocations return to the scheduler at prompt boundaries.

## Long-run results

Calibration pooled 64 prompts. At each layer, denoising step, CFG branch and head, each prompt contributes the coordinate mean across all 10,800 normalized pre-RoPE tokens. Compare its concatenated 2D high-pair coordinates with the pool for that same cell. Medians and tenth percentiles are over prompts, all 30 layers, 20 steps, both branches and 12 heads; group boundaries are t 0–7, h 22–28 and w 43–49. Q and K are separate.

| Side | Pair group | Median cosine | P10 cosine |
|---|---|---:|---:|
| Q | t | 0.970994 | 0.875884 |
| Q | h | 0.992557 | 0.969481 |
| Q | w | 0.992834 | 0.966834 |
| K | t | 0.978754 | 0.895643 |
| K | h | 0.993214 | 0.969184 |
| K | w | 0.993446 | 0.967174 |

Temporal centers are less directionally stable across prompts than spatial centers, especially in the lower tail. This measures the direction of token averages across prompts.

Stage 0 velocity references: 32 held-out prompts × 20 entering teacher states, with one seed per prompt. Relative squared velocity error over all latent elements for one prompt, teacher entering state, denoise step and branch or guided combination. Average each step across prompts and overall equally across all prompt-step cells. High current means replace both normalized pre-RoPE Q/K on 22 high pairs across all 10800 tokens, all heads/layers, each branch/step independently, no exemptions (matches the earlier benchmark). FP32 teacher uses the same cached BF16 inputs upcast to FP32.

| Reference | Conditional | Unconditional | CFG 5 |
|---|---:|---:|---:|
| Teacher bf16 vs. fp32 | 0.000140878 | 0.0001488545 | 0.002731035 |
| Current high-pair means vs. native bf16 | 0.1008203 | 0.1012649 | 0.3499369 |

The completed comparison below pairs the static calibration arm with the frozen teacher. The trained arms and full six-arm evaluation are reported when their outputs are available.

VBench measures the benchmark's single video per seed and prompt. Aesthetic/imaging use all 45 frames (with their official normalizations); subject/background compare first/previous-frame embeddings over 44 transitions; motion smoothness uses AMT over 22 reconstructions; dynamic degree uses RAFT over 22 flow transitions after subsampling to 23 frames; text-video consistency uses ViCLIP on eight midpoint frames and the caption. See `handoff.md` §11 for the pinned evaluator and exact scopes.

Each comparison uses the same 100 prompt/seed units across arms, 10,000 paired prompt resamples and percentile 95% intervals (NumPy 20261109).

| Dimension | baseline | static |
|---|---:|---:|
| subject_consistency | 0.963999 | 0.991220 |
| background_consistency | 0.976466 | 0.991434 |
| motion_smoothness | 0.984463 | 0.991962 |
| dynamic_degree | 0.330000 | 0.000000 |
| aesthetic_quality | 0.588595 | 0.271404 |
| imaging_quality | 0.660679 | 0.486135 |
| overall_consistency | 0.232555 | 0.026384 |

| Dimension | Comparison | Paired mean difference | 95% CI |
|---|---|---:|---|
| subject_consistency | static_minus_baseline | +0.027221 | [+0.018944, +0.036092] |
| background_consistency | static_minus_baseline | +0.014968 | [+0.010736, +0.019253] |
| motion_smoothness | static_minus_baseline | +0.007499 | [+0.004637, +0.010788] |
| dynamic_degree | static_minus_baseline | -0.330000 | [-0.420000, -0.240000] |
| aesthetic_quality | static_minus_baseline | -0.317191 | [-0.334745, -0.299354] |
| imaging_quality | static_minus_baseline | -0.174544 | [-0.199351, -0.148707] |
| overall_consistency | static_minus_baseline | -0.206171 | [-0.220326, -0.192148] |

Same prompt and seed in each arm vs frozen baseline. L2 ratio over all final post-step latent coordinates, then average over prompts. Frame PSNR is 10log10(255^2/RGB MSE) over all channels/pixels of each aligned decoded MP4 frame; average finite dB values over all frames/prompts, count infinite identical frames separately.

| Arm | Prompt pairs | Aligned frames | Latent relative L2 | Mean finite frame PSNR (dB) | Identical frames |
|---|---:|---:|---:|---:|---:|
| static | 100 | 4500 | 1.142688 | 8.869 | 0 |

The static centers reduced dynamic degree from 0.33 to 0.00. VBench found no dynamic videos in the static arm across these 100 prompts. Text–video consistency and aesthetic quality both fell on all 100 prompt-and-seed pairs. The rise in subject and background consistency coincided with a drop in motion, text alignment, and visual quality. These results measure pooled centers applied to the frozen model. The learned weights and trained controls are pending.

All 200 videos passed the 45-frame, 480×480, 16 fps and checksum checks. Every static record has finite latents for all 20 sampler steps and 2,204 replacement calls, covering 29 nonexempt layers, 19 nonexempt denoising steps, both CFG branches, and Q and K. The static generation and seven VBench runs total 4,787 worker seconds (1.33 GPU-hours) across the interrupted and resumed attempts. The GPU window includes process setup and video writes. The paired per-prompt evidence is in `distill-results/static-paired/`.
