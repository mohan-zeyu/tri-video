# Wan2.1 mean-vector distillation

Stage 0 captured all 3,641 teacher trajectories. Stages 1, 2 and 3 completed their 2,000, 5,000 and 5,000 updates, respectively. The dense control completed 5,000 off-policy updates. Five arms have audited results for all 100 matched prompts and seven dimensions: the frozen teacher, static means, Stage 1, final Stage 3 and dense control. The run stopped before the deletion control at the resource decision point. The approved limits were four concurrent GPUs, 72 total GPU-hours and 300 GB of incremental disk. During the final student benchmark, the user reduced the active device set to GPUs 1, 3 and 6.

## Question and setup

Can Wan2.1-T2V-1.3B recover video quality after replacing the high-frequency parts of self-attention Q and K with learned, token-independent vectors? Frozen teacher: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at `0fad780a534b6463e45facd96134c9f345acfa5b`. Generation uses 45 frames at 480×480, 20 UniPC steps (flow shift 3), CFG 5 with an empty negative prompt, a bf16 transformer and a tiled fp32 VAE. Each step on each CFG branch evaluates 10,800 tokens and full native attention across all 64 rotary pairs.

The student replaces 22 pairs (44 coordinates: t 0–7, h 22–28, w 43–49) of **both** Q and K after QK-RMSNorm and before 3D RoPE. A separate 44-coordinate vector applies to every token at a fixed layer, step, CFG branch and head. The 42 other pairs and V stay content-dependent. Layer 0 and denoising step 1 are exempt. There is no attention kernel or inference speedup claim.

The manifests contain 3,641 distinct text prompts: 1,641 VBench prompts and 2,000 OpenVid captions, divided into 64 calibration, 32 validation and 3,545 additional training prompts. Rechecked normalized text uniqueness and disjointness from benchmark prompts and p00–p03; the six-arm evaluation manifest preserves every original benchmark prompt and seed. All 3,641 trajectory records passed the prompt and seed check and contain 20 entering states of shape `[16,12,60,60]`.

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

Stage 0 measured precision and current-mean velocity references over the 32 validation prompts, both branches and all 20 denoising steps. Stage 2 cleared its guided-error gate at step 2,500. Each benchmark comparison below uses 10,000 paired prompt resamples with NumPy seed 20261109 and measures same-seed final latent relative L2 and frame PSNR against the teacher.

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

Complete teacher capture, including prompt encoding, 20 denoising steps with both CFG branches and artifact writes; excludes VAE decode. The sum is across prompt wall times and does not denote elapsed wall time when workers run concurrently.

| Split | Trajectories | Median seconds | Sum of prompt seconds |
|---|---:|---:|---:|
| calib | 64 | 21.09 | 1392.7 |
| val | 32 | 20.85 | 698.7 |
| train | 3545 | 12.48 | 48793.3 |

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

Calibration, the held-out velocity references, and trajectory capture used 15.8067 GPU-hours from observed worker launch-to-exit times, including setup, retries, and the encoder parity probe. The allocator used up to four idle devices. The workload retains 136.60 GB of trajectory data, measured across the complete capture directory.

### Stage 1 means training

The median layer-local relative squared error fell from 0.280204 to 0.218263, a 22.10% reduction against the runbook’s 20% threshold. Each layer’s error compares its complete attention output over all video tokens and output coordinates, using the same frozen teacher inputs. The comparison averages the conditional and unconditional errors over 32 held-out prompts and 19 nonexempt denoising steps, then takes the median over the 29 nonexempt layers. Layer 0 and the first step are exempt. The perturbation replaces only the selected 22 normalized pre-RoPE pairs in both Q and K.

Stage 1 ran 2,000 updates at a fixed learning rate of 0.001 on GPUs 3 and 1. Each update used two independently sampled prompt-and-step states, one per rank, with both CFG branches. Elapsed time from the worker launch through exit was 5,549 s, equivalent to 3.08 GPU-hours across both ranks. Peak PyTorch allocation was 7.39 GiB on each rank. The initial Stage 2 attempt ended after 29 s when another lab process had allocated 11.49 GiB on one of its GPUs. Stage 2 restarted from the completed Stage 1 means on idle GPUs 3 and 6. Per-layer scores, the training log samples, and the configuration are in `distill-results/stage1/`.

### Stage 2 held-out validation

At optimizer step 2,500, the guided relative squared velocity error was 0.035898, clearing the 0.174968 threshold. Validation averages ratios over 32 held-out prompts and 20 cached entering teacher states per prompt. Each ratio covers all latent velocity coordinates. The guided column combines conditional and unconditional velocities at CFG 5 before computing the ratio. `distill-results/stage2/validation.jsonl` also retains the 20 denoising-step means for each checkpoint.

| Optimizer step | Conditional | Unconditional | Guided |
|---:|---:|---:|---:|
| 250 | 0.014305 | 0.017708 | 0.054301 |
| 500 | 0.012941 | 0.012885 | 0.043466 |
| 750 | 0.014037 | 0.013656 | 0.047797 |
| 1000 | 0.012497 | 0.012011 | 0.045220 |
| 1250 | 0.010652 | 0.010337 | 0.039658 |
| 1500 | 0.010823 | 0.010566 | 0.039968 |
| 1750 | 0.010055 | 0.009906 | 0.039741 |
| 2000 | 0.010034 | 0.010003 | 0.039204 |
| 2250 | 0.009077 | 0.009055 | 0.036135 |
| 2500 | 0.008831 | 0.009028 | 0.035898 |
| 2750 | 0.008438 | 0.008743 | 0.034415 |
| 3000 | 0.007776 | 0.007680 | 0.031819 |
| 3250 | 0.007532 | 0.007311 | 0.032380 |
| 3500 | 0.007674 | 0.007778 | 0.031138 |
| 3750 | 0.006960 | 0.006908 | 0.030547 |
| 4000 | 0.006680 | 0.006483 | 0.029268 |
| 4250 | 0.006494 | 0.006181 | 0.028811 |
| 4500 | 0.006414 | 0.006166 | 0.028006 |
| 4750 | 0.006289 | 0.006056 | 0.027869 |
| 5000 | 0.006186 | 0.005931 | 0.027649 |

The Stage 2 final guided error was 0.027649 over the same 640 held-out prompt-and-step states. Training used two prompt-and-step samples per update, each with both CFG branches. The base transformer learning rate was `1e-5`, and the mean learning rate was `2e-4`. They warmed up over 500 updates and used cosine decay to 10% of their base rates. Across the successful runs before and after the gate, the two ranks used 23,661 elapsed seconds, or 13.15 GPU-hours, including validation, checkpoints and final EMA export. Peak PyTorch allocation was 19.58 GiB per rank.

### Stage 1 paired video benchmark

The means-only student and teacher each generated 100 videos with the same prompt and seed manifest. Each VBench score uses the dimension-specific frame sampling described above. The paired columns use 10,000 prompt resamples with NumPy seed 20261109. Each static and Stage 1 result uses the same frozen transformer and replaces the selected high pre-RoPE pairs in both Q and K.

| Dimension | Teacher | Stage 1 | Stage 1 − teacher [95% CI] | Stage 1 − static [95% CI] |
|---|---:|---:|---:|---:|
| subject_consistency | 0.963999 | 0.987142 | +0.023143 [+0.014815, +0.031980] | -0.004078 [-0.004653, -0.003521] |
| background_consistency | 0.976466 | 0.989032 | +0.012566 [+0.008151, +0.016980] | -0.002401 [-0.003512, -0.001311] |
| motion_smoothness | 0.984463 | 0.989720 | +0.005258 [+0.002434, +0.008548] | -0.002242 [-0.002469, -0.002031] |
| dynamic_degree | 0.330000 | 0.000000 | -0.330000 [-0.420000, -0.240000] | +0.000000 [+0.000000, +0.000000] |
| aesthetic_quality | 0.588595 | 0.271410 | -0.317185 [-0.334832, -0.299144] | +0.000006 [-0.004341, +0.004224] |
| imaging_quality | 0.660679 | 0.484571 | -0.176108 [-0.200114, -0.150651] | -0.001564 [-0.007275, +0.004107] |
| overall_consistency | 0.232555 | 0.023345 | -0.209210 [-0.223406, -0.195191] | -0.003039 [-0.005935, -0.000241] |

The mean final latent relative L2 was 1.171124 against the teacher, taking one full post-step latent ratio per prompt. Mean decoded-frame PSNR was 8.701 dB across all 4,500 matched frames from the 100 prompt-and-seed pairs, with 0 identical frames. All 200 videos passed the 45-frame, 480×480, 16 fps and checksum checks. Each Stage 1 video has 20 finite steps and 2,204 replacements across 29 active layers, 19 active denoising steps, both branches and Q and K.

The Stage 1 improvement in isolated attention output error did not carry over to restored video motion or text alignment. Dynamic degree remained zero in both means-only arms, and Stage 1 text–video consistency fell by 0.00304 versus static means. The interval for the aesthetic change spans zero. The paired scores and agreement records are in `distill-results/s1-paired/`.

### Stage 2 held-out video agreement

Seven of eight scheduled validation prompts completed before the device timeouts. For the seven completed prompt-and-seed pairs, the Stage 2 EMA student had a mean final latent relative L2 of 0.588189 over all final latent coordinates per pair. Mean decoded-frame PSNR was 13.464 dB over all 315 aligned frames. This diagnostic uses the validation split and its teacher renderings. The VBench arms use the separate 100-prompt benchmark.

### Stage 3 on-policy training

Stage 3 completed 5,000 updates on two ranks, using the Stage 2 EMA export as initialization. The loss, base learning rates, warmup and cosine schedule were the same as Stage 2. Each rollout refresh proposed 64 prompt-and-seed trajectories, each with 20 entering latents. Over 401 logged optimizer steps after step 1,000, 48.0% of the two sampled states per logged update came from student rollouts. The target still used the frozen teacher at the sampled state.

After checkpoint 2,000, the rollout split used GPUs 1 and 5 for part of two refreshes while GPUs 3 and 6 continued as training ranks. The replica test matched all 20 entering latents bit for bit on three fixed prompt-and-seed rollouts from the same Stage 2 EMA snapshot. The first split finished in 267.85 s after the export, and the second timed out at 330 s when the spare devices were contested. Both training ranks regenerated that refresh’s remaining jobs using the same prompt and seed lists. The run charged 1,340 GPU-seconds for the two spare-worker allocations.

The two trainer ranks ran roughly 9.77 hours between Stage 3 launch and final checkpoint, including rollouts, validation and restart. This is roughly 19.54 GPU-hours before the separate spare-worker charge. The resumed run used at most 19.58 GiB of PyTorch allocations per rank. The median logged optimizer step took 3.66 s, excluding time before each optimizer step such as rollout refreshes. The mean Euclidean norm of the 44 selected high-pair coordinates, averaged across all layers, denoising steps, CFG branches and heads, changed from 10.72936 to 10.72580 for Q and 11.76144 to 11.75797 for K. These are table vector norms, including exempt slots, with no token averaging at measurement time.

The table uses the same 640 held-out teacher states and CFG-5 relative squared velocity errors as Stage 2.

| Optimizer step | Conditional | Unconditional | Guided |
|---:|---:|---:|---:|
| 250 | 0.006817 | 0.006565 | 0.029448 |
| 500 | 0.010304 | 0.009766 | 0.035846 |
| 750 | 0.008976 | 0.008637 | 0.036808 |
| 1000 | 0.009967 | 0.009901 | 0.037512 |
| 1250 | 0.009617 | 0.009222 | 0.037400 |
| 1500 | 0.009435 | 0.009461 | 0.036514 |
| 1750 | 0.009647 | 0.009608 | 0.037056 |
| 2000 | 0.008668 | 0.008507 | 0.037017 |
| 2250 | 0.009690 | 0.009199 | 0.035845 |
| 2500 | 0.008514 | 0.008356 | 0.037032 |
| 2750 | 0.007459 | 0.007278 | 0.032017 |
| 3000 | 0.007650 | 0.007718 | 0.033970 |
| 3250 | 0.006815 | 0.006648 | 0.030924 |
| 3500 | 0.006885 | 0.006860 | 0.031200 |
| 3750 | 0.006556 | 0.006341 | 0.029898 |
| 4000 | 0.006435 | 0.006190 | 0.029437 |
| 4250 | 0.006202 | 0.006050 | 0.028711 |
| 4500 | 0.006131 | 0.005947 | 0.029119 |
| 4750 | 0.006097 | 0.005917 | 0.028629 |
| 5000 | 0.005997 | 0.005781 | 0.028238 |

### Final Stage 3 paired video benchmark

The teacher and Stage 3 EMA student each contributed 100 videos matched by prompt and seed, using the original 100-prompt manifest. Each score averages 100 per-video values using the metric-specific VBench frame sampling described above. The same 10,000 bootstrap samples of the 100 matched prompts, with NumPy seed 20261109, keep all arms together within each sampled prompt. The static and Stage 1 comparisons use their audited per-prompt results and the same 100 teacher video checksums. All three center-replacement arms intervene on the same 22 high pre-RoPE pairs in Q and K. The final student uses its trained transformer and learned constants.

| Dimension | Teacher | Stage 3 | Stage 3 − teacher [95% CI] | Stage 3 − static [95% CI] | Stage 3 − Stage 1 [95% CI] |
|---|---:|---:|---:|---:|---:|
| subject_consistency | 0.963999 | 0.958942 | -0.005056 [-0.011926, +0.002087] | -0.032278 [-0.041341, -0.023791] | -0.028200 [-0.037269, -0.019814] |
| background_consistency | 0.976466 | 0.974867 | -0.001599 [-0.004941, +0.001841] | -0.016566 [-0.020690, -0.012616] | -0.014165 [-0.018447, -0.010204] |
| motion_smoothness | 0.984463 | 0.985119 | +0.000657 [-0.001284, +0.002433] | -0.006843 [-0.009766, -0.004313] | -0.004601 [-0.007500, -0.002072] |
| dynamic_degree | 0.330000 | 0.210000 | -0.120000 [-0.210000, -0.030000] | +0.210000 [+0.130000, +0.290000] | +0.210000 [+0.130000, +0.290000] |
| aesthetic_quality | 0.588595 | 0.517914 | -0.070681 [-0.086809, -0.053861] | +0.246510 [+0.228104, +0.264980] | +0.246504 [+0.228028, +0.265220] |
| imaging_quality | 0.660679 | 0.652749 | -0.007931 [-0.025771, +0.010023] | +0.166614 [+0.137478, +0.193499] | +0.168177 [+0.139276, +0.195316] |
| overall_consistency | 0.232555 | 0.219662 | -0.012893 [-0.020745, -0.005342] | +0.193278 [+0.178278, +0.208606] | +0.196317 [+0.181334, +0.211642] |

The final student exceeded both static means and Stage 1 on motion incidence, aesthetics, imaging quality and text–video consistency. It scored lower than both on subject and background consistency and motion smoothness. Against the teacher, the paired intervals were below zero for motion incidence, aesthetics and text–video consistency. The four remaining intervals included zero. The contribution of on-policy training to video quality remains unresolved because Stage 2 has only the separate seven-prompt validation video diagnostic.

Across the 100 seed-matched pairs, the mean final latent relative L2 was 0.586365, taking the norm ratio across all coordinates of each final latent and then averaging over prompts. Mean decoded-frame PSNR was 13.773 dB over all 4,500 aligned frames, with zero identical frames. The video audit confirmed 45 frames, 480×480, 16 fps and matching checksums for all 200 videos. Each Stage 3 video had 20 finite denoising states and 2,204 replacement calls across 29 active layers, 19 active steps, both CFG branches and Q and K. Median student render-and-decode time was 15.71 seconds per prompt, excluding text encoding and MP4 export.

The main resumed GPU 1 worker used 4,166 seconds for rendering and VBench. The first attempt on GPU 5 used 271 seconds and saved four completed videos before a shared-host out-of-memory failure. A later GPU 5 worker saved one more video before being stopped at the user's request, after 51 seconds. Total charged accelerator time was 4,488 GPU-seconds, or 1.25 GPU-hours. The CPU finished the video audit and agreement analysis after the scorer released the GPU. Artifacts are in `distill-results/student-paired/`. The lab's prompt-level videos are under `out/learned-center-wan-student/videos/`.

### Dense control training

The dense control started from the Wan2.1 teacher checkpoint and used full Q and K with every rotary pair active, including layer 0 and the first denoising step. It ran the Stage 2 objective for 5,000 off-policy updates, using two prompt-and-step samples per optimizer update, each with both CFG branches. Its transformer learning rate was `1e-5` with 500 warmup updates and cosine decay to 10% of the base rate. The two RTX 5090 ranks ran for 22,637 elapsed seconds including setup, validation, checkpoints and EMA export, or 12.58 GPU-hours. Peak PyTorch allocation was 19.58 GiB per rank.

This curve uses the same 640 held-out teacher states and CFG-5 relative squared velocity errors as Stages 2 and 3. Every 250-update checkpoint, including its 20 per-denoise-step values, is in `distill-results/dense/validation.jsonl`.

| Optimizer step | Conditional | Unconditional | Guided |
|---:|---:|---:|---:|
| 250 | 0.003609 | 0.006080 | 0.027740 |
| 1000 | 0.006606 | 0.006540 | 0.034220 |
| 2000 | 0.004874 | 0.005090 | 0.027256 |
| 2500 | 0.003757 | 0.003967 | 0.023889 |
| 3000 | 0.002944 | 0.003052 | 0.021911 |
| 4000 | 0.002008 | 0.001898 | 0.018360 |
| 5000 | 0.001458 | 0.001334 | 0.015493 |

### Dense control paired video benchmark

Each mean covers one video per prompt on the same 100 prompt-and-seed pairs and the VBench frame sampling specified above. The direct Stage 3 comparison uses the same prompt indices, teacher video checksums and 10,000 paired prompt resamples with NumPy seed 20261109.

| Dimension | Teacher | Dense | Dense − teacher [95% CI] | Stage 3 − dense [95% CI] |
|---|---:|---:|---:|---:|
| subject_consistency | 0.963999 | 0.964285 | +0.000286 [-0.004789, +0.005414] | -0.005342 [-0.010979, +0.000146] |
| background_consistency | 0.976466 | 0.977676 | +0.001210 [-0.001541, +0.004178] | -0.002808 [-0.005425, -0.000076] |
| motion_smoothness | 0.984463 | 0.985506 | +0.001043 [-0.000675, +0.002753] | -0.000386 [-0.001586, +0.000865] |
| dynamic_degree | 0.330000 | 0.250000 | -0.080000 [-0.150000, -0.020000] | -0.040000 [-0.120000, +0.040000] |
| aesthetic_quality | 0.588595 | 0.578063 | -0.010532 [-0.022586, +0.001557] | -0.060149 [-0.075305, -0.044672] |
| imaging_quality | 0.660679 | 0.670020 | +0.009341 [-0.007529, +0.026302] | -0.017272 [-0.034703, -0.000962] |
| overall_consistency | 0.232555 | 0.234363 | +0.001808 [-0.005027, +0.008694] | -0.014701 [-0.023318, -0.006229] |

Stage 3 trailed dense in background consistency, aesthetics, imaging quality and text–video consistency with paired intervals below zero. The other three intervals spanned zero. Dense and Stage 3 both registered less motion than the teacher, with paired intervals below zero for motion incidence. The dense-control and Stage 3 training histories differ in Q/K coverage, initialization and on-policy sampling, so this comparison measures the complete recipes.

Dense mean final latent relative L2 was 0.566338 across the 100 teacher pairs, using all final latent coordinates per pair. It achieved 14.202 dB mean PSNR across all 4,500 temporally aligned decoded frames, with zero identical frames. The audit verified 45 frames at 480×480 and 16 fps plus matching checksums for all 200 videos. Each dense video had 20 finite denoising steps and zero Q/K replacements. Rendering and all seven VBench dimensions used 3,978 GPU-seconds (1.11 GPU-hours) on GPU 3. The summary, direct paired intervals and agreement are in `distill-results/dense-paired/`. The lab videos are under `out/learned-center-wan-dense/videos/`.

### Resource stop and remaining control

The conservative allocation ledger totals 257,395 GPU-seconds (71.50 GPU-hours) through the dense paired benchmark. It charges worker elapsed time per allocated device, including retries, an hour reserved for early pilots and a timing margin. Calibration, references and trajectory capture account for 56,905 GPU-seconds. The total leaves about 0.50 approved GPU-hours. The lab scratch and pilot directories occupied 260.07 GB at the final check. The driver's deletion preflight projected roughly 49 GB of additional checkpoint and export headroom, so it stopped before allocating devices. A 5,000-update deletion run and 100-prompt VBench comparison are estimated to take about 14 GPU-hours based on the dense control's measured allocation. The claim that learned replacement beats deletion remains untested. Full EMA exports, validation curves and the completed paired benchmark outputs were preserved.

| Allocation | GPU-seconds charged |
|---|---:|
| Calibration, references and capture | 56,905 |
| Stage 1 training | 11,098 |
| Stage 2 training and early restart | 47,380 |
| Stage 3 training, restart and spare rollout workers | 71,648 |
| Dense training | 45,274 |
| Static benchmark | 4,787 |
| Stage 1 benchmark | 7,264 |
| Stage 3 benchmark | 4,488 |
| Dense benchmark | 3,978 |
| Stage 2 held-out videos and timeout grace | 721 |
| Stage 3 rollout parity probe | 72 |
| Early pilot allowance and scheduling margin | 3,780 |
| Total | 257,395 |

## Narrow and deletion controls

The follow-up brief calls for two 5,000-update Stage 2 controls, narrow first and deletion second, using teacher initialization, two ranks and the dense loss, optimizer and sampling recipe. Both controls use off-policy teacher states throughout. The Stage 3 student also trained on its own rollouts, so that comparison includes a training-history difference.

Narrow zeroes the 44 coordinates of the selected 22 rotary pairs per head in each token's Q and K projection output before native QK-RMSNorm and RoPE. It applies to all 20 denoising steps in 29 of 30 self-attention layers, with layer 0 exempt. Across the 12 heads, 528 of the 1,536 Q and K outputs are masked in each active layer. The effective removal is 1,623,072 projection weight and bias parameters per active layer, or 47,069,088 across 29 layers, from `2 × 528 × (1,536 + 1)` per layer. The implementation masks the rows' outputs and still stores the dense matrices.

Native Wan RMSNorm takes its mean square across all 1,536 features for one token and projection. Under narrow, only 1,008 live features contribute to the numerator, corresponding to 84 per head, while the denominator remains 1,536. Deletion zeroes the same pairs after RMSNorm, so all projected features participate in the norm statistic. Deletion also retains the first-step exemption. The difference between these controls includes both the norm statistic and first-step coverage.

On CPU, the narrow hooks matched the manual zero-padded RMSNorm calculation on a tiny Wan model. The same test checked exact zero Q/K projection outputs and exact zero gradients in the masked Q/K weight and bias rows, nonzero gradients in kept rows, the layer-0 exemption, and masking with no step or branch context. The existing learned-center tests also passed on the pinned lab runtime.

## Per-pair selection study

### A. Calibration and selection

The frozen bf16 Wan2.1 teacher replayed all 20 cached entering states for each of 64 calibration prompts, with both CFG branches. For each prompt, layer, denoising step, branch, head, and Q or K side, the collector measured normalized pre-RoPE pair means and energies across all 10,800 video tokens. The masks span all 14,400 layer–step–branch–head cells (30 layers × 20 steps × 2 branches × 12 heads). These statistics include layer 0 and step 1.

For a cell, side, and rotary pair, the pooled static mean `mu_f` is the average of the 64 prompt-level token means. For each prompt, `err_pool_p,f` is the average squared L2 distance of its token pairs to `mu_f`. The damage score is the prompt-average of this error multiplied by the partner side’s prompt-average pair energy. This product is the proxy for expected squared logit perturbation before common scaling. Fixed high selects the original 22 high-frequency pairs on each side. Separate selects the 22 lowest-damage pairs independently on Q and K in every cell. Shared selects the 22 lowest sums of Q and K damage and uses those pairs on both sides.

Cross-prompt stability compares the own-mean selection or damage ranking for one prompt and cell with the pooled selection or ranking in that cell. Jaccard compares the chosen 22-pair sets. Spearman ranks all 64 rotary pairs. The median and 10th percentile each summarize 921,600 prompt–cell comparisons per side.

| Side | Jaccard median | Jaccard p10 | Damage Spearman median | Damage Spearman p10 |
|---|---:|---:|---:|---:|
| Q | 0.760 | 0.571 | 0.927 | 0.823 |
| K | 0.760 | 0.571 | 0.931 | 0.825 |

The 0.76 median Jaccard on both sides is between the brief’s 0.5 input-dependence threshold and its 0.8 strong-stability threshold. Pooled damage captures much of the prompt-level ordering, and selected sets still vary across prompts.

Separate retains 40.76% of fixed-high Q selections and 42.27% of fixed-high K selections, counted across all cells. The composition below is the share of each variant and side’s 22 selections per cell, pooled over all cells. Axis entries show high, middle, and low frequency thirds in that order.

| Selection | Temporal H/M/L | Height H/M/L | Width H/M/L |
|---|---|---|---|
| Separate Q | 18.65% / 10.48% / 7.59% | 10.07% / 10.46% / 7.76% | 12.05% / 15.01% / 7.95% |
| Separate K | 19.37% / 10.24% / 7.34% | 10.45% / 10.37% / 7.61% | 12.45% / 14.72% / 7.45% |
| Shared, each side | 18.94% / 10.50% / 7.49% | 9.86% / 10.65% / 7.62% | 12.11% / 15.15% / 7.67% |

Each total damage sums the score over all selected Q and K pairs and all 14,400 cells, including the cells exempted at inference. For separate Q and K individually, this sum is 0.112% and 6.92% of the corresponding fixed-high sum, respectively.

| Selection | Summed Q + K damage | Fraction of fixed high |
|---|---:|---:|
| fixed_high | 590,506,173 | 100.000% |
| separate | 1,294,064 | 0.219% |
| shared | 1,361,080 | 0.230% |

The RMS concentration is `||mu_f|| / sqrt(mean_p E_p ||x_f||²)` for one pair and cell. Values below average those ratios over all pairs in the indicated subset and all cells, separately per side. Low damage in this study does not require a high concentration because partner energy also enters the score.

| Side | Separate selected | Separate unselected | Fixed high selected |
|---|---:|---:|---:|
| Q | 0.4816 | 0.5648 | 0.7244 |
| K | 0.4886 | 0.5362 | 0.7392 |

Separate Q and K share an average of 17.57 selected pairs per cell out of 22 on each side. The median, 10th percentile, and 90th percentile over all cells are 18, 14, and 20 pairs. A one-sided replacement still requires a dot product. Only the shared pairs are position-only, and no inference speed was measured.

Each layer ratio divides the total fixed-high Q + K damage by the total separate damage for that layer, summed over every denoising step, CFG branch, head, and selected pair. The intersection column averages the number of shared pairs per cell over the same steps, branches, and heads.

| Layer | Fixed high / separate damage | Q–K intersection, mean pairs |
|---|---:|---:|
| 0 | 1,134.973 | 21.07 |
| 1 | 23.638 | 19.01 |
| 2 | 30.711 | 18.40 |
| 3 | 7.757 | 16.62 |
| 4 | 7.308 | 17.00 |
| 5 | 36.647 | 16.97 |
| 6 | 3.208 | 17.70 |
| 7 | 5.745 | 17.31 |
| 8 | 13.769 | 17.06 |
| 9 | 9.562 | 16.14 |
| 10 | 2.281 | 16.95 |
| 11 | 5.299 | 17.56 |
| 12 | 6.416 | 17.44 |
| 13 | 2.872 | 18.37 |
| 14 | 4.289 | 17.89 |
| 15 | 3.252 | 18.99 |
| 16 | 2.742 | 17.16 |
| 17 | 7.771 | 17.15 |
| 18 | 2.376 | 18.01 |
| 19 | 5.783 | 16.95 |
| 20 | 17.170 | 17.71 |
| 21 | 12.088 | 16.49 |
| 22 | 2.376 | 17.36 |
| 23 | 4.421 | 18.06 |
| 24 | 3.456 | 16.62 |
| 25 | 2.618 | 17.43 |
| 26 | 6.124 | 16.12 |
| 27 | 101.188 | 17.21 |
| 28 | 60,783.206 | 18.56 |
| 29 | 9,932.877 | 17.86 |

Collection on one RTX 5090 used 986 GPU-seconds (0.274 GPU-hours), including model loading and all 64 prompt files. CPU selection verified that every Q and K mask picks exactly 22 pairs in every cell and that all 64 statistics records completed. Lab artifacts are under `out/distill-pair-stats/` and `out/distill-selection/` in the selection checkout. The tracked numeric artifact is `distill-results/selection/selection-report.json`.

### B. Training-free velocity screen

Each value is the mean over 32 held-out prompts and all 20 cached teacher entering states per prompt. The error at one state is the squared L2 velocity difference divided by the reference velocity energy, using every latent coordinate. For guided error, conditional and unconditional predictions are combined at CFG 5 before computing the ratio. The frozen bf16 teacher is the reference for the static-mean and deletion arms. The selected 22 pairs per side are replaced in normalized pre-RoPE Q and K for every token. This six-arm screen exempts layer 0 and step 1.

| Selection / mode | Conditional | Unconditional | Guided CFG 5 |
|---|---:|---:|---:|
| fixed_high/mean | 0.099291 | 0.098567 | 0.283408 |
| fixed_high/delete | 0.308276 | 0.307376 | 0.414549 |
| separate/mean | 0.010352 | 0.010486 | 0.036740 |
| separate/delete | 0.015286 | 0.014815 | 0.051309 |
| shared/mean | 0.006686 | 0.006748 | 0.032195 |
| shared/delete | 0.009840 | 0.009837 | 0.037419 |

The gate passes. Separate and shared pooled means reduce guided error by 7.71× and 8.80×, respectively, relative to fixed-high pooled means on the 640 states. Shared has the lowest guided error and is the first comparison arm for Stage C.

With a mask held fixed, mean minus deletion in guided error was −0.01457 for separate (95% CI −0.01858 to −0.01134) and −0.00522 for shared (−0.00649 to −0.00413). Each interval resamples 32 matched prompts 10,000 times with NumPy seed 20261109, retaining all 20 states of each sampled prompt. The pooled means improve velocity matching relative to zeros in this training-free screen. The result measures the frozen model on cached states.

The second replay applies the same hooks in all 30 layers and all 20 steps. The Stage 0 current-mean reference recomputes the 22 fixed-high pair means per prompt, state, layer, branch and head from the tokens in that forward. The precision reference compares the dense native bf16 and fp32 teacher velocities. All rows use the same 640 held-out prompt–state cells.

| All-layer, all-step condition | Conditional | Unconditional | Guided CFG 5 |
|---|---:|---:|---:|
| Native teacher bf16 vs. fp32 | 0.000141 | 0.000149 | 0.002731 |
| Fixed-high current-token means vs. native bf16 | 0.100820 | 0.101265 | 0.349937 |
| Fixed-high pooled means vs. native bf16 | 0.132161 | 0.132640 | 0.398662 |
| Separate pooled means vs. native bf16 | 0.010646 | 0.010964 | 0.040984 |

Stage B used 3,072 GPU-seconds for the six-arm screen and 1,288 GPU-seconds for the all-layer, all-step replay on GPU 4. A+B used 5,346 GPU-seconds (1.485 GPU-hours), including model loads. Both runs finished all required prompts and steps without lingering claims. The complete scores, per-step means, scopes and arm manifests are committed under `distill-results/selection/`.

### Shared 22-pair short training

The single shared-mask run used commit `1550308`, two RTX 5090 ranks on GPUs 4 and 2, and teacher initialization. It completed 500 off-policy updates with learning rate 3e-5, a 50-update warmup, no feature loss, and the shared 22-pair mask on both Q and K. Each update used two sampled teacher prompt–state items, one per rank and both CFG branches per item. The run completed in one launch with finite losses and saved both the final EMA export and a resumable optimizer checkpoint.

From launch through exit, the run took 2,026 seconds across the two GPUs (4,052 GPU-seconds, 1.126 GPU-hours). The median logged optimizer step took 3.566 seconds across the 50 logged steps from updates 10 through 500, excluding held-out validation and checkpoint writes. Peak PyTorch-allocated memory was 19.013 GiB on rank 0 and 19.010 GiB on rank 1.

The table reports guided relative squared velocity error over all latent coordinates at each cached teacher entering state, after combining both CFG branches at guidance 5. Every prompt contributes its 20 entering states. The short-run validation uses the live weights and the first 8 held-out prompts, d0064–d0071. The historical comparisons shown with 32 prompts average over the full validation set.

| Condition | Held-out prompts | Guided error |
|---|---:|---:|
| Shared short training, update 250 | 8 | 0.057455 |
| Shared short training, update 500 | 8 | 0.035533 |
| Shared pooled means, frozen teacher weights, matched subset | 8 | 0.029888 |
| Shared pooled means, frozen teacher weights, full screen | 32 | 0.032195 |
| Fixed-high Stage 2, update 5,000 | 32 | 0.027649 |
| Dense Stage 2, update 5,000 | 32 | 0.015493 |
| Native teacher bf16 vs. fp32 | 32 | 0.002731 |

The guided error fell by 38.2% from update 250 to 500 on the eight-prompt set. At update 500, error was +0.005645 above the training-free result on those same prompts and states. This run has no update-zero validation through the trainer, so the exact source of the gap remains unresolved.

To measure mean-vector drift from initialization to update 500, the calculation takes one Euclidean norm over the 44 selected coordinates per active layer, step, branch and head. It averages over 29 layers × 19 steps × 2 branches × 12 heads for each side, using the final live center checkpoint. The vectors are token-independent.

| Side | Initial selected norm | Update-500 selected norm | Mean change |
|---|---:|---:|---:|
| Q | 3.65641945 | 3.65638529 | -0.00003416 |
| K | 3.32870444 | 3.32869009 | -0.00001435 |

The logged whole-table norms include all 128 coordinates and the exempt cells. From the first logged update to update 500, Q moved from 12.271144 to 12.271112 and K from 12.714284 to 12.714255. The requested arguments, history and validation curves are in `distill-results/fast-shared22/`. The lab EMA and optimizer state are under `out/distill-fast-shared22/` in the selection checkout. The work stopped after reporting the requested short run.

### Training-free shared22 and shared32 video benchmark

Both shared arms used the frozen Wan2.1 teacher transformer and pooled static means, evaluated on the first 30 prompts in the benchmark manifest with one original seed per prompt. For each layer, denoising step, CFG branch, head and side, each pair mean pools all 10,800 video tokens within each of 64 calibration prompts and then averages the prompt means. Shared22 and shared32 rank pairs by summed Q and K damage, select 22 or 32 pairs respectively, and use the same selected pairs on both sides of each cell. The constant replacement is after QK-RMSNorm and before RoPE.

Each of the 60 generated videos passed the checksum, finite-latent and frame audit at 45 frames, 480×480 and 16 fps. Every new video recorded 2,204 interventions, matching 29 active self-attention layers × 19 active denoising steps × two CFG branches × two sides. Each video’s selection block records equal Q, K and Q–K intersection counts of 22 or 32 pairs per cell, averaged over the full mask tables. Layer 0 and denoising step 1 are exempt from replacement. The audited selection blocks are in `distill-results/static-shared30/selection-records.json`.

Each VBench mean uses one video per prompt and that dimension’s frame sampling specified above. Teacher, trained fixed-high (the final 5,000-update Stage 3 EMA), dense Stage 2, and static fixed-high scores are the audited per-prompt values from their existing 100-prompt benchmarks, restricted to the same first 30 prompt-and-seed pairs. Each difference is candidate minus reference and uses 10,000 paired prompt resamples with NumPy seed 20261109. Differences below about 0.02 in aesthetics are not resolvable with this 30-prompt screen.

| Dimension | Teacher | Trained fixed-high | Dense | Static fixed-high | Shared22 | Shared32 |
|---|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.961865 | 0.947142 | 0.962550 | 0.991295 | 0.952154 | 0.942813 |
| background_consistency | 0.976412 | 0.969787 | 0.976944 | 0.991485 | 0.971805 | 0.969995 |
| motion_smoothness | 0.982059 | 0.981423 | 0.982006 | 0.991790 | 0.985126 | 0.987243 |
| dynamic_degree | 0.333333 | 0.333333 | 0.200000 | 0.000000 | 0.200000 | 0.066667 |
| aesthetic_quality | 0.580512 | 0.517949 | 0.568441 | 0.267319 | 0.539369 | 0.398174 |
| imaging_quality | 0.662574 | 0.639815 | 0.658788 | 0.485986 | 0.615462 | 0.414821 |
| overall_consistency | 0.228075 | 0.225531 | 0.234424 | 0.030281 | 0.227472 | 0.172105 |

**Shared22 paired differences with 95% intervals**

| Dimension | vs. teacher | vs. trained fixed-high | vs. dense | vs. static fixed-high |
|---|---:|---:|---:|---:|
| subject_consistency | -0.00971 [-0.01991, -0.00029] | +0.00501 [-0.01067, +0.02100] | -0.01040 [-0.02455, +0.00182] | -0.03914 [-0.06232, -0.01959] |
| background_consistency | -0.00461 [-0.00998, +0.00014] | +0.00202 [-0.00556, +0.00943] | -0.00514 [-0.01210, +0.00146] | -0.01968 [-0.03003, -0.01083] |
| motion_smoothness | +0.00307 [-0.00110, +0.00824] | +0.00370 [-0.00103, +0.01012] | +0.00312 [-0.00202, +0.00959] | -0.00666 [-0.01046, -0.00334] |
| dynamic_degree | -0.13333 [-0.26667, -0.03333] | -0.13333 [-0.30000, +0.03333] | +0.00000 [-0.13333, +0.13333] | +0.20000 [+0.06667, +0.36667] |
| aesthetic_quality | -0.04114 [-0.07217, -0.01192] | +0.02142 [-0.01437, +0.05811] | -0.02907 [-0.06469, +0.00694] | +0.27205 [+0.24027, +0.30509] |
| imaging_quality | -0.04711 [-0.08084, -0.01284] | -0.02435 [-0.05412, +0.00433] | -0.04333 [-0.07579, -0.01304] | +0.12948 [+0.07235, +0.18142] |
| overall_consistency | -0.00060 [-0.01241, +0.01091] | +0.00194 [-0.01304, +0.01519] | -0.00695 [-0.02030, +0.00476] | +0.19719 [+0.17209, +0.22324] |

**Shared32 paired differences with 95% intervals**

| Dimension | vs. teacher | vs. trained fixed-high | vs. dense | vs. static fixed-high |
|---|---:|---:|---:|---:|
| subject_consistency | -0.01905 [-0.03084, -0.00666] | -0.00433 [-0.01853, +0.01144] | -0.01974 [-0.03218, -0.00751] | -0.04848 [-0.06555, -0.03287] |
| background_consistency | -0.00642 [-0.01217, -0.00068] | +0.00021 [-0.00599, +0.00708] | -0.00695 [-0.01279, -0.00098] | -0.02149 [-0.02857, -0.01503] |
| motion_smoothness | +0.00518 [-0.00117, +0.01310] | +0.00582 [-0.00056, +0.01355] | +0.00524 [-0.00144, +0.01309] | -0.00455 [-0.00749, -0.00192] |
| dynamic_degree | -0.26667 [-0.43333, -0.13333] | -0.26667 [-0.43333, -0.13333] | -0.13333 [-0.26667, -0.03333] | +0.06667 [+0.00000, +0.16667] |
| aesthetic_quality | -0.18234 [-0.21683, -0.14925] | -0.11978 [-0.14911, -0.08986] | -0.17027 [-0.20781, -0.13214] | +0.13085 [+0.09351, +0.17053] |
| imaging_quality | -0.24775 [-0.31082, -0.18428] | -0.22499 [-0.28047, -0.17028] | -0.24397 [-0.30584, -0.18302] | -0.07116 [-0.13585, -0.00303] |
| overall_consistency | -0.05597 [-0.07698, -0.03636] | -0.05343 [-0.07599, -0.03245] | -0.06232 [-0.08340, -0.04269] | +0.14182 [+0.11275, +0.16983] |

Shared22 had lower aesthetics, imaging quality, subject consistency and motion incidence than the teacher on the paired intervals. Its text–video consistency was 0.22747 versus 0.22807 for the teacher, with the difference interval spanning zero. All seven shared22 differences versus the trained fixed-high student had intervals spanning zero. Against dense, shared22 had lower imaging quality by 0.04333 (95% CI −0.07579 to −0.01304). Its point estimates for aesthetics and text–video consistency also trailed dense, with both intervals spanning zero.

Shared22 exceeded the static fixed-high arm on aesthetics, imaging quality, text–video consistency, and motion incidence. Shared32 had lower aesthetics, imaging quality, text–video consistency, and motion incidence than the teacher, the trained fixed-high student, and the dense control, with the paired intervals below zero.

For each candidate, final-latent L2 takes the Euclidean norm of the difference over all coordinates of the final latent for one prompt and seed. Relative L2 divides by the teacher latent norm before averaging over prompts. Mean PSNR averages the decoded-frame RGB PSNR over 1,350 matched frames per arm (30 prompts × 45 frames).

| Arm | Mean final latent L2 | Mean relative L2 | Mean frame PSNR (dB) | Identical frames |
|---|---:|---:|---:|---:|
| Shared22 | 467.270 | 0.525782 | 14.949 | 0 |
| Shared32 | 546.265 | 0.614498 | 13.653 | 0 |

Render workers on GPUs 4 and 2 hit their initial time limit. The run resumed from their 51 completed videos and preserved the interrupted files. The 60 videos charged 1,760 GPU-seconds, counting the partial attempts and their continuations. Scoring all seven dimensions on the 60 new videos with audited teacher scores reused charged 431 GPU-seconds on GPU 4. Total GPU use was 2,191 GPU-seconds (0.609 GPU-hours). The lab videos and full records are in `out/learned-center-wan-static-shared30/` in the selection checkout.

### Shared22 full fine-tuning

Shared22 started from the frozen Wan2.1 teacher checkpoint with the pooled static shared 22-pair means. The teacher supplied cached states and velocity targets, and the student transformer and means were trainable. No earlier student weights were used. The run used two FSDP ranks on RTX 5090 GPUs 4 and 5, with two prompt-and-state samples per rank per optimizer update, for four samples with both CFG branches each. It completed 2,500 off-policy updates at a base transformer learning rate of 1e-5 with 250 warmup updates, feature-loss weight 0.1 and EMA decay 0.999.

Training used 36,204 GPU-seconds (10.057 GPU-hours) over 18,102 elapsed seconds, including data loading, validations and EMA saves. The median was 6.902 seconds over the 250 logged optimizer updates from 10 through 2,500. Peak PyTorch allocations were 21.538 GiB on rank 0 and 21.535 GiB on rank 1. All optimizer updates completed in one invocation with finite losses.

Validation measures relative squared velocity error over all latent coordinates at 20 cached teacher states for each of the first eight held-out prompts, d0064–d0071. These 160 states are the same at every checkpoint. The guided column combines both CFG branches at guidance 5 before computing the error. Step zero uses the same training code path as the later live-weight checkpoints.

| Update | Conditional | Unconditional | Guided CFG 5 |
|---:|---:|---:|---:|
| 0 | 0.005625 | 0.005600 | 0.030147 |
| 250 | 0.004925 | 0.005003 | 0.032010 |
| 500 | 0.005119 | 0.005698 | 0.029735 |
| 750 | 0.005403 | 0.004786 | 0.035191 |
| 1,000 | 0.004765 | 0.005181 | 0.029822 |
| 1,250 | 0.003689 | 0.003504 | 0.026262 |
| 1,500 | 0.003692 | 0.003579 | 0.026239 |
| 1,750 | 0.003041 | 0.002914 | 0.024334 |
| 2,000 | 0.002788 | 0.002541 | 0.022687 |
| 2,250 | 0.002478 | 0.002279 | 0.021209 |
| 2,500 | 0.002378 | 0.002143 | 0.020696 |

Both gates passed. At update 1,000, guided error 0.029822 beat step zero (0.030147). At update 1,500, guided error 0.026239 beat update 500 (0.029735). The final live checkpoint reached 0.020696, a 31.3% reduction versus the comparable step-zero result.

Each validation row was appended and committed separately under `distill-results/shared22/`. Each 500-update save kept its EMA transformer and means to fit the storage budget without deleting checkpoints. These exports contain model parameters and masks. The final EMA was used for generation.

#### Matched 30-prompt video benchmark

The final shared22 EMA generated the first 30 manifest prompts with one original seed per prompt. For each fixed layer, denoising step, CFG branch and head, the model replaces the same 22 normalized pre-RoPE pairs in Q and K for every token. All 30 new videos recorded 2,204 calls (29 active layers × 19 active steps × two CFG branches × Q and K), finite latents at all 20 steps, and 45 frames at 480×480 and 16 fps. Per-video selection blocks and checksums are retained with the benchmark artifacts.

Each VBench score averages 30 per-video measurements. Aesthetics and imaging use all 45 frames and text–video consistency samples eight midpoint frames per video. The other dimensions use the same VBench frame sampling as above. The teacher, training-free shared22, trained fixed-high Stage 3 student and dense scores reuse their audited rows on the same prompts and seeds. Paired 95% percentile intervals resample the 30 matched prompts 10,000 times using NumPy seed 20261109.

| Dimension | Teacher | Static shared22 | Trained fixed-high | Dense | Tuned shared22 |
|---|---:|---:|---:|---:|---:|
| subject_consistency | 0.961865 | 0.952154 | 0.947142 | 0.962550 | 0.961931 |
| background_consistency | 0.976412 | 0.971805 | 0.969787 | 0.976944 | 0.976160 |
| motion_smoothness | 0.982059 | 0.985126 | 0.981423 | 0.982006 | 0.983828 |
| dynamic_degree | 0.333333 | 0.200000 | 0.333333 | 0.200000 | 0.266667 |
| aesthetic_quality | 0.580512 | 0.539369 | 0.517949 | 0.568441 | 0.569469 |
| imaging_quality | 0.662574 | 0.615462 | 0.639815 | 0.658788 | 0.682822 |
| overall_consistency | 0.228075 | 0.227472 | 0.225531 | 0.234424 | 0.230451 |

| Dimension | Shared22 − teacher | Shared22 − static shared22 | Shared22 − trained fixed-high | Shared22 − dense |
|---|---:|---:|---:|---:|
| subject_consistency | +0.00007 [-0.00699, +0.00817] | +0.00978 [-0.00266, +0.02199] | +0.01479 [+0.00424, +0.02570] | -0.00062 [-0.00681, +0.00573] |
| background_consistency | -0.00025 [-0.00507, +0.00413] | +0.00435 [-0.00326, +0.01184] | +0.00637 [+0.00302, +0.00959] | -0.00078 [-0.00384, +0.00228] |
| motion_smoothness | +0.00177 [-0.00039, +0.00429] | -0.00130 [-0.00501, +0.00197] | +0.00241 [-0.00097, +0.00659] | +0.00182 [-0.00153, +0.00622] |
| dynamic_degree | -0.06667 [-0.16667, +0.00000] | +0.06667 [-0.06667, +0.20000] | -0.06667 [-0.26667, +0.13333] | +0.06667 [+0.00000, +0.16667] |
| aesthetic_quality | -0.01104 [-0.03190, +0.00899] | +0.03010 [+0.00048, +0.06158] | +0.05152 [+0.02654, +0.07752] | +0.00103 [-0.02542, +0.02675] |
| imaging_quality | +0.02025 [-0.00823, +0.05365] | +0.06736 [+0.02923, +0.11079] | +0.04301 [+0.00240, +0.08456] | +0.02403 [-0.01071, +0.06153] |
| overall_consistency | +0.00238 [-0.00913, +0.01424] | +0.00298 [-0.00835, +0.01579] | +0.00492 [-0.00458, +0.01418] | -0.00397 [-0.01052, +0.00242] |

Fine-tuning lifted aesthetics by 0.03010 and imaging quality by 0.06736 versus static shared22, with both paired intervals above zero. The corresponding gains over trained fixed-high were 0.05152 and 0.04301. The text–video differences versus static shared22 and trained fixed-high had intervals spanning zero. The 30-prompt screen has limited resolution for aesthetic differences below about 0.02.

The dense-interval condition for requesting a full benchmark was met on both dimensions. These are percentile bootstrap intervals for the dense arm mean across 30 videos, with the same 10,000 resamples. The paired difference intervals versus dense also include zero.

| Dimension | Tuned shared22 mean | Dense mean [95% interval] | Shared22 − dense [paired 95% interval] |
|---|---:|---:|---:|
| aesthetic_quality | 0.569469 | 0.568441 [0.531954, 0.599849] | +0.00103 [-0.02542, +0.02675] |
| imaging_quality | 0.682822 | 0.658788 [0.607711, 0.703016] | +0.02403 [-0.01071, +0.06153] |

Same-seed agreement with the teacher gave mean final-latent L2 495.215 and relative L2 0.559703, computed over all final latent coordinates per pair and averaged across 30 prompts. The relative ratio divides by the teacher norm per pair. Mean decoded RGB PSNR was 14.122 dB over the 1,350 aligned frames, with zero identical frames.

The benchmark used 1,170 GPU-seconds on GPU 4, for 37,374 GPU-seconds (10.382 GPU-hours) including training. Training outputs, final EMA backup and benchmark records occupied about 17.51 GB, including linked teacher files in the benchmark count. All EMA milestones were retained. The lab checkpoint is `out/distill-shared22/student/`, and its checksum-verified durable copy is `out/distill-shared22-export/student/` in the selection checkout. Videos and source records are in `out/learned-center-wan-shared22/`. The curve and benchmark artifacts are under `distill-results/shared22/`.

The 100-prompt benchmark awaits approval. Narrow remains paused at update 4,000, and the deletion control is still pending.
