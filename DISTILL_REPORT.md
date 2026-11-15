# Wan2.1 mean-vector distillation

Stage 0 captured all 3,641 teacher trajectories. Stages 1, 2 and 3 completed their 2,000, 5,000 and 5,000 updates, respectively. The paired benchmark is complete for the static means, Stage 1 and final Stage 3 arms against the frozen teacher, covering all 100 prompts and seven dimensions. Dense-control training is in progress. The deletion control has yet to run. The approved limits were four concurrent GPUs, 72 total GPU-hours and 300 GB of incremental disk. During the final student benchmark, the user reduced our active device set to GPUs 1, 3 and 6.

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
