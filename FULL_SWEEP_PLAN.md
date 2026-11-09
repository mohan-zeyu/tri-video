# Complete Wan 3D RoPE experiment

Scope approved: four real prompts, 20 denoising steps, all 30 self-attention
layers, all 12 heads, each axis t/h/w. Fixed seed 0 and CFG scale 5.0.
45 video frames at 480x480 give a 12x30x30 grid: 10,800 video tokens.
Configuration and literal prompts: `configs/full_sweep_wan.json`.

## Exact meaning of each result

Every primary result belongs to one prompt, one denoising step (1-20, with
the scheduler timestep also recorded), one layer, one head, and one axis.
The 12 video-time positions are not denoising steps.

1. Capture normalized pre-RoPE Q/K from the verified conditional CFG pass.
   The stock Wan pipeline performs conditional and unconditional forwards
   separately; hooks must never silently replace one with the other.
2. Among the 2D frequency pairs belonging to the requested axis, select
   `argmax mean_tokens(||q_f||) * mean_tokens(||k_f||)` separately per head.
   Wan's 64 pairs split into 22 temporal, 21 height and 21 width pairs.
3. Plot ALL 10,800 token Q points and ALL 10,800 token K points in the
   selected 2D plane, before and after the model's positional rotation.
   The same pair, token IDs and display scale are used for pre/post plots.
4. For logit prediction, use the FULL 128D head, not just the selected pair
   or requested axis's pairs. Compute each head's Q and K centers in fp32
   across ALL 10,800 tokens, at this same prompt/step/layer.
5. For every query, compare its measured logit with the center prediction
   for every key on the same axis line (other two coordinates unchanged).
   Evaluate both signed offset directions and self. Thus each head has
   10,800 evaluated queries; no query or key subsampling.
6. Plot measured logit mean +/- population standard deviation at every
   signed offset, and the predicted mean over exactly the same pairs.
   Store counts, the origin-query curve, mean per-query Pearson r, its
   quantiles and the number of valid queries. Constant correlation rows
   are undefined rather than assigned a misleading zero.
7. At each prompt/step/layer/axis, plot the selected-pair MRL distribution
   over its 12 heads, separately for Q/K and pre/post. Retain all 64 pair
   MRL values and separate whole-head MRL values in numeric records.

This is a within-input center reconstruction experiment. Its centers are
not independently calibrated; no compression/quality/efficiency claim is
made. No denoising-step or prompt averaging enters the primary figures.

## Execution and artifacts

- Run from the repository root on a configured CUDA data host.
- GPUs 0 and 2 were initially idle. After the pilot, another scheduler
  occupied GPU 0; the launch guard refused to interfere. The full sweep
  therefore uses the then-idle GPUs 2 and 5. Assign whole prompt trajectories to GPUs;
  do not split a sequential denoising trajectory between devices.
- First test the math against explicit attention matrices, then pilot
  layers 0/15/29 at the first step of the real 20-step schedule. Check
  conditional embedding identity, token coverage and attention outputs.
- Then capture the four complete trajectories, two per GPU. Compute the
  metrics while that layer's Q/K are resident, release GPU references,
  and retain full bf16 Q/K on disk for audit and later reanalysis.
- Render in separate bounded CPU workers while capture continues. The
  initial eight-worker pool is supplemented by a 16-worker pool taking
  newest records first, after confirming 128 allowed CPUs and no CPU
  quota. Atomic, source-hash-checked render caches prevent corruption and
  skip figures another pool has completed. Total CPU render workers: 24.
  Each layer/axis gets an all-head sheet, one crop for each of its 12
  heads, and a separate across-head concentration distribution figure.
- Expected coverage: 80 prompt/step combinations, 2,400 layer captures,
  7,200 layer/axis sheets, 86,400 head figures, 7,200 distributions.
  Each head figure contains pre-RoPE scatter, post-RoPE scatter and full
  head logit reconstruction. Full raw Q/K require approximately 159 GB
  in decimal units, plus figures, metadata and final latents.
- Use atomic writes, status files, per-prompt completion manifests and
  explicit coverage verification. A launched job is not a completed run.
- NFS proved to be the bottleneck. Before they started, steps 16-20 of
  p01/p03 were routed through private `/dev/shm/tri-sweep-*` directories.
  Logical output paths and numerical code are unchanged. After all GPU
  and render writers and the coverage audit finish, the storage finalizer
  copies those steps to NFS, verifies every file with SHA-256, replaces
  only its own temporary symlinks with durable directories, then removes
  its private temporary tree. `storage-finalization.json` must say
  `complete` before the experiment is delivered.
- Final deliverables: the complete numeric/raw archive, every requested
  plot, an index for navigating them, validation/coverage records, and
  a report whose summaries state all aggregation dimensions.

## Correction to the previous trajectory report

The earlier `pipeline_probe.py` retained the final Wan CFG forward. In
Diffusers 0.42.1, this is the unconditional forward, not the conditional
forward claimed in the old report. Earlier Wan trajectory results must
therefore not be described as conditional-branch measurements. The new
sweep supersedes them and explicitly verifies the branch by matching
the actual transformer text input to the supplied positive embedding.
