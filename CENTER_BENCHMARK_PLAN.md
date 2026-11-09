# Wan Q/K center-replacement experiments

Study date 9 November 2026.

## Question

Does replacing normalized pre-RoPE video Q/K with their current-token centers degrade generated-video quality? Compare all pairs with the high-frequency thirds previously measured in this repository.

## Three paired arms

| Arm | Replaced pairs in each head |
|---|---|
| Baseline | None |
| All | All 64 pairs, all 128 coordinates |
| High | t pairs 0–7, h pairs 22–28, w pairs 43–49, totaling 22 pairs and 44 coordinates |

Apply replacement after the existing Q/K normalization and before the model's native 3D RoPE. Each coordinate center is an fp32 arithmetic mean over all 10,800 video tokens, cast to native bf16 for inference. Means remain separate for Q/K, each prompt, each of 20 denoising steps, each of 30 layers, each of 12 heads and each conditional/unconditional CFG branch. Q and K have separate centers. Every selected pair's two coordinates are replaced together. V and the attention computation continue through the existing pipeline.

The all-pair arm gives a position-only logit function within each head and forward pass, up to native floating-point rounding. In the high-pair arm, the selected-pair contribution depends on relative 3D position while the other 42 pairs retain token dependence. Centers adapt as the modified generation trajectory evolves.

This uses the arithmetic center definition from TriAttention §3.3. Current-input recomputation is the user's selected initial intervention. The paper derives its centers from calibration data and uses them for key-importance scoring and cache pruning. Interpret this experiment as a test of Q/K replacement in video diffusion.

## Generation

- Wan-AI/Wan2.1-T2V-1.3B-Diffusers, cached revision recorded per output.
- 45 frames, 480 × 480, 16 fps, 20 denoising steps, CFG 5, empty negative prompt.
- 100 distinct official VBench category prompts, sampled before generating outputs with Python seed 20261109.
- 13 prompts each for animal, architecture, food and human. 12 each for lifestyle, plant, scenery and vehicles.
- One deterministic seed per prompt, independent across prompts and shared across the three arms. All three arms start with matching generator state and the same embeddings.
- 300 videos in total. Separate GPU workers claim complete prompt groups, then generate all three arms. Category-interleaved processing ensures the early completed groups cover diverse content.
- Baseline is freshly generated with the same environment and decode settings.
- Float32 VAE with native tiling. bf16 text encoder and transformer. Record software, source/config hashes and model revision.

## Evaluation

Evaluate the six official VBench custom-input quality dimensions. The completed study additionally includes official ViCLIP text-video consistency.

1. Subject consistency, using DINO.
2. Background consistency, using CLIP.
3. Motion smoothness, using AMT.
4. Dynamic degree, using RAFT.
5. Aesthetic quality, using CLIP and the aesthetic predictor.
6. Imaging quality, using MUSIQ.

Report raw dimension scores, paired changes against baseline, and paired bootstrap confidence intervals over prompts. Each score's scope must state the frame aggregation and prompt count. Dynamic degree is a motion-incidence measure and should be interpreted jointly with appearance and smoothness. This is a 100-prompt, one-seed-per-prompt VBench custom-input study. Its scores describe that sample. The official full prompt/seed protocol and all-dimension aggregate score require their own evaluation.

Add text alignment only if the evaluator can be prepared and evaluated in time, documenting its exact implementation and sampling.

## Verification

- Synthetic checks independently verify all-token means and exact replacement coordinates, including batch/head separation and untouched unselected pairs.
- A timed full-resolution prompt pilot estimates generation throughput before full launch.
- Verify both CFG branches against the supplied text embeddings on every transformer forward.
- Verify 120 Q/K normalization-hook calls per denoising step, covering 30 layers × 2 sides × 2 branches. This is 2,400 calls per video.
- Check finite denoising latents, frame shape, video hashes, final latents and exact step coverage.
- For eight prespecified category-diverse prompts, retain pair diagnostics at steps 1, 10 and 20 across every layer/head and both branches. Each pair MRL uses all video tokens. Report diagnostics with this audit-subset scope.
- Validate real-module intervention outputs against independently constructed pre-RoPE mean replacements.
- Preserve completed arms atomically and resume missing work. Report failures explicitly.

## Files

- Frozen generation inputs in `configs/center_benchmark_wan.json`.
- Intervention in `scripts/center_replacement.py`.
- Generation worker in `scripts/run_center_benchmark.py`.
- Official VBench source pinned to commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` under `tmp/center-benchmark/VBench`.
- New experiment outputs will remain separate from the completed concentration sweep.

## Sources

- TriAttention, §3.3 and §4, https://arxiv.org/html/2604.04921v1
- Official VBench source and custom-input documentation, https://github.com/Vchitect/VBench
- Official VBench sampling protocol, https://github.com/Vchitect/VBench/tree/master/prompts
