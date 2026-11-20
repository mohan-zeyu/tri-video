# Widened head (DISTILL_WIDE_PROMPT.md): 128 means under RoPE + 32 unrotated content pairs

Interim evidence for this study. The `DISTILL_REPORT.md` section and the `handoff.md` §16 paragraph are written
after the concurrent no-exemption shared22 study (its own supervisor appends to both files) has published its final
stage, to avoid making its automatic merge conflict.

Lab checkout `/mnt/data/liuzeyu/tri-video-wide-67ed977`, pinned to `1359c17` (branch tip 67ed977 plus the
`generate_learned_center.py` SyntaxError fix). Nothing exempt anywhere: all 30 layers and all 20 denoising steps.

## CPU tests

`cpu-tests.log`: `scripts/test_learned_center.py`, GPU runtime with CUDA hidden, at 67ed977: ALL TESTS OK
(including `test_wide`). The fix in 1359c17 touches only `generate_learned_center.py`, which the suite does not
import; that change was instead checked by loading the shared22 and split42 student checkpoints through the old
(construct + `load_state_dict`) and new (`LearnedCenter.from_checkpoint`) paths, mean and delete modes:
every parameter and buffer identical, same mode / wide / NoPE flags.

## A. Selection v4 (CPU, 52 s)

`select_pairs.py --stats out/distill-pair-stats --out out/distill-selection-v4 --pairs 22 --content-pairs 32
--include-layer0 --include-step1`. Statistics: the existing per-prompt files, pooled over the 64 calibration prompts.

Audit (`selection-audit.json`; per layer-step-branch-head cell, 30 x 20 x 2 x 12 = 14,400 cells, pair-level masks
over the 64 rotary pairs of each head):

- `shared` Q and K masks tensor-equal to the original `out/distill-selection/selection.pt`.
- `wide`: `wide: True`; means mask covers all 64 pairs in every cell, Q = K; exactly 32 content pairs in every cell
  (min = max = 32); in every cell the content pairs are the top 32 by pooled damage_q + damage_k.
- `static-wide/learned_center.pt` loads with `is_wide`, `content_pairs == 32`, `content_index` [30,20,2,12,64]
  consistent with the content mask, and all 30 layers / 20 steps active.

`selection-report.json` → `wide` (pooled over all 14,400 cells; fractions of all selected content pairs):

| Axis | high third | mid third | low third |
|---|---:|---:|---:|
| t | 0.1046 | 0.1002 | 0.1307 |
| h | 0.1227 | 0.0976 | 0.1298 |
| w | 0.1153 | 0.0691 | 0.1301 |

`fixed_high_in_content_fraction` = 0.4982: of the fixed-high pair slots (44 coordinates = 22 pairs per cell, summed over
all cells) 49.8% fall in the content set. The 22 split42 NoPE pairs are contained in the 32 content pairs in
every cell (intersection mean 22.0).

## B. Training-free screen (one GPU, 2,200.9 allocated GPU-seconds = 0.611 GPU-hours incl. loading)

`screen_selections.py --val-prompts 32 --no-exempt --selection out/distill-selection-v4/selection.pt --statistics
out/distill-selection-v4/pair-statistics.pt --variants shared wide --modes mean delete`.

Scope: relative squared velocity error of the whole latent vs the native bf16 teacher at the same teacher entering
state; equal mean over 32 held-out prompts (d0064-d0095) x 20 teacher entering states; conditional and
unconditional branches separately, and combined at CFG w = 5 for guided. Means are the static pooled calibration
means (64 prompts), not trained. No exempt layers or steps. Each Q/K replacement applies to all 10,800 video tokens
of the cell; the wide content channel uses the real post-QK-norm values of the 32 content pairs, unrotated.

| Arm | Conditional | Unconditional | Guided (w=5) |
|---|---:|---:|---:|
| shared/mean | 0.006842 | 0.006987 | 0.034431 |
| shared/delete | 0.010010 | 0.010069 | 0.039689 |
| wide/mean | 0.055053 | 0.052194 | 0.151673 |
| wide/delete | 0.449584 | 0.447210 | 0.571677 |

The shared rows are bit-identical to the no-exemption study's A screen (`distill-results/noexempt/screen`), a
reproducibility check of the same masks, means, prompts and states. All rows finite, so C proceeds. Training-free,
the widened head with pooled means is about 4.4x the shared22 guided error. Zeroing its position channel (content-only
attention on 32 unrotated pairs) is far worse at 0.572, above split42's training-free start of 0.44 (different
architecture and scope).

`supervisor.py` and `gpu-worker.py` are the exact lab runner sources for B/C (idle-GPU leases on GPUs 2 and 3 only;
GPUs 1/6 belong to the running no-exemption study, GPU 4 is excluded).
