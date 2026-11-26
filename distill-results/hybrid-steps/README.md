# Teacher on the early steps: shared32 hybrid benchmark

Part 2 of `DISTILL_PROFILE_PROMPT.md`. This was inference only, with no training. One sampler ran all 20 UniPC steps and handed the denoiser over between steps, so the scheduler kept its multistep history. On the steps listed below, the frozen bf16 teacher ran with `LearnedCenter` off. On all other steps, the shared32 final EMA student ran with its trained means.

- **Code.** `scripts/generate_learned_center.py --teacher-steps ...`, with the CPU test `scripts/test_hybrid_steps.py`.
- **Driver.** `scripts/hybrid_steps.py`.
- **Tables.** Produced by `scripts/report_hybrid_steps.py`.
- **Run.** 25–26 November 2026, in lab checkout `/mnt/data/liuzeyu/tri-video-profile-8b2502c`, with everything written under `out/study-hybrid-steps/`.

Step numbers are 1-based: step 1 is scheduler index 0.

| Arm | Steps 1-3 | Steps 4-20 | Teacher forwards per video | Replacement calls per video |
|---|---|---|---:|---:|
| teacher (frozen baseline, reused) | teacher | teacher | 40 | 0 |
| `s32_means` (reused from `distill-results/shared32/full100/`) | student | student | 0 | 2,400 |
| `s32_teacher3` | teacher | student | 6 | 2,040 |
| `s32_student3` | student | teacher | 34 | 360 |

**Frozen protocol.**
- The 100 evaluation prompts, with one fixed seed each (the original seeds).
- 20 UniPC steps, shift 3, CFG 5, empty negative prompt.
- 45 frames at 480×480, 16 fps.

**Inputs verified before rendering (`ema-verify.json`).** The SHA-256 of `learned_center.pt`, the transformer config and the bf16 weights each match the audited hashes in `distill-results/shared32/`.

## Parity (`parity.json`, `parity/`)

Two prompts were run through the new path: v000 (seed 2026110900) and v013 (seed 2026110913). The tolerance is the one the encoder-parity probes use: `torch.equal` on the final latent, meaning all coordinates of the step-20 state.

| Check | Reference | v000 | v013 | Per video |
|---|---|---|---|---|
| teacher on all 20 steps | frozen teacher baseline | final latents equal, relative L2 0.0, MP4 SHA-256 identical | same | 40 teacher forwards, 0 replacement calls |
| teacher on no step | `s32_means` videos | final latents equal, MP4 SHA-256 identical | same | 0 teacher forwards, 2,400 replacement calls |

## Video audit (`*/selection-records.json`)

Both arms pass the audit on all 100 records:

| Arm | Teacher steps | Teacher forwards | Replacement calls |
|---|---|---:|---:|
| `s32_teacher3` | [1, 2, 3] | 6 | 2,040 |
| `s32_student3` | [4 … 20] | 34 | 360 |

For both arms, latents were finite at all 20 steps and every video has 45 frames of 480×480×3.

## VBench (`*/summary.json`, `*/per-prompt-results.csv`, `*/paired-comparisons.json`)

**How the values are computed**
- Each value is an equal mean over the 100 prompts of VBench's within-video aggregation, with one video per prompt:
  - Aesthetic and imaging quality are scored over all 45 frames; imaging is divided by 100.
  - Text-video consistency (`overall_consistency`) uses eight midpoint frames.
  - Dynamic degree is the fraction of the 100 videos judged dynamic.
- Paired differences are the mean [pointwise 95%] over 10,000 paired prompt resamples (NumPy seed 20261109).
- There is one seed per prompt, so there is no uncertainty across seeds.
- The teacher and `s32_means` videos are the existing ones; they were not regenerated.

| Dimension | Teacher | s32_means | s32_teacher3 | s32_student3 |
|---|---:|---:|---:|---:|
| Aesthetics | 0.5886 | 0.5692 | 0.5751 | 0.5929 |
| Imaging | 0.6607 | 0.6769 | 0.6586 | 0.6808 |
| Text-video | 0.2326 | 0.2314 | 0.2281 | 0.2355 |
| Subject | 0.9640 | 0.9652 | 0.9616 | 0.9688 |
| Background | 0.9765 | 0.9758 | 0.9753 | 0.9785 |
| Motion smoothness | 0.9845 | 0.9863 | 0.9848 | 0.9861 |
| Dynamic degree | 0.3300 | 0.2100 | 0.3000 | 0.2600 |

Dynamic videos out of 100: teacher 33, s32_means 21, s32_teacher3 30, s32_student3 26.

| Dimension | teacher3 − teacher | teacher3 − s32_means | student3 − teacher | student3 − s32_means | teacher3 − student3 |
|---|---|---|---|---|---|
| Aesthetics | -0.0135 [-0.0227, -0.0042] | +0.0059 [-0.0060, +0.0179] | +0.0044 [-0.0086, +0.0171] | +0.0237 [+0.0125, +0.0348] | -0.0179 [-0.0306, -0.0052] |
| Imaging | -0.0021 [-0.0119, +0.0077] | -0.0183 [-0.0363, +0.0002] | +0.0201 [+0.0027, +0.0369] | +0.0039 [-0.0062, +0.0135] | -0.0222 [-0.0409, -0.0035] |
| Text-video | -0.0044 [-0.0091, -0.0001] | -0.0032 [-0.0089, +0.0022] | +0.0029 [-0.0039, +0.0099] | +0.0041 [-0.0007, +0.0089] | -0.0074 [-0.0142, -0.0008] |
| Subject | -0.0024 [-0.0046, -0.0003] | -0.0036 [-0.0114, +0.0043] | +0.0048 [-0.0022, +0.0117] | +0.0036 [+0.0016, +0.0058] | -0.0072 [-0.0143, -0.0001] |
| Background | -0.0012 [-0.0031, +0.0008] | -0.0005 [-0.0045, +0.0032] | +0.0020 [-0.0017, +0.0057] | +0.0027 [+0.0007, +0.0047] | -0.0032 [-0.0071, +0.0005] |
| Motion smoothness | +0.0004 [-0.0001, +0.0009] | -0.0015 [-0.0029, -0.0001] | +0.0016 [+0.0003, +0.0031] | -0.0002 [-0.0008, +0.0003] | -0.0013 [-0.0027, +0.0001] |
| Dynamic degree | -0.0300 [-0.0800, +0.0100] | +0.0900 [+0.0200, +0.1700] | -0.0700 [-0.1500, +0.0100] | +0.0500 [+0.0000, +0.1000] | +0.0400 [-0.0300, +0.1100] |

## Same-seed agreement with the teacher (`*/agreement/`)

For each prompt, the arm's video is compared with the teacher's video for the same prompt and seed:
- **Final-latent relative L2:** the step-20 state over all coordinates, as one ratio per prompt, averaged over the 100 prompts.
- **Frame PSNR:** computed over all 4,500 decoded frames (100 videos × 45). No frame was identical to the teacher's.

| Arm | Final-latent relative L2 | Frame PSNR (dB) |
|---|---:|---:|
| s32_means | 0.5818 | 13.910 |
| s32_teacher3 | 0.2923 | 19.219 |
| s32_student3 | 0.5895 | 13.824 |

## Reading

**Row of the brief's outcome table:** "In between".

For `s32_teacher3`:
- Dynamic degree is inside the teacher's paired interval: -0.03 [-0.08, +0.01], 30 against 33 videos.
- Agreement is clearly better than `s32_means`: relative L2 is halved, 0.5818 → 0.2923, and PSNR is +5.3 dB.
- Aesthetics is not inside the teacher's interval: -0.0135 [-0.0227, -0.0042].
- Text-video and subject consistency also sit slightly below the teacher, with intervals that just exclude zero.

The converse arm says how much each group of steps explains:

- **The trajectory away from the teacher comes from the student's steps 1-3.** With the teacher on steps 4-20, agreement is no better than the full student: 0.5895 against 0.5818. The student's three early steps alone produce a video as far from the teacher's same-seed video as running the student on all 20 steps.
- **The aesthetics deficit comes from the student's steps 4-20.** With the teacher on 4-20, aesthetics is at teacher level: +0.0044 [-0.0086, +0.0171], and +0.0237 [+0.0125, +0.0348] above `s32_means`. With the student on 4-20, it stays below the teacher. Teacher on steps 1-3 recovers only +0.0059 [-0.0060, +0.0179] of the -0.0194 gap.
- **Both step groups contribute to dynamic degree.** The teacher on steps 1-3 restores 9 of the 12 missing dynamic videos; the teacher on steps 4-20 restores 5 of them.
- **Imaging rises with the student on steps 1-3.** It is +0.0201 [+0.0027, +0.0369] above the teacher in `s32_student3`, and `s32_means` was +0.0162 [-0.0021, +0.0337]. With the teacher on steps 1-3, imaging returns to teacher level.

Handing steps 1-3 to the teacher therefore does not close the aesthetics gap of this student. This result does not support the first row's conclusion that a student trained only on steps 4-20 is worth trying. This arm already runs the current student on steps 4-20, and its aesthetics stays below the teacher. Whether a student trained only on those steps would do better there was not tested; it is not in scope and needs approval.

## GPU time (`ledger.tsv`, `budget.json`)

Allocated wall time × GPUs of every GPU job, including model loading. CPU work is excluded.

| Job | GPU-seconds |
|---|---:|
| parity, both checks | 155.656 |
| s32_teacher3 render (two cards) | 1,793.493 |
| s32_teacher3, seven scorers | 500.665 |
| s32_student3 render | 1,723.803 |
| s32_student3, seven scorers | 521.645 |
| **Part 2 total** | **4,695.262 (1.3042 GPU-hours)** |

The s32_student3 render ran on one card. No second card passed the idle rule while the first worker was rendering, so the first worker took all 100 prompts (1,698.8 s). The second worker started afterwards, found no prompts left and exited after loading (25.0 s).
