# Per-pair attention-score attribution: research hand-off

Written 23 November 2026 by the research session on branch `claude/practical-lovelace-yham5v`, so that the
conversation can be compacted. It records what was asked, what was built, what was measured, the mistakes made
on the way, and, in most detail, the last topic: the percentage of the attention score carried by the replaced
pairs in the teacher and in the trained students. Section 6 is the part the owner is still thinking about.

Every number names its population, as `AGENTS.md` requires. "Cell" always means one prompt × layer × denoising
step × CFG branch × head, analyzed with every one of the 10,800 video tokens as a query and every one as a key,
on the full 128-dim post-RoPE logits of that head.

## 1. State in one paragraph

The question was: how much of each self-attention score does each rotary frequency pair contribute in Wan2.1
T2V-1.3B. That is measured for the frozen teacher on 96 prompts (64 calibration, 32 held-out), every layer, step,
branch and head. Follow-ups measured how a head's shares move across denoising steps, how much score the
replaced pair sets (shared22, shared32, fixed-high) carry in the teacher, and how much they carry inside the
three trained students. Everything requested so far is collected, analyzed, checked and acknowledged; nothing is
pending in the hand-off queue. Open items are decisions for the owner (section 8).

## 2. How to resume

- Branch `claude/practical-lovelace-yham5v`. It contains `claude/magical-goodall-dmq9i1` merged at `2daeacc`.
  Push only to this branch.
- Draft pull request `mohan-zeyu/tri-video#2` is the notification channel, not for merge. The research session
  subscribes to it (`subscribe_pr_activity`); a comment "Hand-off to claude" means: run
  `python3 scripts/handoff.py pending --as claude` and follow `.claude/skills/handoff/SKILL.md` (examine, report
  to the owner, ack, one push notification, stop; never chain a new brief without the owner's request).
- Agents: `claude` = research session; `claude-lab` = Claude Code on the 5090 lab host. It was addressed as
  `codex` up to brief `ffc819f`; the owner corrected that, and `AGENTS.md` and the skill were renamed at
  `36823c4`. Send briefs with `scripts/handoff.py send --as claude --to claude-lab --brief <file> ...`.
- The research session has no GPU and cannot reach the lab. Per-cell records exist only on the lab host; the
  repository holds quantile tables, figures and a few CSV files. Do not try to run the model here.
- Local environment for CPU tests: `.venv/bin/python` (torch 2.11 CPU, diffusers 0.42.1); gitignored.
  `python scripts/test_pair_score_attribution.py` must print `ALL TESTS OK` (about three minutes).

## 3. Owner's standing instructions (learned the hard way)

1. **Do not run the experiment in the research session.** Write the code well; the lab agent runs it.
2. **No averaging over cells.** Keep every prompt × layer × step × branch × head. Report distributions
   (minimum, 5%, 25%, median, 75%, 95%, maximum) or counts, never a mean over heads, layers, steps, branches or
   prompts. Inside one cell a share is a ratio of sums over that cell's queries and keys; that is allowed.
3. **State the scope of every number** (`AGENTS.md`).
4. **Keep it simple.** The owner was annoyed when a side quantity blocked the main result twice (section 7).
5. **Figures: one line per individual pair,** a few representative heads, several pages are fine. Summing pairs
   into groups or drawing all 12 heads in one panel "looks chaotic".
6. The lab agent is Claude Code; address it as `claude-lab`.
7. The owner prefers plain quantities ("just the percentage of their attention scores") over derived ones.

## 4. What was built

All in `scripts/`, all covered by `test_pair_score_attribution.py`.

| File | Role |
|---|---|
| `pair_score_attribution.py` | Field definitions and `attribute()` for one prompt/step/branch/layer, all heads |
| `collect_pair_score_attribution.py` | Lab collector: replays stored teacher entering states (both CFG branches), or a trained student with `--student`, or saved sweep captures with `--sweep-root`; one record per prompt with tensors `[30 layers, 20 steps, 2 branches, 12 heads, 64 pairs]` |
| `analyze_pair_score_attribution.py` | Quantile report over cells, selection overlay, atlas figures, `--cell` lookup; `shares()` derives per-cell shares from a record |
| `step_similarity_pair_score_attribution.py` | Step-to-step similarity of each head's 64 shares, quantiles over trajectories |
| `plot_step_trajectories_pair_score_attribution.py` | One page per layer: individual pairs against the step for four representative heads |
| `selection_score_share.py` | Share carried by named pair sets (selection files, set differences, or the pairs a student replaced) |

Briefs sent to the lab: `SCORE_ATTRIBUTION_PROMPT.md` (pasted by the owner, before the hand-off protocol),
`DISTILL_SCORE_STEPS_PROMPT.md` (`cb076e9`), `DISTILL_SCORE_SHARED32_PROMPT.md` (`ffc819f`),
`DISTILL_SCORE_STUDENTS_PROMPT.md` (`d8013ed`).

### 4.1 Definitions

`qr`, `kr`: outputs of `attn1.norm_q` / `attn1.norm_k` (normalized, pre-RoPE), rotated with the model's own RoPE
angles and rounded to bf16 as the model does. Pair contribution and logit of query i, key j in one head:

```
c_f[i,j] = <qr_i[f], kr_j[f]> / sqrt(128)        s[i,j] = sum over the 64 pairs f of c_f[i,j]
p[i,j]   = softmax over all 10,800 keys j of s[i,.]     (the head's attention weights, fp32)
```

Pair order: t pairs 0–21, h 22–42, w 43–63; inside an axis a larger index is a lower angular frequency.

Per cell and pair, stored raw in the records (shares are derived by `shares()`):

- `var_cov`: `sum_g G[f,g]`, `G[f,g] = mean_i Cov_j(c_f[i,.], c_g[i,.])`, exact over all query × key combinations
  from float64 second moments. `var_cov_share = var_cov / row_var` is signed and sums to 1.
- `var_own`: `G[f,f]`; `var_own_share` normalizes over the 64 pairs (non-negative).
- `pair_cov`: `G / sum G`, 64 × 64, float16. For a pair set S, `sum_{f,g in S} pair_cov` is the relative logit
  variance of the part S contributes (`deleted_logit_variance`).
- `adv`: `mean_i A_f[i]`, `A_f[i] = sum_j p[i,j] c_f[i,j] − mean_j c_f[i,j]`. `adv_share = adv / sum_f adv`: share
  of what lifts the attended keys above the average key. Row-centred: the per-query constant is removed.
- `top1`: the same with `p` replaced by the single arg-max key.
- `mean_contribution`: `mean over all i,j of c_f[i,j]` (uncentred).
- `var_share_quantiles`, `adv_share_quantiles`: 5/25/50/75/95% quantiles across the cell's queries of the
  per-query centred shares.
- Center model: Q/K replaced by their token means of the same cell, rotated at the real positions;
  `center_pair_r`, `center_logit_r`, `center_explained_share`.
- `attn_score_share` (added last, the "plain percentage"): see section 6.

## 5. Results so far

### 5.1 Collection (teacher)

64 calibration prompts `d0000`–`d0063` (921,600 cells) and 32 held-out prompts `d0064`–`d0095` (460,800 cells),
frozen bf16 teacher replaying its own stored entering states, both CFG branches, every query. 3.83 GPU-hours,
about 140 s per prompt, 20.3 GB of records. All 576 recomputed attention outputs (step 1, layers 0/15/29, both
branches, every prompt) matched the model bit for bit. Pair energies of `d0000` matched the selection study's
statistics exactly. Audits: `distill-results/score-attribution*/collection-audit.json`, `pilot-audit.json`.

### 5.2 Which pairs carry the score (teacher, calibration, quantiles over 921,600 cells)

- No pair dominates everywhere. Highest median `adv_share`: h26 0.043, w47 0.032, t2 0.0095; every temporal
  pair's median is below the uniform 1/64 = 0.0156.
- Single pairs dominate specific cells: 95th percentile of `adv_share` is 0.37 for t0, 0.24 for h22, 0.22 for
  w43. Largest pair of a cell (by `adv_share`): pair 22 in 21% of cells, pair 0 in 17%, pair 26 in 15%.
- Concentration (`var_own_share`): median cell needs 5 pairs for 50% and 30 for 90% (5%–95% of cells: 1–17 and
  3–47).
- Two kinds of layers (medians over the 30,720 cells of a layer): layers 0–2, 5, 20, 21, 27–29 need 6–13 pairs
  for 90% and the token-mean model reproduces their logits with r 0.84–0.98; layers 6, 10, 13–16, 18, 22–25 need
  37–44 pairs with r 0.46–0.59. Across layers those two quantities correlate at −0.94.
- Wavelength reading (assumes the standard RoPE base 10000, consistent with the recorded lowest frequency):
  t2, h26, w47 have wavelengths about 14.5 frames and 36 tokens against a 12 × 30 × 30 grid; pairs beyond about
  four grid lengths (t5+, h29+, w50+) each carry about half the uniform share and barely rotate.
- Amplitude `E‖q_f‖·E‖k_f‖` ranks pairs like `var_own_share` (per-cell Spearman median 0.95) but its largest
  pair is the largest-score pair in only 72% of cells. The three axis-dominant pairs of the old scatter figures
  carry a median 22% of the logit variance.
- Prompts and branches matter little: per-cell Spearman of a prompt's shares against the across-prompt median
  0.93 (median); conditional and unconditional medians nearly identical. Held-out medians within about 0.01.
- Negative result: a better ranking rule has little headroom. Shared22 carries 6.4% of own variance (median)
  against 4.7% for the best possible 22 pairs.
- Unexplained: a few cells in layers 28–29 have extreme logit row variance (maximum 754,406 against a median
  near 10), at every step. Not investigated.

Sources: `distill-results/score-attribution/report.json`, `REPORT.md`, `figures/`; held-out in
`score-attribution-val/`. `DISTILL_REPORT.md`, "Per-pair attention-score attribution".

### 5.3 Across denoising steps (teacher)

Step-level (profile over the 64 pairs of per-pair medians, medians over 46,080 cells per step, calibration):
neighbouring steps correlate at 0.99 or more; step 1 against 20 is 0.74 for `adv_share`.

Per head (the direct answer; one head's 64 `adv_share` values at two steps, Pearson across the 64 pairs;
quantiles over 46,080 trajectories = 64 prompts × 30 layers × 2 branches × 12 heads):

| Steps | 5% quantile | Median |
|---|---:|---:|
| 1 vs 2 | 0.585 | 0.919 |
| 2 vs 3 | 0.825 | 0.972 |
| 5 vs 6 | 0.991 | 0.998 |
| 10 vs 11 | 0.998 | 0.999 |
| 19 vs 20 | 0.969 | 0.996 |
| 1 vs 10 | 0.204 | 0.696 |
| 10 vs 20 | 0.667 | 0.929 |
| 1 vs 20 | 0.027 | 0.539 |

So the step-level figure overstated per-head stability: followed head by head, step 1 against 20 is 0.54, not
0.74. Steps 1–3 are a separate regime; from about step 5 a head's shares are nearly frozen and drift smoothly.
Median share mass that changes pair: 1% (10→11), 15% (1→2), 35% (1→20). By layer the 1-vs-20 median ranges from
0.92 (layer 2) to 0.31–0.36 (layers 15, 25, 26). Held-out medians within 0.01.

Sources: `step-similarity.json`, `STEP-SIMILARITY.md`, `figures/step-similarity-*.png`, and 90 pages under
`step-trajectories/` (prompts `d0000`, `d0001`, `d0064`, conditional branch, with CSV of every plotted number).
Known flaw: the second title line of those pages is cut off at the right edge; fixed in the script at `26ac7ac`,
pages not regenerated.

### 5.4 Score carried by the replaced pair sets, in the teacher

Calibration, 921,600 cells, a cell's pair shares summed over the set; median cell (5%–95% of cells). Row-centred
shares, no exemptions applied.

| | Shared22 | Shared32 | Fixed-high |
|---|---|---|---|
| `adv_share` | 0.10 (0.003–0.23) | 0.20 (0.006–0.39) | 0.62 (0.14–0.99) |
| `var_cov_share` | 0.08 (0.001–0.19) | 0.17 (0.003–0.35) | 0.61 (0.10–0.99) |
| `deleted_logit_variance` | 0.035 (0.0002–0.11) | 0.083 (0.0005–0.24) | 0.56 (0.06–0.99) |
| score-blind choice of that many pairs | 0.34 | 0.50 | 0.34 |

The ten pairs shared32 adds to shared22 carry a median `adv_share` of 0.088 on their own, almost as much as all
of shared22. Sources: `report.json` (`selection`), `selection-shares-shared32.json`,
`SELECTION-SHARES-shared32.md` in both result directories.

## 6. The last topic in detail: percentage of the attention score, teacher and students

### 6.1 What the owner asked

1. "Tell me just the attention shares of our replaced pairs." I answered with teacher numbers.
2. "No, I need the students' shares. Also, why don't get just the percentage of their attention scores." Two
   changes: measure inside the trained students, and report a plain percentage instead of the row-centred shares.
3. After the result, the owner restated the calculation to check it: per layer/step/branch/head, choose pairs
   by damage, compute the average share of every query attending to every key, then average into one number.
   That restatement is right about the pairs and wrong in two places (6.3). The owner said they are "still
   sticking on" this, so it is not settled.

### 6.2 Why the plain percentage was not reported at first

The raw logit of a head is a sum of 64 signed pair contributions. Softmax ignores any per-query constant, and
the total can be near zero or negative, so "pair's part ÷ total" can blow up or change sign. I therefore used
row-centred shares (`adv_share`, `var_cov_share`). The owner's request is still legitimate, and the records
already held what the plain percentage needs, so it was added as `attn_score_share`.

### 6.3 Exactly how `attn_score_share` is computed

For ONE cell (one prompt, layer, step, branch, head) and a set S of pairs:

```
                 sum over queries i, sum over keys j of   p[i,j] * (sum over f in S of c_f[i,j])
percentage(S) = ───────────────────────────────────────────────────────────────────────────────
                 sum over queries i, sum over keys j of   p[i,j] * s[i,j]
```

- `p[i,j]` is the attention weight query i gives key j (softmax over all 10,800 keys). **Keys are weighted by
  attention, not counted equally**: the keys a query attends to dominate, the ones it ignores barely count.
- **It is a ratio of sums, not an average of per-query percentages.** Numerator and denominator are each summed
  over all 10,800 queries, then divided once.
- `s` is the raw logit (not row-centred), so the per-query constant that softmax ignores is included.
- In code (`analyze_pair_score_attribution.py`, `_shares`): `raw_f = adv_f + mean_contribution_f`, which equals
  `mean over queries of sum_j p[i,j] c_f[i,j]`; `attn_score_share_f = raw_f / sum_g raw_g`; the set's percentage
  is the sum of `attn_score_share_f` over f in S. It is undefined (NaN, counted, left out of quantiles) in cells
  where `sum_g raw_g <= 0.01 * sum_g |raw_g|`, i.e. where the pairs' raw contributions nearly cancel.
- **Which pairs S**: for shared22 and shared32, each layer × step × branch × head has its own set, the pairs with
  the lowest `damage_q + damage_k`; damage was pooled over the 64 calibration prompts, so the set is the same for
  every prompt. Fixed-high is the same 22 pairs everywhere (t 0–7, h 22–28, w 43–49), not chosen by damage.
  S is the union of the Q-side and K-side masks (identical here). For the shared22 and fixed-high students,
  layer 0 and step 1 are exempt (nothing replaced) and those cells are left out.
- **Across cells there is no averaging.** Each prompt × cell gives one percentage; the reported number is the
  median of those, with the 5%–95% range.

How it differs from the owner's restatement: (a) not an equal-weight average over keys, but attention-weighted;
(b) not an average over queries of each query's percentage, but one ratio of sums per cell; (c) the final number
is the median over cells, not their mean.

What does not exist and would need new work: a per-query percentage (the records keep per-query quantiles only
for the centred shares), an equal-weight-over-keys percentage, a share of absolute contributions
(`sum |c_f|`), and anything measured on a student's own generation trajectory.

### 6.4 How the students were measured

`collect_pair_score_attribution.py --student DIR` loads the student's transformer weights and its own
`learned_center.pt` in the mode it was trained in (means), attaches the learned-center hooks first, then the
analysis hooks, so the Q/K analyzed are the ones the student attends with after the selected pairs were replaced
by its learned constants. The student is fed the stored **teacher** entering states at each step (as
`screen_selections.py --student` does), both CFG branches in one batch. Each record also stores `replaced_q` /
`replaced_k` (the pairs replaced per cell, exemptions applied) and a check that those pairs are the same vector
at every token.

Scope: first 16 held-out prompts `d0064`–`d0079`; shared22 and fixed-high 211,584 entries each
(16 × 29 layers × 19 steps × 2 branches × 12 heads), shared32 230,400 (nothing exempt). The teacher column uses
the same pairs and the same cells on the same 16 prompts. 1.88 GPU-hours.

Students: shared22 = tuned shared22 EMA (`out/distill-shared22-export/student/` in
`/mnt/data/liuzeyu/tri-video-selection-67c101b`); shared32 =
`/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/train/student`; fixed-high = preserved Stage 3 EMA
(`out/distill-controls-preserved/s3/student/`). Hashes are in
`distill-results/score-attribution-students/audit.json`.

### 6.5 Result

`attn_score_share` of the replaced pairs, median cell (5%–95% of cells):

| Replaced pairs | In the teacher | In the trained student |
|---|---|---|
| Shared22 (22 pairs) | 10.3% (0.3–21.3) | 3.1% (−1.6–13.6) |
| Shared32 (32 pairs) | 19.5% (0.1–37.6) | 6.9% (−4.1–24.2) |
| Fixed-high (22 pairs) | 57.4% (12.5–128) | 51.7% (6.4–107) |

Row-centred `adv_share`, same scope, teacher → student medians: shared22 0.105 → 0.014, shared32 0.199 → 0.027,
fixed-high 0.600 → 0.558. `var_cov_share`: 0.085 → 0.008, 0.168 → 0.016, 0.592 → 0.540.
`deleted_logit_variance`: 0.037 → 0.003, 0.085 → 0.008, 0.529 → 0.470.

By layer (median `attn_score_share` over the cells of a layer): shared22 student under 1% in layers 1, 2, 5, 17,
20, 28 and 6–9% in layers 13, 18, 22–26; fixed-high student 85–95% in layers 1, 2, 5, 20, 28, 29 and 20–30% in
layers 13–15, 22, 23. By step: shared22 student 2.5% at step 2 rising to 4.8% at step 20; shared32 student
5.9%–9.2%; fixed-high student 50%–57%.

Reading: the shared22 and shared32 students moved their attention off the replaced pairs (about a third of the
teacher share in the plain percentage, about an eighth in the centred share); the fixed-high student still gets
about half of its attention score from position-only constants. That matches the drop experiment
(`DISTILL_REPORT.md`): zeroing trained means changed aesthetics by −0.005 for shared22 (interval spans zero),
−0.037 for shared32, −0.163 for fixed-high; imaging −0.009, −0.042, −0.284 (100 prompts each).

Caveats to carry forward:
- Percentages above 100% or below zero are real: other pairs contribute negatively in those cells. The extremes
  are wild (minimum about −5, maximum about 40); medians and middle ranges are sound. Undefined in at most 0.04%
  of entries per file.
- The students saw teacher latents, not their own trajectories.
- 16 prompts, not 96. The shared32 teacher medians on these 16 match the 32-prompt result (0.199 in both).
- The attention-output recomputation check runs at step 1, which shared22 and fixed-high exempt, so it covers
  replaced pairs only for shared32 (exact there). Constancy of the replaced pairs was confirmed for all three
  (largest deviation 6e-5).
- The shared22 export is matched in the repository only by its `learned_center.pt` hash; its weights match a
  checksum file beside the export on the lab disk.
- Split42 and the no-exemption shared22 student were not measured.

Sources: `distill-results/score-attribution-students/selection-shares-<name>-{student,teacher}.json`, the
matching `SELECTION-SHARES-*.md` (with per-layer and per-step tables) and `audit.json`.

## 7. Mistakes and corrections (do not repeat)

1. I started a CPU pilot in the research container and averaged over heads and layers. The owner stopped both.
   The pilot output and downloaded weights were deleted.
2. The first lab analysis stopped on an overflow in `center_pair_r` (a side quantity): for the lowest-frequency
   pairs the center-model contribution is constant over keys, its stored variance was a tiny negative number, and
   a 1e-300 floor turned the ratio into 1e140. Fixed at `c4754e8`: undefined below a relative variance floor of
   1e-10, counted, excluded from quantiles.
3. My resume instruction then had an ambiguous stop rule ("pairs other than the lowest-frequency pairs"), and the
   lab agent correctly stopped on pair 20. Lesson: stop rules must name the exact condition; do not gate the main
   result on a side quantity.
4. I showed a jagged layout-check figure made from random synthetic data; the owner read it as a result. Label
   synthetic figures unmistakably or do not show them.
5. I reported a step-level similarity (0.74) as if it described heads; the per-head figure is 0.54.
6. I addressed two briefs to `codex`; the lab agent is Claude Code (`claude-lab`).
7. I gave teacher shares when the owner meant the students'.

## 8. Open items (owner decides; none is approved)

1. Whether the `attn_score_share` definition in 6.3 is the percentage the owner wants, or one of the variants
   listed there.
2. Regenerate the 90 step-trajectory pages with the title fix (CPU, minutes).
3. Measure split42 or the no-exemption shared22 student, or students on their own trajectories.
4. Directions discussed, with my opinion: sparse attention guided by pair properties fits the data better than
   step distillation. A single replay could decide both: attention-mass recall under position-only key
   selection per layer/head/step, and cross-step stability of attention outputs. Not briefed.
5. The owner said pair cutting is not to be pursued further; the per-layer budget idea is therefore parked.

## 9. Where things are

Repository: results under `distill-results/score-attribution/`, `score-attribution-val/`,
`score-attribution-students/`; report section in `DISTILL_REPORT.md`; lab log in `handoff.md` §17.

Lab host (5090), read-only inputs for any further analysis:
- Teacher records: `/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution` (64) and
  `.../distill-score-attribution-val` (32).
- Student records: `/mnt/data/liuzeyu/tri-score-steps-cb076e9/out/distill-score-attribution-student-{shared22,shared32,fixedhigh}`
  (16 each, 9.0 GiB).
- Selection masks: shared22 `/mnt/data/liuzeyu/tri-video-removal-f34fc19/out/distill-selection-v2/selection.pt`
  (variant `shared`); shared32 `/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/selection/selection.pt`
  (variant `shared`).

Commits on this branch: `e8657af` code and first brief; `c327f75` collection audit; `c4754e8` overflow fix;
`a1fa2a9` teacher analysis published; `f76714d`, `d8f21ff` step scripts; `2daeacc` merge and hand-off protocol;
`cb076e9` → `4d6be83` step figures and similarity (ack `3e3390d`); `26ac7ac` title fix; `ffc819f` → `57306b9`
shared32 in the teacher (ack `8f1623d`); `36823c4` lab agent renamed; `d8013ed` → `a6d8c9b` students
(ack `5af211f`).
