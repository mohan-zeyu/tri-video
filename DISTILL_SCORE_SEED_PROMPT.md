# Plan and task: is a difference between two prompts caused by the text or by the noise seed?

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). Unlike the earlier briefs of this study, **you write the code for this one** (the owner asked for
that on 25 November 2026), run it, and hand the results back. The research session reviews the code and the
results afterwards. Nothing existing is changed or regenerated in place.

## Why

The single-cell study compared prompts `d0000` and `d0006` (`distill-results/score-attribution-cell/compare/`).
At layer 15 and at layer 29, step 20, the score maps of most pairs differ between the two prompts. But the two
prompts differ in two things at once: the text and the initial noise (each prompt has its own seed). The
comparison cannot say which of the two causes a difference. This task separates them with a 2 × 2 design.

## Design

Two texts and two noise seeds, taken from `configs/distill_prompts.json`:

| | Text | Noise seed |
|---|---|---|
| `T0` / `N0` | text of `d0000` ("a shark is swimming in the ocean, surrealism style") | seed of `d0000`, 1654720988 |
| `T6` / `N6` | text of `d0006` ("The video shows a woman standing on a city street. ...") | seed of `d0006`, 1686442679 |

Four teacher trajectories, all generated new in this task with exactly the protocol of
`scripts/dump_teacher_trajectories.py` (45 frames, 480 × 480, 20 UniPC steps, CFG 5, empty negative prompt, bf16,
CUDA generator seeded with the noise seed):

| Id | Text | Noise | Role |
|---|---|---|---|
| `t0000n0000` | `T0` | `N0` | control: must reproduce the stored `d0000` trajectory |
| `t0006n0006` | `T6` | `N6` | control: must reproduce the stored `d0006` trajectory |
| `t0000n0006` | `T0` | `N6` | crossed |
| `t0006n0000` | `T6` | `N0` | crossed |

Cells: layers 0, 15, 29 × denoising steps 1, 10, 20 (nine settings), conditional branch, head 1 (heads count from
0), frozen bf16 teacher replaying the trajectory's own stored entering state, as `scripts/cell_pair_scores.py
capture` does. 4 trajectories × 9 settings = 36 cells. Every one of the 10,800 tokens is a query and a key.

Comparisons, for each of the nine settings (six per setting, 54 in total), with the two measures of
`scripts/compare_cell_pair_scores.py` (read its docstring; do not redefine them):

| Type | The two comparisons |
|---|---|
| same noise, different text | `t0000n0000` vs `t0006n0000`; `t0000n0006` vs `t0006n0006` |
| same text, different noise | `t0000n0000` vs `t0000n0006`; `t0006n0000` vs `t0006n0006` |
| both differ | `t0000n0000` vs `t0006n0006`; `t0000n0006` vs `t0006n0000` |

- Centred share: each cell's `ctr_att_abs_share` per pair (share of the absolute attention-weighted score after
  removing each query's mean over keys; every query × every key of that cell).
- Map correlation: for ONE query token and one pair, the Pearson correlation over all 10,800 keys between the
  pair's score map in the two cells; given for the centre query and as minimum / median / maximum over the
  3 × 3 × 3 lattice of 27 query tokens.

## Standing rules of this study (binding)

- **No averaging over cells, prompts, comparisons or pairs.** Report single values, or distributions (minimum, 5%,
  25%, median, 75%, 95%, maximum) or counts. The two comparisons of a type are shown side by side, never averaged.
  The only pooling allowed is what the two measures already define inside one cell or one query.
- **Every table, figure title and paragraph states its scope** (`AGENTS.md`): which cell(s), which model, whether
  it uses every query × every key or one query token, and what a quantile runs over.
- Figures: follow `scripts/plot_cell_pair_scores.py` (import its helpers and colours; pairs on the horizontal axis,
  the pair-set strip at the bottom, title block with the scope). Keep each page readable; several pages are fine.
- Keep it simple. Reuse `cell_pair_scores.py` and `compare_cell_pair_scores.py` as they are. If you believe one of
  them or a definition must change, stop and ASK.

## What to build

1. **Trajectories.** `dump_teacher_trajectories.py` can produce them from a small config file with the four ids
   above (`text`, `seed`, `split: "val"`), written to a new directory under `out/` in your task checkout. Known
   pitfall: with no `calib` prompt in the config its `finalize_calibration` fails on an empty list after the
   trajectories are written; a guard for that case is the one change to an existing script that needs no ASK
   (`scripts/test_learned_center.py` must still pass). Never write into the existing trajectory cache.
2. **Cells.** `cell_pair_scores.py capture` and `scores` for the 36 cells, with the three pair sets as in
   `DISTILL_SCORE_CELL_PROMPT.md`, into `distill-results/score-attribution-seed/cells/`, and
   `plot_cell_pair_scores.py` for each.
3. **Comparisons.** `compare_cell_pair_scores.py` for the 54 comparisons, into
   `distill-results/score-attribution-seed/compare/`.
4. **Summary (new code).** One script that reads the 54 comparison CSV files and the 36 `pair-scores.csv` and
   writes, under `distill-results/score-attribution-seed/`:
   - `SEED-VS-PROMPT.md`: per setting and per comparison (all six rows, labelled by type): the Pearson correlation
     across the 64 pairs between the two cells' centred shares; the largest difference of one pair's centred share;
     and the distribution over pairs (the seven quantile levels) of the pair's median-of-27-queries map
     correlation, given three times: over all 64 pairs, over the pairs whose centred share is above 1/64 in the
     comparison's first cell (state how many), and over the 43 low-frequency pairs t6–t21, h29–h42, w51–w63.
   - `seed-vs-prompt.csv`: one row per setting × comparison × pair with every number used.
   - One figure per setting (nine pages): (a) per pair, the centred share in the four cells (text by colour, noise
     by marker shape, a thin line from the smallest to the largest of the four); (b) per pair, the
     median-of-27-queries map correlation of the six comparisons (one colour per type, the two comparisons of a
     type as two markers); the pair-set strip at the bottom.
5. **Tests.** CPU tests for the new code on the tiny random model or on synthetic cells, added to
   `scripts/test_pair_score_attribution.py` or a new test file; both test files must print `ALL TESTS OK`.

## Checks (report every number)

1. Controls against the stored trajectories: for `t0000n0000` vs stored `d0000` and `t0006n0006` vs stored
   `d0006`, the largest absolute difference of `states.pt` at each of the 20 steps, and of the embeddings.
   Step 1 (the initial noise) and the embeddings must be exactly equal; if not, **ASK**. Later steps: report
   whether they are exactly equal; if not, they are not a stop condition (see 4).
2. Crossed trajectories: the step-1 state of `t0000n0006` must equal the step-1 state of `t0006n0006` exactly,
   and that of `t0006n0000` must equal that of `t0000n0000` exactly; the positive embeddings must equal those
   of the control with the same text. If not, **ASK**.
3. Layer 0, step 1 is before the text enters and the state is the noise itself: for each noise, the captured
   Q/K (`cell.npz`) of the two texts must be exactly equal. If not, **ASK**.
4. Rerun floor: compare each control cell with the already committed cell of the same setting where one exists
   (`distill-results/score-attribution-cell/d0000-L..` and `d0006-L..`; six settings each): largest absolute
   difference of Q/K and of the centred shares, and run `compare_cell_pair_scores.py` on those pairs too
   (output under `compare-rerun/`). This says how much a pure rerun changes; it is reported, not gated.
5. Every capture's attention-output validation (`relative_max_err` ≤ 1e-3 is enforced by the script) and, for
   every setting, that the pair sets are identical in the four cells.

## Rules

- 45 GPU-minutes in total, idle GPUs only; never kill or crowd a process you did not start. 2 GB of disk under
  `out/`. Write nothing outside your task checkout.
- **ASK** (hand off with `--status question`) if: a check marked ASK fails, a test fails, you need to change a
  definition or an existing script beyond the one guard named above, or the budget would be exceeded. Nothing
  else is a stop condition; do not judge or interpret the values.
- Commit intermediate progress without trailers, as `AGENTS.md` says.

## Deliverables

1. Code: the config, the summary script, the tests, and a short driver (script or documented commands) that
   reproduces everything from the trajectories on.
2. Commit under `distill-results/score-attribution-seed/`: the three summary outputs and the nine figures; all 54
   comparison folders' `.md`, `.csv` and `.png`; `compare-rerun/`; per cell `pair-scores.csv`,
   `pair-scores.json`, `PAIR-SCORES.md` and `figures/4-one-query-maps.png`; and an `audit.json` with the
   check numbers, trajectory hashes, GPU time and the lab path of everything not committed. Do **not** commit
   `cell.npz` or the other per-cell figures (about 200 MB); keep them on the lab disk and give the path.
3. One paragraph in `handoff.md` §17: what ran, where, GPU time, the check numbers. No interpretation and no
   `DISTILL_REPORT.md` section; the research session examines the output.
4. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
