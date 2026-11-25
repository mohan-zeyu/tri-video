# Plan and task: is a difference between two prompts caused by the text or by the noise seed?

**This version replaces the brief of commit `3cd891d`.** The owner narrowed it on 25 November 2026: "Just do it
on the layer 15 step 1. Don't run too many." If you started on the earlier version, stop, keep nothing of it in
the deliverables, and follow this file only.

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). **You write the small amount of code this needs** (the owner asked for that), run it, and hand
the results back. The research session reviews the code and the results. Nothing existing is changed.

## Why

At layer 15, step 1 the score maps of most pairs differ between prompts `d0000` and `d0006`
(`distill-results/score-attribution-cell/compare/d0000-L15-S01-conditional-H01--vs--d0006-L15-S01-conditional-H01.*`).
The two prompts differ in two things at once: the text and the initial noise (each prompt has its own seed).
This task separates the two, in that one setting only.

## Design

One setting: **layer 15, denoising step 1**, conditional branch, head 1 (heads count from 0), frozen bf16
teacher, every one of the 10,800 tokens as a query and as a key.

Two texts and two noises: `T0`, `N0` = text and initial noise of `d0000`; `T6`, `N6` = those of `d0006`.

| Cell | Text | Noise | Source |
|---|---|---|---|
| `T0N0` | `d0000` | `d0000` | exists: `distill-results/score-attribution-cell/d0000-L15-S01-conditional-H01` |
| `T6N6` | `d0006` | `d0006` | exists: `distill-results/score-attribution-cell/d0006-L15-S01-conditional-H01` |
| `T0N6` | `d0000` | `d0006` | **new**, id `t0000n0006` |
| `T6N0` | `d0006` | `d0000` | **new**, id `t0006n0000` |

At step 1 the entering state is the initial noise itself, so **no sampling run is needed**. A crossed input is
the stored step-1 state (`states.pt[0]`) of the noise prompt with the stored text embeddings (`embeddings.pt`)
of the text prompt, both read from the existing teacher trajectory cache. Only two new forward passes.

Six comparisons with the two measures of `scripts/compare_cell_pair_scores.py` (read its docstring; do not
redefine them):

| Type | The two comparisons |
|---|---|
| same noise, different text | `T0N0` vs `T6N0`; `T0N6` vs `T6N6` |
| same text, different noise | `T0N0` vs `T0N6`; `T6N0` vs `T6N6` |
| both differ | `T0N0` vs `T6N6` (exists already); `T0N6` vs `T6N0` |

## Standing rules of this study (binding)

- **No averaging over cells, comparisons or pairs.** Report single values, or distributions (minimum, 5%, 25%,
  median, 75%, 95%, maximum) or counts. The two comparisons of a type are shown side by side, never averaged.
- **Every table, figure title and paragraph states its scope** (`AGENTS.md`): which cells, which model, whether a
  number uses every query × every key or one query token, and what a quantile runs over.
- Figures follow `scripts/plot_cell_pair_scores.py` (import its helpers and colours; pairs on the horizontal axis,
  the pair-set strip at the bottom, the title block with the scope).
- Keep it simple. Use `cell_pair_scores.py` and `compare_cell_pair_scores.py` as they are. If you believe one of
  them or a definition must change, stop and ASK.

## What to build and run

1. **Crossed inputs.** A small script that writes, under `out/` in your task checkout, a trajectory folder for
   each new id in the layout `TrajectoryStore` reads (`scripts/train_learned_center.py`): the text prompt's
   `embeddings.pt`, the noise prompt's `meta.json` timesteps, and a `states.pt` whose step-1 state is the noise
   prompt's `states.pt[0]`. Only step 1 of these folders is meaningful: make the later states unusable (for
   example NaN) so a wrong `--step` cannot pass silently. Never write into the existing cache.
2. **Cells.** `cell_pair_scores.py capture --step 1 --layer 15 --branch conditional --head 1` with the three pair
   sets as in `DISTILL_SCORE_CELL_PROMPT.md`, then `scores` and `plot_cell_pair_scores.py`, for the two new ids,
   into `distill-results/score-attribution-seed/`.
3. **Comparisons.** `compare_cell_pair_scores.py` for the six comparisons, into
   `distill-results/score-attribution-seed/compare/`.
4. **Summary (new code).** One script that reads the six comparison CSV files and the four `pair-scores.csv` and
   writes, under `distill-results/score-attribution-seed/`:
   - `SEED-VS-PROMPT.md`: for each of the six comparisons, labelled by type: the Pearson correlation across the 64
     pairs between the two cells' centred shares; the largest difference of one pair's centred share; and the
     distribution over pairs (the seven quantile levels) of the pair's median-of-27-queries map correlation,
     given three times: over all 64 pairs, over the pairs whose centred share is above 1/64 in the comparison's
     first cell (state how many), and over the 43 low-frequency pairs t6–t21, h29–h42, w51–w63. Then one table
     with a row per pair and the six median-of-27-queries map correlations as columns.
   - `seed-vs-prompt.csv`: one row per comparison × pair with every number used.
   - `seed-vs-prompt.png`, one page: (a) per pair, the centred share in the four cells (text by colour, noise by
     marker shape, a thin line from the smallest to the largest of the four); (b) per pair, the
     median-of-27-queries map correlation of the six comparisons (one colour per type, the two comparisons of a
     type as two markers); the pair-set strip at the bottom.
5. **Tests.** A CPU test of the new code (tiny random model or synthetic cells), added to
   `scripts/test_pair_score_attribution.py` or a new test file; the test files must print `ALL TESTS OK`.

## Checks (report every number)

1. Each crossed folder: its step-1 state equals the noise prompt's stored `states.pt[0]` exactly, and its
   embeddings equal the text prompt's stored ones exactly. If not, **ASK**.
2. Each capture's attention-output validation (enforced by the script at 1e-3) and that the pair sets of the two
   new cells equal those of the two existing cells.

## Rules

- 10 GPU-minutes, one idle GPU; never kill or crowd a process you did not start. Write nothing outside your task
  checkout.
- **ASK** (hand off with `--status question`) if check 1 fails, a test fails, a command raises an error you
  cannot attribute to your own new code, or you need to change a definition or an existing script. Nothing else
  is a stop condition; do not judge or interpret the values.

## Deliverables

1. Code: the two scripts, the test, and the exact commands in the `handoff.md` paragraph.
2. Commit under `distill-results/score-attribution-seed/`: the two new cell folders whole (including `cell.npz`
   and `figures/`), the six comparison outputs, and the three summary files.
3. One paragraph in `handoff.md` §17: what ran, GPU time, the check results. No interpretation.
4. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
