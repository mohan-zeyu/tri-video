# Task: the same single cell for a second prompt (d0006), with the one-query maps of all pairs

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). This repeats `DISTILL_SCORE_CELL_PROMPT.md` (brief `65786e3`) for one other prompt and adds the
figures. One forward pass of the frozen teacher, under a GPU-minute. It changes no existing output.

## Why

The user looked at the one-query score maps of the first cell (prompt `d0000`) and asked for the same maps for
every pair and for a different prompt.

## Definitions

- The cell: prompt `d0006` ("The video shows a woman standing on a city street. ..."), layer 0, denoising step 1
  (step index 0), conditional branch, head 1 (heads count from 0), frozen bf16 teacher. Every one of the 10,800
  tokens is a query and every one is a key; nothing is pooled over prompts, layers, steps, branches or heads.
- The maps (figures 4, 5, 6) are for ONE query token, the grid centre (frame 6, row 15, column 15): every pixel
  is one query-key score of one pair.
- Columns and pair sets as in the first brief. Inputs `S22`, `S32`, the trajectory cache and `WAN_SNAPSHOT` as
  in the first brief; the record for check B is `d0006.pt` in the same directory as `d0000.pt`.

## 0. Test

```sh
python scripts/test_pair_score_attribution.py     # must print ALL TESTS OK
```

## A. Capture, table, figures (one idle GPU)

```sh
CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py capture --traj <trajectory cache> \
    --id d0006 --layer 0 --step 1 --branch conditional --head 1 \
    --set shared22 $S22 shared --set fixed_high $S22 fixed_high --set shared32 $S32 shared \
    --out distill-results/score-attribution-cell
C=distill-results/score-attribution-cell/d0006-L00-S01-conditional-H01
CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py scores $C
python scripts/plot_cell_pair_scores.py $C          # CPU; writes $C/figures/ (10 PNG, 1 CSV)
```

## B. Check against the existing record

The Python snippet of the first brief, section B, with `d0006.pt` and the `d0006` folder. Both differences must
be at most 1e-3. Report the two numbers.

## Rules

- 15 GPU-minutes, idle GPU only; never kill or crowd a process you did not start. Write nothing outside this
  checkout. Do not touch the `d0000` folder.
- **ASK** (hand off with `--status question`) only if: the test fails, a command raises an error, an input file is
  missing or the `S22` hash differs, or a difference in B exceeds 1e-3. Nothing else is a stop condition; do not
  judge the values or restyle the figures.
- Do not change a script or a definition.

## Deliverables

1. Commit the whole folder `distill-results/score-attribution-cell/d0006-L00-S01-conditional-H01/` (`cell.npz`,
   `pair-scores.csv`, `pair-scores.json`, `PAIR-SCORES.md`, `figures/`).
2. One paragraph in `handoff.md` §17: what ran, GPU time, the validation error, the two numbers of B. No
   interpretation.
3. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
