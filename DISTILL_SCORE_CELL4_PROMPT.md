# Task: the single-cell pair scores at the last layer and the last step, for both prompts

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). Same procedure as `DISTILL_SCORE_CELL3_PROMPT.md` (brief `f4fcb44`), for six more cells. Six
forward passes of the frozen teacher, about two GPU-minutes. It changes no existing output.

## Why

After the middle step and the middle layer, the user asked for the last layer and the last step. Both readings
are covered: each changed alone against the first cell (layer 0, step 1), and both together.

## The six cells

Frozen bf16 teacher, conditional branch, head 1 (heads count from 0), every one of the 10,800 tokens as a query
and as a key; nothing is pooled over prompts, layers, steps, branches or heads.

| Prompts | Layer | Denoising step (1-based) | Folder suffix |
|---|---:|---:|---|
| `d0000`, `d0006` | 29 | 1 | `-L29-S01-conditional-H01` |
| `d0000`, `d0006` | 0 | 20 | `-L00-S20-conditional-H01` |
| `d0000`, `d0006` | 29 | 20 | `-L29-S20-conditional-H01` |

Inputs `S22`, `S32`, the trajectory cache, `WAN_SNAPSHOT` and the record directory `R` as in the first brief
(`DISTILL_SCORE_CELL_PROMPT.md`).

## 0. Test

```sh
python scripts/test_pair_score_attribution.py     # must print ALL TESTS OK
```

## A. Capture, table, figures, check (one idle GPU)

The loop of `DISTILL_SCORE_CELL3_PROMPT.md`, section A, unchanged except for its first line:

```sh
for CELL in "d0000 29 1" "d0006 29 1" "d0000 0 20" "d0006 0 20" "d0000 29 20" "d0006 29 20"; do
```

For every cell both differences of the check must be at most 1e-3. Report the twelve numbers and the six
validation errors.

## Rules

- 20 GPU-minutes, idle GPU only; never kill or crowd a process you did not start. Write nothing outside this
  checkout. Do not touch the existing cell folders or `compare/`.
- **ASK** (hand off with `--status question`) only if: the test fails, a command raises an error, an input file is
  missing or the `S22` hash differs, or a difference in the check exceeds 1e-3. Nothing else is a stop condition;
  do not judge the values or restyle the figures.
- Do not change a script or a definition.

## Deliverables

1. Commit the six folders whole (`cell.npz`, `pair-scores.csv`, `pair-scores.json`, `PAIR-SCORES.md`,
   `figures/`).
2. One paragraph in `handoff.md` §17: what ran, GPU time, the validation errors, the check numbers. No
   interpretation.
3. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
