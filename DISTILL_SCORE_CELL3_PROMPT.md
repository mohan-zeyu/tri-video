# Task: the single-cell pair scores at a middle step and at a middle layer, for both prompts

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). Same procedure as `DISTILL_SCORE_CELL2_PROMPT.md` (brief `a2a1152`), for four more cells. Four
forward passes of the frozen teacher, about two GPU-minutes. It changes no existing output.

## Why

The first two cells (prompts `d0000` and `d0006`, layer 0, step 1, head 1) are before the text enters the model.
The user asked for the same thing at a middle denoising step with the other settings unchanged, and at a middle
layer with the other settings unchanged.

## The four cells

Frozen bf16 teacher, conditional branch, head 1 (heads count from 0), every one of the 10,800 tokens as a query
and as a key; nothing is pooled over prompts, layers, steps, branches or heads.

| Prompt | Layer | Denoising step (1-based) | Folder under `distill-results/score-attribution-cell/` |
|---|---:|---:|---|
| `d0000` | 0 | 10 | `d0000-L00-S10-conditional-H01` |
| `d0006` | 0 | 10 | `d0006-L00-S10-conditional-H01` |
| `d0000` | 15 | 1 | `d0000-L15-S01-conditional-H01` |
| `d0006` | 15 | 1 | `d0006-L15-S01-conditional-H01` |

Inputs `S22`, `S32`, the trajectory cache, `WAN_SNAPSHOT` and the record directory `R` as in the first brief
(`DISTILL_SCORE_CELL_PROMPT.md`).

## 0. Test

```sh
python scripts/test_pair_score_attribution.py     # must print ALL TESTS OK
```

## A. Capture, table, figures, check (one idle GPU)

```sh
for CELL in "d0000 0 10" "d0006 0 10" "d0000 15 1" "d0006 15 1"; do
  set -- $CELL
  CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py capture --traj <trajectory cache> \
      --id $1 --layer $2 --step $3 --branch conditional --head 1 \
      --set shared22 $S22 shared --set fixed_high $S22 fixed_high --set shared32 $S32 shared \
      --out distill-results/score-attribution-cell
  C=$(printf 'distill-results/score-attribution-cell/%s-L%02d-S%02d-conditional-H01' $1 $2 $3)
  CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py scores $C
  python scripts/plot_cell_pair_scores.py $C          # CPU; writes $C/figures/ (10 PNG, 1 CSV)
  python - "$C" "<R>/$1.pt" $2 $3 <<'PY'
import json, sys, torch
folder, record, layer, step = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
r = torch.load(record, map_location="cpu", mmap=True)
rows = json.load(open(folder + "/pair-scores.json"))["rows"][:64]
adv = r["adv"][layer, step - 1, 0, 1].double()
raw = adv + r["mean_contribution"][layer, step - 1, 0, 1].double()
new_adv = torch.tensor([x["ctr_att_net"] for x in rows], dtype=torch.float64)
new_raw = torch.tensor([x["att_net"] for x in rows], dtype=torch.float64)
print(folder, "max |ctr_att_net - adv|", float((new_adv - adv).abs().max()),
      "max |att_net - (adv + mean_contribution)|", float((new_raw - raw).abs().max()))
PY
done
```

For every cell both differences must be at most 1e-3. Report the eight numbers and the four validation errors.

## Rules

- 20 GPU-minutes, idle GPU only; never kill or crowd a process you did not start. Write nothing outside this
  checkout. Do not touch the existing cell folders.
- **ASK** (hand off with `--status question`) only if: the test fails, a command raises an error, an input file is
  missing or the `S22` hash differs, or a difference in the check exceeds 1e-3. Nothing else is a stop condition;
  do not judge the values or restyle the figures.
- Do not change a script or a definition.

## Deliverables

1. Commit the four folders whole (`cell.npz`, `pair-scores.csv`, `pair-scores.json`, `PAIR-SCORES.md`,
   `figures/`).
2. One paragraph in `handoff.md` §17: what ran, GPU time, the validation errors, the check numbers. No
   interpretation.
3. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
