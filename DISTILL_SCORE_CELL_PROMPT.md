# Task: pair scores of one single cell (layer 0, step 1, head 1, prompt d0000)

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). One forward pass of the frozen teacher and one small table. A few GPU-minutes. It changes no
existing output and trains nothing.

## Why

The user finds the results so far too general: quantiles over hundreds of thousands of cells hide the detail,
and the percentage reported last (`attn_score_share`) lets positive and negative pair scores cancel. They asked
to start with one cell only and to see the score of every pair there, with the pairs of the selection options
marked.

## Definitions

- The cell: prompt `d0000` ("a shark is swimming in the ocean, surrealism style"), layer 0, denoising step 1
  (the first stored entering state, step index 0), conditional branch, head 1 (heads count from 0). Frozen bf16
  teacher. Every one of the 10,800 video tokens is a query and every one is a key. Nothing is pooled over prompts,
  layers, steps, branches or heads.
- Pair score: `c_f[i,j] = <qr_i[f], kr_j[f]> / sqrt(128)`, the part of the post-RoPE logit of query i and key j that
  pair f contributes. The columns keep positive and negative scores apart; they are defined at the top of
  `scripts/cell_pair_scores.py`.
- Pair sets marked in the table: what each selection file selects **in this cell** (union of its Q and K masks).
  No exemption is applied. Note for the record only: the trained shared22 and fixed-high students exempt layer 0
  and step 1, so they replace nothing in this cell; the shared32 student does.

## Inputs (read only)

- Teacher snapshot `WAN_SNAPSHOT` and the teacher trajectory cache used by the earlier attribution collections
  (`handoff.md` §17).
- `S22=/mnt/data/liuzeyu/tri-video-removal-f34fc19/out/distill-selection-v2/selection.pt`
  (SHA-256 `4bac7e317935a6b2aa0b4255f2b593cf65f93808c6e7e941bdca73663267d2b1`), variants `shared` and `fixed_high`.
- `S32=/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/selection/selection.pt`, variant `shared`.
- The existing teacher record `R=/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution/d0000.pt`.

## 0. Test

```sh
python scripts/test_pair_score_attribution.py     # CPU, about four minutes; must print ALL TESTS OK
```

## A. Capture and table (one idle GPU)

```sh
CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py capture --traj <trajectory cache> \
    --id d0000 --layer 0 --step 1 --branch conditional --head 1 \
    --set shared22 $S22 shared --set fixed_high $S22 fixed_high --set shared32 $S32 shared \
    --out distill-results/score-attribution-cell
CUDA_VISIBLE_DEVICES=<idle> python scripts/cell_pair_scores.py scores \
    distill-results/score-attribution-cell/d0000-L00-S01-conditional-H01
```

`capture` stops by itself if the attention output of layer 0 recomputed from the captured Q/K differs from the
model's by more than 1e-3 (relative). `scores` writes `pair-scores.csv`, `pair-scores.json` and `PAIR-SCORES.md`
next to `cell.npz`.

## B. One check against the existing record

```sh
python - <<'EOF'
import json, torch
r = torch.load("<R>", map_location="cpu", mmap=True)
rows = json.load(open("distill-results/score-attribution-cell/d0000-L00-S01-conditional-H01/pair-scores.json"))["rows"][:64]
adv = r["adv"][0, 0, 0, 1].double()
raw = adv + r["mean_contribution"][0, 0, 0, 1].double()
new_adv = torch.tensor([x["ctr_att_net"] for x in rows], dtype=torch.float64)
new_raw = torch.tensor([x["att_net"] for x in rows], dtype=torch.float64)
print("max |ctr_att_net - adv|", float((new_adv - adv).abs().max()), "max |att_net - (adv + mean_contribution)|", float((new_raw - raw).abs().max()))
EOF
```

Both differences must be at most 1e-3 (the new table must describe the same cell as the published study). Report
the two numbers.

## Rules

- 15 GPU-minutes, idle GPU only; never kill or crowd a process you did not start. Write nothing outside this
  checkout.
- **ASK** (hand off with `--status question`) only if: the test fails, a command raises an error, an input file is
  missing or the `S22` hash differs, or a difference in B exceeds 1e-3. Nothing else is a stop condition; do not
  judge the values in the table.
- Do not change a script or a definition.

## Deliverables

1. Commit the whole folder `distill-results/score-attribution-cell/d0000-L00-S01-conditional-H01/`: `cell.npz`
   (the captured Q/K of this one head, expected 5–12 MB; commit it so the research session can compute further
   variants without another run; if it is larger than 25 MB, leave it out and say so), `pair-scores.csv`,
   `pair-scores.json`, `PAIR-SCORES.md`.
2. One paragraph in `handoff.md` §17: what ran, GPU time, the validation error, the two numbers of B. No
   interpretation; the research session examines the output.
3. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
