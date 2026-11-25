# Task: attention-score share of the replaced pairs inside the trained students

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. You are `claude-lab`
(`AGENTS.md`). This task replays three finished student exports on stored teacher states and writes new records
and small tables. It changes no existing output and trains nothing.

## Why

So far the attention share of the replaced pairs was measured in the frozen teacher (median `adv_share` 0.10 for
shared22, 0.20 for shared32, 0.62 for fixed-high; `distill-results/score-attribution/`). The user wants the same
measured in the students that actually carry learned means on those pairs, and wants the plain percentage of the
attention score next to the row-centred shares.

## Definitions

- Cell: one prompt × layer × denoising step × CFG branch × head. Every query × every key of the full 128-dim
  post-RoPE logits (`scripts/pair_score_attribution.py`).
- Student replay: the student's transformer weights with its own `learned_center.pt` attached in the mode it
  was trained in (means), fed the stored **teacher** entering states, as `scripts/screen_selections.py --student`
  does. The Q/K analyzed are the ones the student attends with, after the replacement.
- Replaced pairs of a cell: the pairs the student replaces there on Q or K, with its exemptions applied
  (`replaced_q | replaced_k` in the record). Cells where nothing is replaced are left out of the tables.
- `attn_score_share`: the plain percentage. Of the raw logit the cell's queries give to the keys they attend
  (softmax-weighted, summed over queries), the part contributed by the replaced pairs. Undefined, and counted,
  in cells where the pairs' raw contributions nearly cancel.
- `adv_share`, `var_cov_share`, `var_own_share`, `top1_share`, `deleted_logit_variance`: as in the earlier briefs.
- **No averaging.** Every reported number is a quantile over prompt × cell entries. Do not add means over cells.

## Students (read only) and prompts

| Name | Export | Exempt |
|---|---|---|
| `shared22` | the tuned shared22 EMA, `out/distill-shared22-export/student/` in `/mnt/data/liuzeyu/tri-video-selection-67c101b` | layer 0, step 1 |
| `shared32` | `/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/train/student` | none |
| `fixedhigh` | the preserved Stage 3 EMA, `out/distill-controls-preserved/s3/student/` (the one the fixed-high swap used, `handoff.md` §16) | layer 0, step 1 |

Verify each export against the SHA-256 recorded for it in the repository before use
(`distill-results/shared32/training/final-ema.sha256`; the `selection-records.json` and `summary.json` under
`distill-results/shared22/full100/`; the fixed-high swap records under `distill-results/swap/`). If an export
cannot be found, has no recorded hash you can match, or its hash differs, stop and hand off with
`--status question`.

Prompts: the first 16 held-out prompts, `d0064` to `d0079`. The teacher records for them already exist in
`/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution-val`. Trajectory cache as in the
earlier collection (`handoff.md` §17).

## 0. Tests

```sh
python scripts/test_pair_score_attribution.py     # CPU, about three minutes; must print ALL TESTS OK
python scripts/test_learned_center.py             # unchanged code; must still pass
```

## A. One-prompt pilot (one idle GPU)

```sh
CUDA_VISIBLE_DEVICES=<idle> python scripts/collect_pair_score_attribution.py --traj <trajectory cache> \
    --prompts <manifest> --ids d0064 --student <shared22 export> --out out/distill-score-attribution-student-shared22
```

Check in the record:
- shapes as in the teacher records, `filled` all true, `student_mode` is `mean`;
- `attention_output_validation`: six entries, every `relative_max_err` ≤ 1e-3 (report the values);
- `replaced_pairs_constant_max_deviation` ≤ 1e-2 on both sides (the replaced pairs are the same vector at every
  token);
- `replaced_q` has no pair in layer 0 and none at step index 0, and 22 pairs in every other cell.

Record `seconds` and project B. **ASK** before B if the projection for all three students exceeds 4 GPU-hours.

## B. Collection (idle GPUs only, at most three at once, one student per GPU)

```sh
for each student NAME with export DIR:
  CUDA_VISIBLE_DEVICES=<idle> python scripts/collect_pair_score_attribution.py --traj <trajectory cache> \
      --prompts <manifest> --ids d0064 d0065 d0066 d0067 d0068 d0069 d0070 d0071 d0072 d0073 d0074 d0075 d0076 d0077 d0078 d0079 \
      --student DIR --out out/distill-score-attribution-student-NAME
```

Check per student: 16 records, no `.claim` left, every validation entry ≤ 1e-3, the constancy check as in A.
`shared32` must show 32 replaced pairs in every cell including layer 0 and step index 0; `fixedhigh` the pairs
0–7, 22–28 and 43–49 in every non-exempt cell.

## C. Tables (CPU)

```sh
T=/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution-val
IDS="d0064 d0065 d0066 d0067 d0068 d0069 d0070 d0071 d0072 d0073 d0074 d0075 d0076 d0077 d0078 d0079"
for NAME in shared22 shared32 fixedhigh; do
  S=out/distill-score-attribution-student-$NAME
  python scripts/selection_score_share.py --stats $S --record-masks replaced \
      --out distill-results/score-attribution-students --tag $NAME-student
  python scripts/selection_score_share.py --stats $T --ids $IDS --masks-from-record replaced $S/d0064.pt \
      --out distill-results/score-attribution-students --tag $NAME-teacher
done
```

The `-student` file is the share of the replaced pairs inside the student; the `-teacher` file is the share of
the same pairs in the same cells of the frozen teacher on the same 16 prompts.

Check: for each name the two files have the same `entries` and `pairs_per_active_cell`; `entries` is 16 × 13,224
for `shared22` and `fixedhigh` (29 layers × 19 steps × 2 branches × 12 heads) and 16 × 14,400 for `shared32`.
Report `attn_score_share_undefined_entries` for all six files. If it exceeds 5% of the entries in any file, say so
in the hand-off message; do not change the definition.

## Budget and rules

4 GPU-hours and 12 GB of disk under `out/` in this checkout. Idle GPUs only; never kill or crowd a process you
did not start. Write nothing outside this checkout. **ASK** (hand off with `--status question`) if a check
fails, a script raises an error, or before changing any definition or script.

## Deliverables

1. Commit `distill-results/score-attribution-students/` (six JSON, six Markdown) and a small audit file with the
   export hashes, the pilot seconds, GPU time, the validation maxima and the constancy maxima.
2. One paragraph in `handoff.md` §17: what ran, where the records are, GPU time. No `DISTILL_REPORT.md` section
   and no interpretation; the research session examines the output.
3. Hand off to `claude` with `scripts/handoff.py send --as claude-lab ... --ack <this brief's commit> --status done`.
