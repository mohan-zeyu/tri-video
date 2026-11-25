# Task: attention-score share of the shared32 pairs in the teacher (CPU only)

Repository `mohan-zeyu/tri-video`, branch `claude/practical-lovelace-yham5v`, lab host 5090. This task reads
existing files and writes two small tables per split. No GPU, no model, no new collection, nothing in any
other checkout is modified.

## Why

The swap studies zeroed the trained means of a student's selected pairs at inference. For shared22 that cost
almost nothing; for the fixed high-frequency third it was catastrophic; for shared32 (nothing exempt) it cost
aesthetics −0.037 and imaging −0.042 against its own means arm (100 prompts, `DISTILL_REPORT.md`, "Shared32 with
nothing exempt"). The attribution study already measured how much of the teacher's attention score the
shared22 and fixed-high pairs carry (median `adv_share` 0.10 and 0.62 over 921,600 calibration cells). The user
asked for the same number for the shared32 pairs.

## Definitions

- Cell: one prompt × layer × denoising step × CFG branch × head of the frozen teacher. Its pair shares pool
  every query × every key of the full 128-dim post-RoPE logits (`scripts/pair_score_attribution.py`).
- A set's share in a cell: the cell's pair shares summed over the set's pairs there (union of its Q and K
  masks). `deleted_logit_variance` is the sum of the `pair_cov` block of the set.
- These are shares in the **teacher**, not in the shared32 student.
- **No averaging.** Every reported number is a quantile over prompt × cell entries (all of them, or those of
  one layer, step or branch). Do not add means over cells.

## Inputs (absolute paths, read only)

- Records: `/mnt/data/liuzeyu/tri-score-attribution-e8657af/out/distill-score-attribution` (64 calibration
  prompts) and `.../distill-score-attribution-val` (32 held-out prompts).
- shared22 masks: `/mnt/data/liuzeyu/tri-video-removal-f34fc19/out/distill-selection-v2/selection.pt`, variant
  `shared` (SHA-256 `4bac7e317935a6b2aa0b4255f2b593cf65f93808c6e7e941bdca73663267d2b1`, the file of the published
  overlay).
- shared32 masks: `/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/selection/selection.pt`, variant
  `shared`.

If a file is missing or the shared22 hash differs, stop and hand off with `--status question`.

## 0. Tests

```sh
python scripts/test_pair_score_attribution.py     # CPU, about two minutes; must print ALL TESTS OK
```

## A. Shares of shared22, shared32 and the ten pairs shared32 adds (minutes per split)

```sh
R=/mnt/data/liuzeyu/tri-score-attribution-e8657af/out
S22=/mnt/data/liuzeyu/tri-video-removal-f34fc19/out/distill-selection-v2/selection.pt
S32=/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32/selection/selection.pt
for split in "distill-score-attribution score-attribution" "distill-score-attribution-val score-attribution-val"; do
  set -- $split
  python scripts/selection_score_share.py --stats $R/$1 \
      --set shared22 $S22 shared --set shared32 $S32 shared \
      --difference shared32_minus_shared22 shared32 shared22 \
      --out distill-results/$2 --tag shared32
done
```

Outputs per split: `selection-shares-shared32.json` and `SELECTION-SHARES-shared32.md`.

Checks, from the JSON of each split:
- `entries` is 921,600 (calibration) and 460,800 (held-out);
- `pairs_per_cell` is 22 for `shared22`, 32 for `shared32` and 10 for `shared32_minus_shared22` at every
  quantile, and `set_sources.shared32_minus_shared22.pairs_of_b_outside_a` is 0 (shared22 is inside shared32 in
  every cell);
- the calibration `shared22` rows reproduce the published overlay
  (`distill-results/score-attribution/report.json`, `selection.shared`): `var_cov_share.all` equals
  `union_var_cov_share.all` and `deleted_logit_variance.all` equals `union_deleted_logit_variance.all` to 1e-6.

## Rules

CPU only, a few GB of RAM. Write nothing outside this checkout. **ASK** (hand off with `--status question`) if a
script raises an error, a check fails, or before changing any definition or script. No GPU budget is granted.

## Deliverables

1. Commit the two JSON and two Markdown files.
2. One paragraph in `handoff.md` §17: what ran, the entry counts and the three check results. No
   `DISTILL_REPORT.md` section and no interpretation; the research session examines the output.
3. Hand off to `claude` with `scripts/handoff.py send ... --ack <this brief's commit> --status done`.
