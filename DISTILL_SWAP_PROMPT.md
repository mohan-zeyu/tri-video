# Task: fine-tune-then-drop. Do the retained means do anything in the trained students? (training-free swaps, no training)

Repository `mohan-zeyu/tri-video`, branch `claude/magical-goodall-dmq9i1`, lab host with RTX 5090s (32 GB).

## Question

The deletion control answered "drop, then fine-tune": a student trained with zeros on the shared22 pairs reaches the
same quality as the student trained with means (`DISTILL_REPORT.md`, "Shared22 deletion control"). This task answers
the other order, "fine-tune, then drop": take the students as trained and swap what sits on the selected pairs at
inference only. Four arms, all on the shared22 masks with layer 0 and step 1 exempt, no weights changed:

| arm | weights | on the selected pairs | what it is |
|---|---|---|---|
| `s22_means` | tuned shared22 EMA (`out/distill-shared22/student`) | its trained means (`--center-mode mean`) | as benchmarked, reference |
| `s22_zeros` | tuned shared22 EMA | zeros (`--center-mode delete`) | **fine-tune with means, then drop them** |
| `del_zeros` | deletion EMA (`out/distill-shared22-delete/student`) | zeros (`--center-mode delete`) | as benchmarked, reference |
| `del_means` | deletion EMA | the calibration means stored in its center file (`--center-mode mean`) | the symmetric swap: insert means into a student trained without them |

Both center files carry the same masks (verified SHA-256-identical in the deletion mask audit). The deletion student's
`learned_center.pt` holds the pooled calibration means, which the trainer initialises in every mode and which delete
mode never uses; `s22`'s means moved by 0.004% in norm during training, so the two mean sets differ negligibly. Record
`mu_norm_q/k` from both files in the report (the screen writes them to `meta.json`).

Read first: `AGENTS.md`, `DISTILL_AGENT_PROMPT.md` §3 (ground rules), the deletion-control section of
`DISTILL_REPORT.md`, and the new `--student` option of `scripts/screen_selections.py` (this commit).
`python scripts/test_learned_center.py` must pass in the GPU runtime first (CPU, about ten minutes).

## A. Velocity-error screen (about 20 GPU-minutes; two jobs, shardable)

```sh
# tuned shared22 student: as trained (mean) and with its means dropped (delete)
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 --student out/distill-shared22/student \
  --modes mean delete --out out/distill-swap/screen-s22
# deletion student: as trained (delete) and with means inserted (mean)
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 --student out/distill-shared22-delete/student \
  --modes delete mean --out out/distill-swap/screen-del
```
Scope: 32 held-out prompts × 20 teacher entering states × both branches, relative squared velocity error vs the native
bf16 **teacher** (the reference is always the teacher, also when a student is screened), guided w=5. The student arms
on the first 8 prompts should agree with the trainers' final validation rows (0.020696 for s22, 0.020530 for the
deletion student, 8 prompts); state the 8-prompt and 32-prompt numbers separately.

## B. 100-prompt videos for the two swapped arms (about 2 GPU-hours)

`s22_means` and `del_zeros` already exist as the audited `shared22` and `shared22_delete` benchmark arms; do not
regenerate them. Generate the two swaps with `generate_learned_center.py` (add `s22_zeros` and `del_means` to the
benchmark config's `modes`; expected intervention calls 2,204 each; the record's `center_mode` must say `delete` for
`s22_zeros` and `mean` for `del_means`):

```sh
python scripts/generate_learned_center.py --out out/learned-center-wan-swap --arm s22_zeros --student out/distill-shared22/student --center-mode delete
python scripts/generate_learned_center.py --out out/learned-center-wan-swap --arm del_means --student out/distill-shared22-delete/student --center-mode mean
```
Then the seven VBench dimensions on the 100 videos of each swap arm, paired comparisons (10,000 paired prompt
resamples, NumPy seed 20261109) of `s22_zeros` vs `s22_means`, `del_means` vs `del_zeros`, and each swap vs the
teacher, plus same-seed latent relative L2 / PSNR vs the teacher, in the layout of `distill-results/shared22-delete/full100/`.

## How to read it (write this reasoning in the report)

- `s22_zeros` ≈ `s22_means` and `del_means` ≈ `del_zeros`: the means are inert in both trained students; drop them.
- `s22_zeros` < `s22_means` but `del_zeros` ≈ `s22_means` (already known): the student trained with means came to
  rely on them, yet they are not needed, since a student trained without them is as good. The decision is then
  "choose zeros before fine-tuning", not "keep the means".
- `del_means` < `del_zeros`: inserting means into a zero-trained student hurts; consistent with the point above.
Any other pattern: report it and **ASK**.

## Budget and rules

About 2.5 GPU-hours total, idle GPUs only, at most two concurrently; do not disturb running jobs (the pooled C2
fine-tune, the shared22 extension). Preserve all earlier outputs; never regenerate the baseline or the existing
`shared22` / `shared22_delete` arms. **ASK** before anything outside this list. Commit once, with a message listing what
ran. No model identifiers in committed files; no speed claims.

## Deliverables

1. `distill-results/swap/` with both screen `complete.json`/`summary.md`/`meta.json`, and `full100/` for the two swap
   arms (per-prompt CSV, paired comparisons, agreement, selection records, summary).
2. A new `DISTILL_REPORT.md` section "Fine-tune, then drop: swapping means and zeros in the trained students": the
   four-arm screen table, the 100-prompt seven-dimension table and paired intervals, the reading above with the
   numbers; every number with its scope (pairs vs whole Q/K, token average, step average, prompts, branches,
   exemptions).
3. `handoff.md` §16 updated.
4. A final message: what ran, what failed and how it was fixed, the headline numbers, open issues.

## C. The same swap on the fixed-high student (added 18 November; training-free, about 1.3 GPU-hours)

The original study's student (fixed high third, 22 pairs t 0–7 / h 22–28 / w 43–49 on both Q and K, learned means,
Stage 3 EMA after 5,000 updates) is preserved at `out/distill-controls-preserved/s3/student/` (transformer/ + a
44-wide legacy `learned_center.pt`, which the current module loads and converts; verify by its recorded SHA-256 in
`out/distill-controls-preserved/` and by `uses_default_masks == True`, exemptions layer 0 and step 1). Swap its means
for zeros at inference; nothing is trained.

| arm | weights | on the fixed high pairs | what it is |
|---|---|---|---|
| `fh_means` | fixed-high Stage 3 EMA | its trained means (`--center-mode mean`) | as benchmarked (`student` arm, `distill-results/student-paired/`), reference |
| `fh_zeros` | fixed-high Stage 3 EMA | zeros (`--center-mode delete`) | **fine-tune with means, then drop them**, on the fixed set |

C1. Screen (about 20 GPU-minutes):
```sh
CUDA_VISIBLE_DEVICES=<g> python scripts/screen_selections.py --val-prompts 32 --student out/distill-controls-preserved/s3/student \
  --modes mean delete --out out/distill-swap/screen-fh
```
Scope as in A (32 held-out prompts × 20 teacher entering states, both branches, guided w=5, reference = native bf16
teacher). The `student/mean` row should be near the Stage 3 final validation of 0.028238 guided on the same 32
prompts (that row used the live weights; this one uses the EMA export, so a small difference is expected; state both).
Put the two rows in one table with the four rows of A and with the training-free teacher rows for the fixed set
(static means 0.283, zeros 0.415) so the ratio zeros/means can be read before and after training for both pair sets.

C2. 100-prompt videos for `fh_zeros` only (about 1 GPU-hour); `fh_means` is the existing audited `student` arm, do
not regenerate it. Add `fh_zeros` to the benchmark config's `modes`; expected intervention calls 2,204; the record's
`center_mode` must say `delete` and `selection.default_high_masks` must be `true`:
```sh
python scripts/generate_learned_center.py --out out/learned-center-wan-swap --arm fh_zeros --student out/distill-controls-preserved/s3/student --center-mode delete
```
Seven VBench dimensions, paired comparisons (10,000 paired prompt resamples, NumPy seed 20261109) of `fh_zeros` vs
`student` (the fixed-high means student) and vs the teacher, same-seed latent relative L2 / PSNR vs the teacher, in
the layout of `distill-results/swap/full100/`.

How to read it, next to A and B: the fixed high pairs are the most concentrated set (RMS-MRL 0.72 vs 0.48 on the
selected pairs, pooled over the 64 calibration prompts and all tokens, per cell), so the trained student is expected to
miss its means more than shared22 missed its own (shared22: −0.008 subject consistency, quality measures within
their intervals). Report the size of the loss per dimension and say explicitly that this is a fine-tune-then-drop
result; the fixed-high zeros-then-fine-tune arm was never run, so it does not say whether the means are *needed* on
the fixed set. Budget: about 1.3 GPU-hours, one or two idle GPUs, same rules as above. Deliverables: add the screen
files and `full100/` rows to `distill-results/swap/`, a subsection to the same report section, `handoff.md` §16, and
a final message with the numbers. Commit once when C is complete.
