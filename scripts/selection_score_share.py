"""Attention-score share carried by named sets of rotary pairs (e.g. the pairs a student replaces or zeroes).

Reads the per-prompt records of collect_pair_score_attribution.py (frozen teacher) and one or more pair masks
from selection files written by select_pairs.py. A mask set gives, for every layer x step x CFG branch x head,
the pairs selected on Q and on K; the set's pairs in a cell are the union of the two (a pair zeroed on either
side contributes nothing to the logit). For every prompt x cell the script sums the cell's pair shares over
the set:

  attn_score_share  the plain percentage of the attention score: of the raw logit the cell's queries give to the
                  keys they attend (softmax-weighted, summed over queries), the part contributed by the set.
                  Not row-centred, so it includes the per-query constant softmax ignores; undefined (counted,
                  left out of the quantiles) in cells where the pairs' raw contributions nearly cancel
  var_cov_share   signed share of the row-centred logit variance
  var_own_share   share of the pairs' own contribution variances
  adv_share       share of the attention-weighted logit advantage of the attended keys over the mean key
  top1_share      share of the arg-max key's logit margin over the mean key
  deleted_logit_variance   sum_{f,g in set} pair_cov: row-centred variance of the logit part the set contributes,
                           relative to the row variance of the full logit (needs pair_cov in the records)

Each cell's pair shares pool every query x every key of the full 128-dim post-RoPE logits
(pair_score_attribution.py). The shares are those of whatever model the records come from: the teacher, or a
trained student when the records were collected with --student. Nothing is averaged over cells: every number is
a quantile (minimum, 5%, 25%, median, 75%, 95%, maximum) over the prompt x cell entries in which the set has at
least one pair (so a student's exempt cells are left out), or over those of one layer, step or branch.

  --set NAME PATH VARIANT        masks = torch.load(PATH)["variants"][VARIANT] (no exemptions applied)
  --difference NAME A B          the pairs of set A that are not in set B, per cell (e.g. what 32 adds to 22)
  --record-masks NAME            the pairs the student replaced, stored in its own records (replaced_q | replaced_k)
  --masks-from-record NAME FILE  the same masks read from one student record FILE, applied to these records
                                 (e.g. teacher records: the same pairs and cells, before training)

References written per set when its size is the same k in every cell: k/64 (a choice blind to the score) and
the smallest var_own_share any k pairs of the cell can reach.

  python scripts/selection_score_share.py --stats out/distill-score-attribution \
      --set shared22 out/distill-selection-v2/selection.pt shared \
      --set shared32 out/study-shared32/selection/selection.pt shared \
      --difference shared32_minus_shared22 shared32 shared22 \
      --out distill-results/score-attribution --tag shared32
"""

import argparse
import json
import warnings

import numpy as np
import torch

from analyze_pair_score_attribution import LEVEL_NAMES, breakdown, dist, json_ready, load, resolve, shares

MEASURES = ("attn_score_share", "var_cov_share", "var_own_share", "adv_share", "top1_share")


def build(records, steps, masks):
    """masks: {name: bool [layer, step, branch, head, 64] over the analyzed steps and the records' branches}."""
    per_set = {name: {m: [] for m in (*MEASURES, "deleted_logit_variance")} for name in masks}
    active = {name: mask.any(-1) for name, mask in masks.items()}               # cells where the set has pairs
    sizes = {name: sorted({int(v) for v in np.unique(mask.sum(-1)[active[name]])}) for name, mask in masks.items()}
    lowest = {name: [] for name, size in sizes.items() if len(size) == 1}
    has_cov = "pair_cov" in records[0]
    for record in records:                                                     # one prompt at a time (memory)
        derived = shares(record)
        own_sorted = np.sort(derived["var_own_share"][:, steps].astype(np.float64), -1).cumsum(-1)
        cov = record["pair_cov"][:, steps].float() if has_cov else None
        for name, mask in masks.items():
            hide = np.where(active[name], 0.0, np.nan)                         # cells without pairs drop out
            for measure in MEASURES:
                per_set[name][measure].append((derived[measure][:, steps].astype(np.float64) * mask).sum(-1) + hide)
            if has_cov:
                m = torch.from_numpy(mask).float()
                per_set[name]["deleted_logit_variance"].append(torch.einsum("lsbhf,lsbhfg,lsbhg->lsbh", m, cov, m).numpy() + hide)
            if name in lowest:
                lowest[name].append(own_sorted[..., sizes[name][0] - 1] + hide)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                        # a layer or step without active cells gives NaN
        for name, values in per_set.items():
            cells = int(active[name].sum())
            item = {"pairs_per_active_cell": dist(masks[name].sum(-1)[active[name]]).tolist(), "active_cells_per_prompt": cells,
                    "cells_per_prompt": int(active[name].size), "entries": cells * len(records)}
            for measure, stack in values.items():
                if stack:
                    x = np.stack(stack)                                        # [prompt, layer, step, branch, head]
                    if np.isinf(x).any() or (measure != "attn_score_share" and np.isnan(x[:, active[name]]).any()):
                        raise FloatingPointError(f"{name} {measure}")
                    item[measure] = breakdown(x, skip_nan=True)
                    if measure == "attn_score_share":
                        item["attn_score_share_undefined_entries"] = int(np.isnan(x[:, active[name]]).sum())
            if name in lowest:
                item["reference"] = {"pairs": sizes[name][0], "score_blind_share": sizes[name][0] / masks[name].shape[-1],
                                     "smallest_possible_var_own_share": breakdown(np.stack(lowest[name]), skip_nan=True)}
            out[name] = item
    return out


def markdown(report):
    s = report["shape"]
    text = ["# Attention-score share carried by selected pair sets (frozen teacher)\n",
            f"Source: {report['source']}. {s['prompts']} prompts x {s['layers']} layers x {s['steps']} denoising steps x "
            f"{s['branches']} CFG branches x {s['heads']} heads = {report['entries']:,} prompt x cell entries. A cell's pair shares pool "
            "every query x every key of the full 128-dim post-RoPE logits of the model the records come from, and are summed "
            "over the set's pairs in that cell (union of its Q and K masks). Every row is a distribution over the entries in "
            "which the set has at least one pair (column `entries`); nothing is averaged over cells. `attn_score_share` is the "
            "plain percentage of the attention-weighted raw logit; the other shares are relative to each query's mean over keys.\n",
            "| set | pairs per cell | entries | quantity | " + " | ".join(LEVEL_NAMES) + " |\n|---|---|---:|---|" + "---:|" * 7]
    for name, item in report["sets"].items():
        size = item["pairs_per_active_cell"]
        pairs = f"{size[3]:.0f}" if size[0] == size[-1] else f"{size[0]:.0f}-{size[-1]:.0f}"
        for measure in (*MEASURES, "deleted_logit_variance"):
            if measure in item:
                text.append(f"| {name} | {pairs} | {item['entries']:,} | {measure} | " + " | ".join(f"{v:.4f}" for v in item[measure]["all"]) + " |")
        if "reference" in item:
            ref = item["reference"]
            text.append(f"| {name} | {pairs} | {item['entries']:,} | smallest var_own_share any {ref['pairs']} pairs can reach | "
                        + " | ".join(f"{v:.4f}" for v in ref["smallest_possible_var_own_share"]["all"]) + " |")
    text.append("\nA choice of k pairs blind to the score carries k/64 of a share: "
                + ", ".join(f"{name} {item['reference']['score_blind_share']:.3f}" for name, item in report["sets"].items() if "reference" in item) + ".\n")
    text.append("`attn_score_share` is undefined where the pairs' raw contributions nearly cancel: "
                + ", ".join(f"{name} {item['attn_score_share_undefined_entries']:,} of {item['entries']:,} entries" for name, item in report["sets"].items()) + ".\n")
    cell = lambda c: "no active cells" if c[3] is None else f"{c[3]:.3f} [{c[1]:.3f}, {c[5]:.3f}]"
    for measure in ("attn_score_share", "adv_share", "var_cov_share"):
        names = list(report["sets"])
        text.append(f"## {measure} by layer: median [5%, 95%] over the entries of that layer\n")
        text.append("| layer | " + " | ".join(names) + " |\n|---:|" + "---:|" * len(names))
        for layer in range(s["layers"]):
            cells = [report["sets"][n][measure]["by_layer"][layer] for n in names]
            text.append(f"| {layer} | " + " | ".join(cell(c) for c in cells) + " |")
        text.append(f"\n## {measure} by denoising step: median [5%, 95%] over the entries of that step\n")
        text.append("| step | " + " | ".join(names) + " |\n|---:|" + "---:|" * len(names))
        for i, step in enumerate(report["steps_analyzed_zero_based"]):
            cells = [report["sets"][n][measure]["by_step"][i] for n in names]
            text.append(f"| {step + 1} | " + " | ".join(cell(c) for c in cells) + " |")
        text.append("")
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-score-attribution")
    parser.add_argument("--ids", nargs="+", help="restrict to these prompt ids")
    parser.add_argument("--set", nargs=3, action="append", default=[], metavar=("NAME", "PATH", "VARIANT"))
    parser.add_argument("--record-masks", metavar="NAME", help="use replaced_q | replaced_k stored in these (student) records")
    parser.add_argument("--masks-from-record", nargs=2, action="append", default=[], metavar=("NAME", "FILE"))
    parser.add_argument("--difference", nargs=3, action="append", default=[], metavar=("NAME", "A", "B"))
    parser.add_argument("--out", default="distill-results/score-attribution")
    parser.add_argument("--tag", required=True, help="outputs are selection-shares-<tag>.json and .md")
    args = parser.parse_args()
    records, steps = load(args.stats, args.ids)
    branches = len(records[0]["branches"])
    masks, sources = {}, {}
    for name, path, variant in args.set:
        selection = torch.load(resolve(path), map_location="cpu")
        chosen = selection["variants"][variant]
        masks[name] = (chosen["mask_q"] | chosen["mask_k"]).numpy()[:, steps, :branches]
        sources[name] = {"path": str(path), "variant": variant, "pairs_in_file": selection.get("pairs"),
                         "q_equals_k": bool(torch.equal(chosen["mask_q"], chosen["mask_k"]))}
    if args.record_masks:
        first = records[0]["replaced_q"] | records[0]["replaced_k"]
        for r in records:
            if not torch.equal(r["replaced_q"] | r["replaced_k"], first):
                raise SystemExit("the records do not share one set of replaced pairs")
        masks[args.record_masks] = first.numpy()[:, steps, :branches]
        sources[args.record_masks] = {"records": "replaced_q | replaced_k of these records", "student": records[0].get("student"),
                                      "student_mode": records[0].get("student_mode")}
    for name, path in args.masks_from_record:
        other = torch.load(resolve(path), map_location="cpu", mmap=True)
        masks[name] = (other["replaced_q"] | other["replaced_k"]).numpy()[:, steps, :branches]
        sources[name] = {"record": str(path), "student": other.get("student"), "student_mode": other.get("student_mode")}
    if not masks:
        raise SystemExit("give at least one of --set, --record-masks, --masks-from-record")
    for name, a, b in args.difference:
        masks[name] = masks[a] & ~masks[b]
        sources[name] = {"difference": [a, b], "pairs_of_b_outside_a": int((masks[b] & ~masks[a]).sum())}
    shape = records[0]["var_cov"][:, steps].shape
    for name, mask in masks.items():
        if mask.shape != tuple(shape):
            raise SystemExit(f"{name}: mask shape {mask.shape} does not match the records {tuple(shape)}")
    report = {
        "prompts": [r["prompt_id"] for r in records], "steps_analyzed_zero_based": list(steps), "branches": records[0]["branches"],
        "source": records[0]["source"], "quantile_levels": list(LEVEL_NAMES),
        "shape": {"prompts": len(records), "layers": shape[0], "steps": shape[1], "branches": shape[2], "heads": shape[3]},
        "entries": len(records) * int(np.prod(shape[:4])), "set_sources": sources,
        "scope": "per prompt x layer x step x branch x head: the cell's pair shares (every query x every key, full 128-dim "
                 "post-RoPE logits of the model named in `source`) summed over the set's pairs in that cell (union of its Q "
                 "and K masks). 'all' is the distribution over the entries in which the set has at least one pair; by_layer / "
                 "by_step / by_branch over those of one layer, step or branch (null where there are none). No mean over cells.",
        "sets": build(records, steps, masks),
    }
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json_ready(report)                                                # undefined quantiles become null
    (out / f"selection-shares-{args.tag}.json").write_text(json.dumps(report, allow_nan=False) + "\n")
    (out / f"SELECTION-SHARES-{args.tag}.md").write_text(markdown(report))
    print(markdown(report)[:3500])


if __name__ == "__main__":
    main()
