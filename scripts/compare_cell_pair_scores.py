"""Compare the pair scores of TWO cells that differ in one setting (normally the prompt), pair by pair.

Inputs: two folders written by cell_pair_scores.py (cell.npz and pair-scores.csv). Each is one cell: one prompt,
layer, denoising step, CFG branch and head. Nothing is pooled over cells.

Per pair, three things are put side by side:

  centred share   each cell's `ctr_att_abs_share`: the pair's share of the absolute attention-weighted score after
                  removing each query's constant. Every query x every key of that cell.
  raw share       each cell's `att_abs_share` (the same without removing the constant).
  map correlation for ONE query token, the Pearson correlation over all keys between the pair's score map
                  c_f[query, key] in cell A and in cell B. +1: the same picture in both cells; 0: unrelated.
                  Given for the centre query (the one of the map figures) and, to show how much it depends on
                  the query, as minimum / median / maximum over a 3 x 3 x 3 lattice of 27 query tokens.

A pair is called "strong" in a cell when its centred share is above the equal split 1/64.

  python scripts/compare_cell_pair_scores.py <cell folder A> <cell folder B> --out distill-results/score-attribution-cell/compare
"""

import argparse
import csv
import math

import numpy as np
import torch

from cell_pair_scores import AXES, load_cell, resolve
from full_sweep_metrics import rotate
from plot_cell_pair_scores import CENTRED, INK, INK2, MUTED, RAW, axis_names, finish, pair_axis, scope, setup, strip


def rotated(folder):
    tensors, sets, meta = load_cell(folder)
    native = getattr(torch, meta["dtype"]) if meta["dtype"] != "float32" else None
    qr = rotate(tensors["q"][None], tensors["cos"], tensors["sin"], native)[0]
    kr = rotate(tensors["k"][None], tensors["cos"], tensors["sin"], native)[0]
    rows = [r for r in csv.DictReader(open(folder / "pair-scores.csv")) if r["row"] == "pair"]
    added = {r["name"]: r for r in csv.DictReader(open(folder / "pair-scores.csv")) if r["row"] == "sum of pair rows"}
    return qr, kr, sets, meta, rows, added


def score_maps(qr, kr, index):
    """[pairs, keys]: the score each pair adds to the logit of query `index` and every key."""
    q = qr[index].double().reshape(-1, 2)
    k = kr.double().reshape(kr.shape[0], -1, 2)
    return torch.einsum("pc,npc->pn", q, k) / math.sqrt(qr.shape[1])


def correlation(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-300)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--grid", nargs=3, type=int, default=[12, 30, 30], metavar=("FRAMES", "ROWS", "COLUMNS"))
    parser.add_argument("--out", default="distill-results/score-attribution-cell/compare")
    args = parser.parse_args()
    folders = [resolve(args.a), resolve(args.b)]
    (qa, ka, sets, meta_a, rows_a, added_a), (qb, kb, sets_b, meta_b, rows_b, added_b) = rotated(folders[0]), rotated(folders[1])
    frames, height, width = args.grid
    if qa.shape != qb.shape or math.prod(args.grid) != qa.shape[0]:
        raise SystemExit("the two cells must have the same tokens, matching --grid")
    token = lambda t, h, w: (t * height + h) * width + w
    centre = token(frames // 2, height // 2, width // 2)
    lattice = [token(t, h, w) for t in sorted({frames // 6, frames // 2, frames - 1 - frames // 6})
               for h in sorted({height // 6, height // 2, height - 1 - height // 6}) for w in sorted({width // 6, width // 2, width - 1 - width // 6})]
    at_centre = correlation(score_maps(qa, ka, centre), score_maps(qb, kb, centre))
    over = np.stack([correlation(score_maps(qa, ka, i), score_maps(qb, kb, i)) for i in lattice])            # [queries, pairs]
    column = lambda rows, name: np.array([float(r[name]) for r in rows])
    share = {"centred": (column(rows_a, "ctr_att_abs_share"), column(rows_b, "ctr_att_abs_share")),
             "raw": (column(rows_a, "att_abs_share"), column(rows_b, "att_abs_share"))}
    name_a, name_b = folders[0].name, folders[1].name
    same = [k for k in ("prompt_id", "layer", "step", "branch", "head") if meta_a[k] == meta_b[k]]
    differ = [k for k in ("prompt_id", "layer", "step", "branch", "head") if meta_a[k] != meta_b[k]]
    label = lambda m: ", ".join(f"{k.replace('_id', '')} {m[k]}" for k in differ) or "same cell"
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name_a}--vs--{name_b}"
    with open(out / f"{stem}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "axis", *[f"in_{s}" for s in sets], "centred_share_a", "centred_share_b", "raw_share_a", "raw_share_b",
                         "map_correlation_centre_query", "map_correlation_min_27_queries", "map_correlation_median_27_queries", "map_correlation_max_27_queries"])
        for f in range(len(rows_a)):
            writer.writerow([f, AXES[f], *[int(sets[s][f]) for s in sets], *[f"{v:.8g}" for v in (share["centred"][0][f], share["centred"][1][f],
                             share["raw"][0][f], share["raw"][1][f], at_centre[f], over[:, f].min(), np.median(over[:, f]), over[:, f].max())]])
    equal = 1 / len(rows_a)
    strong = [f for f in range(len(rows_a)) if share["centred"][0][f] > equal]
    strong_b = [f for f in range(len(rows_a)) if share["centred"][1][f] > equal]
    pairs_text = lambda fs: ", ".join(f"{AXES[f]}{f}" for f in fs) or "none"
    text = [f"# Pair scores of two cells compared: {label(meta_a)} against {label(meta_b)}\n",
            f"Cell A: `{name_a}`. Cell B: `{name_b}`. Same in both: {', '.join(f'{k} {meta_a[k]}' for k in same)}. {meta_a['source']}. "
            "Shares are per cell over every query × every key (attention-weighted absolute score; centred = after removing each query's "
            "constant). The map correlation is for ONE query token: Pearson correlation over all "
            f"{qa.shape[0]:,} keys between the pair's score map in A and in B; centre query = frame {frames // 2}, row {height // 2}, column "
            f"{width // 2} (token {centre}); the range is over a 3 × 3 × 3 lattice of {len(lattice)} query tokens. Nothing is pooled over cells.\n",
            f"- Pairs with a centred share above the equal split 1/64 (\"strong\"): {len(strong)} in A ({pairs_text(strong)}), holding "
            f"{100 * share['centred'][0][strong].sum():.1f}% of A's centred absolute score; {len(strong_b)} in B ({pairs_text(strong_b)}), holding "
            f"{100 * share['centred'][1][strong_b].sum():.1f}% of B's.",
            f"- Pearson correlation across the 64 pairs between A's and B's centred shares: {np.corrcoef(*share['centred'])[0, 1]:.4f}; "
            f"largest difference of one pair's centred share: {100 * np.abs(share['centred'][0] - share['centred'][1]).max():.2f} percentage points.",
            "- Share of the absolute attention-weighted score per selection option (pairs' shares added), A → B: "
            + "; ".join(f"{s} raw {100 * float(added_a[s]['att_abs_share']):.2f}% → {100 * float(added_b[s]['att_abs_share']):.2f}%, centred "
                        f"{100 * float(added_a[s]['ctr_att_abs_share']):.2f}% → {100 * float(added_b[s]['ctr_att_abs_share']):.2f}%" for s in sets) + ".\n",
            "| pair | selected by | centred share A | centred share B | raw share A | raw share B | map correlation, centre query | min of 27 queries | median | max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in sorted(range(len(rows_a)), key=lambda f: -share["centred"][0][f]):
        chosen = ", ".join(s for s in sets if sets[s][f]) or "—"
        text.append(f"| {AXES[f]}{f} | {chosen} | {100 * share['centred'][0][f]:.2f}% | {100 * share['centred'][1][f]:.2f}% | {100 * share['raw'][0][f]:.2f}% | "
                    f"{100 * share['raw'][1][f]:.2f}% | {at_centre[f]:+.2f} | {over[:, f].min():+.2f} | {np.median(over[:, f]):+.2f} | {over[:, f].max():+.2f} |")
    (out / f"{stem}.md").write_text("\n".join(text) + "\n")

    plt = setup()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, .22 * len(sets) + .25], "hspace": .2})
    x = np.arange(len(rows_a))
    ax = axes[0]
    ax.bar(x - .2, 100 * share["centred"][0], width=.36, color=RAW, label=f"A: {label(meta_a)}")
    ax.bar(x + .2, 100 * share["centred"][1], width=.36, color=CENTRED, label=f"B: {label(meta_b)}")
    ax.axhline(100 * equal, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(63.6, 100 * equal, " equal split 1/64", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylabel("centred share of the absolute score (%)")
    ax.set_title("(a) Each cell on its own: share of each pair in the absolute attention-weighted score, after removing each query's constant. "
                 "Every query × every key", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    pair_axis(ax, labels=False)
    axis_names(ax, 1.085)
    ax = axes[1]
    ax.vlines(x, over.min(0), over.max(0), color=RAW, linewidth=2.6, alpha=.4, label=f"range over {len(lattice)} query tokens")
    ax.scatter(x, np.median(over, 0), s=20, color=RAW, edgecolors="#fcfcfb", linewidths=.6, zorder=3, label="median of those")
    ax.scatter(x, at_centre, s=30, marker="D", color=INK, edgecolors="#fcfcfb", linewidths=.6, zorder=4, label="centre query (the one of the map figures)")
    ax.axhline(0, color=INK2, linewidth=.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("map correlation between A and B")
    ax.set_title("(b) One query at a time: is the pair's score map over the keys the same picture in A and in B?  +1 the same, 0 unrelated",
                 loc="left", fontsize=9.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncols=3)
    pair_axis(ax, labels=False)
    strip(axes[2], sets)
    meta = dict(meta_a, prompt_id=meta_a["prompt_id"] if "prompt_id" in same else f"{meta_a['prompt_id']} (A) and {meta_b['prompt_id']} (B)")
    for key in ("layer", "step", "branch", "head"):
        if key not in same:
            meta[key] = f"{meta_a[key]} (A) and {meta_b[key]} (B)"
    title = scope(meta, f"(a) uses every one of the {qa.shape[0]:,} tokens as a query and as a key, in each cell separately. (b) is per query "
                  "token: Pearson correlation over all keys between the pair's score map in A and in B; no pooling over queries.").replace("ONE cell", "TWO cells")
    fig.subplots_adjust(left=.13, right=.985, bottom=.06)
    finish(fig, "Do the pair scores repeat between the two cells?\n" + title, out / f"{stem}.png")
    plt.close(fig)
    print("\n".join(text[:6]))
    print("WROTE", out / f"{stem}.md")


if __name__ == "__main__":
    main()
