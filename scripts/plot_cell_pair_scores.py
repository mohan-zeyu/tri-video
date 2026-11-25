"""Figures for the pair scores of ONE cell (folder written by cell_pair_scores.py: pair-scores.csv and cell.npz).

Every figure is about that one cell only: one prompt, one layer, one denoising step, one CFG branch, one head of
the model named in the cell's `source`. Nothing is pooled over prompts, layers, steps, branches or heads.

  1-positive-negative.png   per pair: positive and negative scores summed separately. (a) weighted by attention,
                            (b) every query-key combination once, (c) the sizes of (a) on a log scale.
                            Every query x every key.
  2-shares.png              per pair: share of the absolute attention-weighted score, raw and after removing each
                            query's constant; and the same summed over each pair set. Every query x every key.
  3-distributions.png       per pair: quantiles of the score over all query-key combinations, and quantiles
                            across queries of each query's attention-weighted score. Nothing is summed over queries.
  4-one-query-maps.png      ONE query token (default the grid centre): its attention and the score of a few pairs
                            over the keys of its frame and of its row. Recomputed from cell.npz; the numbers of
                            all 64 pairs on those two slices are saved as 4-one-query-maps.csv.
  5-one-query-maps-{t,h,w}.png            the same for every pair, one page per axis, one colour scale for all 64
  6-one-query-maps-own-scale-{t,h,w}.png  the same with each pair on its own colour scale (pattern, not size)

  python scripts/plot_cell_pair_scores.py distill-results/score-attribution-cell/d0000-L00-S01-conditional-H01
"""

import argparse
import csv
import math
import textwrap
from pathlib import Path

import numpy as np
import torch

from cell_pair_scores import AXES, LEVEL_NAMES, load_cell, resolve
from full_sweep_metrics import rotate

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POSITIVE, NEGATIVE, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"           # diverging pair: blue <-> red, gray midpoint
RAW, CENTRED = "#2a78d6", "#eb6834"                                    # categorical slots 1 and 2
BOUNDS = (21.5, 42.5)                                                  # between the t, h and w pairs


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True})
    return plt


def read(folder):
    rows = list(csv.DictReader(open(folder / "pair-scores.csv")))
    pairs = [r for r in rows if r["row"] == "pair"]
    column = lambda name, source=pairs: np.array([float(r[name]) for r in source])
    return rows, pairs, column


def scope(meta, what):
    cell = (f"ONE cell: prompt {meta['prompt_id']}, layer {meta['layer']}, denoising step {meta['step']}, {meta['branch']} branch, "
            f"head {meta['head']} (heads count from 0), {meta['source'].split(',')[0]}.")
    return "\n".join(textwrap.wrap(cell + " " + what, 150))


def pair_axis(ax, labels=True):
    ax.set_xlim(-.7, 63.7)
    ax.grid(axis="x", visible=False)
    for bound in BOUNDS:
        ax.axvline(bound, color=AXIS, linewidth=1)
    ax.set_xticks(range(64))
    ax.set_xticklabels([str(i) for i in range(64)] if labels else [], fontsize=6.2)
    ax.tick_params(axis="x", length=0 if not labels else 2)


def axis_names(ax, y=1.01):
    for centre, name in ((10.5, "t pairs (0–21)"), (32, "h pairs (22–42)"), (53, "w pairs (43–63)")):
        ax.text(centre, y, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK2)


def strip(ax, sets):
    """Which pairs each pair set selects in this cell: one row per set, a filled square per selected pair."""
    names = list(sets)
    for row, name in enumerate(names):
        chosen = np.flatnonzero(sets[name])
        ax.scatter(chosen, np.full(len(chosen), row), marker="s", s=30, color=INK2, linewidths=0)
    ax.set_ylim(len(names) - .4, -.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"selected by {n} ({int(sets[n].sum())})" for n in names], fontsize=8)
    ax.grid(visible=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    pair_axis(ax)
    ax.set_xlabel("rotary pair (lower angular frequency to the right inside each axis)")


def finish(fig, title, path, pad=.62):
    """Title block at the top left; the axes start `pad` inches below its last line (room for axis names and a panel title)."""
    fig.suptitle(title, x=.012, ha="left", fontsize=9.4, y=1 - .08 / fig.get_figheight(), va="top", linespacing=1.4)
    fig.subplots_adjust(top=1 - (.08 + .183 * (title.count("\n") + 1) + pad) / fig.get_figheight())
    fig.savefig(path, dpi=170)


def figure_signs(plt, folder, meta, sets, column, tokens):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12.2), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, .22 * len(sets) + .25], "hspace": .22})
    x = np.arange(64)
    low = min(column("att_neg").min(), column("all_neg").min()) * 1.12
    high = max(column("att_pos").max(), column("all_pos").max()) * 1.08
    panels = (("att", "(a) Weighted by attention: each query-key score counts with the attention the query gives that key"),
              ("all", f"(b) Every query-key combination counts once ({tokens * tokens:,} combinations)"))
    for ax, (prefix, title) in zip(axes, panels):
        ax.bar(x, column(prefix + "_pos"), width=.72, color=POSITIVE, label="sum of the positive scores")
        ax.bar(x, column(prefix + "_neg"), width=.72, color=NEGATIVE, label="sum of the negative scores")
        ax.axhline(0, color=INK2, linewidth=.8)
        ax.set_ylim(low, high)
        ax.set_ylabel("score (logit units)")
        ax.set_title(title, loc="left", fontsize=9.5)
        pair_axis(ax, labels=False)
    axis_names(axes[0], 1.09)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5, ncols=2)
    ax = axes[2]
    pos, neg = column("att_pos"), -column("att_neg")
    floor = 1e-5
    ax.vlines(x, np.maximum(np.minimum(pos, neg), floor), np.maximum(pos, neg), color=AXIS, linewidth=1.4, zorder=1)
    ax.scatter(x, np.maximum(pos, floor), s=30, color=POSITIVE, edgecolors=SURFACE, linewidths=.7, zorder=3, label="positive part")
    ax.scatter(x, np.maximum(neg, floor), s=58, color=NEGATIVE, edgecolors=SURFACE, linewidths=.7, zorder=2, label="size of the negative part")
    ax.set_yscale("log")
    ax.set_ylim(floor * .6, 4)
    ax.set_ylabel("size of the score (log scale)")
    ax.set_title("(c) The same numbers as (a) on a log scale. Where the two dots are close, positive and negative scores nearly cancel (a red rim around a blue dot: equal)", loc="left", fontsize=9.5)
    ax.legend(loc="lower center", frameon=False, fontsize=8.5, ncols=2)
    pair_axis(ax, labels=False)
    strip(axes[3], sets)
    title = scope(meta, f"Every one of the {tokens:,} tokens is a query and every one is a key; full 128-dim post-RoPE logit split into its 64 "
                  "pair scores. Positive and negative scores are summed separately (attention-weighted sums are divided by the number of "
                  "queries, the others by the number of combinations), so they cannot cancel.")
    fig.subplots_adjust(left=.13, right=.985, bottom=.05)
    finish(fig, "Score of each rotary pair, positive and negative parts apart\n" + title, folder / "figures" / "1-positive-negative.png")
    plt.close(fig)


def figure_shares(plt, folder, meta, sets, rows, column, tokens):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.4), gridspec_kw={"height_ratios": [3.2, .22 * len(sets) + .25, .42 * len(sets) + .3], "hspace": .34})
    x = np.arange(64)
    ax = axes[0]
    ax.bar(x - .2, 100 * column("att_abs_share"), width=.36, color=RAW, label="raw score")
    ax.bar(x + .2, 100 * column("ctr_att_abs_share"), width=.36, color=CENTRED,
           label="after removing each query's constant (the part softmax can see)")
    ax.axhline(100 / 64, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(63.6, 100 / 64, " equal split 1/64", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylabel("share of the absolute score (%)")
    ax.set_title("(a) Share of each pair in the absolute attention-weighted score (positive part minus negative part, over the same summed over the 64 pairs)",
                 loc="left", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    pair_axis(ax)
    axis_names(ax, 1.075)
    strip(axes[1], sets)
    axes[1].sharex(ax)
    ax = axes[2]
    names = list(sets)
    added = {r["name"]: r for r in rows if r["row"] == "sum of pair rows"}
    y = np.arange(len(names))
    for offset, key, colour in ((-.19, "att_abs_share", RAW), (.19, "ctr_att_abs_share", CENTRED)):
        values = [100 * float(added[n][key]) for n in names]
        ax.barh(y + offset, values, height=.34, color=colour)
        for yy, v in zip(y + offset, values):
            ax.text(v + .8, yy, f"{v:.2f}%", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({int(sets[n].sum())} pairs)" for n in names])
    ax.set_ylim(len(names) - .45, -.55)
    ax.set_xlim(0, 108)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the absolute attention-weighted score carried by the set's pairs (%) — the pairs' shares added, nothing cancels between pairs")
    ax.set_title("(b) The same shares added over the pairs each option selects in this cell (blue raw, orange after removing each query's constant)",
                 loc="left", fontsize=9.5)
    title = scope(meta, f"Every one of the {tokens:,} tokens is a query and every one is a key; each query-key score is weighted by the attention "
                  "the query gives that key. Shares use absolute scores, so every term is non-negative.")
    fig.subplots_adjust(left=.13, right=.985, bottom=.065)
    finish(fig, "Share of the absolute attention score carried by each pair and by each selection option\n" + title, folder / "figures" / "2-shares.png")
    plt.close(fig)


def ranges(ax, x, column, prefix, colour):
    q = {name: column(f"{prefix}_{name}") for name in LEVEL_NAMES}
    ax.vlines(x, q["min"], q["max"], color=AXIS, linewidth=1, label="minimum to maximum")
    ax.vlines(x, q["q05"], q["q95"], color=colour, linewidth=2.2, alpha=.45, label="5% to 95%")
    ax.vlines(x, q["q25"], q["q75"], color=colour, linewidth=4.6, label="25% to 75%")
    ax.scatter(x, q["median"], s=16, color=INK, edgecolors=SURFACE, linewidths=.6, zorder=3, label="median")
    ax.axhline(0, color=INK2, linewidth=.8)


def figure_distributions(plt, folder, meta, sets, column, tokens):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, .22 * len(sets) + .25], "hspace": .2})
    x = np.arange(64)
    ranges(axes[0], x, column, "all", POSITIVE)
    axes[0].set_title(f"(a) Score c_f[i,j] of the pair over all {tokens * tokens:,} query-key combinations, each counted once", loc="left", fontsize=9.5)
    axes[0].set_ylabel("score (logit units)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5, ncols=4)
    ranges(axes[1], x, column, "qry", POSITIVE)
    axes[1].set_title(f"(b) One number per query: a[i] = sum over keys of attention × score of the pair. Spread across the {tokens:,} queries",
                      loc="left", fontsize=9.5)
    axes[1].set_ylabel("attention-weighted score of one query")
    for ax in axes[:2]:
        pair_axis(ax, labels=False)
    axis_names(axes[0], 1.085)
    strip(axes[2], sets)
    title = scope(meta, f"Every one of the {tokens:,} tokens is a query and every one is a key. Quantiles only: nothing is summed or averaged "
                  "over queries or keys in (a), nothing over queries in (b).")
    fig.subplots_adjust(left=.13, right=.985, bottom=.06)
    finish(fig, "Spread of each pair's score inside the cell\n" + title, folder / "figures" / "3-distributions.png")
    plt.close(fig)


def one_query(tensors, meta, query, grid):
    """Attention [frames, rows, columns] of ONE query token and the score of each of the 64 pairs over every key."""
    frames, height, width = grid
    native = getattr(torch, meta["dtype"]) if meta["dtype"] != "float32" else None
    qr = rotate(tensors["q"][None], tensors["cos"], tensors["sin"], native)[0]
    kr = rotate(tensors["k"][None], tensors["cos"], tensors["sin"], native)[0]
    scale = 1 / math.sqrt(qr.shape[1])
    index = (query[0] * height + query[1]) * width + query[2]
    attention = torch.softmax((qr[index] @ kr.T * scale).double(), 0).reshape(grid).numpy()
    score = np.stack([((qr[index, 2 * f:2 * f + 2] @ kr[:, 2 * f:2 * f + 2].T) * scale).double().reshape(grid).numpy() for f in range(qr.shape[1] // 2)])
    return index, attention, score


def colour_maps():
    from matplotlib.colors import LinearSegmentedColormap
    return (LinearSegmentedColormap.from_list("score", [NEGATIVE, NEUTRAL, POSITIVE]),
            LinearSegmentedColormap.from_list("attention", [NEUTRAL, "#86b6ef", "#2a78d6", "#0d366b"]))


def map_pair(top, bottom, volume, cmap, vmin, vmax, query, grid, title, first):
    """One item: the keys of the query's frame (top) and the keys of the query's row in every frame (bottom)."""
    frames, height, width = grid
    t0, h0, w0 = query
    image = top.imshow(volume[t0], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    bottom.imshow(volume[:, h0], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest", aspect=1.6)
    top.plot(w0, h0, marker="+", color=INK, markersize=9, markeredgewidth=1.4)
    bottom.plot(w0, t0, marker="+", color=INK, markersize=9, markeredgewidth=1.4)
    top.set_title(title, fontsize=8.4, linespacing=1.25)
    for ax in (top, bottom):
        ax.grid(visible=False)
        ax.set_xticks([0, width - 1])
        ax.set_xticklabels(["0", str(width - 1)], fontsize=7)
        ax.tick_params(length=0, labelsize=7)
        for side in ax.spines.values():
            side.set_visible(False)
    top.set_yticks([0, height - 1] if first else [])
    bottom.set_yticks([0, frames - 1] if first else [])
    if first:
        top.set_ylabel(f"key row h  (frame {t0})", fontsize=8.5)
        bottom.set_ylabel(f"key frame t  (row {h0})", fontsize=8.5)
    return image


def colour_bars(fig, attention_image, score_image, note):
    inch = 1 / fig.get_figheight()
    bar = fig.add_axes([.06, .62 * inch, .2, .15 * inch])
    fig.colorbar(attention_image, cax=bar, orientation="horizontal")
    bar.set_xlabel("attention weight", fontsize=8.5)
    bar.tick_params(labelsize=7.5)
    if score_image is not None:
        bar = fig.add_axes([.42, .62 * inch, .4, .15 * inch])
        fig.colorbar(score_image, cax=bar, orientation="horizontal")
        bar.set_xlabel(note, fontsize=8.5)
        bar.tick_params(labelsize=7.5)
    else:
        fig.text(.42, .62 * inch, note, fontsize=8.5, color=INK2, va="bottom", ha="left")


def query_scope(meta, query, index, grid, tail):
    frames, height, width = grid
    cell = (f"ONE query token of ONE cell: prompt {meta['prompt_id']}, layer {meta['layer']}, denoising step {meta['step']}, {meta['branch']} branch, "
            f"head {meta['head']}, {meta['source'].split(',')[0]}. Query at frame {query[0]}, row {query[1]}, column {query[2]} (the cross), token "
            f"{index} of {frames * height * width:,}. Top maps: the {height} × {width} keys of the query's own frame. Bottom maps: the {frames} × {width} "
            "keys of the query's own row in every frame. The horizontal axis of every map is the key column w. Nothing is pooled: every pixel is one query-key score. " + tail)
    return "\n".join(textwrap.wrap(cell, 150))


def figure_maps(plt, folder, meta, tensors, pairs, query, grid):
    index, attention, score = one_query(tensors, meta, query, grid)
    limit = max(float(np.abs(score[f]).max()) for f in pairs)
    diverging, sequential = colour_maps()
    columns = 1 + len(pairs)
    fig, axes = plt.subplots(2, columns, figsize=(2.05 * columns + .6, 6.6), gridspec_kw={"height_ratios": [grid[1], grid[0] * 1.6], "hspace": .32, "wspace": .12})
    first = map_pair(axes[0, 0], axes[1, 0], attention, sequential, 0, float(attention.max()), query, grid, "attention of this query\n(sums to 1 over all keys)", True)
    for col, f in enumerate(pairs, 1):
        image = map_pair(axes[0, col], axes[1, col], score[f], diverging, -limit, limit, query, grid,
                         f"{AXES[f]} pair {f}\nweighted by attention: {float((attention * score[f]).sum()):+.3f}", False)
    for ax in axes[1]:
        ax.set_xlabel("key column w", fontsize=8)
    fig.subplots_adjust(left=.06, right=.985, bottom=.2)
    colour_bars(fig, first, image, "score the pair adds to the logit of this query and that key (same colour scale for every pair; red negative, blue positive)")
    title = query_scope(meta, query, index, grid, f"Pairs that look blank have scores too small to show on the common scale (±{limit:.2f}).")
    finish(fig, "What a pair's score looks like across keys, for one query\n" + title, folder / "figures" / "4-one-query-maps.png", .55)
    plt.close(fig)
    frames, height, width = grid
    with open(folder / "figures" / "4-one-query-maps.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice", "key_frame", "key_row", "key_column", "attention", *[f"pair_{f}" for f in range(len(score))]])
        for t in range(frames):
            for h in range(height):
                for w in range(width):
                    if t == query[0] or h == query[1]:
                        writer.writerow(["frame" if t == query[0] else "row", t, h, w, f"{attention[t, h, w]:.8g}", *[f"{s[t, h, w]:.8g}" for s in score]])
    return {"query_token": index, "colour_limit": limit, "attention_weighted": {f: float((attention * score[f]).sum()) for f in pairs}}


def figure_all_maps(plt, folder, meta, tensors, sets, query, grid, own_scale):
    """Every pair, one page per axis (t, h, w). Common colour scale over the 64 pairs, or each pair on its own scale."""
    index, attention, score = one_query(tensors, meta, query, grid)
    limit = float(np.abs(score).max())
    diverging, sequential = colour_maps()
    columns, written = 8, []
    for axis in ("t", "h", "w"):
        pairs = [f for f in range(len(score)) if AXES[f] == axis]
        blocks = -(-(1 + len(pairs)) // columns)
        fig = plt.figure(figsize=(2.05 * columns + .6, 4.25 * blocks + 2.3))
        spec = fig.add_gridspec(3 * blocks, columns, height_ratios=[10, grid[1], grid[0] * 1.6] * blocks, hspace=.2, wspace=.12)   # title room, frame map, row map
        slot = lambda i: (fig.add_subplot(spec[3 * (i // columns) + 1, i % columns]), fig.add_subplot(spec[3 * (i // columns) + 2, i % columns]))
        first = map_pair(*slot(0), attention, sequential, 0, float(attention.max()), query, grid, "attention of this query\n(sums to 1 over all keys)\n ", True)
        image = None
        for i, f in enumerate(pairs, 1):
            own = float(np.abs(score[f]).max())
            names = [name for name, mask in sets.items() if mask[f]]
            chosen = "no option" if not names else "all options" if len(names) == len(sets) > 1 else ", ".join(names)
            top, bottom = slot(i)
            image = map_pair(top, bottom, score[f], diverging, -(own if own_scale else limit), own if own_scale else limit, query, grid,
                             f"{axis} pair {f}" + (f"  ·  scale ±{own:.3g}" if own_scale else "") + f"\nweighted by attention: {float((attention * score[f]).sum()):+.4f}\n"
                             f"selected by: {chosen}", i % columns == 0)
        fig.subplots_adjust(left=.06, right=.985, bottom=1.15 / fig.get_figheight())
        if own_scale:
            colour_bars(fig, first, None, "Each pair has its OWN colour scale: red = −scale, gray = 0, blue = +scale, with the scale printed in the pair's title.\n"
                        "Colours are not comparable between pairs.")
            tail = "Each pair is drawn on its own colour scale, so the pattern of weak pairs is visible but sizes cannot be compared between panels."
        else:
            colour_bars(fig, first, image, "score the pair adds to the logit of this query and that key (same colour scale for all 64 pairs; red negative, blue positive)")
            tail = f"All 64 pairs share one colour scale (±{limit:.2f}); pairs that look blank have scores too small to show on it."
        name = f"{'6-one-query-maps-own-scale' if own_scale else '5-one-query-maps'}-{axis}.png"
        finish(fig, f"Score of every {axis} pair across keys, for one query\n" + query_scope(meta, query, index, grid, tail), folder / "figures" / name, .12)
        plt.close(fig)
        written.append(name)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cell", help="folder written by cell_pair_scores.py")
    parser.add_argument("--grid", nargs=3, type=int, default=[12, 30, 30], metavar=("FRAMES", "ROWS", "COLUMNS"))
    parser.add_argument("--query", nargs=3, type=int, metavar=("FRAME", "ROW", "COLUMN"), help="default: the grid centre")
    parser.add_argument("--map-pairs", nargs="+", type=int, default=[47, 26, 4, 50, 56, 36, 6])
    args = parser.parse_args()
    folder = resolve(args.cell)
    (folder / "figures").mkdir(exist_ok=True)
    tensors, sets, meta = load_cell(folder)
    rows, pairs, column = read(folder)
    tokens = meta["tokens"]
    if math.prod(args.grid) != tokens:
        raise SystemExit(f"grid {args.grid} does not hold {tokens} tokens")
    frames, height, width = args.grid
    for position, axis_pair in ((1, 43), (width, 22), (height * width, 0)):                       # token order: frame, row, column
        if not np.isclose(float(tensors["cos"][position, axis_pair]), math.cos(1.0), atol=1e-3):
            raise SystemExit("unexpected token order")
    plt = setup()
    figure_signs(plt, folder, meta, sets, column, tokens)
    figure_shares(plt, folder, meta, sets, rows, column, tokens)
    figure_distributions(plt, folder, meta, sets, column, tokens)
    query = args.query or [frames // 2, height // 2, width // 2]
    info = figure_maps(plt, folder, meta, tensors, args.map_pairs, query, tuple(args.grid))
    for own_scale in (False, True):
        figure_all_maps(plt, folder, meta, tensors, sets, query, tuple(args.grid), own_scale)
    print("WROTE", sorted(p.name for p in (folder / "figures").iterdir()), info)


if __name__ == "__main__":
    main()
