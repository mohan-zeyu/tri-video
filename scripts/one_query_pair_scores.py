"""Pair scores for ONE query token with every key: absolute raw scores, their share per pair, and their maps.

Inputs: cell folders written by cell_pair_scores.py (cell.npz). Each cell is one prompt (or one text-noise
combination), layer, denoising step, CFG branch and head. Everything here is for ONE query token i0 of a cell
(default the grid centre) against all N keys. No attention weighting, no centring, no pooling over queries.

    c_f[j]     = <qr_i0[f], kr_j[f]> / sqrt(128)      raw score pair f adds to the logit of query i0 and key j
    |c_f[j]|                                           its absolute value: the absolute map of pair f
    size_f     = sum over all N keys j of |c_f[j]|     every key counts once
    share_f    = size_f / sum over the 64 pairs g of size_g

Between two cells A and B (same query position), per pair, the Pearson correlation over the N keys of
  the absolute maps |c_f| in A and in B      ("absolute")  and
  the signed raw maps c_f in A and in B      ("signed")
The absolute map cannot tell a pattern from its sign-flipped copy; the signed correlation is listed next to it
for that reason.

Outputs in --out: one-query-pairs.csv (one row per cell x pair), one-query-correlations.csv (one row per
comparison x pair), ONE-QUERY.md, one-query.png, and per cell abs-maps-<label>-{t,h,w}.png: the absolute map of
every pair over the keys of the query's frame and of the query's row, one colour scale for all 64 pairs.

Cells are styled in the order given: colours alternate blue / orange, the marker changes every two cells. For the
text x noise design give T0N0, T6N0, T0N6, T6N6: colour = text, marker = noise.

  python scripts/one_query_pair_scores.py --out distill-results/score-attribution-seed/one-query \
      --cell T0N0 <folder> --cell T6N0 <folder> --cell T0N6 <folder> --cell T6N6 <folder> \
      --compare T0N0 T6N0 "same noise, different text" --compare T0N0 T0N6 "same text, different noise" ...
"""

import argparse
import csv
import math
import textwrap

import numpy as np

from cell_pair_scores import AXES, cell_name, load_cell, resolve
from plot_cell_pair_scores import (CENTRED, INK2, MUTED, NEUTRAL, RAW, SURFACE, axis_names, colour_bars, finish, map_pair, one_query,
                                   pair_axis, setup, strip)

CELL_STYLES = [(RAW, "o"), (CENTRED, "o"), (RAW, "^"), (CENTRED, "^")]
TYPE_COLOURS = [RAW, CENTRED, "#1baf7a"]


def correlation(a, b):
    """Pearson correlation along the last axis of [pairs, keys] arrays."""
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def describe(meta):
    return (f"layer {meta['layer']}, denoising step {meta['step']}, {meta['branch']} branch, head {meta['head']} (heads count from 0), "
            f"{meta['source'].split(',')[0].split(' replaying')[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", nargs=2, action="append", required=True, metavar=("LABEL", "FOLDER"))
    parser.add_argument("--compare", nargs=3, action="append", default=[], metavar=("LABEL_A", "LABEL_B", "TYPE"))
    parser.add_argument("--grid", nargs=3, type=int, default=[12, 30, 30], metavar=("FRAMES", "ROWS", "COLUMNS"))
    parser.add_argument("--query", nargs=3, type=int, metavar=("FRAME", "ROW", "COLUMN"), help="default: the grid centre")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-maps", action="store_true", help="skip the per-cell map pages")
    args = parser.parse_args()
    if len(args.cell) > len(CELL_STYLES):
        raise SystemExit(f"at most {len(CELL_STYLES)} cells")
    grid = tuple(args.grid)
    frames, height, width = grid
    query = args.query or [frames // 2, height // 2, width // 2]
    cells = {}
    for label, folder in args.cell:
        tensors, sets, meta = load_cell(resolve(folder))
        if math.prod(grid) != meta["tokens"]:
            raise SystemExit(f"grid {grid} does not hold {meta['tokens']} tokens")
        index, attention, score = one_query(tensors, meta, query, grid)                # score [pairs, frames, rows, columns]
        flat = score.reshape(len(score), -1)
        size = np.abs(flat).sum(1)
        centred = np.abs(flat - flat.mean(1, keepdims=True)).sum(1)
        cells[label] = {"meta": meta, "sets": sets, "folder": cell_name(meta), "attention": attention, "score": score, "flat": flat, "index": index,
                        "size": size, "share": size / size.sum(), "centred_share": centred / centred.sum(), "net": flat.sum(1), "largest": np.abs(flat).max(1)}
    first = next(iter(cells.values()))
    sets, pairs, tokens, index = first["sets"], len(first["size"]), first["meta"]["tokens"], first["index"]
    comparisons = []
    for a, b, kind in args.compare:
        comparisons.append({"a": a, "b": b, "type": kind, "absolute": correlation(np.abs(cells[a]["flat"]), np.abs(cells[b]["flat"])),
                            "signed": correlation(cells[a]["flat"], cells[b]["flat"]),
                            "share_correlation": float(np.corrcoef(cells[a]["share"], cells[b]["share"])[0, 1])})
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    where = f"frame {query[0]}, row {query[1]}, column {query[2]} (token {index} of {tokens:,})"
    name = lambda f: f"{AXES[f]}{f}"

    with open(out / "one-query-pairs.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "folder", "pair", "axis", *[f"in_{s}" for s in sets], "sum_abs_raw_score", "share_of_abs_raw_score",
                         "share_of_abs_centred_score", "sum_raw_score", "largest_abs_raw_score"])
        for label, c in cells.items():
            for f in range(pairs):
                writer.writerow([label, c["folder"], f, AXES[f], *[int(c["sets"][s][f]) for s in sets],
                                 *[f"{v:.8g}" for v in (c["size"][f], c["share"][f], c["centred_share"][f], c["net"][f], c["largest"][f])]])
    with open(out / "one-query-correlations.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "cell_a", "cell_b", "pair", "axis", "correlation_absolute_maps", "correlation_signed_maps"])
        for r in comparisons:
            for f in range(pairs):
                writer.writerow([r["type"], r["a"], r["b"], f, AXES[f], f"{r['absolute'][f]:.8g}", f"{r['signed'][f]:.8g}"])

    levels = (0, .05, .25, .5, .75, .95, 1)
    dist = lambda v: " | ".join(f"{x:+.3f}" for x in np.quantile(v, levels))
    text = ["# One query, every key: absolute raw score of each pair\n",
            f"Cells: " + "; ".join(f"{label} = `{c['folder']}`" for label, c in cells.items()) + f". Each: {describe(first['meta'])}. "
            f"Everything below is for ONE query token, {where}, against all {tokens:,} keys, each key counted once. Raw scores: no "
            "attention weighting, no centring, no pooling over queries or cells.\n",
            "`share` of a pair = sum over the keys of |raw score of the pair| ÷ the same summed over the 64 pairs.\n",
            "## Largest pairs per cell (share of the absolute raw score, this one query)\n",
            "| cell | ten largest pairs | pairs above the equal split 1/64 | share held by them |", "|---|---|---:|---:|"]
    for label, c in cells.items():
        order = np.argsort(-c["share"])[:10]
        above = c["share"] > 1 / pairs
        text.append(f"| {label} | " + ", ".join(f"{name(f)} {100 * c['share'][f]:.1f}%" for f in order) + f" | {int(above.sum())} | {100 * c['share'][above].sum():.1f}% |")
    if comparisons:
        text += ["\n## Comparisons (same query token in both cells)\n",
                 "`shares`: Pearson correlation across the 64 pairs between the two cells' shares. Map correlations are per pair, over the "
                 f"{tokens:,} keys; the seven columns are the minimum, 5%, 25%, median, 75%, 95% and maximum over the 64 pairs (not a mean).\n",
                 "| type | A vs B | shares | maps | min | 5% | 25% | median | 75% | 95% | max |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in comparisons:
            text.append(f"| {r['type']} | {r['a']} vs {r['b']} | {r['share_correlation']:.4f} | absolute | {dist(r['absolute'])} |")
            text.append(f"| {r['type']} | {r['a']} vs {r['b']} | | signed | {dist(r['signed'])} |")
    text += ["\n## Every pair\n", "| pair | " + " | ".join(f"share {label}" for label in cells) + " |" + "".join(f" {r['a']} vs {r['b']}, absolute |" for r in comparisons),
             "|---|" + "---:|" * (len(cells) + len(comparisons))]
    for f in range(pairs):
        text.append(f"| {name(f)} | " + " | ".join(f"{100 * c['share'][f]:.2f}%" for c in cells.values()) + " |" + "".join(f" {r['absolute'][f]:+.2f} |" for r in comparisons))
    (out / "ONE-QUERY.md").write_text("\n".join(text) + "\n")

    plt = setup()
    x = np.arange(pairs)
    rows = 3 if comparisons else 1
    fig, axes = plt.subplots(rows + 1, 1, figsize=(14, 3.6 * rows + 2.6), sharex=True,
                             gridspec_kw={"height_ratios": [3] * rows + [.22 * len(sets) + .25], "hspace": .22})
    ax = axes[0]
    shares = np.stack([c["share"] for c in cells.values()])
    if len(cells) > 1:
        ax.vlines(x, 100 * shares.min(0), 100 * shares.max(0), color=MUTED, linewidth=.8, zorder=1)
    spread = np.linspace(-.2, .2, len(cells)) if len(cells) > 1 else [0]
    for offset, (label, c), (colour, marker) in zip(spread, cells.items(), CELL_STYLES):
        ax.scatter(x + offset, 100 * c["share"], s=22, marker=marker, color=colour, edgecolors=SURFACE, linewidths=.5, zorder=3, label=f"{label}: {c['folder'].split('-L')[0]}")
    ax.axhline(100 / pairs, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(pairs - .4, 100 / pairs, " equal split 1/64", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylabel("share of the absolute raw score (%)")
    ax.set_title("(a) Each cell on its own, ONE query: sum over all keys of |raw score of the pair|, as a share of the same over the 64 pairs",
                 loc="left", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=2)
    pair_axis(ax, labels=False)
    axis_names(ax, 1.09)
    if comparisons:
        kinds = list(dict.fromkeys(r["type"] for r in comparisons))
        if len(kinds) > len(TYPE_COLOURS):
            raise SystemExit(f"at most {len(TYPE_COLOURS)} comparison types")
        offsets = dict(zip(kinds, np.linspace(-.27, .27, len(kinds)) if len(kinds) > 1 else [0]))
        panels = (("absolute", "(b) Is the pair's ABSOLUTE map |raw score| over the keys the same picture in the two cells?  +1 the same, 0 unrelated"),
                  ("signed", "(c) The same for the SIGNED raw score map. Where (b) is high and (c) is not, the pattern is there but its sign differs"))
        for ax, (key, title) in zip(axes[1:3], panels):
            seen = {}
            for r in comparisons:
                second = seen.get(r["type"], 0)
                seen[r["type"]] = second + 1
                colour = TYPE_COLOURS[kinds.index(r["type"])]
                ax.scatter(x + offsets[r["type"]], r[key], s=16 if second == 0 else 20, marker="o" if second == 0 else "D", color=colour if second == 0 else SURFACE,
                           edgecolors=colour, linewidths=.5 if second == 0 else 1.1, zorder=3, label=f"{r['type']}: {r['a']} vs {r['b']}")
            ax.axhline(0, color=INK2, linewidth=.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("correlation over the keys")
            ax.set_title(title, loc="left", fontsize=9.5)
            pair_axis(ax, labels=False)
        axes[1].legend(loc="lower left", frameon=False, fontsize=7.6, ncols=3)
    strip(axes[-1], sets)
    scope = (f"{len(cells)} cell{'s' if len(cells) > 1 else ''}, each {describe(first['meta'])}. ONE query token, {where}, against all {tokens:,} keys, every "
             "key counted once. Raw scores: no attention weighting, no centring, no pooling over queries. Every marker is one pair in one cell (a) "
             "or one pair in one comparison (b, c).")
    fig.subplots_adjust(left=.13, right=.985, bottom=.06 if comparisons else .14)
    finish(fig, "One query, every key: absolute raw score of each pair\n" + "\n".join(textwrap.wrap(scope, 150)), out / "one-query.png")
    plt.close(fig)

    if not args.no_maps:
        from matplotlib.colors import LinearSegmentedColormap
        sequential = LinearSegmentedColormap.from_list("absolute", [NEUTRAL, "#86b6ef", "#2a78d6", "#0d366b"])
        columns = 8
        for label, c in cells.items():
            limit = float(np.abs(c["score"]).max())
            for axis in ("t", "h", "w"):
                members = [f for f in range(pairs) if AXES[f] == axis]
                blocks = -(-(1 + len(members)) // columns)
                fig = plt.figure(figsize=(2.05 * columns + .6, 4.25 * blocks + 2.3))
                spec = fig.add_gridspec(3 * blocks, columns, height_ratios=[10, height, frames * 1.6] * blocks, hspace=.2, wspace=.12)
                slot = lambda i: (fig.add_subplot(spec[3 * (i // columns) + 1, i % columns]), fig.add_subplot(spec[3 * (i // columns) + 2, i % columns]))
                head = map_pair(*slot(0), c["attention"], sequential, 0, float(c["attention"].max()), query, grid, "attention of this query\n(sums to 1 over all keys)\n ", True)
                image = None
                for i, f in enumerate(members, 1):
                    names = [s for s in c["sets"] if c["sets"][s][f]]
                    chosen = "no option" if not names else "all options" if len(names) == len(c["sets"]) > 1 else ", ".join(names)
                    image = map_pair(*slot(i), np.abs(c["score"][f]), sequential, 0, limit, query, grid,
                                     f"{axis} pair {f}\nshare of |raw score|: {100 * c['share'][f]:.2f}%\nselected by: {chosen}", i % columns == 0)
                fig.subplots_adjust(left=.06, right=.985, bottom=1.15 / fig.get_figheight())
                colour_bars(fig, head, image, f"|raw score| the pair adds to the logit of this query and that key (one colour scale for all 64 pairs, 0 to {limit:.2f})")
                scope = (f"ONE query token of ONE cell: {label} = {c['folder']} ({describe(c['meta'])}). Query at {where} (the cross). Top maps: the "
                         f"{height} × {width} keys of the query's own frame. Bottom maps: the {frames} × {width} keys of the query's own row in every frame. The "
                         "horizontal axis of every map is the key column w. Every pixel is the absolute value of one query-key raw score of one pair; "
                         "nothing is pooled. The share in each title is the pair's sum over ALL keys of this query ÷ the same over the 64 pairs.")
                finish(fig, f"Absolute raw score of every {axis} pair across keys, for one query\n" + "\n".join(textwrap.wrap(scope, 150)),
                       out / f"abs-maps-{label}-{axis}.png", .12)
                plt.close(fig)
    print("\n".join(text[:8 + len(cells)]))
    print("WROTE", out)


if __name__ == "__main__":
    main()
