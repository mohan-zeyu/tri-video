"""Individual pair score shares against denoising step, for a few representative heads of each layer.

Reads the per-prompt records of collect_pair_score_attribution.py. For ONE prompt and ONE CFG branch it writes
one page per layer. A page has one panel per selected head; in a panel x = denoising step, y = score share, and
every line is ONE rotary pair of that head. Every point is a single cell value: that prompt, layer, step,
branch, head and pair, with the share pooled over every query x every key of the full 128-dim post-RoPE logits
(pair_score_attribution.py). Pairs are never summed and nothing is averaged over heads, layers, steps, branches
or prompts.

Which heads and pairs are drawn (the selection only decides what is shown; the plotted values are untouched):
  heads   The 12 heads of the layer are ordered by how concentrated they are: the median over the analyzed
          steps of the head's largest pair share. --heads N (default 4) takes N heads evenly spaced through
          that order, from the most to the least concentrated, so sharp and diffuse heads are both shown.
  pairs   Per selected head: the three largest pairs at the first, the middle and the last analyzed step,
          so a pair that only matters early or late is included; if more than --pairs (default 6) remain, those
          with the largest share at any step are kept.

Measure (--measure): adv_share (default; share of the attention-weighted logit advantage of the attended keys
over the mean key), var_cov_share, var_own_share or top1_share. Line colour is the pair's axis (t, h, w); the
line style separates pairs of the same axis. Pairs are named axis + model pair index (t0-t21, h22-h42, w43-w63;
inside an axis a larger index is a lower angular frequency).

Outputs under --out: <prompt>-<branch>-<measure>-layer-NN.png and one CSV per prompt and branch with every
plotted number (layer, head, head_order, pair, step, value).

  python scripts/plot_step_trajectories_pair_score_attribution.py --stats out/distill-score-attribution \
      --out distill-results/score-attribution/step-trajectories --prompts d0000
"""

import argparse

import numpy as np

from analyze_pair_score_attribution import load, resolve, shares

AXES = {"t": (0, 22), "h": (22, 43), "w": (43, 64)}
AXIS_COLOR = {"t": "#2a78d6", "h": "#eb6834", "w": "#1baf7a"}          # validated for all-pairs separation
STYLES = ("-", (0, (5, 1.8)), (0, (1.2, 1.4)), (0, (5, 1.6, 1.2, 1.6)), (0, (8, 2.4)), (0, (3, 1.2, 1, 1.2, 1, 1.2)))
MEASURES = {"adv_share": "share of the attention-weighted logit advantage of the attended keys over the mean key",
            "var_cov_share": "signed share of the row-centred logit variance",
            "var_own_share": "share of the pairs' own contribution variances",
            "top1_share": "share of the arg-max key's logit margin over the mean key"}


def pair_name(pair):
    axis = next(a for a, (lo, hi) in AXES.items() if lo <= pair < hi)
    return axis, f"{axis}{pair}"


def select_heads(x, count):
    """x [step, head, 64] -> [(head, position in the concentration order)], most concentrated first."""
    concentration = np.median(x.max(-1), 0)
    order = np.argsort(-concentration, kind="stable")
    positions = sorted(set(np.round(np.linspace(0, len(order) - 1, min(count, len(order)))).astype(int).tolist()))
    return [(int(order[p]), p) for p in positions]


def select_pairs(x, count):
    """x [step, 64] of one head -> sorted pair indices: top three at the first, middle and last step."""
    steps = sorted({0, (x.shape[0] - 1) // 2, x.shape[0] - 1})
    candidates = sorted({int(p) for s in steps for p in np.argsort(-x[s], kind="stable")[:3]})
    if len(candidates) > count:
        candidates = sorted(sorted(candidates, key=lambda p: -x[:, p].max())[:count])
    return candidates


def draw(panels, steps, heads_total, title, path):
    """panels: [(head, order position, pairs, values [step, len(pairs)])]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, muted, grid, axis_color = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "font.size": 9,
                         "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": axis_color, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid, "grid.linewidth": .8, "axes.axisbelow": True})
    columns = min(2, len(panels))
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.3 * columns + .4, 3.25 * rows + 1.7), squeeze=False, sharex=True)
    x = [s + 1 for s in steps]
    for ax, (head, position, pairs, values) in zip(axes.flat, panels):
        used = {}
        for column, pair in enumerate(pairs):
            axis, name = pair_name(pair)
            style = STYLES[used.setdefault(axis, 0) % len(STYLES)]
            used[axis] += 1
            ax.plot(x, values[:, column], color=AXIS_COLOR[axis], linestyle=style, linewidth=2, marker="o", markersize=3.6,
                    markeredgecolor=surface, markeredgewidth=.7, label=name)
        ax.axhline(1 / 64, color=axis_color, linewidth=1)
        ax.text(x[-1] + .45, 1 / 64, "1/64", fontsize=7.5, color=muted, va="center", ha="left")
        if values.min() < 0:
            ax.axhline(0, color=axis_color, linewidth=1)
        rank = ("most concentrated head" if position == 0 else "least concentrated head" if position == heads_total - 1
                else f"concentration rank {position + 1} of {heads_total}")
        ax.set_title(f"head {head}  ·  {rank}", loc="left", fontsize=9.5)
        ax.set_xlim(x[0] - .5, x[-1] + .5)
        ax.set_ylim(min(0, float(values.min()) * 1.08), max(float(values.max()) * 1.08, 1.6 / 64))
        ax.set_xticks([s for s in x if s == x[0] or s % 5 == 0])
        ax.grid(axis="x", visible=False)
        ax.legend(loc="center left", bbox_to_anchor=(1.045, .5), frameon=False, fontsize=8.5, handlelength=3.2, title="pair", title_fontsize=8.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("denoising step")
    for ax in axes[:, 0]:
        ax.set_ylabel("score share of the pair")
    fig.suptitle(title, x=.012, ha="left", fontsize=9.6, y=.995, va="top", linespacing=1.4)
    fig.subplots_adjust(left=.075, right=.9, top=1 - 1.5 / fig.get_figheight(), bottom=.52 / fig.get_figheight(), wspace=.34, hspace=.24)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-score-attribution")
    parser.add_argument("--out", default="distill-results/score-attribution/step-trajectories")
    parser.add_argument("--prompts", nargs="+", help="prompt ids; default: the first record")
    parser.add_argument("--branches", nargs="+", default=["conditional"])
    parser.add_argument("--layers", nargs="+", type=int, help="default: every layer")
    parser.add_argument("--measure", default="adv_share", choices=list(MEASURES))
    parser.add_argument("--heads", type=int, default=4, help="heads per layer page")
    parser.add_argument("--pairs", type=int, default=6, help="most pairs drawn per head")
    parser.add_argument("--csv-only", action="store_true", help="write the numbers without figures (no matplotlib needed)")
    args = parser.parse_args()
    records, steps = load(args.stats, args.prompts)
    if not args.prompts:
        records = records[:1]
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        share = shares(record)[args.measure][:, steps].astype(np.float64)              # [layer, step, branch, head, 64]
        if not np.isfinite(share).all():
            raise FloatingPointError(f"{args.measure} of {record['prompt_id']}")
        for branch in args.branches:
            b = record["branches"].index(branch)
            stem = f"{record['prompt_id']}-{branch}-{args.measure}"
            lines = ["layer,head,head_order,pair,step,value"]
            for layer in (range(share.shape[0]) if args.layers is None else args.layers):
                x = share[layer, :, b]                                                 # [step, head, 64]
                panels = []
                for head, position in select_heads(x, args.heads):
                    pairs = select_pairs(x[:, head], args.pairs)
                    values = x[:, head][:, pairs]
                    panels.append((head, position, pairs, values))
                    lines += [f"{layer},{head},{position + 1},{pair},{steps[s] + 1},{values[s, c]:.6g}"
                              for c, pair in enumerate(pairs) for s in range(len(steps))]
                if not args.csv_only:
                    title = (f"Layer {layer}: {args.measure} of individual pairs against denoising step, {len(panels)} of the {x.shape[1]} heads\n"
                             f"prompt {record['prompt_id']}, {branch} branch, {record['source']}. {args.measure} = {MEASURES[args.measure]}.\n"
                             "Each point is one cell and one pair (this prompt, layer, step, head): pooled over every query x every key, full "
                             "128-dim post-RoPE logits; no averaging, pairs not summed.\n"
                             "Heads: evenly spaced through the order of the head's largest pair share (median over steps). "
                             "Pairs: the three largest at the first, middle and last step.")
                    draw(panels, steps, x.shape[1], title, out / f"{stem}-layer-{layer:02d}.png")
            (out / f"{stem}.csv").write_text("\n".join(lines) + "\n")
            print("wrote", stem, flush=True)


if __name__ == "__main__":
    main()
