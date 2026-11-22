"""Per-head score share against denoising step, one panel per layer.

Reads the per-prompt records of collect_pair_score_attribution.py and draws, for ONE prompt and ONE CFG branch,
a grid with one panel per layer: x = denoising step, y = the share of the attention score carried by a set of
rotary pairs, one line per head. Every point is one cell (that prompt, layer, step, branch, head): the cell's
pair shares (pooled over every query x every key, full 128-dim post-RoPE logits; pair_score_attribution.py)
summed over the pairs of the set. Nothing is averaged over heads, layers, steps, branches or prompts.

Pair sets (--sets):
  high     the 22 highest-frequency pairs (t 0-7, h 22-28, w 43-49)
  mid      t 8-14, h 29-35, w 50-56        low      t 15-21, h 36-42, w 57-63
  t, h, w  all pairs of one axis (0-21, 22-42, 43-63)
  largest  the single largest pair share of the cell (the pair may differ between cells)
  pairs:22,26   any explicit list of pair indices

Measure (--measure): adv_share (default; share of the attention-weighted logit advantage of the attended keys
over the mean key), var_cov_share, var_own_share or top1_share.

Outputs under --out, per prompt, branch and set: a PNG and a CSV (layer, head, step, value) with the plotted
numbers.

  python scripts/plot_step_trajectories_pair_score_attribution.py --stats out/distill-score-attribution \
      --out distill-results/score-attribution/step-trajectories --prompts d0000
"""

import argparse

import numpy as np

from analyze_pair_score_attribution import load, resolve, shares

THIRDS = {"high": (range(0, 8), range(22, 29), range(43, 50)), "mid": (range(8, 15), range(29, 36), range(50, 57)),
          "low": (range(15, 22), range(36, 43), range(57, 64))}
AXES = {"t": range(0, 22), "h": range(22, 43), "w": range(43, 64)}
MEASURES = {"adv_share": "share of the attention-weighted logit advantage over the mean key",
            "var_cov_share": "signed share of the row-centred logit variance",
            "var_own_share": "share of the pairs' own contribution variances",
            "top1_share": "share of the arg-max key's logit margin over the mean key"}
# eight validated categorical hues; heads beyond eight reuse the first hues with a dashed line (hue x dash)
HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def pair_set(name):
    """-> (pair indices or None for 'largest', description)."""
    if name in THIRDS:
        t, h, w = THIRDS[name]
        return [*t, *h, *w], (f"the {name}-frequency third: t {t.start}-{t.stop - 1}, h {h.start}-{h.stop - 1}, "
                              f"w {w.start}-{w.stop - 1}")
    if name in AXES:
        return list(AXES[name]), f"all {name}-axis pairs ({AXES[name].start}-{AXES[name].stop - 1})"
    if name == "largest":
        return None, "the cell's single largest pair (the pair may differ between cells)"
    if name.startswith("pairs:"):
        ids = sorted({int(v) for v in name[6:].split(",")})
        if not ids or min(ids) < 0 or max(ids) > 63:
            raise SystemExit(f"bad pair list {name}")
        return ids, "pair" + ("s " if len(ids) > 1 else " ") + ", ".join(map(str, ids))
    raise SystemExit(f"unknown set {name}")


def values(share, ids):
    """share [layer, step, branch, head, 64] -> [layer, step, branch, head]."""
    return share.max(-1) if ids is None else share[..., ids].sum(-1)


def draw(v, steps, title, path):
    """v [layer, step, head]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid, axis_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "font.size": 8.5,
                         "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": axis_color, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid, "grid.linewidth": .7, "axes.axisbelow": True})
    layers, _, heads = v.shape
    columns = min(6, layers)
    rows = -(-layers // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.75 * columns + .6, 2.05 * rows + 1.5), sharex=True, sharey=True, squeeze=False)
    x = [s + 1 for s in steps]
    low, high = float(v.min()), float(v.max())
    pad = .04 * max(high - low, 1e-6)
    for layer, ax in enumerate(axes.flat):
        if layer >= layers:
            ax.set_visible(False)
            continue
        for head in range(heads):
            ax.plot(x, v[layer, :, head], color=HUES[head % len(HUES)], linewidth=1.5,
                    linestyle="-" if head < len(HUES) else (0, (4, 1.6)), label=f"head {head}")
        if low < 0 < high:
            ax.axhline(0, color=axis_color, linewidth=.9)
        ax.set_title(f"layer {layer}", loc="left", fontsize=9, pad=3)
        ax.set_xlim(x[0] - .6, x[-1] + .6)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks([s for s in x if s == x[0] or s % 5 == 0])
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(heads, 12), frameon=False, fontsize=8.5, handlelength=2.2,
               columnspacing=1.3, bbox_to_anchor=(.5, .002))
    fig.supxlabel("denoising step", fontsize=9.5, color=ink2, y=.58 / fig.get_figheight())
    fig.supylabel("share of the head's attention score", fontsize=9.5, color=ink2, x=.006)
    fig.suptitle(title, x=.012, ha="left", fontsize=9.8, y=.995, va="top", linespacing=1.4)
    top = 1 - 1.05 / fig.get_figheight()
    fig.subplots_adjust(left=.05, right=.995, top=top, bottom=1.05 / fig.get_figheight(), wspace=.08, hspace=.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-score-attribution")
    parser.add_argument("--out", default="distill-results/score-attribution/step-trajectories")
    parser.add_argument("--prompts", nargs="+", help="prompt ids; default: the first record")
    parser.add_argument("--branches", nargs="+", help="default: every branch in the records")
    parser.add_argument("--measure", default="adv_share", choices=list(MEASURES))
    parser.add_argument("--sets", nargs="+", default=["high", "t", "h", "w", "largest"])
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
        for branch in args.branches or record["branches"]:
            b = record["branches"].index(branch)
            for name in args.sets:
                ids, description = pair_set(name)
                v = values(share, ids)[:, :, b]                                        # [layer, step, head]
                stem = f"{record['prompt_id']}-{branch}-{args.measure}-{name.replace(':', '-').replace(',', '_')}"
                lines = ["layer,head,step,value"] + [f"{layer},{head},{steps[s] + 1},{v[layer, s, head]:.6g}"
                                                     for layer in range(v.shape[0]) for head in range(v.shape[2]) for s in range(v.shape[1])]
                (out / f"{stem}.csv").write_text("\n".join(lines) + "\n")
                if not args.csv_only:
                    title = (f"{args.measure} carried by {description}, per head, against denoising step\n"
                             f"prompt {record['prompt_id']}, {branch} branch, {record['source']}. {args.measure} = {MEASURES[args.measure]}.\n"
                             "Each point is one cell (this prompt, layer, step, head): its pair shares, pooled over every query x "
                             "every key of the full 128-dim post-RoPE logits, summed over the set. No averaging over heads, layers, "
                             "steps or prompts.")
                    draw(v, steps, title, out / f"{stem}.png")
                print("wrote", stem, flush=True)


if __name__ == "__main__":
    main()
