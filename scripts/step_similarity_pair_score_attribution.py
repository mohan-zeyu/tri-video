"""How similar a head's pair score shares are between denoising steps.

Reads the per-prompt records of collect_pair_score_attribution.py. A trajectory is one prompt x layer x CFG
branch x head followed through the analyzed denoising steps; at every step it has one vector of 64 pair shares
(each share pooled over every query x every key of that cell, full 128-dim post-RoPE logits; see
pair_score_attribution.py). For every pair of steps (a, b) of the SAME trajectory:

  pearson      Pearson correlation across the 64 pairs between the share vectors at a and b
  spearman     Spearman rank correlation across the 64 pairs
  moved_share  0.5 * sum_f |share_a[f] - share_b[f]|: the share mass that changes pair (0 = identical vectors)
  top5 / top22 number of pairs common to the 5 (22) largest shares at a and at b

Nothing is averaged over trajectories. Every number reported is a quantile (minimum, 5%, 25%, median, 75%, 95%,
maximum) over the trajectories of a named population: all of them, or those of one layer or one branch.
For scale, the same statistics between two different PROMPTS at the same layer, step, branch and head are in
report.json of analyze_pair_score_attribution.py (cross_prompt_stability).

Outputs under --out: step-similarity.json ([measure][metric]["all"] is [step a, step b, 7 levels];
"by_layer" and "by_branch" are given for the step pairs listed in "step_pairs"), STEP-SIMILARITY.md, and
figures/step-similarity-<measure>.png unless --no-figures.

  python scripts/step_similarity_pair_score_attribution.py --stats out/distill-score-attribution \
      --out distill-results/score-attribution
"""

import argparse
import json

import numpy as np

from analyze_pair_score_attribution import LEVEL_NAMES, dist, json_ready, load, ranks, resolve, shares, top_mask

MEASURES = ("var_cov_share", "var_own_share", "adv_share", "top1_share")
METRICS = ("pearson", "spearman", "moved_share", "top5", "top22")


def correlation(x):
    """x [..., steps, 64] -> [..., steps, steps] Pearson across the last dimension between every two steps."""
    centred = x - x.mean(-1, keepdims=True)
    centred = centred / np.sqrt((centred ** 2).sum(-1, keepdims=True))
    return centred @ np.swapaxes(centred, -1, -2)


def step_matrices(x):
    """x [..., steps, 64] float64 share vectors of each trajectory -> {metric: [..., steps, steps]}."""
    masks = {k: top_mask(x, k).astype(np.float64) for k in (5, 22)}
    return {
        "pearson": correlation(x), "spearman": correlation(ranks(x)),
        "moved_share": .5 * np.abs(x[..., :, None, :] - x[..., None, :, :]).sum(-1),
        "top5": masks[5] @ np.swapaxes(masks[5], -1, -2), "top22": masks[22] @ np.swapaxes(masks[22], -1, -2),
    }


def step_pairs(count):
    """Index pairs reported by layer and by branch: every adjacent transition, then first/middle/last."""
    middle = count // 2 - 1
    extra = [(0, middle), (middle, count - 1), (0, count - 1)]
    return [(a, a + 1) for a in range(count - 1)] + [p for p in extra if p[1] - p[0] > 1]


def build(records, steps):
    pairs = step_pairs(len(steps))
    out = {}
    for measure in MEASURES:
        per_prompt = []
        for record in records:                                                      # one prompt at a time (memory)
            # [layer, branch, head, step, 64]: one row of 64 shares per step of each trajectory
            x = np.moveaxis(shares(record)[measure][:, steps].astype(np.float64), 1, 3)
            if not np.isfinite(x).all():
                raise FloatingPointError(f"{measure} of {record['prompt_id']}")
            per_prompt.append(step_matrices(x))
        out[measure] = {}
        for metric in METRICS:
            m = np.stack([matrices[metric] for matrices in per_prompt])            # [prompt, layer, branch, head, a, b]
            selected = np.stack([m[..., a, b] for a, b in pairs], -1)                # [prompt, layer, branch, head, pair of steps]
            by_layer = dist(np.moveaxis(selected, 1, 0).reshape(m.shape[1], -1, len(pairs)), axis=1)
            by_branch = dist(np.moveaxis(selected, 2, 0).reshape(m.shape[2], -1, len(pairs)), axis=1)
            out[measure][metric] = {"all": dist(m.reshape(-1, *m.shape[-2:]), axis=0).tolist(),
                                    "by_layer": by_layer.tolist(), "by_branch": by_branch.tolist()}
    shape = (len(records), *x.shape[:4])
    return {
        "prompts": [r["prompt_id"] for r in records], "steps_analyzed_zero_based": list(steps),
        "branches": records[0]["branches"], "source": records[0]["source"],
        "trajectories": int(np.prod(shape[:4])),
        "shape": {"prompts": shape[0], "layers": shape[1], "branches": shape[2], "heads": shape[3], "steps": shape[4]},
        "quantile_levels": list(LEVEL_NAMES), "step_pairs": [[steps[a], steps[b]] for a, b in pairs],
        "scope": "trajectory = one prompt x layer x CFG branch x head across the analyzed denoising steps; each step "
                 "contributes that cell's 64 pair shares (pooled over every query x every key, full 128-dim post-RoPE "
                 "logits). Each metric compares the two 64-vectors of one trajectory at two steps. 'all' is "
                 "[step a, step b, level]: quantiles over every trajectory; by_layer is [layer, step pair, level] and "
                 "by_branch [branch, step pair, level] over the trajectories of that layer or branch, for the step "
                 "pairs in step_pairs (zero-based indices). No mean over trajectories is taken anywhere.",
        "measures": out,
    }


def markdown(report):
    s, steps = report["shape"], report["steps_analyzed_zero_based"]
    position = {tuple(p): i for i, p in enumerate(report["step_pairs"])}
    count = len(steps)
    middle = count // 2 - 1
    shown = [p for p in dict.fromkeys([(0, 1), (1, 2), (middle, middle + 1), (count - 2, count - 1), (0, middle), (middle, count - 1), (0, count - 1)])
             if (steps[p[0]], steps[p[1]]) in position]
    text = ["# Similarity of pair score shares between denoising steps\n",
            f"Source: {report['source']}. {report['trajectories']:,} trajectories = {s['prompts']} prompts x {s['layers']} layers x "
            f"{s['branches']} CFG branches x {s['heads']} heads, each followed through {s['steps']} denoising steps. At every step a "
            "trajectory has 64 pair shares, each pooled over every query x every key of that cell (full 128-dim post-RoPE "
            "logits). Each row compares the two 64-vectors of the same trajectory at two steps and gives the quantiles over all "
            "trajectories. Nothing is averaged over trajectories. Steps are numbered from 1. Full step x step tables and the "
            "per-layer and per-branch quantiles are in `step-similarity.json`.\n"]
    names = {"pearson": "Pearson across the 64 pairs", "spearman": "Spearman across the 64 pairs",
             "moved_share": "share mass that changes pair (0.5 x L1 distance)",
             "top5": "pairs in common among the 5 largest", "top22": "pairs in common among the 22 largest"}
    for measure, metrics in report["measures"].items():
        text.append(f"## {measure}\n")
        text.append("| metric | steps | " + " | ".join(LEVEL_NAMES) + " |\n|---|---|" + "---:|" * 7)
        for metric in METRICS:
            table = metrics[metric]["all"]
            for a, b in shown:
                digits = 1 if metric.startswith("top") else 4
                text.append(f"| {names[metric]} | {steps[a] + 1} vs {steps[b] + 1} | " + " | ".join(f"{v:.{digits}f}" for v in table[a][b]) + " |")
        last = position[(steps[0], steps[-1])]
        text.append(f"\nPer layer, steps {steps[0] + 1} vs {steps[-1] + 1}, Pearson across the 64 pairs: quantiles over the "
                    f"{s['prompts'] * s['branches'] * s['heads']:,} trajectories of that layer.\n")
        text.append("| layer | " + " | ".join(LEVEL_NAMES) + " |\n|---:|" + "---:|" * 7)
        for layer, rows in enumerate(metrics["pearson"]["by_layer"]):
            text.append(f"| {layer} | " + " | ".join(f"{v:.4f}" for v in rows[last]) + " |")
        text.append("")
    return "\n".join(text)


def figures(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    surface, ink2 = "#fcfcfb", "#52514e"
    cmap = LinearSegmentedColormap.from_list("blue", [surface, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "font.size": 9,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    out.mkdir(parents=True, exist_ok=True)
    steps = [s + 1 for s in report["steps_analyzed_zero_based"]]
    ticks = [i for i in range(len(steps)) if i == 0 or steps[i] % 5 == 0]
    for measure, metrics in report["measures"].items():
        table = np.array(metrics["pearson"]["all"])                                # [a, b, level]
        low = float(np.floor(table[..., 1].min() * 10) / 10)
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
        for ax, level, name in zip(axes, (1, 2, 3), ("5% quantile", "25% quantile", "median")):
            image = ax.imshow(table[..., level], cmap=cmap, vmin=low, vmax=1, interpolation="nearest")
            ax.set_title(f"{name} over trajectories", loc="left", fontsize=9.5)
            ax.set_xticks(ticks, [steps[i] for i in ticks])
            ax.set_yticks(ticks, [steps[i] for i in ticks])
            ax.set_xlabel("denoising step")
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_ylabel("denoising step")
        bar = fig.colorbar(image, ax=axes.tolist(), fraction=.02, pad=.02)
        bar.outline.set_visible(False)
        bar.set_label("Pearson across the 64 pairs")
        fig.suptitle(f"{measure}: similarity of one head's 64 pair shares between two denoising steps\n"
                     f"{report['trajectories']:,} trajectories (prompt x layer x CFG branch x head); each share pooled over every "
                     "query x every key; quantiles over trajectories, no mean", x=.01, ha="left", fontsize=9.5)
        fig.subplots_adjust(left=.05, right=.9, top=.82, bottom=.12, wspace=.18)
        fig.savefig(out / f"step-similarity-{measure}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-score-attribution")
    parser.add_argument("--ids", nargs="+", help="restrict to these prompt ids")
    parser.add_argument("--out", default="distill-results/score-attribution")
    parser.add_argument("--no-figures", action="store_true", help="skip figures (no matplotlib needed)")
    args = parser.parse_args()
    records, steps = load(args.stats, args.ids)
    if len(steps) < 2:
        raise SystemExit("need at least two analyzed steps")
    report = build(records, steps)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step-similarity.json").write_text(json.dumps(json_ready(report), allow_nan=False) + "\n")
    (out / "STEP-SIMILARITY.md").write_text(markdown(report))
    if not args.no_figures:
        figures(report, out / "figures")
    print(markdown(report)[:2500])


if __name__ == "__main__":
    main()
