"""Is a difference between two prompts caused by the text or by the noise seed? Summary of a 2 x 2 design.

Four cells that differ only in their input: two texts (T0, T6) crossed with two initial noises (N0, N6). Every cell
is the same layer, denoising step, CFG branch and head of the same model (folders written by cell_pair_scores.py).
Six comparisons, each written by compare_cell_pair_scores.py (its two measures are used as they are, nothing is
redefined here):

  same noise, different text   T0N0 vs T6N0,  T0N6 vs T6N6
  same text, different noise   T0N0 vs T0N6,  T6N0 vs T6N6
  both differ                  T0N0 vs T6N6,  T0N6 vs T6N0

Per comparison (the two comparisons of a type are always shown side by side, never averaged):

  share correlation   Pearson correlation across the 64 pairs between the two cells' centred shares
                      (`ctr_att_abs_share`; each share uses every query x every key of its own cell)
  largest difference  of one pair's centred share between the two cells, in percentage points, and which pair
  map correlation     per pair, the median over 27 query tokens of the Pearson correlation over all keys between
                      the pair's score map in the two cells (`map_correlation_median_27_queries`). Its
                      distribution over pairs (min, 5%, 25%, median, 75%, 95%, max; linear interpolation) is given
                      over all 64 pairs, over the pairs whose centred share is above 1/64 in the comparison's first
                      cell, and over the 43 low-frequency pairs t6-t21, h29-h42, w51-w63.

Writes SEED-VS-PROMPT.md, seed-vs-prompt.csv (one row per comparison x pair, every number used) and
seed-vs-prompt.png. It refuses to run if the four cells are not the same layer, step, branch and head or if their
pair sets differ.

  python scripts/seed_vs_prompt.py --cell T0N0 <folder> --cell T6N6 <folder> --cell T0N6 <folder> --cell T6N0 <folder> \
      --compare distill-results/score-attribution-seed/compare --out distill-results/score-attribution-seed
"""

import argparse
import csv
import textwrap

import numpy as np
import torch

from cell_pair_scores import AXES, LEVEL_NAMES, LEVELS, load_cell, resolve
from pair_score_attribution import quantiles
from plot_cell_pair_scores import CENTRED, INK, INK2, MUTED, RAW, SURFACE, axis_names, finish, pair_axis, setup, strip

LABELS = ("T0N0", "T6N6", "T0N6", "T6N0")
COMPARISONS = (("same noise, different text", "T0N0", "T6N0"), ("same noise, different text", "T0N6", "T6N6"),
               ("same text, different noise", "T0N0", "T0N6"), ("same text, different noise", "T6N0", "T6N6"),
               ("both differ", "T0N0", "T6N6"), ("both differ", "T0N6", "T6N0"))
TYPES = ("same noise, different text", "same text, different noise", "both differ")
TYPE_COLOURS = {TYPES[0]: RAW, TYPES[1]: CENTRED, TYPES[2]: "#1baf7a"}     # categorical slots 1, 2, 3
LOW_FREQUENCY = list(range(6, 22)) + list(range(29, 43)) + list(range(51, 64))  # t6-t21, h29-h42, w51-w63
SAME = ("layer", "step", "branch", "head", "tokens")


def distribution(values):
    """The seven quantile levels of `values` (linear interpolation), or None when there are no values."""
    if len(values) == 0:
        return None
    return quantiles(torch.tensor(np.asarray(values, dtype=np.float64)), 0, LEVELS).numpy()


def load(cells, compare):
    info = {}
    for label, folder in cells.items():
        _, sets, meta = load_cell(folder)
        rows = [r for r in csv.DictReader(open(folder / "pair-scores.csv")) if r["row"] == "pair"]
        info[label] = {"folder": folder, "sets": sets, "meta": meta,
                       "centred": np.array([float(r["ctr_att_abs_share"]) for r in rows]),
                       "raw": np.array([float(r["att_abs_share"]) for r in rows])}
    first = info[LABELS[0]]
    for label in LABELS[1:]:
        other = info[label]
        for key in SAME:
            if other["meta"][key] != first["meta"][key]:
                raise SystemExit(f"{label} differs from {LABELS[0]} in {key}: {other['meta'][key]} vs {first['meta'][key]}")
        if sorted(other["sets"]) != sorted(first["sets"]) or not all(np.array_equal(other["sets"][s], first["sets"][s]) for s in first["sets"]):
            raise SystemExit(f"{label} has other pair sets than {LABELS[0]}")
    results = []
    for kind, a, b in COMPARISONS:
        path = compare / f"{info[a]['folder'].name}--vs--{info[b]['folder'].name}.csv"
        rows = list(csv.DictReader(open(path)))
        column = lambda name: np.array([float(r[name]) for r in rows])
        share_a, share_b = column("centred_share_a"), column("centred_share_b")
        # the comparison file must describe the same cells as the four pair-scores.csv (written with 8 significant digits)
        if np.abs(share_a - info[a]["centred"]).max() > 1e-7 or np.abs(share_b - info[b]["centred"]).max() > 1e-7:
            raise SystemExit(f"{path.name} does not match the centred shares of {a} and {b}")
        median = column("map_correlation_median_27_queries")
        strong = np.flatnonzero(share_a > 1 / len(rows))
        difference = np.abs(share_a - share_b)
        results.append({"type": kind, "a": a, "b": b, "file": path.name, "rows": rows, "share_a": share_a, "share_b": share_b,
                        "raw_a": column("raw_share_a"), "raw_b": column("raw_share_b"), "centre": column("map_correlation_centre_query"),
                        "min27": column("map_correlation_min_27_queries"), "median27": median, "max27": column("map_correlation_max_27_queries"),
                        "share_correlation": float(np.corrcoef(share_a, share_b)[0, 1]),
                        "largest_difference": float(difference.max()), "largest_difference_pair": int(difference.argmax()), "strong": strong,
                        "subsets": {"all 64 pairs": distribution(median),
                                    f"pairs above 1/64 in {a} ({len(strong)})": distribution(median[strong]),
                                    f"43 low-frequency pairs": distribution(median[LOW_FREQUENCY])}})
    return info, results


def describe(info):
    meta = info[LABELS[0]]["meta"]
    return (f"layer {meta['layer']}, denoising step {meta['step']}, {meta['branch']} branch, head {meta['head']} (heads count from 0), "
            f"frozen bf16 teacher")


def inputs(info):
    return "; ".join(f"{label} = `{info[label]['folder'].name}`" for label in LABELS)


def write_csv(path, info, results):
    sets = info[LABELS[0]]["sets"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "type", "cell_a", "cell_b", "folder_a", "folder_b", "pair", "axis", *[f"in_{s}" for s in sets],
                         "low_frequency", "above_1_64_in_a", "centred_share_a", "centred_share_b", "centred_share_difference",
                         "raw_share_a", "raw_share_b", "map_correlation_centre_query", "map_correlation_min_27_queries",
                         "map_correlation_median_27_queries", "map_correlation_max_27_queries",
                         "comparison_share_correlation_64_pairs", "comparison_largest_share_difference", "comparison_largest_difference_pair"])
        for index, r in enumerate(results, 1):
            strong = set(r["strong"].tolist())
            for f in range(len(r["rows"])):
                writer.writerow([index, r["type"], r["a"], r["b"], info[r["a"]]["folder"].name, info[r["b"]]["folder"].name, f, AXES[f],
                                 *[int(sets[s][f]) for s in sets], int(f in LOW_FREQUENCY), int(f in strong),
                                 *[f"{v:.8g}" for v in (r["share_a"][f], r["share_b"][f], r["share_a"][f] - r["share_b"][f], r["raw_a"][f],
                                                        r["raw_b"][f], r["centre"][f], r["min27"][f], r["median27"][f], r["max27"][f],
                                                        r["share_correlation"], r["largest_difference"])],
                                 r["largest_difference_pair"]])


def write_markdown(path, info, results):
    sets = info[LABELS[0]]["sets"]
    tokens = info[LABELS[0]]["meta"]["tokens"]
    text = ["# Text or noise seed? Four cells, six comparisons\n",
            f"Four cells, each {describe(info)}: {inputs(info)}. T0 / T6 = the stored text embeddings of prompt d0000 / d0006; "
            "N0 / N6 = the stored step-1 entering state (the initial noise) of d0000 / d0006. Crossed cells use one prompt's text "
            "with the other prompt's noise.\n",
            f"Scope of every number: a centred share uses every one of the {tokens:,} tokens as a query and as a key of ONE cell "
            "(`ctr_att_abs_share`). A map correlation is for ONE query token: the Pearson correlation over all keys between the pair's "
            "score map in the two cells of a comparison; `median of 27` is its median over a 3 × 3 × 3 lattice of 27 query tokens. "
            "Distributions below run over pairs (linear-interpolation quantiles). Nothing is averaged over cells, comparisons or pairs; "
            "the two comparisons of a type stand side by side.\n",
            "## Per comparison\n",
            "| # | type | A vs B | share correlation across 64 pairs | largest difference of one pair's centred share |",
            "|---:|---|---|---:|---:|"]
    for index, r in enumerate(results, 1):
        f = r["largest_difference_pair"]
        text.append(f"| {index} | {r['type']} | {r['a']} vs {r['b']} | {r['share_correlation']:.4f} | "
                    f"{100 * r['largest_difference']:.2f} pp ({AXES[f]}{f}: {100 * r['share_a'][f]:.2f}% vs {100 * r['share_b'][f]:.2f}%) |")
    text += ["", "## Distribution over pairs of the median-of-27-queries map correlation\n",
             "Pairs above 1/64 are those whose centred share is above the equal split in the comparison's first cell (A); the count "
             "is in brackets. Low-frequency pairs: t6–t21, h29–h42, w51–w63 (43 pairs).\n",
             "| # | type | A vs B | pairs | " + " | ".join(LEVEL_NAMES) + " |",
             "|---:|---|---|---|" + "---:|" * len(LEVEL_NAMES)]
    for index, r in enumerate(results, 1):
        for subset, values in r["subsets"].items():
            cells = " | ".join(f"{v:+.3f}" for v in values) if values is not None else " | ".join("—" for _ in LEVEL_NAMES)
            text.append(f"| {index} | {r['type']} | {r['a']} vs {r['b']} | {subset} | {cells} |")
    for index, r in enumerate(results, 1):
        strong = ", ".join(f"{AXES[f]}{f}" for f in r["strong"]) or "none"
        text.append(f"\n- Comparison {index} ({r['a']} vs {r['b']}): pairs above 1/64 in {r['a']}: {strong}.")
    text += ["", "## Per pair: median-of-27-queries map correlation in each comparison\n",
             "| pair | selected by | " + " | ".join(f"{i}. {r['a']} vs {r['b']}" for i, r in enumerate(results, 1)) + " |",
             "|---|---|" + "---:|" * len(results)]
    for f in range(len(results[0]["rows"])):
        chosen = ", ".join(s for s in sets if sets[s][f]) or "—"
        text.append(f"| {AXES[f]}{f} | {chosen} | " + " | ".join(f"{r['median27'][f]:+.3f}" for r in results) + " |")
    text.append("\nColumns 1–2: same noise, different text. 3–4: same text, different noise. 5–6: both differ.")
    path.write_text("\n".join(text) + "\n")


def write_figure(path, info, results):
    plt = setup()
    sets = info[LABELS[0]]["sets"]
    tokens = info[LABELS[0]]["meta"]["tokens"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.8), sharex=True, gridspec_kw={"height_ratios": [3, 3, .22 * len(sets) + .25], "hspace": .2})
    x = np.arange(64)
    ax = axes[0]
    shares = np.stack([info[label]["centred"] for label in LABELS])
    ax.vlines(x, 100 * shares.min(0), 100 * shares.max(0), color=MUTED, linewidth=.8, zorder=1)
    style = {"T0": RAW, "T6": CENTRED, "N0": "o", "N6": "^"}
    for label in LABELS:
        text_, noise = label[:2], label[2:]
        offset = -.14 if noise == "N0" else .14
        ax.scatter(x + offset, 100 * info[label]["centred"], s=22, marker=style[noise], color=style[text_], edgecolors=SURFACE,
                   linewidths=.5, zorder=3, label=f"{label}: text {'d0000' if text_ == 'T0' else 'd0006'}, noise {'d0000' if noise == 'N0' else 'd0006'}")
    ax.axhline(100 / 64, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(63.6, 100 / 64, " equal split 1/64", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylabel("centred share of the absolute score (%)")
    ax.set_title("(a) Each cell on its own: centred share of each pair (colour = text, marker = noise; the thin line spans the four cells). "
                 "Every query × every key", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=2)
    pair_axis(ax, labels=False)
    axis_names(ax, 1.085)
    ax = axes[1]
    offsets = {TYPES[0]: -.27, TYPES[1]: 0, TYPES[2]: .27}
    seen = {}
    for r in results:
        second = seen.get(r["type"], 0)
        seen[r["type"]] = second + 1
        ax.scatter(x + offsets[r["type"]], r["median27"], s=16 if second == 0 else 20, marker="o" if second == 0 else "D",
                   color=TYPE_COLOURS[r["type"]] if second == 0 else SURFACE, edgecolors=TYPE_COLOURS[r["type"]],
                   linewidths=.5 if second == 0 else 1.1, zorder=3, label=f"{r['type']}: {r['a']} vs {r['b']}")
    ax.axhline(0, color=INK2, linewidth=.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("map correlation, median of 27 queries")
    ax.set_title("(b) Per comparison: is the pair's score map over the keys the same picture in the two cells? Median over 27 query tokens; "
                 "+1 the same, 0 unrelated", loc="left", fontsize=9.5)
    ax.legend(loc="lower left", frameon=False, fontsize=7.6, ncols=3)
    pair_axis(ax, labels=False)
    strip(axes[2], sets)
    scope = (f"FOUR cells, each {describe(info)}. T0 / T6 = text embeddings of d0000 / d0006; N0 / N6 = their stored step-1 entering "
             f"states (the initial noise). (a) uses every one of the {tokens:,} tokens as a query and as a key, in each cell separately. "
             "(b) is per comparison: for each of 27 query tokens, the Pearson correlation over all keys between the pair's score map in the "
             "two cells; the marker is the median over those 27 queries. Nothing is averaged over cells, comparisons or pairs.")
    fig.subplots_adjust(left=.13, right=.985, bottom=.06)
    finish(fig, "Text or noise seed: do the pair scores repeat when only the text, only the noise, or both change?\n"
           + "\n".join(textwrap.wrap(scope, 150)), path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", nargs=2, action="append", required=True, metavar=("LABEL", "FOLDER"), help=f"one of {', '.join(LABELS)}")
    parser.add_argument("--compare", default="distill-results/score-attribution-seed/compare")
    parser.add_argument("--out", default="distill-results/score-attribution-seed")
    args = parser.parse_args()
    cells = {label: resolve(folder) for label, folder in args.cell}
    if sorted(cells) != sorted(LABELS):
        raise SystemExit(f"need exactly the cells {', '.join(LABELS)}")
    info, results = load(cells, resolve(args.compare))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed-vs-prompt.csv", info, results)
    write_markdown(out / "SEED-VS-PROMPT.md", info, results)
    write_figure(out / "seed-vs-prompt.png", info, results)
    for index, r in enumerate(results, 1):
        print(f"{index} {r['type']}: {r['a']} vs {r['b']} share correlation {r['share_correlation']:.4f}, "
              f"largest difference {100 * r['largest_difference']:.2f} pp, strong pairs in A {len(r['strong'])}")
    print("WROTE", out / "SEED-VS-PROMPT.md")


if __name__ == "__main__":
    main()
