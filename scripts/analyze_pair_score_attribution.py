"""Full-resolution analysis of the per-prompt records of collect_pair_score_attribution.py.

Nothing here is a mean over cells. A cell is one prompt x layer x denoising step x CFG branch x head; its 64 pair
shares come from every query and every key of that cell (pair_score_attribution.py). Every table is a
distribution over a named population of cells: the quantiles LEVELS (minimum, 5%, 25%, median, 75%, 95%,
maximum) or counts. The per-cell values themselves stay in the per-prompt records; `--cell` prints any of them.

Shares derived per cell (64 values each):
  var_cov_share    var_cov / row_var: signed share of the row-centred logit variance (sums to 1)
  var_own_share    var_own / sum_f var_own: share of the pairs' own contribution variances (>= 0, sums to 1)
  adv_share        adv / adv_total: share of the attention-weighted logit advantage over the mean key (sums to 1)
  top1_share       top1 / top1_total: share of the arg-max key's logit margin over the mean key (sums to 1)
  amplitude_share  norm_q * norm_k normalized over the 64 pairs (the sweep's dominant-pair amplitude)

Outputs under --out: report.json (all distributions, each with its scope), REPORT.md (tables), figures/.

  python scripts/analyze_pair_score_attribution.py --stats out/distill-score-attribution \
      --selection out/distill-selection/selection.pt --out distill-results/score-attribution
  python scripts/analyze_pair_score_attribution.py --stats out/distill-score-attribution \
      --cell d0000 15 9 conditional 0          # prompt, layer, 0-based step, branch, head
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
LEVEL_NAMES = ("min", "p05", "p25", "median", "p75", "p95", "max")
SHARES = ("var_cov_share", "var_own_share", "adv_share", "top1_share", "amplitude_share")
AXES = {"t": (0, 22), "h": (22, 43), "w": (43, 64)}
DIMS = {"layer": 1, "step": 2, "branch": 3}                      # axes of [prompt, layer, step, branch, head]
# center_pair_r is a correlation with the center model's pair contribution. For a low-frequency pair whose Q and K
# token means are nearly in phase, that contribution is almost constant over keys: its row variance falls to the
# rounding noise of the stored moments (the centers are rotated in fp32, relative variance noise about 4e-15) and
# can even be stored as a tiny negative number. The correlation is then undefined, not large. It is reported only
# where center_var_own exceeds CENTER_R_FLOOR x (|E q_f| |E k_f|)^2 / d, the largest square the center contribution
# can take; at the floor the rounding noise in r is below 0.01. Undefined entries are NaN and are counted.
CENTER_R_FLOOR = 1e-10


def resolve(path):
    return Path(path) if os.path.isabs(str(path)) else ROOT / path


def shares(record, center_r_floor=CENTER_R_FLOOR):
    """Per-cell shares and derived per-cell quantities of one record, [layer, step, branch, head, ...] (computed in
    float64, kept as float32)."""
    f = {k: v.double().numpy() for k, v in record.items() if torch.is_tensor(v) and k not in ("pair_cov", "filled")}
    amplitude = f["norm_q"] * f["norm_k"]
    own_sum = f["var_own"].sum(-1)
    center_var = f["center_var_cov"].sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):          # cells of steps that were not analyzed are empty
        return {k: v.astype(np.float32) for k, v in _shares(f, amplitude, own_sum, center_var, center_r_floor).items()}


def _shares(f, amplitude, own_sum, center_var, center_r_floor):
    center_square = (f["center_norm_q"] * f["center_norm_k"]) ** 2 / (2 * f["var_own"].shape[-1])
    defined = (f["center_var_own"] > center_r_floor * center_square) & (f["var_own"] > 0)
    return {
        "var_cov_share": f["var_cov"] / f["row_var"][..., None],
        "var_own_share": f["var_own"] / own_sum[..., None],
        "adv_share": f["adv"] / f["adv_total"][..., None],
        "top1_share": f["top1"] / f["top1_total"][..., None],
        "amplitude_share": amplitude / amplitude.sum(-1, keepdims=True),
        "center_pair_r": np.where(defined, f["center_cross_own"] / np.sqrt(np.where(defined, f["var_own"] * f["center_var_own"], 1.0)), np.nan),
        "pair_mrl_min": np.minimum(f["center_norm_q"] / f["norm_q"], f["center_norm_k"] / f["norm_k"]),
        "interference_ratio": f["row_var"] / own_sum,
        "center_logit_r": f["center_cross_cov"].sum(-1) / np.sqrt(f["row_var"] * center_var),
        "center_explained_share": f["center_cross_cov"].sum(-1) / f["row_var"],
        "row_var": f["row_var"], "adv_total": f["adv_total"], "top1_total": f["top1_total"],
        "effective_keys": f["effective_keys"], "pmax": f["pmax"], "top1_self": f["top1_self"],
    }


def load(stats, ids=None):
    files = sorted(p for p in resolve(stats).glob("*.pt") if ids is None or p.stem in ids)
    if not files:
        raise RuntimeError(f"no records under {stats}")
    records = [torch.load(f, map_location="cpu", mmap=True) for f in files]
    steps = records[0]["steps_analyzed"]
    for r in records:
        if r["steps_analyzed"] != steps or r["branches"] != records[0]["branches"]:
            raise RuntimeError("records differ in analyzed steps or branches; analyze them separately")
    return records, steps


def dist(x, axis=None, skip_nan=False):
    """Quantiles LEVELS of x; the levels become the last dimension. skip_nan: over the defined entries only."""
    return np.moveaxis((np.nanquantile if skip_nan else np.quantile)(x, LEVELS, axis=axis), 0, -1)


def grouped(x, dim, per_pair, skip_nan=False):
    """Distribution over all cells sharing one index of `dim`. x: [prompt, layer, step, branch, head(, pair)]."""
    moved = np.moveaxis(x, dim, 0)
    flat = moved.reshape(moved.shape[0], -1, x.shape[-1]) if per_pair else moved.reshape(moved.shape[0], -1)
    return dist(flat, axis=1, skip_nan=skip_nan)


def breakdown(x, per_pair=False, skip_nan=False):
    """Overall and per-layer / per-step / per-branch distributions over cells."""
    overall = dist(x.reshape(-1, x.shape[-1]), axis=0, skip_nan=skip_nan) if per_pair else dist(x, skip_nan=skip_nan)
    return {"all": overall.tolist(), **{f"by_{name}": grouped(x, dim, per_pair, skip_nan).tolist() for name, dim in DIMS.items()}}


def check_finite(data):
    """Every derived quantity must be finite; center_pair_r may be NaN (undefined) but never infinite."""
    for name, value in data.items():
        bad = np.isinf(value) if name == "center_pair_r" else ~np.isfinite(value)
        if bad.any():
            where = [int(i) for i in np.argwhere(bad)[0]]
            raise FloatingPointError(f"{name}: {int(bad.sum())} non-finite entries of {value.size}; first at "
                                     f"[prompt, layer, step, branch, head(, pair)] = {where}")


def json_ready(value):
    """NaN (an undefined quantile of an all-undefined population) becomes null; anything infinite is an error."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def ranks(x):
    return np.argsort(np.argsort(x, -1), -1).astype(np.float64)


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.maximum(np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1)), 1e-12)


def top_mask(x, k):
    index = np.argsort(x, -1)[..., -k:]
    mask = np.zeros(x.shape, dtype=bool)
    np.put_along_axis(mask, index, True, -1)
    return mask


def selection_overlay(records, steps, data, selection, pairs_total=64):
    """Score carried by each selection variant's pairs, per prompt x cell (no pooling over cells)."""
    out = {}
    damage = (selection["damage_q"] + selection["damage_k"]).double().numpy()[:, steps]
    branches = data["var_cov_share"].shape[3]
    for name, masks in selection["variants"].items():
        union = (masks["mask_q"] | masks["mask_k"]).numpy()[:, steps, :branches]            # [L, S, B, H, 64]
        both = (masks["mask_q"] & masks["mask_k"]).numpy()[:, steps, :branches]
        item = {"pairs_in_union": dist(union.sum(-1)).tolist(), "pairs_in_intersection": dist(both.sum(-1)).tolist()}
        for measure in ("var_cov_share", "var_own_share", "adv_share", "top1_share"):
            item[f"union_{measure}"] = breakdown((data[measure] * union).sum(-1))
        if "pair_cov" in records[0]:
            m = torch.from_numpy(union).float()
            deleted = np.stack([torch.einsum("lsbhf,lsbhfg,lsbhg->lsbh", m, r["pair_cov"][:, steps].float(), m).numpy() for r in records])
            item["union_deleted_logit_variance"] = breakdown(deleted)
        out[name] = item
    k = int(selection.get("pairs", 22))
    lowest = np.sort(data["var_own_share"], -1)[..., :k].sum(-1)
    out["reference"] = {
        "lowest_own_share_set": breakdown(lowest), "pairs": k,
        "spearman_damage_vs_var_own": breakdown(spearman(np.broadcast_to(damage[None, :, :, :branches], data["var_own"].shape), data["var_own"])),
        "scope": "per prompt x cell. union = pairs selected on Q or K in that cell (a pair zeroed on either side "
                 "contributes nothing); union_<share> = sum of the cell's pair shares over the union; "
                 "union_deleted_logit_variance = sum_{f,g in union} pair_cov = row-centred variance of the removed "
                 "logit part / row variance of the full logit, every query x key; lowest_own_share_set = the "
                 "smallest value any set of `pairs` pairs can reach in var_own_share; Spearman across the 64 "
                 "pairs of a cell between damage_q + damage_k (select_pairs.py, pooled over its calibration "
                 "prompts) and this prompt's var_own. Selection exemptions (layer 0, step 1) are not applied.",
    }
    return out


def build_report(records, steps, selection=None, center_r_floor=CENTER_R_FLOOR):
    per_prompt = [shares(r, center_r_floor) for r in records]
    data = {k: np.stack([p[k][:, steps] for p in per_prompt]) for k in per_prompt[0]}     # [prompt, layer, step, branch, head, ...]
    data["var_own"] = np.stack([r["var_own"].double().numpy()[:, steps] for r in records])
    check_finite(data)
    n = data["var_cov_share"].shape
    report = {
        "prompts": [r["prompt_id"] for r in records], "steps_analyzed_zero_based": list(steps),
        "branches": records[0]["branches"], "query_stride": records[0]["query_stride"], "source": records[0]["source"],
        "shape": {"prompts": n[0], "layers": n[1], "steps": n[2], "branches": n[3], "heads": n[4], "pairs": n[5]},
        "cells": int(np.prod(n[:5])), "quantile_levels": list(LEVEL_NAMES), "cell_scope": records[0]["scope"],
        "reading": "every array ends with the 7 quantile levels; 'all' pools every prompt x layer x step x branch x "
                   "head cell; by_layer / by_step / by_branch pool the remaining cells; per-pair arrays are "
                   "[..., pair, level]. Distributions are over cells, never means over cells.",
    }
    report["pair_share"] = {m: breakdown(data[m], per_pair=True) for m in SHARES}
    r = data["center_pair_r"]
    undefined = np.isnan(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                           # an all-undefined population gives NaN
        report["pair_center_r"] = breakdown(r, per_pair=True, skip_nan=True)
    report["pair_center_r"].update(
        undefined_entries=int(undefined.sum()), entries=int(r.size),
        undefined_entries_per_pair=undefined.reshape(-1, r.shape[-1]).sum(0).tolist(),
        defined_entries_outside_unit_interval=int((np.abs(r[~undefined]) > 1 + 1e-4).sum()),
        variance_floor=center_r_floor,
        scope="per (cell, pair): pooled row-centred correlation, over every query x every key, between the pair's actual "
              "contribution and its center-model contribution. Undefined (excluded from these quantiles and counted "
              "here) where the center-model contribution's row variance is not above variance_floor x "
              "(|E q_f| |E k_f|)^2 / 128, i.e. within rounding noise of zero; quantiles are over the defined entries")
    own = np.sort(data["var_own_share"], -1)[..., ::-1].cumsum(-1)
    k50, k90 = (own < .5).sum(-1) + 1, (own < .9).sum(-1) + 1
    report["concentration"] = {
        "largest_pair_var_own_share": breakdown(own[..., 0]), "largest_4_pairs_var_own_share": breakdown(own[..., 3]),
        "largest_22_pairs_var_own_share": breakdown(own[..., 21]),
        "pairs_for_50_percent": breakdown(k50), "pairs_for_90_percent": breakdown(k90),
        "pairs_for_50_percent_counts": np.bincount(k50.reshape(-1), minlength=65).tolist(),
        "pairs_for_90_percent_counts": np.bincount(k90.reshape(-1), minlength=65).tolist(),
        "largest_pair_adv_share": breakdown(data["adv_share"].max(-1)),
        "scope": "per cell: the 64 var_own_share values sorted descending and accumulated; counts[k] = number of "
                 "cells needing exactly k pairs",
    }
    report["largest_pair_counts"] = {
        m: {"all": np.bincount(data[m].argmax(-1).reshape(-1), minlength=64).tolist(),
            "by_layer": [np.bincount(data[m][:, layer].argmax(-1).reshape(-1), minlength=64).tolist() for layer in range(n[1])]}
        for m in SHARES}
    report["largest_pair_counts"]["scope"] = "number of cells whose largest share is pair f"
    report["sign"] = {m: {"negative_entries": int((data[m] < 0).sum()), "entries": int(data[m].size),
                          "most_negative": float(data[m].min())} for m in ("var_cov_share", "adv_share", "top1_share")}
    report["head"] = {m: breakdown(data[m]) for m in ("interference_ratio", "center_logit_r", "center_explained_share",
                                                      "row_var", "adv_total", "top1_total", "effective_keys", "pmax", "top1_self")}
    report["head"]["scope"] = ("per cell. interference_ratio = row_var / sum_f var_own (>1: pairs reinforce each other); "
                               "center_logit_r = pooled row-centred correlation of the center-model logit with the actual "
                               "logit over every query x key; center_explained_share = Cov(center logit, actual logit) / "
                               "row_var; effective_keys, pmax, top1_self are means / fractions over the cell's queries")
    amp = data["amplitude_share"]
    report["amplitude"] = {
        "spearman_vs_var_own_share": breakdown(spearman(amp, data["var_own_share"])),
        "spearman_vs_adv_share": breakdown(spearman(amp, data["adv_share"])),
        "cells_largest_amplitude_is_largest_var_own": int((amp.argmax(-1) == data["var_own_share"].argmax(-1)).sum()),
        "cells_largest_amplitude_is_largest_adv": int((amp.argmax(-1) == data["adv_share"].argmax(-1)).sum()),
        "axis_dominant_three_pairs": {m: breakdown(sum(np.take_along_axis(
            data[m][..., a:b], amp[..., a:b].argmax(-1)[..., None], -1)[..., 0] for a, b in AXES.values()))
            for m in ("var_cov_share", "adv_share")},
        "scope": "per cell; Spearman across its 64 pairs; axis_dominant_three_pairs = summed share of the three "
                 "pairs with the largest amplitude inside t, h and w (the pairs of the sweep's scatter figures)",
    }
    edges = np.linspace(0, 1, 21)
    r_edges = np.linspace(-1, 1, 41)
    report["center_r_vs_mrl_counts"] = {
        "counts": np.histogram2d(data["pair_mrl_min"][~undefined], np.clip(r[~undefined], -1, 1), bins=(edges, r_edges))[0].astype(int).tolist(),
        "mrl_edges": edges.tolist(), "r_edges": r_edges.tolist(),
        "scope": "number of (cell, pair) entries with a defined center_pair_r, by min(MRL_Q, MRL_K) of the pair (rows) "
                 "and the pooled row-centred correlation between the pair's actual and center-model contribution (columns)"}
    if n[0] > 1:
        stability = {}
        for m in ("var_own_share", "adv_share"):
            median = np.median(data[m], 0, keepdims=True)
            overlap = top_mask(data[m], 22) & top_mask(np.broadcast_to(median, data[m].shape), 22)
            stability[m] = {
                "spearman_prompt_vs_prompt_median": breakdown(spearman(data[m], np.broadcast_to(median, data[m].shape))),
                "largest_22_overlap_with_prompt_median": breakdown(overlap.sum(-1)),
                "range_over_prompts": dist((data[m].max(0) - data[m].min(0)).reshape(-1, 64), axis=0).tolist()}
        stability["scope"] = ("per layer x step x branch x head cell: each prompt's 64 shares against the per-pair median "
                              "over prompts (Spearman across pairs; number of pairs shared by the two largest-22 sets); "
                              "range_over_prompts[pair] = distribution over cells of max - min over prompts")
        report["cross_prompt_stability"] = stability
    if selection is not None:
        report["selection"] = selection_overlay(records, steps, data, selection)
    return report, data


def row(values, digits=4):
    return " | ".join(f"{v:.{digits}f}" for v in values)


def markdown(report):
    s = report["shape"]
    text = ["# Per-pair attention-score attribution\n",
            f"Source: {report['source']}. {s['prompts']} prompts x {s['layers']} layers x {s['steps']} denoising steps x "
            f"{s['branches']} CFG branches ({', '.join(report['branches'])}) x {s['heads']} heads = {report['cells']:,} cells.\n",
            "A cell's 64 pair shares use the full 128-dim post-RoPE logits of every query x every key of that cell, "
            "relative to each query's mean over keys. Every table is a distribution over cells (minimum, 5%, 25%, "
            "median, 75%, 95%, maximum). Nothing is averaged over cells; definitions are in "
            "`scripts/pair_score_attribution.py`, all per-layer / per-step / per-branch distributions in `report.json`.\n"]
    header = "| pair | axis | " + " | ".join(LEVEL_NAMES) + " |\n|---:|---|" + "---:|" * 7
    for measure in ("var_cov_share", "adv_share", "var_own_share", "top1_share", "amplitude_share"):
        text.append(f"## {measure}: distribution over all {report['cells']:,} cells, per pair\n")
        text.append(header)
        for pair, values in enumerate(report["pair_share"][measure]["all"]):
            axis = next(a for a, (lo, hi) in AXES.items() if lo <= pair < hi)
            text.append(f"| {pair} | {axis} | {row(values)} |")
        text.append("")
    text.append("## Per-cell quantities: distribution over all cells\n")
    text.append("| quantity | " + " | ".join(LEVEL_NAMES) + " |\n|---|" + "---:|" * 7)
    for name, block in (("pairs needed for 50% of own variance", report["concentration"]["pairs_for_50_percent"]),
                        ("pairs needed for 90% of own variance", report["concentration"]["pairs_for_90_percent"]),
                        ("largest pair, var_own_share", report["concentration"]["largest_pair_var_own_share"]),
                        ("largest 4 pairs, var_own_share", report["concentration"]["largest_4_pairs_var_own_share"]),
                        ("largest 22 pairs, var_own_share", report["concentration"]["largest_22_pairs_var_own_share"]),
                        ("three axis-dominant pairs, var_cov_share", report["amplitude"]["axis_dominant_three_pairs"]["var_cov_share"]),
                        ("three axis-dominant pairs, adv_share", report["amplitude"]["axis_dominant_three_pairs"]["adv_share"]),
                        ("Spearman amplitude vs var_own_share", report["amplitude"]["spearman_vs_var_own_share"]),
                        ("interference ratio", report["head"]["interference_ratio"]),
                        ("center-model logit r", report["head"]["center_logit_r"]),
                        ("center-model explained share", report["head"]["center_explained_share"]),
                        ("effective attended keys", report["head"]["effective_keys"]),
                        ("arg-max key is the query itself (fraction of queries)", report["head"]["top1_self"])):
        text.append(f"| {name} | {row(block['all'])} |")
    c = report["pair_center_r"]
    text.append(f"\n`center_pair_r` is undefined for {c['undefined_entries']:,} of {c['entries']:,} (cell, pair) entries: the "
                f"center-model contribution of that pair is constant over keys to within rounding noise (row variance not above "
                f"{c['variance_floor']:g} x its largest possible square). Its quantiles in `report.json` are over the defined "
                f"entries; `undefined_entries_per_pair` gives the count for each pair.\n")
    if "selection" in report:
        text.append("## Score carried by the selected pairs (union of the Q and K masks), per prompt x cell\n")
        text.append("| variant | quantity | " + " | ".join(LEVEL_NAMES) + " |\n|---|---|" + "---:|" * 7)
        for variant, item in report["selection"].items():
            if variant == "reference":
                text.append(f"| reference | smallest possible var_own_share of {item['pairs']} pairs | {row(item['lowest_own_share_set']['all'])} |")
                continue
            for key in ("union_var_cov_share", "union_adv_share", "union_var_own_share", "union_deleted_logit_variance"):
                if key in item:
                    text.append(f"| {variant} | {key} | {row(item[key]['all'])} |")
        text.append("\n" + report["selection"]["reference"]["scope"] + "\n")
    return "\n".join(text)


def figures(report, records, steps, out, atlas_prompts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    surface, ink, ink2, grid, axis_color, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#2a78d6"
    sequential = LinearSegmentedColormap.from_list("blue", [surface, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    diverging = LinearSegmentedColormap.from_list("blue-red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "font.size": 9,
                         "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": axis_color, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid, "grid.linewidth": .8, "axes.axisbelow": True})
    out.mkdir(parents=True, exist_ok=True)
    s = report["shape"]
    scope = (f"{s['prompts']} prompts x {s['layers']} layers x {s['steps']} steps x {s['branches']} branches x {s['heads']} heads = "
             f"{report['cells']:,} cells; each cell: every query x every key, full 128-dim post-RoPE logits")
    for measure in SHARES:
        q = np.array(report["pair_share"][measure]["all"])                                  # [64, 7]
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
        for ax, (name, (a, b)) in zip(axes, AXES.items()):
            x = np.arange(1, b - a + 1)
            ax.fill_between(x, q[a:b, 1], q[a:b, 5], color=blue, alpha=.14, linewidth=0, label="5%–95% of cells")
            ax.fill_between(x, q[a:b, 2], q[a:b, 4], color=blue, alpha=.3, linewidth=0, label="25%–75% of cells")
            ax.plot(x, q[a:b, 3], color=blue, linewidth=2, label="median cell")
            ax.axhline(1 / 64, color=axis_color, linewidth=1)
            ax.set_title(f"{name} axis", loc="left", fontsize=9.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel(measure)
        axes[2].legend(loc="upper right", fontsize=8, frameon=False)
        fig.supxlabel("pair rank inside the axis (1 = highest angular frequency); grey line = 1/64", fontsize=9, color=ink2)
        fig.suptitle(f"{measure}: distribution over cells, per pair (no mean over cells)\n{scope}", x=.01, ha="left", fontsize=9.5)
        fig.subplots_adjust(left=.07, right=.99, top=.83, bottom=.13, wspace=.06)
        fig.savefig(out / f"pair-share-distribution-{measure}.png", dpi=180)
        plt.close(fig)
    for record in records[:atlas_prompts]:
        derived = shares(record)
        for b, branch in enumerate(record["branches"]):
            for measure, cmap in (("var_own_share", sequential), ("adv_share", diverging)):
                x = derived[measure][:, steps, b]                                           # [layer, step, head, pair]
                columns = min(len(steps), 10)
                rows = -(-len(steps) // columns)
                fig, axes = plt.subplots(rows, columns, figsize=(1.9 * columns + .8, 4.6 * rows + .9), squeeze=False, sharex=True, sharey=True)
                limit = float(np.quantile(np.abs(x), .999))
                norm = TwoSlopeNorm(0, -limit, limit) if measure == "adv_share" else plt.Normalize(0, limit)
                for i, ax in enumerate(axes.flat):
                    ax.grid(False)
                    if i >= len(steps):
                        ax.set_visible(False)
                        continue
                    image = ax.imshow(x[:, i].reshape(-1, 64), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                    ax.set_title(f"step {steps[i] + 1}", loc="left", fontsize=9)
                    ax.set_xticks([0, 22, 43], ["t", "h", "w"])
                    ax.set_yticks(np.arange(0, s["layers"] * s["heads"], 5 * s["heads"]), [str(v) for v in range(0, s["layers"], 5)])
                for ax in axes[:, 0]:
                    ax.set_ylabel("layer (each row of pixels = one head)")
                bar = fig.colorbar(image, cax=fig.add_axes([.945, .3, .008, .4]))
                bar.outline.set_visible(False)
                fig.suptitle(f"{measure}, prompt {record['prompt_id']}, {branch} branch: every layer x head (rows) and pair (columns), one panel per "
                             f"denoising step. One pixel = one head's pair share from every query x every key; colour clipped at the 99.9% quantile.",
                             x=.01, ha="left", fontsize=9.5)
                fig.subplots_adjust(left=.04, right=.93, top=.93 if rows > 1 else .88, bottom=.04, wspace=.05, hspace=.08)
                fig.savefig(out / f"atlas-{record['prompt_id']}-{branch}-{measure}.png", dpi=150)
                plt.close(fig)


def print_cell(records, pid, layer, step, branch, head):
    record = next(r for r in records if r["prompt_id"] == pid)
    b = record["branches"].index(branch)
    if not record["filled"][layer, step, b]:
        raise SystemExit("cell was not analyzed")
    derived = shares(record)
    cell = (layer, step, b, head)
    print(f"prompt {pid} layer {layer} step {step + 1} (index {step}) {branch} head {head}: every query x every key of this cell")
    for name in ("row_var", "adv_total", "top1_total", "interference_ratio", "center_logit_r", "effective_keys", "pmax", "top1_self"):
        print(f"  {name} = {float(derived[name][cell]):.6g}")
    print("  share quantiles across queries: levels", record["quantile_levels"])
    print("pair axis  var_cov_share  var_own_share  adv_share  top1_share  amplitude_share  center_r  mrl_min  "
          "var_share[q05 q50 q95]  adv_share[q05 q50 q95]")
    for pair in range(64):
        axis = next(a for a, (lo, hi) in AXES.items() if lo <= pair < hi)
        vq, aq = record["var_share_quantiles"][cell][pair], record["adv_share_quantiles"][cell][pair]
        print(f"{pair:4d} {axis:>4}  " + "  ".join(f"{float(derived[m][cell][pair]):+.5f}" for m in SHARES)
              + f"  {float(derived['center_pair_r'][cell][pair]):+.3f}  {float(derived['pair_mrl_min'][cell][pair]):.3f}"
              + f"  [{float(vq[0]):+.4f} {float(vq[2]):+.4f} {float(vq[4]):+.4f}]  [{float(aq[0]):+.4f} {float(aq[2]):+.4f} {float(aq[4]):+.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-score-attribution")
    parser.add_argument("--ids", nargs="+", help="restrict to these prompt ids")
    parser.add_argument("--selection", help="selection.pt from select_pairs.py, to measure the score its pairs carry")
    parser.add_argument("--out", default="distill-results/score-attribution")
    parser.add_argument("--atlas-prompts", type=int, default=2, help="prompts drawn as full per-cell atlases")
    parser.add_argument("--no-figures", action="store_true", help="skip figures (no matplotlib needed)")
    parser.add_argument("--center-r-floor", type=float, default=CENTER_R_FLOOR,
                        help="relative variance below which a pair's center-model contribution counts as constant")
    parser.add_argument("--cell", nargs=5, metavar=("PROMPT", "LAYER", "STEP", "BRANCH", "HEAD"), help="print one cell and exit")
    args = parser.parse_args()
    records, steps = load(args.stats, args.ids)
    if args.cell:
        print_cell(records, args.cell[0], int(args.cell[1]), int(args.cell[2]), args.cell[3], int(args.cell[4]))
        return
    selection = torch.load(resolve(args.selection), map_location="cpu") if args.selection else None
    report, _ = build_report(records, steps, selection, args.center_r_floor)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(json_ready(report), allow_nan=False) + "\n")
    (out / "REPORT.md").write_text(markdown(report))
    if not args.no_figures:
        figures(report, records, steps, out / "figures", args.atlas_prompts)
    print(markdown(report)[:3000])


if __name__ == "__main__":
    main()
