"""Figures + summary table for the pre-RoPE Q/K probe results.

Reads out/<model>.json written by probe_video.py; writes out/<model>-*.png and
prints a markdown summary. Colors follow the dataviz reference palette:
sequential blue ramp for heatmaps, categorical slots 1-3 (blue/orange/aqua)
for the three models, blue/orange for the pre/post pair.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CAT = {"wan": "#2a78d6", "ltx": "#eb6834", "cog": "#1baf7a"}
NICE = {"wan": "Wan2.1-1.3B", "ltx": "LTX-Video-2B", "cog": "CogVideoX-2b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 9,
})

MODELS = sys.argv[1:] or ["wan", "ltx", "cog"]
OUT = Path(__file__).resolve().parent.parent / "out"
results = {m: json.loads((OUT / f"{m}.json").read_text()) for m in MODELS}


def run_of(m, tag="t_mid-zeros-s0"):
    return next(x for x in results[m]["runs"] if x["tag"] == tag)


def heads_matrix(m, key):
    """[layers, heads] for the reference run."""
    run = run_of(m)
    layers = sorted(int(k) for k in run["by_layer"])
    return np.array([run["by_layer"][str(l)][key] for l in layers])


def all_heads(m, key, tag="t_mid-zeros-s0"):
    run = run_of(m, tag)
    vals = [l[key] for l in run["by_layer"].values() if key in l]
    if not vals:
        return np.array([])
    return np.concatenate(vals)


def cmap_seq():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ)


# ---------------------------------------------------------------- fig 1: heatmaps
for m in MODELS:
    n = 1 + (len(MODELS) > 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.16 * heads_matrix(m, "q_mrl").shape[0] + 1.6))
    for ax, key, title in zip(axes, ["q_mrl", "k_mrl"],
                              ["Query MRL (pre-RoPE)", "Key MRL (pre-RoPE)"]):
        mat = heads_matrix(m, key)
        im = ax.imshow(mat, aspect="auto", cmap=cmap_seq(), vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_title(f"{NICE[m]} — {title}", fontsize=10, loc="left")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label("MRL = ||E[x]|| / E[||x||]", color=INK2)
    fig.savefig(OUT / f"{m}-mrl-heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{m}-mrl-heatmap.png")

# ---------------------------------------------------------------- fig 2: pre vs post
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=False)
for ax, qk, name in zip(axes, ["q", "k"], ["Query", "Key"]):
    bins = np.linspace(0, 1, 41)
    # two series only: pre (blue) vs post (orange), pooled across rope models
    pre = np.concatenate([all_heads(m, f"{qk}_mrl") for m in MODELS] or [np.array([])])
    post = np.concatenate([all_heads(m, f"{qk}_mrl_post") for m in MODELS if all_heads(m, f"{qk}_mrl_post").size] or [np.array([])])
    ax.hist(pre, bins=bins, color="#2a78d6", alpha=0.85, label="pre-RoPE")
    if post.size:
        ax.hist(post, bins=bins, color="#eb6834", alpha=0.60, label="post-RoPE")
    ax.set_title(f"{name} MRL — video DiT heads", fontsize=10, loc="left")
    ax.set_xlabel("MRL")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("heads")
fig.savefig(OUT / "mrl-pre-vs-post.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote mrl-pre-vs-post.png")

# ---------------------------------------------------------------- fig 3: Q<->K alignment
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bins = np.linspace(-1, 1, 61)
for key, label, color in [("qk_center_cos", "cos(center Q, center K)", "#2a78d6"),
                          ("qk_same_pos_cos", "cos(q_t, k_t) same position", "#eb6834"),
                          ("qk_cross_cos", "E cos(q_t, k_s) all pairs", "#1baf7a")]:
    vals = np.concatenate([all_heads(m, key) for m in MODELS])
    ax.hist(vals, bins=bins, alpha=0.55, label=label, color=color)
ax.set_title("Q↔K directional alignment, pre-RoPE (all video DiT heads)", fontsize=10, loc="left")
ax.set_xlabel("cosine similarity")
ax.set_ylabel("heads")
ax.legend(frameon=False, fontsize=8)
fig.savefig(OUT / "qk-alignment.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote qk-alignment.png")

# ---------------------------------------------------------------- fig 4: per-model summary bars
fig, ax = plt.subplots(figsize=(7.2, 3.0))
labels, pre_q, post_q = [], [], []
for m in MODELS:
    labels.append(NICE[m])
    pre_q.append((all_heads(m, "q_mrl") > 0.95).mean() * 100)
    post_q.append((all_heads(m, "q_mrl_post") > 0.95).mean() * 100)
x = np.arange(len(labels))
ax.bar(x - 0.19, pre_q, width=0.34, color="#2a78d6", label="pre-RoPE")
ax.bar(x + 0.19, post_q, width=0.34, color="#eb6834", label="post-RoPE")
for xi, v in zip(x - 0.19, pre_q):
    ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
ax.axhline(90, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.text(len(labels) - 0.45, 91.5, "Qwen3-8B reference (paper)", fontsize=7.5, color=MUTED, ha="right")
ax.set_xticks(x, labels)
ax.set_ylabel("% heads with query MRL > 0.95")
ax.set_ylim(0, 108)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_title("Share of heads whose queries concentrate around one direction", fontsize=10, loc="left")
fig.savefig(OUT / "frac-concentrated.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote frac-concentrated.png")

# ---------------------------------------------------------------- markdown summary
rows = []
header = ("| model | stat | q MRL pre | k MRL pre | q MRL post | k MRL post | "
          "cos(Q̄,K̄) | E cos(q,k) | %q>0.95 | %k>0.95 |")
rows.append(header)
rows.append("|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    def s(key):
        v = all_heads(m, key)
        return f"{v.mean():.3f} / {np.median(v):.3f}" if v.size else "n/a (no RoPE)"
    rows.append(
        f"| {NICE[m]} ({results[m]['layers']}L×{results[m]['heads']}H, "
        f"T={run_of(m)['tokens']}) | mean/med | {s('q_mrl')} | {s('k_mrl')} | "
        f"{s('q_mrl_post')} | {s('k_mrl_post')} | {s('qk_center_cos')} | "
        f"{s('qk_cross_cos')} | {(all_heads(m,'q_mrl')>0.95).mean()*100:.0f}% | "
        f"{(all_heads(m,'k_mrl')>0.95).mean()*100:.0f}% |")
summary = "\n".join(rows)
print()
print(summary)
(OUT / "summary.md").write_text(summary + "\n")
print("\nwrote out/summary.md")
