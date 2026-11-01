"""K-side report: per-band MRL on every (layer, head, band) + logit statistics.

Figure 1 per model (k-band-report-<m>.png):
  (a) the literal ask — heatmap [layer, head x band] of K per-band MRL, every
      cell = one (layer, head, rotary pair); head boundaries ruled off
  (b) K dominant-band MRL, [layer, head]
  (c) K band profile: [layer, band] averaged over heads (Wan axis split marked)

Figure 2 (band-logit-stats.png): band amplitude a_f = ||q_f||*||k_f|| vs
concentration R_f — are the bands that carry the logits the concentrated ones?
  (a) per-model scatter (pooled pairs, log-x amplitude vs MRL) with Pearson r
  (b) top-4 band amplitude share distribution per model

Stats JSON: correlations, amplitude concentration, K-vs-Q asymmetry.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_grad_enabled(False)
OUT = Path(__file__).resolve().parent.parent / "out"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = {"q": "#2a78d6", "k": "#eb6834"}
NICE = {"wan": "Wan2.1-1.3B", "ltx": "LTX-Video-2B", "cog": "CogVideoX-2b"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.grid": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9})

from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)


def load(m):
    d = torch.load(OUT / f"{m}-pair-mrl.pt", map_location="cpu", weights_only=False)
    return (np.array(d["k_pair_mrl"]), np.array(d["q_pair_mrl"]),
            np.array(d["k_pair_norm"]), np.array(d["q_pair_norm"]),
            np.array([[h["k"] for h in row] for row in d["dom_pair_mrl"]]),
            np.array([[h["q"] for h in row] for row in d["dom_pair_mrl"]]))


def fig1(m):
    km, qm, kn, qn, kdom, qdom = load(m)
    L, H, P = km.shape
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.42, wspace=0.22,
                          left=0.06, right=0.965, top=0.90, bottom=0.07)

    # (a) every (layer, head, band) cell for K
    ax = fig.add_subplot(gs[0, :])
    grid = km.transpose(0, 1, 2).reshape(L, H * P)
    im = ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    for h in range(1, H):
        ax.axvline(h * P - 0.5, color=SURF, lw=0.7)
    ax.set_title(f"{NICE[m]} — K per-band MRL: every (layer × head × band) cell "
                 f"[heads ruled off, {P} bands each]", fontsize=10, loc="left")
    ax.set_ylabel("layer")
    ax.set_xlabel(f"head-band index (head 0 … {H-1})")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.008)
    cb.set_label("K band MRL $R_f$", color=INK2)

    # (b) dominant-band MRL per (layer, head), K vs Q
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(kdom, aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("K dominant-band MRL [layer × head]", fontsize=10, loc="left")
    ax.set_ylabel("layer")
    ax.grid(False)

    # (c) band profile averaged over heads
    ax = fig.add_subplot(gs[1, 1])
    prof = km.mean(axis=1)                       # [L, P]
    ax.imshow(prof, aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    if m == "wan":                                # axis split t22/h21/w21 of 64 pairs
        for b, lab in [(22, "t|h"), (43, "h|w")]:
            ax.axvline(b - 0.5, color=INK, lw=0.9)
        ax.text(11, -0.7, "time", ha="center", fontsize=8, color=INK2)
        ax.text(32.5, -0.7, "height", ha="center", fontsize=8, color=INK2)
        ax.text(53.5, -0.7, "width", ha="center", fontsize=8, color=INK2)
    ax.set_title("K band MRL, head-averaged [layer × band]", fontsize=10, loc="left")
    ax.set_xlabel("band index")
    ax.grid(False)
    fig.savefig(OUT / f"k-band-report-{m}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"k-band-report-{m}.png")


def fig2(stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
    ax = axes[0]
    cols = {"wan": "#2a78d6", "ltx": "#eb6834", "cog": "#1baf7a"}
    for m, c in cols.items():
        s = stats[m]
        ax.scatter(s["amp_pct"]["x"], s["corr"]["y"], s=10, alpha=0.30, color=c, label=NICE[m])
        ax.scatter([np.median(s["amp_pct"]["x"])], [np.median(s["corr"]["y"])], s=70,
                   color=c, edgecolor=INK, linewidth=0.8, zorder=5)
    ax.set_xlabel("band amplitude share: top-4 a_f / Σ a_f  (per head)")
    ax.set_ylabel("Pearson r( R_f , a_f ) per head")
    ax.set_title("Do the bands that carry the logits stay concentrated?", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    labels = list(stats)
    for i, m in enumerate(labels):
        s = stats[m]
        ax.scatter([i - 0.12], [s["k_dom_mean"]], s=70, color=CAT["k"], zorder=3)
        ax.scatter([i + 0.12], [s["q_dom_mean"]], s=70, color=CAT["q"], zorder=3)
        ax.vlines(i, s["q_dom_mean"], s["k_dom_mean"], color=GRID, lw=1.2, zorder=1)
    ax.scatter([], [], s=70, color=CAT["k"], label="K dominant-band MRL")
    ax.scatter([], [], s=70, color=CAT["q"], label="Q dominant-band MRL")
    ax.axhline(0.977, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(labels) - 0.45, 0.979, "Qwen3-8B (paper)", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(range(len(labels)), [NICE[m] for m in labels])
    ax.set_ylim(0.85, 1.0)
    ax.set_title("K follows the feature as strongly as Q", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "band-logit-stats.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote band-logit-stats.png")


def main():
    stats = {}
    for m in ["wan", "ltx", "cog"]:
        km, qm, kn, qn, kdom, qdom = load(m)
        L, H, P = km.shape
        amp = qn * kn                                # a_f per (L,H,P)
        total = amp.sum(-1, keepdims=True)
        top4 = np.sort(amp, -1)[:, :, -4:].sum(-1) / total[..., 0]   # [L,H]
        # per-head Pearson r between R_f and a_f across bands (K side)
        a = amp - amp.mean(-1, keepdims=True)
        r_num = (a * (km - km.mean(-1, keepdims=True))).sum(-1)
        r_den = np.sqrt((a * a).sum(-1) * ((km - km.mean(-1, keepdims=True)) ** 2).sum(-1))
        r = r_num / np.maximum(r_den, 1e-9)
        stats[m] = {
            "k_dom_mean": round(float(kdom.mean()), 4),
            "q_dom_mean": round(float(qdom.mean()), 4),
            "k_all_band_mean": round(float(km.mean()), 4),
            "q_all_band_mean": round(float(qm.mean()), 4),
            "k_frac_bands_gt095": round(float((km > 0.95).mean()), 4),
            "top4_amp_share_mean": round(float(top4.mean()), 4),
            "top4_amp_share_median": round(float(np.median(top4)), 4),
            "corr": {"x": top4.reshape(-1).tolist(), "y": r.reshape(-1).tolist(),
                     "mean": round(float(r.mean()), 4), "median": round(float(np.median(r)), 4)},
            "amp_pct": {"x": top4.reshape(-1).tolist()},
            "kdom_by_layer_first_last": [kdom[0].mean().round(4).item(), kdom[-1].mean().round(4).item()],
        }
        fig1(m)
    fig2(stats)
    (OUT / "band-logit-stats.json").write_text(json.dumps(stats, indent=2)[:4000] + "\n...\n")
    slim = {m: {k: v for k, v in s.items() if k not in ("corr", "amp_pct")} |
               {"corr_mean": s["corr"]["mean"], "corr_median": s["corr"]["median"]}
            for m, s in stats.items()}
    (OUT / "band-logit-stats.json").write_text(json.dumps(slim, indent=2) + "\n")
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
