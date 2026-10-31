"""Recompute the FULL per-rotary-pair MRL data from stored subsample tensors.

The probe JSONs only stored the mean over pairs (q_pair_mrl_mean). This script
rebuilds, for every (layer, head): the per-pair MRL vector R_f (f = 0..D/2-1),
the dominant-pair MRL (paper's selection: top band by ||q_f||*||k_f||), and
per-axis aggregations where the rope layout defines them.

  Wan:  head_dim 128 -> 64 interleaved pairs; axis split of the 64 pairs is
        t: 22, h: 21, w: 21 (WanRotaryPosEmbed: t_dim=44, h_dim=w_dim=42)
  LTX:  rope runs on the flattened C=H*D stream -> 1024 pairs total,
        t/h/w blocks of 341 (+1 pad pair); each head's 32 pairs carve a
        contiguous slice, so heads land on different axes
  Cog:  no RoPE; pairs are still computed as geometric 2D pairs (no axis map)

Source: out/<m>-tensors.pt -> runs[tag][layer]["_subsample_q"/"_subsample_k"],
bf16 [H, 512, D] — estimates from a 512-token subsample of T.

Usage: python pair_detail.py wan ltx cog   (writes out/<m>-pair-mrl.pt + .json)
"""
import json
import sys
from pathlib import Path

import torch

torch.set_grad_enabled(False)
OUT = Path(__file__).resolve().parent.parent / "out"


def pair_mrl_full(x):
    """x: [H, n, D] float -> per-pair MRL [H, P] and per-pair norms [H, P]."""
    z = x.unflatten(-1, (-1, 2))                       # [H, n, P, 2]
    mean = z.mean(1)                                   # [H, P, 2]
    norm = z.norm(dim=-1)                              # [H, n, P]
    mrl = mean.norm(dim=-1) / norm.mean(1).clamp_min(1e-12)
    return mrl, norm.mean(1)                           # [H, P], [H, P]


def axis_of_pairs(which, H, P):
    """Map pair index -> axis label, per head where the mapping varies."""
    if which == "wan":
        # same split for every head: t=0..21, h=22..42, w=43..63
        ax = ["t"] * 22 + ["h"] * 21 + ["w"] * 21
        return {h: ax for h in range(H)}
    if which == "ltx":
        # flattened stream: 1024 pairs = t 0..340, h 341..681, w 682..1022, pad 1023
        total = H * P
        per = total // 3 if total % 3 == 0 else total // 3
        labels = ["t"] * per + ["h"] * per + ["w"] * (total - 2 * per - 1) + ["pad"]
        # head h owns flattened pairs [h*P, (h+1)*P)
        return {h: labels[h * P:(h + 1) * P] for h in range(H)}
    return {h: ["none"] * P for h in range(H)}          # cog: no rope axes


def process(which):
    tensors = torch.load(OUT / f"{which}-tensors.pt", map_location="cpu", weights_only=False)
    tag = "t_mid-zeros-s0"
    layers = sorted(tensors[tag].keys(), key=int)
    H, D = tensors[tag][layers[0]]["_subsample_q"].shape[0], tensors[tag][layers[0]]["_subsample_q"].shape[2]
    P = D // 2
    axis_map = axis_of_pairs(which, H, P)

    out = {"layers": [int(l) for l in layers], "heads": H, "pairs": P,
           "q_pair_mrl": [], "k_pair_mrl": [],         # [L][H][P]
           "q_pair_norm": [], "k_pair_norm": [],
           "dom_pair_mrl": [], "dom_pair_idx": [],     # [L][H] paper-style
           "per_axis_mrl": []}                          # [L][H] {axis: mean R}
    for l in layers:
        q = tensors[tag][l]["_subsample_q"].float()
        k = tensors[tag][l]["_subsample_k"].float()
        mq, nq = pair_mrl_full(q)
        mk, nk = pair_mrl_full(k)
        imp = nq * nk                                    # band importance
        dom = imp.argmax(-1)                             # [H]
        out["q_pair_mrl"].append(mq.tolist())
        out["k_pair_mrl"].append(mk.tolist())
        out["q_pair_norm"].append(nq.tolist())
        out["k_pair_norm"].append(nk.tolist())
        out["dom_pair_mrl"].append([{"q": qv, "k": kv} for qv, kv in zip(
            mq.gather(-1, dom[:, None]).squeeze(-1).tolist(),
            mk.gather(-1, dom[:, None]).squeeze(-1).tolist())])
        out["dom_pair_idx"].append(dom.tolist())
        ax_rows = []
        for h in range(H):
            agg = {}
            for ax in ("t", "h", "w", "pad", "none"):
                idx = [i for i, a in enumerate(axis_map[h]) if a == ax]
                if idx:
                    agg[ax] = {"q": float(mq[h, idx].mean()), "k": float(mk[h, idx].mean()),
                               "count": len(idx)}
            ax_rows.append(agg)
        out["per_axis_mrl"].append(ax_rows)

    # headline summaries
    import numpy as np
    allq = np.array(out["q_pair_mrl"])                  # [L, H, P]
    allk = np.array(out["k_pair_mrl"])
    domq = np.array([[d["q"] for d in row] for row in out["dom_pair_mrl"]])
    domk = np.array([[d["k"] for d in row] for row in out["dom_pair_mrl"]])
    summ = {
        "model": which, "source": f"{tag} 512-token subsamples, bf16",
        "q_pair_mrl_mean_over_all": round(float(allq.mean()), 4),
        "k_pair_mrl_mean_over_all": round(float(allk.mean()), 4),
        "q_pair_mrl_median": round(float(np.median(allq)), 4),
        "q_frac>0.95_allpairs": round(float((allq > 0.95).mean()), 4),
        "q_frac>0.9_allpairs": round(float((allq > 0.9).mean()), 4),
        "dom_q_mrl_mean": round(float(domq.mean()), 4),
        "dom_q_mrl_median": round(float(np.median(domq)), 4),
        "dom_q_frac>0.95": round(float((domq > 0.95).mean()), 4),
        "dom_q_frac>0.9": round(float((domq > 0.9).mean()), 4),
        "dom_k_mrl_mean": round(float(domk.mean()), 4),
        "dom_k_frac>0.95": round(float((domk > 0.95).mean()), 4),
        "per_axis_mean": {},
    }
    for ax in ("t", "h", "w", "pad", "none"):
        vals_q, vals_k = [], []
        for lrow in out["per_axis_mrl"]:
            for hagg in lrow:
                if ax in hagg:
                    vals_q.append(hagg[ax]["q"]); vals_k.append(hagg[ax]["k"])
        if vals_q:
            summ["per_axis_mean"][ax] = {"q": round(float(np.mean(vals_q)), 4),
                                         "k": round(float(np.mean(vals_k)), 4)}
    torch.save(out, OUT / f"{which}-pair-mrl.pt")
    (OUT / f"{which}-pair-mrl-summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["wan", "ltx", "cog"]:
        process(m)
