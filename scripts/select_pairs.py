"""Rank rotary pairs by replacement damage and select, per (layer, step, branch, head), which pairs to
replace on the Q side and on the K side.

Input: per-prompt statistics from scripts/collect_pair_statistics.py (calibration prompts). Definitions, for one
cell c = (layer, step, branch, head), side x in {q, k}, pair f, prompt p:

  xbar_p,f          per-prompt coordinate mean of x_f over all 10,800 tokens      (2 coords)
  mu_f              pooled static mean = mean over prompts of xbar_p,f           (what a static student deploys)
  E_p||x_f||^2      per-prompt mean token energy of the pair
  err_pool_p,f      = E_p||x_f - mu_f||^2  = E_p||x_f||^2 - 2 mu_f . xbar_p,f + ||mu_f||^2   (error of the pooled mean)
  err_own_p,f       = E_p||x_f - xbar_p,f||^2 = E_p||x_f||^2 - ||xbar_p,f||^2             (error of the prompt's own mean)
  damage_q,f        = mean_p err_pool_q,p,f  x  mean_p E_p||k_f||^2       (and symmetrically damage_k,f)
  rms_mrl_x,f       = ||mu_f|| / sqrt(mean_p E_p||x_f||^2)               (RMS-normalized concentration of the pooled mean)

The damage is the expected squared logit perturbation (up to 1/sqrt(d)) from replacing that side's pair by the
pooled mean: replacement error energy on one side times the partner side's energy. Lower damage = safer to replace.

Selections (each a bool mask [layers, steps, branches, heads, 64] per side):
  fixed_high   the original third: pairs t 0-7, h 22-28, w 43-49 on both sides
  separate     per cell and side, the --pairs lowest-damage pairs (Q and K chosen independently)
  shared       per cell, the --pairs lowest (damage_q + damage_k) pairs, same set on both sides
  split        per cell, the --means-pairs (default 42) lowest (damage_q + damage_k) pairs are replaced on both
               sides, and the remaining pairs are marked NoPE (kept as content, no rotary embedding): the constant
               pairs then carry all positional signal and the NoPE pairs all content
  wide         widened head: constants on ALL 64 pairs (the full 128-dim mean under RoPE, a position channel) plus
               the --content-pairs (default 32) HIGHEST-damage pairs as an unrotated content channel appended to
               Q and K (LearnedCenter wide mode; V and to_out unchanged)
  pooled       per (layer, head), the --pairs lowest sum over steps and branches of (damage_q + damage_k), same
               set on both sides and at every step and branch: the only kind of selection that can be realised by
               dropping rows of to_q/to_k (narrow mode) instead of masking at run time

Also written: pooled statistics (pair-statistics.pt: full 128-dim pooled means, usable as --calibration), static
learned_center.pt exports per variant, and selection-report.json with cross-prompt stability (Jaccard of each
prompt's own-mean selection with the pooled one; Spearman of per-prompt damage vs pooled damage), the Q/K
intersection size under 'separate', and the frequency composition of the selected sets.
"""

import argparse
import json
from pathlib import Path

import torch

from full_sweep import save_torch, write_json
from learned_center import LearnedCenter, high_pair_mask

ROOT = Path(__file__).resolve().parents[1]
THIRDS = {"t": (range(0, 8), range(8, 15), range(15, 22)), "h": (range(22, 29), range(29, 36), range(36, 43)), "w": (range(43, 50), range(50, 57), range(57, 64))}


def pair_view(x):
    """[..., 128] -> [..., 64, 2]"""
    return x.view(*x.shape[:-1], 64, 2)


def spearman(a, b):
    """Spearman rank correlation along the last dim (no tie handling beyond argsort order)."""
    ra = a.argsort(-1).argsort(-1).double()
    rb = b.argsort(-1).argsort(-1).double()
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def lowest_k_mask(score, k):
    """bool mask of the k smallest entries along the last dim."""
    idx = score.topk(k, dim=-1, largest=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def composition(mask):
    """fraction of selected pairs per axis third; mask [..., 64] bool. Returns dict axis -> (high, mid, low) fractions of the selected set."""
    total = mask.sum().item()
    out = {}
    for axis, thirds in THIRDS.items():
        out[axis] = [round(mask[..., list(r)].sum().item() / max(total, 1), 4) for r in thirds]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-pair-stats", help="directory of per-prompt .pt files")
    parser.add_argument("--out", default="out/distill-selection")
    parser.add_argument("--pairs", type=int, default=22, help="pairs replaced per side per cell (22 = the original third)")
    parser.add_argument("--means-pairs", type=int, default=42, help="'split' variant: pairs replaced by constants per cell; the other 64-N pairs become NoPE")
    parser.add_argument("--content-pairs", type=int, default=32, help="'wide' variant: highest-damage pairs kept as the unrotated content channel")
    parser.add_argument("--exempt-layer0", action="store_true", default=True)
    parser.add_argument("--include-layer0", action="store_true")
    parser.add_argument("--include-step1", action="store_true")
    args = parser.parse_args()
    stats_dir = ROOT / args.stats if not str(args.stats).startswith("/") else Path(args.stats)
    out = ROOT / args.out if not str(args.out).startswith("/") else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in stats_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"no statistics under {stats_dir}")
    records = [torch.load(f, map_location="cpu") for f in files]
    ids = [r["prompt_id"] for r in records]
    means = {s: torch.stack([r[f"mean_{s}"] for r in records]).double() for s in ("q", "k")}      # [P, L, S, B, H, 128]
    energy = {s: torch.stack([r[f"energy_{s}"] for r in records]).double() for s in ("q", "k")}   # [P, L, S, B, H, 64]
    P, L, S, B, H, _ = means["q"].shape
    pooled = {s: means[s].mean(0) for s in ("q", "k")}                                               # [L, S, B, H, 128]
    pooled_energy = {s: energy[s].mean(0) for s in ("q", "k")}
    err_pool, err_own, damage_pooled, damage_own, mrl = {}, {}, {}, {}, {}
    for s, partner in (("q", "k"), ("k", "q")):
        mu = pair_view(pooled[s])                                       # [L,S,B,H,64,2]
        xb = pair_view(means[s])                                        # [P,L,S,B,H,64,2]
        err_pool[s] = (energy[s] - 2 * (mu * xb).sum(-1) + mu.square().sum(-1)).clamp_min(0)      # [P,...,64]
        err_own[s] = (energy[s] - xb.square().sum(-1)).clamp_min(0)
        damage_pooled[s] = err_pool[s].mean(0) * pooled_energy[partner]                            # [L,S,B,H,64]
        damage_own[s] = err_own[s] * energy[partner]                                               # [P,L,S,B,H,64]
        mrl[s] = mu.square().sum(-1).sqrt() / pooled_energy[s].sqrt().clamp_min(1e-12)
    k = args.pairs
    variants = {
        "fixed_high": {side: high_pair_mask().expand(L, S, B, H, 64).clone() for side in ("mask_q", "mask_k")},
        "separate": {"mask_q": lowest_k_mask(damage_pooled["q"], k), "mask_k": lowest_k_mask(damage_pooled["k"], k)},
    }
    shared = lowest_k_mask(damage_pooled["q"] + damage_pooled["k"], k)
    variants["shared"] = {"mask_q": shared, "mask_k": shared.clone()}
    pooled_cell = lowest_k_mask((damage_pooled["q"] + damage_pooled["k"]).sum((1, 2)), k)               # [L, H, 64]
    pooled_mask = pooled_cell[:, None, None].expand(L, S, B, H, 64).clone()
    variants["pooled"] = {"mask_q": pooled_mask, "mask_k": pooled_mask.clone()}
    split_means = lowest_k_mask(damage_pooled["q"] + damage_pooled["k"], args.means_pairs)
    variants["split"] = {"mask_q": split_means, "mask_k": split_means.clone(), "mask_nope": ~split_means}
    content = ~lowest_k_mask(damage_pooled["q"] + damage_pooled["k"], 64 - args.content_pairs)       # the highest-damage pairs
    all_pairs = torch.ones(L, S, B, H, 64, dtype=torch.bool)
    variants["wide"] = {"mask_q": all_pairs, "mask_k": all_pairs.clone(), "mask_nope": content, "wide": True}
    # removal criterion (zeros instead of the pooled mean): energy x partner energy, no mean subtracted
    removal_shared = lowest_k_mask(pooled_energy["q"] * pooled_energy["k"] * 2, k)

    # ---- stability and structure report ---------------------------------------------------------------
    report = {"prompts": P, "ids": ids, "pairs_per_side": k, "cells": L * S * B * H,
              "scope": "per cell = (layer, step, branch, head); statistics over all 10,800 tokens per prompt; pooled over calibration prompts; exemptions are not applied here (masks cover every cell)"}
    for s in ("q", "k"):
        own_sel = lowest_k_mask(damage_own[s], k)                                                  # [P, L,S,B,H,64]
        pooled_sel = variants["separate"][f"mask_{s}"]
        inter = (own_sel & pooled_sel).sum(-1).double()
        jacc = inter / (own_sel | pooled_sel).sum(-1).double()
        rho = spearman(damage_own[s], damage_pooled[s].expand_as(damage_own[s]))
        report[f"stability_{s}"] = {
            "jaccard_prompt_vs_pooled_median": float(jacc.median()), "jaccard_p10": float(jacc.flatten().quantile(0.1)),
            "jaccard_mean": float(jacc.mean()),
            "spearman_damage_median": float(rho.median()), "spearman_p10": float(rho.flatten().quantile(0.1)),
            "scope": "each prompt's own-mean selection/damage vs the pooled selection/damage, per cell; median/p10 over prompts x cells"}
        report[f"composition_{s}_separate"] = composition(pooled_sel)
        report[f"mrl_{s}_selected_mean"] = float(mrl[s][pooled_sel].mean())
        report[f"mrl_{s}_unselected_mean"] = float(mrl[s][~pooled_sel].mean())
        report[f"mrl_{s}_fixed_high_mean"] = float(mrl[s][variants["fixed_high"][f"mask_{s}"]].mean())
        report[f"damage_{s}_selected_over_fixed_high"] = float(damage_pooled[s][pooled_sel].sum() / damage_pooled[s][variants["fixed_high"][f"mask_{s}"]].sum())
    inter = (variants["separate"]["mask_q"] & variants["separate"]["mask_k"]).sum(-1).double()      # [L,S,B,H]
    report["separate_q_k_intersection"] = {"mean_pairs": float(inter.mean()), "median": float(inter.median()),
                                           "p10": float(inter.flatten().quantile(0.1)), "p90": float(inter.flatten().quantile(0.9)),
                                           "per_layer_mean": [round(float(v), 2) for v in inter.mean((1, 2, 3))],
                                           "scope": "number of pairs selected on both Q and K in the same cell, of the --pairs per side"}
    report["composition_shared"] = composition(shared)
    report["composition_pooled"] = composition(pooled_cell)
    report["wide"] = {"content_pairs": args.content_pairs, "composition_content": composition(content),
                      "fixed_high_in_content_fraction": float((content & variants["fixed_high"]["mask_q"]).sum() / variants["fixed_high"]["mask_q"].sum()),
                      "content_vs_split_nope_intersection_mean": float((content & ~split_means).sum(-1).double().mean()),
                      "scope": "per cell; content = the highest-damage pairs by damage_q + damage_k; means cover all 64 pairs"}
    report["split"] = {"means_pairs": args.means_pairs, "nope_pairs": 64 - args.means_pairs,
                       "composition_means": composition(split_means), "composition_nope": composition(~split_means),
                       "shared_in_split_means_fraction": float((split_means & shared).sum() / shared.sum()),
                       "fixed_high_in_nope_fraction": float(((~split_means) & variants["fixed_high"]["mask_q"]).sum() / variants["fixed_high"]["mask_q"].sum()),
                       "scope": "per cell; the split means set is the lowest-damage superset of the shared set when means_pairs >= pairs"}
    report["pooled_vs_shared_intersection"] = {"mean_pairs": float((pooled_mask & shared).sum(-1).double().mean()),
                                               "p10": float((pooled_mask & shared).sum(-1).double().flatten().quantile(0.1)),
                                               "scope": "pairs the step/branch-invariant pooled set shares with the per-cell shared set, per cell, of --pairs"}
    report["shared_vs_removal_criterion_intersection"] = {"mean_pairs": float((removal_shared & shared).sum(-1).double().mean()),
                                                          "scope": "per cell, pairs in common between the shared selection (pooled-mean replacement damage) and the selection by energy x partner energy (zero replacement damage)"}
    report["damage_pooled_over_shared"] = float((damage_pooled["q"][pooled_mask].sum() + damage_pooled["k"][pooled_mask].sum()) / (damage_pooled["q"][shared].sum() + damage_pooled["k"][shared].sum()))
    report["fixed_high_in_separate_q_fraction"] = float((variants["separate"]["mask_q"] & variants["fixed_high"]["mask_q"]).sum() / variants["fixed_high"]["mask_q"].sum())
    report["fixed_high_in_separate_k_fraction"] = float((variants["separate"]["mask_k"] & variants["fixed_high"]["mask_k"]).sum() / variants["fixed_high"]["mask_k"].sum())
    report["damage_total_by_variant"] = {name: float(damage_pooled["q"][v["mask_q"]].sum() + damage_pooled["k"][v["mask_k"]].sum()) for name, v in variants.items()}
    report["per_layer_mean_damage_fixed_over_separate"] = [round(float((damage_pooled["q"][l][variants["fixed_high"]["mask_q"][l]].sum() + damage_pooled["k"][l][variants["fixed_high"]["mask_k"][l]].sum()) /
                                                                     (damage_pooled["q"][l][variants["separate"]["mask_q"][l]].sum() + damage_pooled["k"][l][variants["separate"]["mask_k"][l]].sum()).clamp_min(1e-30)), 3) for l in range(L)]

    # ---- outputs ----------------------------------------------------------------------------------------
    save_torch(out / "pair-statistics.pt", {"mean_q": pooled["q"].float(), "mean_k": pooled["k"].float(),
        "energy_q": pooled_energy["q"].float(), "energy_k": pooled_energy["k"].float(), "prompts": ids,
        "scope": "pooled over calibration prompts of per-prompt all-token statistics; separate layer/step/branch/head; full 128 coordinates"})
    save_torch(out / "selection.pt", {"variants": {n: {k_: (v_.bool() if torch.is_tensor(v_) else v_) for k_, v_ in v.items()} for n, v in variants.items()}, "pairs": k,
        "damage_q": damage_pooled["q"].float(), "damage_k": damage_pooled["k"].float(), "mrl_q": mrl["q"].float(), "mrl_k": mrl["k"].float(),
        "criterion": "lowest damage = mean_p E_p||x_f - mu_f||^2 x mean_p E_p||partner_f||^2 per cell and side", "prompts": ids})
    exempt_layers = () if args.include_layer0 else (0,)
    exempt_steps = () if args.include_step1 else (0,)
    for name, masks in variants.items():
        center = LearnedCenter(layers=L, steps=S, branches=B, heads=H, exempt_layers=exempt_layers, exempt_steps=exempt_steps,
                               mode="mean", selection=masks)
        center.init_from_calibration({"mean_q": pooled["q"].float(), "mean_k": pooled["k"].float()})
        save_torch(out / f"static-{name}" / "learned_center.pt", {"state_dict": center.state_dict(), "mode": "mean", "variant": name})
    write_json(out / "selection-report.json", report)
    print(json.dumps({k_: v_ for k_, v_ in report.items() if k_ not in ("ids",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
