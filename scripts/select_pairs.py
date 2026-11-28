"""Rank rotary pairs and select, per (layer, step, branch, head), which pairs get which role (means, zeros at
inference, unrotated content), on the Q side and on the K side.

Input: per-prompt statistics from scripts/collect_pair_statistics.py (calibration prompts). Definitions, for one
cell c = (layer, step, branch, head), side x in {q, k}, pair f, prompt p:

  xbar_p,f          per-prompt coordinate mean of x_f over all 10,800 tokens      (2 coords)
  mu_f              pooled static mean = mean over prompts of xbar_p,f           (what a static student deploys)
  E_p||x_f||^2      per-prompt mean token energy of the pair
  err_pool_p,f      = E_p||x_f - mu_f||^2  = E_p||x_f||^2 - 2 mu_f . xbar_p,f + ||mu_f||^2   (error of the pooled mean)
  err_own_p,f       = E_p||x_f - xbar_p,f||^2 = E_p||x_f||^2 - ||xbar_p,f||^2             (error of the prompt's own mean)
  damage_q,f        = mean_p err_pool_q,p,f  x  mean_p E_p||k_f||^2       (and symmetrically damage_k,f)
  rms_mrl_x,f       = ||mu_f|| / sqrt(mean_p E_p||x_f||^2)               (RMS-normalized concentration of the pooled mean)

The policies (criterion, combine, pool, pairs, rank, role) live in scripts/pair_selection.py. The historical
variants are presets there and are always written:
  fixed_high   the original third: pairs t 0-7, h 22-28, w 43-49 on both sides
  separate     per cell and side, the --pairs lowest-damage pairs (Q and K chosen independently)
  shared       per cell, the --pairs lowest (damage_q + damage_k) pairs, same set on both sides
  pooled       per (layer, head), the --pairs lowest sum over steps and branches of (damage_q + damage_k)
  split        the --means-pairs lowest-damage pairs -> means; the rest -> content without RoPE (NoPE)
  wide         widened head: means on all 64 pairs + the --content-pairs highest-damage pairs as unrotated content
--spec adds a variant named "policy" from a selection spec (JSON file or inline JSON; see pair_selection.py), e.g.
  '{"policies": [{"criterion": "error", "combine": "shared", "pairs": 16, "role": "mean"}]}'

Also written: pooled statistics (pair-statistics.pt: full 128-dim pooled means, usable as --calibration), static
learned_center.pt exports per variant (exemptions from --bypass or --include-*), and selection-report.json with
cross-prompt stability (Jaccard of each prompt's own-mean selection with the pooled one; Spearman of per-prompt score
vs pooled score, for the policy variant's criterion as well as for damage), the Q/K intersection size under
'separate', composition by frequency third, and overlaps between variants.
"""

import argparse
import json
from pathlib import Path

import torch

from full_sweep import save_torch, write_json
from learned_center import LearnedCenter
from pair_selection import PRESETS, PooledStatistics, build_selection, describe, high_pair_mask, lowest_k_mask, normalize_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]
THIRDS = {"t": (range(0, 8), range(8, 15), range(15, 22)), "h": (range(22, 29), range(29, 36), range(36, 43)), "w": (range(43, 50), range(50, 57), range(57, 64))}


def spearman(a, b):
    """Spearman rank correlation along the last dim (no tie handling beyond argsort order)."""
    ra = a.argsort(-1).argsort(-1).double()
    rb = b.argsort(-1).argsort(-1).double()
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def composition(mask):
    """fraction of selected pairs per axis third; mask [..., 64] bool. Returns dict axis -> (high, mid, low) fractions of the selected set."""
    total = mask.sum().item()
    out = {}
    for axis, thirds in THIRDS.items():
        out[axis] = [round(mask[..., list(r)].sum().item() / max(total, 1), 4) for r in thirds]
    return out


def stability(stats, side, criterion, pairs, rank, pooled_mask):
    """Each prompt's own-mean selection vs the pooled selection: Jaccard and Spearman, per cell; median/p10 over prompts x cells."""
    own_score = stats.per_prompt_score(side, criterion)
    if own_score is None:
        return None
    own_sel = lowest_k_mask(own_score, pairs, largest=(rank == "highest"))
    inter = (own_sel & pooled_mask).sum(-1).double()
    jacc = inter / (own_sel | pooled_mask).sum(-1).double()
    rho = spearman(own_score, stats.side_score(side, criterion).expand_as(own_score))
    return {"jaccard_prompt_vs_pooled_median": float(jacc.median()), "jaccard_p10": float(jacc.flatten().quantile(0.1)),
            "jaccard_mean": float(jacc.mean()), "spearman_score_median": float(rho.median()), "spearman_p10": float(rho.flatten().quantile(0.1)),
            "scope": "each prompt's own-mean selection/score vs the pooled selection/score, per cell; median/p10 over prompts x cells"}


def parse_bypass(text):
    if not text:
        return {}
    b = json.loads(Path(text).read_text()) if Path(text).exists() else json.loads(text)
    return b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", default="out/distill-pair-stats", help="directory of per-prompt .pt files")
    parser.add_argument("--out", default="out/distill-selection")
    parser.add_argument("--pairs", type=int, default=22, help="pairs replaced per side per cell for shared/separate/pooled (22 = the original third)")
    parser.add_argument("--means-pairs", type=int, default=42, help="'split' variant: pairs replaced by constants per cell; the other 64-N pairs become NoPE")
    parser.add_argument("--content-pairs", type=int, default=32, help="'wide' variant: highest-damage pairs kept as the unrotated content channel")
    parser.add_argument("--spec", help="selection spec (JSON file or inline JSON) written as variant 'policy'")
    parser.add_argument("--bypass", help="bypass rules (JSON file or inline JSON) for the static exports, e.g. '{\"layers\": [0], \"steps\": [0]}'")
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
    stats = PooledStatistics(means, energy)
    P, (L, S, B, H) = stats.P, stats.shape
    damage_pooled = {s: stats.side_score(s, "damage") for s in ("q", "k")}
    k = args.pairs

    # ---- presets with the command-line sizes, plus the optional policy variant --------------------------------
    specs = {}
    for name, preset in PRESETS.items():
        spec = json.loads(json.dumps(preset))
        size = {"split": args.means_pairs, "wide": args.content_pairs}.get(name, k)
        for p in spec["policies"]:
            if "complement_of" in p or p["criterion"] in ("all", "fixed_high"):
                continue
            p["pairs"] = size
        specs[name] = spec
    if args.spec:
        specs["policy"] = normalize_spec(json.loads(Path(args.spec).read_text()) if Path(args.spec).exists() else json.loads(args.spec))
    built = {name: build_selection(stats, spec) for name, spec in specs.items()}
    variants = {name: {"mask_q": b["mask_q"], "mask_k": b["mask_k"], "mask_nope": b["mask_nope"], "wide": b["wide"],
                       "spec": b["spec"], "spec_hash": spec_hash(b["spec"]), "description": describe(b["spec"])}
                for name, b in built.items()}
    shared = variants["shared"]["mask_q"]
    fixed = variants["fixed_high"]["mask_q"]

    # ---- stability and structure report ---------------------------------------------------------------
    report = {"prompts": P, "ids": ids, "pairs_per_side": k, "cells": L * S * B * H,
              "variants": {n: {"description": v["description"], "spec_hash": v["spec_hash"], "spec": v["spec"],
                               "mean_pairs_per_cell_q": float(v["mask_q"].sum(-1).double().mean()), "mean_pairs_per_cell_k": float(v["mask_k"].sum(-1).double().mean()),
                               "nope_pairs_per_cell": float(v["mask_nope"].sum(-1).double().mean()), "wide": v["wide"],
                               "composition_mean_q": composition(v["mask_q"]), "composition_nope": composition(v["mask_nope"]),
                               "shared_pairs_in_mean_q": float((v["mask_q"] & shared).sum(-1).double().mean()),
                               "fixed_high_in_mean_q_fraction": float((v["mask_q"] & fixed).sum() / fixed.sum()),
                               "mrl_q_mean_pairs": float(stats.mrl["q"][v["mask_q"]].mean()) if v["mask_q"].any() else None,
                               "energy_q_mean_pairs_over_all": float(stats.pooled_energy["q"][v["mask_q"]].mean() / stats.pooled_energy["q"].mean()) if v["mask_q"].any() else None,
                               "damage_q_total": float(damage_pooled["q"][v["mask_q"]].sum() + damage_pooled["k"][v["mask_k"]].sum())}
                           for n, v in variants.items()},
              "scope": "per cell = (layer, step, branch, head); statistics over all 10,800 tokens per prompt; pooled over calibration prompts; exemptions are not applied here (masks cover every cell); energy ratios are the selected pairs' pooled energy over the all-pair pooled energy"}
    sep = variants["separate"]
    for s in ("q", "k"):
        report[f"stability_{s}"] = stability(stats, s, "damage", k, "lowest", sep[f"mask_{s}"])
        report[f"composition_{s}_separate"] = composition(sep[f"mask_{s}"])
        report[f"mrl_{s}_selected_mean"] = float(stats.mrl[s][sep[f"mask_{s}"]].mean())
        report[f"mrl_{s}_unselected_mean"] = float(stats.mrl[s][~sep[f"mask_{s}"]].mean())
        report[f"mrl_{s}_fixed_high_mean"] = float(stats.mrl[s][fixed].mean())
        report[f"damage_{s}_selected_over_fixed_high"] = float(damage_pooled[s][sep[f"mask_{s}"]].sum() / damage_pooled[s][fixed].sum())
    inter = (sep["mask_q"] & sep["mask_k"]).sum(-1).double()      # [L,S,B,H]
    report["separate_q_k_intersection"] = {"mean_pairs": float(inter.mean()), "median": float(inter.median()),
                                           "p10": float(inter.flatten().quantile(0.1)), "p90": float(inter.flatten().quantile(0.9)),
                                           "per_layer_mean": [round(float(v), 2) for v in inter.mean((1, 2, 3))],
                                           "scope": "number of pairs selected on both Q and K in the same cell, of the --pairs per side"}
    report["composition_shared"] = composition(shared)
    report["fixed_high_in_separate_q_fraction"] = float((sep["mask_q"] & fixed).sum() / fixed.sum())
    report["fixed_high_in_separate_k_fraction"] = float((sep["mask_k"] & fixed).sum() / fixed.sum())
    report["damage_total_by_variant"] = {name: float(damage_pooled["q"][v["mask_q"]].sum() + damage_pooled["k"][v["mask_k"]].sum()) for name, v in variants.items()}
    report["per_layer_mean_damage_fixed_over_separate"] = [round(float((damage_pooled["q"][l][fixed[l]].sum() + damage_pooled["k"][l][fixed[l]].sum()) /
                                                                     (damage_pooled["q"][l][sep["mask_q"][l]].sum() + damage_pooled["k"][l][sep["mask_k"][l]].sum()).clamp_min(1e-30)), 3) for l in range(L)]
    pooled_mask = variants["pooled"]["mask_q"]
    report["composition_pooled"] = composition(pooled_mask[:, 0, 0])
    report["pooled_vs_shared_intersection"] = {"mean_pairs": float((pooled_mask & shared).sum(-1).double().mean()),
                                               "p10": float((pooled_mask & shared).sum(-1).double().flatten().quantile(0.1)),
                                               "scope": "pairs the step/branch-invariant pooled set shares with the per-cell shared set, per cell, of --pairs"}
    removal_shared = lowest_k_mask(stats.side_score("q", "energy") + stats.side_score("k", "energy"), k)
    report["shared_vs_removal_criterion_intersection"] = {"mean_pairs": float((removal_shared & shared).sum(-1).double().mean()),
                                                          "scope": "per cell, pairs in common between the shared selection (pooled-mean replacement damage) and the selection by energy x partner energy (zero replacement damage)"}
    report["damage_pooled_over_shared"] = float((damage_pooled["q"][pooled_mask].sum() + damage_pooled["k"][pooled_mask].sum()) / (damage_pooled["q"][shared].sum() + damage_pooled["k"][shared].sum()))
    split_means, content = variants["split"]["mask_q"], variants["wide"]["mask_nope"]
    report["split"] = {"means_pairs": args.means_pairs, "nope_pairs": 64 - args.means_pairs,
                       "composition_means": composition(split_means), "composition_nope": composition(~split_means),
                       "shared_in_split_means_fraction": float((split_means & shared).sum() / shared.sum()),
                       "fixed_high_in_nope_fraction": float(((~split_means) & fixed).sum() / fixed.sum()),
                       "scope": "per cell; the split means set is the lowest-damage superset of the shared set when means_pairs >= pairs"}
    report["wide"] = {"content_pairs": args.content_pairs, "composition_content": composition(content),
                      "fixed_high_in_content_fraction": float((content & fixed).sum() / fixed.sum()),
                      "content_vs_split_nope_intersection_mean": float((content & ~split_means).sum(-1).double().mean()),
                      "scope": "per cell; content = the highest-damage pairs by damage_q + damage_k; means cover all 64 pairs"}
    if "policy" in variants:
        pol = variants["policy"]
        first = next((p for p in pol["spec"]["policies"] if "complement_of" not in p), None)
        report["policy"] = {"description": pol["description"], "spec": pol["spec"], "spec_hash": pol["spec_hash"],
                            "mean_pairs_in_common_with_shared_per_cell": float((pol["mask_q"] & shared).sum(-1).double().mean()),
                            "mean_pairs_in_common_with_fixed_high_per_cell": float((pol["mask_q"] & fixed).sum(-1).double().mean()),
                            "energy_q_mean_pairs_over_all": report["variants"]["policy"]["energy_q_mean_pairs_over_all"],
                            "energy_q_shared_pairs_over_all": report["variants"]["shared"]["energy_q_mean_pairs_over_all"],
                            "stability_q": stability(stats, "q", first["criterion"], int(first["pairs"]), first["rank"], pol["mask_q"]) if first else None,
                            "stability_k": stability(stats, "k", first["criterion"], int(first["pairs"]), first["rank"], pol["mask_k"]) if first else None,
                            "scope": "policy variant vs the shared (damage) and fixed-high sets, per cell; energy ratio = pooled energy of the selected pairs over the all-pair pooled energy"}

    # ---- outputs ----------------------------------------------------------------------------------------
    save_torch(out / "pair-statistics.pt", {"mean_q": stats.pooled["q"].float(), "mean_k": stats.pooled["k"].float(),
        "energy_q": stats.pooled_energy["q"].float(), "energy_k": stats.pooled_energy["k"].float(), "prompts": ids,
        "scope": "pooled over calibration prompts of per-prompt all-token statistics; separate layer/step/branch/head; full 128 coordinates"})
    save_torch(out / "selection.pt", {"variants": {n: {k_: (v_.bool() if torch.is_tensor(v_) else v_) for k_, v_ in v.items()} for n, v in variants.items()}, "pairs": k,
        "damage_q": damage_pooled["q"].float(), "damage_k": damage_pooled["k"].float(), "mrl_q": stats.mrl["q"].float(), "mrl_k": stats.mrl["k"].float(),
        "error_q": stats.mean_err["q"].float(), "error_k": stats.mean_err["k"].float(),
        "criterion": "see each variant's spec (pair_selection.py); damage = mean_p E_p||x_f - mu_f||^2 x mean_p E_p||partner_f||^2 per cell and side", "prompts": ids})
    bypass = parse_bypass(args.bypass)
    if args.bypass:                                   # explicit rules replace the legacy defaults entirely
        exempt_layers, exempt_steps = tuple(bypass.get("layers", ())), tuple(bypass.get("steps", ()))
    else:
        exempt_layers, exempt_steps = (() if args.include_layer0 else (0,)), (() if args.include_step1 else (0,))
    for name, masks in variants.items():
        center = LearnedCenter(layers=L, steps=S, branches=B, heads=H, exempt_layers=exempt_layers, exempt_steps=exempt_steps,
                               mode="mean", selection=masks, bypass=bypass or None)
        center.init_from_calibration({"mean_q": stats.pooled["q"].float(), "mean_k": stats.pooled["k"].float()})
        save_torch(out / f"static-{name}" / "learned_center.pt", center.checkpoint_record(extra={"variant": name}))
    write_json(out / "selection-report.json", report)
    print(json.dumps({k_: v_ for k_, v_ in report.items() if k_ not in ("ids", "variants")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
