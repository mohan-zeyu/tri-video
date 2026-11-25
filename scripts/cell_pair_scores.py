"""Scores of the 64 rotary pairs in ONE cell, with positive and negative scores kept apart.

One cell = one prompt, one layer, one denoising step, one CFG branch, one head. Nothing here is pooled over
prompts, layers, steps, branches or heads. Inside the cell every video token is a query and every video token is
a key (N x N query-key combinations; N = 10,800 for Wan 480p, so 116,640,000).

    c_f[i,j] = <qr_i[f], kr_j[f]> / sqrt(128)    the score pair f adds to the logit of query i and key j
    s[i,j]   = sum over the 64 pairs f of c_f[i,j]                the full 128-dim post-RoPE logit
    p[i,j]   = softmax over all keys j of s[i,.]                  the head's attention weights (fp32)

qr / kr are the attn1.norm_q / norm_k outputs rotated with the model's own RoPE angles and rounded to the model's
dtype, as in pair_score_attribution.py. Pair order: t 0-21, h 22-42, w 43-63.

A score x[i,j] (one pair's c_f, a pair set's sum, or the logit s) is summarized without letting signs cancel:

  all_*   every query-key combination counts once (divided by N*N):
          all_pos = sum of max(x,0), all_neg = sum of min(x,0), all_net = all_pos + all_neg, all_abs = all_pos - all_neg,
          all_pos_frac = fraction of combinations with x > 0, all_min ... all_max = quantiles of x over the combinations
  att_*   each combination weighted by the attention p[i,j] the query gives the key (divided by N; a query's
          weights sum to 1): att_pos, att_neg, att_net, att_abs as above; att_pos_mass = attention mass on x > 0
  qry_*   no pooling over queries: a[i] = sum_j p[i,j] x[i,j] is one number per query; qry_min ... qry_max are its
          quantiles across the N queries, qry_pos_frac the fraction of queries with a[i] > 0
  ctr_*   the same for x[i,j] - mean_j x[i,.], the part softmax can see (a per-query constant does not change
          the attention): ctr_all_abs, ctr_att_pos, ctr_att_neg, ctr_att_net, ctr_att_abs

Shares over the 64 pairs (pair rows only): *_abs_share = the pair's *_abs / the sum of *_abs over the 64 pairs
(every term is >= 0, nothing cancels); att_net_share is the earlier `attn_score_share` and ctr_att_net_share the
earlier `adv_share`, both signed.

Rows: the 64 pairs; the logit; and for every named pair set (a) "sum of pair rows": the pairs' rows added, so
nothing cancels between pairs, and (b) "set score": the pairs' scores added inside each query-key combination
first (as the logit does), then summarized.

  capture  (lab, one GPU)  replay the stored teacher entering state of one prompt and step through the frozen bf16
           teacher (both CFG branches in one batch, as the collector does), keep Q/K of one layer, branch and head,
           check that the layer's attention output recomputed from them matches the model, and write cell.npz
           (pre-RoPE Q/K [N, 128], cos/sin [N, 64], the pair sets of this cell from --set NAME PATH VARIANT).
  scores   (CPU or GPU)    cell.npz -> pair-scores.csv (every column), PAIR-SCORES.md, pair-scores.json

  python scripts/cell_pair_scores.py capture --id d0000 --layer 0 --step 1 --head 1 --out distill-results/score-attribution-cell \
      --set shared22 <selection.pt> shared --set fixed_high <selection.pt> fixed_high --set shared32 <selection32.pt> shared
  python scripts/cell_pair_scores.py scores distill-results/score-attribution-cell/d0000-L00-S01-conditional-H01
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from full_sweep_metrics import rotate
from pair_score_attribution import quantiles

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("conditional", "unconditional")
LEVELS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
LEVEL_NAMES = ("min", "q05", "q25", "median", "q75", "q95", "max")
ADDITIVE = ("all_pos", "all_neg", "all_net", "all_abs", "att_pos", "att_neg", "att_net", "att_abs",
            "ctr_all_abs", "ctr_att_pos", "ctr_att_neg", "ctr_att_net", "ctr_att_abs")
SHARES = ("all_abs_share", "att_abs_share", "ctr_all_abs_share", "ctr_att_abs_share", "att_net_share", "ctr_att_net_share")
AXES = ["t"] * 22 + ["h"] * 21 + ["w"] * 21


def resolve(path):
    return Path(path) if os.path.isabs(str(path)) else ROOT / path


def score_stats(x, p):
    """x [N, N] scores of query i (row) and key j (column); p [N, N] attention weights. One float per column."""
    n = x.shape[0]
    total = lambda t: float(t.sum(dtype=torch.float64))
    pos, neg, positive = x.clamp(min=0), x.clamp(max=0), x > 0
    out = {"all_pos": total(pos) / n ** 2, "all_neg": total(neg) / n ** 2, "all_pos_frac": total(positive) / n ** 2,
           "att_pos": total(p * pos) / n, "att_neg": total(p * neg) / n, "att_pos_mass": total(p * positive) / n}
    del pos, neg, positive
    per_query = (p * x).sum(1, dtype=torch.float64)                              # a[i], one number per query
    out["qry_pos_frac"] = float((per_query > 0).double().mean())
    out.update({f"qry_{name}": float(v) for name, v in zip(LEVEL_NAMES, quantiles(per_query, 0, LEVELS))})
    out.update({f"all_{name}": float(v) for name, v in zip(LEVEL_NAMES, quantiles(x.flatten(), 0, LEVELS))})
    centred = x - x.mean(1, keepdim=True)
    out["ctr_all_abs"] = total(centred.abs()) / n ** 2
    out["ctr_att_pos"], out["ctr_att_neg"] = total(p * centred.clamp(min=0)) / n, total(p * centred.clamp(max=0)) / n
    for prefix in ("all", "att", "ctr_att"):
        out[prefix + "_net"] = out[prefix + "_pos"] + out[prefix + "_neg"]
        out[prefix + "_abs"] = out[prefix + "_pos"] - out[prefix + "_neg"]
    return out


def cell_scores(q, k, cos, sin, native=None, sets=None, device="cpu"):
    """q, k [N, D] normalized pre-RoPE Q/K of one head; cos, sin [N, D/2]; sets {name: bool [D/2]}.

    Returns (rows, info): one row per pair, one for the logit, two per pair set (see the module docstring).
    """
    sets = sets or {}
    qr = rotate(q[None].to(device), cos.to(device), sin.to(device), native)[0]
    kr = rotate(k[None].to(device), cos.to(device), sin.to(device), native)[0]
    n, d = qr.shape
    scale = 1 / math.sqrt(d)
    logits = qr @ kr.T * scale
    logp = torch.log_softmax(logits, 1)
    p = logp.exp()
    entropy = -(p * logp).sum(1, dtype=torch.float64)
    del logp
    rebuilt = torch.zeros_like(logits)
    set_score = {name: torch.zeros_like(logits) for name in sets}
    rows = []
    for f in range(d // 2):
        c = qr[:, 2 * f:2 * f + 2] @ kr[:, 2 * f:2 * f + 2].T * scale
        rows.append({"row": "pair", "name": str(f), "axis": AXES[f] if d == 128 else "", **{f"in_{s}": int(m[f]) for s, m in sets.items()},
                     **score_stats(c, p)})
        rebuilt += c
        for name, mask in sets.items():
            if mask[f]:
                set_score[name] += c
    for column in ("all_abs", "att_abs", "ctr_all_abs", "ctr_att_abs"):
        whole = sum(r[column] for r in rows)
        for r in rows:
            r[column + "_share"] = r[column] / whole
    for column in ("att_net", "ctr_att_net"):                                    # signed shares; undefined when the sum nearly cancels
        whole, size = sum(r[column] for r in rows), sum(abs(r[column]) for r in rows)
        for r in rows:
            r[column + "_share"] = r[column] / whole if whole > 0.01 * size else None
    pair_rows = list(rows)
    rows.append({"row": "logit", "name": "all 64 pairs", **score_stats(logits, p)})
    for name, mask in sets.items():
        members = [r for r, m in zip(pair_rows, mask) if m]
        added = {c: (None if any(r[c] is None for r in members) else sum(r[c] for r in members)) for c in (*ADDITIVE, *SHARES)}
        rows.append({"row": "sum of pair rows", "name": name, "pairs": len(members), **added})
        rows.append({"row": "set score", "name": name, "pairs": len(members), **score_stats(set_score[name], p)})
    info = {"tokens": n, "combinations": n * n, "pairs": d // 2,
            "largest_abs_difference_between_logit_and_sum_of_pair_scores": float((rebuilt - logits).abs().max()),
            "logit_quantiles_over_combinations": dict(zip(LEVEL_NAMES, (rows[len(pair_rows)][f"all_{n_}"] for n_ in LEVEL_NAMES))),
            "attention_effective_keys_quantiles_over_queries": dict(zip(LEVEL_NAMES, quantiles(entropy.exp(), 0, LEVELS).tolist())),
            "attention_largest_weight_quantiles_over_queries": dict(zip(LEVEL_NAMES, quantiles(p.max(1).values.double(), 0, LEVELS).tolist()))}
    return rows, info


def cell_name(meta):
    return f"{meta['prompt_id']}-L{meta['layer']:02d}-S{meta['step']:02d}-{meta['branch']}-H{meta['head']:02d}"


@torch.inference_mode()
def capture(args):
    from diffusers import WanTransformer3DModel
    from train_learned_center import TrajectoryStore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = WanTransformer3DModel.from_pretrained(args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                  local_files_only=True).to(device).eval()
    store = TrajectoryStore(resolve(args.traj), [args.id])
    item = store.item(args.id, args.step - 1)
    attn, seen = model.blocks[args.layer].attn1, {}
    keep = lambda name: (lambda module, inputs, output: seen.__setitem__(name, output))
    handles = [model.rope.register_forward_hook(keep("rope")), attn.norm_q.register_forward_hook(keep("q")),
               attn.norm_k.register_forward_hook(keep("k")), attn.to_v.register_forward_hook(keep("v")), attn.register_forward_hook(keep("out"))]
    x = item["x"].to(device=device, dtype=torch.bfloat16).unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
    text = torch.cat([item["positive"], item["negative"]]).to(device=device, dtype=torch.bfloat16)
    model(hidden_states=x, timestep=torch.full((2,), float(item["timestep"]), device=device), encoder_hidden_states=text, return_dict=False)
    for handle in handles:
        handle.remove()
    branch = BRANCHES.index(args.branch)
    heads, dim = model.config.num_attention_heads, model.config.attention_head_dim
    cos, sin = seen["rope"][0][0, :, 0, 0::2].float(), seen["rope"][1][0, :, 0, 1::2].float()            # [N, 64]
    q, k, v = [seen[name][branch].unflatten(-1, (heads, dim)).transpose(0, 1) for name in ("q", "k", "v")]  # [H, N, D]
    # the tensors kept are the ones the model attends with: recompute this layer's attention output from them
    qr, kr = [rotate(t, cos, sin, t.dtype).to(t.dtype)[None] for t in (q, k)]
    reference = F.scaled_dot_product_attention(qr, kr, v[None])[0].transpose(0, 1).flatten(1)
    reference = attn.to_out[1](attn.to_out[0](reference))
    error, size = float((reference - seen["out"][branch]).abs().max()), float(seen["out"][branch].abs().max())
    validation = {"max_abs_err": error, "output_scale": size, "relative_max_err": error / max(size, 1e-12)}
    if validation["relative_max_err"] > 1e-3:
        raise RuntimeError(f"attention recomputation mismatch {validation}")
    meta = {"prompt_id": args.id, "layer": args.layer, "step": args.step, "branch": args.branch, "head": args.head,
            "timestep": float(item["timestep"]), "dtype": str(q.dtype).replace("torch.", ""), "tokens": q.shape[1],
            "source": "frozen teacher replaying its stored entering state of this prompt and step, bf16, both CFG branches in one batch",
            "attention_output_validation": validation, "sets": {}}
    arrays = {}
    for name, path, variant in args.set:
        chosen = torch.load(resolve(path), map_location="cpu")["variants"][variant]
        mask = chosen["mask_q"][args.layer, args.step - 1, branch, args.head] | chosen["mask_k"][args.layer, args.step - 1, branch, args.head]
        arrays["set_" + name] = mask.numpy()
        meta["sets"][name] = {"path": str(path), "variant": variant, "pairs": int(mask.sum()),
                              "q_equals_k": bool(torch.equal(chosen["mask_q"], chosen["mask_k"]))}
    out = resolve(args.out) / cell_name(meta)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cell.npz", q=q[args.head].float().cpu().numpy(), k=k[args.head].float().cpu().numpy(),
                        cos=cos.cpu().numpy(), sin=sin.cpu().numpy(), meta=json.dumps(meta), **arrays)
    print("CAPTURED", out / "cell.npz", json.dumps(validation), flush=True)
    return out


def load_cell(folder):
    z = np.load(Path(folder) / "cell.npz")
    meta = json.loads(str(z["meta"]))
    tensors = {name: torch.from_numpy(z[name]) for name in ("q", "k", "cos", "sin")}
    sets = {name: z["set_" + name].astype(bool) for name in meta["sets"]}
    return tensors, sets, meta


def fmt(value, spec="+.4f"):
    return "" if value is None else format(value, spec)


def markdown(rows, info, meta, sets):
    n = info["tokens"]
    where = (f"prompt `{meta['prompt_id']}`, layer {meta['layer']}, denoising step {meta['step']} of the stored trajectory (timestep "
             f"{meta['timestep']:.1f}), {meta['branch']} branch, head {meta['head']} (heads count from 0)")
    text = [f"# Pair scores of one cell: {cell_name(meta)}\n",
            f"One cell only: {where}. Source: {meta['source']}. Every one of the {n:,} video tokens is a query and every one is a key "
            f"({info['combinations']:,} query-key combinations), full 128-dim post-RoPE logit split exactly into its 64 pair scores. "
            "Nothing is pooled over prompts, layers, steps, branches or heads.\n",
            "`c_f[i,j]` is the score pair f adds to the logit of query i and key j. Positive and negative scores are summed "
            "separately, so they cannot cancel:\n",
            "- **attention-weighted** (`att`): each query-key score is weighted by the attention the query gives that key, summed over "
            f"keys and queries, divided by the {n:,} queries. `+` adds only the positive scores, `−` only the negative ones.",
            "- **every combination once** (`all`): each query-key score counts once, divided by the number of combinations.",
            "- **share of |score|**: (`+` minus `−`) of the pair divided by the same summed over the 64 pairs. Every term is "
            "non-negative.",
            "- **centred**: the same after subtracting, for each query, the pair's mean score over all keys. Softmax ignores "
            "that per-query constant, so this is the part of the score that can move attention.",
            "- **per query**: `a[i] = sum_j p[i,j] c_f[i,j]` is one number per query; the three values are its 5%, median and "
            "95% quantiles across the queries (no pooling over queries).\n",
            f"Checks: attention output of this layer recomputed from the captured Q/K, relative maximum error "
            f"{meta['attention_output_validation']['relative_max_err']:.2e}; largest difference between the logit and the sum of the "
            f"64 pair scores {info['largest_abs_difference_between_logit_and_sum_of_pair_scores']:.2e}. Attention of this head: "
            f"effective number of keys per query (exp of entropy) median "
            f"{info['attention_effective_keys_quantiles_over_queries']['median']:.0f} "
            f"(5%–95% of queries {info['attention_effective_keys_quantiles_over_queries']['q05']:.0f}–"
            f"{info['attention_effective_keys_quantiles_over_queries']['q95']:.0f}).\n"]
    if sets:
        text.append("Pair sets of this cell (union of the Q and K masks of the selection file; no exemption applied): "
                    + "; ".join(f"`{name}` {int(mask.sum())} pairs: {', '.join(str(i) for i in np.flatnonzero(mask))}" for name, mask in sets.items()) + ".\n")
    names = list(sets)
    head = ["pair", "axis", *names, "att +", "att −", "att net", "att share of \\|score\\|", "all +", "all −", "all share of \\|score\\|",
            "centred att share of \\|score\\|", "per query 5%", "per query median", "per query 95%"]
    line = lambda cells: "| " + " | ".join(cells) + " |"
    body = lambda r: [fmt(r["att_pos"]), fmt(r["att_neg"]), fmt(r["att_net"]), fmt(r.get("att_abs_share"), ".2%"), fmt(r["all_pos"]), fmt(r["all_neg"]),
                      fmt(r.get("all_abs_share"), ".2%"), fmt(r.get("ctr_att_abs_share"), ".2%"), fmt(r.get("qry_q05")), fmt(r.get("qry_median")), fmt(r.get("qry_q95"))]
    text += ["## The 64 pairs\n", line(head), "|" + "---:|" * len(head)]
    for r in rows:
        if r["row"] == "pair":
            text.append(line([r["name"], r["axis"], *("x" if r[f"in_{s}"] else "" for s in names), *body(r)]))
    text += ["\n## Pair sets and the whole logit\n",
             "`sum of pair rows`: the rows of the set's pairs added, nothing cancels between pairs. `set score`: the pairs' scores are "
             "added inside each query-key combination first, as the logit adds them, and then split into `+` and `−`.\n"]
    head = ["row", "set", "pairs", *head[2 + len(names):]]
    text += [line(head), "|---|---|" + "---:|" * (len(head) - 2)]
    for r in rows:
        if r["row"] != "pair":
            text.append(line([r["row"], r["name"], str(r.get("pairs", info["pairs"])), *body(r)]))
    text.append("\nEvery column, including the quantiles of each pair's score over all combinations, is in `pair-scores.csv`.\n")
    return "\n".join(text)


@torch.inference_mode()
def scores(args):
    folder = resolve(args.cell)
    tensors, sets, meta = load_cell(folder)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    native = getattr(torch, meta["dtype"]) if meta["dtype"] != "float32" else None
    rows, info = cell_scores(tensors["q"], tensors["k"], tensors["cos"], tensors["sin"], native, sets, device)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(folder / "pair-scores.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows({key: (format(v, ".8g") if isinstance(v, float) else v) for key, v in row.items()} for row in rows)
    report = {"cell": meta, "scope": "one prompt, one layer, one denoising step, one CFG branch, one head; every video token as a query x "
              "every video token as a key; full 128-dim post-RoPE logit split into its 64 pair scores; no pooling over cells",
              "columns": __doc__.split("A score x[i,j]")[1].split("  capture  ")[0].strip(), "info": info,
              "sets": {name: np.flatnonzero(mask).tolist() for name, mask in sets.items()}, "rows": rows}
    (folder / "pair-scores.json").write_text(json.dumps(report, allow_nan=False) + "\n")
    text = markdown(rows, info, meta, sets)
    (folder / "PAIR-SCORES.md").write_text(text)
    print(text[:6000])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("capture")
    c.add_argument("--traj", default="out/distill-traj-wan")
    c.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"))
    c.add_argument("--id", default="d0000", help="prompt id")
    c.add_argument("--layer", type=int, default=0, help="0-29")
    c.add_argument("--step", type=int, default=1, help="denoising step, 1-20 (1 = first)")
    c.add_argument("--branch", choices=BRANCHES, default="conditional")
    c.add_argument("--head", type=int, default=1, help="0-11")
    c.add_argument("--set", nargs=3, action="append", default=[], metavar=("NAME", "PATH", "VARIANT"))
    c.add_argument("--out", default="distill-results/score-attribution-cell")
    s = commands.add_parser("scores")
    s.add_argument("cell", help="folder written by capture (holds cell.npz)")
    s.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    capture(args) if args.command == "capture" else scores(args)


if __name__ == "__main__":
    main()
