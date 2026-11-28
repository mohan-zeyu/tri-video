"""Selection policies: which rotary pairs of each (layer, step, CFG branch, head) cell get which role.

A selection spec is a list of policies applied in order; each policy scores pairs on the pooled calibration
statistics, ranks them inside a cell (or inside a pool of cells) and assigns a role to the chosen pairs:

    {"criterion": "damage", "combine": "shared", "pool": [], "pairs": 16, "rank": "lowest", "role": "mean"}

criterion  per-side score of a pair (lower = safer to replace by its mean):
    damage            mean_p err_pool_x,f  x  mean_p E_p||partner_f||^2     (replacement error x partner energy; the original)
    error             mean_p err_pool_x,f                                    (replacement error alone, no partner factor)
    error_normalized  mean_p err_pool_x,f / mean_p E_p||x_f||^2               (= 1 - RMS-MRL^2: pure concentration, energy ignored)
    energy            mean_p E_p||x_f||^2  x  mean_p E_p||partner_f||^2       (zero-replacement damage)
    fixed_high        0 on the 22 high-frequency pairs (t 0-7, h 22-28, w 43-49), 1 elsewhere (the original fixed third)
    all               0 everywhere (used with "pairs": 64)
combine    shared   one set for Q and K from score_q + score_k        separate  each side ranks its own score
pool       axes summed before ranking: any of "steps", "branches", "heads" (the chosen set is then the same across them)
pairs      how many pairs per cell get the role (1..64)
rank       lowest (default) or highest
role       mean          replaced by the learned constant (mask_q / mask_k)
           content_nope  kept as content without rotary embedding (mask_nope)
complement {"complement_of": "mean", "role": "content_nope"} gives the pairs NOT in that role another role
wide       top-level flag of the selection: widened head (means on all pairs + content pairs appended; learned_center.py)

Presets reproduce the historical variants exactly (tested): shared, separate, pooled, split, wide, fixed_high.

Definitions (per cell c, side x, pair f, prompt p; see select_pairs.py):
    mu_f = mean_p xbar_p,f (pooled static mean),  err_pool_p,f = E_p||x_f - mu_f||^2,  E_p||x_f||^2 = per-prompt mean token energy.
"""

import copy
import json

import torch

from center_replacement import HIGH_PAIRS

PAIRS = 64
AXES = {"steps": 1, "branches": 2, "heads": 3}
CRITERIA = ("damage", "error", "error_normalized", "energy", "fixed_high", "all")
ROLES = ("mean", "content_nope")

PRESETS = {
    "shared":     {"wide": False, "policies": [{"criterion": "damage", "combine": "shared", "pairs": 22, "rank": "lowest", "role": "mean"}]},
    "separate":   {"wide": False, "policies": [{"criterion": "damage", "combine": "separate", "pairs": 22, "rank": "lowest", "role": "mean"}]},
    "pooled":     {"wide": False, "policies": [{"criterion": "damage", "combine": "shared", "pool": ["steps", "branches"], "pairs": 22, "rank": "lowest", "role": "mean"}]},
    "split":      {"wide": False, "policies": [{"criterion": "damage", "combine": "shared", "pairs": 42, "rank": "lowest", "role": "mean"},
                                               {"complement_of": "mean", "role": "content_nope"}]},
    "wide":       {"wide": True, "policies": [{"criterion": "all", "combine": "shared", "pairs": 64, "role": "mean"},
                                              {"criterion": "damage", "combine": "shared", "pairs": 32, "rank": "highest", "role": "content_nope"}]},
    "fixed_high": {"wide": False, "policies": [{"criterion": "fixed_high", "combine": "shared", "pairs": 22, "rank": "lowest", "role": "mean"}]},
}


def pair_view(x):
    return x.view(*x.shape[:-1], PAIRS, 2)


def lowest_k_mask(score, k, largest=False):
    """bool mask of the k smallest (or largest) entries along the last dim; ties broken by torch.topk order."""
    idx = score.topk(k, dim=-1, largest=largest).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def high_pair_mask():
    mask = torch.zeros(PAIRS, dtype=torch.bool)
    mask[list(HIGH_PAIRS)] = True
    return mask


class PooledStatistics:
    """Per-cell statistics pooled over calibration prompts. means/energy: dict side -> [P, L, S, B, H, 128] / [P, ..., 64]."""

    def __init__(self, means, energy):
        self.P, self.L, self.S, self.B, self.H, _ = means["q"].shape
        self.pooled = {s: means[s].mean(0) for s in ("q", "k")}
        self.pooled_energy = {s: energy[s].mean(0) for s in ("q", "k")}
        self.err_pool, self.err_own = {}, {}
        for s in ("q", "k"):
            mu = pair_view(self.pooled[s])
            xb = pair_view(means[s])
            self.err_pool[s] = (energy[s] - 2 * (mu * xb).sum(-1) + mu.square().sum(-1)).clamp_min(0)   # [P, L,S,B,H,64]
            self.err_own[s] = (energy[s] - xb.square().sum(-1)).clamp_min(0)
        self.mean_err = {s: self.err_pool[s].mean(0) for s in ("q", "k")}
        self.mrl = {s: pair_view(self.pooled[s]).square().sum(-1).sqrt() / self.pooled_energy[s].sqrt().clamp_min(1e-12) for s in ("q", "k")}

    @property
    def shape(self):
        return (self.L, self.S, self.B, self.H)

    def side_score(self, side, criterion):
        """[L,S,B,H,64] double; lower = better to replace."""
        partner = "k" if side == "q" else "q"
        if criterion == "damage":
            return self.mean_err[side] * self.pooled_energy[partner]
        if criterion == "error":
            return self.mean_err[side].clone()
        if criterion == "error_normalized":
            return self.mean_err[side] / self.pooled_energy[side].clamp_min(1e-12)
        if criterion == "energy":
            return self.pooled_energy[side] * self.pooled_energy[partner]
        if criterion == "fixed_high":
            base = (~high_pair_mask()).double() + torch.arange(PAIRS, dtype=torch.float64) * 1e-6     # HIGH first, then by index
            return base.expand(*self.shape, PAIRS).clone()
        if criterion == "all":
            return torch.zeros(*self.shape, PAIRS, dtype=torch.float64)
        raise ValueError(f"unknown criterion {criterion!r}; one of {CRITERIA}")

    def per_prompt_score(self, side, criterion):
        """Each prompt's own-mean score [P, L,S,B,H,64] (for the cross-prompt stability report); None if undefined."""
        partner = "k" if side == "q" else "q"
        if criterion == "damage":
            return self.err_own[side] * self.pooled_energy[partner].unsqueeze(0)
        if criterion == "error":
            return self.err_own[side]
        if criterion == "error_normalized":
            return self.err_own[side] / self.pooled_energy[side].clamp_min(1e-12).unsqueeze(0)
        return None


def normalize_spec(spec):
    """Accept a preset name, a preset name with overrides, or a full {"wide": bool, "policies": [...]} dict."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown selection preset {spec!r}")
        spec = {"preset": spec}
    spec = copy.deepcopy(spec)
    if "preset" in spec:
        base = copy.deepcopy(PRESETS[spec.pop("preset")])
        over = spec.pop("policies", None)
        base.update(spec)
        if over is not None:
            base["policies"] = over
        spec = base
    if "policies" not in spec:
        raise ValueError("selection spec needs 'policies' (or a preset)")
    spec.setdefault("wide", False)
    for p in spec["policies"]:
        if "complement_of" in p:
            if p.get("role") not in ROLES or p["complement_of"] not in ROLES:
                raise ValueError(f"bad complement policy {p}")
            continue
        p.setdefault("combine", "shared")
        p.setdefault("pool", [])
        p.setdefault("rank", "lowest")
        if p.get("criterion") not in CRITERIA:
            raise ValueError(f"bad criterion in {p}")
        if p.get("role") not in ROLES:
            raise ValueError(f"bad role in {p}")
        if p["combine"] not in ("shared", "separate") or p["rank"] not in ("lowest", "highest"):
            raise ValueError(f"bad combine/rank in {p}")
        if not 1 <= int(p.get("pairs", 0)) <= PAIRS:
            raise ValueError(f"pairs must be 1..64 in {p}")
        if any(a not in AXES for a in p["pool"]):
            raise ValueError(f"pool axes must be among {list(AXES)} in {p}")
    return spec


def _rank(score, pairs, rank, pool):
    """Rank inside each cell after summing over the pooled axes; the chosen set is broadcast back over them."""
    axes = tuple(AXES[a] for a in pool)
    pooled = score.sum(axes, keepdim=True) if axes else score
    mask = lowest_k_mask(pooled, pairs, largest=(rank == "highest"))
    return mask.expand_as(score).clone()


def build_selection(stats, spec):
    """-> {"mask_q", "mask_k", "mask_nope": [L,S,B,H,64] bool, "wide": bool, "scores": {role: ...}, "spec": normalized}"""
    spec = normalize_spec(spec)
    shape = (*stats.shape, PAIRS)
    masks = {"mean_q": torch.zeros(shape, dtype=torch.bool), "mean_k": torch.zeros(shape, dtype=torch.bool),
             "content_nope": torch.zeros(shape, dtype=torch.bool)}
    for p in spec["policies"]:
        if "complement_of" in p:
            src = p["complement_of"]
            comp = ~(masks["mean_q"] | masks["mean_k"]) if src == "mean" else ~masks["content_nope"]
            if p["role"] == "mean":
                masks["mean_q"] |= comp
                masks["mean_k"] |= comp
            else:
                masks["content_nope"] |= comp
            continue
        sq, sk = stats.side_score("q", p["criterion"]), stats.side_score("k", p["criterion"])
        if p["combine"] == "shared":
            m = _rank(sq + sk, int(p["pairs"]), p["rank"], p["pool"])
            mq, mk = m, m.clone()
        else:
            mq, mk = _rank(sq, int(p["pairs"]), p["rank"], p["pool"]), _rank(sk, int(p["pairs"]), p["rank"], p["pool"])
        if p["role"] == "mean":
            masks["mean_q"] |= mq
            masks["mean_k"] |= mk
        else:
            if not torch.equal(mq, mk):
                raise ValueError("content_nope must be the same set on Q and K (use combine=shared)")
            masks["content_nope"] |= mq
    out = {"mask_q": masks["mean_q"], "mask_k": masks["mean_k"], "mask_nope": masks["content_nope"], "wide": bool(spec["wide"]), "spec": spec}
    if not out["wide"] and bool((out["mask_nope"] & (out["mask_q"] | out["mask_k"])).any()):
        raise ValueError("a content_nope pair cannot also be a mean pair unless the selection is wide")
    return out


def describe(spec):
    spec = normalize_spec(spec)
    parts = []
    for p in spec["policies"]:
        if "complement_of" in p:
            parts.append(f"complement of {p['complement_of']} -> {p['role']}")
        else:
            pool = f", pooled over {'/'.join(p['pool'])}" if p["pool"] else ""
            parts.append(f"{p['pairs']} {p['rank']} by {p['criterion']} ({p['combine']}{pool}) -> {p['role']}")
    return ("widened head; " if spec["wide"] else "") + "; ".join(parts)


def spec_hash(spec):
    import hashlib
    return hashlib.sha256(json.dumps(normalize_spec(spec), sort_keys=True).encode()).hexdigest()[:16]
