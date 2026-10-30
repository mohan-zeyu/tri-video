"""TriAttention-style causal check, adapted to video DiTs (3D RoPE).

Paper (Sec 3.2-3.3): if pre-RoPE Q/K concentrate around centers q-bar, k-bar,
then approximating q_t ~ q-bar, k_s ~ k-bar makes every attention logit a
function of the rotation-angle difference alone:

    s_hat(t, s) = sum_f Re( conj(q-bar_f) * e^{i(theta_s,f - theta_t,f)} * k-bar_f ) / sqrt(D)

For 1D RoPE the angle difference is a function of token distance only (their
"trigonometric series"). Video models use 3D RoPE (t,h,w axes), so the series
is a function of the 3D offset -- the analog of the paper's distance
preference. The per-pair angles theta_{t,f} are read directly from the model's
own rope tensors: cos = freqs_cos[t, 2f], sin = freqs_sin[t, 2f+1]
(interleaved convention shared by all three models).

This script: run one forward at t_mid, stash full pre-RoPE q,k for selected
layers, then per head report the Pearson correlation between actual rotated
logits q_t R_t^T R_s k_s and the center-only prediction, over randomly sampled
(t, s) pairs; also emit attention-vs-offset marginals for the most
concentrated heads (figures).

Usage: python analysis_series.py --model wan --layers 0 15 29 --out ../out/wan-series.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_video import ADAPTERS  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)


def angles_from_freqs(freqs):
    """freqs: (cos, sin), each [T, 2D] interleaved-duplicated.
    Returns theta [T, D/2] and a (cos, sin) pair extractor consistent with the
    models' processors: cos = cos[..., 0::2], sin = sin[..., 1::2]."""
    cos, sin = freqs[0].float(), freqs[1].float()
    if cos.dim() == 4:  # wan returns [1, T, 1, 2D]
        cos, sin = cos[0, :, 0], sin[0, :, 0]
    elif cos.dim() == 3:  # ltx returns [1, T, 2D]
        cos, sin = cos[0], sin[0]
    c, s = cos[..., 0::2], sin[..., 1::2]          # [T, D/2]
    return torch.atan2(s, c)


def predict_logits(q_bar, k_bar, theta_h, qidx, chunk=24):
    """q_bar, k_bar: [H, D] float32 centers; theta_h: [H, T, P] per-head angles;
    qidx: [Q]. Returns s_hat: [H, Q, T] (unscaled by 1/sqrt(D))."""
    H, D = q_bar.shape
    zq = torch.view_as_complex(q_bar.reshape(H, D // 2, 2).float())[:, :, None, None]
    zk = torch.view_as_complex(k_bar.reshape(H, D // 2, 2).float())[:, :, None, None]
    rows = []
    for i in range(0, len(qidx), chunk):
        qt = qidx[i:i + chunk]                                 # [q]
        dth = (theta_h.unsqueeze(1) - theta_h[:, qt].unsqueeze(2))  # [H, q, T, P]
        e = torch.polar(torch.ones_like(dth), dth).permute(0, 3, 1, 2)  # [H, P, q, T]
        prod = (zq.conj() * e * zk).sum(1)                      # [H, q, T] complex
        rows.append(torch.view_as_real(prod)[..., 0])
    return torch.cat(rows, dim=1)                               # [H, Q, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 5, 10])
    ap.add_argument("--queries", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ad = ADAPTERS[args.model]()
    if not ad.has_rope:
        raise SystemExit(f"{args.model} runs without RoPE; series analysis not applicable")
    model = ad.load().to("cuda")
    device = "cuda"

    stash = {}
    freqs_box = {}
    handles = [ad.rope_module().register_forward_hook(
        lambda m, x, y: freqs_box.update(freqs=[f.detach() for f in y]))]

    def grab(layer, name):
        def hook(m, x, y):
            stash.setdefault(layer, {})[name] = y.detach()
        return hook

    for layer in args.layers:
        attn = ad.self_attn(layer)
        qm, km, _ = ad.qk_modules(attn)
        handles.append(qm.register_forward_hook(grab(layer, "q")))
        handles.append(km.register_forward_hook(grab(layer, "k")))

    model(**ad.make_inputs(ad.t_mid, "zeros", 0, device))
    freqs = freqs_box["freqs"]
    theta = angles_from_freqs(freqs)

    result = {"model": ad.name, "layers": args.layers, "timestep": ad.t_mid,
              "queries": args.queries, "heads": [], "marginals": {}}
    T = None
    for layer in args.layers:
        box = stash[layer]
        attn = ad.self_attn(layer)
        H = attn.heads
        q_raw, k_raw = box["q"], box["k"]
        if args.model == "cog":
            nv = freqs[0].shape[0]
            q_raw, k_raw = q_raw[:, -nv:], k_raw[:, -nv:]
        D = q_raw.shape[-1] // H
        q = q_raw.unflatten(2, (H, D))[0].permute(1, 0, 2).float()   # [H, T, D]
        k = k_raw.unflatten(2, (H, D))[0].permute(1, 0, 2).float()
        T = q.shape[1]
        assert theta.shape[0] == T, (theta.shape, T)

        # actual logits from rotated tensors (the model's own rotation)
        qr = ad.rope_apply(q, freqs)
        kr = ad.rope_apply(k, freqs)

        # sample queries, correlate their logit rows with center-only prediction
        P_total = theta.shape[-1]
        if P_total == D // 2:      # same freqs broadcast to every head (wan)
            theta_h = theta[None].expand(H, -1, -1)
        else:                      # ltx: heads carve contiguous pair blocks
            theta_h = theta.view(T, H, D // 2).permute(1, 0, 2)
        qidx = torch.randperm(T, device=device)[: args.queries]
        actual = torch.einsum("hqd,hkd->hqk", qr[:, qidx], kr) / math.sqrt(D)
        pred = predict_logits(q.mean(1), k.mean(1), theta_h, qidx) / math.sqrt(D)
        a = actual - actual.mean(-1, keepdim=True)
        p = pred - pred.mean(-1, keepdim=True)
        r = (a * p).sum(-1) / (a.norm(dim=-1) * p.norm(dim=-1)).clamp_min(1e-9)
        r = r.mean(1)                                          # mean over queries -> [H]

        # pre-RoPE concentration for context
        q_mrl = q.mean(1).norm(dim=-1) / q.norm(dim=-1).mean(1)

        for h in range(H):
            result["heads"].append({"layer": layer, "head": h,
                                    "recon_r": round(r[h].item(), 4),
                                    "q_mrl": round(q_mrl[h].item(), 4)})

        # marginals for the best head: mean actual logit vs (dt, dh, dw) offsets
        best = int(q_mrl.argmax())
        grid = token_grid(args.model, ad, T)
        if grid is not None:
            ft, ht, wt = grid
            logits = torch.einsum("d,kd->k", qr[best, qidx[0]], kr[best]) / math.sqrt(D)
            marg = {}
            for name, coord in [("dt", ft), ("dh", ht), ("dw", wt)]:
                off = coord[qidx[0]].item() - coord           # signed offsets
                uniq = off.unique()
                marg[name] = [[int(u.item()), float(logits[off == u].mean().item()),
                               int((off == u).sum().item())] for u in uniq]
            result["marginals"][f"L{layer}H{best}"] = marg
        del q, k, qr, kr, actual, pred
        print(f"layer {layer}: mean r={r.mean().item():.3f} "
              f"median r={r.median().item():.3f} (best head {best}, mrl {q_mrl[best]:.3f})", flush=True)

    rs = torch.tensor([h["recon_r"] for h in result["heads"]])
    result["summary"] = {"mean_r": round(rs.mean().item(), 4),
                         "median_r": round(rs.median().item(), 4),
                         "frac_r>0.5": round((rs > 0.5).float().mean().item(), 4),
                         "frac_r>0.7": round((rs > 0.7).float().mean().item(), 4)}
    for h in handles:
        h.remove()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), "wrote", args.out, flush=True)


def token_grid(which, ad, T):
    """(f, h, w) coordinate per token, row-major as produced by patchify."""
    if which == "wan":
        f = (ad.frames - 1) // 4 + 1
        h, w = ad.height // 16, ad.width // 16      # latent/8 then patch 2
        f, h, w = f // 1, h, w
    elif which == "ltx":
        f = (ad.frames - 1) // 8 + 1
        h, w = ad.height // 32, ad.width // 32      # latent/32, patch 1x1x1
    else:
        f = ((ad.frames - 1) // 4 + 1) // 2
        h, w = ad.height // 32, ad.width // 32      # latent/8 then patch 4
    if f * h * w != T:
        print(f"grid mismatch {f}x{h}x{w} != {T}; skipping marginals")
        return None
    ft = torch.arange(f, device="cuda").view(f, 1, 1).expand(f, h, w).reshape(-1)
    ht = torch.arange(h, device="cuda").view(1, h, 1).expand(f, h, w).reshape(-1)
    wt = torch.arange(w, device="cuda").view(1, 1, w).expand(f, h, w).reshape(-1)
    return ft, ht, wt


if __name__ == "__main__":
    main()
