"""Exact per-token, per-pair and full-head metrics for the Wan 3D sweep.

No prompt, layer, head or denoising-step pooling occurs here. Axis logit
statistics include EVERY query and EVERY key on its same-axis line.
"""
import math

import torch

AXES = ("t", "h", "w")


def rotate(x, cos, sin, native_dtype=None):
    """[H,N,D], [N,D/2] -> RoPE output; optional native bf16 rounding."""
    a, b = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    y = torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)
    return y.to(native_dtype).float() if native_dtype is not None else y


def pair_stats(x):
    z = x.float().reshape(*x.shape[:-1], -1, 2)
    center = z.mean(1)
    norms = z.square().sum(-1).sqrt().mean(1)
    resultant = (center.square().sum(-1).sqrt() / norms.clamp_min(1e-20)).clamp(0,1)
    return resultant, norms, center.flatten(-2)


def axis_lines(x, grid, axis):
    """Return [head, other-coordinate line, axis coordinate, whole head D]."""
    shaped = x.reshape(x.shape[0], *grid, x.shape[-1])
    perm = [0] + [i+1 for i in range(3) if i != axis] + [axis+1, 4]
    return shaped.permute(perm).reshape(x.shape[0], -1, grid[axis], x.shape[-1])


def pearson_rows(a, b):
    aa = a-a.mean(-1, keepdim=True)
    bb = b-b.mean(-1, keepdim=True)
    denominator = aa.square().sum(-1).sqrt()*bb.square().sum(-1).sqrt()
    return torch.where(denominator > 1e-12, (aa*bb).sum(-1)/denominator, torch.nan)


def axis_logits(qr, kr, center_qr, center_kr, grid, axis):
    """Full-head logits, all queries, same other coordinates, signed offsets.

    Prediction uses means over ALL video tokens at this one layer/head/step,
    and sums ALL pairs. Positions from all three axes are retained.
    """
    ql, kl = axis_lines(qr, grid, axis), axis_lines(kr, grid, axis)
    actual = torch.matmul(ql, kl.transpose(-1, -2))/math.sqrt(qr.shape[-1])
    # Compute predictions on every line, preserving native angle tensors and
    # avoiding an assumption of perfect floating-point translation invariance.
    cql, ckl = axis_lines(center_qr, grid, axis), axis_lines(center_kr, grid, axis)
    predicted = torch.matmul(cql, ckl.transpose(-1, -2))/math.sqrt(qr.shape[-1])
    r = pearson_rows(actual, predicted)
    extent, lines = grid[axis], ql.shape[1]
    delta = list(range(1-extent, extent))
    mean, std, prediction, counts = [], [], [], []
    for d in delta:
        # torch diagonal offset = key_index-query_index = -delta.
        a = actual.diagonal(offset=-d, dim1=-2, dim2=-1).flatten(1)
        b = predicted.diagonal(offset=-d, dim1=-2, dim2=-1).flatten(1)
        mean.append(a.mean(1))
        std.append(a.std(1, unbiased=False))
        prediction.append(b.mean(1))
        counts.append(lines*(extent-abs(d)))
    mean, std, prediction = [torch.stack(z, dim=1) for z in (mean, std, prediction)]
    # Query at origin along this axis, all other coordinates zero: no pooling.
    result = {
        "offsets": delta, "pairs_per_offset": counts,
        "measured_mean": mean.tolist(), "measured_std": std.tolist(),
        "predicted_mean": prediction.tolist(),
        "per_query_r_mean": torch.nanmean(r, dim=(1,2)).tolist(),
        "per_query_r_p10": torch.nanquantile(r.flatten(1), .1, dim=1).tolist(),
        "per_query_r_median": torch.nanquantile(r.flatten(1), .5, dim=1).tolist(),
        "per_query_r_p90": torch.nanquantile(r.flatten(1), .9, dim=1).tolist(),
        "valid_queries": torch.isfinite(r).sum((1,2)).tolist(),
        "total_queries_per_head": math.prod(grid),
        "curve_r": pearson_rows(mean, prediction).tolist(),
        "origin_query_measured": actual[:,0,0].tolist(),
        "origin_query_predicted": predicted[:,0,0].tolist(),
        "origin_query_offsets": [-i for i in range(extent)],
    }
    return result


def analyze(q, k, cos, sin, grid, pair_axes):
    """Analyze one conditional forward, one layer, all heads, all tokens."""
    qf, kf = q.float(), k.float()
    qr = rotate(qf, cos, sin, q.dtype)
    kr = rotate(kf, cos, sin, k.dtype)
    rq, nq, cq = pair_stats(qf)
    rk, nk, ck = pair_stats(kf)
    rqp, _, _ = pair_stats(qr)
    rkp, _, _ = pair_stats(kr)
    score = nq*nk
    center_qr = rotate(cq[:,None].expand_as(qf), cos, sin)
    center_kr = rotate(ck[:,None].expand_as(kf), cos, sin)
    result = {"tokens": q.shape[1], "heads": q.shape[0], "head_dim": q.shape[2],
              "q_pair_mrl_pre": rq.tolist(), "k_pair_mrl_pre": rk.tolist(),
              "q_pair_mrl_post": rqp.tolist(), "k_pair_mrl_post": rkp.tolist(),
              "band_score": score.tolist(), "center_q": cq.tolist(), "center_k": ck.tolist(),
              "q_whole_mrl": (cq.norm(dim=-1)/qf.norm(dim=-1).mean(1)).tolist(),
              "k_whole_mrl": (ck.norm(dim=-1)/kf.norm(dim=-1).mean(1)).tolist(),
              "axes": {}}
    for axis, name in enumerate(AXES):
        idx = torch.tensor([i for i,a in enumerate(pair_axes) if a == name], device=q.device)
        assert len(idx)>0
        dominant = idx[score[:,idx].argmax(-1)]
        head = torch.arange(q.shape[0], device=q.device)
        item = {"pair_indices": idx.tolist(), "dominant_pair": dominant.tolist(),
                "q_dom_mrl_pre": rq[head,dominant].tolist(), "k_dom_mrl_pre": rk[head,dominant].tolist(),
                "q_dom_mrl_post": rqp[head,dominant].tolist(), "k_dom_mrl_post": rkp[head,dominant].tolist(),
                "q_all_axis_pairs_mrl_mean": rq[:,idx].mean(1).tolist(),
                "k_all_axis_pairs_mrl_mean": rk[:,idx].mean(1).tolist()}
        item.update(axis_logits(qr, kr, center_qr, center_kr, grid, axis))
        result["axes"][name] = item
    return result
