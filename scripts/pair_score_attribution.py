"""How much of a Wan self-attention score each RoPE frequency pair contributes.

`attribute` analyzes ONE cell group: one prompt, one denoising step, one CFG branch, one layer, every head
separately. It never pools prompts, steps, branches, layers or heads, and it uses every video token as a query
and every video token as a key (all three axes, not axis lines).

The post-RoPE logit of query i and key j in one head is an exact sum over the 64 two-dimensional pairs f of the
128-dim head (d = 128):

    s[i,j] = sum_f c_f[i,j],        c_f[i,j] = <qr_i[f], kr_j[f]> / sqrt(d)

qr/kr are the normalized pre-RoPE Q/K (attn1.norm_q / norm_k outputs) rotated with the model's own angles and
rounded to the model's dtype, as in full_sweep_metrics.py; dot products are fp32 (attention) or fp64 (moments).

Softmax over keys ignores a per-query constant, so every "share" below is relative to the query's own mean over
keys (row-centred). What one number aggregates is stated per field; nothing is a mean over cells.

Per pair f, returned for every head ([H, 64] unless noted):

  var_cov      sum_g G[f,g], with G[f,g] = mean_i Cov_j(c_f[i,.], c_g[i,.]) taken exactly over all N x N
               query-key combinations from second moments:
               G[f,g] = (1/d) sum_{a in f, b in g} Cov_j(kr)[a,b] * mean_i(qr qr^T)[a,b].
               sum_f var_cov = mean over queries of the population variance of the query's logit row
               (= row_var). var_cov / row_var is the pair's signed covariance share (sums to 1).
  var_own      G[f,f], the pair's own row-centred contribution variance (>= 0).
  pair_cov     G / row_var as [H, 64, 64] float16 (optional). For any pair set S, sum_{f,g in S} pair_cov is the
               exact relative logit-variance change of deleting S on both sides.
  adv          mean over queries of A_f[i] = sum_j p[i,j] c_f[i,j] - mean_j c_f[i,j], p = softmax_j s[i,.] over
               all keys (the head's real attention pattern up to fp32 arithmetic).
               sum_f A_f[i] = E_p[s] - E_uniform[s] >= 0. adv / sum_f adv is the attention-advantage share
               (ratio of sums over queries): which pairs lift the attended keys above the average key.
  top1         the same with p replaced by the single arg-max key (logit margin of the top key).
  mean_contribution  mean of c_f over all N x N combinations (uncentred). adv + mean_contribution is the pair's
               part of the attention-weighted mean raw logit, which includes the constant softmax ignores.
  var_share_quantiles, adv_share_quantiles   [H, 64, 5]: quantiles (QUANTILES) ACROSS QUERIES of the per-query
               share Cov_j(c_f[i,.], s[i,.]) / Var_j(s[i,.]) and A_f[i] / sum_g A_g[i]. These show the spread
               over tokens that var_cov and adv sum over.
  center_var_cov, center_var_own   the same G for the center model: Q/K replaced by their fp32 means over all
               tokens of this cell (the TriAttention trigonometric series), rotated at the real positions.
  center_cross_own    mean_i Cov_j(actual c_f, center c_f): with var_own and center_var_own it gives the pooled
               row-centred correlation between a pair's actual and center-predicted contribution.
  center_cross_cov    mean_i Cov_j(center c_f, actual s): summed over f and divided by row_var it is the share
               of logit variance linearly explained by the center prediction.
  energy_q/k   E_tokens ||x_f||^2;  norm_q/k  E_tokens ||x_f||;  center_norm_q/k  ||E_tokens x_f||  (pre-RoPE).
               norm_q * norm_k is the dominant-pair amplitude of the sweep; center_norm / norm is the pair MRL;
               energy and center norms give select_pairs.py's own-mean replacement error.

Per head ([H] unless noted): row_var, adv_total, top1_total, entropy and effective_keys (mean over queries of the
attention entropy and of its exponential), pmax (mean over queries of the largest attention weight), top1_self
(fraction of queries whose arg-max key is the query itself), row_var_quantiles and adv_total_quantiles ([H, 5],
across queries).
"""
import math

import torch

from full_sweep_metrics import pair_stats, rotate

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

SCOPE = {
    "cell": "one prompt, one denoising step, one CFG branch, one layer, one head; no pooling across any of them",
    "logit": "full 128-dim head, post-RoPE, divided by sqrt(128); exact sum of the 64 pair contributions",
    "queries": "every video token (attention/top-1 fields: every query_stride-th token)",
    "keys": "every video token on all three axes; not restricted to axis lines",
    "row_centring": "contributions are taken relative to each query's own mean over all keys",
    "variance_fields": "exact over all query x key combinations, float64 second moments",
    "attention_fields": "softmax over all keys of the fp32 logits; pooled fields are means over queries of "
                        "per-query sums, so their ratios are ratios of sums over queries",
    "quantile_fields": "quantiles across queries of per-query shares, levels " + str(QUANTILES),
    "centers": "fp32 mean of normalized pre-RoPE Q/K over all tokens of the same cell",
}


def pair_blocks(matrix):
    """[..., D, D] -> [..., D/2, D/2], summing each 2x2 pair block."""
    d = matrix.shape[-1]
    return matrix.reshape(*matrix.shape[:-2], d // 2, 2, d // 2, 2).sum((-3, -1))


def pair_sum(x):
    """[..., D] -> [..., D/2], summing the two coordinates of each pair."""
    return x.reshape(*x.shape[:-1], -1, 2).sum(-1)


def quantiles(x, dim, levels=QUANTILES):
    """Linear-interpolation quantiles along `dim` (no size limit); a new last dim indexes the levels."""
    ordered = x.sort(dim).values.movedim(dim, -1)
    position = torch.tensor(levels, dtype=torch.float64, device=x.device) * (ordered.shape[-1] - 1)
    low, high = position.floor().long(), position.ceil().long()
    weight = (position - low).to(ordered.dtype)
    return ordered[..., low] * (1 - weight) + ordered[..., high] * weight


def variance_decomposition(qr, kr, center_qr, center_kr):
    """Exact row-centred covariance of pair contributions over all queries x all keys.

    Returns G [H, 2P, 2P] float64 (actual pairs first, then center-model pairs) and Cov_j(kr) [H, D, D].
    """
    d = qr.shape[-1]
    zq = torch.cat((qr.double(), center_qr.double()), -1)
    zk = torch.cat((kr.double(), center_kr.double()), -1)
    n = zq.shape[1]
    second_q = zq.transpose(1, 2) @ zq / n
    mean_k = zk.mean(1)
    cov_k = zk.transpose(1, 2) @ zk / n - mean_k[:, :, None] * mean_k[:, None, :]
    return pair_blocks(cov_k * second_q) / d, cov_k[:, :d, :d]


def query_variance_shares(qr, cov_k):
    """Per query: covariance of each pair's contribution with the full logit over keys [H, N, P], row variance."""
    q64 = qr.double()
    cov = pair_sum(q64 * (q64 @ cov_k)) / qr.shape[-1]
    return cov, cov.sum(-1)


def attention_attribution(qr, kr, chunk=1350, query_stride=1):
    """Softmax- and top-1-weighted pair contributions relative to the mean key, for every analyzed query."""
    heads, n, d = qr.shape
    scale = 1 / math.sqrt(d)
    qr, kr = qr.float(), kr.float()
    key_t = kr.transpose(1, 2).contiguous()
    mean_key = kr.mean(1, keepdim=True)
    index = torch.arange(0, n, query_stride, device=qr.device)
    adv, top, entropy, pmax, self_top = [], [], [], [], []
    for start in range(0, len(index), chunk):
        ids = index[start:start + chunk]
        qc = qr[:, ids]
        logits = torch.matmul(qc, key_t) * scale                            # [H, B, N]
        arg = logits.argmax(-1)
        logp = torch.log_softmax(logits, -1)
        del logits
        p = logp.exp()
        entropy.append(-(p * logp).sum(-1))
        del logp
        pmax.append(p.max(-1).values)
        weighted_key = torch.matmul(p, kr)                                  # [H, B, D]
        del p
        adv.append(pair_sum(qc * (weighted_key - mean_key)) * scale)        # [H, B, P]
        top_key = torch.gather(kr, 1, arg[..., None].expand(-1, -1, d))
        top.append(pair_sum(qc * (top_key - mean_key)) * scale)
        self_top.append(arg == ids[None])
    entropy = torch.cat(entropy, 1).double()
    return {"adv_per_query": torch.cat(adv, 1).double(), "top1": torch.cat(top, 1).double().mean(1),
            "entropy": entropy.mean(1), "effective_keys": entropy.exp().mean(1),
            "pmax": torch.cat(pmax, 1).double().mean(1), "top1_self": torch.cat(self_top, 1).double().mean(1)}


def attribute(q, k, cos, sin, chunk=1350, query_stride=1, pair_cov=True):
    """All fields for one prompt/step/branch/layer. q, k: [H, N, D] normalized pre-RoPE; cos, sin: [N, D/2].

    Pair order is the model's: t pairs 0-21, h 22-42, w 43-63, angular frequency decreasing inside each axis.
    """
    qf, kf = q.float(), k.float()
    native = q.dtype if q.dtype != torch.float32 else None
    qr, kr = rotate(qf, cos, sin, native), rotate(kf, cos, sin, native)
    _, norm_q, cq = pair_stats(qf)
    _, norm_k, ck = pair_stats(kf)
    center_qr = rotate(cq[:, None].expand_as(qf), cos, sin)
    center_kr = rotate(ck[:, None].expand_as(kf), cos, sin)
    pairs, d = q.shape[-1] // 2, q.shape[-1]
    g, cov_k = variance_decomposition(qr, kr, center_qr, center_kr)
    actual, center, cross = g[:, :pairs, :pairs], g[:, pairs:, pairs:], g[:, :pairs, pairs:]
    row_var = actual.sum((1, 2))
    query_cov, query_var = query_variance_shares(qr, cov_k)
    attention = attention_attribution(qr, kr, chunk, query_stride)
    adv_query = attention.pop("adv_per_query")
    adv_total = adv_query.sum(-1)
    as_pairs = lambda x: x.double().reshape(*x.shape[:-1], pairs, 2)
    result = {
        "var_cov": actual.sum(2), "var_own": actual.diagonal(dim1=-2, dim2=-1), "row_var": row_var,
        "var_share_quantiles": quantiles(query_cov / query_var.clamp_min(1e-30)[..., None], 1),
        "row_var_quantiles": quantiles(query_var, 1),
        "adv": adv_query.mean(1), "adv_total": adv_total.mean(1),
        "adv_share_quantiles": quantiles(adv_query / adv_total.clamp_min(1e-30)[..., None], 1),
        "adv_total_quantiles": quantiles(adv_total, 1),
        "mean_contribution": pair_sum(qr.double().mean(1) * kr.double().mean(1)) / math.sqrt(d),
        "center_var_cov": center.sum(2), "center_var_own": center.diagonal(dim1=-2, dim2=-1),
        "center_cross_own": cross.diagonal(dim1=-2, dim2=-1), "center_cross_cov": cross.sum(1),
        "energy_q": as_pairs(qf).square().sum(-1).mean(1), "energy_k": as_pairs(kf).square().sum(-1).mean(1),
        "norm_q": norm_q.double(), "norm_k": norm_k.double(),
        "center_norm_q": as_pairs(cq).norm(dim=-1), "center_norm_k": as_pairs(ck).norm(dim=-1),
    }
    result.update(attention)
    result["top1_total"] = result["top1"].sum(-1)
    if pair_cov:
        result["pair_cov"] = (actual / row_var[:, None, None]).to(torch.float16)
    return result
