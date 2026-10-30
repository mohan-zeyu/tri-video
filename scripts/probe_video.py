"""Probe pre-RoPE Q/K geometry in video diffusion transformers.

Reference: TriAttention (arXiv 2604.04921), Observation 3.1 — in causal LLMs,
pre-RoPE Q and K concentrate around non-zero centers (MRL = ||E[x]||/E[||x||]
-> 1), stable across positions/contexts. This probe asks whether video DiTs
share that geometry:

  intra concentration  MRL of Q and K across tokens, per (layer, head), plus
                       the paper's per-frequency-band MRL (rotary pairs)
  Q<->K alignment      cos(E[q], E[k]); mean same-position cos(q_t, k_t);
                       exact E_{t,s} cos(q_t, k_s) over all token pairs
  contrast             same statistics after applying the model's own RoPE
  stability            center direction across two noise seeds / text inputs
  validation           attention output recomputed from captured pre-RoPE
                       tensors + replicated rotation must match the module's
                       real output (proves the capture point is correct)

Capture point = output of norm_q/norm_k (Wan, LTX) or to_q/to_k (CogVideoX-2b
has no QK-norm) — exactly the tensors that enter apply_rotary_emb inside the
attention processor. CogVideoX rotates only the video slice of the joint
[text; video] stream, so video tokens are sliced out for both pre/post stats.

Run inside this repo's venv:  python probe_video.py --model wan --out ../out/wan.json
"""
import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

SUBSAMPLE = 512  # tokens kept per (layer, head) for histograms


# ---------------------------------------------------------------- metrics
def head_stats(q, k):
    """q, k: [H, T, D] float32. Returns dict of [H] tensors, pre- or post-RoPE."""
    T = q.shape[1]
    mq, mk = q.mean(1), k.mean(1)
    qn, kn = q.norm(dim=-1), k.norm(dim=-1)
    qh = q / qn.clamp_min(1e-12).unsqueeze(-1)
    kh = k / kn.clamp_min(1e-12).unsqueeze(-1)
    sqh, skh = qh.sum(1), kh.sum(1)
    half = T // 2
    return {
        "q_mrl": mq.norm(dim=-1) / qn.mean(1),
        "k_mrl": mk.norm(dim=-1) / kn.mean(1),
        "q_mean_norm": qn.mean(1), "k_mean_norm": kn.mean(1),
        "q_center_norm": mq.norm(dim=-1), "k_center_norm": mk.norm(dim=-1),
        "qk_center_cos": F.cosine_similarity(mq, mk, dim=-1),
        # exact cross expectation over all (t, s) pairs via sums of unit vectors
        "qk_cross_cos": (sqh * skh).sum(-1) / (T * T),
        "qk_same_pos_cos": (qh * kh).sum(-1).mean(1),
        "q_split_cos": F.cosine_similarity(q[:, :half].mean(1), q[:, half:].mean(1), dim=-1),
        "k_split_cos": F.cosine_similarity(k[:, :half].mean(1), k[:, half:].mean(1), dim=-1),
    }


def pair_mrl(x):
    """Paper-style per-frequency-band MRL; all three models use interleaved
    rotary pairs (2i, 2i+1). x: [H, T, D] -> [H, D/2]."""
    z = x.unflatten(-1, (-1, 2))
    return z.mean(1).norm(dim=-1) / z.norm(dim=-1).mean(1).clamp_min(1e-12)


# ---------------------------------------------------------------- adapters
class WanAdapter:
    name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    frames, height, width = 45, 480, 480  # latent (12, 60, 60) -> 10800 tokens
    text_len, text_dim = 512, 4096        # umt5 feature dim (config text_dim)
    t_high, t_mid, t_low = 1000, 500, 100
    has_rope = True

    def load(self):
        from diffusers import WanTransformer3DModel
        self.model = WanTransformer3DModel.from_pretrained(
            self.name, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        return self.model

    def blocks(self):
        return list(self.model.blocks)

    def self_attn(self, i):
        return self.model.blocks[i].attn1

    def rope_module(self):
        return self.model.rope

    def qk_modules(self, attn):
        return attn.norm_q, attn.norm_k, "post_qknorm"

    def rope_apply(self, q, freqs):
        """q [H, T, D] float -> rotated, matching WanAttnProcessor (bf16 math)."""
        cos, sin = freqs[0].float(), freqs[1].float()          # [T, 2D]
        x = q.permute(1, 0, 2)[None]                            # [1, T, H, D]
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = cos[..., 0::2], sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out[0].permute(1, 0, 2)

    def make_inputs(self, timestep, text_mode, seed, device):
        g = torch.Generator(device=device).manual_seed(seed)
        f, h, w = (self.frames - 1) // 4 + 1, self.height // 8, self.width // 8
        lat = torch.randn(1, 16, f, h, w, generator=g, device=device).to(torch.bfloat16)
        text = torch.zeros(1, self.text_len, self.text_dim, device=device, dtype=torch.bfloat16)
        if text_mode == "randn":
            text = torch.randn(text.shape, generator=g, device=device).to(torch.bfloat16)
        return dict(hidden_states=lat, timestep=torch.tensor([timestep], device=device, dtype=torch.bfloat16),
                    encoder_hidden_states=text)

    def validate(self, attn, box, freqs):
        """Recompute attn1 (self, no mask, no img ctx in T2V) from captures."""
        H = attn.heads
        q = box["q"].unflatten(2, (H, -1)).transpose(1, 2)     # [1, H, T, D] bf16
        k = box["k"].unflatten(2, (H, -1)).transpose(1, 2)
        v = box["v"].unflatten(2, (H, -1)).transpose(1, 2)
        qr = self.rope_apply(q[0].float(), freqs).to(q.dtype)[None]
        kr = self.rope_apply(k[0].float(), freqs).to(k.dtype)[None]
        o = F.scaled_dot_product_attention(qr, kr, v)          # default scale 1/sqrt(D)
        o = o.transpose(1, 2).flatten(2, 3)
        o = attn.to_out[1](attn.to_out[0](o))
        return (o - box["out"]).abs().max().item(), box["out"].abs().max().item()


class LTXAdapter:
    name = "Lightricks/LTX-Video"
    # 257f 512x832 -> latent (33, 16, 26) -> 3432 tokens (LTX VAE /32 spatial, /8 temporal)
    frames, height, width = 257, 512, 832
    text_len, text_dim = 128, 4096
    t_high, t_mid, t_low = 1000, 500, 100
    has_rope = True

    def load(self):
        from diffusers import LTXVideoTransformer3DModel
        self.model = LTXVideoTransformer3DModel.from_pretrained(
            self.name, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        return self.model

    def blocks(self):
        return list(self.model.transformer_blocks)

    def self_attn(self, i):
        return self.model.transformer_blocks[i].attn1

    def rope_module(self):
        return self.model.rope

    def qk_modules(self, attn):
        return attn.norm_q, attn.norm_k, "post_qknorm"

    def rope_apply(self, q, freqs):
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb as rope
        H, T, D = q.shape
        out = rope(q.permute(1, 0, 2).reshape(1, T, H * D), freqs)   # fp32 math inside
        return out.reshape(T, H, D).permute(1, 0, 2).float()

    def make_inputs(self, timestep, text_mode, seed, device):
        # this LTX revision takes pre-patchified tokens [B, S, C]; patch (1,1,1)
        g = torch.Generator(device=device).manual_seed(seed)
        f = (self.frames - 1) // 8 + 1
        h, w = self.height // 32, self.width // 32
        in_ch = self.model.config.in_channels
        lat = torch.randn(1, in_ch, f, h, w, generator=g, device=device).to(torch.bfloat16)
        tokens = lat.permute(0, 2, 3, 4, 1).flatten(1, 3)      # [1, f*h*w, C]
        text = torch.zeros(1, self.text_len, self.text_dim, device=device, dtype=torch.bfloat16)
        if text_mode == "randn":
            text = torch.randn(text.shape, generator=g, device=device).to(torch.bfloat16)
        return dict(hidden_states=tokens, timestep=torch.tensor([timestep], device=device, dtype=torch.bfloat16),
                    encoder_hidden_states=text, encoder_attention_mask=None,
                    num_frames=f, height=h, width=w,
                    rope_interpolation_scale=(8 / 25, 32.0, 32.0))

    def validate(self, attn, box, freqs):
        H = attn.heads
        q = box["q"].unflatten(2, (H, -1)).transpose(1, 2)
        k = box["k"].unflatten(2, (H, -1)).transpose(1, 2)
        v = box["v"].unflatten(2, (H, -1)).transpose(1, 2)
        qr = self.rope_apply(q[0].float(), freqs).to(q.dtype)[None]
        kr = self.rope_apply(k[0].float(), freqs).to(k.dtype)[None]
        o = F.scaled_dot_product_attention(qr, kr, v)
        o = o.transpose(1, 2).flatten(2, 3)
        o = attn.to_out[1](attn.to_out[0](o))
        return (o - box["out"]).abs().max().item(), box["out"].abs().max().item()


class CogAdapter:
    """CogVideoX-2b: this diffusers checkpoint runs WITHOUT RoPE
    (use_rotary_positional_embeddings=False; sinusoidal pos-emb added in
    patch_embed) — it serves as the no-RoPE control. QK LayerNorm exists and
    its modules see the 4D [B, H, S, D] joint [text; video] stream."""
    name = "zai-org/CogVideoX-2b"
    # 61f 480x720 -> latent (16, 60, 90); spatial patch 2 -> 16*30*45 video tokens
    frames, height, width = 61, 480, 720
    text_len, text_dim = 226, 4096
    t_high, t_mid, t_low = 990, 500, 100
    has_rope = False

    def load(self):
        from diffusers import CogVideoXTransformer3DModel
        self.model = CogVideoXTransformer3DModel.from_pretrained(
            self.name, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        return self.model

    def blocks(self):
        return list(self.model.transformer_blocks)

    def self_attn(self, i):
        return self.model.transformer_blocks[i].attn1

    def rope_module(self):
        return None

    def qk_modules(self, attn):
        # LayerNorm QK-norm; its input/output are [B, H, S, D] over the joint stream
        return attn.norm_q, attn.norm_k, "post_qknorm_joint_stream"

    def rope_apply(self, q, freqs):
        raise RuntimeError("CogVideoX-2b checkpoint runs without RoPE")

    def make_inputs(self, timestep, text_mode, seed, device):
        g = torch.Generator(device=device).manual_seed(seed)
        f = (self.frames - 1) // 4 + 1
        h, w = self.height // 8, self.width // 8
        in_ch = self.model.config.in_channels
        lat = torch.randn(1, in_ch, f, h, w, generator=g, device=device).to(torch.bfloat16)
        text = torch.zeros(1, self.text_len, self.text_dim, device=device, dtype=torch.bfloat16)
        if text_mode == "randn":
            text = torch.randn(text.shape, generator=g, device=device).to(torch.bfloat16)
        return dict(hidden_states=lat, timestep=torch.tensor([timestep], device=device, dtype=torch.bfloat16),
                    encoder_hidden_states=text)

    def validate(self, attn, box, freqs):
        """No RoPE in this checkpoint: plain SDPA over the joint stream."""
        tl = self.text_len
        q, k = box["q"], box["k"]                              # [1, H, S, D]
        _, H, S, D = q.shape
        v = box["v"].unflatten(2, (H, D)).transpose(1, 2)      # to_v gives [1, S, C]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).flatten(2, 3)
        o = attn.to_out[1](attn.to_out[0](o))
        video_out = o[:, tl:]
        real = box["out"][0] if isinstance(box["out"], tuple) else box["out"]
        return (video_out - real).abs().max().item(), real.abs().max().item()


ADAPTERS = {"wan": WanAdapter, "ltx": LTXAdapter, "cog": CogAdapter}


# ---------------------------------------------------------------- probe core
def probe(which, out_path, device="cuda"):
    import diffusers

    torch.manual_seed(0)
    ad = ADAPTERS[which]()
    model = ad.load().to(device)
    n_layers = len(ad.blocks())
    attn0 = ad.self_attn(0)
    heads = attn0.heads

    result = {
        "adapter": which, "model": ad.name, "torch": torch.__version__,
        "diffusers": diffusers.__version__, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0), "dtype": "bfloat16",
        "layers": n_layers, "heads": heads, "frames": ad.frames,
        "height": ad.height, "width": ad.width, "runs": [], "validation": {},
    }

    freqs_box = {}
    handles = []
    if ad.has_rope:
        handles.append(ad.rope_module().register_forward_hook(
            lambda m, x, y: freqs_box.update(freqs=[f.detach() for f in y])))

    # per-layer stash filled by projection hooks, drained by the attn-module hook
    boxes = {i: {} for i in range(n_layers)}
    val_layers = {0, n_layers // 2, n_layers - 1}

    def grab(box, name):
        def hook(m, x, y):
            box[name] = y.detach() if torch.is_tensor(y) else tuple(t.detach() for t in y)
        return hook

    for i in range(n_layers):
        attn = ad.self_attn(i)
        qm, km, mode = ad.qk_modules(attn)
        handles.append(qm.register_forward_hook(grab(boxes[i], "q")))
        handles.append(km.register_forward_hook(grab(boxes[i], "k")))
        if i in val_layers:
            handles.append(attn.to_v.register_forward_hook(grab(boxes[i], "v")))
            handles.append(attn.register_forward_hook(grab(boxes[i], "out")))
        result.setdefault("capture_stage", {})[str(i)] = mode

    def forward(timestep, text_mode, seed):
        model(**ad.make_inputs(timestep, text_mode, seed, device))

    def drain(tag):
        """Compute stats for every layer from this forward's captures."""
        rows = {}
        for i in range(n_layers):
            box = boxes[i]
            q_raw, k_raw = box.pop("q"), box.pop("k")
            H = ad.self_attn(i).heads
            if which == "cog":
                # norm_q output is [1, H, S, D] over the joint [text; video]
                # stream; keep the video tail
                q = q_raw[0, :, ad.text_len:].float()                    # [H, T, D]
                k = k_raw[0, :, ad.text_len:].float()
            else:
                D = q_raw.shape[-1] // H
                q = q_raw.unflatten(2, (H, D))[0].permute(1, 0, 2).float()   # [H, T, D]
                k = k_raw.unflatten(2, (H, D))[0].permute(1, 0, 2).float()
            T = q.shape[1]
            pre = head_stats(q, k)
            row = {"tokens": T}
            for key in pre:
                row[key] = [round(v, 6) for v in pre[key].tolist()]
            if ad.has_rope:
                freqs = freqs_box["freqs"]
                assert freqs[0].shape[-2] == T, (freqs[0].shape, T)
                post = head_stats(ad.rope_apply(q, freqs), ad.rope_apply(k, freqs))
                for key in post:
                    row[key + "_post"] = [round(v, 6) for v in post[key].tolist()]
            row["q_pair_mrl_mean"] = [round(v, 6) for v in pair_mrl(q).mean(-1).tolist()]
            row["k_pair_mrl_mean"] = [round(v, 6) for v in pair_mrl(k).mean(-1).tolist()]
            # full centers + a token subsample for later analysis
            row["_center_q"] = q.mean(1).to(torch.bfloat16).cpu()
            row["_center_k"] = k.mean(1).to(torch.bfloat16).cpu()
            idx = torch.randperm(T, device=q.device)[:SUBSAMPLE]
            row["_subsample_q"] = q[:, idx].to(torch.bfloat16).cpu()
            row["_subsample_k"] = k[:, idx].to(torch.bfloat16).cpu()
            rows[i] = row
            if i in val_layers and "v" in box and "out" in box:
                err, scale = ad.validate(ad.self_attn(i),
                                         {"q": q_raw, "k": k_raw, "v": box["v"], "out": box["out"]},
                                         freqs_box.get("freqs"))
                result["validation"][f"layer{i}"] = {
                    "max_abs_err": err, "output_scale": scale,
                    "rel_err": err / max(scale, 1e-9)}
                box.pop("v"); box.pop("out")
            del q, k
        return rows

    # ---- sweeps ------------------------------------------------------------
    t0 = time.time()
    plan = [
        ("t_mid-zeros-s0", ad.t_mid, "zeros", 0),       # primary reference run
        ("t_high-zeros-s0", ad.t_high, "zeros", 0),
        ("t_low-zeros-s0", ad.t_low, "zeros", 0),
        ("t_mid-randn-s0", ad.t_mid, "randn", 0),       # text sensitivity
        ("t_mid-zeros-s1", ad.t_mid, "zeros", 1),       # noise seed stability
    ]
    all_rows = {}
    for tag, ts, tm, sd in plan:
        forward(ts, tm, sd)
        rows = drain(tag)
        all_rows[tag] = rows
        r0 = rows[0]
        result["runs"].append({
            "tag": tag, "timestep": ts, "text": tm, "seed": sd,
            "tokens": r0["tokens"],
            "by_layer": {str(i): {k: v for k, v in r.items() if not k.startswith("_")}
                         for i, r in rows.items()}})
        mrl_q = sum(sum(r["q_mrl"]) / len(r["q_mrl"]) for r in rows.values()) / n_layers
        mrl_k = sum(sum(r["k_mrl"]) / len(r["k_mrl"]) for r in rows.values()) / n_layers
        print(f"[{which}] {tag}: T={r0['tokens']} heads={heads} "
              f"mean q_mrl={mrl_q:.3f} k_mrl={mrl_k:.3f}", flush=True)
        torch.cuda.empty_cache()

    # cross-seed / cross-text center stability at t_mid (full centers)
    for other, label in [("t_mid-zeros-s1", "vs_seed"), ("t_mid-randn-s0", "vs_text")]:
        cq = [F.cosine_similarity(all_rows["t_mid-zeros-s0"][i]["_center_q"].float(),
                                  all_rows[other][i]["_center_q"].float(), dim=-1).mean().item()
              for i in range(n_layers)]
        ck = [F.cosine_similarity(all_rows["t_mid-zeros-s0"][i]["_center_k"].float(),
                                  all_rows[other][i]["_center_k"].float(), dim=-1).mean().item()
              for i in range(n_layers)]
        result[f"center_stability_{label}"] = {"q": cq, "k": ck}

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    for h in handles:
        h.remove()
    del model
    gc.collect(); torch.cuda.empty_cache()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({tag: {i: {k: v for k, v in r.items() if k.startswith("_")}
                      for i, r in rows.items()} for tag, rows in all_rows.items()},
               out.with_suffix("").as_posix() + "-tensors.pt")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out} in {result['elapsed_seconds']}s; validation: {result['validation']}", flush=True)
    return result


def summarize(result_path):
    """Headline numbers for the mid-timestep reference run + paper comparison."""
    r = json.loads(Path(result_path).read_text())
    run = next(x for x in r["runs"] if x["tag"] == "t_mid-zeros-s0")
    keys = ["q_mrl", "k_mrl", "q_pair_mrl_mean", "k_pair_mrl_mean",
            "qk_center_cos", "qk_cross_cos", "qk_same_pos_cos",
            "q_mrl_post", "k_mrl_post", "qk_center_cos_post", "qk_cross_cos_post",
            "q_split_cos", "k_split_cos"]
    agg = {}
    for key in keys:
        vals = [v for l in run["by_layer"].values() for v in l.get(key, [])]
        if not vals:
            continue
        t = torch.tensor(vals)
        agg[key] = {"mean": round(t.mean().item(), 4), "median": round(t.median().item(), 4),
                    "p10": round(t.quantile(.1).item(), 4), "p90": round(t.quantile(.9).item(), 4),
                    "frac>0.95": round((t > 0.95).float().mean().item(), 4),
                    "frac>0.9": round((t > 0.9).float().mean().item(), 4)}
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    probe(args.model, args.out)
    print(json.dumps(summarize(args.out), indent=2))
