"""Pre-RoPE Q/K geometry inside a REAL text-to-video denoising trajectory.

Upgrades over probe_video.py (noise latents + zero text):
  - real prompt -> real T5/UMT5 embeddings (the actual conditioning stream)
  - real scheduler trajectory: latents are progressively denoised video, not
    pure noise at every step (tests timestep-insensitivity honestly)
  - same capture points (norm_q/norm_k, exactly the tensors entering
    apply_rotary_emb) and the same statistics + bit-exact validation

Uses callback_on_step_end to drain per-step captures. Statistics kept for
every KEEP_STEPS-th step to bound output size.

Usage: python pipeline_probe.py --model wan --prompt "..." --out ../out/wan-pipeline.json
"""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys_path = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(sys_path))
from probe_video import head_stats, pair_mrl  # noqa: E402

torch.set_grad_enabled(False)

KEEP_EVERY = 1          # drain every step (steps are few)
SUBSAMPLE = 512


class PipeWan:
    name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    frames, height, width = 45, 480, 480
    steps, guidance = 20, 5.0
    has_rope = True

    def load(self, device):
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(local_snapshot(self.name), torch_dtype=torch.bfloat16)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.transformer = pipe.transformer
        return pipe

    def attns(self):
        return [(i, b.attn1) for i, b in enumerate(self.transformer.blocks)]

    def rope_module(self):
        return self.transformer.rope

    def qk_modules(self, attn):
        return attn.norm_q, attn.norm_k, "post_qknorm"

    def rope_apply(self, q, freqs):
        cos, sin = freqs[0].float(), freqs[1].float()      # [1, T, 1, 2D]
        x = q.permute(1, 0, 2)[None]                        # [1, T, H, D]
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = cos[..., 0::2], sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out[0].permute(1, 0, 2)

    def run(self, pipe, prompt, device):
        g = torch.Generator(device=device).manual_seed(0)
        return pipe(prompt=prompt, negative_prompt="", num_frames=self.frames,
                    height=self.height, width=self.width,
                    num_inference_steps=self.steps, guidance_scale=self.guidance,
                    generator=g).frames[0]

    def validate(self, attn, box, freqs):
        """Full-batch (CFG: [neg, pos]) recompute of attn1 output."""
        H = attn.heads
        B = box["q"].shape[0]
        qs, ks, vs = [], [], []
        for b in range(B):
            qs.append(box["q"][b].unflatten(1, (H, -1)).transpose(0, 1))
            ks.append(box["k"][b].unflatten(1, (H, -1)).transpose(0, 1))
            vs.append(box["v"][b].unflatten(1, (H, -1)).transpose(0, 1))
        q = torch.stack(qs)
        k = torch.stack(ks)
        v = torch.stack(vs)
        qr = torch.stack([self.rope_apply(q[b].float(), freqs) for b in range(B)]).to(q.dtype)
        kr = torch.stack([self.rope_apply(k[b].float(), freqs) for b in range(B)]).to(k.dtype)
        o = F.scaled_dot_product_attention(qr, kr, v)
        o = o.transpose(1, 2).flatten(2, 3)
        o = attn.to_out[1](attn.to_out[0](o))
        return (o - box["out"]).abs().max().item(), box["out"].abs().max().item()


class PipeCog:
    """CogVideoX-2b joint-stream model (no RoPE in this checkpoint)."""
    name = "zai-org/CogVideoX-2b"
    frames, height, width = 49, 480, 720
    steps, guidance = 30, 6.0
    has_rope = False

    def load(self, device):
        from diffusers import CogVideoXPipeline
        pipe = CogVideoXPipeline.from_pretrained(local_snapshot(self.name), torch_dtype=torch.bfloat16)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.transformer = pipe.transformer
        return pipe

    def attns(self):
        return [(i, b.attn1) for i, b in enumerate(self.transformer.transformer_blocks)]

    def rope_module(self):
        return None

    def qk_modules(self, attn):
        return attn.norm_q, attn.norm_k, "post_qknorm_joint_stream"

    def run(self, pipe, prompt, device):
        g = torch.Generator(device=device).manual_seed(0)
        return pipe(prompt=prompt, num_frames=self.frames, height=self.height,
                    width=self.width, num_inference_steps=self.steps,
                    guidance_scale=self.guidance, generator=g).frames[0]

    def preencode(self, pipe, prompt, device):
        """Encode on CPU and free the T5 before the denoising loop (~4.8GB)."""
        pipe.text_encoder.to("cpu")
        pos, neg = pipe.encode_prompt(prompt, device="cpu", num_videos_per_prompt=1,
                                      do_classifier_free_guidance=True, negative_prompt="")
        torch.cuda.empty_cache()
        return pos.to(device), neg.to(device)

    def validate(self, attn, box, freqs):
        tl = self.text_len
        q, k = box["q"], box["k"]                            # [1, H, S, D]
        _, H, S, D = q.shape
        v = box["v"].unflatten(2, (H, D)).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).flatten(2, 3)
        o = attn.to_out[1](attn.to_out[0](o))
        real = box["out"][0] if isinstance(box["out"], tuple) else box["out"]
        return (o[:, tl:] - real).abs().max().item(), real.abs().max().item()


PIPES = {"wan": PipeWan, "cog": PipeCog}


def local_snapshot(repo_id):
    """Resolve repo id to its cached snapshot dir (bypasses hub metadata)."""
    root = Path("/mnt/data/liuzeyu/hf-cache/hub") / f"models--{repo_id.replace('/', '--')}"
    rev = (root / "refs" / "main").read_text().strip()
    snap = root / "snapshots" / rev
    assert (snap / "model_index.json").exists(), snap
    return str(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PIPES))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    ad = PIPES[args.model]()
    pipe = ad.load(device)
    attns = ad.attns()
    n_layers = len(attns)

    result = {"model": ad.name, "prompt": args.prompt, "frames": ad.frames,
              "height": ad.height, "width": ad.width, "steps": ad.steps,
              "torch": torch.__version__, "dtype": "bfloat16",
              "by_step": {}, "validation": {}}

    freqs_box = {}
    handles = []
    if ad.has_rope:
        handles.append(ad.rope_module().register_forward_hook(
            lambda m, x, y: freqs_box.update(freqs=[f.detach() for f in y])))

    boxes = {i: {} for i in range(n_layers)}
    val_layers = {0, n_layers // 2, n_layers - 1}

    def grab(box, name):
        def hook(m, x, y):
            box[name] = y.detach() if torch.is_tensor(y) else tuple(t.detach() for t in y)
        return hook

    for i, attn in attns:
        qm, km, _ = ad.qk_modules(attn)
        handles.append(qm.register_forward_hook(grab(boxes[i], "q")))
        handles.append(km.register_forward_hook(grab(boxes[i], "k")))
        if i in val_layers:
            handles.append(attn.to_v.register_forward_hook(grab(boxes[i], "v")))
            handles.append(attn.register_forward_hook(grab(boxes[i], "out")))

    step_box = {"i": 0}

    def drain(step_idx, validate=False):
        rows = {}
        for i, attn in attns:
            box = boxes[i]
            q_raw, k_raw = box.get("q"), box.get("k")
            if q_raw is None:
                continue
            H = attn.heads
            B = q_raw.shape[0]                     # 2 under CFG: [neg, pos]
            if args.model == "cog":
                tl = 226
                ad.text_len = tl
                q = q_raw[B - 1, :, tl:].float()   # positive-prompt branch
                k = k_raw[B - 1, :, tl:].float()
                D = q.shape[-1]
            else:
                D = q_raw.shape[-1] // H
                q = q_raw[B - 1].unflatten(1, (H, D)).permute(1, 0, 2).float()
                k = k_raw[B - 1].unflatten(1, (H, D)).permute(1, 0, 2).float()
            T = q.shape[1]
            pre = head_stats(q, k)
            row = {"tokens": T}
            for key in pre:
                row[key] = [round(v, 6) for v in pre[key].tolist()]
            if ad.has_rope:
                freqs = freqs_box["freqs"]
                post = head_stats(ad.rope_apply(q, freqs), ad.rope_apply(k, freqs))
                for key in post:
                    row[key + "_post"] = [round(v, 6) for v in post[key].tolist()]
            pm_q, pm_k = pair_mrl(q), pair_mrl(k)
            row["q_pair_mrl_mean"] = [round(v, 6) for v in pm_q.mean(-1).tolist()]
            row["k_pair_mrl_mean"] = [round(v, 6) for v in pm_k.mean(-1).tolist()]
            # paper-style dominant band: top pair by ||q_f||*||k_f||
            zq = q.unflatten(-1, (-1, 2))
            zk = k.unflatten(-1, (-1, 2))
            imp = zq.norm(dim=-1).mean(1) * zk.norm(dim=-1).mean(1)   # [H, P]
            dom = imp.argmax(-1)
            row["q_dom_mrl"] = [round(v, 6) for v in pm_q.gather(-1, dom[:, None]).squeeze(-1).tolist()]
            row["k_dom_mrl"] = [round(v, 6) for v in pm_k.gather(-1, dom[:, None]).squeeze(-1).tolist()]
            row["_center_q"] = q.mean(1).to(torch.bfloat16).cpu()
            rows[i] = row
            if validate and i in val_layers and "v" in box and "out" in box:
                err, scale = ad.validate(attn, {"q": q_raw, "k": k_raw,
                                                "v": box["v"], "out": box["out"]},
                                         freqs_box.get("freqs"))
                result["validation"][f"step{step_idx}_layer{i}"] = {
                    "max_abs_err": err, "output_scale": scale,
                    "rel_err": err / max(scale, 1e-9)}
            del q, k
        result["by_step"][str(step_idx)] = {
            str(i): {k: v for k, v in r.items() if not k.startswith("_")}
            for i, r in rows.items()}
        result.setdefault("_centers", {})[str(step_idx)] = {
            str(i): r["_center_q"] for i, r in rows.items()}

    def on_step_end(pipe, step_idx, timestep, callback_kwargs):
        if step_idx % KEEP_EVERY == 0:
            drain(step_idx, validate=(step_idx == 0))
            step_box["i"] = step_idx + 1
        return callback_kwargs

    t0 = time.time()
    video = ad.run_with_callback(pipe, args.prompt, device, on_step_end)
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    for h in handles:
        h.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.pop("_centers"), str(out).replace(".json", "") + "-centers.pt")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out, f"in {result['elapsed_seconds']}s; validation:",
          json.dumps(result["validation"]), flush=True)


def _run_with_callback(self, pipe, prompt, device, cb):
    """Bind callback into the model-specific pipeline call."""
    if self.name.startswith("Wan"):
        g = torch.Generator(device=device).manual_seed(0)
        return pipe(prompt=prompt, negative_prompt="", num_frames=self.frames,
                    height=self.height, width=self.width,
                    num_inference_steps=self.steps, guidance_scale=self.guidance,
                    generator=g, callback_on_step_end=cb, output_type="latent")
    g = torch.Generator(device=device).manual_seed(0)
    if hasattr(self, "preencode"):
        pos, neg = self.preencode(pipe, prompt, device)
        return pipe(num_frames=self.frames, height=self.height, width=self.width,
                    num_inference_steps=self.steps, guidance_scale=self.guidance,
                    generator=g, prompt_embeds=pos, negative_prompt_embeds=neg,
                    callback_on_step_end=cb, output_type="latent")
    return pipe(prompt=prompt, num_frames=self.frames, height=self.height,
                width=self.width, num_inference_steps=self.steps,
                guidance_scale=self.guidance, generator=g,
                callback_on_step_end=cb, output_type="latent")


for _cls in PIPES.values():
    _cls.run_with_callback = _run_with_callback


if __name__ == "__main__":
    main()
