"""Dump Wan teacher trajectories for distillation, one GPU per worker, restartable.

For each prompt in configs/distill_prompts.json (optionally filtered by --split / --ids):
  traj/<id>/embeddings.pt   {"positive": [1,512,4096] bf16, "negative": [1,512,4096] bf16}
  traj/<id>/states.pt       [20, 16, 12, 60, 60] bf16: the latent ENTERING each denoising step (captured from
                            the transformer's conditional forward input), i.e. states[0] is the initial noise
  traj/<id>/meta.json       timesteps (20), seed, text, hashes, environment
For prompts with split == "calib", additionally (hooks on attn1.norm_q / norm_k, both CFG branches):
  traj/<id>/center-means.pt {"mean_q","mean_k": [30 layers, 20 steps, 2 branches, 12 heads, 44] fp32}
                            = mean over all 10,800 current video tokens of the normalized pre-RoPE HIGH coordinates
                            (same quantity scripts/center_replacement.py uses, but stored instead of applied)
Finally (any worker, when all calib prompts are done): traj/calibration.pt with the mean over calib prompts and
the cosine spread of per-prompt means around it, per pair group.

Settings fixed to the repo's benchmark protocol: 45 frames 480x480, 20 UniPC steps, CFG 5, empty negative
prompt, bf16 transformer, CUDA generator seeded per prompt. No VAE decode.
"""

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from full_sweep import save_torch, write_json
from learned_center import HIGH_DIMS

ROOT = Path(__file__).resolve().parents[1]
GRID = (12, 30, 30)
TOKENS = GRID[0] * GRID[1] * GRID[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Capture:
    def __init__(self, model, positive, negative, steps, calibrate):
        self.positive, self.negative, self.steps, self.calibrate = positive, negative, steps, calibrate
        self.states, self.timesteps = [], []
        self.branch, self.step = None, -1
        self.counts = {}
        if calibrate:
            self.mean_q = torch.zeros(30, steps, 2, 12, HIGH_DIMS.numel(), dtype=torch.float64)
            self.mean_k = torch.zeros_like(self.mean_q)
        self.handles = [model.register_forward_pre_hook(self.pre, with_kwargs=True)]
        if calibrate:
            for layer, block in enumerate(model.blocks):
                for side in ("q", "k"):
                    self.handles.append(getattr(block.attn1, "norm_" + side).register_forward_hook(self.hook(layer, side)))

    def pre(self, module, args, kwargs):
        text = kwargs["encoder_hidden_states"]
        if torch.equal(text, self.positive):
            self.branch = 0
            self.step += 1
            self.states.append(kwargs["hidden_states"].detach().to(torch.bfloat16).cpu())
            self.timesteps.append(float(kwargs["timestep"].flatten()[0]))
        elif torch.equal(text, self.negative):
            self.branch = 1
        else:
            raise RuntimeError("Unknown CFG branch")
        patch = module.config.patch_size
        grid = tuple(kwargs["hidden_states"].shape[i + 2] // patch[i] for i in range(3))
        if grid != GRID:
            raise RuntimeError(f"Unexpected grid {grid}")

    def hook(self, layer, side):
        def apply(module, inputs, output):
            if output.shape[1] != TOKENS:
                raise RuntimeError("Unexpected token population")
            key = (self.step, self.branch, layer, side)
            if key in self.counts:
                raise RuntimeError(f"Duplicate hook call {key}")
            self.counts[key] = 1
            high = output[0].float().view(TOKENS, 12, 128)[:, :, HIGH_DIMS]          # [tokens, heads, 44]
            target = self.mean_q if side == "q" else self.mean_k
            target[layer, self.step, self.branch] = high.mean(0, dtype=torch.float64).cpu()
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()


def finalize_calibration(traj, config):
    calib_ids = [p["id"] for p in config["prompts"] if p["split"] == "calib"]
    files = [traj / pid / "center-means.pt" for pid in calib_ids]
    if not all(f.exists() for f in files) or (traj / "calibration.pt").exists():
        return
    means = [torch.load(f) for f in files]
    mean_q = torch.stack([m["mean_q"] for m in means]).float()                 # [P, 30, 20, 2, 12, 44]
    mean_k = torch.stack([m["mean_k"] for m in means]).float()
    pooled_q, pooled_k = mean_q.mean(0), mean_k.mean(0)
    pairs = HIGH_DIMS.view(22, 2)
    groups = {"t": slice(0, 8), "h": slice(8, 15), "w": slice(15, 22)}
    spread = {}
    for side, per, pooled in (("q", mean_q, pooled_q), ("k", mean_k, pooled_k)):
        for name, sl in groups.items():
            a = per.view(*per.shape[:-1], 22, 2)[..., sl, :].flatten(-2)
            b = pooled.view(*pooled.shape[:-1], 22, 2)[..., sl, :].flatten(-2)
            cos = torch.nn.functional.cosine_similarity(a, b.unsqueeze(0).expand_as(a), dim=-1)
            spread[f"{side}_{name}_cos_prompt_vs_pooled_median"] = float(cos.median())
            spread[f"{side}_{name}_cos_prompt_vs_pooled_p10"] = float(cos.flatten().quantile(0.1))
            ratio = b.norm(dim=-1) / a.norm(dim=-1).mean(0).clamp_min(1e-12)
            spread[f"{side}_{name}_pooled_norm_over_mean_prompt_norm_median"] = float(ratio.median())
    save_torch(traj / "calibration.pt", {"mean_q": pooled_q, "mean_k": pooled_k, "prompts": calib_ids,
        "scope": "mean over calib prompts of per-prompt all-token means; separate layer/step/branch/head; HIGH 44 coords"})
    write_json(traj / "calibration-spread.json", {"prompts": len(calib_ids), "spread": spread,
        "scope": "cosine between each prompt's all-token mean and the pooled mean, per layer/step/branch/head, within pair group; median and 10th percentile over all cells"})
    print("CALIBRATION COMPLETE", len(calib_ids), json.dumps(spread)[:400], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/distill_prompts.json")
    parser.add_argument("--out", default="out/distill-traj-wan")
    parser.add_argument("--split", nargs="*", help="only these splits (calib/val/train)")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    traj = ROOT / args.out
    traj.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    from diffusers import WanPipeline
    import diffusers, transformers

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub" / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
    revision = (cache / "refs/main").read_text().strip() if (cache / "refs/main").exists() else "unknown"
    snapshot = Path(os.environ.get("WAN_SNAPSHOT", str(cache / "snapshots" / revision)))
    pipe = WanPipeline.from_pretrained(snapshot, vae=None, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    environment = {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                   "snapshot": revision, "gpu": torch.cuda.get_device_name(0), "hostname": socket.gethostname(), "pid": os.getpid(),
                   "config_sha256": digest(config_path), "code_sha256": {n: digest(ROOT / "scripts" / n) for n in ("dump_teacher_trajectories.py", "learned_center.py")}}
    done = 0
    for prompt in config["prompts"]:
        if args.split and prompt["split"] not in args.split:
            continue
        if args.ids and prompt["id"] not in args.ids:
            continue
        pdir = traj / prompt["id"]
        if (pdir / "meta.json").exists():
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        claim = pdir / "claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            started = time.time()
            pipe.text_encoder.to("cuda")
            positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt="", do_classifier_free_guidance=True,
                                                    device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if torch.equal(positive, negative):
                raise RuntimeError("CFG embeddings identical")
            capture = Capture(pipe.transformer, positive, negative, args.steps, prompt["split"] == "calib")
            try:
                final = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, height=480, width=480, num_frames=45,
                             num_inference_steps=args.steps, guidance_scale=5.0, output_type="latent",
                             generator=torch.Generator(device="cuda").manual_seed(prompt["seed"])).frames
            finally:
                capture.close()
            if len(capture.states) != args.steps:
                raise RuntimeError(f"captured {len(capture.states)} states")
            states = torch.cat(capture.states, 0)
            if not torch.isfinite(states.float()).all() or not torch.isfinite(final.float()).all():
                raise RuntimeError("nonfinite latents")
            save_torch(pdir / "embeddings.pt", {"positive": positive.cpu(), "negative": negative.cpu()})
            save_torch(pdir / "states.pt", states)
            save_torch(pdir / "final-latents.pt", final.to(torch.bfloat16).cpu())
            if capture.calibrate:
                expected = args.steps * 2 * 30 * 2
                if len(capture.counts) != expected:
                    raise RuntimeError(f"calibration hooks {len(capture.counts)} != {expected}")
                save_torch(pdir / "center-means.pt", {"mean_q": capture.mean_q.float(), "mean_k": capture.mean_k.float()})
            write_json(pdir / "meta.json", {"prompt": prompt, "timesteps": capture.timesteps, "steps": args.steps,
                "states_shape": list(states.shape), "seconds": time.time() - started, "environment": environment,
                "states_sha256": digest(pdir / "states.pt")})
            done += 1
            print("COMPLETE", prompt["id"], prompt["split"], round(time.time() - started, 1), flush=True)
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    finalize_calibration(traj, config)
    print("WORKER COMPLETE", done, flush=True)


if __name__ == "__main__":
    main()
