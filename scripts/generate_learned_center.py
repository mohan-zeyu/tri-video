"""Generate one evaluation arm of the learned-center study on the 100-prompt benchmark, one GPU per worker.

Output layout matches scripts/run_center_benchmark.py so the existing evaluator/summarizer can be reused:
  <out>/videos/<arm>/<id>.mp4
  <out>/records/<id>/<arm>/{complete.json, final-latents.pt}

Arms (--arm NAME, with --student DIR and --center-mode):
  baseline       teacher weights, replacement off (or --link-baseline to hard-link the existing baseline videos)
  static         teacher weights + calibrated means (learned_center.pt from a stage-0/1 directory), mode mean
  student_s1     teacher weights + stage-1 trained means
  student        stage-3 student weights + trained means (mode mean)
  dense_control  fine-tuned weights trained with --center-mode off, replacement off
  narrow         weights trained with --center-mode narrow, zeros before RMSNorm on the checkpoint's masks
                 (the original control: fixed HIGH pairs at every step; with a selection: per-cell masks)
  deletion       weights trained with --center-mode delete, mode delete
The arm name is free; the weights/means/mode come from the arguments. Every record stores the
LearnedCenter state hash, the mode, exempt layers/steps and the number of replacement hook calls.

Same protocol as the benchmark: 45 frames 480x480, 20 UniPC steps, CFG 5, empty negative prompt, bf16
transformer, fp32 tiled VAE, CUDA generator seeded per prompt, same prompt seeds as center_benchmark_wan.json.
"""

import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from full_sweep import save_torch, write_json
from learned_center import LearnedCenter

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/center_benchmark_wan.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--student", help="directory with learned_center.pt and optionally transformer/ (bf16 safetensors)")
    parser.add_argument("--center-mode", choices=("mean", "delete", "narrow", "off"), default="mean")
    parser.add_argument("--link-baseline", help="existing benchmark root whose videos/baseline and records/*/baseline are hard-linked as this arm")
    parser.add_argument("--prompts", nargs="*")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--resident-text-encoder", action="store_true",
                        help="keep the text encoder on CUDA between prompts when device memory permits")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    dest = ROOT / args.out
    dest.mkdir(parents=True, exist_ok=True)

    if args.arm not in config["modes"]:
        raise ValueError(f"Unknown benchmark mode {args.arm}")
    if args.link_baseline:
        if args.arm != "baseline":
            raise ValueError("--link-baseline requires --arm baseline")
        source = ROOT / args.link_baseline
        if source.resolve() == dest.resolve():
            raise ValueError("Baseline and destination must be different directories")
        linked = 0
        for prompt in config["prompts"]:
            video = source / "videos" / "baseline" / (prompt["id"] + ".mp4")
            record = source / "records" / prompt["id"] / "baseline"
            if not video.exists() or not (record / "complete.json").exists():
                raise RuntimeError(f"missing baseline for {prompt['id']}")
            if json.loads((record / "complete.json").read_text())["prompt"] != prompt:
                raise RuntimeError(f"frozen baseline prompt or seed differs for {prompt['id']}")
            (dest / "videos" / args.arm).mkdir(parents=True, exist_ok=True)
            (dest / "records" / prompt["id"] / args.arm).mkdir(parents=True, exist_ok=True)
            target_video = dest / "videos" / args.arm / (prompt["id"] + ".mp4")
            if not target_video.exists():
                os.link(video, target_video)
            for f in record.iterdir():
                t = dest / "records" / prompt["id"] / args.arm / f.name
                if not t.exists():
                    os.link(f, t)
            linked += 1
        print("LINKED", linked, "baseline prompts as", args.arm, flush=True)
        return

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    import diffusers, transformers
    from diffusers.utils import export_to_video

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub" / ("models--" + config["model"].replace("/", "--"))
    revision = (cache / "refs/main").read_text().strip() if (cache / "refs/main").exists() else "unknown"
    snapshot = Path(os.environ.get("WAN_SNAPSHOT", str(cache / "snapshots" / revision)))
    student_dir = Path(args.student) if args.student else None
    weights_source = "teacher"
    if student_dir and (student_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists():
        transformer = WanTransformer3DModel.from_pretrained(student_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
        weights_source = str(student_dir / "transformer")
    else:
        transformer = WanTransformer3DModel.from_pretrained(snapshot, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(snapshot, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()

    center_hash = None
    if student_dir and (student_dir / "learned_center.pt").exists():
        saved = torch.load(student_dir / "learned_center.pt", map_location="cpu")
        # masks (per-cell Q/K selections), content pairs, wide flag and exemptions come from the checkpoint
        center = LearnedCenter.from_checkpoint(saved, mode=args.center_mode).to("cuda")
        if (center.layers, center.steps, center.heads, center.head_dim) != (len(transformer.blocks), config["steps"], transformer.config.num_attention_heads, transformer.config.attention_head_dim):
            raise RuntimeError("learned_center.pt shape does not match the model/config")
        center_hash = digest(student_dir / "learned_center.pt")
    elif args.center_mode == "mean":
        raise RuntimeError("--center-mode mean needs --student with learned_center.pt")
    else:
        center = LearnedCenter(layers=len(transformer.blocks), steps=config["steps"], heads=transformer.config.num_attention_heads,
                               head_dim=transformer.config.attention_head_dim, mode=args.center_mode).to("cuda")
    center.attach(pipe.transformer)
    exempt = {"layers": [int(i) for i in torch.nonzero(~center.layer_active).flatten()], "steps": [int(i) for i in torch.nonzero(~center.step_active).flatten()]}
    bypass = center.bypass_spec
    selection_info = {"default_high_masks": center.uses_default_masks, "cell_invariant_masks": center.cell_invariant(),
                      "pairs_per_cell_q": float(center.selected_pairs("q").sum(-1).float().mean()),
                      "pairs_per_cell_k": float(center.selected_pairs("k").sum(-1).float().mean()),
                      "q_k_intersection_mean": float((center.selected_pairs("q") & center.selected_pairs("k")).sum(-1).float().mean()),
                      "nope_pairs_per_cell": float(center.selected_pairs("nope").sum(-1).float().mean()),
                      "wide_head": center.is_wide, "content_pairs": center.content_pairs if center.is_wide else 0}
    environment = {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                   "snapshot": revision, "weights": weights_source, "learned_center_sha256": center_hash, "center_mode": args.center_mode,
                   "exempt": exempt, "bypass": bypass, "selection_spec": center.selection_spec, "selection": selection_info,
                   "gpu": torch.cuda.get_device_name(0), "hostname": socket.gethostname(), "pid": os.getpid(),
                   "resident_text_encoder": args.resident_text_encoder,
                   "config_sha256": digest(config_path), "code_sha256": {n: digest(ROOT / "scripts" / n) for n in ("generate_learned_center.py", "learned_center.py")}}
    active_layers = int(center.layer_active.sum()) if args.center_mode != "off" else 0
    active_steps = int(center.step_active.sum()) if args.center_mode != "off" else 0
    active_branches = int(sum(bool(center.cell_active[:, :, b].any()) for b in range(center.branches))) if args.center_mode != "off" else 0
    # a hook/processor counts once per layer x branch forward x step where any head of that cell is active
    expected_calls = active_layers * 2 * active_branches * active_steps if args.center_mode != "off" else 0   # layers x (q,k) x branches x steps
    expected_nope_calls = active_layers * active_branches * active_steps if (center.uses_nope and args.center_mode != "off") else 0
    scope = {
        "mean": ("widened head: full 128-dim constant vector under RoPE plus %d unrotated content pairs appended to Q/K, per layer/step/branch/head" % center.content_pairs) if center.is_wide else
                "constant vectors on selected Q/K pairs after RMSNorm, per layer/step/branch/head"
                + ("; the other pairs keep content without rotary embedding (NoPE)" if center.uses_nope else ""),
        "delete": ("widened head with the position channel zeroed: attention on %d unrotated content pairs only" % center.content_pairs) if center.is_wide else
                  "zeros on selected Q/K pairs after RMSNorm and before RoPE, per layer/step/branch/head"
                  + ("; the other pairs keep content without rotary embedding (NoPE): position-blind attention" if center.uses_nope else ""),
        "narrow": "zeros on selected Q/K pairs before native full-width RMSNorm, per layer/step/branch/head"
                  + ("; masks identical at every step and branch, equivalent to dropped projection rows"
                     if center.cell_invariant() and bool(center.step_active.all()) else ""),
        "off": "full Q/K with no replacements",
    }[args.center_mode] + ("; bypass: " + center.describe_bypass() if args.center_mode != "off" else "")

    completed = 0
    for prompt in config["prompts"]:
        if args.prompts and prompt["id"] not in args.prompts:
            continue
        pdir = dest / "records" / prompt["id"]
        arm = pdir / args.arm
        if (arm / "complete.json").exists():
            continue
        arm.mkdir(parents=True, exist_ok=True)
        claim = arm / "claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            pipe.text_encoder.to("cuda")
            positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt=config["negative_prompt"], do_classifier_free_guidance=True,
                                                    device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
            if not args.resident_text_encoder:
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            center.set_rollout(positive, negative)
            center.calls = 0
            center.nope_calls = 0
            started = time.time()
            latents_holder = {}
            checked_steps = []

            def end_step(p, i, t, kw):
                lat = kw["latents"].float()
                if not torch.isfinite(lat).all():
                    raise RuntimeError("nonfinite latents")
                latents_holder["final"] = lat.detach()
                checked_steps.append(i)
                return kw
            frames = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, height=config["height"], width=config["width"],
                          num_frames=config["frames"], num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
                          generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]), output_type="np",
                          callback_on_step_end=end_step).frames[0]
            torch.cuda.synchronize()
            seconds = time.time() - started
            if checked_steps != list(range(config["steps"])) or center.rollout["step"] != config["steps"] - 1:
                raise RuntimeError("incomplete denoising trajectory")
            center.clear_rollout()
            if center.calls != expected_calls:
                raise RuntimeError(f"replacement calls {center.calls} != expected {expected_calls}")
            if center.nope_calls != expected_nope_calls:
                raise RuntimeError(f"NoPE processor calls {center.nope_calls} != expected {expected_nope_calls}")
            if not np.isfinite(frames).all() or frames.shape != (config["frames"], config["height"], config["width"], 3):
                raise RuntimeError(f"bad frames {frames.shape}")
            video_dir = dest / "videos" / args.arm
            video_dir.mkdir(parents=True, exist_ok=True)
            video = video_dir / (prompt["id"] + ".mp4")
            tmp = video.with_name(video.stem + ".pending.mp4")
            export_to_video(frames, str(tmp), fps=config["fps"], quality=9, macro_block_size=1)
            tmp.replace(video)
            save_torch(arm / "final-latents.pt", latents_holder["final"].cpu())
            write_json(arm / "complete.json", {"prompt": prompt, "mode": args.arm, "environment": environment,
                "inference_decode_seconds": seconds, "finite_step_latents": len(checked_steps), "intervention_calls": center.calls, "expected_intervention_calls": expected_calls,
                "nope_calls": center.nope_calls, "expected_nope_calls": expected_nope_calls,
                "video_sha256": digest(video), "video_bytes": video.stat().st_size, "frames": list(frames.shape),
                "center_scope": scope,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
            completed += 1
            print("COMPLETE", prompt["id"], args.arm, round(seconds, 1), flush=True)
        except BaseException as error:
            write_json(arm / "failed.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise
        finally:
            claim.rmdir()
        if args.max_prompts and completed >= args.max_prompts:
            break
    print("WORKER COMPLETE", completed, flush=True)


if __name__ == "__main__":
    main()
