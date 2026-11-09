"""Restartable Wan benchmark with center-guided original-K/V pruning."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
import torch

from triattention_wan_queryblock_pruning import KVPruningIntervention
from full_sweep import save_torch, write_json

MODES = ("baseline", "tri50", "tri75")

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", nargs="*")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["tri50", "tri75"])
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    calibration_path = ROOT / config["calibration_path"]
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    dest = ROOT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from diffusers import AutoencoderKLWan, WanPipeline
    import diffusers
    import transformers
    from diffusers.utils import export_to_video

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub" / ("models--" + config["model"].replace("/", "--"))
    revision = (cache / "refs/main").read_text().strip()
    snapshot = Path(os.environ.get("WAN_SNAPSHOT", str(cache / "snapshots" / revision)))
    print("Loading", snapshot, flush=True)
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    if (len(pipe.transformer.blocks), pipe.transformer.config.num_attention_heads, pipe.transformer.config.attention_head_dim) != (30, 12, 128):
        raise RuntimeError("Unexpected model architecture")
    environment = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                   "transformers": transformers.__version__, "snapshot": revision,
                   "gpu": torch.cuda.get_device_name(0), "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "hostname": socket.gethostname(), "pid": os.getpid(),
                   "config_sha256": digest(config_path),
                   "calibration_sha256": digest(calibration_path),
                   "code_sha256": {name: digest(ROOT / "scripts" / name) for name in ("run_triattention_queryblock_benchmark.py", "triattention_wan_queryblock_pruning.py")}}
    completed = 0
    for prompt in config["prompts"]:
        if args.prompts and prompt["id"] not in args.prompts:
            continue
        pdir = dest / "records" / prompt["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        if all((pdir / mode / "complete.json").exists() for mode in args.modes):
            continue
        claim = pdir / "claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json(claim / "owner.json", environment)
        try:
            pipe.text_encoder.to("cuda")
            positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt=config["negative_prompt"],
                do_classifier_free_guidance=True, device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if torch.equal(positive, negative):
                raise RuntimeError("CFG embeddings unexpectedly identical")
            for mode in args.modes:
                arm = pdir / mode
                if (arm / "complete.json").exists():
                    continue
                arm.mkdir(exist_ok=True)
                print("START", prompt["id"], mode, flush=True)
                write_json(arm / "running.json", {"environment": environment, "start_unix": time.time()})
                ratio = {"baseline": 1.0, "tri50": 0.5, "tri75": 0.75}[mode]
                intervention = KVPruningIntervention(pipe.transformer, positive, negative, ratio, calibration)
                intervention.started = time.time()
                try:
                    frames = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                        height=config["height"], width=config["width"], num_frames=config["frames"],
                        num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
                        generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]),
                        output_type="np", callback_on_step_end=intervention.end_step).frames[0]
                    torch.cuda.synchronize()
                    inference_seconds = time.time() - intervention.started
                    if intervention.step != config["steps"] or not np.isfinite(frames).all():
                        raise RuntimeError("Incomplete or nonfinite output")
                    expected_shape = (config["frames"], config["height"], config["width"], 3)
                    if frames.shape != expected_shape:
                        raise RuntimeError(f"Unexpected frames {frames.shape}")
                    video_dir = dest / "videos" / mode
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video = video_dir / (prompt["id"] + ".mp4")
                    temp_video = video.with_name(video.stem + ".pending.mp4")
                    export_to_video(frames, str(temp_video), fps=config["fps"], quality=9, macro_block_size=1)
                    temp_video.replace(video)
                    save_torch(arm / "final-latents.pt", intervention.final_latents.cpu())
                    if intervention.tape:
                        np.savez_compressed(arm / "pair-diagnostics.npz", **intervention.tape)
                    write_json(arm / "complete.json", {"prompt": prompt, "mode": mode, "environment": environment,
                        "inference_decode_seconds": inference_seconds, "wall_seconds": time.time() - intervention.started,
                        "grid": intervention.grid, "tokens": int(np.prod(intervention.grid)),
                        "intervention_calls": len(intervention.counts), "steps": intervention.records,
                        "ranking_pairs": list(range(64)), "retention_ratio": ratio,
                        "retained_keys_per_head": round(10800 * ratio),
                        "protected_sink_tokens": 900, "protected_live_query_tokens": 2700, "query_chunk_frames": 3,
                        "ranked_keep_tokens_by_query_chunk": [round(10800 * ratio) - 2700] + [round(10800 * ratio) - 3600] * 3,
                        "video_sha256": digest(video), "video_bytes": video.stat().st_size,
                        "center_scope": "Calibrated Q means/mean norms, separate CFG branch/layer/head/pair, pooled four disjoint prompts x20steps x10800tokens. Native actual Q and selected original post-RoPE K/V perform attention.",
                        "frames": list(frames.shape), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
                    print("COMPLETE", prompt["id"], mode, round(inference_seconds, 2), flush=True)
                finally:
                    intervention.close()
                del intervention, frames
                torch.cuda.empty_cache()
            completed += 1
        except BaseException as error:
            write_json(pdir / "failed.json", {"error": repr(error), "traceback": traceback.format_exc(), "environment": environment})
            raise
        finally:
            (claim / "owner.json").unlink(missing_ok=True)
            claim.rmdir()
        if args.max_prompts and completed >= args.max_prompts:
            break
    print("WORKER COMPLETE", completed, flush=True)


if __name__ == "__main__":
    main()
