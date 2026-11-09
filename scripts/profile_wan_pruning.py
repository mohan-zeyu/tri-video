"""Profile the frozen Wan pruning policy on actual generation trajectories."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from triattention_wan_queryblock_pruning import (
    KVPruningIntervention,
    importance_scores,
    select_indices,
    temporal_coefficients,
)


class StageTimer:
    def __init__(self):
        self.events = []

    @contextmanager
    def region(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        wall = time.perf_counter()
        yield
        elapsed = time.perf_counter() - wall
        end.record()
        self.events.append((name, start, end, elapsed))

    def summary(self):
        torch.cuda.synchronize()
        result = defaultdict(lambda: {"calls": 0, "stream_ms": 0.0, "cpu_wall_ms": 0.0})
        for name, start, end, wall in self.events:
            result[name]["calls"] += 1
            result[name]["stream_ms"] += start.elapsed_time(end)
            result[name]["cpu_wall_ms"] += wall * 1000
        return dict(result)


class ProfiledReference(KVPruningIntervention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = StageTimer()

    def dispatch(self, query, key, value, *args, **kwargs):
        if self.layer is None:
            return self.original_dispatch(query, key, value, *args, **kwargs)
        identity = (self.step, self.branch, self.layer)
        if identity in self.counts:
            raise RuntimeError("Duplicate self-attention dispatch")
        self.counts[identity] = 1
        keep_count = round(query.shape[1] * self.ratio)
        outputs = []
        per_frame = self.grid[1] * self.grid[2]
        for query_start in range(0, self.grid[0], 3):
            query_end = min(query_start + 3, self.grid[0])
            lookup = (self.branch, self.layer, query_start)
            with self.timer.region("coefficients"):
                if lookup not in self.coefficients:
                    mean = self.calibration["q_mean"][self.branch, self.layer]
                    residual = self.calibration["q_abs_mean"][self.branch, self.layer] - mean.norm(dim=-1)
                    self.coefficients[lookup] = (
                        temporal_coefficients(mean, self.grid[0], query.device, query_start, query_end), residual
                    )
                coefficient, residual = self.coefficients[lookup]
            with self.timer.region("score"):
                scores = importance_scores(self.pre_key, coefficient, residual, self.grid)
            with self.timer.region("selection"):
                indices = select_indices(scores, keep_count, self.grid, self.generator, query_start, query_end)
            with self.timer.region("gather"):
                gather = indices.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, 128)
                retained_key = key.gather(1, gather)
                retained_value = value.gather(1, gather)
                live_query = query[:, query_start * per_frame:query_end * per_frame]
            with self.timer.region("attention"):
                outputs.append(self.original_dispatch(live_query, retained_key, retained_value, *args, **kwargs))
        with self.timer.region("concatenation"):
            return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--implementations", nargs="+", choices=["native", "reference50", "reference75", "profile50", "profile75"], default=["native", "reference50", "reference75"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", default="v000")
    parser.add_argument("--capture", help="Save one actual first-step/layer-0 conditional Q/K/V input")
    args = parser.parse_args()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("configs/triattention_pruning_wan_queryblocks.json").read_text())
    prompt = next(p for p in config["prompts"] if p["id"] == args.prompt)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from diffusers import AutoencoderKLWan, WanPipeline
    import diffusers.models.transformers.transformer_wan as wan
    calibration = torch.load(config["calibration_path"], map_location="cpu", weights_only=True)
    snapshot = "/dev/shm/tri-wan-snapshot"
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt="",
        do_classifier_free_guidance=True, device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    result = {"prompt": prompt, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "scope": "One prompt/seed, 20 denoise steps, all30layers/12heads/bothCFG branches, all10800tokens. Each pruned bank keeps5400or8100 original K/V tokens per head. Full generation+VAE decode excludes text encoding and file I/O. One unreported warmup per arm then repeated measured trajectories.",
        "runs": {}}
    captured = {}
    if args.capture:
        original = wan.dispatch_attention_fn
        pre_key = []
        def capture_pre_key(module, inputs, output):
            if not pre_key and not captured:
                pre_key.append(output)
        handle = pipe.transformer.blocks[0].attn1.norm_k.register_forward_hook(capture_pre_key)
        def capture(query, key, value, *a, **kw):
            if pre_key and not captured:
                captured.update({"query": query.cpu(), "key": key.cpu(), "value": value.cpu(), "pre_key": pre_key[-1].cpu()})
            return original(query, key, value, *a, **kw)
        wan.dispatch_attention_fn = capture
    for name in args.implementations:
        values = []
        for repetition in range(-1, args.repeats):
            controller = None
            if name != "native":
                cls = ProfiledReference if name.startswith("profile") else KVPruningIntervention
                controller = cls(pipe.transformer, positive, negative, .5 if name.endswith("50") else .75, calibration)
                controller.started = time.time()
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                output = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                    height=480, width=480, num_frames=45, num_inference_steps=20, guidance_scale=5,
                    generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]), output_type="np",
                    callback_on_step_end=controller.end_step if controller else None).frames
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repetition >= 0:
                    item = {"seconds": elapsed, "decoded_sha256": hashlib.sha256(output.tobytes()).hexdigest()}
                    if controller:
                        item["self_attention_dispatches"] = len(controller.counts)
                        item["latent_sha256"] = hashlib.sha256(controller.final_latents.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    if hasattr(controller, "timer"):
                        item["stages"] = controller.timer.summary()
                    values.append(item)
                    if controller and repetition == 0:
                        torch.save(controller.final_latents.cpu(), destination / (name + "-final-latents.pt"))
                print("TIMING", name, repetition, elapsed, flush=True)
            finally:
                if controller:
                    controller.close()
            if args.capture and captured:
                wan.dispatch_attention_fn = original
                handle.remove()
                torch.save(captured, args.capture)
                captured.clear()
                pre_key.clear()
                args.capture = None
            del output, controller
        result["runs"][name] = {"median_seconds": statistics.median(x["seconds"] for x in values), "runs": values}
        (destination / "timings.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
