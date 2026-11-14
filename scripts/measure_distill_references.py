"""Measure the two Stage 0 velocity references on the frozen validation states."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from center_replacement import replace_with_center
from full_sweep import write_json
from train_learned_center import TrajectoryStore, rel_sq


def predict(model, item, device, dtype):
    x = item["x"].to(device=device, dtype=dtype).unsqueeze(0).expand(2, -1, -1, -1, -1)
    text = torch.cat([item["positive"], item["negative"]]).to(device=device, dtype=dtype)
    t = torch.full((2,), item["timestep"], device=device, dtype=torch.float32)
    return model(hidden_states=x.contiguous(), timestep=t, encoder_hidden_states=text, return_dict=False)[0].float()


def errors(output, reference):
    guided = lambda v: (v[1] + 5 * (v[0] - v[1])).unsqueeze(0)
    return torch.cat((rel_sq(output, reference), rel_sq(guided(output), guided(reference))))


def summarize(out, val, steps):
    paths = [out / f"{pid}.json" for pid in val]
    if not all(p.exists() for p in paths):
        return
    rows = [json.loads(p.read_text()) for p in paths]
    results = {}
    for metric in ("teacher_bf16_vs_fp32", "high_current_mean_vs_teacher_bf16"):
        data = torch.tensor([r[metric] for r in rows], dtype=torch.float64)
        assert tuple(data.shape) == (len(val), steps, 3) and torch.isfinite(data).all()
        results[metric] = {"per_step": data.mean(0).tolist(), "mean": data.mean((0, 1)).tolist()}
    write_json(out / "complete.json", {"ids": val, "steps": steps, "columns": ["cond", "uncond", "guided_cfg5"],
        "scope": "Relative squared velocity error over all latent elements for one prompt, teacher entering state, denoise step and branch or guided combination. Average each step across prompts and overall equally across all prompt-step cells. High current means replace both normalized pre-RoPE Q/K on 22 high pairs across all 10800 tokens, all heads/layers, each branch/step independently, no exemptions (matches the earlier benchmark). FP32 teacher uses the same cached BF16 inputs upcast to FP32.", **results})


@torch.inference_mode()
def main():
    from diffusers import WanTransformer3DModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="out/distill-traj-wan")
    parser.add_argument("--prompts", default="configs/distill_prompts.json")
    parser.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"), required="WAN_SNAPSHOT" not in os.environ)
    parser.add_argument("--out", default="out/distill-reference")
    parser.add_argument("--max-prompts", type=int, help="limit completed work per worker for smoke checks")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # FP32 dense native SDPA can otherwise select the O(N^2) math path. Verify
    # memory-efficient dispatch explicitly at the real sequence length.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    val = [p["id"] for p in json.loads(Path(args.prompts).read_text())["prompts"] if p["split"] == "val"]
    store = TrajectoryStore(args.traj, val)
    if store.ids != val:
        raise RuntimeError("Validation trajectories are incomplete")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"subfolder": "transformer", "local_files_only": True}
    bf16 = WanTransformer3DModel.from_pretrained(args.snapshot, dtype=torch.bfloat16, **kwargs).to(device).eval()
    fp32 = WanTransformer3DModel.from_pretrained(args.snapshot, dtype=torch.float32, **kwargs).to(device).eval()
    enabled = False
    def high_hook(_module, _args, output):
        return replace_with_center(output, bf16.config.num_attention_heads, "high") if enabled else output
    for block in bf16.blocks:
        for side in ("q", "k"):
            getattr(block.attn1, "norm_" + side).register_forward_hook(high_hook)
    done = 0
    for pid in val:
        target = out / f"{pid}.json"
        if target.exists():
            continue
        claim = out / f"{pid}.claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            if target.exists():
                continue
            started = time.monotonic()
            noise, high = [], []
            for step in range(store.steps):
                item = store.item(pid, step)
                enabled = False
                native = predict(bf16, item, device, torch.bfloat16)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    precision = predict(fp32, item, device, torch.float32)
                enabled = True
                replaced = predict(bf16, item, device, torch.bfloat16)
                noise.append(errors(native, precision).tolist())
                high.append(errors(replaced, native).tolist())
            data = torch.tensor([noise, high])
            if not torch.isfinite(data).all():
                raise FloatingPointError(pid)
            write_json(target, {"prompt_id": pid, "seconds": time.monotonic() - started,
                "teacher_bf16_vs_fp32": noise, "high_current_mean_vs_teacher_bf16": high})
            print("COMPLETE", pid, round(time.monotonic() - started, 2), flush=True)
            done += 1
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    summarize(out, val, store.steps)


if __name__ == "__main__":
    main()
