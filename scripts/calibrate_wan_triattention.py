"""Collect native normalized pre-RoPE Q means/norms from four held-out prompts."""

import argparse
import json
from pathlib import Path

import torch
from full_sweep import save_torch, write_json


def main():
    from diffusers import AutoencoderKLWan, WanPipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path("configs/full_sweep_wan.json").read_text())
    prompt = next(p for p in config["prompts"] if p["id"] == args.prompt)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = "/dev/shm/tri-wan-snapshot"
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt="", do_classifier_free_guidance=True, device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu")
    q_sum = torch.zeros(2, 30, 12, 64, 2, dtype=torch.float64)
    norm_sum = torch.zeros(2, 30, 12, 64, dtype=torch.float64)
    counts = torch.zeros(2, 30, dtype=torch.int64)
    state = {"branch": None, "step": 0}
    def start(module, inputs, kwargs):
        text = kwargs["encoder_hidden_states"]
        if torch.equal(text, positive):state["branch"] = 0
        elif torch.equal(text, negative):state["branch"] = 1
        else:raise RuntimeError("Unknown conditioning")
    handles = [pipe.transformer.register_forward_pre_hook(start, with_kwargs=True)]
    def capture(layer):
        def hook(module, inputs, output):
            b = state["branch"]
            q = output.float().reshape(1, 10800, 12, 64, 2)
            q_sum[b, layer] += q.sum((0, 1), dtype=torch.float64).cpu()
            norm_sum[b, layer] += q.norm(dim=-1).sum((0, 1), dtype=torch.float64).cpu()
            counts[b, layer] += 10800
        return hook
    for layer, block in enumerate(pipe.transformer.blocks):
        handles.append(block.attn1.norm_q.register_forward_hook(capture(layer)))
    def end(pipe, step, timestep, kwargs):
        state["step"] += 1
        print(args.prompt, "STEP", step + 1, flush=True)
        return kwargs
    pipe(prompt_embeds=positive, negative_prompt_embeds=negative, height=480, width=480, num_frames=45,
         num_inference_steps=20, guidance_scale=5, generator=torch.Generator(device="cuda").manual_seed(0),
         output_type="latent", callback_on_step_end=end)
    for handle in handles:handle.remove()
    assert (counts == 20 * 10800).all()
    root = Path(args.out);root.mkdir(parents=True, exist_ok=True)
    save_torch(root / (args.prompt + ".pt"), {"q_sum": q_sum, "norm_sum": norm_sum, "token_count": counts})
    write_json(root / (args.prompt + ".json"), {"prompt": prompt, "seed": 0, "steps": 20, "tokens_per_forward": 10800,
        "scope": "Separate CFG branch/layer/head/pair. Sum all10800 tokens and all20 steps; no head/layer/branch pooling."})
    print("CALIBRATION COMPLETE", args.prompt, flush=True)


if __name__ == "__main__":
    main()
