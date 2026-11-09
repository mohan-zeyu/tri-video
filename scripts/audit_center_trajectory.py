"""Independent processor replay against saved complete generation trajectories.

Uses a fresh no-hook baseline and an independently written coordinate-index
replacement processor, with actual Wan 3D angles and all 10,800 tokens.
The reference uses flat coordinate means and the native attention dispatcher.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


class IndependentCenterProcessor:
    def __init__(self, mode, axis_pair_counts):
        self.mode = mode
        self.calls = 0
        self.rotated_calls = 0
        self.axis_pair_counts = axis_pair_counts

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        if encoder_hidden_states is not None or attn.add_k_proj is not None:
            raise RuntimeError("Replay processor expects video self-attention")
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(hidden_states))
        value = attn.to_v(hidden_states)
        tokens = hidden_states.shape[1]
        if tokens != 10800:
            raise RuntimeError(tokens)
        if self.mode == "all":
            query = query.float().mean(1, keepdim=True).to(query.dtype).expand_as(query)
            key = key.float().mean(1, keepdim=True).to(key.dtype).expand_as(key)
        elif self.mode == "high":
            # Derive coordinate indices from native axis dimensions. Compute
            # flat coordinate means directly, independently of the head reshape.
            pair_ids = []
            offset = 0
            for count in self.axis_pair_counts:
                pair_ids.extend(range(offset, offset + (count + 2) // 3))
                offset += count
            coordinates = [head * 128 + 2 * pair + component
                           for head in range(attn.heads)
                           for pair in pair_ids for component in (0, 1)]
            query_center = query.float().mean(1, keepdim=True).to(query.dtype)
            key_center = key.float().mean(1, keepdim=True).to(key.dtype)
            query = query.clone()
            key = key.clone()
            query[..., coordinates] = query_center[..., coordinates]
            key[..., coordinates] = key_center[..., coordinates]
        else:
            raise ValueError(self.mode)
        query = query.unflatten(-1, (attn.heads, 128))
        key = key.unflatten(-1, (attn.heads, 128))
        value = value.unflatten(-1, (attn.heads, 128))
        if rotary_emb is None:
            raise RuntimeError("Missing native RoPE")
        cosine, sine = rotary_emb
        def rotate(x):
            first = x[..., 0::2]
            second = x[..., 1::2]
            result = torch.empty_like(x)
            result[..., 0::2] = first * cosine[..., 0::2] - second * sine[..., 1::2]
            result[..., 1::2] = first * sine[..., 1::2] + second * cosine[..., 0::2]
            return result
        query = rotate(query)
        key = rotate(key)
        self.rotated_calls += 1
        self.calls += 1
        output = dispatch_attention_fn(query, key, value, attn_mask=attention_mask,
                                       dropout_p=0.0, is_causal=False, backend=None,
                                       parallel_config=None)
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


def main():
    from diffusers import AutoencoderKLWan, WanPipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out/center-benchmark-wan-20261109")
    parser.add_argument("--snapshot", default="/dev/shm/tri-wan-snapshot")
    parser.add_argument("--prompt", default="v000")
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads((root / "config.json").read_text())
    prompt = next(p for p in config["prompts"] if p["id"] == args.prompt)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    vae = AutoencoderKLWan.from_pretrained(args.snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(args.snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    positive, negative = pipe.encode_prompt(prompt=prompt["text"], negative_prompt=config["negative_prompt"],
        do_classifier_free_guidance=True, device="cuda", num_videos_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    native = [block.attn1.processor for block in pipe.transformer.blocks]
    pair_counts = [getattr(pipe.transformer.rope, axis + "_dim") // 2 for axis in ("t", "h", "w")]
    result = {"prompt": prompt, "grid": [12, 30, 30], "tokens": 10800,
              "layers": 30, "heads": 12, "steps": 20, "cfg_branches": 2,
              "native_axis_pair_counts": pair_counts,
              "scope": "One actual prompt and seed. Each replacement center averages all video tokens at this single step/layer/head/CFG branch. All layers, heads and 20 steps execute. No cross-step/prompt pooling.",
              "checks": []}
    for mode in ("baseline", "all", "high"):
        for layer, block in enumerate(pipe.transformer.blocks):
            block.attn1.set_processor(native[layer] if mode == "baseline" else IndependentCenterProcessor(mode, pair_counts))
        captured = []
        def step_end(pipe, step, timestep, kwargs):
            captured.append({"step": step + 1, "latent_rms": float(kwargs["latents"].float().square().mean().sqrt())})
            return kwargs
        started = time.time()
        actual = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
            height=config["height"], width=config["width"], num_frames=config["frames"],
            num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]),
            output_type="latent", callback_on_step_end=step_end).frames.cpu()
        path = root / "records" / prompt["id"] / mode / "final-latents.pt"
        expected = torch.load(path, map_location="cpu", weights_only=True)
        difference = actual.float() - expected.float()
        check = {"mode": mode, "comparison": "Fresh native no-hook baseline" if mode == "baseline" else "Independent explicit processor vs saved hook-based generation",
                 "bit_exact": bool(torch.equal(actual, expected)), "max_abs_error": float(difference.abs().max()),
                 "relative_l2_error": float(difference.norm() / expected.float().norm()),
                 "seconds": time.time() - started, "steps": captured,
                 "reference_latents_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if mode != "baseline":
            processors = [block.attn1.processor for block in pipe.transformer.blocks]
            check["attention_calls"] = sum(p.calls for p in processors)
            check["native_rope_calls"] = sum(p.rotated_calls for p in processors)
            if check["attention_calls"] != 1200 or check["native_rope_calls"] != 1200:
                raise RuntimeError(check)
        result["checks"].append(check)
        out = root / "implementation-audit"
        out.mkdir(exist_ok=True)
        (out / (prompt["id"] + "-trajectory-replay.json")).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in check.items() if k != "steps"}), flush=True)
    if not all(v["bit_exact"] for v in result["checks"]):
        raise RuntimeError("Independent full-trajectory replay differed from saved outputs")


if __name__ == "__main__":
    main()
