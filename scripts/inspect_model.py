"""Inspect installed diffusers model classes: forward signatures, module layout,
processor source (where RoPE is applied), rope module output shapes.

Downloads only the transformer subfolder. Usage:
  python inspect_model.py wan|ltx|cog
"""
import inspect
import sys

import torch

torch.set_grad_enabled(False)

NAME = {"wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "ltx": "Lightricks/LTX-Video",
        "cog": "zai-org/CogVideoX-2b"}


def main(which):
    if which == "wan":
        from diffusers import WanTransformer3DModel as M
        sub = "transformer"
    elif which == "ltx":
        from diffusers import LTXVideoTransformer3DModel as M
        sub = "transformer"
    else:
        from diffusers import CogVideoXTransformer3DModel as M
        sub = "transformer"
    model = M.from_pretrained(NAME[which], subfolder=sub, torch_dtype=torch.bfloat16)
    print("=== config ===")
    print({k: v for k, v in model.config.items() if not k.startswith("_")})
    print("=== forward signature ===")
    print(inspect.signature(model.forward))
    print("=== blocks/attn layout ===")
    blocks = model.blocks if which == "wan" else model.transformer_blocks
    b0 = blocks[0]
    for n, ch in b0.named_children():
        print(f"block0.{n}: {type(ch).__name__}")
    attn = b0.attn1
    print("--- attn1 children ---")
    for n, ch in attn.named_children():
        print(f"attn1.{n}: {type(ch).__name__}")
    print("attn1.heads =", getattr(attn, "heads", None))
    print("=== processor ===")
    proc = attn.processor
    print(type(proc).__module__, type(proc).__name__)
    src = inspect.getsource(type(proc).__call__)
    print(src[:4000])
    print("=== rope module ===")
    rope = model.rope if which != "cog" else model.rotary_emb
    print(type(rope).__module__, type(rope).__name__)
    print(inspect.getsource(type(rope))[:3500])


if __name__ == "__main__":
    main(sys.argv[1])
