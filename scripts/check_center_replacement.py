"""Independent native-Wan attention checks on the actual GPU runtime."""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from center_replacement import HIGH_PAIRS, replace_with_center


def main():
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    args = parser.parse_args()
    torch.manual_seed(20261109)
    torch.set_grad_enabled(False)
    attn = WanAttention(dim=1536, heads=12, dim_head=128, processor=WanAttnProcessor()).to("cuda", dtype=torch.bfloat16).eval()
    if args.snapshot:
        from diffusers import WanTransformer3DModel
        model = WanTransformer3DModel.from_pretrained(args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        attn = model.blocks[0].attn1
    tokens = 36
    hidden = torch.randn(2, tokens, 1536, device="cuda", dtype=torch.bfloat16)
    phase = torch.randn(1, tokens, 1, 64, device="cuda")
    cos = phase.cos().repeat_interleave(2, dim=-1)
    sin = phase.sin().repeat_interleave(2, dim=-1)

    def rotate(x):
        shaped = x.unflatten(-1, (12, 128))
        a, b = shaped.unflatten(-1, (64, 2)).unbind(-1)
        out = torch.empty_like(shaped)
        out[..., 0::2] = a * cos[..., 0::2] - b * sin[..., 1::2]
        out[..., 1::2] = a * sin[..., 1::2] + b * cos[..., 0::2]
        return out.transpose(1, 2)

    results = []
    for mode in ("baseline", "all", "high"):
        q = attn.norm_q(attn.to_q(hidden))
        k = attn.norm_k(attn.to_k(hidden))
        reference = []
        ids = list(range(64)) if mode == "all" else list(HIGH_PAIRS) if mode == "high" else []
        for x in (q, k):
            paired = x.unflatten(-1, (12, 64, 2)).clone()
            # Independently compute token means per coordinate, batch and head.
            for batch in range(2):
                for head in range(12):
                    for pair in ids:
                        center = x.unflatten(-1, (12, 64, 2))[batch, :, head, pair].float().mean(0).to(x.dtype)
                        paired[batch, :, head, pair] = center
            reference.append(paired.flatten(-3))
        assert torch.equal(replace_with_center(q, 12, mode), reference[0])
        assert torch.equal(replace_with_center(k, 12, mode), reference[1])
        value = attn.to_v(hidden).unflatten(-1, (12, 128)).transpose(1, 2)
        expected = F.scaled_dot_product_attention(rotate(reference[0]), rotate(reference[1]), value)
        expected = expected.transpose(1, 2).flatten(2)
        expected = attn.to_out[1](attn.to_out[0](expected))
        hooks = [module.register_forward_hook(lambda module, inputs, output: replace_with_center(output, 12, mode))
                 for module in (attn.norm_q, attn.norm_k)]
        actual = attn(hidden, rotary_emb=(cos, sin))
        for hook in hooks:
            hook.remove()
        error = float((actual - expected).abs().max())
        if error != 0:
            raise AssertionError((mode, error))
        results.append({"mode": mode, "native_attention_output_max_abs_error": error,
                        "token_count": tokens, "batches": 2, "heads": 12, "head_dim": 128})
    # Full-center trigonometric logits equal direct rotation in fp32, across 3D offsets.
    qcenter = torch.randn(12, 64, 2, device="cuda")
    kcenter = torch.randn_like(qcenter)
    axis_ids = torch.tensor([0] * 22 + [1] * 21 + [2] * 21, device="cuda")
    frequency = torch.exp(-torch.arange(64, device="cuda").float() / 9)
    coords = torch.tensor([[0, 0, 0], [1, 2, 3], [2, 1, 0]], device="cuda")
    angle = coords[:, axis_ids] * frequency
    def rotate_center(center):
        a, b = center.unbind(-1)
        return torch.stack((a[None] * angle[:, None].cos() - b[None] * angle[:, None].sin(),
                            a[None] * angle[:, None].sin() + b[None] * angle[:, None].cos()), dim=-1)
    qr, kr = rotate_center(qcenter), rotate_center(kcenter)
    direct = torch.einsum("ihfc,jhfc->hij", qr, kr) / math.sqrt(128)
    delta = angle[:, None] - angle[None, :]
    dot = (qcenter * kcenter).sum(-1)
    cross = qcenter[..., 0] * kcenter[..., 1] - qcenter[..., 1] * kcenter[..., 0]
    trig = (dot[:, None, None] * delta[None].cos() + cross[:, None, None] * delta[None].sin()).sum(-1) / math.sqrt(128)
    error = float((direct - trig).abs().max())
    assert error < 2e-6, error
    print(json.dumps({"native_checks": results, "three_axis_trig_max_abs_error": error,
                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                      "weights": "pretrained layer 0" if args.snapshot else "randomly initialized native Wan module"}, indent=2))


if __name__ == "__main__":
    main()
