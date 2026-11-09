"""Replace normalized pre-RoPE Wan Q/K coordinates with current-token means."""

import torch


HIGH_PAIRS = tuple(range(0, 8)) + tuple(range(22, 29)) + tuple(range(43, 50))
MODES = ("baseline", "all", "high")


def replace_with_center(x, heads, mode):
    """x is [batch,tokens,heads*head_dim], with independent batch/head means.

    Each selected 2D pair uses its two arithmetic coordinate means over all
    current video tokens. There is no averaging across heads, layers, CFG
    branches, prompts or denoising steps. Accumulate in fp32, cast to x.dtype.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "baseline":
        return x
    shaped = x.unflatten(-1, (heads, -1))
    if shaped.shape[-1] != 128:
        raise ValueError(f"Expected Wan 128D heads, got {shaped.shape}")
    center = shaped.mean(dim=1, keepdim=True, dtype=torch.float32).to(x.dtype)
    if mode == "all":
        return center.expand_as(shaped).reshape_as(x).contiguous()
    pairs = shaped.unflatten(-1, (64, 2))
    result = pairs.clone()
    result[..., HIGH_PAIRS, :] = center.unflatten(-1, (64, 2))[..., HIGH_PAIRS, :]
    return result.flatten(-3)


def pair_diagnostics(x, heads):
    """All-token pair MRL and relative mean-replacement squared error [B,H,F]."""
    pairs = x.float().unflatten(-1, (heads, 64, 2))
    center = pairs.mean(dim=1)
    squared_norm = pairs.square().sum(-1)
    mean_norm = squared_norm.sqrt().mean(dim=1)
    energy = squared_norm.mean(dim=1)
    center_energy = center.square().sum(-1)
    return {
        "mrl": center_energy.sqrt() / mean_norm.clamp_min(1e-20),
        "relative_squared_error": (energy - center_energy).clamp_min(0) / energy.clamp_min(1e-20),
        "center": center,
    }
