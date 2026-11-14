"""Learned per-step "mean vector" replacement of Wan's high-frequency Q/K rotary pairs.

Student rule, applied in every self-attention layer l, head h, denoising step s and CFG branch b (except
exempt layers/steps): after Wan's QK-RMSNorm and before 3D RoPE,

    q[:, h, HIGH] = mu_q[l, s, b, h]      k[:, h, HIGH] = mu_k[l, s, b, h]

HIGH is the 44 coordinates of rotary pairs t 0-7, h 22-28, w 43-49 (22 pairs), the same set as the repo's
training-free high-pair arm. mu_q/mu_k are fp32 parameters of shape [layers, steps, branches, heads, 44].

Modes: "mean" (learned constants), "delete" (zeros; the control that tests whether the mean carries anything),
"off" (teacher behaviour; the dense control).

The module is attached with forward hooks on each block's attn1.norm_q / attn1.norm_k, exactly the hook points
of scripts/run_center_benchmark.py. The step/branch context is set explicitly by the training loop
(set_context) or inferred during pipeline rollouts (set_rollout + a transformer pre-hook that compares the
text embeddings to the registered conditional/unconditional embeddings and counts conditional forwards).
"""

import torch
from torch import nn

from center_replacement import HIGH_PAIRS

HIGH_DIMS = torch.tensor([2 * f + c for f in HIGH_PAIRS for c in (0, 1)])
assert HIGH_DIMS.numel() == 44


class LearnedCenter(nn.Module):
    def __init__(self, layers=30, steps=20, branches=2, heads=12, head_dim=128,
                 exempt_layers=(0,), exempt_steps=(0,), mode="mean"):
        super().__init__()
        self.layers, self.steps, self.branches, self.heads, self.head_dim = layers, steps, branches, heads, head_dim
        self.mu_q = nn.Parameter(torch.zeros(layers, steps, branches, heads, HIGH_DIMS.numel()))
        self.mu_k = nn.Parameter(torch.zeros(layers, steps, branches, heads, HIGH_DIMS.numel()))
        layer_active = torch.ones(layers, dtype=torch.bool)
        layer_active[list(exempt_layers)] = False
        step_active = torch.ones(steps, dtype=torch.bool)
        step_active[list(exempt_steps)] = False
        self.register_buffer("layer_active", layer_active)
        self.register_buffer("step_active", step_active)
        self.register_buffer("high_dims", HIGH_DIMS.clone())
        self.mode = mode
        self.step = None      # LongTensor [B], zero-based denoising step per batch element
        self.branch = None    # LongTensor [B], 0 = conditional, 1 = unconditional
        self.handles = []
        self.rollout = None
        self.calls = 0

    # ---- context -------------------------------------------------------------------------------------
    def set_context(self, step, branch):
        self.step = torch.as_tensor(step).reshape(-1).long()
        self.branch = torch.as_tensor(branch).reshape(-1).long()

    def set_rollout(self, positive, negative):
        """Infer (step, branch) from the pipeline's forwards: conditional then unconditional per step."""
        self.rollout = {"positive": positive, "negative": negative, "step": -1}

    def clear_rollout(self):
        self.rollout = None

    def _transformer_pre_hook(self, module, args, kwargs):
        if self.rollout is None:
            return
        text = kwargs["encoder_hidden_states"].to(self.rollout["positive"].dtype)   # pipelines may cast to the model dtype
        if torch.equal(text, self.rollout["positive"]):
            self.rollout["step"] += 1
            branch = 0
        elif torch.equal(text, self.rollout["negative"]):
            branch = 1
        else:
            raise RuntimeError("Unknown CFG branch during rollout")
        batch = kwargs["hidden_states"].shape[0]
        self.set_context(torch.full((batch,), self.rollout["step"]), torch.full((batch,), branch))

    # ---- replacement -----------------------------------------------------------------------------------
    def replace(self, x, layer, side):
        """x: [B, tokens, heads*head_dim] normalized pre-RoPE Q or K. Returns the student tensor."""
        if self.mode == "off" or not bool(self.layer_active[layer]):
            return x
        if self.step is None or self.branch is None:
            raise RuntimeError("LearnedCenter context not set")
        batch, tokens, width = x.shape
        step = self.step.to(x.device)
        branch = self.branch.to(x.device)
        if step.numel() != batch:
            raise RuntimeError(f"context batch {step.numel()} != tensor batch {batch}")
        active = self.step_active.to(x.device)[step]                      # [B]
        if not bool(active.any()):
            return x
        mu = self.mu_q if side == "q" else self.mu_k
        if self.mode == "mean":
            values = mu[layer][step, branch]                              # [B, heads, 44] fp32
        elif self.mode == "delete":
            values = torch.zeros(batch, self.heads, HIGH_DIMS.numel(), device=x.device)
        else:
            raise ValueError(self.mode)
        values = values.to(x.dtype).unsqueeze(1).expand(batch, tokens, self.heads, -1)
        shaped = x.view(batch, tokens, self.heads, self.head_dim)
        original = shaped[..., self.high_dims]
        values = torch.where(active.view(batch, 1, 1, 1), values, original)
        out = shaped.clone()
        out[..., self.high_dims] = values
        self.calls += 1
        return out.view(batch, tokens, width)

    # ---- attachment ------------------------------------------------------------------------------------
    def attach(self, transformer):
        self.detach()
        if len(transformer.blocks) != self.layers:
            raise ValueError(f"model has {len(transformer.blocks)} blocks, module has {self.layers}")
        for layer, block in enumerate(transformer.blocks):
            for side in ("q", "k"):
                module = getattr(block.attn1, "norm_" + side)
                self.handles.append(module.register_forward_hook(self._hook(layer, side)))
        self.handles.append(transformer.register_forward_pre_hook(self._transformer_pre_hook, with_kwargs=True))
        return self

    def _hook(self, layer, side):
        def apply(module, inputs, output):
            return self.replace(output, layer, side)
        return apply

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    # ---- initialization --------------------------------------------------------------------------------
    @torch.no_grad()
    def init_from_calibration(self, calibration):
        """calibration: dict with mean_q/mean_k of shape [layers, steps, branches, heads, 44] (fp32)."""
        for name in ("mean_q", "mean_k"):
            if tuple(calibration[name].shape) != tuple(self.mu_q.shape):
                raise ValueError(f"{name} shape {tuple(calibration[name].shape)} != {tuple(self.mu_q.shape)}")
        self.mu_q.copy_(calibration["mean_q"].float())
        self.mu_k.copy_(calibration["mean_k"].float())

    def position_bias_tables(self, layer, step, branch, rope_freqs):
        """Equivalent additive logit tables implied by the constants (for analysis). rope_freqs: [64] angular
        frequencies per pair. Returns dict axis -> callable(delta) giving sum_f |q_f||k_f| cos(w_f delta + phi_f)/sqrt(d)."""
        q = self.mu_q[layer, step, branch].detach().float().view(self.heads, -1, 2)
        k = self.mu_k[layer, step, branch].detach().float().view(self.heads, -1, 2)
        zq = torch.complex(q[..., 0], q[..., 1])
        zk = torch.complex(k[..., 0], k[..., 1])
        z = zq * zk.conj() / self.head_dim ** 0.5                         # [heads, 22]
        pairs = torch.tensor(HIGH_PAIRS)
        axis_of = lambda f: "t" if f < 22 else "h" if f < 43 else "w"
        tables = {}
        for axis in ("t", "h", "w"):
            sel = torch.tensor([i for i, f in enumerate(pairs.tolist()) if axis_of(f) == axis])
            w = rope_freqs[pairs[sel]].float()
            zs = z[:, sel]
            tables[axis] = lambda delta, w=w, zs=zs: (zs * torch.exp(1j * w * float(delta))).real.sum(-1)
        return tables
