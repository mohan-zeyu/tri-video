"""Learned per-step "mean vector" replacement of selected Wan Q/K rotary pairs.

Student rule, applied in every self-attention layer l, head h, denoising step s and CFG branch b (except
exempt layers/steps): after Wan's QK-RMSNorm and before 3D RoPE,

    q[:, h, S_q[l,s,b,h]] = mu_q[l, s, b, h, S_q[l,s,b,h]]      k[:, h, S_k[l,s,b,h]] = mu_k[l, s, b, h, S_k[l,s,b,h]]

S_q and S_k are boolean coordinate masks (per layer, step, branch, head; 128 coordinates = 64 rotary pairs
x 2). The default masks select HIGH on both sides: the 44 coordinates of pairs t 0-7, h 22-28, w 43-49, the
repo's original high-frequency third. A selection produced by scripts/select_pairs.py gives Q and K their
own per-cell sets. mu_q/mu_k are fp32 parameters of shape [layers, steps, branches, heads, 128]; entries
outside the mask are unused.

Modes: "mean" (learned constants on the mask), "delete" (post-norm zeros on the mask), "narrow" (pre-norm
zeros on the fixed HIGH set at every step; not combinable with a selection), "off" (dense attention).

Mean and deletion hook attn1.norm_q / attn1.norm_k. Narrow hooks attn1.to_q / attn1.to_k on non-exempt layers
and leaves native RMSNorm and RoPE in place. The step/branch context is set explicitly by the training loop
(set_context) or inferred during pipeline rollouts (set_rollout + a transformer pre-hook that compares the
text embeddings to the registered conditional/unconditional embeddings and counts conditional forwards).

Checkpoint compatibility: state dicts written by the earlier 44-wide version (mu_* of shape [..., 44], no
masks) load into this module and reproduce its behaviour exactly.
"""

import torch
from torch import nn

from center_replacement import HIGH_PAIRS

HIGH_DIMS = torch.tensor([2 * f + c for f in HIGH_PAIRS for c in (0, 1)])
assert HIGH_DIMS.numel() == 44
PAIRS = 64


def pair_mask_to_dims(pair_mask):
    """[..., 64] bool over rotary pairs -> [..., 128] bool over interleaved coordinates."""
    return pair_mask.repeat_interleave(2, dim=-1)


def high_pair_mask():
    mask = torch.zeros(PAIRS, dtype=torch.bool)
    mask[list(HIGH_PAIRS)] = True
    return mask


class LearnedCenter(nn.Module):
    def __init__(self, layers=30, steps=20, branches=2, heads=12, head_dim=128,
                 exempt_layers=(0,), exempt_steps=(0,), mode="mean", selection=None):
        """selection: optional dict with 'mask_q' and 'mask_k', bool tensors of shape
        [layers, steps, branches, heads, 64] (pairs) or [..., 128] (coordinates)."""
        super().__init__()
        self.layers, self.steps, self.branches, self.heads, self.head_dim = layers, steps, branches, heads, head_dim
        if head_dim != 2 * PAIRS:
            raise ValueError("LearnedCenter assumes 128-dim heads (64 rotary pairs)")
        shape = (layers, steps, branches, heads, head_dim)
        self.mu_q = nn.Parameter(torch.zeros(shape))
        self.mu_k = nn.Parameter(torch.zeros(shape))
        default = pair_mask_to_dims(high_pair_mask()).expand(shape).clone()
        mask_q, mask_k = default, default.clone()
        if selection is not None:
            if mode == "narrow":
                raise ValueError("narrow mode uses the fixed HIGH projection rows; it takes no selection")
            mask_q, mask_k = (self._coerce_mask(selection[k], shape) for k in ("mask_q", "mask_k"))
        self.register_buffer("mask_q", mask_q)
        self.register_buffer("mask_k", mask_k)
        layer_active = torch.ones(layers, dtype=torch.bool)
        layer_active[list(exempt_layers)] = False
        step_active = torch.ones(steps, dtype=torch.bool)
        if mode != "narrow":
            step_active[list(exempt_steps)] = False
        self.register_buffer("layer_active", layer_active)
        self.register_buffer("step_active", step_active)
        self.register_buffer("high_dims", HIGH_DIMS.clone())
        projection_high_dims = (torch.arange(heads)[:, None] * head_dim + HIGH_DIMS).flatten()
        self.register_buffer("projection_high_dims", projection_high_dims, persistent=False)
        self.mode = mode
        self.step = None      # LongTensor [B], zero-based denoising step per batch element
        self.branch = None    # LongTensor [B], 0 = conditional, 1 = unconditional
        self.handles = []
        self.rollout = None
        self.calls = 0

    @staticmethod
    def _coerce_mask(mask, shape):
        mask = torch.as_tensor(mask).bool()
        if mask.shape[-1] == PAIRS:
            mask = pair_mask_to_dims(mask)
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(shape)}")
        if not torch.equal(mask[..., 0::2], mask[..., 1::2]):
            raise ValueError("masks must select whole rotary pairs (both coordinates)")
        return mask.clone()

    @property
    def uses_default_masks(self):
        default = pair_mask_to_dims(high_pair_mask()).to(self.mask_q.device)
        return bool((self.mask_q == default).all()) and bool((self.mask_k == default).all())

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
        """Replace selected pre-RoPE Q/K entries at the mode's projection or norm hook."""
        if self.mode == "off" or not bool(self.layer_active[layer]):
            return x
        if self.mode == "narrow":
            self.calls += 1
            return x.index_fill(-1, self.projection_high_dims, 0)
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
        mask = (self.mask_q if side == "q" else self.mask_k)[layer][step, branch]   # [B, heads, 128]
        mask = mask & active.view(batch, 1, 1)
        if self.mode == "mean":
            values = mu[layer][step, branch].to(x.dtype)                 # [B, heads, 128]
        elif self.mode == "delete":
            values = torch.zeros(batch, self.heads, self.head_dim, dtype=x.dtype, device=x.device)
        else:
            raise ValueError(self.mode)
        shaped = x.view(batch, tokens, self.heads, self.head_dim)
        out = torch.where(mask.unsqueeze(1), values.unsqueeze(1), shaped)
        self.calls += 1
        return out.reshape(batch, tokens, width)

    # ---- attachment ------------------------------------------------------------------------------------
    def attach(self, transformer):
        self.detach()
        if len(transformer.blocks) != self.layers:
            raise ValueError(f"model has {len(transformer.blocks)} blocks, module has {self.layers}")
        hook_prefix = "to_" if self.mode == "narrow" else "norm_"
        for layer, block in enumerate(transformer.blocks):
            if self.mode == "narrow" and not bool(self.layer_active[layer]):
                continue
            for side in ("q", "k"):
                module = getattr(block.attn1, hook_prefix + side)
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

    # ---- initialization and checkpoints --------------------------------------------------------------
    @torch.no_grad()
    def init_from_calibration(self, calibration):
        """calibration: dict with mean_q/mean_k of shape [layers, steps, branches, heads, 128] (full means,
        from scripts/collect_pair_statistics.py) or [..., 44] (HIGH-only means from the original dump)."""
        for name, param in (("mean_q", self.mu_q), ("mean_k", self.mu_k)):
            value = calibration[name].float().to(param.device)
            if tuple(value.shape) == tuple(param.shape):
                param.copy_(value)
            elif tuple(value.shape) == tuple(param.shape[:-1]) + (HIGH_DIMS.numel(),):
                param.zero_()
                param[..., self.high_dims] = value
            else:
                raise ValueError(f"{name} shape {tuple(value.shape)} incompatible with {tuple(param.shape)}")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept the earlier 44-wide layout (mu_* [..., 44], no masks)."""
        for name in ("mu_q", "mu_k"):
            key = prefix + name
            if key in state_dict and state_dict[key].shape[-1] == HIGH_DIMS.numel():
                full = torch.zeros(self.mu_q.shape, dtype=state_dict[key].dtype)
                full[..., HIGH_DIMS] = state_dict[key]
                state_dict[key] = full
        for name in ("mask_q", "mask_k"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = pair_mask_to_dims(high_pair_mask()).expand(self.mu_q.shape).clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def from_checkpoint(cls, saved, mode=None, **kwargs):
        """Build a module matching a saved {"state_dict", "mode"} record (either layout)."""
        state = saved["state_dict"]
        layers, steps, branches, heads = state["mu_q"].shape[:4]
        module = cls(layers=layers, steps=steps, branches=branches, heads=heads, mode=mode or saved.get("mode", "mean"), **kwargs)
        module.load_state_dict(state)
        return module

    def selected_pairs(self, side):
        """[layers, steps, branches, heads, 64] bool: pairs replaced on this side (ignoring exemptions)."""
        return (self.mask_q if side == "q" else self.mask_k)[..., 0::2]

    def position_bias_tables(self, layer, step, branch, rope_freqs):
        """Equivalent additive logit tables implied by the constants on pairs selected on BOTH sides (analysis).
        rope_freqs: [64] angular frequencies per pair. Returns dict axis -> callable(delta) giving
        sum_f |q_f||k_f| cos(w_f delta + phi_f)/sqrt(d) per head; one-sided pairs are not position-only and are skipped."""
        q = self.mu_q[layer, step, branch].detach().float().view(self.heads, PAIRS, 2)
        k = self.mu_k[layer, step, branch].detach().float().view(self.heads, PAIRS, 2)
        both = (self.selected_pairs("q")[layer, step, branch] & self.selected_pairs("k")[layer, step, branch]).float()   # [heads, 64]
        z = torch.complex(q[..., 0], q[..., 1]) * torch.complex(k[..., 0], k[..., 1]).conj() / self.head_dim ** 0.5 * both
        axis_pairs = {"t": range(0, 22), "h": range(22, 43), "w": range(43, 64)}
        tables = {}
        for axis, pairs in axis_pairs.items():
            sel = torch.tensor(list(pairs))
            w = rope_freqs[sel].float()
            zs = z[:, sel]
            tables[axis] = lambda delta, w=w, zs=zs: (zs * torch.exp(1j * w * float(delta))).real.sum(-1)
        return tables
