"""Learned per-step "mean vector" replacement of selected Wan Q/K rotary pairs.

Student rule, applied in every self-attention layer l, head h, denoising step s and CFG branch b (except
exempt layers/steps): after Wan's QK-RMSNorm and before 3D RoPE,

    q[:, h, S_q[l,s,b,h]] = mu_q[l, s, b, h, S_q[l,s,b,h]]      k[:, h, S_k[l,s,b,h]] = mu_k[l, s, b, h, S_k[l,s,b,h]]

S_q and S_k are boolean coordinate masks (per layer, step, branch, head; 128 coordinates = 64 rotary pairs
x 2). The default masks select HIGH on both sides: the 44 coordinates of pairs t 0-7, h 22-28, w 43-49, the
repo's original high-frequency third. A selection produced by scripts/select_pairs.py gives Q and K their
own per-cell sets. mu_q/mu_k are fp32 parameters of shape [layers, steps, branches, heads, 128]; entries
outside the mask are unused.

A third, optional mask S_nope (same for Q and K) marks pairs that keep their content but receive NO rotary
embedding (NoPE): the attention processor is wrapped so that RoPE multiplies those pairs by 1 instead of
e^{i w pos}. With S_q = S_k = S_mean and S_nope = the complement, the logit splits into a pure content term
(the NoPE pairs, position-blind) and a pure position term (the constant pairs under RoPE, a per-head additive bias
sum_f |mu_q,f||mu_k,f| cos(w_f delta + phi_f), computed by position_bias_tables). Setting the constants to zero
then removes every positional signal from that attention layer.

Modes: "mean" (learned constants on the mask), "delete" (post-norm zeros on the mask), "narrow" (pre-norm
zeros on the mask, i.e. the projection output is zeroed before Wan's RMSNorm, so the norm statistics are taken
over the remaining coordinates), "off" (dense attention).

Mean and deletion hook attn1.norm_q / attn1.norm_k. Narrow hooks attn1.to_q / attn1.to_k on non-exempt layers
and leaves native RMSNorm and RoPE in place. Narrow takes the same per-cell masks and step exemptions as the
other modes; when its masks do not depend on step or branch and no step is exempt (the original narrow control,
or a "pooled" selection), it is exactly equivalent to deleting those rows of to_q/to_k, and it then also works
without a step/branch context. The step/branch context is set explicitly by the training loop (set_context) or
inferred during pipeline rollouts (set_rollout + a transformer pre-hook that compares the text embeddings to the
registered conditional/unconditional embeddings and counts conditional forwards).

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
        mask_nope = torch.zeros(shape, dtype=torch.bool)
        if selection is not None:
            mask_q, mask_k = (self._coerce_mask(selection[k], shape) for k in ("mask_q", "mask_k"))
            if selection.get("mask_nope") is not None:
                mask_nope = self._coerce_mask(selection["mask_nope"], shape)
                if bool((mask_nope & (mask_q | mask_k)).any()):
                    raise ValueError("a NoPE pair cannot also be a replaced pair")
        self.register_buffer("mask_q", mask_q)
        self.register_buffer("mask_k", mask_k)
        self.register_buffer("mask_nope", mask_nope)
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
        self.wrapped = []      # (attention module, original processor) pairs replaced by NoPE wrappers
        self.rollout = None
        self.calls = 0
        self.nope_calls = 0

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

    def cell_invariant(self, layer=None):
        """True when the masks of this layer (or every layer) are the same at every step and branch."""
        layers = range(self.layers) if layer is None else (layer,)
        for mask in (self.mask_q, self.mask_k):
            for l in layers:
                m = mask[l]
                if not torch.equal(m, m[0:1, 0:1].expand_as(m)):
                    return False
        return True

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
        batch, tokens, width = x.shape
        masks = self.mask_q if side == "q" else self.mask_k
        if self.step is None or self.branch is None:
            # Context-free use is only defined for narrow with step/branch-invariant masks and no exempt step:
            # then the removal is a fixed property of the layer (equivalent to dropping projection rows).
            if self.mode == "narrow" and bool(self.step_active.all()) and self.cell_invariant(layer):
                mask = masks[layer, 0, 0].unsqueeze(0).expand(batch, -1, -1)      # [B, heads, 128]
                step = branch = None
            else:
                raise RuntimeError("LearnedCenter context not set")
        else:
            step = self.step.to(x.device)
            branch = self.branch.to(x.device)
            if step.numel() != batch:
                raise RuntimeError(f"context batch {step.numel()} != tensor batch {batch}")
            active = self.step_active.to(x.device)[step]                      # [B]
            if not bool(active.any()):
                return x
            mask = masks[layer][step, branch] & active.view(batch, 1, 1)      # [B, heads, 128]
        if self.mode == "mean":
            mu = self.mu_q if side == "q" else self.mu_k
            values = mu[layer][step, branch].to(x.dtype)                 # [B, heads, 128]
        elif self.mode in ("delete", "narrow"):
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
            if self.uses_nope and bool(self.layer_active[layer]) and self.mode != "off":
                attn = block.attn1
                original = attn.processor
                attn.set_processor(NoPEProcessor(original, self, layer))
                self.wrapped.append((attn, original))
        self.handles.append(transformer.register_forward_pre_hook(self._transformer_pre_hook, with_kwargs=True))
        return self

    @property
    def uses_nope(self):
        return bool(self.mask_nope.any())

    def nope_mask_for(self, layer, batch, device):
        """[B, heads, 128] bool: pairs that must not be rotated for the current context, or None if nothing to do."""
        if self.step is None or self.branch is None:
            raise RuntimeError("LearnedCenter context not set")
        step = self.step.to(device)
        branch = self.branch.to(device)
        if step.numel() != batch:
            raise RuntimeError(f"context batch {step.numel()} != tensor batch {batch}")
        active = self.step_active.to(device)[step]
        if not bool(active.any()):
            return None
        return self.mask_nope[layer][step, branch] & active.view(batch, 1, 1)

    def _hook(self, layer, side):
        def apply(module, inputs, output):
            return self.replace(output, layer, side)
        return apply

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for attn, original in self.wrapped:
            attn.set_processor(original)
        self.wrapped = []

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
        if prefix + "mask_nope" not in state_dict:
            state_dict[prefix + "mask_nope"] = torch.zeros(self.mu_q.shape, dtype=torch.bool)
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
        """[layers, steps, branches, heads, 64] bool: pairs replaced on this side, or NoPE pairs for side "nope" (ignoring exemptions)."""
        return {"q": self.mask_q, "k": self.mask_k, "nope": self.mask_nope}[side][..., 0::2]

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


class NoPEProcessor:
    """Wraps a Wan attention processor so that the pairs in LearnedCenter.mask_nope receive no rotary embedding.

    The wrapper rewrites the rotary tensors passed to the original processor: on NoPE coordinates cos -> 1 and
    sin -> 0 (tuple format, diffusers >= 0.35, freqs [1, T, 1, 128] against q/k [B, T, H, 128]) or the complex
    factor -> 1 (older format, freqs [1, 1, T, 64] against q/k [B, H, T, 128]). Every other pair sees the native
    rotary values and dtype, so non-NoPE pairs are bit-identical to the unwrapped model. The mask is per batch
    element (step, branch) and per head, as the other masks.
    """

    def __init__(self, original, center, layer):
        self.original = original
        self.center = center
        self.layer = layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs):
        if rotary_emb is not None and self.center.mode != "off":
            batch, tokens = hidden_states.shape[0], hidden_states.shape[1]
            nope = self.center.nope_mask_for(self.layer, batch, hidden_states.device)       # [B, H, 128] or None
            if nope is not None and bool(nope.any()):
                rotary_emb = self.rewrite(rotary_emb, nope, batch, tokens)
                self.center.nope_calls += 1
        return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)

    @staticmethod
    def rewrite(rotary_emb, nope, batch, tokens):
        if isinstance(rotary_emb, (tuple, list)):
            cos, sin = rotary_emb
            if cos.dim() != 4 or cos.shape[1] != tokens:
                raise RuntimeError(f"unexpected rotary layout {tuple(cos.shape)}")
            m = nope[:, None, :, :]                                                     # [B, 1, H, 128]
            cos = torch.where(m, torch.ones((), dtype=cos.dtype, device=cos.device), cos.to(nope.device))
            sin = torch.where(m, torch.zeros((), dtype=sin.dtype, device=sin.device), sin.to(nope.device))
            return (cos, sin)
        freqs = rotary_emb
        if not torch.is_complex(freqs) or freqs.dim() != 4 or freqs.shape[2] != tokens:
            raise RuntimeError(f"unexpected rotary layout {tuple(freqs.shape)}")
        m = nope[..., 0::2][:, :, None, :]                                              # [B, H, 1, 64]
        return torch.where(m, torch.ones((), dtype=freqs.dtype, device=freqs.device), freqs)
