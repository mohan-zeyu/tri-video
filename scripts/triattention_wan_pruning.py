"""TriAttention key ranking with original Q and selected original post-RoPE K/V."""

import time
import torch


def temporal_coefficients(q_mean, frames, device):
    """[H,64,2] Q centers -> [T,H,128] mean rotated proxies.

    Matches official LongLive offsets 1,2,4,...128, reference frame T,
    delta_t=T-key_frame+offset and delta_h=delta_w=0. All64 pairs contribute.
    """
    omega = torch.zeros(64, device=device)
    omega[:22] = 10000.0 ** (-torch.arange(22, device=device).float() / 22)
    offsets = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device)
    angle = ((frames - torch.arange(frames, device=device))[:, None, None]
             + offsets[None, :, None]) * omega[None, None, :]
    cosine = angle.cos().mean(1)[:, None]
    sine = angle.sin().mean(1)[:, None]
    real, imag = q_mean.unbind(-1)
    return torch.stack((real[None] * cosine - imag[None] * sine,
                        imag[None] * cosine + real[None] * sine), -1).flatten(-2)


def importance_scores(pre_key, coefficient, residual, grid):
    """Return [B,H,N] scores, using each actual pre-RoPE key's64 pairs."""
    batch, tokens, width = pre_key.shape
    frames, height, columns = grid
    keys = pre_key.float().reshape(batch, frames, height * columns, 12, 128)
    trig = torch.einsum("btshd,thd->btsh", keys, coefficient)
    pair_norm = keys.unflatten(-1, (64, 2)).square().sum(-1).sqrt()
    correction = torch.einsum("btshf,hf->btsh", pair_norm, residual)
    return (trig + correction).permute(0, 3, 1, 2).reshape(batch, 12, tokens)


def select_indices(scores, keep_count, grid, generator):
    """Protect first frame and last3 frames, rank remaining tokens per head."""
    frames, height, width = grid
    per_frame = height * width
    tokens = frames * per_frame
    if keep_count == tokens:
        return torch.arange(tokens, device=scores.device)[None, None].expand(scores.shape[0], scores.shape[1], -1)
    protected = torch.cat((torch.arange(per_frame, device=scores.device),
                           torch.arange((frames - 3) * per_frame, tokens, device=scores.device)))
    candidates = torch.arange(per_frame, (frames - 3) * per_frame, device=scores.device)
    count = keep_count - protected.numel()
    if count <= 0:
        raise ValueError("Budget must leave an active ranked region")
    values = scores[..., candidates]
    values = (values - values.mean(-1, keepdim=True)) / values.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    noise = torch.rand(values.shape, generator=generator, device="cpu", dtype=torch.float32).to(values.device)
    values = values + noise * 1e-6
    ranked = candidates[values.topk(count, dim=-1).indices]
    fixed = protected[None, None].expand(scores.shape[0], scores.shape[1], -1)
    return torch.cat((fixed, ranked), dim=-1).sort(-1).values


class KVPruningIntervention:
    """Insert K/V gathering inside the unmodified native Wan attention path."""
    def __init__(self, model, positive, negative, ratio, calibration):
        import diffusers.models.transformers.transformer_wan as wan
        self.module = wan
        self.original_dispatch = wan.dispatch_attention_fn
        self.model = model
        self.positive = positive
        self.negative = negative
        self.ratio = ratio
        self.calibration = {k: calibration[k].to("cuda") for k in ("q_mean", "q_abs_mean")}
        self.step = 0
        self.branch = None
        self.layer = None
        self.pre_key = None
        self.counts = {}
        self.records = []
        self.tape = {}
        self.generator = torch.Generator(device="cpu").manual_seed(0)
        self.coefficients = {}
        self.handles = [model.register_forward_pre_hook(self.start, with_kwargs=True)]
        for layer, block in enumerate(model.blocks):
            self.handles.append(block.attn1.register_forward_pre_hook(self.enter(layer)))
            self.handles.append(block.attn1.norm_k.register_forward_hook(self.capture_key))
            self.handles.append(block.attn1.register_forward_hook(self.leave))
        wan.dispatch_attention_fn = self.dispatch

    def start(self, module, inputs, kwargs):
        text = kwargs["encoder_hidden_states"]
        if torch.equal(text, self.positive):
            self.branch = 0
        elif torch.equal(text, self.negative):
            self.branch = 1
        else:
            raise RuntimeError("Unknown CFG branch")
        latents = kwargs["hidden_states"]
        self.grid = tuple(latents.shape[i + 2] // module.config.patch_size[i] for i in range(3))

    def enter(self, layer):
        def hook(module, inputs):
            self.layer = layer
            self.pre_key = None
        return hook

    def capture_key(self, module, inputs, output):
        self.pre_key = output

    def leave(self, module, inputs, output):
        self.layer = None
        self.pre_key = None

    def dispatch(self, query, key, value, *args, **kwargs):
        if self.layer is None:
            return self.original_dispatch(query, key, value, *args, **kwargs)
        identity = (self.step, self.branch, self.layer)
        if identity in self.counts:
            raise RuntimeError("Duplicate self-attention dispatch")
        self.counts[identity] = 1
        tokens = query.shape[1]
        keep_count = round(tokens * self.ratio)
        if keep_count == tokens:
            return self.original_dispatch(query, key, value, *args, **kwargs)
        lookup = (self.branch, self.layer)
        if lookup not in self.coefficients:
            mean = self.calibration["q_mean"][self.branch, self.layer]
            residual = self.calibration["q_abs_mean"][self.branch, self.layer] - mean.norm(dim=-1)
            self.coefficients[lookup] = (temporal_coefficients(mean, self.grid[0], query.device), residual)
        coefficient, residual = self.coefficients[lookup]
        scores = importance_scores(self.pre_key, coefficient, residual, self.grid)
        indices = select_indices(scores, keep_count, self.grid, self.generator)
        gather = indices.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, 128)
        retained_key = key.gather(1, gather)
        retained_value = value.gather(1, gather)
        return self.original_dispatch(query, retained_key, retained_value, *args, **kwargs)

    def end_step(self, pipe, step, timestep, kwargs):
        expected = {(step, branch, layer) for branch in (0, 1) for layer in range(30)}
        actual = {k for k in self.counts if k[0] == step}
        if actual != expected:
            raise RuntimeError("Incomplete attention coverage")
        latent = kwargs["latents"].float()
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite latents")
        self.records.append({"step": step + 1, "timestep": float(timestep),
                             "latent_rms": float(latent.square().mean().sqrt()),
                             "elapsed_seconds": time.time() - self.started})
        self.final_latents = kwargs["latents"].detach()
        self.step += 1
        print("STEP", step + 1, round(time.time() - self.started, 2), flush=True)
        return kwargs

    def close(self):
        self.module.dispatch_attention_fn = self.original_dispatch
        for handle in self.handles:
            handle.remove()
