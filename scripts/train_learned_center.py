"""Distill Wan2.1-T2V-1.3B into the learned per-step mean-vector student (see DISTILL_PLAN.md).

Stages
  1  layer-local: teacher network frozen, only mu_q/mu_k train. For each sampled (prompt, step) the teacher is
     run once with the replacement OFF, capturing every block's attn1 input and output (both CFG branches);
     then each layer's attn1 is re-run with the replacement ON and L_attn = sum_l ||A_S - A_T||^2 / ||A_T||^2.
  2  off-policy: all student weights + mu train on teacher-trajectory states x_s with
     L = L_vel + lambda_cfg * L_cfg + lambda_feat * L_feat   (relative squared errors, see DISTILL_PLAN.md section 3)
  3  on-policy: as stage 2, but every --rollout-every steps each rank rolls out --rollouts-per-rank videos with the
     current student (full 20-step CFG-5 UniPC, no gradient), stores the 20 entering latents, and training batches
     draw --onpolicy-frac of their samples from that FIFO buffer; the target is always the teacher's velocity at the
     sampled state.

Parallelism: torchrun with one process per GPU. The student is sharded with FSDP2 (fully_shard per block; fp32
sharded master weights, bf16 compute, fp32 gradient reduction); the frozen bf16 teacher is replicated. The small
LearnedCenter module is replicated and its gradients are all-reduced manually. --no-fsdp runs a single process
with fp32 weights and bf16 autocast (CPU tests, or one large GPU).

Controls: --center-mode off trains the dense control (same recipe, no replacement); --center-mode delete trains
the deletion control. --include-layer0 / --include-step1 lift the default exemptions.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from full_sweep import save_torch, write_json
from learned_center import LearnedCenter

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------------------
# distributed helpers
# ----------------------------------------------------------------------------------------------------------
def dist_init(no_fsdp):
    if no_fsdp or "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    return rank, world, device


def all_reduce_mean(value, world):
    tensor = torch.as_tensor(value, dtype=torch.float64,
                             device=torch.device("cuda", torch.cuda.current_device())
                             if world > 1 and dist.get_backend() == "nccl" else "cpu")
    if world > 1:
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world
    return tensor


def log(rank, *parts):
    if rank == 0:
        print(*parts, flush=True)


# ----------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------
class TrajectoryStore:
    """Teacher trajectories written by scripts/dump_teacher_trajectories.py."""

    def __init__(self, root, prompt_ids):
        self.root = Path(root)
        self.ids = [pid for pid in prompt_ids if (self.root / pid / "meta.json").exists()]
        if not self.ids:
            raise RuntimeError(f"no complete trajectories under {root}")
        meta = json.loads((self.root / self.ids[0] / "meta.json").read_text())
        self.timesteps = meta["timesteps"]
        self.steps = len(self.timesteps)
        self._emb_cache = {}

    def embeddings(self, pid):
        if pid not in self._emb_cache:
            if len(self._emb_cache) > 256:
                self._emb_cache.clear()
            self._emb_cache[pid] = torch.load(self.root / pid / "embeddings.pt")
        return self._emb_cache[pid]

    def sample(self, rng):
        pid = rng.choice(self.ids)
        step = rng.randrange(self.steps)
        return self.item(pid, step)

    def item(self, pid, step):
        states = torch.load(self.root / pid / "states.pt", mmap=True)
        emb = self.embeddings(pid)
        return {"pid": pid, "step": step, "x": states[step].clone(), "timestep": self.timesteps[step],
                "positive": emb["positive"], "negative": emb["negative"], "source": "teacher"}


class OnPolicyBuffer:
    """FIFO of student rollouts: <dir>/<refresh>_<rank>_<i>.pt holding {"states": [steps,...], "pid": str}."""

    def __init__(self, directory, capacity_files, store):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity_files
        self.store = store

    def files(self):
        return sorted(self.dir.glob("*.pt"))

    def add(self, name, states, pid):
        tmp = self.dir / (name + ".tmp")
        torch.save({"states": states.to(torch.bfloat16).cpu(), "pid": pid}, tmp)
        tmp.replace(self.dir / (name + ".pt"))

    def prune(self):
        files = self.files()
        for f in files[: max(0, len(files) - self.capacity)]:
            f.unlink(missing_ok=True)

    def sample(self, rng):
        files = self.files()
        if not files:
            return None
        record = torch.load(rng.choice(files), mmap=True)
        step = rng.randrange(record["states"].shape[0])
        emb = self.store.embeddings(record["pid"])
        return {"pid": record["pid"], "step": step, "x": record["states"][step].clone(), "timestep": self.store.timesteps[step],
                "positive": emb["positive"], "negative": emb["negative"], "source": "student"}


# ----------------------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------------------
def load_models(args, device):
    """Teacher: bf16 via from_pretrained (diffusers keeps rope/time embedder/norms in fp32, as the pipeline does).
    Student: fp32 master weights."""
    from diffusers import WanTransformer3DModel
    snapshot = Path(args.snapshot)
    teacher = WanTransformer3DModel.from_pretrained(snapshot, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
    teacher = teacher.to(device).eval().requires_grad_(False)
    student = None if args.stage == 1 else WanTransformer3DModel.from_pretrained(
        snapshot, subfolder="transformer", torch_dtype=torch.float32, local_files_only=True)
    return teacher, student, snapshot


def shard_student(student, device, world, no_fsdp):
    if no_fsdp or world == 1:
        return student.to(device)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mesh = init_device_mesh(device.type, (world,))
    # A block's RoPE tables are fp32. FSDP casting them during the first
    # forward but not during checkpoint recompute changes both dtype and math.
    # Hidden states and text reach the blocks in bf16 already.
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)

    def cast_model_inputs(module, args, kwargs):
        # Pipeline rollouts see fp32 master parameters outside the FSDP forward.
        # Keep timestep untouched; casting it to bf16 rounds large schedule values.
        return args, {k: (v.to(torch.bfloat16) if k in ("hidden_states", "encoder_hidden_states")
                          and v is not None else v) for k, v in kwargs.items()}

    student.register_forward_pre_hook(cast_model_inputs, with_kwargs=True)
    student = student.to(device)
    for block in student.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(student, mesh=mesh, mp_policy=policy)
    return student


# ----------------------------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------------------------
def rel_sq(a, b):
    """||a-b||^2 / ||b||^2 per batch element, fp32."""
    a, b = a.float(), b.float()
    num = (a - b).flatten(1).square().sum(1)
    den = b.flatten(1).square().sum(1).clamp_min(1e-12)
    return num / den


class FeatureTap:
    def __init__(self, model, layers):
        self.features = {}
        self.handles = [model.blocks[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, layer):
        def apply(module, inputs, output):
            self.features[layer] = output
        return apply

    def close(self):
        for h in self.handles:
            h.remove()


STUDENT_AUTOCAST = False   # set in main: True when the student is an unsharded fp32 model (no FSDP mixed precision)


def forward_both_branches(model, center, item, device, step_index, student=False):
    x = item["x"].to(device=device, dtype=torch.bfloat16).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    text = torch.cat([item["positive"], item["negative"]], 0).to(device=device, dtype=torch.bfloat16)
    timestep = torch.full((2,), float(item["timestep"]), device=device, dtype=torch.float32)
    if center is not None:
        center.set_context(torch.tensor([step_index, step_index]), torch.tensor([0, 1]))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(student and STUDENT_AUTOCAST)):
        return model(hidden_states=x, timestep=timestep, encoder_hidden_states=text, return_dict=False)[0]


def distillation_loss(student, teacher, center, item, device, args, taps):
    """Returns (loss, metrics) for one (prompt, step) sample, both CFG branches as a batch of 2."""
    with torch.no_grad():
        v_t = forward_both_branches(teacher, None, item, device, item["step"]).float()
        feats_t = {l: f.detach() for l, f in taps["teacher"].features.items()}
    v_s = forward_both_branches(student, center, item, device, item["step"], student=True).float()
    feats_s = taps["student"].features
    l_vel = rel_sq(v_s, v_t)                                     # [2]
    guided_t = v_t[1] + args.guidance * (v_t[0] - v_t[1])
    guided_s = v_s[1] + args.guidance * (v_s[0] - v_s[1])
    l_cfg = rel_sq(guided_s.unsqueeze(0), guided_t.unsqueeze(0))[0]
    l_feat = torch.zeros((), device=device)
    if args.lambda_feat > 0 and feats_s:
        l_feat = torch.stack([rel_sq(feats_s[l], feats_t[l]).mean() for l in feats_s]).mean()
    loss = l_vel.sum() + args.lambda_cfg * l_cfg + args.lambda_feat * l_feat
    metrics = {"vel_cond": l_vel[0].item(), "vel_uncond": l_vel[1].item(), "cfg": l_cfg.item(), "feat": l_feat.item()}
    return loss, metrics


class AttentionTap:
    """Stage 1: capture each block's attn1 input (args) and output, plus the rotary embedding."""

    def __init__(self, model):
        self.inputs, self.outputs = {}, {}
        self.handles = []
        for l, block in enumerate(model.blocks):
            self.handles.append(block.attn1.register_forward_pre_hook(self._pre(l), with_kwargs=True))
            self.handles.append(block.attn1.register_forward_hook(self._post(l)))

    def _pre(self, layer):
        def apply(module, args, kwargs):
            self.inputs[layer] = (args, kwargs)
        return apply

    def _post(self, layer):
        def apply(module, inputs, output):
            self.outputs[layer] = output.detach()
        return apply

    def close(self):
        for h in self.handles:
            h.remove()


def stage1_loss(teacher, center, item, device, backward_scale=None):
    """Layer-local mean training. The teacher's own attn1 modules are used as the student attention, with the
    replacement hooks active; the network weights are frozen."""
    tap = AttentionTap(teacher)
    center.mode_backup, center.mode = center.mode, "off"
    with torch.no_grad():
        forward_both_branches(teacher, center, item, device, item["step"])
    center.mode = center.mode_backup
    tap_inputs, tap_outputs = tap.inputs, tap.outputs
    tap.close()
    center.set_context(torch.tensor([item["step"]] * 2), torch.tensor([0, 1]))
    per_layer = []
    for l, block in enumerate(teacher.blocks):
        args, kwargs = tap_inputs.pop(l)
        target = tap_outputs.pop(l)
        with torch.set_grad_enabled(backward_scale is not None):
            out = block.attn1(*args, **kwargs)
            loss_l = rel_sq(out, target).mean()
            # The inputs are detached teacher states. Each layer's gradient can be
            # accumulated immediately without retaining 30 attention graphs.
            if backward_scale is not None and loss_l.requires_grad:
                (loss_l * backward_scale).backward()
        per_layer.append(loss_l.item())
    return sum(per_layer), per_layer


# ----------------------------------------------------------------------------------------------------------
# rollouts (stage 3)
# ----------------------------------------------------------------------------------------------------------
def make_pipeline(student, args):
    from diffusers import UniPCMultistepScheduler, WanPipeline
    scheduler = UniPCMultistepScheduler.from_pretrained(args.snapshot, subfolder="scheduler", local_files_only=True)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=None, scheduler=scheduler, transformer=student)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def rollouts(student, center, store, buffer, args, device, rank, world, refresh, rng):
    student.eval()
    pipe = make_pipeline(student, args)
    shape = store.item(store.ids[0], 0)["x"].shape                                  # [16, T, H, W]
    frames = (shape[1] - 1) * 4 + 1
    height, width = shape[2] * 8, shape[3] * 8
    captured = []

    def pre(module, a, kw):
        if torch.equal(kw["encoder_hidden_states"].to(center.rollout["positive"].dtype), center.rollout["positive"]):
            captured.append(kw["hidden_states"][0].detach().to(torch.bfloat16).cpu())
    for i in range(args.rollouts_per_rank):
        pid = rng.choice(store.ids)
        emb = store.embeddings(pid)
        positive, negative = emb["positive"].to(device, torch.bfloat16), emb["negative"].to(device, torch.bfloat16)
        center.set_rollout(positive, negative)
        captured.clear()
        handle = student.register_forward_pre_hook(pre, with_kwargs=True)
        seed = args.seed + 1_000_003 * refresh + 1000 * rank + i
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=STUDENT_AUTOCAST):
                pipe(prompt_embeds=positive, negative_prompt_embeds=negative, height=height, width=width, num_frames=frames,
                     num_inference_steps=store.steps, guidance_scale=args.guidance, output_type="latent",
                     generator=torch.Generator(device=device.type).manual_seed(seed))
        finally:
            handle.remove()
            center.clear_rollout()
        if len(captured) != store.steps:
            raise RuntimeError(f"rollout captured {len(captured)} states")
        states = torch.stack(captured, 0)
        if torch.isfinite(states.float()).all():
            buffer.add(f"r{refresh:04d}_{rank:02d}_{i:02d}", states, pid)
    if world > 1:
        dist.barrier()
    if rank == 0:
        buffer.prune()
    if world > 1:
        dist.barrier()
    reshard(student)
    if args.stage != 1:
        student.train()


# ----------------------------------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------------------------------
def reshard(student):
    """FSDP2 keeps the root module unsharded after a forward without backward (e.g. validation or rollouts);
    reshard before touching parameters directly (EMA, swaps, state dicts)."""
    if student is not None and hasattr(student, "reshard"):
        student.reshard()


def full_student_state(student, world, no_fsdp):
    if no_fsdp or world == 1:
        return {k: v.detach().cpu() for k, v in student.state_dict().items()}
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(student, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def ema_parameters(student, center):
    return [*student.parameters(), *center.parameters()]


@contextmanager
def use_ema(student, center, ema):
    reshard(student)
    if ema is None:
        yield
        return
    params = ema_parameters(student, center)
    # Move the backup off GPU: during export/rollouts it serves no compute.
    backup = [p.detach().to_local().cpu() if hasattr(p, "to_local") else p.detach().to("cpu", copy=True)
              for p in params]
    with torch.no_grad():
        for p, e in zip(params, ema):
            p.copy_(e)
    try:
        yield
    finally:
        reshard(student)
        with torch.no_grad():
            for p, b in zip(params, backup):
                local = p.to_local() if hasattr(p, "to_local") else p
                local.copy_(b)


def save_student(student, center, out_dir, snapshot, rank, world, no_fsdp, ema=None, tag="student"):
    """Export paired transformer and center EMA weights, with manifest written last."""
    with use_ema(student, center, ema):
        state = full_student_state(student, world, no_fsdp)
        if rank == 0:
            from safetensors.torch import save_file
            target = Path(out_dir) / tag / "transformer"
            target.mkdir(parents=True, exist_ok=True)
            # Diffusers intentionally keeps these parameter groups fp32 at
            # inference; don't round their learned weights on the way to disk.
            keep_fp32 = student._keep_in_fp32_modules or []
            tensors = {k: v.to(torch.float32 if any(part in keep_fp32 for part in k.split("."))
                               else torch.bfloat16).contiguous() for k, v in state.items()}
            pending = target / "diffusion_pytorch_model.pending.safetensors"
            save_file(tensors, str(pending), metadata={"format": "pt"})
            pending.replace(target / "diffusion_pytorch_model.safetensors")
            write_json(target / "config.json", json.loads((Path(snapshot) / "transformer" / "config.json").read_text()))
            save_torch(target.parent / "learned_center.pt", {"state_dict": center.state_dict(), "mode": center.mode})
    if world > 1:
        dist.barrier()


def save_training_state(student, opt_model, center, opt_center, step, out_dir, rank, world, no_fsdp, ema, rng):
    root = Path(out_dir) / "checkpoint"
    ckpt = root / f"step-{step:06d}"
    reshard(student)
    if rank == 0:
        ckpt.mkdir(parents=True, exist_ok=True)
        save_torch(ckpt / "center.pt", {"step": step, "center": center.state_dict(), "opt_center": opt_center.state_dict()})
    ckpt.mkdir(parents=True, exist_ok=True)
    save_torch(ckpt / f"rng-rank{rank:02d}.pt", rng.getstate())
    if opt_model is not None:
        if no_fsdp or world == 1:
            if rank == 0:
                save_torch(ckpt / "model.pt", {"model": student.state_dict(), "opt": opt_model.state_dict(),
                           "ema": [e.cpu() for e in ema] if ema else None})
        else:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict
            msd, osd = get_state_dict(student, opt_model)
            dcp.save({"model": msd, "optim": osd, "ema": {str(i): e for i, e in enumerate(ema)} if ema else {}},
                     checkpoint_id=str(ckpt / "model"))
    if world > 1:
        dist.barrier()
    if rank == 0:
        latest = root / "latest.json"
        previous = json.loads(latest.read_text())["directory"] if latest.exists() else None
        write_json(latest, {"directory": ckpt.name, "step": step})
        if previous and previous != ckpt.name:
            shutil.rmtree(root / previous)
    if world > 1:
        dist.barrier()


def load_training_state(student, opt_model, center, opt_center, out_dir, world, no_fsdp, ema, rank, rng):
    ckpt = Path(out_dir) / "checkpoint"
    latest = ckpt / "latest.json"
    if latest.exists():
        ckpt /= json.loads(latest.read_text())["directory"]
    if not (ckpt / "center.pt").exists():
        raise FileNotFoundError(f"No complete checkpoint under {ckpt}")
    saved = torch.load(ckpt / "center.pt", map_location="cpu")
    center.load_state_dict(saved["center"])
    opt_center.load_state_dict(saved["opt_center"])
    rng_state = ckpt / f"rng-rank{rank:02d}.pt"
    if rng_state.exists():
        rng.setstate(torch.load(rng_state, map_location="cpu"))
    if opt_model is None:
        return saved["step"]
    if no_fsdp or world == 1:
        record = torch.load(ckpt / "model.pt", map_location="cpu")
        student.load_state_dict(record["model"])
        opt_model.load_state_dict(record["opt"])
        if ema and record.get("ema"):
            for e, r in zip(ema, record["ema"]):
                e.copy_(r.to(e.device))
        return saved["step"]
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    msd, osd = get_state_dict(student, opt_model)
    payload = {"model": msd, "optim": osd, "ema": {str(i): e for i, e in enumerate(ema)} if ema else {}}
    dcp.load(payload, checkpoint_id=str(ckpt / "model"))
    set_state_dict(student, opt_model, model_state_dict=payload["model"], optim_state_dict=payload["optim"])
    if ema:
        for i, e in enumerate(ema):
            e.copy_(payload["ema"][str(i)])
    return saved["step"]


# ----------------------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def validate_attention(teacher, center, store, val_ids, device, rank, world):
    """Layer-local error on paired held-out teacher inputs, for active steps only."""
    items = [(pid, s) for pid in val_ids for s in range(store.steps) if bool(center.step_active[s])]
    sums = torch.zeros(center.layers, dtype=torch.float64, device=device)
    for pid, s in items[rank::world]:
        _, per_layer = stage1_loss(teacher, center, store.item(pid, s), device)
        sums += torch.tensor(per_layer, dtype=sums.dtype, device=device)
    if world > 1:
        dist.all_reduce(sums)
    means = sums.cpu() / len(items)
    return {"per_layer": means.tolist(), "median_active_layer": float(means[center.layer_active.cpu()].quantile(0.5)),
            "items": len(items), "prompt_ids": val_ids,
            "scope": "Arithmetic mean over held-out prompts, nonexempt steps and both CFG branches of each layer's relative squared attention output error over all tokens and output coordinates. Uses identical frozen teacher inputs per layer; only normalized pre-RoPE high Q/K pairs are changed. Median over nonexempt layers."}


@torch.no_grad()
def validate(student, teacher, center, store, val_ids, args, device, rank, world, cache):
    """Per-step relative velocity errors (cond, uncond, guided) on held-out teacher states. Each rank evaluates
    an equal share of (prompt, step) items; teacher velocities are cached after the first call."""
    items = [(pid, s) for pid in val_ids for s in range(store.steps)]
    per_rank = math.ceil(len(items) / world)
    # FSDP ranks need the same forward count; padded items participate in
    # collectives without contributing to the metric.
    mine = [items[i] if i < len(items) else items[-1]
            for i in range(rank * per_rank, (rank + 1) * per_rank)]
    student.eval()
    sums = torch.zeros(store.steps, 3, dtype=torch.float64, device=device)
    counts = torch.zeros(store.steps, dtype=torch.float64, device=device)
    for i, (pid, s) in enumerate(mine, rank * per_rank):
        item = store.item(pid, s)
        key = (pid, s)
        if key not in cache:
            cache[key] = forward_both_branches(teacher, None, item, device, s).float().cpu()
        v_t = cache[key].to(device)
        v_s = forward_both_branches(student, center, item, device, s, student=True).float()
        l_vel = rel_sq(v_s, v_t)
        g_t = v_t[1] + args.guidance * (v_t[0] - v_t[1])
        g_s = v_s[1] + args.guidance * (v_s[0] - v_s[1])
        l_cfg = rel_sq(g_s.unsqueeze(0), g_t.unsqueeze(0))[0]
        if i < len(items):
            sums[s] += torch.stack([l_vel[0], l_vel[1], l_cfg]).double()
            counts[s] += 1
    if world > 1:
        dist.all_reduce(sums)
        dist.all_reduce(counts)
    mean = (sums / counts.clamp_min(1).unsqueeze(1)).cpu()
    reshard(student)
    if args.stage != 1:
        student.train()
    return {"per_step": mean.tolist(), "mean": mean.mean(0).tolist(), "items": int(counts.sum().item()),
            "scope": "held-out prompts, teacher-trajectory states, both CFG branches; columns cond, uncond, guided(w=5); relative squared error"}


# ----------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--traj", default="out/distill-traj-wan")
    parser.add_argument("--prompts", default="configs/distill_prompts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"), help="Wan snapshot dir with transformer/ and scheduler/")
    parser.add_argument("--init-student", help="directory with transformer/ + learned_center.pt from a previous stage")
    parser.add_argument("--calibration", help="traj/calibration.pt for mu init (default: <traj>/calibration.pt)")
    parser.add_argument("--center-mode", choices=("mean", "delete", "off"), default="mean")
    parser.add_argument("--include-layer0", action="store_true")
    parser.add_argument("--include-step1", action="store_true")
    parser.add_argument("--steps", type=int, default=5000, help="optimizer steps")
    parser.add_argument("--accum", type=int, default=1, help="samples per rank per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-center", type=float, help="default: 1e-3 for stage 1, 2e-4 otherwise")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lambda-cfg", type=float, default=1.0)
    parser.add_argument("--lambda-feat", type=float, default=0.1)
    parser.add_argument("--feat-layers", type=int, nargs="*", default=[5, 10, 15, 20, 25])
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--onpolicy-frac", type=float, default=0.5)
    parser.add_argument("--onpolicy-ramp", type=int, default=1000)
    parser.add_argument("--rollout-every", type=int, default=200)
    parser.add_argument("--rollouts-per-rank", type=int, default=9)
    parser.add_argument("--buffer-files", type=int, default=500, help="FIFO capacity in rollout files (x steps states)")
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--no-fsdp", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="checkpoint and stop at this optimizer step, retaining the full-run LR schedule")
    parser.add_argument("--budget-seconds", type=float, help="wall-time budget for this invocation; stop at a checkpoint before it is spent")
    args = parser.parse_args()
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    if args.budget_seconds is not None and args.budget_seconds <= 0:
        parser.error("--budget-seconds must be positive")
    if args.lr_center is None:
        args.lr_center = 1e-3 if args.stage == 1 else 2e-4

    rank, world, device = dist_init(args.no_fsdp)
    global STUDENT_AUTOCAST
    STUDENT_AUTOCAST = args.no_fsdp or world == 1
    out = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)
    rng = random.Random(args.seed * 7919 + rank)

    prompts = json.loads((ROOT / args.prompts).read_text())["prompts"] if not os.path.isabs(args.prompts) else json.loads(Path(args.prompts).read_text())["prompts"]
    train_ids = [p["id"] for p in prompts if p["split"] != "val"]
    val_ids = [p["id"] for p in prompts if p["split"] == "val"]
    traj_root = ROOT / args.traj if not os.path.isabs(args.traj) else Path(args.traj)
    store = TrajectoryStore(traj_root, train_ids)
    val_store_ids = [pid for pid in val_ids if (traj_root / pid / "meta.json").exists()]
    log(rank, f"trajectories: train {len(store.ids)}, val {len(val_store_ids)}, steps {store.steps}")

    teacher, student, snapshot = load_models(args, device)
    layers, heads = len(teacher.blocks), teacher.config.num_attention_heads
    center = LearnedCenter(layers=layers, steps=store.steps, heads=heads, head_dim=teacher.config.attention_head_dim,
                           exempt_layers=() if args.include_layer0 else (0,), exempt_steps=() if args.include_step1 else (0,),
                           mode=args.center_mode).to(device)
    if args.init_student:
        init_dir = Path(args.init_student)
        weights_file = init_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        if weights_file.exists() and student is not None:
            from safetensors.torch import load_file
            weights = load_file(str(weights_file))
            missing, unexpected = student.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
            if unexpected or any("blocks" in m for m in missing):
                raise RuntimeError(f"init mismatch: missing {missing[:3]} unexpected {unexpected[:3]}")
            log(rank, "student weights initialized from", weights_file)
        else:
            log(rank, "no transformer weights in", init_dir, "(stage-1 checkpoint): student starts from the teacher weights")
        saved = torch.load(init_dir / "learned_center.pt", map_location="cpu")
        center.load_state_dict(saved["state_dict"])
        log(rank, "mu initialized from", init_dir / "learned_center.pt")
    else:
        calib_path = Path(args.calibration) if args.calibration else traj_root / "calibration.pt"
        if calib_path.exists():
            center.init_from_calibration(torch.load(calib_path, map_location="cpu"))
            log(rank, "mu initialized from", calib_path)
        else:
            log(rank, "WARNING: no calibration file; mu initialized to zero (= deletion)")

    if args.stage == 1:
        # the teacher module doubles as the student attention; only mu trains
        center.attach(teacher)
        opt_model, ema, taps = None, None, None
        student = None
    else:
        if not args.no_checkpointing:
            student.enable_gradient_checkpointing()
        student = shard_student(student, device, world, args.no_fsdp)
        student.train()
        center.attach(student)
        opt_model = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, foreach=False)
        ema = [p.detach().clone() for p in ema_parameters(student, center)] if args.ema > 0 else None
        taps = {"teacher": FeatureTap(teacher, args.feat_layers if args.lambda_feat > 0 else []),
                "student": FeatureTap(student, args.feat_layers if args.lambda_feat > 0 else [])}
    opt_center = torch.optim.Adam(center.parameters(), lr=args.lr_center)

    start = load_training_state(student, opt_model, center, opt_center, out, world, args.no_fsdp, ema, rank, rng) if args.resume else 0
    log(rank, "START_OPTIMIZER_STEP", start + 1)
    buffer = OnPolicyBuffer(out / "onpolicy", args.buffer_files, store) if args.stage == 3 else None
    active_steps = [s for s in range(store.steps) if bool(center.step_active[s])]
    if not active_steps:
        raise RuntimeError("no active steps")
    val_cache = {}
    history = []
    if rank == 0:
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    def lr_scale(step):
        if args.stage == 1:
            return 1.0
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    t0 = time.monotonic()
    if args.stage == 1 and val_store_ids and not args.resume:
        result = validate_attention(teacher, center, store, val_store_ids, device, rank, world)
        if rank == 0:
            write_json(out / "attention-before.json", result)
    stop_at = args.stop_after or args.steps
    last_step = start
    budget_stop = False
    for step in range(start, stop_at):
        if args.stage == 3 and step % args.rollout_every == 0:
            with use_ema(student, center, ema):
                rollouts(student, center, store, buffer, args, device, rank, world, step // args.rollout_every, rng)
        step_started = time.monotonic()
        scale = lr_scale(step)
        for group in opt_center.param_groups:
            group["lr"] = args.lr_center * scale
        if opt_model is not None:
            for group in opt_model.param_groups:
                group["lr"] = args.lr * scale
        opt_center.zero_grad(set_to_none=True)
        if opt_model is not None:
            opt_model.zero_grad(set_to_none=True)
        metrics_acc = {}
        for _ in range(args.accum):
            use_onpolicy = (args.stage == 3 and buffer is not None and
                            rng.random() < args.onpolicy_frac * min(1.0, (step + 1) / max(1, args.onpolicy_ramp)))
            item = buffer.sample(rng) if use_onpolicy else None
            if item is None:
                item = store.sample(rng)
            if args.stage == 1:
                item = store.item(item["pid"], rng.choice(active_steps))
            if args.stage == 1:
                loss, per_layer = stage1_loss(teacher, center, item, device, backward_scale=1 / args.accum)
                metrics = {"attn_total": loss, "attn_layer_median": float(torch.tensor(per_layer).median())}
            else:
                loss, metrics = distillation_loss(student, teacher, center, item, device, args, taps)
                (loss / args.accum).backward()
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f"nonfinite loss at step {step + 1}: {metrics}")
            metrics["onpolicy"] = float(item["source"] == "student")
            for k, v in metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v / args.accum
        # center grads are replicated: average across ranks. Every rank must join the collective, so a rank whose
        # sample hit an exempt step (no gradient on mu) contributes zeros.
        for p in center.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if world > 1:
            for p in center.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if opt_model is not None and args.clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
            metrics_acc["grad_norm"] = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
        torch.nn.utils.clip_grad_norm_(center.parameters(), args.clip if args.clip > 0 else float("inf"))
        opt_center.step()
        if opt_model is not None:
            opt_model.step()
        if ema is not None:
            reshard(student)
            with torch.no_grad():
                for e, p in zip(ema, ema_parameters(student, center)):
                    e.lerp_(p.detach(), 1 - args.ema)
        metrics_acc["mu_norm_q"] = float(center.mu_q.detach().norm(dim=-1).mean())
        metrics_acc["mu_norm_k"] = float(center.mu_k.detach().norm(dim=-1).mean())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics_acc["optimizer_step_seconds"] = time.monotonic() - step_started
        last_step = step + 1
        if (step + 1) % args.log_every == 0 or step in (start, args.steps - 1):
            reduced = {k: float(all_reduce_mean(v, world)) for k, v in sorted(metrics_acc.items())}
            reduced.update({"step": step + 1, "lr_scale": scale, "elapsed_s": round(time.monotonic() - t0, 1)})
            history.append(reduced)
            log(rank, json.dumps(reduced))
        if (step + 1) % args.val_every == 0 and val_store_ids and args.stage != 1:
            result = validate(student, teacher, center, store, val_store_ids, args, device, rank, world, val_cache)
            result["step"] = step + 1
            log(rank, "VAL", json.dumps({"step": step + 1, "mean_cond_uncond_guided": result["mean"]}))
            if rank == 0:
                with open(out / "validation.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")
        if args.budget_seconds:
            # Reserve the larger of 5 minutes or 10 recent optimizer steps for
            # validation/checkpoint I/O and for workers to exit together.
            budget_stop = time.monotonic() - t0 >= args.budget_seconds - max(300, 10 * metrics_acc["optimizer_step_seconds"])
            if world > 1:
                stop_tensor = torch.tensor(int(budget_stop), device=device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                budget_stop = bool(stop_tensor.item())
        if (step + 1) % args.save_every == 0 or step + 1 == stop_at or budget_stop:
            if args.stage == 1:
                if rank == 0:
                    (out / "student").mkdir(exist_ok=True)
                    save_torch(out / "student" / "learned_center.pt", {"state_dict": center.state_dict(), "mode": center.mode})
            else:
                if step + 1 == args.steps:
                    save_student(student, center, out, snapshot, rank, world, args.no_fsdp, ema=ema, tag="student")
            save_training_state(student, opt_model, center, opt_center, step + 1, out, rank, world, args.no_fsdp, ema, rng)
            if rank == 0:
                with open(out / "history.jsonl", "a") as f:
                    for h in history:
                        f.write(json.dumps(h) + "\n")
                history.clear()
        if budget_stop:
            break
    if args.stage == 1 and val_store_ids:
        result = validate_attention(teacher, center, store, val_store_ids, device, rank, world)
        if rank == 0:
            write_json(out / "attention-after.json", result)
    if device.type == "cuda":
        print(json.dumps({"rank": rank, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "wall_seconds": time.monotonic() - t0}), flush=True)
    log(rank, "TRAINING COMPLETE" if last_step == args.steps else "TRAINING PAUSED", args.stage, last_step)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
