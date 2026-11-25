"""Timers for profiling one distillation update of scripts/train_learned_center.py (--profile).

Phases (per sample: data, teacher_forward, student_forward, loss, recompute, backward_rest; per update: center_sync,
clip, optimizer, ema, bookkeeping) are delimited by `mark(name)`: torch.cuda.synchronize(device) and then
time.perf_counter(), so every phase holds the wall time since the previous boundary and the phases of one update
partition its wall time by construction. The independent check is the trainer's own `optimizer_step_seconds`
(time.monotonic from `step_started` to the synchronize after the EMA).

Per-block times (timers "blocks"): forward pre/post hooks on every student block record CUDA events on the current
stream (no synchronize inside the forward, so FSDP prefetch overlap is left alone); a block forward that runs while
`.backward()` is active is gradient-checkpoint recomputation. `recompute` is the sum of those block times, and
`backward_rest` is the wall time of `.backward()` minus `recompute`.

FSDP collectives: torch.distributed.all_gather_into_tensor / reduce_scatter_tensor (the functions FSDP2's
_fsdp_collectives.py calls in this torch) are wrapped to count bytes, attributed to the phase in progress.
"""
import json
import time
from collections import defaultdict
from contextlib import nullcontext

import torch

SAMPLE_PHASES = ("data", "teacher_forward", "student_forward", "loss", "recompute", "backward_rest")
UPDATE_PHASES = ("center_sync", "clip", "optimizer", "ema", "bookkeeping")
PHASES = SAMPLE_PHASES + UPDATE_PHASES


class UpdateProfiler:
    def __init__(self, device, timers, out, rank, arm, warmup, updates, torch_updates=0):
        if timers not in ("none", "phases", "blocks"):
            raise ValueError(timers)
        self.device, self.timers, self.out, self.rank, self.arm = device, timers, out, rank, arm
        self.warmup, self.updates, self.torch_updates = warmup, updates, torch_updates
        self.cuda = device.type == "cuda"
        self.sync = timers != "none"
        self.rows = []
        self.update = None
        self.in_backward = False
        self.block_depth = 0
        self.handles = []
        self.comm = defaultdict(lambda: defaultdict(float))     # phase -> {all_gather_bytes, reduce_scatter_bytes, calls}
        self._patched = []
        self.torch_prof = None
        self.marks = []

    # ---- clock
    def _now(self):
        if self.sync and self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    # ---- per-block hooks
    def attach_blocks(self, student):
        if self.timers != "blocks":
            return
        for index, block in enumerate(student.blocks):
            self.handles.append(block.register_forward_pre_hook(self._block_pre(index), prepend=True))
            self.handles.append(block.register_forward_hook(self._block_post(index), always_call=True))

    def _stamp(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def _block_pre(self, index):
        def hook(module, args):
            if self.update is not None:
                self.block_depth += 1
                self.update["block_open"][index] = (self._stamp(), self.in_backward)
        return hook

    def _block_post(self, index):
        def hook(module, args, output):
            if self.update is not None and index in self.update["block_open"]:
                start, backward = self.update["block_open"].pop(index)
                self.update["block_pairs"].append((self.sample, index, backward, start, self._stamp()))
                self.block_depth -= 1
        return hook

    # ---- collectives
    def patch_collectives(self):
        import torch.distributed as dist
        for name, key in (("all_gather_into_tensor", "all_gather_bytes"), ("reduce_scatter_tensor", "reduce_scatter_bytes")):
            original = getattr(dist, name)

            def wrapper(output, input, *a, _original=original, _key=key, **kw):
                # bytes of the full (unsharded) tensor: the gathered output, or the reduce-scatter input
                full = output if _key == "all_gather_bytes" else input
                self._comm_pending[_key + ("_recompute" if self.block_depth and self.in_backward else "")] += full.numel() * full.element_size()
                self._comm_pending[_key.replace("bytes", "calls")] += 1
                return _original(output, input, *a, **kw)
            setattr(dist, name, wrapper)
            self._patched.append((dist, name, original))
        self._comm_pending = defaultdict(float)

    def unpatch(self):
        for module, name, original in self._patched:
            setattr(module, name, original)
        self._patched = []
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---- update bracketing
    def begin_update(self, step):
        self.sample = -1
        self.update = {"step": step, "phases": defaultdict(float), "samples": [], "block_open": {}, "block_pairs": [], "comm": defaultdict(lambda: defaultdict(float))}
        if hasattr(self, "_comm_pending"):
            self._comm_pending.clear()
        index = step - self.first_step
        if self.torch_updates and index == self.warmup:
            self._start_torch()
        self.t = self._now()

    def mark(self, phase):
        t = self._now()
        per_sample = phase in SAMPLE_PHASES or phase == "backward"
        target = self.update["samples"][self.sample] if per_sample and self.sample >= 0 else self.update["phases"]
        target[phase] = target.get(phase, 0.0) + (t - self.t)
        self.t = t
        if self.torch_prof is not None:
            with torch.profiler.record_function("mark::" + phase):
                pass
        if hasattr(self, "_comm_pending") and self._comm_pending:
            for k, v in self._comm_pending.items():
                if k.endswith("_recompute"):
                    self.update["comm"]["recompute"][k[:-len("_recompute")]] += v
                else:
                    self.update["comm"][phase if phase != "backward" else "backward_rest"][k] += v
            self._comm_pending.clear()

    def begin_sample(self):
        self.sample += 1
        self.update["samples"].append({})

    def backward(self):
        """Context manager around `.backward()`; records the wall time as the provisional phase 'backward'."""
        profiler = self

        class _Backward:
            def __enter__(self_inner):
                profiler.in_backward = True

            def __exit__(self_inner, *exc):
                profiler.in_backward = False
                if exc[0] is None:
                    profiler.mark("backward")
                return False
        return _Backward() if self.update is not None else nullcontext()

    def end_update(self, optimizer_step_seconds, losses=None):
        """Called after the trainer's synchronize and its optimizer_step_seconds read."""
        self.mark("bookkeeping")
        u = self.update
        block_rows = []
        for sample, index, backward, start, end in u["block_pairs"]:
            seconds = start.elapsed_time(end) / 1000 if self.cuda else end - start
            block_rows.append({"sample": sample, "block": index, "pass": "recompute" if backward else "first", "seconds": seconds})
        for i, s in enumerate(u["samples"]):
            if "backward" in s:
                if self.timers == "blocks":
                    s["recompute"] = sum(r["seconds"] for r in block_rows if r["sample"] == i and r["pass"] == "recompute")
                    s["backward_rest"] = s.pop("backward") - s["recompute"]
                else:
                    s["backward_total"] = s.pop("backward")
        index = u["step"] - self.first_step
        measured = index >= self.warmup
        base = {"arm": self.arm, "rank": self.rank, "update": index, "optimizer_step": u["step"] + 1, "measured": measured}
        rows = []
        if self.sync:
            for i, s in enumerate(u["samples"]):
                for phase, seconds in s.items():
                    rows.append({**base, "kind": "sample_phase", "sample": i, "phase": phase, "seconds": seconds})
            for phase, seconds in u["phases"].items():
                rows.append({**base, "kind": "update_phase", "phase": phase, "seconds": seconds})
            phase_sum = sum(sum(s.values()) for s in u["samples"]) + sum(u["phases"].values())
            rows.append({**base, "kind": "phase_sum", "seconds": phase_sum})
        rows.append({**base, "kind": "update_wall", "seconds": optimizer_step_seconds})
        if self.timers == "blocks":
            for r in block_rows:
                rows.append({**base, "kind": "block", **r})
        for phase, counts in u["comm"].items():
            rows.append({**base, "kind": "comm", "phase": phase, **{k: float(v) for k, v in counts.items()}})
        if losses:
            rows.append({**base, "kind": "losses", **losses})
        self.rows.extend(rows)
        with open(self.out / f"phase-times-rank{self.rank}.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if self.torch_prof is not None and index == self.warmup + self.torch_updates - 1:
            self._stop_torch()
        self.update = None

    # ---- torch.profiler (P2)
    def _start_torch(self):
        if self.rank != 0:
            return
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.cuda else [])
        self.torch_prof = profile(activities=activities, record_shapes=False, with_stack=False)
        self.torch_prof.__enter__()

    def _stop_torch(self):
        prof, self.torch_prof = self.torch_prof, None
        prof.__exit__(None, None, None)
        summarize_torch_profile(prof, self.out, self.torch_updates)


def summarize_torch_profile(prof, out, updates):
    """Top 20 CUDA operators by self time, FSDP ranges, and NCCL kernel time attributed to the phase in which it ran
    (between consecutive mark:: events on the host timeline; phases end with a synchronize)."""
    averages = prof.key_averages()
    def self_device(e):
        return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))
    top = sorted(averages, key=self_device, reverse=True)[:20]
    total_self = sum(self_device(e) for e in averages)
    ops = [{"name": e.key, "self_device_seconds": self_device(e) / 1e6, "calls": e.count,
            "share_of_profiled_self_device_time": self_device(e) / total_self if total_self else None} for e in top]
    fsdp = sorted(({"name": e.key, "cpu_seconds": e.cpu_time_total / 1e6, "calls": e.count} for e in averages if e.key.startswith("FSDP::")),
                  key=lambda r: -r["cpu_seconds"])
    events = prof.profiler.kineto_results.events() if prof.profiler and prof.profiler.kineto_results else []
    marks, kernels = [], []
    for ev in events:
        name = ev.name()
        if name.startswith("mark::"):
            marks.append((ev.start_ns(), name[len("mark::"):]))
        elif ev.device_type() == torch.autograd.DeviceType.CUDA and "nccl" in name.lower():
            kernels.append((ev.start_ns(), ev.duration_ns(), name))
    marks.sort()
    by_phase = defaultdict(lambda: defaultdict(float))
    for start, duration, name in kernels:
        phase = next((p for t, p in marks if t >= start), "after_last_mark")
        kind = "all_gather" if "allgather" in name.lower() else "reduce_scatter" if "reducescatter" in name.lower() else "other"
        by_phase[phase][kind + "_seconds"] += duration / 1e9
    result = {"scope": "rank 0, %d measured updates after warm-up under torch.profiler; times summed over those updates" % updates,
              "profiled_updates": updates, "top_cuda_ops_by_self_time": ops, "fsdp_ranges_cpu": fsdp[:40],
              "nccl_kernel_seconds_by_phase": {k: dict(v) for k, v in by_phase.items()},
              "note": "'backward' collects NCCL kernels inside .backward() (recompute and backward_rest together)"}
    (out / "profiler-top-ops.json").write_text(json.dumps(result, indent=1) + "\n")
    try:
        prof.export_chrome_trace(str(out / "trace-rank0.json"))
    except Exception as error:   # the trace is optional evidence; the summary above is the deliverable
        (out / "trace-error.txt").write_text(repr(error))
