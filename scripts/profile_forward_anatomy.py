#!/usr/bin/env python3
"""P5 of the update profile: anatomy of one teacher forward (both CFG branches as a batch of 2, no gradient, one GPU).

Forwards run through the trainer's own forward_both_branches on stored training states drawn with the trainer's
TrajectoryStore.sample. Module forward pre/post hooks call torch.cuda.synchronize(device) and read
time.perf_counter(); the scaled-dot-product call is timed by wrapping the attention dispatch function the Wan
processor calls. Categories, per forward and summed over the 30 blocks:
  attn1_qkv_norms   to_q, to_k, to_v and norm_q, norm_k of self-attention
  attn1_sdpa        the scaled-dot-product call of self-attention
  attn1_to_out      to_out of self-attention
  attn1_rope_rest   self-attention minus the three above: RoPE application, head reshapes, casts
  attn2             cross-attention (whole module)
  ffn               feed-forward (whole module)
  block_rest        block minus attn1, attn2 and ffn: norms, modulation, residual arithmetic
  patch_embedding, condition_embedder, rope (outside the blocks)
  output_rest       forward minus everything above: output norm/modulation, proj_out, unpatchify
The categories partition the instrumented forward wall time. An uninstrumented forward on the first state must equal
the instrumented one (torch.equal); its wall time is reported to show the timer overhead.
"""
import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_learned_center import ROOT, TrajectoryStore, forward_both_branches


class Anatomy:
    def __init__(self, model, device):
        import diffusers.models.transformers.transformer_wan as wan
        self.device, self.wan = device, wan
        self.cur = None
        self.open = {}
        self.handles = []
        self.attention = None        # "attn1" | "attn2" while inside that module
        self.timing = False
        self._dispatch = wan.dispatch_attention_fn

        def timed_dispatch(*a, **kw):
            if not self.timing or self.attention != "attn1":
                return self._dispatch(*a, **kw)
            t0 = self._now()
            out = self._dispatch(*a, **kw)
            self.cur["attn1_sdpa"] += self._now() - t0
            return out
        wan.dispatch_attention_fn = timed_dispatch
        self._hook(model, "forward_total")
        for name in ("patch_embedding", "condition_embedder", "rope"):
            self._hook(getattr(model, name), name)
        for block in model.blocks:
            self._hook(block, "block")
            self._hook(block.attn1, "attn1", attention="attn1")
            self._hook(block.attn2, "attn2", attention="attn2")
            self._hook(block.ffn, "ffn")
            for name in ("to_q", "to_k", "to_v", "norm_q", "norm_k"):
                self._hook(getattr(block.attn1, name), "attn1_qkv_norms")
            for module in block.attn1.to_out:
                self._hook(module, "attn1_to_out")

    def _now(self):
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _hook(self, module, key, attention=None):
        def pre(m, args):
            if self.timing:
                if attention:
                    self.attention = attention
                self.open[id(m)] = self._now()

        def post(m, args, output):
            if self.timing and id(m) in self.open:
                self.cur[key] += self._now() - self.open.pop(id(m))
                if attention:
                    self.attention = None
        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def run(self, fn):
        self.cur = defaultdict(float)
        self.timing = True
        try:
            out = fn()
        finally:
            self.timing = False
        raw = dict(self.cur)
        c = {"attn1_qkv_norms": raw.get("attn1_qkv_norms", 0.0), "attn1_sdpa": raw.get("attn1_sdpa", 0.0),
             "attn1_to_out": raw.get("attn1_to_out", 0.0)}
        c["attn1_rope_rest"] = raw["attn1"] - c["attn1_qkv_norms"] - c["attn1_sdpa"] - c["attn1_to_out"]
        c["attn2"], c["ffn"] = raw["attn2"], raw["ffn"]
        c["block_rest"] = raw["block"] - raw["attn1"] - raw["attn2"] - raw["ffn"]
        for name in ("patch_embedding", "condition_embedder", "rope"):
            c[name] = raw.get(name, 0.0)
        c["output_rest"] = raw["forward_total"] - raw["block"] - c["patch_embedding"] - c["condition_embedder"] - c["rope"]
        c["forward_total"] = raw["forward_total"]
        return out, c

    def close(self):
        for h in self.handles:
            h.remove()
        self.wan.dispatch_attention_fn = self._dispatch


def sdpa_backend(device):
    """Which scaled-dot-product backend dispatches on a [2, 12, 10800, 128] bf16 input, and the kernel names it runs."""
    q, k, v = (torch.randn(2, 12, 10800, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
    choice = torch._fused_sdp_choice(q, k, v, None, 0.0, False)
    names = {int(v): k for k, v in torch.nn.attention.SDPBackend.__members__.items()}
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(device)
    kernels = sorted({e.key for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0})
    return {"input": "[2, 12, 10800, 128] bf16, no mask, no dropout, not causal", "fused_sdp_choice": names.get(int(choice), str(choice)),
            "kernels": kernels}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--prompts", default="configs/distill_prompts.json")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forwards", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2, help="forwards run and discarded first")
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    out = Path(a.out)
    if "distill-profile" not in out.resolve().parts or (out.exists() and any(out.iterdir())):
        sys.exit("--out must be a new directory under a distill-profile/ directory")
    out.mkdir(parents=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    from diffusers import WanTransformer3DModel
    teacher = WanTransformer3DModel.from_pretrained(a.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                    local_files_only=True).to(device).eval().requires_grad_(False)
    prompts = json.loads((ROOT / a.prompts).read_text())["prompts"]
    store = TrajectoryStore(a.traj, [p["id"] for p in prompts if p["split"] != "val"])
    rng = random.Random(a.seed * 7919)          # the trainer's rank-0 sample stream
    items = [store.sample(rng) for _ in range(a.warmup + a.forwards)]
    plain = []
    for item in items[:a.warmup + 1]:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        y = forward_both_branches(teacher, None, item, device, item["step"])
        torch.cuda.synchronize(device)
        plain.append((time.perf_counter() - t0, y))
    reference = plain[-1][1]
    anatomy = Anatomy(teacher, device)
    rows = []
    for i, item in enumerate(items):
        y, c = anatomy.run(lambda: forward_both_branches(teacher, None, item, device, item["step"]))
        if i == a.warmup:
            equal = torch.equal(y, reference)
        if i >= a.warmup:
            rows.append({"forward": i - a.warmup, "pid": item["pid"], "step": item["step"] + 1, **c})
    anatomy.close()
    uninstrumented = []
    for item in items[a.warmup:]:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        forward_both_branches(teacher, None, item, device, item["step"])
        torch.cuda.synchronize(device)
        uninstrumented.append(time.perf_counter() - t0)
    keys = [k for k in rows[0] if k not in ("forward", "pid", "step")]
    total = statistics.median(r["forward_total"] for r in rows)
    summary = {k: {"median_seconds": statistics.median(r[k] for r in rows), "mean_seconds": statistics.mean(r[k] for r in rows),
                   "share_of_median_total": statistics.median(r[k] for r in rows) / total} for k in keys}
    parts = [k for k in keys if k != "forward_total"]
    result = {"scope": ("teacher (bf16, frozen), one GPU, no gradient; %d forwards after %d discarded warm-ups, each one stored training "
                        "state (trainer rank-0 sample stream, seed %d) with both CFG branches as a batch of 2, 10,800 video tokens; "
                        "per-forward times summed over the 30 blocks; median and mean across forwards; synchronize at every hook"
                        % (a.forwards, a.warmup, a.seed)),
              "outputs_equal_uninstrumented": bool(equal),
              "uninstrumented_forward_seconds": {"median": statistics.median(uninstrumented), "mean": statistics.mean(uninstrumented)},
              "instrumented_forward_seconds": {"median": total, "mean": statistics.mean(r["forward_total"] for r in rows)},
              "categories_sum_to_total_max_abs_error": max(abs(sum(r[k] for k in parts) - r["forward_total"]) for r in rows),
              "summary": summary, "rows": rows, "sdpa": sdpa_backend(device),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "gpu": torch.cuda.get_device_name(device)}
    (out / "forward-anatomy.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows",)}, indent=1), flush=True)
    if not equal:
        sys.exit("instrumented forward differs from the uninstrumented one")


if __name__ == "__main__":
    main()
