"""Per-pair Q/K statistics of the Wan teacher on stored trajectories, for pair selection and static means.

Replays the saved entering latents of the calibration prompts (out/distill-traj-wan/<id>/states.pt, 20 steps)
through the frozen bf16 teacher, both CFG branches, with hooks on attn1.norm_q / norm_k (normalized pre-RoPE
Q/K, the same tensors the replacement acts on). For every (layer, step, branch, head) it records

  mean_q, mean_k   [30, 20, 2, 12, 128]  fp32 coordinate mean over all 10,800 video tokens (full 128 dims)
  energy_q, energy_k [30, 20, 2, 12, 64] fp32 mean over tokens of ||x_f||^2 per rotary pair f

one file per prompt: <out>/<id>.pt. From these, scripts/select_pairs.py derives pair MRL, mean-replacement
error, the damage ranking, the selections and the pooled static means. One GPU per worker, resumable.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from full_sweep import save_torch, write_json
from train_learned_center import TrajectoryStore

ROOT = Path(__file__).resolve().parents[1]


class PairStats:
    def __init__(self, model, steps):
        self.layers, self.heads = len(model.blocks), model.config.num_attention_heads
        self.head_dim = model.config.attention_head_dim
        shape = (self.layers, steps, 2, self.heads)
        self.mean = {s: torch.zeros(*shape, self.head_dim, dtype=torch.float64) for s in ("q", "k")}
        self.energy = {s: torch.zeros(*shape, self.head_dim // 2, dtype=torch.float64) for s in ("q", "k")}
        self.context = None
        self.seen = set()
        self.handles = [getattr(b.attn1, "norm_" + s).register_forward_hook(self._hook(l, s))
                        for l, b in enumerate(model.blocks) for s in ("q", "k")]

    def _hook(self, layer, side):
        def apply(module, inputs, output):
            step = self.context
            for b in range(output.shape[0]):              # batch = [conditional, unconditional]
                key = (layer, step, b, side)
                if key in self.seen:
                    raise RuntimeError(f"duplicate {key}")
                self.seen.add(key)
                x = output[b].double().view(-1, self.heads, self.head_dim)                 # [tokens, heads, 128]
                self.mean[side][layer, step, b] = x.mean(0).cpu()
                self.energy[side][layer, step, b] = x.view(x.shape[0], self.heads, -1, 2).square().sum(-1).mean(0).cpu()
        return apply

    def close(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def main():
    from diffusers import WanTransformer3DModel
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default="out/distill-traj-wan")
    parser.add_argument("--prompts", default="configs/distill_prompts.json")
    parser.add_argument("--split", default="calib")
    parser.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"))
    parser.add_argument("--out", default="out/distill-pair-stats")
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    prompts = json.loads((ROOT / args.prompts).read_text())["prompts"] if not os.path.isabs(args.prompts) else json.loads(Path(args.prompts).read_text())["prompts"]
    ids = [p["id"] for p in prompts if p["split"] == args.split]
    traj = ROOT / args.traj if not os.path.isabs(args.traj) else Path(args.traj)
    store = TrajectoryStore(traj, ids)
    out = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = WanTransformer3DModel.from_pretrained(args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    done = 0
    for pid in store.ids:
        target = out / f"{pid}.pt"
        if target.exists():
            continue
        claim = out / f"{pid}.claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            started = time.time()
            stats = PairStats(model, store.steps)
            for step in range(store.steps):
                item = store.item(pid, step)
                stats.context = step
                x = item["x"].to(device=device, dtype=torch.bfloat16).unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
                text = torch.cat([item["positive"], item["negative"]]).to(device=device, dtype=torch.bfloat16)
                t = torch.full((2,), float(item["timestep"]), device=device)
                model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)
            stats.close()
            expected = stats.layers * store.steps * 2 * 2
            if len(stats.seen) != expected:
                raise RuntimeError(f"hook calls {len(stats.seen)} != {expected}")
            record = {"prompt_id": pid, "steps": store.steps, "timesteps": store.timesteps,
                      "mean_q": stats.mean["q"].float(), "mean_k": stats.mean["k"].float(),
                      "energy_q": stats.energy["q"].float(), "energy_k": stats.energy["k"].float(),
                      "scope": "normalized pre-RoPE Q/K at attn1.norm_q/norm_k; mean and per-pair energy over all video tokens; separate layer/step/branch/head; teacher trajectory states"}
            if not all(torch.isfinite(record[k]).all() for k in ("mean_q", "mean_k", "energy_q", "energy_k")):
                raise RuntimeError("nonfinite statistics")
            save_torch(target, record)
            print("COMPLETE", pid, round(time.time() - started, 1), flush=True)
            done += 1
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    write_json(out / f"worker-{os.getpid()}.json", {"completed": done, "split": args.split, "ids": store.ids})
    print("WORKER COMPLETE", done, flush=True)


if __name__ == "__main__":
    main()
