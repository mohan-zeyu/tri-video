"""Per-pair attention-score attribution of the Wan teacher, for every layer, step, CFG branch and head.

Default source: the stored teacher trajectories (out/distill-traj-wan/<id>/states.pt, 20 entering states),
replayed through the frozen bf16 teacher with both CFG branches, exactly as scripts/collect_pair_statistics.py
does. Hooks on attn1.norm_q / norm_k give the normalized pre-RoPE Q/K of all 10,800 video tokens; the model's
own RoPE angles come from a hook on model.rope. For every (layer, step, branch) the 12 heads are analyzed by
pair_score_attribution.attribute (read its docstring for every field and what it aggregates).

One file per prompt, <out>/<id>.pt, holding one tensor per field with leading dimensions

    [layer 30, step 20, branch 2 (0 = conditional, 1 = unconditional), head 12, ...]

e.g. var_cov, adv, top1 [30, 20, 2, 12, 64]; *_share_quantiles [30, 20, 2, 12, 64, 5]; row_var [30, 20, 2, 12];
pair_cov [30, 20, 2, 12, 64, 64] float16 (about 118 MB per prompt; --no-pair-cov drops it). Nothing is averaged
over prompts, layers, steps, branches or heads; within a cell every token is a query and every token is a key.

Alternative source --sweep-root out/full-sweep-wan-20261101: the saved conditional-branch captures of p00-p03
(pXX/step-NNN/layer-NNN.pt). Those records have a branch dimension of size 1 (conditional only).

At the first analyzed step, three layers of every prompt recompute the attention output from the hooked
Q/K/V and the replicated rotation and compare it with the module's real output (attention_output_validation),
so the analyzed tensors are verified to be the ones the model attends with. One GPU per worker, resumable,
shardable: start one process per idle GPU with the same arguments.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from full_sweep import save_torch, write_json
from full_sweep_metrics import rotate
from pair_score_attribution import QUANTILES, SCOPE, attribute

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("conditional", "unconditional")


def resolve(path):
    return Path(path) if os.path.isabs(str(path)) else ROOT / path


class Record:
    """Per-prompt tensors [layers, steps, branches, heads, ...], filled one (layer, step, branch) at a time."""

    def __init__(self, layers, steps, branches):
        self.shape = (layers, steps, branches)
        self.fields = {}
        self.filled = torch.zeros(self.shape, dtype=torch.bool)

    def put(self, layer, step, branch, result):
        if self.filled[layer, step, branch]:
            raise RuntimeError(f"duplicate cell {(layer, step, branch)}")
        for name, value in result.items():
            value = value.detach().cpu()
            dtype = torch.float16 if name == "pair_cov" else torch.float32
            if name not in self.fields:
                self.fields[name] = torch.zeros(*self.shape, *value.shape, dtype=dtype)
            self.fields[name][layer, step, branch] = value.to(dtype)
        self.filled[layer, step, branch] = True

    def finish(self, expected=None):
        expected = self.filled.numel() if expected is None else expected
        if int(self.filled.sum()) != expected:
            raise RuntimeError(f"cells filled {int(self.filled.sum())} != {expected}")
        for name, value in self.fields.items():
            if not torch.isfinite(value[self.filled]).all():
                raise FloatingPointError(name)
        return dict(self.fields, filled=self.filled)


class ScoreAttribution:
    """Forward hooks that analyze every self-attention layer of each transformer call."""

    def __init__(self, model, record, chunk, query_stride, pair_cov, validate_step):
        self.record, self.chunk, self.query_stride, self.pair_cov = record, chunk, query_stride, pair_cov
        self.heads, self.head_dim = model.config.num_attention_heads, model.config.attention_head_dim
        layers = len(model.blocks)
        self.validate_step, self.validate_layers = validate_step, {0, layers // 2, layers - 1}
        self.step, self.freqs, self.box, self.validation = None, None, {}, {}
        self.handles = [model.rope.register_forward_hook(self._rope)]
        for layer, block in enumerate(model.blocks):
            attn = block.attn1
            for name in ("q", "k"):
                self.handles.append(getattr(attn, "norm_" + name).register_forward_hook(self._grab(layer, name)))
            self.handles.append(attn.to_v.register_forward_hook(self._grab(layer, "v")))
            self.handles.append(attn.register_forward_hook(self._consume(layer)))

    def _rope(self, module, inputs, output):
        cos, sin = output
        self.freqs = cos[0, :, 0, 0::2].float(), sin[0, :, 0, 1::2].float()         # [tokens, 64]

    def _grab(self, layer, name):
        def hook(module, inputs, output):
            if name != "v" or (self.step == self.validate_step and layer in self.validate_layers):
                self.box.setdefault(layer, {})[name] = output.detach()
        return hook

    def _validate(self, attn, q, k, v, cos, sin, output):
        """Module output recomputed from the hooked tensors; q, k [H, N, D], v [N, H*D], output [N, H*D]."""
        qr = rotate(q, cos, sin, q.dtype).to(q.dtype)[None]
        kr = rotate(k, cos, sin, k.dtype).to(k.dtype)[None]
        value = v.unflatten(-1, (self.heads, self.head_dim)).transpose(0, 1)[None]
        reference = F.scaled_dot_product_attention(qr, kr, value)[0].transpose(0, 1).flatten(1)
        reference = attn.to_out[1](attn.to_out[0](reference))
        error, scale = float((reference - output).abs().max()), float(output.abs().max())
        result = {"max_abs_err": error, "output_scale": scale, "relative_max_err": error / max(scale, 1e-12)}
        if result["relative_max_err"] > 1e-3:
            raise RuntimeError(f"attention recomputation mismatch {result}")
        return result

    def _consume(self, layer):
        def hook(attn, inputs, output):
            box = self.box.pop(layer)
            cos, sin = self.freqs
            for branch in range(box["q"].shape[0]):
                q, k = [box[name][branch].unflatten(-1, (self.heads, self.head_dim)).transpose(0, 1) for name in ("q", "k")]
                if q.shape[1] != cos.shape[0]:
                    raise RuntimeError(f"tokens {q.shape[1]} != rotary positions {cos.shape[0]}")
                if "v" in box:
                    self.validation[f"step{self.step}-layer{layer}-{BRANCHES[branch]}"] = self._validate(
                        attn, q, k, box["v"][branch], cos, sin, output[branch])
                self.record.put(layer, self.step, branch, attribute(q, k, cos, sin, self.chunk, self.query_stride, self.pair_cov))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def metadata(args, **extra):
    return dict(extra, quantile_levels=list(QUANTILES), query_stride=args.query_stride, scope=SCOPE,
                pair_order="t pairs 0-21, h 22-42, w 43-63; angular frequency decreases with index inside an axis",
                dimensions="[layer, step, branch, head, ...]; step index 0 = first denoising step")


def replay(args, out, device):
    """Both CFG branches of every stored entering state of the selected distillation prompts."""
    from diffusers import WanTransformer3DModel
    from train_learned_center import TrajectoryStore
    prompts = json.loads(resolve(args.prompts).read_text())["prompts"]
    ids = args.ids or [p["id"] for p in prompts if p["split"] == args.split]
    store = TrajectoryStore(resolve(args.traj), ids)
    model = WanTransformer3DModel.from_pretrained(args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                  local_files_only=True).to(device).eval()
    steps = list(range(store.steps)) if args.steps is None else args.steps
    done = 0
    for pid in store.ids:
        target, claim = out / f"{pid}.pt", out / f"{pid}.claim"
        if target.exists():
            continue
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            started = time.time()
            record = Record(len(model.blocks), store.steps, 2)
            hooks = ScoreAttribution(model, record, args.chunk, args.query_stride, not args.no_pair_cov, steps[0])
            try:
                for step in steps:
                    item = store.item(pid, step)
                    hooks.step = step
                    x = item["x"].to(device=device, dtype=torch.bfloat16).unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
                    text = torch.cat([item["positive"], item["negative"]]).to(device=device, dtype=torch.bfloat16)
                    t = torch.full((2,), float(item["timestep"]), device=device)
                    model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)
            finally:
                hooks.close()
            fields = record.finish(len(model.blocks) * len(steps) * 2)
            fields.update(metadata(args, prompt_id=pid, branches=list(BRANCHES), steps_analyzed=steps,
                                   timesteps=store.timesteps, attention_output_validation=hooks.validation,
                                   seconds=time.time() - started,
                                   source="teacher replay of stored entering states, bf16, both CFG branches in one batch"))
            save_torch(target, fields)
            print("COMPLETE", pid, round(time.time() - started, 1), flush=True)
            done += 1
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    return done, store.ids


def sweep(args, out, device):
    """Saved conditional-branch Q/K of the verified full sweep (p00-p03)."""
    root = resolve(args.sweep_root)
    rope = torch.load(root / "rope.pt", map_location="cpu", weights_only=True)
    cos, sin = rope["cos"].to(device), rope["sin"].to(device)
    ids = args.ids or sorted(p.name for p in root.iterdir() if (p / "complete.json").exists())
    done = 0
    for pid in ids:
        target, claim = out / f"{pid}.pt", out / f"{pid}.claim"
        if target.exists():
            continue
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            started = time.time()
            found = sorted(int(p.name[5:]) for p in (root / pid).glob("step-*"))
            steps = found if args.steps is None else [s + 1 for s in args.steps]
            layers = sorted(int(p.stem[6:]) for p in (root / pid / f"step-{steps[0]:03d}").glob("layer-*.pt"))
            record = Record(len(layers), len(found), 1)
            for step in steps:
                for layer in layers:
                    raw = torch.load(root / pid / f"step-{step:03d}" / f"layer-{layer:03d}.pt", map_location="cpu", weights_only=True)
                    q, k = raw["q"].to(device), raw["k"].to(device)
                    record.put(layer, step - 1, 0, attribute(q, k, cos, sin, args.chunk, args.query_stride, not args.no_pair_cov))
            fields = record.finish(len(layers) * len(steps))
            fields.update(metadata(args, prompt_id=pid, branches=[BRANCHES[0]], steps_analyzed=[s - 1 for s in steps],
                                   seconds=time.time() - started,
                                   source="saved native bf16 pre-RoPE Q/K of the verified conditional full sweep"))
            save_torch(target, fields)
            print("COMPLETE", pid, round(time.time() - started, 1), flush=True)
            done += 1
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    return done, ids


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default="out/distill-traj-wan")
    parser.add_argument("--prompts", default="configs/distill_prompts.json")
    parser.add_argument("--split", default="calib")
    parser.add_argument("--ids", nargs="+", help="explicit prompt ids instead of a split")
    parser.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"))
    parser.add_argument("--sweep-root", help="analyze saved full-sweep captures instead of replaying trajectories")
    parser.add_argument("--out", default="out/distill-score-attribution")
    parser.add_argument("--steps", nargs="+", type=int, help="0-based step indices; default every step")
    parser.add_argument("--chunk", type=int, default=1350, help="queries per attention chunk (GPU memory)")
    parser.add_argument("--query-stride", type=int, default=1, help="attention/top-1 fields use every n-th query; 1 = every query")
    parser.add_argument("--no-pair-cov", action="store_true", help="do not store the 64 x 64 pair covariance")
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    done, ids = (sweep if args.sweep_root else replay)(args, out, device)
    write_json(out / f"worker-{os.getpid()}.json", {"completed": done, "ids": ids, "arguments": vars(args)})
    print("WORKER COMPLETE", done, flush=True)


if __name__ == "__main__":
    main()
