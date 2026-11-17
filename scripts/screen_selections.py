"""Training-free screen of pair selections: velocity error of static-mean (and deletion) replacement on the
frozen teacher, over held-out teacher states.

For each validation prompt and each of the 20 entering states, both CFG branches: the native bf16 teacher
velocity, then one forward per arm. Arms = (selection variant from select_pairs.py) x (mode): mode "mean" uses the
pooled static means, mode "delete" zeros the same pairs after RMSNorm, mode "narrow" zeros them before RMSNorm
(the norm then runs over the remaining coordinates; with the "pooled" variant this equals dropping projection
rows). Error = relative squared velocity error (cond, uncond,
guided w=5), exactly the Stage 0 reference metric (scripts/measure_distill_references.py), so results are
comparable with: teacher bf16-vs-fp32 noise 0.0027 and training-free high current means 0.350 (guided).

Exemptions follow the student by default (layer 0 and step 1 untouched); --no-exempt matches the reference's
all-layer/all-step scope; --all-steps keeps layer 0 exempt but touches every step (what a weight-level removal
would do, since dropped rows cannot be restored at one step). One GPU per worker, resumable per prompt.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from full_sweep import write_json
from learned_center import LearnedCenter
from train_learned_center import TrajectoryStore, rel_sq

ROOT = Path(__file__).resolve().parents[1]


def predict(model, item, device):
    x = item["x"].to(device=device, dtype=torch.bfloat16).unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
    text = torch.cat([item["positive"], item["negative"]]).to(device=device, dtype=torch.bfloat16)
    t = torch.full((2,), float(item["timestep"]), device=device)
    return model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)[0].float()


def errors(output, reference):
    guided = lambda v: (v[1] + 5 * (v[0] - v[1])).unsqueeze(0)
    return torch.cat((rel_sq(output, reference), rel_sq(guided(output), guided(reference)))).tolist()


def summarize(out, ids, arms, steps):
    paths = [out / f"{pid}.json" for pid in ids]
    if not all(p.exists() for p in paths):
        return None
    rows = [json.loads(p.read_text()) for p in paths]
    results = {}
    for arm in arms:
        data = torch.tensor([r["arms"][arm] for r in rows], dtype=torch.float64)       # [P, steps, 3]
        results[arm] = {"per_step": data.mean(0).tolist(), "mean": data.mean((0, 1)).tolist()}
    summary = {"ids": ids, "steps": steps, "columns": ["cond", "uncond", "guided_cfg5"], "arms": results,
               "scope": "relative squared velocity error vs the native bf16 teacher at the same teacher entering state; mean over prompts per step, and over all prompt-step cells; static pooled means from calibration prompts; exemptions as recorded in meta"}
    write_json(out / "complete.json", summary)
    lines = ["| arm | cond | uncond | guided (w=5) |", "|---|---:|---:|---:|"]
    for arm in arms:
        m = results[arm]["mean"]
        lines.append(f"| {arm} | {m[0]:.5f} | {m[1]:.5f} | {m[2]:.5f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


@torch.inference_mode()
def main():
    from diffusers import WanTransformer3DModel
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default="out/distill-traj-wan")
    parser.add_argument("--prompts", default="configs/distill_prompts.json")
    parser.add_argument("--snapshot", default=os.environ.get("WAN_SNAPSHOT"))
    parser.add_argument("--selection", default="out/distill-selection/selection.pt")
    parser.add_argument("--statistics", default="out/distill-selection/pair-statistics.pt")
    parser.add_argument("--variants", nargs="*", default=["fixed_high", "separate", "shared"])
    parser.add_argument("--modes", nargs="*", default=["mean", "delete"])
    parser.add_argument("--val-prompts", type=int, default=32, help="first N validation prompts")
    parser.add_argument("--no-exempt", action="store_true", help="apply to layer 0 and step 1 too (reference scope)")
    parser.add_argument("--all-steps", action="store_true", help="keep layer 0 exempt but apply at every step (weight-removal scope)")
    parser.add_argument("--out", default="out/distill-selection/screen")
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    prompts = json.loads((ROOT / args.prompts).read_text())["prompts"] if not os.path.isabs(args.prompts) else json.loads(Path(args.prompts).read_text())["prompts"]
    ids = [p["id"] for p in prompts if p["split"] == "val"][: args.val_prompts]
    traj = ROOT / args.traj if not os.path.isabs(args.traj) else Path(args.traj)
    store = TrajectoryStore(traj, ids)
    if store.ids != ids:
        raise RuntimeError("validation trajectories incomplete")
    out = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = WanTransformer3DModel.from_pretrained(args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    selection = torch.load(ROOT / args.selection if not os.path.isabs(args.selection) else args.selection, map_location="cpu")
    statistics = torch.load(ROOT / args.statistics if not os.path.isabs(args.statistics) else args.statistics, map_location="cpu")
    L, H, D = len(model.blocks), model.config.num_attention_heads, model.config.attention_head_dim
    exempt_layers, exempt_steps = ((), ()) if args.no_exempt else ((0,), () if args.all_steps else (0,))
    arms = {}
    for variant in args.variants:
        for mode in args.modes:
            if mode not in ("mean", "delete", "narrow"):
                raise ValueError(mode)
            center = LearnedCenter(layers=L, steps=store.steps, heads=H, head_dim=D, exempt_layers=exempt_layers, exempt_steps=exempt_steps,
                                   mode=mode, selection=selection["variants"][variant]).to(device)
            if mode == "mean":
                center.init_from_calibration(statistics)
            arms[f"{variant}/{mode}"] = center
    write_json(out / "meta.json", {"arms": list(arms), "val_prompts": ids, "exempt_layers": list(exempt_layers), "exempt_steps": list(exempt_steps),
                                   "selection": str(args.selection), "statistics": str(args.statistics), "pairs_per_side": selection.get("pairs")})
    done = 0
    for pid in ids:
        target = out / f"{pid}.json"
        if target.exists():
            continue
        claim = out / f"{pid}.claim"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        try:
            started = time.time()
            result = {arm: [] for arm in arms}
            for step in range(store.steps):
                item = store.item(pid, step)
                native = predict(model, item, device)
                for name, center in arms.items():
                    center.attach(model)
                    center.set_context(torch.tensor([step, step]), torch.tensor([0, 1]))
                    try:
                        result[name].append(errors(predict(model, item, device), native))
                    finally:
                        center.detach()
            if not torch.isfinite(torch.tensor([result[a] for a in arms])).all():
                raise FloatingPointError(pid)
            write_json(target, {"prompt_id": pid, "seconds": time.time() - started, "arms": result})
            print("COMPLETE", pid, round(time.time() - started, 1), flush=True)
            done += 1
        finally:
            claim.rmdir()
        if args.max_prompts and done >= args.max_prompts:
            break
    summarize(out, ids, list(arms), store.steps)


if __name__ == "__main__":
    main()
