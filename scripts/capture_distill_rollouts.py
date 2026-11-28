"""Capture planned Stage 3 rollouts from one fixed EMA snapshot on a spare GPU."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from learned_center import LearnedCenter
from train_learned_center import OnPolicyBuffer, TrajectoryStore, capture_rollout, make_pipeline


def main():
    from diffusers import WanTransformer3DModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--guidance", type=float, default=5.0)
    args = parser.parse_args()

    jobs = json.loads(args.jobs.read_text())
    store = TrajectoryStore(args.traj, list(dict.fromkeys(job["prompt_id"] for job in jobs)))
    device = torch.device("cuda")
    student = WanTransformer3DModel.from_pretrained(
        args.student, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True,
    )
    # FSDP's mixed precision policy casts every parameter during forward.
    # Cast only parameters here; leave Wan's native RoPE buffers at their
    # original precision, as the trainer does.
    for param in student.parameters():
        param.data = param.data.to(torch.bfloat16)
    student = student.to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.student / "learned_center.pt", map_location="cpu")
    center = LearnedCenter(
        layers=len(student.blocks), steps=store.steps, heads=student.config.num_attention_heads,
        head_dim=student.config.attention_head_dim, mode=checkpoint["mode"],
    )
    center.load_state_dict(checkpoint["state_dict"])
    center = center.to(device)
    center.attach(student)
    pipe = make_pipeline(student, SimpleNamespace(snapshot=args.snapshot))
    buffer = OnPolicyBuffer(args.out, len(jobs), store)
    for job in jobs:
        states = capture_rollout(pipe, center, store, job["prompt_id"], job["seed"], device, args.guidance)
        if not torch.isfinite(states.float()).all():
            raise ValueError(f"Nonfinite rollout for {job['name']}")
        buffer.add(job["name"], states, job["prompt_id"])
        print(f"ROLLOUT COMPLETE {job['name']} {job['prompt_id']}", flush=True)


if __name__ == "__main__":
    main()
