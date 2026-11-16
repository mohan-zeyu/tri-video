"""Pair audited Wan benchmark arms by prompt and seed, retaining one shared bootstrap."""

import argparse
import json
from pathlib import Path

import numpy as np

from full_sweep import write_json


def read_arm(name, path):
    summary = json.loads(Path(path).read_text())
    if name not in summary["config"]["modes"]:
        raise ValueError(f"{path} does not contain {name}")
    return summary


def teacher_checksums(summary):
    return {
        video["path"]: video["sha256"] for video in summary["videos"]
        if video["path"].startswith("videos/baseline/")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs=2, metavar=("ARM", "SUMMARY"), required=True)
    parser.add_argument("--reference", nargs=2, metavar=("ARM", "SUMMARY"), action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate_name, _ = args.candidate
    candidate = read_arm(*args.candidate)
    prompts = candidate["config"]["prompts"]
    ids = [p["id"] for p in prompts]
    teacher = teacher_checksums(candidate)
    if not ids or len(set(ids)) != len(ids) or set(teacher) != {f"videos/baseline/{pid}.mp4" for pid in ids}:
        raise ValueError("Expected distinct audited teacher prompt-and-seed pairs")
    dimensions = candidate["dimensions"]
    indices = np.random.default_rng(20261109).integers(0, len(ids), size=(10000, len(ids)))
    results = {"candidate": candidate_name, "prompts": len(ids), "replicates": len(indices), "seed": 20261109,
               "dimensions": {d: {} for d in dimensions}}
    for name, path in args.reference:
        reference = read_arm(name, path)
        reference_prompts = {p["id"]: p for p in reference["config"]["prompts"]}
        reference_teacher = teacher_checksums(reference)
        if any(reference_prompts.get(p["id"]) != p for p in prompts) or any(reference_teacher.get(path) != digest for path, digest in teacher.items()):
            raise ValueError(f"{name} has different prompts, seeds or teacher video checksums")
        if reference["config"]["vbench_commit"] != candidate["config"]["vbench_commit"]:
            raise ValueError(f"{name} has a different evaluator revision")
        if set(reference["dimensions"]) != set(dimensions):
            raise ValueError(f"{name} has different dimensions")
        for d in dimensions:
            a = dimensions[d]["per_prompt"]
            b = reference["dimensions"][d]["per_prompt"]
            delta = np.array([a[p][candidate_name] - b[p][name] for p in ids])
            if not np.isfinite(delta).all():
                raise ValueError(f"Nonfinite paired difference for {d}/{name}")
            results["dimensions"][d][f"{candidate_name}_minus_{name}"] = {
                "mean": float(delta.mean()),
                "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
    write_json(args.out, results)


if __name__ == "__main__":
    main()
