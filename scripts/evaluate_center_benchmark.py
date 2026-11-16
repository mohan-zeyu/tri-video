"""Evaluate complete matched prompt groups with unmodified VBench functions."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

from full_sweep import write_json


def read_audited_baseline(previous, root, config, dimension, source_hash):
    """Reuse frozen teacher scores after matching every prompt, video and scorer."""
    audited = json.loads((previous / "summary.json").read_text())
    for key in ("prompts", "vbench_commit"):
        if audited["config"][key] != config[key]:
            raise RuntimeError(f"Audited baseline differs on {key}")
    checksums = {v["path"]: v["sha256"] for v in audited["videos"]}
    evaluation = previous / "evaluation" / dimension
    batches = list(evaluation.glob("batch-*-results.json"))
    if not batches:
        raise RuntimeError(f"No original VBench batches for {dimension}")
    for path in batches:
        batch = json.loads(path.read_text())
        if batch["source_sha256"] != source_hash or batch["vbench_commit"] != config["vbench_commit"]:
            raise RuntimeError(f"Original VBench scorer mismatch in {path}")
    complete = json.loads((evaluation / "complete.json").read_text())
    rows = [row for row in complete["per_video"] if row["arm"] == "baseline"]
    by_prompt = {row["prompt_id"]: row for row in rows}
    ids = {p["id"] for p in config["prompts"]}
    if set(by_prompt) != ids or len(rows) != len(ids):
        raise RuntimeError(f"Incomplete audited baseline scores for {dimension}")
    scale = 100 if dimension == "imaging_quality" else 1
    for prompt in config["prompts"]:
        pid = prompt["id"]
        video_path = f"videos/baseline/{pid}.mp4"
        record = json.loads((root / "records" / pid / "baseline" / "complete.json").read_text())
        if record["prompt"] != prompt or record["video_sha256"] != checksums[video_path]:
            raise RuntimeError(f"Audited teacher video/seed mismatch for {pid}")
        row = by_prompt[pid]
        if row["video_results"] / scale != audited["dimensions"][dimension]["per_prompt"][pid]["baseline"]:
            raise RuntimeError(f"Raw score differs from the audit for {dimension}/{pid}")
        by_prompt[pid] = {**row, "video_path": str(root / video_path)}
    return by_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--batch-prompts", type=int, default=10)
    parser.add_argument("--reuse-baseline-from", type=Path, help="reuse teacher rows from a previously audited benchmark root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = json.loads(Path(args.config).read_text())
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from vbench.utils import init_submodules
    module = importlib.import_module("vbench." + args.dimension)
    compute = getattr(module, "compute_" + args.dimension)
    source_file = Path(module.__file__).resolve()
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    reused_baseline = None
    scored_arms = config["modes"]
    if args.reuse_baseline_from:
        if "baseline" not in scored_arms or len(scored_arms) < 2:
            raise RuntimeError("Baseline reuse requires baseline and a newly scored arm")
        reused_baseline = read_audited_baseline(args.reuse_baseline_from, root, config, args.dimension, source_hash)
        scored_arms = [arm for arm in scored_arms if arm != "baseline"]
    output = root / "evaluation" / args.dimension
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    seen = set()
    for path in sorted(output.glob("batch-*-results.json")):
        batch = json.loads(path.read_text())
        summaries.extend(batch["per_video"])
        seen.update(row["prompt_id"] for row in batch["per_video"])
    modules = init_submodules([args.dimension], local=True, read_frame=False)[args.dimension]
    while True:
        pending = []
        for prompt in config["prompts"]:
            if prompt["id"] in seen:
                continue
            if all((root / "records" / prompt["id"] / arm / "complete.json").exists() for arm in config["modes"]):
                pending.append(prompt)
        if pending:
            batch = pending[:args.batch_prompts]
            metadata = []
            by_path = {}
            for prompt in batch:
                for arm in scored_arms:
                    video = root / "videos" / arm / (prompt["id"] + ".mp4")
                    by_path[str(video)] = (prompt["id"], arm)
                    metadata.append({"prompt_en": prompt["text"], "dimension": [args.dimension], "video_list": [str(video)]})
            number = len(list(output.glob("batch-*-results.json")))
            info_path = output / f"batch-{number:03d}-full-info.json"
            write_json(info_path, metadata)
            started = time.time()
            score, rows = compute(str(info_path), torch.device("cuda"), modules)
            annotated = []
            for row in rows:
                prompt_id, arm = by_path[row["video_path"]]
                annotated.append({**row, "prompt_id": prompt_id, "arm": arm})
            if len(annotated) != len(metadata):
                raise RuntimeError("Evaluator omitted videos")
            reused = {}
            if reused_baseline is not None:
                annotated.extend(reused_baseline[p["id"]] for p in batch)
                reused["reused_baseline_from"] = str(args.reuse_baseline_from.resolve())
            write_json(output / f"batch-{number:03d}-results.json", {
                "dimension": args.dimension, "official_batch_score": float(score),
                "elapsed_seconds": time.time() - started, "per_video": annotated,
                "source_file": str(source_file), "source_sha256": source_hash,
                "vbench_commit": config["vbench_commit"], **reused})
            summaries.extend(annotated)
            seen.update(p["id"] for p in batch)
            write_json(output / "progress.json", {"dimension": args.dimension, "complete_prompts": len(seen),
                "per_video": summaries, "updated_unix": time.time()})
            print("EVALUATED", args.dimension, len(seen), "prompts", flush=True)
            torch.cuda.empty_cache()
            continue
        if len(seen) == len(config["prompts"]):
            write_json(output / "complete.json", {"dimension": args.dimension,
                "prompts": len(seen), "videos": len(summaries), "per_video": summaries,
                "aggregation": "Original VBench frame aggregation within each video, paired means over 100 prompts separately per arm. MUSIQ per-video raw scores require division by 100 for official scale."})
            return
        if not args.watch:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
