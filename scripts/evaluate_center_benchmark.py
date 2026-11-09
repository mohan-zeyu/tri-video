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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--batch-prompts", type=int, default=10)
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
                for arm in config["modes"]:
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
            write_json(output / f"batch-{number:03d}-results.json", {
                "dimension": args.dimension, "official_batch_score": float(score),
                "elapsed_seconds": time.time() - started, "per_video": annotated,
                "source_file": str(Path(module.__file__).resolve()),
                "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                "vbench_commit": config["vbench_commit"]})
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
