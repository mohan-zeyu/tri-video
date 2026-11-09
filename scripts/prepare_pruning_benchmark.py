"""Merge held-out calibration and attach verified matching baseline artifacts."""

import hashlib
import json
import os
from pathlib import Path

import torch
from full_sweep import save_torch, write_json


def main():
    root = Path("out/triattention-pruning-wan-20261109")
    old = Path("out/center-benchmark-wan-20261109")
    config = json.loads(Path("configs/triattention_pruning_wan.json").read_text())
    original = json.loads((old / "config.json").read_text())
    for name in ("model", "frames", "height", "width", "steps", "guidance_scale", "negative_prompt", "fps", "prompts"):
        if config[name] != original[name]:
            raise RuntimeError(f"Baseline mismatch {name}")
    names = ("p00", "p01", "p02", "p03")
    items = [torch.load(root / "calibration" / (name + ".pt"), weights_only=True, map_location="cpu") for name in names]
    counts = sum(x["token_count"] for x in items)
    assert (counts == 864000).all()
    mean = (sum(x["q_sum"] for x in items) / counts[..., None, None, None]).float()
    mean_norm = (sum(x["norm_sum"] for x in items) / counts[..., None, None]).float()
    assert torch.all(mean_norm >= mean.norm(dim=-1) - 1e-5)
    calibration = root / "calibration/calibration.pt"
    if not calibration.exists():
        save_torch(calibration, {"q_mean": mean, "q_abs_mean": mean_norm, "token_count": counts})
    prompts = [json.loads((root / "calibration" / (name + ".json")).read_text())["prompt"] for name in names]
    assert not ({p["text"] for p in prompts} & {p["text"] for p in config["prompts"]})
    write_json(root / "calibration/manifest.json", {"prompts": prompts, "tokens_per_branch_layer": 864000,
        "steps_per_prompt": 20, "q_mean_shape": list(mean.shape),
        "pooling": "Four prompts x20steps x10800tokens, independentCFGbranch/layer/head/pair. Evaluation prompts excluded.",
        "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest()})
    write_json(root / "config.json", config)
    for prompt in config["prompts"]:
        if prompt["id"] == "v000":
            continue
        source = old / "records" / prompt["id"] / "baseline"
        dest = root / "records" / prompt["id"] / "baseline"
        dest.mkdir(parents=True, exist_ok=True)
        data = json.loads((source / "complete.json").read_text())
        assert data["prompt"] == prompt and data["tokens"] == 10800 and len(data["steps"]) == 20
        data["baseline_reuse"] = {"source": str(source.resolve()),
            "source_complete_sha256": hashlib.sha256((source / "complete.json").read_bytes()).hexdigest()}
        write_json(dest / "complete.json", data)
        for origin, target in ((source / "final-latents.pt", dest / "final-latents.pt"),
            (old / "videos/baseline" / (prompt["id"] + ".mp4"), root / "videos/baseline" / (prompt["id"] + ".mp4"))):
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(origin, target)
    print("Calibration merged;99matchingbaseline references linked", flush=True)


if __name__ == "__main__":
    main()
