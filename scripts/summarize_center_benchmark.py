"""Audit all paired generations and aggregate official VBench per-video scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from full_sweep import write_json

ARMS = ("baseline", "all", "high")
HIGH_PAIRS = list(range(8)) + list(range(22, 29)) + list(range(43, 50))
GROUPS = {"t": {"high": list(range(8)), "middle": list(range(8, 15)), "low": list(range(15, 22))},
          "h": {"high": list(range(22, 29)), "middle": list(range(29, 36)), "low": list(range(36, 43))},
          "w": {"high": list(range(43, 50)), "middle": list(range(50, 57)), "low": list(range(57, 64))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra-dimension", action="append", default=[])
    args = parser.parse_args()
    root = Path(args.root)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_ids = [p["id"] for p in config["prompts"]]
    if len(set(config_ids)) != 100 or len({p["text"] for p in config["prompts"]}) != 100:
        raise RuntimeError("Expected 100 distinct prompts")
    records = {}
    code_hashes = set()
    video_checks = []
    import cv2
    for prompt in config["prompts"]:
        for arm in ARMS:
            directory = root / "records" / prompt["id"] / arm
            data = json.loads((directory / "complete.json").read_text())
            if data["environment"]["config_sha256"] != expected_hash:
                raise RuntimeError("Generation config mismatch")
            code_hashes.add(json.dumps(data["environment"]["code_sha256"], sort_keys=True))
            if data["intervention_calls"] != 2400 or len(data["steps"]) != 20 or data["tokens"] != 10800:
                raise RuntimeError("Incomplete intervention coverage")
            if data["prompt"] != prompt or data["mode"] != arm:
                raise RuntimeError("Prompt/seed/arm mismatch")
            expected_pairs = list(range(64)) if arm == "all" else HIGH_PAIRS if arm == "high" else []
            if data["selected_pairs"] != expected_pairs:
                raise RuntimeError("Wrong selected pairs")
            video = root / "videos" / arm / (prompt["id"] + ".mp4")
            sha = hashlib.sha256(video.read_bytes()).hexdigest()
            if sha != data["video_sha256"]:
                raise RuntimeError(f"Video checksum mismatch {video}")
            cap = cv2.VideoCapture(str(video))
            actual = [round(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)]
            cap.release()
            if actual != [45, 480, 480, 16]:
                raise RuntimeError(f"Video metadata mismatch {video}: {actual}")
            video_checks.append({"path": str(video.relative_to(root)), "sha256": sha, "frames_width_height_fps": actual})
            records[prompt["id"], arm] = data
    if len(code_hashes) != 1:
        raise RuntimeError("Mixed generation code versions")
    dimensions = config["dimensions"] + args.extra_dimension
    rng = np.random.default_rng(20261109)
    indices = rng.integers(0, 100, size=(10000, 100))
    summary = {"config": config, "dimensions": {}, "videos": video_checks,
               "coverage": {"prompts": 100, "videos": len(video_checks), "qk_hook_calls": sum(d["intervention_calls"] for d in records.values()),
                            "frames_per_video": 45, "tokens_per_forward": 10800, "denoising_steps": 20, "layers": 30, "heads": 12},
               "generation_code_sha256": json.loads(next(iter(code_hashes))),
               "bootstrap": {"replicates": 10000, "seed": 20261109, "unit": "paired prompt with its one fixed seed; retain all arms together", "interval": "2.5th and 97.5th percentiles of resampled mean paired difference"},
               "timing": {}, "audit": {}}
    for arm in ARMS:
        values = [records[p["id"], arm]["inference_decode_seconds"] for p in config["prompts"]]
        summary["timing"][arm] = {"mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values)}
    rows = []
    for dimension in dimensions:
        source = root / "evaluation" / dimension / "complete.json"
        data = json.loads(source.read_text())
        seen = {(row["prompt_id"], row["arm"]): row for row in data["per_video"]}
        if len(data["per_video"]) != 300 or len(seen) != 300 or set(seen) != set(records):
            raise RuntimeError(f"Incomplete paired metric {dimension}")
        scale = 100 if dimension == "imaging_quality" else 1
        values = {arm: np.array([seen[pid, arm]["video_results"] / scale for pid in config_ids], dtype=float) for arm in ARMS}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError("Nonfinite evaluator score")
        item = {"means": {arm: float(v.mean()) for arm, v in values.items()}, "paired": {}, "categories": {}, "per_prompt": {}}
        for name, arm_a, arm_b in (("all_minus_baseline", "all", "baseline"), ("high_minus_baseline", "high", "baseline"), ("high_minus_all", "high", "all")):
            delta = values[arm_a] - values[arm_b]
            bounds = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            item["paired"][name] = {"mean": float(delta.mean()), "ci95": bounds.tolist(), "lower_prompts": int((delta < 0).sum()), "equal_prompts": int((delta == 0).sum()), "higher_prompts": int((delta > 0).sum())}
        for category in sorted({p["category"] for p in config["prompts"]}):
            select = [i for i, p in enumerate(config["prompts"]) if p["category"] == category]
            item["categories"][category] = {"prompts": len(select), "means": {arm: float(v[select].mean()) for arm, v in values.items()}}
        for i, prompt in enumerate(config["prompts"]):
            item["per_prompt"][prompt["id"]] = {arm: float(v[i]) for arm, v in values.items()}
            rows.append({"dimension": dimension, "prompt_id": prompt["id"], "category": prompt["category"], "prompt": prompt["text"], "seed": prompt["seed"],
                         **{arm: float(v[i]) for arm, v in values.items()}, "all_minus_baseline": float(values["all"][i] - values["baseline"][i]), "high_minus_baseline": float(values["high"][i] - values["baseline"][i])})
        summary["dimensions"][dimension] = item
    audited = [p for p in config["prompts"] if p["audit"]]
    for arm in ARMS:
        accum = {}
        for prompt in audited:
            with np.load(root / "records" / prompt["id"] / arm / "pair-diagnostics.npz") as tape:
                for step in (1, 10, 20):
                    for side in ("q", "k"):
                        for metric in ("mrl", "relative_squared_error"):
                            # Conditional branch, individual pair metrics, all 30 layers/12 heads.
                            array = np.stack([tape[f"s{step:02d}_conditional_L{layer:02d}_{side}_{metric}"][0] for layer in range(30)])
                            for axis, bands in GROUPS.items():
                                for band, ids in bands.items():
                                    accum.setdefault((step, side, metric, axis, band), []).append(float(array[..., ids].mean()))
        summary["audit"][arm] = [{"step": key[0], "side": key[1], "metric": key[2], "axis": key[3], "band": key[4],
                                   "mean": float(np.mean(values)), "prompt_values": values}
                                  for key, values in accum.items()]
    summary["audit_scope"] = {"prompt_ids": [p["id"] for p in audited], "steps": [1, 10, 20], "branch": "conditional",
        "within_pair": "All 10800 current video tokens at fixed prompt/step/layer/head/side. MRL = norm(mean vector)/mean norm. Relative squared error = mean squared distance from the fp32 center / mean squared token norm.",
        "summary": "Arithmetic mean of individual pair metrics over specified band pairs, 12 heads, 30 layers and 8 prespecified prompts, separately at each listed step and Q/K side. No cross-step pooling in step profiles. Each arm uses its own evolving trajectory."}
    summary["native_validation"] = json.loads((root / "native-pretrained-validation.json").read_text())
    if any(row["native_attention_output_max_abs_error"] != 0 for row in summary["native_validation"]["native_checks"]):
        raise RuntimeError("Native attention audit failed")
    write_json(root / "summary.json", summary)
    with (root / "per-prompt-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "config.json", config)
    print(json.dumps({"coverage": summary["coverage"], "means": {d: v["means"] for d, v in summary["dimensions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
