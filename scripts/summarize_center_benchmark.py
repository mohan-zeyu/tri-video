"""Audit all paired generations and aggregate official VBench per-video scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from center_replacement import HIGH_PAIRS

import numpy as np

from full_sweep import write_json

LEGACY_ARMS = ("baseline", "all", "high")
GROUPS = {"t": {"high": list(range(8)), "middle": list(range(8, 15)), "low": list(range(15, 22))},
          "h": {"high": list(range(22, 29)), "middle": list(range(29, 36)), "low": list(range(36, 43))},
          "w": {"high": list(range(43, 50)), "middle": list(range(50, 57)), "low": list(range(57, 64))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra-dimension", action="append", default=[])
    parser.add_argument("--baseline-config", help="original config for a hard-linked legacy teacher baseline")
    parser.add_argument("--generation-config", help="original generation config when reporting a subset of its arms")
    args = parser.parse_args()
    root = Path(args.root)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    generation_path = Path(args.generation_config) if args.generation_config else config_path
    generation_bytes = generation_path.read_bytes()
    expected_hash = hashlib.sha256(generation_bytes).hexdigest()
    arms = config["modes"]
    if "baseline" not in arms or len(set(arms)) != len(arms):
        raise RuntimeError("Expected unique modes including baseline")
    if args.generation_config:
        generation = json.loads(generation_bytes)
        if not set(arms).issubset(generation["modes"]):
            raise RuntimeError("Evaluation requests arms absent from the generation config")
        generation["modes"] = arms
        if generation != config:
            raise RuntimeError("Evaluation config must match the generation config except for the arm list")
    legacy = tuple(arms) == LEGACY_ARMS
    baseline_hash = expected_hash
    if args.baseline_config:
        original = Path(args.baseline_config).read_bytes()
        original_config = json.loads(original)
        for key in ("prompts", "model", "frames", "height", "width", "steps", "guidance_scale", "negative_prompt", "fps"):
            if original_config[key] != config[key]:
                raise RuntimeError(f"Baseline protocol differs: {key}")
        baseline_hash = hashlib.sha256(original).hexdigest()
    comparisons = [(arm, "baseline") for arm in arms if arm != "baseline"]
    if legacy:
        comparisons.append(("high", "all"))
    elif {"student", "static", "deletion", "dense_control"}.issubset(arms):
        comparisons.extend(("student", a) for a in ("static", "deletion", "dense_control"))
    config_ids = [p["id"] for p in config["prompts"]]
    if len(set(config_ids)) != 100 or len({p["text"] for p in config["prompts"]}) != 100:
        raise RuntimeError("Expected 100 distinct prompts")
    records = {}
    code_hashes = {}
    video_checks = []
    import cv2
    for prompt in config["prompts"]:
        for arm in arms:
            directory = root / "records" / prompt["id"] / arm
            data = json.loads((directory / "complete.json").read_text())
            if data["environment"]["config_sha256"] != (baseline_hash if arm == "baseline" else expected_hash):
                raise RuntimeError("Generation config mismatch")
            code_hashes.setdefault(arm, set()).add(json.dumps(data["environment"]["code_sha256"], sort_keys=True))
            if data["prompt"] != prompt or data["mode"] != arm:
                raise RuntimeError("Prompt/seed/arm mismatch")
            if legacy or (arm == "baseline" and args.baseline_config):
                if data["intervention_calls"] != 4 * 30 * config["steps"] or len(data["steps"]) != config["steps"] or data["tokens"] != 10800:
                    raise RuntimeError("Incomplete legacy hook coverage")
                expected_pairs = list(range(64)) if arm == "all" else list(HIGH_PAIRS) if arm == "high" else []
                if data["selected_pairs"] != expected_pairs:
                    raise RuntimeError("Wrong selected pairs")
            else:
                if data["intervention_calls"] != data["expected_intervention_calls"] or data.get("finite_step_latents") != config["steps"]:
                    raise RuntimeError("Incomplete coverage or nonfinite latents")
            video = root / "videos" / arm / (prompt["id"] + ".mp4")
            sha = hashlib.sha256(video.read_bytes()).hexdigest()
            if sha != data["video_sha256"]:
                raise RuntimeError(f"Video checksum mismatch {video}")
            cap = cv2.VideoCapture(str(video))
            actual = [round(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)]
            cap.release()
            if actual != [config["frames"], config["width"], config["height"], config["fps"]]:
                raise RuntimeError(f"Video metadata mismatch {video}: {actual}")
            video_checks.append({"path": str(video.relative_to(root)), "sha256": sha, "frames_width_height_fps": actual})
            records[prompt["id"], arm] = data
    if any(len(hashes) != 1 for hashes in code_hashes.values()) or (
            legacy and len(set.union(*code_hashes.values())) != 1):
        raise RuntimeError("Mixed generation code versions")
    dimensions = config["dimensions"] + args.extra_dimension
    rng = np.random.default_rng(20261109)
    n_prompts = len(config_ids)
    n_videos = n_prompts * len(arms)
    indices = rng.integers(0, n_prompts, size=(10000, n_prompts))
    summary = {"config": config, "dimensions": {}, "videos": video_checks,
               "coverage": {"prompts": n_prompts, "videos": len(video_checks), "qk_hook_calls": sum(d["intervention_calls"] for d in records.values()),
                            "frames_per_video": 45, "tokens_per_forward": 10800, "denoising_steps": 20, "layers": 30, "heads": 12},
               "generation_code_sha256": json.loads(next(iter(code_hashes["baseline"]))) if legacy else
                   {arm: json.loads(next(iter(hashes))) for arm, hashes in code_hashes.items()},
               "bootstrap": {"replicates": 10000, "seed": 20261109, "unit": "paired prompt with its one fixed seed; retain all arms together", "interval": "2.5th and 97.5th percentiles of resampled mean paired difference"},
               "timing": {}, "audit": {}}
    for arm in arms:
        values = [records[p["id"], arm]["inference_decode_seconds"] for p in config["prompts"]]
        summary["timing"][arm] = {"mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values)}
    rows = []
    for dimension in dimensions:
        source = root / "evaluation" / dimension / "complete.json"
        data = json.loads(source.read_text())
        seen = {(row["prompt_id"], row["arm"]): row for row in data["per_video"]}
        if len(data["per_video"]) != n_videos or len(seen) != n_videos or set(seen) != set(records):
            raise RuntimeError(f"Incomplete paired metric {dimension}")
        scale = 100 if dimension == "imaging_quality" else 1
        values = {arm: np.array([seen[pid, arm]["video_results"] / scale for pid in config_ids], dtype=float) for arm in arms}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError("Nonfinite evaluator score")
        item = {"means": {arm: float(v.mean()) for arm, v in values.items()}, "paired": {}, "categories": {}, "per_prompt": {}}
        for arm_a, arm_b in comparisons:
            name = f"{arm_a}_minus_{arm_b}"
            delta = values[arm_a] - values[arm_b]
            bounds = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            item["paired"][name] = {"mean": float(delta.mean()), "ci95": bounds.tolist(), "lower_prompts": int((delta < 0).sum()), "equal_prompts": int((delta == 0).sum()), "higher_prompts": int((delta > 0).sum())}
        for category in sorted({p["category"] for p in config["prompts"]}):
            select = [i for i, p in enumerate(config["prompts"]) if p["category"] == category]
            item["categories"][category] = {"prompts": len(select), "means": {arm: float(v[select].mean()) for arm, v in values.items()}}
        for i, prompt in enumerate(config["prompts"]):
            item["per_prompt"][prompt["id"]] = {arm: float(v[i]) for arm, v in values.items()}
            rows.append({"dimension": dimension, "prompt_id": prompt["id"], "category": prompt["category"], "prompt": prompt["text"], "seed": prompt["seed"],
                         **{arm: float(v[i]) for arm, v in values.items()},
                         **{f"{a}_minus_{b}": float(values[a][i] - values[b][i])
                            for a, b in comparisons if not legacy or b == "baseline"}})
        summary["dimensions"][dimension] = item
    if legacy:
        audited = [p for p in config["prompts"] if p["audit"]]
        for arm in arms:
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
