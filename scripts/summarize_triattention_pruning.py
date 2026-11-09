"""Verify the query-block pruning study and aggregate paired VBench scores."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import cv2
import torch
from full_sweep import write_json

ROOT = Path("out/triattention-pruning-wan-queryblocks-20261109")
ARMS = ("baseline", "tri50", "tri75")


def main():
    config_path = Path("configs/triattention_pruning_wan_queryblocks.json")
    config = json.loads(config_path.read_text())
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    calibration_path = Path(config["calibration_path"])
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    source_hashes = set()
    timings = {a: [] for a in ARMS}
    video_checks = []
    reuse = 0
    for prompt in config["prompts"]:
        for arm in ARMS:
            directory = ROOT / "records" / prompt["id"] / arm
            record = json.loads((directory / "complete.json").read_text())
            assert record["prompt"] == prompt
            assert record["tokens"] == 10800 and len(record["steps"]) == 20
            if arm != "baseline":
                assert record["intervention_calls"] == 1200
                assert record["environment"]["config_sha256"] == config_sha
                assert record["environment"]["calibration_sha256"] == calibration_sha
                assert record["ranking_pairs"] == list(range(64))
                assert record["retained_keys_per_head"] == {"tri50":5400,"tri75":8100}[arm]
                source_hashes.add(json.dumps(record["environment"]["code_sha256"], sort_keys=True))
            elif "baseline_reuse" in record:
                reuse += 1
            video = ROOT / "videos" / arm / (prompt["id"] + ".mp4")
            sha = hashlib.sha256(video.read_bytes()).hexdigest()
            assert sha == record["video_sha256"], video
            cap = cv2.VideoCapture(str(video))
            metadata = [round(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS)]
            cap.release()
            assert metadata == [45,480,480,16], (video,metadata)
            video_checks.append({"path":str(video.relative_to(ROOT)),"sha256":sha})
            timings[arm].append(record["inference_decode_seconds"])
    assert len(source_hashes) == 1
    native = torch.load(ROOT / "records/v000/baseline/final-latents.pt", map_location="cpu", weights_only=True)
    original_native = torch.load(Path("out/center-benchmark-wan-20261109/records/v000/baseline/final-latents.pt"), map_location="cpu", weights_only=True)
    assert torch.equal(native, original_native)
    rng = np.random.default_rng(20261109)
    bootstrap = rng.integers(0,100,size=(10000,100))
    result = {"config":config,"coverage":{"prompts":100,"videos":300,"new_pruned_videos":200,"reused_baseline_videos":reuse,
        "layers":30,"heads":12,"steps":20,"cfg_branches":2,"tokens":10800},"dimensions":{},"videos":video_checks,
        "generation_source_sha256":json.loads(next(iter(source_hashes))),"calibration_sha256":calibration_sha,
        "native_baseline_verification":{"prompt":"v000","final_latents_bit_exact":True,"scope":"Full20-step trajectory, all30layers/12heads/bothCFG branches, all10800tokens. New full-retention controller versus original native baseline."},
        "calibration":json.loads((ROOT/"calibration/manifest.json").read_text()),
        "selection_verification":json.loads((ROOT/"selection-verification.json").read_text()),
        "bootstrap":{"unit":"100 paired prompt/seed units, all arms retained together","replicates":10000,"seed":20261109,"interval":"percentile95%"},
        "timing":{a:{"mean":statistics.mean(v),"median":statistics.median(v),"min":min(v),"max":max(v)} for a,v in timings.items()}}
    rows = []
    for dimension in config["dimensions"] + ["overall_consistency"]:
        data = json.loads((ROOT/"evaluation"/dimension/"complete.json").read_text())
        lookup = {(r["prompt_id"],r["arm"]):r for r in data["per_video"]}
        assert len(lookup) == len(data["per_video"]) == 300
        divisor = 100 if dimension == "imaging_quality" else 1
        values = {a:np.array([lookup[p["id"],a]["video_results"]/divisor for p in config["prompts"]],float) for a in ARMS}
        assert all(np.isfinite(v).all() for v in values.values())
        item = {"means":{a:float(v.mean()) for a,v in values.items()},"paired":{},"per_prompt":{},"categories":{}}
        for label,a,b in (("tri50_minus_baseline","tri50","baseline"),("tri75_minus_baseline","tri75","baseline"),("tri75_minus_tri50","tri75","tri50")):
            delta = values[a]-values[b]
            item["paired"][label] = {"mean":float(delta.mean()),"ci95":np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist(),
                "lower_prompts":int((delta<0).sum()),"higher_prompts":int((delta>0).sum())}
        for i,prompt in enumerate(config["prompts"]):
            scores = {a:float(v[i]) for a,v in values.items()}
            item["per_prompt"][prompt["id"]] = scores
            rows.append({"dimension":dimension,"prompt_id":prompt["id"],"prompt":prompt["text"],"category":prompt["category"],"seed":prompt["seed"],**scores})
        for category in sorted({p["category"] for p in config["prompts"]}):
            indices = [i for i,p in enumerate(config["prompts"]) if p["category"]==category]
            item["categories"][category] = {"prompts":len(indices),"means":{a:float(v[indices].mean()) for a,v in values.items()}}
        result["dimensions"][dimension] = item
    previous = json.loads(Path("out/center-benchmark-wan-20261109/summary.json").read_text())
    assert previous["config"]["prompts"] == config["prompts"]
    result["previous_replacement"] = {d:{"means":v["means"],"per_prompt":v["per_prompt"]} for d,v in previous["dimensions"].items()}
    write_json(ROOT/"summary.json",result)
    with (ROOT/"per-prompt-results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({"coverage":result["coverage"],"means":{d:v["means"] for d,v in result["dimensions"].items()}},indent=2),flush=True)


if __name__ == "__main__":
    main()
