"""Export all saved pair MRL values without loading a model or raw Q/K."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("q_pair_mrl_pre", "k_pair_mrl_pre", "q_pair_mrl_post", "k_pair_mrl_post")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out/full-sweep-wan-20261101")
    args = parser.parse_args()
    root = ROOT / args.root
    config = json.loads((root / "config.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    expected_hashes = {item["path"]: item["sha256"] for item in coverage["files"]}
    destination = root / "mrl-heatmaps" / "inputs"
    destination.mkdir(parents=True, exist_ok=True)

    def export_condition(condition):
        prompt_id, step = condition
        values = {metric: [] for metric in METRICS}
        sources = []
        pair_axes = None
        for layer in range(30):
            relative = f"{prompt_id}/step-{step:03d}/layer-{layer:03d}.json"
            source = (root / relative).read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            assert digest == expected_hashes[relative], relative
            record = json.loads(source)
            assert (record["prompt_id"], record["step"], record["layer"]) == (prompt_id, step, layer)
            assert (record["tokens"], record["heads"], record["head_dim"]) == (10800, 12, 128)
            assert record["branch"] == "conditional"
            axes = {axis: record["axes"][axis]["pair_indices"] for axis in ("t", "h", "w")}
            if pair_axes is not None:
                assert axes == pair_axes
            pair_axes = axes
            for metric in METRICS:
                matrix = record[metric]
                assert len(matrix) == 12 and all(len(row) == 64 for row in matrix)
                assert all(0 <= number <= 1 for row in matrix for number in row)
                values[metric].append(matrix)
            sources.append({"path": relative, "sha256": digest})
        payload = {
            "schema": 1, "prompt_id": prompt_id, "step": step,
            "model_timestep": record["model_timestep"], "tokens": 10800,
            "shape": [30, 12, 64], "pair_axes": pair_axes,
            "mrl": values, "source_layers": sources,
            "definition": "Each source value is 2D-pair MRL over all 10,800 tokens at one prompt/step/layer/head. No prompt, step, layer, head or pair averaging in these stored values."
        }
        name = f"{prompt_id}-step-{step:03d}.json.gz"
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        with gzip.open(destination / name, "wb") as stream:
            stream.write(encoded)
        return {"path": name, "sha256": hashlib.sha256(encoded).hexdigest(), "prompt_id": prompt_id, "step": step}

    conditions = [(prompt["id"], step) for prompt in config["prompts"] for step in range(1, config["steps"] + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(export_condition, conditions))
    manifest = {
        "conditions": files, "source_layer_count": len(files) * 30,
        "coverage_sha256": hashlib.sha256((root / "coverage.json").read_bytes()).hexdigest(),
        "rope_sha256": hashlib.sha256((root / "rope.pt").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = ROOT / "tmp" / "mrl-heatmap-inputs.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(destination, arcname="inputs")
    print(json.dumps({"conditions": len(files), "verified_source_layers": len(files) * 30, "bundle": str(archive)}))


if __name__ == "__main__":
    main()
