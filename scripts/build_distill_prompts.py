"""Build the distillation prompt set: VBench suite prompts plus OpenVid-1M captions, text-disjoint from the
100 evaluation prompts and the 4 calibration prompts.

Sources (download separately, pass paths):
  --vbench   prompts/all_dimension.txt and prompts/all_category.txt from the pinned VBench commit
             fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490 (MIT licence)
  --openvid  data/train/OpenVid-1M.csv from HF dataset nkp37/OpenVid-1M (CC-BY-4.0); the caption column is
             used, nothing else. A head slice of the file is enough.

Selection: every VBench prompt whose normalized text is not an evaluation/calibration prompt; OpenVid captions
with 8-120 words, no URLs, deduplicated by normalized text, sampled without replacement with a fixed seed.
Output: configs/distill_prompts.json with id, text, source, seed (sha256 of id mod 2^31), split
(train / val / calib). The first --calib prompts of the shuffled list are the calibration subset (also trained
on); the next --val prompts are held out from training.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def seed_of(prompt_id):
    return int(hashlib.sha256(prompt_id.encode()).hexdigest(), 16) % (2 ** 31)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vbench", nargs="+", required=True, help="VBench prompt .txt files")
    parser.add_argument("--openvid", required=True, help="OpenVid-1M.csv (or a head slice of it)")
    parser.add_argument("--openvid-count", type=int, default=2000)
    parser.add_argument("--calib", type=int, default=64)
    parser.add_argument("--val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--out", default="configs/distill_prompts.json")
    args = parser.parse_args()

    excluded = set()
    for config in ("configs/center_benchmark_wan.json", "configs/full_sweep_wan.json"):
        for prompt in json.loads((ROOT / config).read_text())["prompts"]:
            excluded.add(normalize(prompt["text"]))
    print(f"excluded evaluation/calibration prompts: {len(excluded)}", file=sys.stderr)

    seen = set(excluded)
    prompts = []
    for path in args.vbench:
        for line in Path(path).read_text().splitlines():
            text = line.strip()
            key = normalize(text)
            if not text or key in seen:
                continue
            seen.add(key)
            prompts.append({"text": text, "source": "vbench"})
    vbench_count = len(prompts)

    csv.field_size_limit(1 << 24)
    candidates = []
    with open(args.openvid, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("caption") or "").strip().replace("\n", " ")
            words = text.split()
            if not (8 <= len(words) <= 120) or "http" in text.lower():
                continue
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(text)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    for text in candidates[: args.openvid_count]:
        prompts.append({"text": text, "source": "openvid"})
    print(f"vbench {vbench_count}, openvid {len(prompts) - vbench_count} of {len(candidates)} candidates", file=sys.stderr)

    rng.shuffle(prompts)
    for index, prompt in enumerate(prompts):
        prompt["id"] = f"d{index:04d}"
        prompt["seed"] = seed_of(prompt["id"])
        prompt["split"] = "calib" if index < args.calib else "val" if index < args.calib + args.val else "train"
    out = {
        "experiment": "wan-learned-center-distillation",
        "selection": __doc__.strip(),
        "selection_seed": args.seed,
        "counts": {"total": len(prompts), "vbench": vbench_count, "openvid": len(prompts) - vbench_count,
                   "calib": args.calib, "val": args.val, "train": len(prompts) - args.val},
        "note": "calib prompts are also training prompts; val prompts are never trained on; all are text-disjoint from the 100 evaluation prompts and p00-p03",
        "prompts": prompts,
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(prompts)} prompts", file=sys.stderr)


if __name__ == "__main__":
    main()
