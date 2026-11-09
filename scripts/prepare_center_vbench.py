"""Prepare the six official VBench evaluator checkpoints, with resumable files."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

CACHE = Path(os.environ.get("VBENCH_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "out/center-benchmark-wan-20261109/evaluator-cache")))
FILES = {
    "clip_model/ViT-B-32.pt": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "clip_model/ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "dino_model/dino_vitbase16_pretrain.pth": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "amt_model/amt-s.pth": "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
    "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth": "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
    "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth",
    "raft_model/models.zip": "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
}


def fetch(item):
    name, url = item
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.with_suffix(path.suffix + ".sha256").exists():
        return name
    temporary = path.with_suffix(path.suffix + ".part")
    errors = []
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tri-video-research"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                expected = response.headers.get("Content-Length")
                checksum = hashlib.sha256()
                size = 0
                while chunk := response.read(4 << 20):
                    output.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
            if expected and size != int(expected):
                raise RuntimeError(f"Size mismatch {size}/{expected}")
            sha = checksum.hexdigest()
            if name.startswith("clip_model/") and sha != url.split("/")[-2]:
                raise RuntimeError("CLIP checkpoint SHA256 mismatch")
            temporary.replace(path)
            path.with_suffix(path.suffix + ".sha256").write_text(sha + "\n")
            print(json.dumps({"file": name, "bytes": size, "sha256": sha, "url": url}), flush=True)
            return name
        except Exception as error:
            errors.append(repr(error))
            print("RETRY", name, errors[-1], flush=True)
            time.sleep(2)
    raise RuntimeError(f"Unable to fetch {name}: {errors}")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=7) as executor:
        list(executor.map(fetch, FILES.items()))
    with zipfile.ZipFile(CACHE / "raft_model/models.zip") as archive:
        for member in archive.infolist():
            target = (CACHE / "raft_model" / member.filename).resolve()
            if not target.is_relative_to((CACHE / "raft_model").resolve()):
                raise RuntimeError("Unsafe archive path")
        archive.extractall(CACHE / "raft_model")
    dino = CACHE / "dino_model/facebookresearch_dino_main"
    if not dino.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dino", str(dino)], check=True)
    print("EVALUATOR CHECKPOINTS READY", flush=True)


if __name__ == "__main__":
    main()
