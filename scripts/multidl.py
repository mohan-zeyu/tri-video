"""Multi-range parallel downloader that writes proper HF-cache structures.

Bypasses the local proxy; pulls each file from HF_ENDPOINT (hf-mirror) with N
concurrent HTTP range requests. Blob names are LFS sha256s, so integrity is
verified by hashing. Idempotent: complete blobs are skipped.

Usage: env -u http_proxy -u https_proxy python3 multidl.py <repo> [<repo> ...]
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

ENDPOINT = "https://hf-mirror.com"
CACHE = "/mnt/data/liuzeyu/hf-cache/hub"
STREAMS = 14          # concurrent range requests per file
CHUNK = 1 << 20

PROXY_HANDLER = urllib.request.ProxyHandler({})  # force no proxy
OPENER = urllib.request.build_opener(PROXY_HANDLER)


def api(url):
    with OPENER.open(urllib.request.Request(url, headers={"User-Agent": "multidl"}), timeout=60) as r:
        return json.load(r)


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "multidl"})
    with OPENER.open(req, timeout=120) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def fetch_range(url, start, end, out_path, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "multidl"})
            with OPENER.open(req, timeout=180) as r, open(out_path, "wb") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
            if os.path.getsize(out_path) == end - start + 1:
                return out_path
        except Exception as e:
            time.sleep(2 + 2 * attempt)
            print(f"  retry {attempt+1} range {start}-{end}: {e}", flush=True)
    raise RuntimeError(f"range {start}-{end} failed after {tries} tries")


def download_file(repo, rev, fname, size, sha):
    model_dir = os.path.join(CACHE, f"models--{repo.replace('/', '--')}")
    blob = os.path.join(model_dir, "blobs", sha)
    snap = os.path.join(model_dir, "snapshots", rev, fname)
    if os.path.exists(blob) and os.path.getsize(blob) == size:
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        if not os.path.lexists(snap):
            os.symlink(os.path.relpath(blob, os.path.dirname(snap)), snap)
        return f"skip {fname}"
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    url = f"{ENDPOINT}/{repo}/resolve/{rev}/{fname}"
    t0 = time.time()
    if size < 8 << 20:  # small file: single GET (no range games)
        fetch_range(url, 0, size - 1, blob + ".tmp0")
        os.replace(blob + ".tmp0", blob)
    else:
        step = (size + STREAMS - 1) // STREAMS
        ranges = [(i * step, min((i + 1) * step, size) - 1) for i in range(STREAMS)]
        parts = [f"{blob}.tmp{i}" for i in range(len(ranges))]
        with cf.ThreadPoolExecutor(STREAMS) as ex:
            list(ex.map(lambda a: fetch_range(url, a[0][0], a[0][1], a[1]),
                        zip(ranges, parts)))
        with open(blob, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while True:
                        buf = f.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                os.remove(p)
    got = os.path.getsize(blob)
    assert got == size, (fname, got, size)
    if len(sha) == 64 and not sha.startswith("fc7de"):  # LFS: name is content sha256
        h = hashlib.sha256()
        with open(blob, "rb") as f:
            while True:
                buf = f.read(8 << 20)
                if not buf:
                    break
                h.update(buf)
        assert h.hexdigest() == sha, f"sha mismatch {fname}"
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    if not os.path.lexists(snap):
        os.symlink(os.path.relpath(blob, os.path.dirname(snap)), snap)
    dt = time.time() - t0
    return f"got  {fname}  {size/1e9:.2f}GB in {dt:.0f}s = {size/dt/1e6:.1f}MB/s"


def download_repo(repo):
    info = api(f"{ENDPOINT}/api/models/{repo}?blobs=true")
    rev = info["sha"]
    model_dir = os.path.join(CACHE, f"models--{repo.replace('/', '--')}")
    os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
    with open(os.path.join(model_dir, "refs", "main"), "w") as f:
        f.write(rev)
    siblings = [f for f in info["siblings"]
                if f["rfilename"].startswith("transformer/")
                and f["rfilename"].endswith((".safetensors", ".json"))]
    big = [f for f in siblings if (f.get("size") or 0) >= 8 << 20 and f.get("lfs")]
    small = [f for f in siblings if f not in big]
    print(f"[{repo}] rev {rev[:12]} big={len(big)} small={len(small)}", flush=True)
    for f in big:  # sequential files, STREAMS ranges each
        oid = f["lfs"]["sha256"]
        print(f"[{repo}] {download_file(repo, rev, f['rfilename'], f['size'], oid)}", flush=True)
    if small:  # correct git-blob names via the hub library itself
        from huggingface_hub import snapshot_download
        snapshot_download(repo, allow_patterns=["transformer/*.json"],
                          revision=rev, max_workers=4)
    print(f"[{repo}] COMPLETE", flush=True)


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        download_repo(repo)
