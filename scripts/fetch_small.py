"""Fetch small non-LFS files into a proper HF-cache layout (git sha1 blobs).

The hub snapshot_download hung on a dead connection; this does the same job
with short timeouts: API blobId -> blobs/<sha1> + snapshot symlink.
Usage: python3 fetch_small.py <repo> <prefix-or-file> [...]
"""
import hashlib
import json
import os
import sys
import urllib.request

ENDPOINT = "https://hf-mirror.com"
CACHE = "/mnt/data/liuzeyu/hf-cache/hub"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fetchsmall"})
    with OPENER.open(req, timeout=45) as r:
        return json.load(r)


def fetch(repo, rev, name):
    url = f"{ENDPOINT}/{repo}/resolve/{rev}/{name}"
    model_dir = os.path.join(CACHE, f"models--{repo.replace('/', '--')}")
    snap = os.path.join(model_dir, "snapshots", rev, name)
    if os.path.lexists(snap):
        return f"have {name}"
    req = urllib.request.Request(url, headers={"User-Agent": "fetchsmall"})
    for attempt in range(5):
        try:
            with OPENER.open(req, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:
            print(f"  retry {attempt+1} {name}: {e}", flush=True)
    else:
        raise RuntimeError(f"{name} failed")
    sha1 = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    blob = os.path.join(model_dir, "blobs", sha1)
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    with open(blob, "wb") as f:
        f.write(data)
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    os.symlink(os.path.relpath(blob, os.path.dirname(snap)), snap)
    return f"got  {name} ({len(data)}B sha1 {sha1[:10]})"


def main(repo, patterns):
    info = api(f"{ENDPOINT}/api/models/{repo}?blobs=true")
    rev = info["sha"]
    wanted = [f["rfilename"] for f in info["siblings"]
              if any(f["rfilename"] == p or f["rfilename"].startswith(p) for p in patterns)]
    wanted.sort()
    print(f"[{repo}] rev {rev[:12]}: {len(wanted)} files", flush=True)
    for name in wanted:
        print(f"[{repo}] {fetch(repo, rev, name)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
