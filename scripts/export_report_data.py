"""Export auditable report data; optionally capture fresh, position-matched Q/K.

Run on the machine holding the original tensor dumps:
  HF_HOME=/mnt/data/liuzeyu/hf-cache HF_HUB_OFFLINE=1 \
    /dev/shm/trivenv/bin/python scripts/export_report_data.py --capture wan ltx
No original measurements are overwritten. GPU work is limited to one forward
per requested model, with hooks on layer 0 only. Head 0 is fixed in advance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEST = OUT / "report"


def capture(model):
    from probe_video import ADAPTERS

    torch.manual_seed(0)
    ad = ADAPTERS[model]()
    from diffusers import WanTransformer3DModel, LTXVideoTransformer3DModel
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / ("models--" + ad.name.replace("/", "--"))
    snapshot = cache / "snapshots" / (cache / "refs/main").read_text().strip()
    cls = {"wan": WanTransformer3DModel, "ltx": LTXVideoTransformer3DModel}[model]
    net = cls.from_pretrained(str(snapshot), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True).eval().to("cuda")
    ad.model = net
    attn = ad.self_attn(0)
    box = {}

    def hook(key):
        return lambda m, x, y: box.update({key: y.detach()})

    handles = [ad.qk_modules(attn)[0].register_forward_hook(hook("q")),
               ad.qk_modules(attn)[1].register_forward_hook(hook("k")),
               ad.rope_module().register_forward_hook(lambda m, x, y: box.update(freqs=y))]
    with torch.no_grad():
        net(**ad.make_inputs(ad.t_mid, "zeros", 0, "cuda"))
    h = attn.heads
    q, k = [box[s][0].unflatten(-1, (h, -1)).permute(1, 0, 2).float() for s in ("q", "k")]
    qr, kr = ad.rope_apply(q, box["freqs"]), ad.rope_apply(k, box["freqs"])
    # Contribution score follows paper Appendix B.7, Eq. 26.
    zq, zk = q[0].reshape(q.shape[1], -1, 2), k[0].reshape(k.shape[1], -1, 2)
    score = zq.norm(dim=-1).mean(0) * zk.norm(dim=-1).mean(0)
    band = int(score.argmax())
    idx = torch.randperm(q.shape[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(2026))[:512]
    result = {"layer": 0, "head": 0, "band": band, "tokens": q.shape[1],
              "torch": torch.__version__, "diffusers": __import__("diffusers").__version__,
              "model": ad.name, "snapshot": snapshot.name, "dtype": "bfloat16",
              "input": "t_mid-zeros-s0", "sample_indices": idx.tolist(),
              "selection": "argmax E||q_f|| E||k_f||, fixed layer 0 / head 0"}
    for name, x in (("q", q), ("k", k), ("q_post", qr), ("k_post", kr)):
        result[name] = x[0, idx, 2*band:2*band+2].cpu().tolist()
    # Slice along the dominant band's axis, keeping the other coordinates
    # fixed. These are signed coordinate offsets, not flattened distances.
    from analysis_series import angles_from_freqs, predict_logits, token_grid
    grid = token_grid(model, ad, q.shape[1])
    f, ht, wt = grid
    coords = [f, ht, wt]
    axis = (0 if band < 22 else 1 if band < 43 else 2) if model == "wan" else 0
    coordinate = coords[axis]
    extent = int(coordinate.max()) + 1
    edge = torch.where(coordinate == extent - 1)[0]
    qidx = edge[torch.linspace(0, len(edge)-1, 16, device="cuda").long()]
    theta = angles_from_freqs(box["freqs"])
    theta_h = theta[None].expand(h, -1, -1) if theta.shape[-1] == q.shape[-1]//2 else theta.view(q.shape[1], h, -1).permute(1, 0, 2)
    pred = predict_logits(q.mean(1), k.mean(1), theta_h, qidx)[0] / np.sqrt(q.shape[-1])
    actual = torch.einsum("qd,kd->qk", qr[0, qidx], kr[0]) / np.sqrt(q.shape[-1])
    keys = []
    for query in qidx:
        mask = torch.ones(q.shape[1], dtype=torch.bool, device="cuda")
        for j in range(3):
            if j != axis:
                mask &= coords[j] == coords[j][query]
        keys.append(torch.where(mask)[0])
    keys = torch.stack(keys)
    a, p = actual.gather(1, keys).flip(-1), pred.gather(1, keys).flip(-1)
    aa, pp = a-a.mean(-1, keepdim=True), p-p.mean(-1, keepdim=True)
    rr = (aa*pp).sum(-1)/(aa.norm(dim=-1)*pp.norm(dim=-1)).clamp_min(1e-12)
    result["axis_slice"] = {"axis": ["t", "h", "w"][axis], "offset": list(range(extent)), "actual": a.cpu().tolist(),
                              "predicted": p.cpu().tolist(), "mean_r": float(rr.mean()),
                              "query_indices": qidx.tolist(), "protocol": "16 fixed lines along dominant-band axis; query at maximal coordinate; other coordinates fixed; includes self"}
    for handle in handles:
        handle.remove()
    del net, q, k, qr, kr, box
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs="*", choices=["wan", "ltx"], default=[])
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    data = {"source_sha256": {}, "pairs": {}}
    for model in ("wan", "ltx", "cog"):
        path = OUT / f"{model}-pair-mrl.pt"
        data["pairs"][model] = torch.load(path, map_location="cpu", weights_only=False)
        data["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (DEST / "band-data.json").write_text(json.dumps(data) + "\n")
    for model in args.capture:
        result = capture(model)
        (DEST / f"{model}-visual-capture.json").write_text(json.dumps(result) + "\n")
        print(model, "captured", result["band"], "axis-slice r", result["axis_slice"]["mean_r"], flush=True)


if __name__ == "__main__":
    main()
