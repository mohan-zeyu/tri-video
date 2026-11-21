"""Compare same-seed final latents and decoded MP4 frames against the linked teacher."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from full_sweep import write_json


def frame_psnr(candidate, baseline, shape):
    streams = [cv2.VideoCapture(str(baseline)), cv2.VideoCapture(str(candidate))]
    scores = []
    try:
        for _ in range(shape[0]):
            (oka, a), (okb, b) = (s.read() for s in streams)
            if not oka or not okb or a.shape != shape[1:] or b.shape != a.shape:
                raise RuntimeError(f"Missing or malformed frame: {candidate}")
            mse = np.mean(np.square(a.astype(np.float32) - b, dtype=np.float32), dtype=np.float64)
            # Identical frames have infinite PSNR; retain their count explicitly.
            scores.append(None if mse == 0 else float(10 * np.log10(255**2 / mse)))
        if any(s.read()[0] for s in streams):
            raise RuntimeError(f"Extra video frames: {candidate}")
    finally:
        for s in streams:
            s.release()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = args.root
    out = root / "agreement"
    out.mkdir(exist_ok=True)
    arms = [a for a in config["modes"] if a != "baseline"]
    shape = (config["frames"], config["height"], config["width"], 3)
    for prompt in config["prompts"]:
        pid = prompt["id"]
        records = root / "records" / pid
        videos = root / "videos"
        target = out / f"{pid}.json"
        if target.exists():
            continue
        def latent(arm):
            d = records / arm
            record = json.loads((d / "complete.json").read_text())
            if record["prompt"] != prompt:
                raise RuntimeError(f"Prompt/seed mismatch: {pid}/{arm}")
            z = torch.load(d / "final-latents.pt", map_location="cpu").float()
            expected_shape = (1, 16, (shape[0] - 1) // 4 + 1, shape[1] // 8, shape[2] // 8)
            if tuple(z.shape) != expected_shape or not z.isfinite().all():
                raise RuntimeError(f"Invalid final latent: {pid}/{arm}")
            return z
        baseline = latent("baseline")
        denominator = baseline.norm()
        if denominator <= 0:
            raise RuntimeError(f"Zero teacher final latent: {pid}")
        result = {}
        for arm in arms:
            video = lambda a: videos / a / (pid + ".mp4")
            scores = frame_psnr(video(arm), video("baseline"), shape)
            result[arm] = {"final_latent_relative_l2": float((latent(arm) - baseline).norm() / denominator),
                           "frame_psnr_db": scores, "identical_frames": sum(x is None for x in scores)}
        write_json(target, {"prompt_id": pid, "seed": prompt["seed"], "arms": result})
    records = [json.loads((out / f"{p['id']}.json").read_text())["arms"] for p in config["prompts"]]
    summary = {}
    for arm in arms:
        rows = [r[arm] for r in records]
        psnr = [v for row in rows for v in row["frame_psnr_db"] if v is not None]
        summary[arm] = {"prompts": len(rows), "frames": len(rows) * shape[0],
                        "mean_final_latent_relative_l2": float(np.mean([r["final_latent_relative_l2"] for r in rows])),
                        "mean_finite_frame_psnr_db": float(np.mean(psnr)) if psnr else None,
                        "identical_frames": sum(r["identical_frames"] for r in rows)}
    write_json(out / "complete.json", {"arms": summary,
        "scope": "Same prompt and seed in each arm vs frozen baseline. L2 ratio over all final post-step latent coordinates, then average over prompts. Frame PSNR is 10log10(255^2/RGB MSE) over all channels/pixels of each aligned decoded MP4 frame; average finite dB values over all frames/prompts, count infinite identical frames separately."})


if __name__ == "__main__":
    main()
