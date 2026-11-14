"""Append verified stage measurements to the pilot report after lab workers finish."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    repo, scratch = args.repo, args.scratch
    report = repo / "DISTILL_REPORT.md"
    original = report.read_text().split("\n## Long-run results\n", 1)[0]
    lines = ["", "## Long-run results", ""]

    def load(relative):
        p = repo / relative
        return json.loads(p.read_text()) if p.exists() else None

    spread = load("out/distill-stage0/calibration-spread.json") or load("distill-results/calibration-spread.json")
    if spread:
        lines += [f"Calibration pooled {spread['prompts']} prompts. At each layer, denoising step, CFG branch and head, each prompt contributes the coordinate mean across all 10,800 normalized pre-RoPE tokens. Compare its concatenated 2D high-pair coordinates with the pool for that same cell. Medians and tenth percentiles are over prompts, all 30 layers, 20 steps, both branches and 12 heads; group boundaries are t 0–7, h 22–28 and w 43–49. Q and K are separate.", "", "| Side | Pair group | Median cosine | P10 cosine |", "|---|---|---:|---:|"]
        for side in ("q", "k"):
            for axis in ("t", "h", "w"):
                d = spread["spread"]
                k = f"{side}_{axis}_cos_prompt_vs_pooled_"
                lines.append(f"| {side.upper()} | {axis} | {d[k+'median']:.6f} | {d[k+'p10']:.6f} |")
        lines += ["", "Temporal centers are less directionally stable across prompts than spatial centers, especially in the lower tail. This measures the direction of token averages across prompts.", ""]

    ref = load("out/distill-stage0/reference.json") or load("distill-results/reference.json")
    if ref:
        lines += [f"Stage 0 velocity references: {len(ref['ids'])} held-out prompts × {ref['steps']} entering teacher states, with one seed per prompt. {ref['scope']}", "", "| Reference | Conditional | Unconditional | CFG 5 |", "|---|---:|---:|---:|"]
        for k, name in (("teacher_bf16_vs_fp32", "Teacher bf16 vs. fp32"), ("high_current_mean_vs_teacher_bf16", "Current high-pair means vs. native bf16")):
            a, b, c = ref[k]["mean"]
            lines.append(f"| {name} | {a:.7g} | {b:.7g} | {c:.7g} |")
        lines.append("")

    timing = load("out/distill-training/stage0-timing.json")
    if timing:
        lines += ["Complete teacher capture, including prompt encoding, 20 denoising steps with both CFG branches and artifact writes; excludes VAE decode. The sum is across prompt wall times and does not denote elapsed wall time when workers run concurrently.", "", "| Split | Trajectories | Median seconds | Sum of prompt seconds |", "|---|---:|---:|---:|"]
        lines += [f"| {split} | {r['count']} | {r['median_seconds']:.2f} | {r['sum_seconds']:.1f} |" for split, r in timing.items()]
        lines.append("")

    bef = load("out/distill-training/attention-before.json")
    aft = load("out/distill-training/attention-after.json")
    if bef and aft:
        start, end = bef['median_active_layer'], aft['median_active_layer']
        lines += [f"Stage 1 median per-layer error: {start:.6f} → {end:.6f} ({(start-end)/start:.2%} reduction; 20% gate {'met' if (start-end)/start>=.2 else 'not met'}). {aft['scope']}", ""]

    for stage in ('s2', 's3', 'dense', 'delete'):
        p = scratch / stage / 'validation.jsonl'
        if not p.exists():
            continue
        rows = {d['step']:d for line in p.read_text().splitlines() if line.strip() for d in [json.loads(line)]}
        lines += [f"{stage} validation. {next(iter(rows.values()))['scope']}", "", "| Optimizer step | Conditional | Unconditional | CFG 5 |", "|---:|---:|---:|---:|"]
        for step, d in sorted(rows.items()):
            a, b, c = d['mean']
            lines.append(f"| {step} | {a:.6f} | {b:.6f} | {c:.6f} |")
        lines.append("")

    full_root = "out/learned-center-wan"
    static_root = "out/learned-center-wan-static"
    summary = load(f"{full_root}/summary.json")
    agreement = load(f"{full_root}/agreement/complete.json")
    if not summary:
        summary = load(f"{static_root}/summary.json")
        agreement = load(f"{static_root}/agreement/complete.json")
        if summary:
            lines += ["The completed comparison below pairs the static calibration arm with the frozen teacher. The trained arms and full six-arm evaluation are reported when their outputs are available.", ""]
    if summary:
        arms = summary['config']['modes']
        lines += ["VBench measures the benchmark's single video per seed and prompt. Aesthetic/imaging use all 45 frames (with their official normalizations); subject/background compare first/previous-frame embeddings over 44 transitions; motion smoothness uses AMT over 22 reconstructions; dynamic degree uses RAFT over 22 flow transitions after subsampling to 23 frames; text-video consistency uses ViCLIP on eight midpoint frames and the caption. See `handoff.md` §11 for the pinned evaluator and exact scopes.", "", "Each comparison uses the same 100 prompt/seed units across arms, 10,000 paired prompt resamples and percentile 95% intervals (NumPy 20261109).", "", "| Dimension | " + " | ".join(arms) + " |", "|---|" + "---:|"*len(arms)]
        for d, item in summary['dimensions'].items():
            lines.append(f"| {d} | " + " | ".join(f"{item['means'][a]:.6f}" for a in arms) + " |")
        lines += ["", "| Dimension | Comparison | Paired mean difference | 95% CI |", "|---|---|---:|---|"]
        for d, item in summary['dimensions'].items():
            for k, c in item['paired'].items():
                lines.append(f"| {d} | {k} | {c['mean']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
        lines.append("")
    if agreement:
        lines += [agreement['scope'], "", "| Arm | Prompt pairs | Aligned frames | Latent relative L2 | Mean finite frame PSNR (dB) | Identical frames |", "|---|---:|---:|---:|---:|---:|"]
        for a, r in agreement['arms'].items():
            psnr = r['mean_finite_frame_psnr_db']
            lines.append(f"| {a} | {r['prompts']} | {r['frames']} | {r['mean_final_latent_relative_l2']:.6f} | {f'{psnr:.3f}' if psnr is not None else '—'} | {r['identical_frames']} |")
        lines.append("")
    if len(lines) == 3:
        return
    pending = report.with_suffix('.pending.md')
    pending.write_text(original.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n")
    pending.replace(report)


if __name__ == '__main__':
    main()
