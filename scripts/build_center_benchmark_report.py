"""Build the detailed Markdown, HTML and PDF center-replacement report."""

import argparse
import html
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

ARMS = ("baseline", "all", "high")
COLORS = {"baseline": "#31566f", "all": "#bf5055", "high": "#d8952c"}
NAMES = {"baseline": "Baseline", "all": "All 64 pairs", "high": "High 22 pairs"}
LABELS = {"aesthetic_quality": "Aesthetic quality", "imaging_quality": "Imaging quality",
          "overall_consistency": "Text-video consistency", "subject_consistency": "Subject consistency",
          "background_consistency": "Background consistency", "motion_smoothness": "Motion smoothness",
          "dynamic_degree": "Dynamic degree"}
ORDER = tuple(LABELS)
SCOPES = {
    "aesthetic_quality": "CLIP ViT-L/14 plus the official aesthetic linear predictor. Average all 45 frame scores, divide by 10, then average 100 per-video scores per arm.",
    "imaging_quality": "MUSIQ-SPAQ. Average all 45 frame scores per video, divide by 100, then average 100 per-video scores per arm. Official longer-edge preprocessing retains the 480 x 480 resolution.",
    "overall_consistency": "Official ViCLIP evaluator. Sample the middle frame of each of eight equal temporal segments. Compute video/text feature cosine similarity for the actual prompt, then average 100 video scores per arm.",
    "subject_consistency": "DINO ViT-B/16. For each of frames 1-44, average its nonnegative cosine similarity with the previous frame and with frame 0. Average 44 transition scores per video, then 100 videos per arm.",
    "background_consistency": "CLIP ViT-B/32. Use the same previous/first-frame cosine aggregation over 44 transitions, then average 100 videos per arm.",
    "motion_smoothness": "Official AMT-S. Interpolate odd-index frames from adjacent even-index frames. Average absolute reconstruction error over 22 interpolated frames and pixels, convert to 1 - error/255, then average 100 videos per arm.",
    "dynamic_degree": "Official RAFT. Subsample the 16-fps video to 8 fps, yielding 23 frames and 22 flow transitions. Threshold the mean of the largest 5% flow magnitudes at 11.25 pixels and require six transitions above threshold. Average the 100 resulting motion-incidence booleans per arm.",
}


def number(value):
    return f"{value:.4f}"


def interval(item):
    low, high = item["ci95"]
    return f"{item['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def make_figures(root, data):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    dimensions = [d for d in ORDER if d in data["dimensions"]]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    positions = np.arange(len(dimensions))
    for offset, arm in ((-.12, "all"), (.12, "high")):
        items = [data["dimensions"][d]["paired"][arm + "_minus_baseline"] for d in dimensions]
        means = np.array([v["mean"] for v in items])
        bounds = np.array([v["ci95"] for v in items])
        ax.errorbar(means, positions + offset, xerr=np.array([means - bounds[:, 0], bounds[:, 1] - means]),
                    fmt="o", ms=5, capsize=3, color=COLORS[arm], label=NAMES[arm])
    ax.axvline(0, color="#78848c", lw=1)
    ax.set_yticks(positions, [LABELS[d] for d in dimensions])
    ax.invert_yaxis()
    ax.set_xlabel("Paired score change from baseline (95% prompt-bootstrap CI)")
    ax.grid(axis="x", alpha=.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "paired-deltas.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.7), sharex=True, sharey="row")
    for column, axis in enumerate(("t", "h", "w")):
        for side, linestyle in (("q", "-"), ("k", "--")):
            for band, color in (("high", "#31566f"), ("middle", "#b46c28"), ("low", "#8e5a89")):
                for row, metric in enumerate(("mrl", "relative_squared_error")):
                    values = [next(v["mean"] for v in data["audit"]["baseline"]
                                   if (v["step"], v["side"], v["axis"], v["band"], v["metric"]) == (step, side, axis, band, metric)) for step in (1, 10, 20)]
                    axes[row, column].plot([1, 10, 20], values, linestyle=linestyle, marker="o", markersize=3,
                                           color=color, label=f"{band} {side.upper()}")
        axes[0, column].set_title(axis)
        axes[1, column].set_xlabel("Denoising step")
        for ax in axes[:, column]:
            ax.set_xticks([1, 10, 20])
            ax.set_ylim(0, 1)
            ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Mean individual-pair MRL")
    axes[1, 0].set_ylabel("Mean pair relative squared error")
    axes[0, 2].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "baseline-pair-diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    categories = list(data["dimensions"]["imaging_quality"]["categories"])
    fig, ax = plt.subplots(figsize=(8.5, 3.1))
    for i, arm in enumerate(ARMS):
        values = [data["dimensions"]["imaging_quality"]["categories"][c]["means"][arm] for c in categories]
        ax.bar(np.arange(8) + (i - 1) * .24, values, width=.23, color=COLORS[arm], label=NAMES[arm])
    ax.set_xticks(np.arange(8), categories, fontsize=8)
    ax.set_ylabel("Mean MUSIQ score / 100")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(figures / "imaging-by-category.png")
    plt.close(fig)
    stills = root / "stills"
    stills.mkdir(exist_ok=True)
    audited = [p for p in data["config"]["prompts"] if p["audit"]]
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 23)
    for prompt in audited:
        canvas = PILImage.new("RGB", (1440, 524), "white")
        draw = ImageDraw.Draw(canvas)
        for i, arm in enumerate(ARMS):
            image_path = stills / f"{prompt['id']}-{arm}.png"
            if not image_path.exists():
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "1.375", "-i",
                    str(root / "videos" / arm / f"{prompt['id']}.mp4"), "-frames:v", "1", "-y", str(image_path)], check=True)
            canvas.paste(PILImage.open(image_path), (i * 480, 44))
            draw.text((i * 480 + 12, 10), NAMES[arm], font=font, fill=COLORS[arm])
        canvas.save(figures / f"{prompt['id']}-comparison.png")


def content(data):
    dims = data["dimensions"]
    findings = []
    for dimension in ("aesthetic_quality", "imaging_quality", "overall_consistency"):
        if dimension not in dims:
            continue
        entry = dims[dimension]
        findings.append((LABELS[dimension] + " falls under both interventions",
            f"Baseline {number(entry['means']['baseline'])}, all-pair {number(entry['means']['all'])}, high-pair {number(entry['means']['high'])}. "
            f"All-pair is lower on {entry['paired']['all_minus_baseline']['lower_prompts']}/100 prompts and high-pair on {entry['paired']['high_minus_baseline']['lower_prompts']}/100 prompts. "
            "Each number averages 100 per-video scores, one fixed seed per prompt."))
    aesthetic = dims["aesthetic_quality"]["paired"]["high_minus_all"]
    imaging = dims["imaging_quality"]["paired"]["high_minus_all"]
    findings.append(("High-only replacement has different effects on aesthetics and imaging quality",
        "Relative to all-pair replacement, high-only changes aesthetic quality by " + interval(aesthetic) +
        " and imaging quality by " + interval(imaging) + ". These are paired differences over the same 100 prompts, with 95% bootstrap intervals."))
    high_mrl = []
    high_error = []
    for axis in ("t", "h", "w"):
        for side in ("q", "k"):
            for metric, destination in (("mrl", high_mrl), ("relative_squared_error", high_error)):
                destination.append(float(np.mean([v["mean"] for v in data["audit"]["baseline"]
                    if v["axis"] == axis and v["side"] == side and v["band"] == "high" and v["metric"] == metric])))
    findings.append(("High-frequency concentration still permits substantial replacement error",
        f"Across the six high-band axis/QK views, mean MRL spans {min(high_mrl):.3f}-{max(high_mrl):.3f}, "
        f"while mean pairwise squared-error fraction spans {min(high_error):.3f}-{max(high_error):.3f}. "
        "Each mean pools eight prespecified baseline prompts, steps 1/10/20, all 30 layers, 12 heads and that axis's 7-8 high-frequency pairs. Each pair statistic uses all 10,800 tokens."))
    findings.append(("Read consistency together with appearance and motion",
        "Dynamic degree is " + ", ".join(NAMES[a] + " " + number(dims["dynamic_degree"]["means"][a]) for a in ARMS) + ". "
        "Similar degraded frames can receive high subject/background consistency or motion-smoothness scores. "
        "Use those dimensions alongside aesthetic quality, imaging quality, motion incidence and the video examples."))
    methods = [
        "Wan2.1-T2V-1.3B-Diffusers. Generate 45 frames at 480 x 480, 16 fps, with 20 denoising steps, CFG 5 and an empty negative prompt. The video-token grid is 12 x 30 x 30, giving 10,800 tokens per self-attention forward.",
        "For each normalized pre-RoPE Q or K, form a separate fp32 coordinate-wise mean over all 10,800 current video tokens. Keep each prompt, denoising step, layer, head and conditional/unconditional CFG branch separate. Cast the mean to bf16 and broadcast it to the selected token coordinates. Q and K use their own centers.",
        "The all-pair arm replaces all 64 pairs, totaling 128 coordinates per head. The high-pair arm replaces temporal pairs 0-7, height pairs 22-28 and width pairs 43-49, totaling 22 pairs and 44 coordinates. These are the same highest-frequency thirds used in the earlier heatmaps. The other 42 pairs use the current trajectory's token vectors.",
        "Apply the intervention to both Q and K in all 30 layers, all 12 heads, all 20 denoising steps and both CFG branches. The native pipeline computes 3D RoPE, scaled full-head logits, softmax, value aggregation and the output projection. Each of the 10,800 query tokens attends to all 10,800 key tokens in each head, summing all 64 pair contributions. Centers adapt to each modified trajectory.",
        "Sample 100 distinct prompts before generation from the eight official VBench content categories. Use sampling seed 20261109, with 13 prompts each from animal, architecture, food and human, and 12 each from lifestyle, plant, scenery and vehicles. Seeds differ across prompts. Each prompt shares its seed and text embeddings across the three arms.",
        "For each metric, compute its official per-video score, then take an equal-weight mean across 100 prompts separately for each arm. Pair each intervention with the baseline for the same prompt and seed. Resample these 100 paired prompt units 10,000 times with seed 20261109 and use percentile 95% confidence intervals. Frames, tokens, layers, heads and denoising steps are kept within their prompt unit during this bootstrap.",
    ]
    limits = [
        "This study covers 100 category prompts, one seed per prompt, 45-frame 480 x 480 videos and the stated 20-step schedule. The six quality dimensions and optional text-video dimension are evaluated in custom-input mode. The official full VBench protocol uses its dimension-specific prompt lists and multiple videos per prompt.",
        "Both Q and K are replaced jointly. Centers are recomputed separately at every step and branch. The measured quality loss therefore occurs with current-step centers. Temporal center drift, Q-only replacement, K-only replacement and calibrated center reuse each require their own intervention.",
        "The arithmetic mean replacement removes variation in both direction and magnitude. MRL quantifies concentration of a token population around its mean direction. The separately reported pairwise squared-error fraction measures the representation change more directly.",
        "Eight category-diverse audit prompts and the middle frame index 22 were fixed before result inspection. Their PDF images illustrate individual frames. The HTML gallery exposes all 300 full videos. Per-prompt CSV scores and exact configuration accompany the report.",
        "The implementation runs the full native attention path. Reported timing includes generation and VAE decoding, with the observed worker and evaluation concurrency. These measurements describe this intervention implementation.",
    ]
    return findings, methods, limits


def write_markdown(root, data):
    findings, methods, limits = content(data)
    lines = ["# Wan Q/K center replacement", "", "9 November 2026 | 100 paired prompts | 300 videos", "", "## Findings", ""]
    for title, evidence in findings:
        lines += ["### " + title, "", evidence, ""]
    lines += ["## Intervention and aggregation", ""]
    for paragraph in methods:
        lines += [paragraph, ""]
    lines += ["For each selected pair f, use mu_Q[f] = mean_tokens Q[f] and mu_K[f] = mean_tokens K[f]. "
              "The pair means are two-coordinate vectors. Concatenating them gives the same mean as averaging the full head coordinate-wise. "
              "For full replacement, the ideal logit is sum_f mu_Q[f]^T R(theta_K[f] - theta_Q[f]) mu_K[f] / sqrt(128), "
              "where the angle difference uses that pair's t, h or w relative coordinate. Native bf16 rounding applies during inference.", "", "## VBench results", "",
              "All scores use the official dimension scale. Changes and confidence intervals are absolute score differences. Each arm has 100 matched videos.", "",
              "| Dimension | Baseline | All pairs | High pairs | All minus baseline, 95% CI | High minus baseline, 95% CI |",
              "|---|---:|---:|---:|---|---|"]
    for dimension in ORDER:
        if dimension in data["dimensions"]:
            d = data["dimensions"][dimension]
            lines.append(f"| {LABELS[dimension]} | {number(d['means']['baseline'])} | {number(d['means']['all'])} | {number(d['means']['high'])} | {interval(d['paired']['all_minus_baseline'])} | {interval(d['paired']['high_minus_baseline'])} |")
    lines += ["", "![Paired differences](figures/paired-deltas.png)", "", "### Exact evaluator populations", ""]
    for dimension in ORDER:
        if dimension in data["dimensions"]:
            lines += ["- **" + LABELS[dimension] + "**. " + SCOPES[dimension]]
    lines += ["", "## Pair-level concentration and replacement error", "", data["audit_scope"]["within_pair"], "", data["audit_scope"]["summary"], "",
              "The audit contains prompts " + ", ".join(data["audit_scope"]["prompt_ids"]) + ". Each table entry additionally averages the three audited steps with equal weight.", "",
              "| Axis/band | Q MRL | K MRL | Q relative squared error | K relative squared error |", "|---|---:|---:|---:|---:|"]
    for axis in ("t", "h", "w"):
        for band in ("high", "middle", "low"):
            values = [np.mean([v["mean"] for v in data["audit"]["baseline"] if v["axis"] == axis and v["band"] == band and v["side"] == side and v["metric"] == metric]) for metric in ("mrl", "relative_squared_error") for side in ("q", "k")]
            lines.append("| " + axis + " " + band + " | " + " | ".join(number(v) for v in values) + " |")
    lines += ["", "![Baseline pair diagnostics](figures/baseline-pair-diagnostics.png)", "", "## Content categories", "",
              "Imaging-quality means below use 12 or 13 prompts per category, all 45 frames per video, and one fixed seed per prompt.", "",
              "| Category | Prompts | Baseline | All pairs | High pairs |", "|---|---:|---:|---:|---:|"]
    for category, value in data["dimensions"]["imaging_quality"]["categories"].items():
        lines.append(f"| {category} | {value['prompts']} | " + " | ".join(number(value["means"][a]) for a in ARMS) + " |")
    lines += ["", "## Prespecified examples", "", "All panels show decoded frame 22 of 45, using the same prompt and seed in all three arms.", ""]
    for prompt in data["config"]["prompts"]:
        if prompt["audit"]:
            lines += [f"### {prompt['id']} / {prompt['category']}", "", prompt["text"], "", f"Seed {prompt['seed']}", "", f"![Paired example](figures/{prompt['id']}-comparison.png)", ""]
    lines += ["## Verification and scope", "", f"All {data['coverage']['videos']} videos passed SHA256 and 45-frame / 480 x 480 / 16-fps checks. "
              f"All {data['coverage']['qk_hook_calls']:,} Q/K hook calls passed exact step/layer/branch coverage. Every transformer input was matched to its supplied positive or negative text embedding.", "",
              "The pretrained layer-0 attention audit uses 36 synthetic tokens in two independent batches, 12 heads and 128 dimensions per head. "
              "Baseline, all-pair and high-pair outputs each match the independent native-RoPE/SDPA recomputation exactly. "
              "The separate fp32 3D trigonometric-logit identity check has maximum absolute error " + f"{data['native_validation']['three_axis_trig_max_abs_error']:.3g}.", ""]
    for paragraph in limits:
        lines += [paragraph, ""]
    lines += ["## Reproduction and sources", "", "The configuration contains every prompt, category, seed and intervention pair ID. "
              "The summary contains raw per-prompt metric values, bootstrap definitions, audit scope and video hashes. "
              "Generation code hashes are recorded in every output and checked for one consistent version.", "",
              "```sh", "# Run from the repository root on the CUDA host.", "# One generation worker per visible GPU. Completed arms are preserved.",
              "CUDA_VISIBLE_DEVICES=2 WAN_SNAPSHOT=/dev/shm/tri-wan-snapshot \\",
              "  /dev/shm/tri-center-runtime/bin/python scripts/run_center_benchmark.py \\",
              "  --config configs/center_benchmark_wan.json --out out/center-benchmark-wan-20261109", "```", "",
              "- TriAttention, section 3.3 gives the calibration-data Q/K means and section 4 develops key-importance scoring. https://arxiv.org/html/2604.04921v1", "",
              "- VBench source, commit " + data["config"]["vbench_commit"] + ". https://github.com/Vchitect/VBench", "",
              "- Official VBench custom-input documentation and full sampling protocol. https://github.com/Vchitect/VBench#new-evaluate-your-own-videos", ""]
    (root / "REPORT.md").write_text("\n".join(lines))


def write_html(root, data):
    findings, methods, limits = content(data)
    esc = html.escape
    dimensions = [d for d in ORDER if d in data["dimensions"]]
    rows = "".join("<tr><th>" + esc(LABELS[d]) + "</th>" + "".join("<td>" + number(data["dimensions"][d]["means"][a]) + "</td>" for a in ARMS) +
                   "<td>" + interval(data["dimensions"][d]["paired"]["all_minus_baseline"]) + "</td><td>" + interval(data["dimensions"][d]["paired"]["high_minus_baseline"]) + "</td></tr>" for d in dimensions)
    findings_html = "".join("<h3>" + esc(t) + "</h3><p>" + esc(e) + "</p>" for t, e in findings)
    method_html = "".join("<p>" + esc(p) + "</p>" for p in methods)
    scope_html = "".join("<h3>" + esc(LABELS[d]) + "</h3><p>" + esc(SCOPES[d]) + "</p>" for d in dimensions)
    limit_html = "".join("<p>" + esc(p) + "</p>" for p in limits)
    encoded = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan Q/K center replacement</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#1d2a33;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:44px 24px}h1{font-size:38px;line-height:1.15;margin-bottom:12px}h2{margin-top:48px;border-bottom:1px solid #ccd5db;padding-bottom:8px}h3{font-size:19px;margin-bottom:4px}p{max-width:1000px}.meta,.scope{color:#52616d;font-size:14px}.card{background:white;border:1px solid #d7dfe5;padding:24px;border-radius:10px;margin:22px 0}a{color:#285a80}table{border-collapse:collapse;font-size:14px;width:100%}td,th{text-align:left;padding:10px 8px;border-bottom:1px solid #d9dfe4;font-variant-numeric:tabular-nums}th{font-weight:600}.scroll{overflow:auto}img.chart{width:100%;max-width:980px;display:block;margin:20px auto}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%;background:#172129;border-radius:5px}.videos h3{margin:8px 0}select,button{padding:8px 12px;font:inherit;margin:0 8px 10px 0;max-width:100%}#prompt{width:760px}#prompt-text{font-weight:550}.method{max-width:1030px}footer{margin-top:40px;color:#52616d;font-size:13px}@media(max-width:760px){.videos{grid-template-columns:1fr}h1{font-size:30px}main{padding:24px 14px}.card{padding:14px}}
</style><main><h1>Wan Q/K center replacement</h1><p class="meta">9 November 2026 · 100 paired prompts · 300 videos · Current-step token means</p>
<p><a href="report.pdf">Detailed PDF</a> · <a href="REPORT.md">Markdown report</a> · <a href="per-prompt-results.csv">All per-prompt scores</a> · <a href="summary.json">Complete numerical summary</a> · <a href="config.json">Exact prompts and configuration</a></p>
<section class="card"><h2 style="margin-top:0">Findings</h2>FINDINGS</section>
<h2>Paired VBench results</h2><p class="scope">Each arm averages 100 per-video scores, with one matching prompt/seed per row of the underlying data. Intervals use 10,000 paired prompt bootstrap samples. Changes are absolute score differences.</p>
<div class="scroll"><table><thead><tr><th>Dimension</th><th>Baseline</th><th>All pairs</th><th>High pairs</th><th>All − baseline, 95% CI</th><th>High − baseline, 95% CI</th></tr></thead><tbody>ROWS</tbody></table></div>
<img class="chart" src="figures/paired-deltas.png" alt="Paired dimension differences and confidence intervals">
<h2 id="gallery">Explore all 300 videos</h2><p class="scope">Choose any of the 100 prompts. Each comparison uses the same seed and all 45 frames. The PDF examples use eight prespecified category-diverse prompts and frame index 22.</p>
<div class="card"><select id="category" aria-label="Category"><option value="">All categories</option></select><select id="prompt" aria-label="Prompt"></select><p id="prompt-text"></p><p id="seed" class="scope"></p><button id="play">Play all</button><button id="pause">Pause all</button><button id="restart">Restart all</button><div class="videos">
<div><h3>Baseline</h3><video id="baseline" controls muted loop playsinline preload="metadata"></video></div><div><h3>All 64 pairs</h3><video id="all" controls muted loop playsinline preload="metadata"></video></div><div><h3>High 22 pairs</h3><video id="high" controls muted loop playsinline preload="metadata"></video></div></div><div class="scroll"><table id="selected-scores"></table></div></div>
<h2>Intervention and exact averaging</h2><div class="method">METHODS</div>
<h2>Baseline concentration and replacement error</h2><p>AUDIT</p><img class="chart" src="figures/baseline-pair-diagnostics.png" alt="Baseline MRL and pairwise mean-replacement error at three denoising steps">
<h2>Imaging quality across content categories</h2><p class="scope">Twelve or thirteen prompts per category, one fixed seed each. MUSIQ averages all 45 decoded frames per video.</p><img class="chart" src="figures/imaging-by-category.png" alt="Imaging quality by content category">
<details><summary>Exact evaluator frame populations and aggregation</summary>SCOPES</details><h2>Verification and scope</h2><p>All 300 videos pass video SHA256, 45-frame, 480 × 480 and 16-fps checks. All 720,000 Q/K hook calls pass exact step/layer/branch coverage. Pretrained Wan layer-0 attention outputs match independent recomputation exactly for all three arms. <a href="native-pretrained-validation.json">Native validation data</a></p>AUDIT_APPENDIX LIMITS
<footer>TriAttention center definition and method · <a href="https://arxiv.org/html/2604.04921v1">Paper</a><br>Official evaluator source · <a href="https://github.com/Vchitect/VBench">VBench</a> · Pinned commit COMMIT</footer>
<script id="dataset" type="application/json">DATA</script><script>
const data=JSON.parse(document.getElementById('dataset').textContent),arms=['baseline','all','high'],labels=LABELS;
const category=document.getElementById('category'),prompt=document.getElementById('prompt'),videos=arms.map(a=>document.getElementById(a));
for(const c of [...new Set(data.config.prompts.map(p=>p.category))].sort()){const o=document.createElement('option');o.value=c;o.textContent=c;category.append(o)}
function populate(){prompt.replaceChildren();for(const p of data.config.prompts.filter(p=>!category.value||p.category===category.value)){const o=document.createElement('option');o.value=p.id;o.textContent=p.id+' · '+p.category+' · '+p.text;prompt.append(o)}update()}
function update(){const p=data.config.prompts.find(p=>p.id===prompt.value);document.getElementById('prompt-text').textContent=p.text;document.getElementById('seed').textContent='Seed '+p.seed+' · '+p.category+' · 20 denoising steps · 30 layers × 12 heads · both CFG branches';for(let i=0;i<3;i++){videos[i].pause();videos[i].src='videos/'+arms[i]+'/'+p.id+'.mp4';videos[i].load()}
const table=document.getElementById('selected-scores');table.replaceChildren();const heading=document.createElement('tr');for(const s of ['Per-video score','Baseline','All pairs','High pairs']){const th=document.createElement('th');th.textContent=s;heading.append(th)}table.append(heading);for(const d of ORDER){if(!data.dimensions[d])continue;const tr=document.createElement('tr');for(const s of [labels[d],...arms.map(a=>data.dimensions[d].per_prompt[p.id][a].toFixed(4))]){const td=document.createElement('td');td.textContent=s;tr.append(td)}table.append(tr)}}
category.onchange=populate;prompt.onchange=update;document.getElementById('play').onclick=()=>{const t=videos[0].currentTime;for(const v of videos){v.currentTime=t;v.play().catch(()=>{})}};document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());document.getElementById('restart').onclick=()=>{for(const v of videos){v.currentTime=0;v.play().catch(()=>{})}};populate();
</script></main></html>"""
    replacements = {"FINDINGS": findings_html, "ROWS": rows, "METHODS": method_html,
                    "AUDIT": esc(data["audit_scope"]["within_pair"] + " " + data["audit_scope"]["summary"]),
                    "SCOPES": scope_html, "LIMITS": limit_html, "COMMIT": data["config"]["vbench_commit"],
                    "DATA": encoded, "LABELS": json.dumps(LABELS), "ORDER": json.dumps(ORDER)}
    audit_note = root / "implementation-audit/README.md"
    replacements["AUDIT_APPENDIX"] = (
        '<h3>Independent implementation audit and official-code comparison</h3>'
        '<p>Actual 3D phase and Q/K rotation checks pass. A separately written processor reproduces the saved final latents exactly for all three arms of one full prompt trajectory. '
        'Official TriAttention uses Q centers to rank actual cached keys and retains original K/V representations. These completed trials directly replace selected Q/K coordinates. '
        '<a href="implementation-audit/README.md">Read the complete audit and pinned source comparison</a> · '
        '<a href="implementation-audit/v000-trajectory-replay.json">Full-trajectory checks</a> · '
        '<a href="implementation-audit/actual-first-step.json">Real-activation checks</a></p>'
    ) if audit_note.exists() else ""
    # Substitute only template markers, before user/source data is inserted.
    import re
    document = re.sub(r"\b(?:FINDINGS|ROWS|METHODS|AUDIT_APPENDIX|AUDIT|SCOPES|LIMITS|COMMIT|DATA|LABELS|ORDER)\b", lambda match: replacements[match.group()], document)
    (root / "index.html").write_text(document)


def write_pdf(root, data, output):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak
    from reportlab.lib.pagesizes import A4
    findings, methods, limits = content(data)
    width = 503
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=12.5, spaceAfter=7)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5)
    heading = ParagraphStyle("heading", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=12, spaceBefore=6)
    subheading = ParagraphStyle("subheading", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=7)
    title = ParagraphStyle("title", parent=heading, fontSize=25, leading=29, spaceBefore=10, spaceAfter=14)
    story = []
    def p(text, style=body):
        story.append(Paragraph(html.escape(text), style))
    def image(name, maximum_height=None):
        path = root / "figures" / name
        im = PILImage.open(path)
        w = width
        h = w * im.height / im.width
        if maximum_height and h > maximum_height:
            w *= maximum_height / h
            h = maximum_height
        story.append(Image(str(path), width=w, height=h))
    def table(rows, widths, font_size=8):
        style = ParagraphStyle("cell", parent=small, fontSize=font_size, leading=font_size + 2, spaceAfter=0)
        t = Table([[Paragraph(html.escape(str(v)), style) for v in row] for row in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#31566f")),
                               ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#aab8c0")),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5"))]))
        story.append(t)
        story.append(Spacer(1, 10))
    p("Wan Q/K center replacement", title)
    p("100 paired prompts | 300 videos | Wan2.1-T2V-1.3B | 9 November 2026", small)
    p("Current-step means across every layer and head", heading)
    for headline, evidence in findings:
        p(headline, subheading)
        p(evidence)
    p("Experiment at a glance", subheading)
    table([["Arm", "Q/K pairs replaced per head", "Center population"],
           ["Baseline", "0", "Native token Q/K"],
           ["All", "64 pairs / 128 coordinates", "All 10,800 tokens at this forward"],
           ["High", "22 pairs / 44 coordinates", "All 10,800 tokens at this forward"]], [70, 174, 259])
    p("Every center is separate for Q/K, prompt, denoising step, layer, head and CFG branch. Both Q and K are replaced jointly. All 30 layers, 12 heads, 20 steps and both branches receive the intervention.")
    story.append(PageBreak())
    p("1. Intervention and evaluation protocol", heading)
    for paragraph in methods:
        p(paragraph)
    p("Center and logit definitions", subheading)
    p("For each selected 2D pair f, mu_Q[f] = mean_tokens Q[f] and mu_K[f] = mean_tokens K[f]. Coordinate-wise averaging of the full 128D head gives the same centers as averaging its 64 pairs separately.")
    p("With all pairs replaced, the ideal logit is the sum of mu_Q[f]^T R(theta_K[f] - theta_Q[f]) mu_K[f] / sqrt(128). Each pair's angle difference uses its t, h or w relative position. Native bf16 rounding is applied during inference.")
    p("TriAttention derives calibration-data means for its analysis and key-importance method. This experiment recomputes means from the current video input at every forward, as selected for these first two trials.")
    story.append(PageBreak())
    p("2. Paired VBench results", heading)
    p("Scores are on the official dimension scale. Each mean uses 100 videos, one fixed seed per prompt, with all three arms paired. Delta intervals resample prompt units 10,000 times.")
    rows = [["Dimension", "Baseline", "All", "High", "All - baseline, 95% CI", "High - baseline, 95% CI"]]
    for dimension in ORDER:
        if dimension in data["dimensions"]:
            d = data["dimensions"][dimension]
            rows.append([LABELS[dimension], *[number(d["means"][a]) for a in ARMS], interval(d["paired"]["all_minus_baseline"]), interval(d["paired"]["high_minus_baseline"])])
    table(rows, [95, 48, 43, 43, 137, 137], 7.2)
    image("paired-deltas.png", 250)
    p("Consistency and smoothness can remain high for nearly uniform or consistently degraded frames. Dynamic degree records motion incidence. Jointly inspect appearance, motion, text alignment and the full videos.", small)
    image("imaging-by-category.png", 165)
    p("Category means use 12 or 13 prompts per category. Each MUSIQ video score averages its 45 decoded frames and is divided by 100.", small)
    story.append(PageBreak())
    p("3. Concentration and representation change", heading)
    p(data["audit_scope"]["within_pair"])
    p(data["audit_scope"]["summary"])
    p("The audit contains eight prespecified category-diverse prompts. The figure keeps steps 1, 10 and 20 separate. The table additionally averages these three steps with equal weight.")
    image("baseline-pair-diagnostics.png", 255)
    rows = [["Axis / band", "Q MRL", "K MRL", "Q error fraction", "K error fraction"]]
    for axis in ("t", "h", "w"):
        for band in ("high", "middle", "low"):
            values = [np.mean([v["mean"] for v in data["audit"]["baseline"] if v["axis"] == axis and v["band"] == band and v["side"] == side and v["metric"] == metric]) for metric in ("mrl", "relative_squared_error") for side in ("q", "k")]
            rows.append([axis + " / " + band, *[number(v) for v in values]])
    table(rows, [111, 88, 88, 108, 108], 8)
    p("Each error fraction belongs to one pair and head before averaging. It measures the energy removed by replacing that pair's token vectors with their fp32 center. The final inference center is rounded to bf16. High directional concentration alone leaves variation in direction and magnitude that the replacement removes.", small)
    audited = [p for p in data["config"]["prompts"] if p["audit"]]
    for start in range(0, 8, 2):
        story.append(PageBreak())
        p(f"4. Prespecified examples ({start + 1}-{start + 2} of 8)", heading)
        p("Decoded frame 22 of 45. Match prompt and seed across the three arms. Examples and frame index were fixed before reviewing results.", small)
        for prompt in audited[start:start + 2]:
            p(prompt["id"] + " / " + prompt["category"] + " / seed " + str(prompt["seed"]), subheading)
            p(prompt["text"], small)
            image(prompt["id"] + "-comparison.png", 205)
            d = data["dimensions"]["imaging_quality"]["per_prompt"][prompt["id"]]
            p("Per-video MUSIQ / 100, averaged over all 45 frames. Baseline " + number(d["baseline"]) + ", all " + number(d["all"]) + ", high " + number(d["high"]) + ".", small)
    story.append(PageBreak())
    p("5. Exact evaluator populations", heading)
    for dimension in ORDER:
        if dimension in data["dimensions"]:
            p(LABELS[dimension], subheading)
            p(SCOPES[dimension], small)
    p("All pairwise changes subtract each prompt's matched baseline before prompt-level averaging. The aggregate statistics pool prompts after the official per-video aggregation. No Q/K pair, head, layer or denoising-step average enters VBench scoring.")
    story.append(PageBreak())
    p("6. Verification, scope and reproduction", heading)
    p("All 300 videos passed independent video hashes and frame-count, size and fps checks. The 720,000 Q/K hook calls passed exact step/layer/branch coverage. Every transformer call verified its conditioning embedding. Every denoising step produced finite latents.")
    p("Pretrained Wan layer 0 was checked with 36 synthetic tokens, two batches, 12 heads and 128 dimensions per head. All three arms matched independent native-RoPE/SDPA recomputation exactly. The separate fp32 3D trigonometric identity check had maximum absolute error " + f"{data['native_validation']['three_axis_trig_max_abs_error']:.3g}.")
    for paragraph in limits:
        p(paragraph, small)
    p("Recorded inference plus decode timing", subheading)
    table([["Arm", "Mean seconds/video", "Median seconds/video"], *[[NAMES[a], f"{data['timing'][a]['mean_seconds']:.2f}", f"{data['timing'][a]['median_seconds']:.2f}"] for a in ARMS]], [180, 160, 163])
    p("Reproducibility files", subheading)
    p("config.json lists all 100 prompts and seeds. per-prompt-results.csv contains every metric and paired change. summary.json records aggregation scopes, confidence intervals and video hashes. The HTML report contains all 300 videos. Generation uses scripts/run_center_benchmark.py and scripts/center_replacement.py, with hashes recorded and verified for every output.", small)
    p("Sources", subheading)
    p("TriAttention, sections 3.3 and 4. https://arxiv.org/html/2604.04921v1", small)
    p("VBench official evaluator and custom-input protocol. https://github.com/Vchitect/VBench", small)
    p("Pinned VBench source commit " + data["config"]["vbench_commit"], small)
    output.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667782"))
        canvas.drawString(46, 24, "Wan Q/K center replacement | 9 November 2026")
        canvas.drawRightString(549, 24, str(doc.page))
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=46, rightMargin=46, topMargin=37, bottomMargin=39,
                      title="Wan Q/K center replacement", author="TriAttention video study").build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    data = json.loads((root / "summary.json").read_text())
    make_figures(root, data)
    write_markdown(root, data)
    write_html(root, data)
    write_pdf(root, data, Path(args.pdf))
    print("Report artifacts created", root, args.pdf)


if __name__ == "__main__":
    main()
