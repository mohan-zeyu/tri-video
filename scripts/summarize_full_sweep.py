"""Verify complete coverage, summarize explicit pooling, and build an atlas.

Call after both capture workers AND all CPU rendering workers finish.
Reads numeric layer records, retains one row per prompt/step/layer/axis/head.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/matplotlib-full-sweep"))


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--root",default="out/full-sweep-wan-20261101")
    args=ap.parse_args(); root=ROOT/args.root
    cfg=json.loads((root/"config.json").read_text())
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[]; validations=[]; files=[]; numeric_hash=hashlib.sha256()
    expected_code={name:hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() for name in ("full_sweep.py","full_sweep_metrics.py")}
    expected={"prompts":len(cfg["prompts"]),"steps_per_prompt":cfg["steps"],"layers":30,"heads":12,"axes":3,"tokens":10800}
    counts={"layer_captures":0,"head_figures":0,"sheets":0,"distributions":0,"conditional_forwards":0,"unconditional_forwards":0,"raw_bytes":0}
    first_step_layer0=[]
    for prompt in cfg["prompts"]:
        pid=prompt["id"]; completion=json.loads((root/pid/"complete.json").read_text())
        assert completion["steps_completed"]==cfg["steps"] and completion["layer_captures"]==cfg["steps"]*30
        for step in range(1,cfg["steps"]+1):
            stepdir=root/pid/f"step-{step:03d}"
            s=json.loads((stepdir/"step.json").read_text())
            assert [x["branch"] for x in s["forwards"]]==["conditional","unconditional"]
            assert all(x["embedding_verified"] for x in s["forwards"])
            counts["conditional_forwards"]+=1; counts["unconditional_forwards"]+=1
            for layer in range(30):
                path=stepdir/f"layer-{layer:03d}.json"; source=path.read_bytes(); numeric_hash.update(source)
                d=json.loads(source); assert d["branch"]=="conditional" and (d["tokens"],d["heads"],d["head_dim"])==(10800,12,128)
                assert (d["step"],d["layer"],d["prompt_id"])==(step,layer,pid)
                assert d["prompt"]==prompt["text"] and tuple(d["grid"])==(12,30,30)
                assert d["code_sha256"]==expected_code
                raw=path.with_suffix(".pt"); assert raw.stat().st_size>2*12*10800*128*2
                counts["raw_bytes"]+=raw.stat().st_size
                rendered=json.loads(path.with_suffix(".rendered.json").read_text())
                assert rendered["metrics_sha256"]==hashlib.sha256(source).hexdigest()
                assert rendered["tokens_subsampled"] is False
                for k in ("sheets","head_figures","distributions"):counts[k]+=rendered["counts"][k]
                counts["layer_captures"]+=1
                if d["validation"] is not None:validations.append(d["validation"])
                if step==1 and layer==0:first_step_layer0.append(np.array(d["center_q"]))
                files.append({"path":str(path.relative_to(root)),"sha256":hashlib.sha256(source).hexdigest(),"raw_bytes":raw.stat().st_size})
                for axis,item in d["axes"].items():
                    n=d["grid"][("t","h","w").index(axis)]
                    assert sum(item["pairs_per_offset"])==d["tokens"]*n
                    assert item["total_queries_per_head"]==d["tokens"]
                    folder=path.with_suffix("")/axis
                    # Directory listings verify every expected output name
                    # without 100,800 serial NFS stat round-trips.
                    names={p.name for p in folder.iterdir()}
                    assert {"all-heads.png","concentration-distribution.png","heads"}<=names
                    head_names={p.name for p in (folder/"heads").iterdir() if p.suffix==".png"}
                    assert head_names=={f"head-{h:02d}.png" for h in range(12)}
                    for head in range(12):
                        row={"prompt_id":pid,"step":step,"scheduler_timestep":s["scheduler_timestep"],"layer":layer,"axis":axis,"head":head,
                             "dominant_pair":item["dominant_pair"][head],"tokens":d["tokens"],"whole_head_dimensions":128,"axis_queries":d["tokens"],
                             "axis_keys_per_query":n,"axis_total_logits":d["tokens"]*n}
                        for k in ("q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","q_all_axis_pairs_mrl_mean","k_all_axis_pairs_mrl_mean","per_query_r_mean","per_query_r_median","per_query_r_p10","per_query_r_p90","curve_r","valid_queries"):
                            row[k]=item[k][head]
                        rows.append(row)
    assert counts["layer_captures"]==2400 and counts["head_figures"]==86400 and counts["sheets"]==7200 and counts["distributions"]==7200,counts
    assert len(validations)==len(cfg["prompts"])*36,len(validations)
    assert max(x["relative_max_err"] for x in validations)<=1e-3
    from verify_full_sweep_samples import verify
    spot_checks=verify(root)
    # First self-attention in the first denoising step precedes text cross
    # attention, so equal initial noise/seed must produce identical Q centers.
    for center in first_step_layer0[1:]:np.testing.assert_array_equal(center,first_step_layer0[0])
    with (root/"per-head-results.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for axis in ("t","h","w"):
        selected=[r for r in rows if r["axis"]==axis]
        summary[axis]={"cells":len(selected)}
        for key in ("q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","per_query_r_mean","curve_r"):
            values=np.array([r[key] for r in selected if r[key] is not None])
            summary[axis][key]={"mean":float(values.mean()),"median":float(np.median(values)),"valid_cells":len(values)}
        for side in ("q","k"):
            summary[axis][side+"_dominant_pre_fraction_gt_095"]=float(np.mean([r[side+"_dom_mrl_pre"]>.95 for r in selected]))
    coverage={"expected":expected,"observed":counts,"rows":len(rows),"attention_output_validation_checks":len(validations),
              "max_attention_relative_error":max(x["relative_max_err"] for x in validations),"bit_exact_checks":sum(x["max_abs_err"]==0 for x in validations),
              "first_step_first_layer_centers_identical_across_prompts":True,"numeric_records_combined_sha256":numeric_hash.hexdigest(),
              "independent_real_tensor_spot_checks":spot_checks,
              "summary_pooling":"Each axis: pool 4 prompts x 20 steps x 30 layers x 12 heads. Each r cell first averages per-query correlations across all 10,800 token queries. No undefined correlation is replaced by zero.",
              "capture_code_sha256":expected_code,"config_sha256":hashlib.sha256((root/"config.json").read_bytes()).hexdigest(),"summary":summary,"files":files}
    (root/"coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11.5,"axes.facecolor":"#eaeaf2","axes.grid":True,"grid.color":"white","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"figure.facecolor":"white"})
    figures=root/"summary";figures.mkdir(exist_ok=True)
    colors=["#4c72b0","#dd8452","#55a868","#c44e52"]
    fig,axs=plt.subplots(2,3,figsize=(11,5.6),sharex=True,sharey=True)
    for col,axis in enumerate(("t","h","w")):
        for ridx,side in enumerate(("q","k")):
            ax=axs[ridx,col]
            for color,prompt in zip(colors,cfg["prompts"]):
                means=[np.mean([r[side+"_dom_mrl_pre"] for r in rows if r["prompt_id"]==prompt["id"] and r["step"]==step and r["axis"]==axis]) for step in range(1,21)]
                ax.plot(range(1,21),means,color=color,label=prompt["id"],lw=1.5)
            ax.set(title=f"{side.upper()} · {axis}-axis dominant pair",ylim=(0,1),xlim=(1,20))
            if ridx==1:ax.set_xlabel("Denoising step (1–20)")
            if col==0:ax.set_ylabel("Mean selected-pair MRL")
    axs[0,0].legend(ncol=2,fontsize=9.5)
    fig.suptitle("Within each prompt and step: mean over 30 layers × 12 heads; pair MRL uses all 10,800 tokens",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(figures/"concentration-trajectories.png",dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,2.8),sharex=True,sharey=True)
    for ax,axis in zip(axs,("t","h","w")):
        for color,prompt in zip(colors,cfg["prompts"]):
            means=[]
            for step in range(1,21):
                vals=[r["per_query_r_mean"] for r in rows if r["prompt_id"]==prompt["id"] and r["step"]==step and r["axis"]==axis and r["per_query_r_mean"] is not None]
                means.append(np.mean(vals))
            ax.plot(range(1,21),means,color=color,label=prompt["id"],lw=1.5)
        ax.set(title=f"{axis}-axis full-head reconstruction",xlabel="Denoising step (1–20)",ylim=(-1,1),xlim=(1,20))
    axs[0].set_ylabel("Mean per-query Pearson r");axs[0].legend(fontsize=9.5,ncol=2)
    fig.suptitle("Within each prompt/step: mean over layers/heads; each cell evaluates every query and same-axis-line key",fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(figures/"reconstruction-trajectories.png",dpi=190);plt.close(fig)
    write_index(root,cfg)
    examples=root/"examples";examples.mkdir(exist_ok=True)
    fixed=root/"p03/step-020/layer-015"
    shutil.copy2(fixed.with_suffix(".json"),examples/"layer-015.json")
    for a in ("t","h","w"):shutil.copy2(fixed/a/"heads/head-00.png",examples/f"{a}-head-00.png")
    table=["| Axis | Q pair R, pre | K pair R, pre | Q pair R, post | K pair R, post | Mean query r |", "|---|---:|---:|---:|---:|---:|"]
    for axis,z in summary.items():
        table.append("| "+axis+" | "+" | ".join(f"{z[k]['mean']:.4f}" for k in ("q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","per_query_r_mean"))+" |")
    means=[summary[a][side+"_dom_mrl_pre"]["mean"] for a in ("t","h","w") for side in ("q","k")]
    findings=f"Across the six Q/K-by-axis summaries, mean selected-2D-pair concentration ranges from {min(means):.3f} to {max(means):.3f}. Mean full-128D-head per-query reconstruction correlation is "+" / ".join(f"{summary[a]['per_query_r_mean']['mean']:.3f}" for a in ("t","h","w"))+" for t / h / w. Each axis summary pools 4 prompts × 20 steps × 30 layers × 12 heads; every underlying pair concentration or query evaluation uses all 10,800 video tokens at its individual prompt/step/layer/head."
    report=["# Complete Wan 3D RoPE experiment", "Four real prompts · 20 denoising steps per prompt · every layer/head/axis · all tokens", "## Scope and provenance",
        "Wan2.1-T2V-1.3B, 45 frames at 480×480, CFG 5.0 and seed 0. Four prompts share the same initial noise, isolating prompt changes at this seed. Each step has a 12×30×30 grid of 10,800 video tokens. Video time and diffusion timestep are distinct coordinates. Every conditional transformer input was matched exactly to its supplied positive text embedding; unconditional CFG forwards were excluded from captures.",
        "## Measurement definitions", "Each primary figure belongs to one prompt, one denoising step, one layer, one head and one axis. No step/prompt pooling occurs there. The 2D scatter uses every token in the strongest frequency pair belonging to that axis, selected by mean_token(||q_f||) × mean_token(||k_f||). Pre/post plots use the identical pair and tokens. Pair R = ||mean_token(x_f)|| / mean_token(||x_f||). Distributions contain the 12 heads of this one layer/step/prompt/axis, separately for Q/K and pre/post.",
        "Logit reconstruction uses the entire 128D head, summing all 64 pairs. Measured logits are recomputed with fp32 dot products from the captured Q/K after native bf16 RoPE rounding. The prediction rotates fp32 centers without bf16 rounding. Those Q/K centers are fp32 averages over all 10,800 video tokens at this same prompt/step/layer/head. Every token acts as a query; keys range over its whole axis line, fixing the other two coordinates. This evaluates 129,600 temporal and 324,000 height or width logits per head/layer/step/prompt. Both signed offset directions and self are included. These are pre-softmax logits, not attention probabilities.",
        "Primary reconstruction curves average the measured and predicted logits over all query–key pairs at each signed offset. Shading is the population standard deviation of individual measured logits at that offset. The head reconstruction score first correlates each individual query's measured and predicted logit vectors over its same-axis keys, then averages the valid query correlations. This differs from correlating the two already-averaged curves; both statistics and valid-query counts are retained. The single origin-query curves are also retained separately.",
        "## Coverage and validation",f"Completed: {counts['layer_captures']:,} layer captures, {counts['sheets']:,} all-head sheets, {counts['head_figures']:,} individual head figures and {counts['distributions']:,} concentration distributions. Each individual head figure has pre/post scatter plus logit reconstruction. Full bf16 Q/K are retained for every layer/step/prompt ({counts['raw_bytes']/1e9:.1f} GB). All outputs are indexed by prompt, step, layer, axis and head.",
        f"Attention-output recomputation checks: {len(validations)}, covering all 30 layers at the first step of each prompt and layers 0/15/29 at steps 10/20. {coverage['bit_exact_checks']} checks are bit-exact; maximum relative error is {coverage['max_attention_relative_error']:.6g}. First-step, first-layer Q centers are identical across all prompts, as expected before cross-attention when initial noise and timestep are identical. Numerical tests independently compare axis logit matrices, signed-offset counts and per-query correlations against explicit calculations.",
        "## Results",findings, "The following table pools **4 prompts × 20 steps × 30 layers × 12 heads = 28,800 cells per axis**. R columns average selected-2D-pair MRL values. The r column averages each cell's mean over its valid token-query correlations, using full-head logits. It does not use the correlation of the plotted mean curves. These are descriptive pooled summaries; primary per-step/per-head data remain separate.","\n".join(table),
        "![Concentration trajectories](summary/concentration-trajectories.png)", "Each colored line is one prompt. Each point averages selected-pair R over 30 layers and 12 heads at one denoising step; Q and K and the three axes are separated. Every underlying pair R uses all 10,800 tokens. The dominant pair is reselected separately at each prompt/step/layer/head/axis; these curves do not track one fixed band through the trajectory.",
        "![Reconstruction trajectories](summary/reconstruction-trajectories.png)", "Each colored line is one prompt. Each point averages the full-head per-query reconstruction score over 30 layers and 12 heads, at one step and one axis. Its underlying query correlations use every token as a query and all same-axis-line keys.",
        "## Fixed examples across all three axes", "These examples use prompt p03, step 20, layer 15, head 0, fixed before inspection. Each scatter uses every token in the axis-selected 2D pair. Logits use the entire 128D head and Q/K centers averaged over all 10,800 tokens at that same condition. Curves average all same-axis-line query–key logits at each signed offset; shading is their measured population standard deviation.",
        "![Temporal example](examples/t-head-00.png)","![Height example](examples/h-head-00.png)","![Width example](examples/w-head-00.png)",
        "In these examples, center predictions underestimate the zero-offset peak. Pearson correlation measures shape agreement, not matching logit magnitudes. A weakly concentrated selected 2D pair can coexist with more correlated full-head reconstruction.",
        "## Interpretation limits", "These predictions use centers estimated from the same input being reconstructed. They establish within-input approximation quality, not cross-prompt calibration transfer. A selected strong band does not imply that every band or the whole head is equally concentrated. Axis-line evaluation covers all three directions but not arbitrary mixed-axis query–key offsets. Four prompts and one shared seed do not establish robustness across seeds or a broad prompt population. No KV pruning, generated-video quality comparison, memory saving or inference speedup is claimed.",
        "## Navigation and reproducibility", "Open `index.html` to select prompt, denoising step, layer, axis and individual head, or view all heads together and the corresponding concentration distribution. `per-head-results.csv` has one row per prompt/step/layer/axis/head. `coverage.json` records coverage, validation and hashes of every numeric layer record. Full Q/K tensors and full pair-level metrics are stored under `pXX/step-XXX/layer-XXX.{pt,json}`. `rope.pt` stores the exact model rotary angles and grid. Per-prompt metadata includes the literal prompt, seed, model snapshot, environment, GPU and code hashes. Step metadata records both the sequential denoising index and actual scheduler timestep.",
        "Figure subjects and pair-ranking/MRL definitions follow [TriAttention §3 and Appendix B](https://arxiv.org/abs/2604.04921). The original LLM experiment uses positive, logarithmically spaced 1D text distances and calibration centers. Our video experiment uses three coordinate axes, same-input centers, both signed directions and self, evaluating every same-axis-line query/key. Its correlations are not a matched LLM benchmark.",
        "The old Wan `pipeline_probe.py` trajectory was incorrectly labelled conditional: its hooks retained the final, unconditional CFG forward. This verified conditional sweep supersedes that interpretation. The earlier synthetic-input plots are separate measurements and are not relabelled by this correction."]
    (root/"REPORT.md").write_text("\n\n".join(report)+"\n")
    if (root/"report-figures/selection.json").exists():
        from make_paper_figures import update_report
        update_report(root,json.loads((root/"report-figures/selection.json").read_text()))
    if (root/"mrl-heatmaps/manifest.json").exists():
        from build_mrl_heatmaps import update_report as update_mrl_report
        update_mrl_report(root/"REPORT.md",json.loads((root/"mrl-heatmaps/manifest.json").read_text()))
    if (root/"mrl-heatmaps/pattern-analysis.json").exists():
        from mrl_pattern_summary import load_summary,update_markdown
        update_markdown(root/"REPORT.md",load_summary(root))
    (root/"complete.json").write_text(json.dumps({"completed_unix":time.time(),"coverage":counts,"rows":len(rows),"numeric_records_combined_sha256":numeric_hash.hexdigest()},indent=2)+"\n")
    print(json.dumps({"coverage":counts,"validation":len(validations),"summary":summary},indent=2))


def write_index(root,cfg):
    # Self-contained local artifact: all paths are relative, no network calls.
    options="".join(f'<option value="{p["id"]}">{p["id"]}: {html.escape(p["text"])}</option>' for p in cfg["prompts"])
    page=(ROOT/"scripts/full_sweep_atlas.html").read_text()
    analysis=""
    if (root/"mrl-heatmaps/pattern-analysis.json").exists():
        from mrl_pattern_summary import html_section,load_summary
        analysis=html_section(load_summary(root))
    page=page.replace("<!-- MRL_ANALYSIS -->",analysis)
    (root/"index.html").write_text(page.replace("OPTIONS",options))
    shutil.copy2(ROOT/"scripts/mrl-heatmaps.js",root/"mrl-heatmaps.js")


if __name__=="__main__":main()
