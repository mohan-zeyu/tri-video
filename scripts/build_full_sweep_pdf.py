"""Create a concise PDF from the verified full-sweep report bundle.

Requires reportlab and pillow; no GPU or torch. Copy the final coverage,
config, summary images and the three documented example images locally.
  python scripts/build_full_sweep_pdf.py --root output/full-sweep-wan-20261101
"""
import argparse
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="output/full-sweep-wan-20261101")
    ap.add_argument("--output",default="output/pdf/wan-full-sweep-report.pdf")
    args=ap.parse_args();root=ROOT/args.root;output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/"config.json").read_text());audit=json.loads((root/"coverage.json").read_text())
    selection=json.loads((root/"report-figures/selection.json").read_text())
    heatmaps=json.loads((root/"mrl-heatmaps/manifest.json").read_text())
    analysis=None
    if (root/"mrl-heatmaps/pattern-analysis.json").exists():
        from mrl_pattern_summary import load_summary
        analysis=load_summary(root)
    body=ParagraphStyle("body",fontName="Times-Roman",fontSize=10.4,leading=13.2,spaceAfter=7)
    heading=ParagraphStyle("heading",parent=body,fontName="Times-Bold",fontSize=12.5,leading=15,spaceBefore=8,spaceAfter=7)
    caption=ParagraphStyle("caption",parent=body,fontSize=8.6,leading=10.6,spaceAfter=7)
    title=ParagraphStyle("title",parent=body,fontName="Times-Bold",fontSize=21,leading=24,alignment=TA_CENTER,spaceAfter=8)
    sub=ParagraphStyle("sub",parent=body,fontSize=10,leading=12,alignment=TA_CENTER,textColor=colors.HexColor("#555555"),spaceAfter=13)
    story=[]
    def p(text,style=body):story.append(Paragraph(text,style))
    def image(file,width=492):
        path=root/file; im=PILImage.open(path)
        story.append(Image(str(path),width=width,height=width*im.height/im.width))
    def table(data,widths,fontsize=9.2,padding=6):
        wrapped=[[Paragraph(str(x),ParagraphStyle("cell",parent=body,fontSize=fontsize,leading=fontsize+2,spaceAfter=0)) for x in row] for row in data]
        t=Table(wrapped,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),padding),("BOTTOMPADDING",(0,0),(-1,-1),padding),
            ("LINEABOVE",(0,0),(-1,0),.7,colors.black),("LINEBELOW",(0,0),(-1,0),.4,colors.black),("LINEBELOW",(0,-1),(-1,-1),.7,colors.black)]))
        story.append(t);story.append(Spacer(1,8))

    p("Q/K concentration and logit reconstruction<br/>through a full video denoising trajectory",title)
    p("Wan2.1-T2V-1.3B · complete prompt / step / layer / head / axis sweep<br/>Captured 1 November 2026 · report updated 8 November 2026",sub)
    p("<b>Scope.</b> Four real text prompts, 20 denoising steps per prompt, all 30 self-attention layers, all 12 heads and all three positional axes. Every primary measurement retains all 10,800 video tokens. This evaluates the geometric observations behind TriAttention; it does not test KV pruning or compression performance.")
    p("1. Setup and prompts",heading)
    p("Generate latent trajectories for 45 frames at 480 × 480, using the model's real text encoder and scheduler. CFG scale is 5.0; seed 0 is reused across prompts. The patch grid is 12 temporal × 30 height × 30 width positions. A temporal token offset is distinct from a denoising-step offset.")
    table([["<b>Prompt</b>","<b>Text supplied to the real text encoder</b>"],*[[x["id"],x["text"]] for x in cfg["prompts"]]],[55,437],9.0)
    p("2. Exact source of each measurement",heading)
    p("<b>Capture:</b> normalized Q/K immediately before RoPE, from the prompt-conditioned CFG forward. The actual transformer text input is compared exactly with the supplied positive embedding on every call. Conditional and unconditional passes are recorded separately; only the conditional pass is analyzed.")
    p("<b>2D geometry:</b> reselect the strongest axis-specific frequency pair separately at every prompt, step, layer and head, using mean-token Q norm × mean-token K norm. Draw every token's Q and K in that pair, before/after RoPE, with identical pair, token IDs and scale. Pair concentration is the norm of the token-mean 2D vector divided by the mean token 2D norm. Axis distributions contain the 12 selected-pair values from this one layer, prompt and step.")
    p("<b>Full-head prediction:</b> average the entire 128D Q/K vectors across all 10,800 tokens, separately at each prompt/step/layer/head. Rotate these centers at the actual query/key positions and sum all 64 pair contributions. Measured logits use actual token vectors after native bf16 rotation, with fp32 dot products. These are pre-softmax logits.")
    story.append(PageBreak())

    p("3. Complete coverage and aggregate results",heading)
    c=audit["observed"]
    p(f"The archive contains <b>{c['layer_captures']:,} layer captures</b>, <b>{c['head_figures']:,} head figures</b>, {c['sheets']:,} all-head sheets and {c['distributions']:,} across-head distributions. Full bf16 Q/K are retained for every layer/step/prompt ({c['raw_bytes']/1e9:.1f} GB). Numeric records and figure counts were checked against the full Cartesian product.")
    means=[audit["summary"][a][side+"_dom_mrl_pre"]["mean"] for a in ("t","h","w") for side in ("q","k")]
    correlations=" / ".join(f"{audit['summary'][a]['per_query_r_mean']['mean']:.3f}" for a in ("t","h","w"))
    p(f"Across Q/K and axes, mean selected-pair R ranges from {min(means):.3f} to {max(means):.3f}. Mean full-head per-query correlation is {correlations} for t / h / w. These pool the prompt/step/layer/head cells defined in Table 1.")
    p("<b>Table 1.</b> Each axis pools 4 prompts × 20 steps × 30 layers × 12 heads = 28,800 cells. R averages selected-2D-pair concentration. The r column first correlates each query's actual/predicted full-head logits over its axis-line keys, averages valid queries within the cell, then averages cells. Undefined correlations are excluded, never set to zero.",caption)
    rows=[["<b>Axis</b>","<b>Q R pre</b>","<b>K R pre</b>","<b>Q R post</b>","<b>K R post</b>","<b>Mean query r</b>"]]
    for a in ("t","h","w"):
        z=audit["summary"][a]
        rows.append([a,*[f"{z[k]['mean']:.4f}" for k in ("q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","per_query_r_mean")]])
    table(rows,[42,86,86,86,86,106],9.0)
    image(Path("summary/concentration-trajectories.png"),width=470)
    p("Figure 1. Each line is one prompt. Each point averages selected-pair R over 30 layers and 12 heads at one step; Q/K and axes remain separate. Every pair statistic uses all 10,800 tokens. Pairs are reselected at each step; these are not fixed-band trajectories. No prompt/step pooling enters a point.",caption)
    image(Path("summary/reconstruction-trajectories.png"),width=470)
    p("Figure 2. Each colored line is one prompt. Each point averages the full-head per-query reconstruction score over layers and heads at this step and axis. This is not the correlation of two already-averaged logit curves.",caption)
    story.append(PageBreak())

    p("4. Complete A/B/C/D examples",heading)
    p("<b>Panel map, following TriAttention Figure 2:</b> A, pre-RoPE 2D pairs; B, post-RoPE pairs; C, concentration across heads; D, logit prediction. Each example combines all four panels at one prompt, denoising step, layer and axis.")
    p("<b>4.1 Fixed examples.</b> Retain p03, step 20, layer 15, head 0 for all three axes. A/B contain every token in one axis-dominant pair. C contains 12 Q and 12 K pre-RoPE pair-MRL values: each head selects its own axis-dominant pair using C_f, and each R uses all 10,800 tokens. One head is 8.33%. There is no layer/step/prompt pooling. Layer, head and pair indices are zero-based; steps are numbered 1-20.")
    p("D uses the full 128D head (all 64 pairs); its centers average all 10,800 tokens at this one prompt/step/layer/head. Curves average all same-axis-line query-key logits per signed offset; shading is measured population SD. Mean query r averages 10,800 individual query correlations, rather than correlating the two plotted mean curves.",caption)
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["fixed"]
        image(Path(z["report_figure"]))
        p(f"<b>{axis}-axis.</b> Pair {z['dominant_pair']}; pre-RoPE pair R_Q={z['q_dom_mrl_pre']:.6f}, R_K={z['k_dom_mrl_pre']:.6f}; full-head mean query r={z['per_query_r_mean']:.6f}.",caption)
    p("The fixed examples underestimate the zero-offset peak. Pearson r measures shape agreement, not matching logit magnitudes. Concentration in one selected pair and reconstruction using the full head are different measurements.",caption)
    story.append(PageBreak())

    p("4.2 Additional best reconstruction examples",heading)
    p("For best prediction performance, maximize <b>full-head mean per-query Pearson r</b> independently for each axis over all 4 prompts x 20 steps x 30 layers x 12 heads = <b>28,800 cells per axis</b>, requiring all 10,800 queries to have valid correlations. The displayed pair is the winning head's C_f-selected axis-dominant pair; r evaluates all 128 dimensions. These maxima were selected after the sweep and are not representative averages.")
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["best_reconstruction"]
        image(Path(z["report_figure"]))
        p(f"<b>{axis}:</b> pair {z['dominant_pair']}; R_Q={z['q_dom_mrl_pre']:.6f}, R_K={z['k_dom_mrl_pre']:.6f}; full-head mean query r=<b>{z['per_query_r_mean']:.6f}</b>. Same all-token definitions as 4.1.",caption)
    p("4.3 Best concentrated pair, ranked separately",heading)
    p("Maximize min(R_Q,R_K) over the 28,800 <b>axis-dominant selected pairs</b> per axis, not every non-dominant band. Each R is pre-RoPE MRL over all 10,800 tokens at one condition. These winners differ from the best full-head predictions above.",caption)
    rows=[["<b>Axis</b>","<b>Prompt / step / layer / head / pair</b>","<b>Pair R_Q / R_K</b>"]]
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["best_pair"]
        rows.append([axis,f"{z['prompt_id']} / {z['step']} / {z['layer']} / {z['head']} / {z['dominant_pair']}",f"{z['q_dom_mrl_pre']:.6f} / {z['k_dom_mrl_pre']:.6f}"])
    table(rows,[35,282,175],8.6,padding=3)
    p("The linked gallery includes all nine A/B/C/D figures: three fixed examples, three best full-head reconstructions and three best concentrated pairs. Selection rules, exact values and source hashes are retained in report-figures/selection.json.",caption)
    story.append(PageBreak())

    for axis_index, axis in enumerate(("t","h","w"),start=1):
        p(f"5.{axis_index} Layer-by-head MRL across {axis}-axis frequencies",heading)
        if axis_index==1:
            p("Each heatmap has layer on the y-axis and head on the x-axis, separately for Q and K. Each underlying R_f is the norm of the token-mean 2D pair divided by the mean token-pair norm, using all 10,800 tokens at one prompt/step/layer/head. A band cell is the arithmetic mean of these individual pair R_f values over the listed pairs.")
            p("High, middle and low bands are nearly equal-count thirds of actual angular frequency, sorted from fast to slow rotation independently within t/h/w. The saved RoPE angles determine the ordering. The temporal axis has 8/7/7 pairs, while height and width each have 7/7/7. Pair index increases toward lower frequency within each axis.",caption)
        else:
            p("Each cell represents one layer/head at the stated prompt and denoising step. It averages the all-token 2D-pair MRL values over the explicitly listed frequency band. Q and K remain separate. The 0-1 color scale is shared across axes, frequency bands, prompts, steps and pre/post-RoPE views.")
        image(Path(f"mrl-heatmaps/figures/p03-step-020-{axis}-pre.png"),width=470)
        p(f"<b>Fixed {axis}-axis example.</b> Prompt p03, denoising step 20, pre-RoPE. All 30 layers and 12 heads appear individually. Each pair statistic uses all 10,800 tokens at that one condition. The panels average only the listed frequency pairs.",caption)
        p("<b>Pooled descriptive means.</b> Each table entry averages 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 band-mean cells. Q/K, frequency group and pre/post-RoPE stage are separate.",caption)
        rows=[["<b>Band</b>","<b>Pair IDs</b>","<b>Q pre</b>","<b>K pre</b>","<b>Q post</b>","<b>K post</b>"]]
        for band,definition in heatmaps["groups"][axis]["bands"].items():
            pairs=definition["pairs"]
            values=[heatmaps["summary"][axis][stage][side][band]["mean"] for stage in ("pre","post") for side in ("q","k")]
            rows.append([band,f"{min(pairs)}-{max(pairs)}",*[f"{value:.4f}" for value in values]])
        table(rows,[65,95,83,83,83,83],8.8,padding=4)
        p('The website exposes all 80 prompt/step conditions, pre/post stages and each individual pair. <link href="wan-mrl-heatmaps-complete.pdf" color="#4c72b0">The complete heatmap appendix</link> contains 160 landscape A3 pages, one per prompt/step/stage, each covering all three axes, both Q/K and all frequency groups.',caption)
        story.append(PageBreak())

    if analysis is not None:
        analysis_body=ParagraphStyle("analysis_body",parent=body,fontSize=9.3,leading=11.2,spaceAfter=3)
        analysis_scope=ParagraphStyle("analysis_scope",parent=caption,fontSize=8.1,leading=9.5,spaceAfter=5,textColor=colors.HexColor("#4d5661"))
        analysis_heading=ParagraphStyle("analysis_heading",parent=heading,fontSize=10.2,leading=12,spaceBefore=4,spaceAfter=3)
        p(analysis["title"],heading)
        p(analysis["definition"],analysis_body)
        pair_table=analysis["pair_table"]
        table([[f"<b>{cell}</b>" for cell in pair_table["headers"]],*pair_table["rows"]],[45,149,149,149],8.1,padding=2)
        for finding in analysis["findings"]:
            p(finding["title"],analysis_heading)
            p(finding["text"],analysis_body)
            if "table" in finding:
                content=finding["table"]
                table([[f"<b>{cell}</b>" for cell in content["headers"]],*content["rows"]],[52,110,110,110,110],8.1,padding=2)
            p(finding["scope"],analysis_scope)
        p("Computed profiles and averaging definitions are stored in mrl-heatmaps/pattern-analysis.json and linked from the website's Heatmap analysis section.",analysis_scope)
        story.append(PageBreak())

    p("6. Validation, interpretation and reproducibility",heading)
    p("Each query is evaluated against every key on its axis line, fixing the other two coordinates. Each head/step/layer/prompt evaluates 10,800 × 12 temporal logits and 10,800 × 30 logits for each spatial axis. Both signed directions and self are included; no query/key subsampling is used.")
    p(f"<b>Attention-output checks.</b> {audit['attention_output_validation_checks']} recomputations cover every layer at the first step of each prompt, plus layers 0/15/29 at steps 10 and 20. {audit['bit_exact_checks']} checks are bit-exact; the maximum relative error is {audit['max_attention_relative_error']:.6g}. Both CFG inputs are verified against their supplied embeddings on every step. Equal first-step/first-layer Q centers across all prompts confirm the expected shared-noise behavior before text cross-attention.")
    p("<b>Independent numerical checks.</b> Small explicit attention matrices verify the axis-line extraction, all signed-offset pair counts, the origin-query curve and the mean of per-query Pearson correlations. Constant predicted rows are treated as undefined. All-token Q/K are saved so selection, concentration and logits can be reanalyzed without repeating generation.")
    p("<b>What the results test.</b> Within each head, prompt and denoising step, the experiment asks whether normalized Q/K concentrate in axis-specific dominant pairs and whether replacing full token vectors by their within-input centers preserves axis-dependent logit structure. High concentration in a selected pair need not imply uniformly strong concentration across all bands, or accurate logits for every query.")
    p("<b>Limits.</b> Centers are estimated from the same input being reconstructed. This tests within-input approximation, not transfer from an independent calibration set. Three exhaustive axis-line tests do not cover arbitrary mixed-axis offsets. Four prompts with one shared seed do not establish broad prompt/seed robustness. The experiment returns latents; it does not compare decoded-video quality, prune KV pairs, or claim a memory/throughput improvement.")
    p('The definitions of dominant-band contribution, pair MRL and center-based logit reconstruction follow <link href="https://arxiv.org/abs/2604.04921" color="#4c72b0">Mao et al., TriAttention, section 3 and Appendix B</link>. The paper evaluates 1D text distances 1, 2, 4, ... with calibration centers. Here we use three latent-coordinate axes, same-input centers, both signed directions and self, evaluating every same-axis-line query/key. These correlations are not a matched LLM benchmark.')
    p("<b>Correction to the previous report.</b> The older Wan pipeline probe retained the final, unconditional CFG forward while describing it as conditional. This new verified conditional sweep supersedes that trajectory interpretation. The separate earlier synthetic-input geometry captures are unaffected.")
    p("Artifact map",heading)
    table([["<b>Artifact</b>","<b>Contents and provenance</b>"],
        ["index.html","Select prompt, step, layer, axis and head; view individual figures, sheets and distributions."],
        ["report-figures/","Nine A/B/C/D examples; selection.json records the ranking rules, values and source hashes."],
        ["mrl-heatmaps/","Every prompt/step's 30 x 12 x 64 pair-MRL values, exact frequency groups and source hashes. Complete heatmap appendix PDF is linked alongside the report."],
        ["per-head-results.csv","One row per prompt/step/layer/axis/head; selected pair, pair R, full-head reconstruction statistics and counts."],
        ["coverage.json","Verified coverage, validation checks and hashes of every numeric layer record."],
        ["pXX/step-XXX/","Full bf16 Q/K, pair metrics, exact scheduler timestep, input latents, conditional/unconditional call checks."],
        ["metadata.json / rope.pt","Literal prompts, seed, model snapshot, package/GPU/code metadata; exact rotary angles, grid and pair-axis map."]],[125,367],9.0)
    p("Result root relative to the data-host checkout is out/full-sweep-wan-20261101. Reproduce using configs/full_sweep_wan.json and scripts/launch_full_sweep.py; inspect FULL_SWEEP_PLAN.md for exact procedures. Summary aggregation is explicit; the primary figures retain individual prompt and denoising-step identity.",caption)

    def footer(canvas,doc):
        canvas.saveState();canvas.setFillColor(colors.black)
        canvas.setStrokeColor(colors.HexColor("#333333"));canvas.setLineWidth(.4);canvas.line(51,804,544,804)
        canvas.setFont("Times-Roman",8);canvas.drawString(51,811,"Complete Wan Q/K geometry and 3D RoPE reconstruction")
        canvas.drawRightString(544,811,"Research report · revised 2026-11-08");canvas.drawCentredString(297.5,28,str(doc.page))
        canvas.restoreState()
    document=SimpleDocTemplate(str(output),pagesize=(595.28,841.89),leftMargin=51,rightMargin=51,topMargin=48,bottomMargin=43,
        title="Complete Wan 3D RoPE experiment",author="TriAttention video reproduction")
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)


if __name__=="__main__":main()
