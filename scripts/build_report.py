"""Redraw the TriAttention-style report from recorded measurements.

Requires numpy, matplotlib, reportlab and pillow. No model or GPU required.
  python scripts/build_report.py
Inputs are original out/*.json plus exports from export_report_data.py.
Outputs: REPORT.md, out/report/*.png|pdf|svg, a numeric audit JSON, and
output/pdf/triattention-video-report.pdf. The original report is archived.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / "tmp/cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIG = OUT / "report"
PDF = ROOT / "output/pdf"
BLUE, ORANGE, RED = "#4c72b0", "#dd8452", "#c44e52"
NAMES = {"wan": "Wan2.1-1.3B", "ltx": "LTX-Video-2B", "cog": "CogVideoX-2b"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.facecolor": "#eaeaf2",
    "figure.facecolor": "white", "axes.edgecolor": "white", "grid.color": "white",
    "axes.grid": True, "grid.linewidth": .8, "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False, "legend.frameon": False, "savefig.dpi": 220,
    "pdf.fonttype": 42, "svg.fonttype": "none"})


def read(path):
    return json.loads(path.read_text())


def flat(rows, key):
    return np.array([v for row in rows.values() for v in row[key]])


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def overview(model, bands, capture):
    fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.0), gridspec_kw={"wspace": .42})
    fig.subplots_adjust(left=.055, right=.985, bottom=.21, top=.79)
    fig.suptitle(f"{NAMES[model]}", fontsize=13, fontweight="bold", y=.99)
    norm = max(np.linalg.norm(np.array(capture[k]), axis=-1).max() for k in ("q", "k"))
    for ax, post, letter in ((axs[0], False, "A"), (axs[1], True, "B")):
        for side, col in (("q", BLUE), ("k", ORANGE)):
            points = np.array(capture[side + ("_post" if post else "")]) / norm
            ax.scatter(points[:, 0], points[:, 1], s=4, alpha=.28, color=col, label=side.upper(), rasterized=True, linewidths=0)
        ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xlabel="Re", ylabel="Im", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({letter}) {'Post' if post else 'Pre'}-RoPE Q/K", pad=9)
        if not post:
            ax.legend(loc="lower left", fontsize=8, handletextpad=.2)
            rs = [np.linalg.norm(np.mean(capture[k], axis=0)) / np.mean(np.linalg.norm(capture[k], axis=-1)) for k in ("q", "k")]
            ax.text(.97, .97, f"$R_Q$={rs[0]:.2f}\n$R_K$={rs[1]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax = axs[2]
    dq = np.array([[c["q"] for c in row] for row in bands["dom_pair_mrl"]]).ravel()
    dk = np.array([[c["k"] for c in row] for row in bands["dom_pair_mrl"]]).ravel()
    bins = np.linspace(0, 1, 21)
    ax.hist([dq, dk], bins=bins, weights=[np.ones(len(dq))*100/len(dq), np.ones(len(dk))*100/len(dk)], color=[BLUE, ORANGE], label=["Q", "K"], rwidth=.88)
    ax.set(xlim=(0, 1.02), ylim=(0, 100), xlabel="Concentration $R_f$", ylabel="Heads (%)", xticks=[0, .5, 1])
    ax.set_title("(C) Dominant-band MRL", pad=9)
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[3]
    series = capture["axis_slice"]
    actual, pred = np.array(series["actual"]), np.array(series["predicted"])
    x = np.array(series["offset"])
    ax.fill_between(x, actual.mean(0)-actual.std(0), actual.mean(0)+actual.std(0), color=ORANGE, alpha=.18, linewidth=0)
    ax.plot(x, actual.mean(0), color=ORANGE, ls="--", lw=1.7, label="Measured")
    ax.plot(x, pred.mean(0), color=BLUE, ls=":", lw=1.9, label="Center prediction")
    ax.set(xlabel=f"{dict(t='Temporal', h='Height', w='Width')[series['axis']]} offset $\\Delta_{series['axis']}$", ylabel="Attention logit", xlim=(0, x[-1]))
    ax.set_title("(D) Logit reconstruction", pad=9)
    ax.legend(fontsize=7.3, loc="best")
    ax.text(.96, .97, f"$\\bar r$={series['mean_r']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    save(fig, f"figure-{'1' if model == 'wan' else '2'}-{model}-geometry")


def recon_hist(series):
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.35), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=.22, left=.08, right=.98, bottom=.24, top=.82)
    for ax, model in zip(axs, ("wan", "ltx")):
        r = np.array([h["recon_r"] for h in series[model]["heads"]])
        ax.hist(r, bins=np.linspace(-1, 1, 31), color=BLUE, alpha=.85, weights=np.ones(len(r))*100/len(r))
        ax.axvline(r.mean(), color=RED, ls="--", lw=1.4)
        ax.set_title(f"{NAMES[model]}  ·  {len(r)} heads", fontsize=11)
        ax.set(xlim=(-1, 1), xlabel="Reconstruction correlation $\\bar r$", xticks=[-1, -.5, 0, .5, 1])
        ax.text(.04, .91, f"Mean {r.mean():.3f}\nMedian {np.median(r):.3f}", transform=ax.transAxes, va="top", fontsize=9)
    axs[0].set_ylabel("Heads (%)")
    save(fig, "figure-3-reconstruction")


def trajectories(pipes):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 2.85), sharey=True)
    fig.subplots_adjust(wspace=.2, left=.08, right=.98, bottom=.30, top=.84)
    for ax, model in zip(axs, ("wan", "cog")):
        p = pipes[model]
        x = sorted(map(int, p["by_step"]))
        for side, col in (("q", BLUE), ("k", ORANGE)):
            dom = [flat(p["by_step"][str(s)], side+"_dom_mrl").mean() for s in x]
            whole = [flat(p["by_step"][str(s)], side+"_mrl").mean() for s in x]
            ax.plot(x, dom, color=col, lw=1.8, label=f"{side.upper()} dominant")
            ax.plot(x, whole, color=col, lw=1.5, ls="--", label=f"{side.upper()} whole")
        ax.set(title=NAMES[model]+(" (no RoPE)" if model == "cog" else ""), xlabel="Denoising step", ylim=(.65, 1.005), xlim=(0, x[-1]))
    axs[0].set_ylabel("Mean concentration $R$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(.53, -.035))
    save(fig, "figure-4-real-trajectories")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    band_export = read(FIG / "band-data.json")
    bands = band_export["pairs"]
    captures = {m: read(FIG/f"{m}-visual-capture.json") for m in ("wan", "ltx")}
    raw = {m: read(OUT/f"{m}.json") for m in NAMES}
    series = {m: read(OUT/f"{m}-series.json") for m in ("wan", "ltx")}
    pipes = {m: read(OUT/f"{m}-pipeline.json") for m in ("wan", "cog")}
    stats = {}
    for model, b in bands.items():
        q, k = [np.array(b[side+"_pair_mrl"]) for side in ("q", "k")]
        dq, dk = [np.array([[c[side] for c in row] for row in b["dom_pair_mrl"]]) for side in ("q", "k")]
        ref = next(r for r in raw[model]["runs"] if r["tag"] == "t_mid-zeros-s0")
        stats[model] = {"q_all": float(q.mean()), "k_all": float(k.mean()), "q_dom": float(dq.mean()), "k_dom": float(dk.mean()),
                        "q_gt95": float((dq>.95).mean()*100), "k_gt95": float((dk>.95).mean()*100),
                        "q_whole": float(flat(ref["by_layer"], "q_mrl").mean()), "k_whole": float(flat(ref["by_layer"], "k_mrl").mean())}
    for model in captures:
        overview(model, bands[model], captures[model])
    recon_hist(series)
    trajectories(pipes)
    manifest = {"statistics": stats, "pair_sources_sha256": band_export["source_sha256"],
                "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(OUT/f"{m}.json" for m in NAMES), *(OUT/f"{m}-series.json" for m in series), *(OUT/f"{m}-pipeline.json" for m in pipes), *(FIG/f"{m}-visual-capture.json" for m in captures)]},
                "figure_protocol": {m: {k: captures[m][k] for k in ("layer", "head", "band", "tokens", "input", "selection", "torch", "diffusers", "snapshot", "dtype")} for m in captures}}
    manifest["reconstruction_statistics"] = {m: {"mean": float(np.mean([h["recon_r"] for h in s["heads"]])), "median": float(np.median([h["recon_r"] for h in s["heads"]])), "heads": len(s["heads"])} for m, s in series.items()}
    manifest["median_convention"] = "Midpoint of the two central observations for even counts; original series summaries used torch's lower median."
    (FIG/"report-audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_report(stats, captures, series, pipes)
    print("Wrote REPORT.md, four figures (PNG/PDF/SVG), audit JSON and three-page PDF.")


def write_report(stats, captures, series, pipes):
    paragraphs = {
        "abstract": "Pre-RoPE Q/K concentration transfers to the tested video diffusion transformers. Dominant-band query MRL is 0.972 in Wan and 0.922 in LTX; center-only logit reconstruction reaches mean correlations of 0.752 and 0.724. Wan also retains strong concentration during a real-prompt denoising trajectory. These measurements support the geometric premise of TriAttention; compression quality and efficiency remain untested.",
        "setup": "We study Wan2.1-T2V-1.3B (30 layers, 12 heads, 128 dimensions/head) and LTX-Video-2B (28 layers, 32 heads, 64 dimensions/head), both with 3D RoPE. CogVideoX-2b (30 layers, 30 heads, 64 dimensions/head) uses no RoPE in this checkpoint and serves as a comparison. Transformer-only probes use bf16 Gaussian latents, zero text features, and a mid diffusion timestep: 10,800 / 13,728 / 21,600 video tokens, respectively. Hooks capture normalized Q/K immediately before rotation in the RoPE models, or the video portion of the joint stream in Cog. Stored attention-output recomputations report zero error at three checked layers per model; pipeline checks cover the first step only.",
        "metric": "For each 2D pair f, R_f = ||E[x_f]|| / E[||x_f||], computed across tokens. We select one dominant pair per head by C_f = E[||q_f||] E[||k_f||] (paper Appendix B.7). Table 1 reports both this selected pair and all-pair means, so selection does not hide weaker bands. Pair estimates use stored 512-token bf16 subsamples; whole-vector statistics use all tokens. Cog's pairs are coordinate pairs, not rotary frequency bands.",
        "concentration": "Both Q and K concentrate strongly in dominant bands, while all-pair averages are lower. This is a band-selective property, not uniform concentration in every 2D subspace. RoPE reduces whole-vector query MRL from 0.857 to 0.631 in Wan and from 0.810 to 0.204 in LTX. Cog also concentrates without RoPE, so these observations do not show that RoPE causes concentration.",
        "reconstruction": "Replacing each token's pre-RoPE Q/K by its head center yields a trigonometric prediction over the spatiotemporal offset. With complex pair centers and model-provided angles, the prediction is:",
        "validation": "The stored evaluation uses 96 sampled queries against all keys in four layers: Wan [0, 5, 15, 29], LTX [0, 5, 14, 27]. Correlations are computed across keys for each query, then averaged within each head. Mean/median correlations are 0.752/0.830 for Wan and 0.724/0.815 for LTX. Centers and evaluation logits come from the same forward pass; this tests within-input reconstruction, not transfer from an independent calibration set. The paper uses logarithmically spaced 1D distances, so these values are not a matched LLM benchmark.",
        "real": "Existing full-pipeline runs encode a real prompt and follow the scheduler for 20 Wan steps or 30 Cog steps, measuring the positive conditioning branch and returning latents. Wan's dominant-band Q/K MRL averages 0.954/0.940 across steps; query MRL rises from 0.938 at step 0 to 0.968 at step 19. Cog's query value remains near 0.83, below its noise-probe value of 0.921; its key value declines from 0.914 to 0.871. The plots show head means, not uncertainty over prompts.",
        "stability": "Concentration within a sequence does not establish a fixed center across contexts. Across two iid noise seeds, mean whole-vector center cosine is about 0.999. Changing zero text features to random features moves centers substantially: Q/K cosine is 0.407/0.469 in Wan and 0.445/0.514 in LTX. Random features are an out-of-distribution perturbation, not a real-prompt benchmark. Each pipeline run covers only one prompt and one seed; LTX has no real-prompt trajectory here. Wan and Cog also differ in architecture, text conditioning, and prompt, so their contrast cannot isolate a causal effect of RoPE.",
        "scope": "The evidence supports pre-RoPE concentration and useful center-based logit reconstruction in the tested video models. It does not establish fixed centers across diverse prompts, top-key retention, video quality after pruning, or a memory/throughput gain. A compression evaluation needs independent multi-prompt calibration, held-out query/key ranking, and quality-versus-budget measurements. Two existing noise-probe rerun issues (Wan frequency-shape assertion and Cog latent layout) remain documented in README; this report does not alter those experiment scripts.",
    }
    paragraphs["abstract"] = paragraphs["abstract"].replace("0.922 in LTX", f"{stats['ltx']['q_dom']:.3f} in LTX")
    paragraphs["validation"] = paragraphs["validation"].replace("0.752/0.830", "0.752/0.848").replace("0.724/0.815", "0.724/0.816")
    captions = {
        "wan": f"Figure 1. Wan Q/K geometry, following the subjects of TriAttention Figure 2. (A-B) Fresh, position-matched samples at layer 0, head 0, band {captures['wan']['band']}; the same 512 tokens and common norm scale are used before/after rotation. One zero-text input, not multiple contexts. (C) Dominant-band MRL across all 360 heads from stored subsamples. (D) Mean measured and center-predicted logits along the {captures['wan']['axis_slice']['axis']} axis, holding other coordinates fixed; 16 deterministic query lines, including self. Shading is +/-1 standard deviation of measured logits across lines; r is the mean per-line Pearson correlation.",
        "ltx": f"Figure 2. LTX uses the same A-D layout. Geometry uses layer 0, head 0, band {captures['ltx']['band']}; the histogram covers all 896 heads. The reconstruction example varies the {captures['ltx']['axis_slice']['axis']} coordinate only. Layer/head selection is fixed before inspection. These local slices illustrate the mechanism and are separate from the all-key evaluation below.",
        "recon": "Figure 3. Distribution of per-head reconstruction correlation, matching the subject of TriAttention Figure 3. Four sampled layers contribute 48 Wan heads and 128 LTX heads. Red dashed lines mark means. These are within-input, all-key measurements, not the paper's distance-balanced protocol.",
        "trajectory": "Figure 4. Real-prompt concentration over denoising. Solid: dominant pair, selected at each head and step; dashed: whole vector. Blue: Q; orange: K. Each point averages all layer/head cells. The two models use different prompts and trajectories, so this is descriptive evidence, not a controlled RoPE ablation.",
    }
    captions["ltx"] += " This head is a counterexample: its K pair is dispersed and its temporal-slice correlation is -0.04. Strong average concentration does not imply accurate reconstruction in every head."
    table_rows = []
    for m in NAMES:
        s = stats[m]
        table_rows.append([NAMES[m], f"{s['q_all']:.3f} / {s['k_all']:.3f}", f"{s['q_dom']:.3f} / {s['k_dom']:.3f}", f"{s['q_gt95']:.1f}% / {s['k_gt95']:.1f}%"])
    table_md = "| Model | All-pair R (Q / K) | Dominant R (Q / K) | Dominant R > 0.95 (Q / K) |\n|---|---|---|---|\n" + "\n".join("| "+" | ".join(r)+" |" for r in table_rows)
    md = ["# Pre-RoPE concentration in video diffusion transformers", "*A focused reproduction of TriAttention's geometric observations · 1 November 2026*", "**Abstract.** "+paragraphs['abstract'], "## 1. Setup and measurement", paragraphs['setup'], paragraphs['metric'], "## 2. Q/K concentration", "![Wan geometry](out/report/figure-1-wan-geometry.png)", captions['wan'], "**Table 1.** Noise-probe concentration; means over layer × head (and pair for all-pair columns).", table_md, paragraphs['concentration'], "![LTX geometry](out/report/figure-2-ltx-geometry.png)", captions['ltx'], "## 3. From centers to attention logits", paragraphs['reconstruction'], "$$\\hat{s}(i,j)=\\frac{1}{\\sqrt d}\\sum_f\\Re\\!\\left[\\overline{\\mu_{q,f}}\\,e^{i(\\theta_{j,f}-\\theta_{i,f})}\\mu_{k,f}\\right].$$", "Here the overline denotes complex conjugation. The model's angles encode the 3D token positions; a flattened token distance is not sufficient.", paragraphs['validation'], "![Reconstruction distributions](out/report/figure-3-reconstruction.png)", captions['recon'], "## 4. Real-prompt validation and limits", paragraphs['real'], "![Real-prompt trajectories](out/report/figure-4-real-trajectories.png)", captions['trajectory'], paragraphs['stability'], "**Conclusion.** "+paragraphs['scope'], "## Sources and reproduction", "Figure subjects and plotting style follow [Mao et al., TriAttention, §3, Figures 2–3 and Appendix B.7](https://arxiv.org/abs/2604.04921). All plotted points and curves are our video-model measurements; no paper results are reused as video results.", "Original data: `out/{wan,ltx,cog}.json`, `*-pair-mrl.pt`, `{wan,ltx}-series.json`, `{wan,cog}-pipeline.json`. Fresh geometry captures and source hashes: `out/report/*-visual-capture.json`, `band-data.json`, `report-audit.json`. The prior report is preserved at `out/report/previous-report.md`.", "Regenerate the report and vector figures with `python scripts/build_report.py` (numpy, matplotlib, reportlab, pillow). Export/capture on the data host with `HF_HOME=/mnt/data/liuzeyu/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /dev/shm/trivenv/bin/python scripts/export_report_data.py --capture wan ltx`. Existing data need not be rerun to rebuild the report."]
    path = ROOT/"REPORT.md"
    if not (FIG/"previous-report.md").exists() and path.exists():
        (FIG/"previous-report.md").write_text(path.read_text())
    path.write_text("\n\n".join(md)+"\n")

    # Fixed three-page layout, with real text and vector-type typography.
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.3, leading=13.2, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=body, fontName="Times-Bold", fontSize=12.3, leading=15.0, spaceBefore=9, spaceAfter=6)
    caption = ParagraphStyle("caption", parent=body, fontSize=8.5, leading=10.7, spaceAfter=8)
    title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=20, leading=23, alignment=TA_CENTER, spaceAfter=6)
    subtitle = ParagraphStyle("subtitle", parent=body, fontSize=9.5, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=12)
    story = []
    def p(text, style=body):
        story.append(Paragraph(text.replace("×", "&#215;").replace("+/-", "&#177;"), style))
    def picture(name, width=492):
        from PIL import Image as PILImage
        f = FIG/(name+".png")
        im = PILImage.open(f)
        story.append(Image(str(f), width=width, height=width*im.height/im.width))
    p("Pre-RoPE concentration in<br/>video diffusion transformers", title)
    p("A focused reproduction of TriAttention's geometric observations · 1 November 2026", subtitle)
    p("<b>Abstract.</b> "+paragraphs['abstract'])
    p("1. Setup and measurement", heading)
    p(paragraphs['setup'])
    p(paragraphs['metric'].replace("R_f = ||E[x_f]|| / E[||x_f||]", "R<sub>f</sub> = ||E[x<sub>f</sub>]|| / E[||x<sub>f</sub>||]").replace("C_f = E[||q_f||] E[||k_f||]", "C<sub>f</sub> = E[||q<sub>f</sub>||] E[||k<sub>f</sub>||]"))
    p("2. Q/K concentration", heading)
    picture("figure-1-wan-geometry")
    p(captions['wan'], caption)
    p("<b>Table 1.</b> Noise-probe means. Each cell gives Q / K.", caption)
    tab = Table([["Model", "All pairs", "Dominant pair", "Dominant > 0.95"], *table_rows], colWidths=[132, 108, 116, 136], rowHeights=[21, 20, 20, 20])
    tab.setStyle(TableStyle([("FONTNAME", (0,0),(-1,0),"Times-Bold"), ("FONTNAME",(0,1),(-1,-1),"Times-Roman"), ("FONTSIZE",(0,0),(-1,-1),9), ("ALIGN",(1,0),(-1,-1),"CENTER"), ("LINEABOVE",(0,0),(-1,0),.7,colors.black), ("LINEBELOW",(0,0),(-1,0),.4,colors.black), ("LINEBELOW",(0,-1),(-1,-1),.7,colors.black), ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(tab)
    story.append(PageBreak())
    p("2. Q/K concentration (continued)", heading)
    p(paragraphs['concentration'])
    picture("figure-2-ltx-geometry")
    p(captions['ltx'], caption)
    p("3. From centers to attention logits", heading)
    p(paragraphs['reconstruction'])
    eq = plt.figure(figsize=(8, .6))
    eq.text(.5, .5, r"$\hat{s}(i,j)=\frac{1}{\sqrt{d}}\sum_f\mathrm{Re}\!\left[\overline{\mu_{q,f}}\,e^{i(\theta_{j,f}-\theta_{i,f})}\mu_{k,f}\right]$", ha="center", va="center", fontsize=17)
    save(eq, "center-logit-equation")
    picture("center-logit-equation", width=260)
    story.append(Spacer(1, 6))
    p("Here mu denotes a complex 2D pair center and the overline denotes conjugation. The model's angles encode 3D token positions; a flattened token distance is not sufficient.")
    p(paragraphs['validation'])
    picture("figure-3-reconstruction", width=440)
    p(captions['recon'], caption)
    story.append(PageBreak())
    p("4. Real-prompt validation and limits", heading)
    p(paragraphs['real'])
    picture("figure-4-real-trajectories")
    p(captions['trajectory'], caption)
    p("Context stability", heading)
    p(paragraphs['stability'])
    p("Conclusion", heading)
    p(paragraphs['scope'])
    p("Sources and reproduction", heading)
    p('Visual subjects and style: <link href="https://arxiv.org/abs/2604.04921" color="#4c72b0">Mao et al., TriAttention, section 3, Figures 2-3, Appendix B.7</link>. All plots use our video-model measurements.', caption)
    p('Data: out/{wan,ltx,cog}.json; *-pair-mrl.pt; {wan,ltx}-series.json; {wan,cog}-pipeline.json. Fresh samples and source hashes: out/report/*-visual-capture.json and report-audit.json. Regenerate with python scripts/build_report.py. Full protocol and capture command are in REPORT.md; previous-report.md preserves the earlier report.', caption)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#333333"))
        canvas.setLineWidth(.4)
        canvas.line(51, 804, 544, 804)
        canvas.setFont("Times-Roman", 8)
        canvas.drawString(51, 811, "Pre-RoPE Q/K geometry in video diffusion transformers")
        canvas.drawRightString(544, 811, "Research report · 2026-11-01")
        canvas.drawCentredString(297.5, 29, str(doc.page))
    doc = SimpleDocTemplate(str(PDF/"triattention-video-report.pdf"), pagesize=(595.28, 841.89), leftMargin=51, rightMargin=51, topMargin=48, bottomMargin=42, title="Pre-RoPE concentration in video diffusion transformers", author="TriAttention video reproduction")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
