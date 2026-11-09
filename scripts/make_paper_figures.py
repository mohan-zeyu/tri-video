"""Compose paper-style A/B/C/D examples from the existing full sweep.

No model forwards or token subsampling. C pools the 12 heads at one
prompt/step/layer/axis; D uses full-head (128D) logits, not one 2D pair.
Best pair concentration and best full-head reconstruction are ranked
separately, over all 28,800 saved cells per axis, with deterministic ties.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/matplotlib-full-sweep"))


def select_examples(root):
    source=root/"per-head-results.csv"
    with source.open() as f: rows=list(csv.DictReader(f))
    rows.sort(key=lambda r:(r["prompt_id"],int(r["step"]),int(r["layer"]),int(r["head"])))
    result={"source_csv_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "definitions":{
                "fixed":"p03, step 20, layer 15, head 0, retained unchanged",
                "best_pair":"largest min(R_Q, R_K), where each R is selected-2D-pair pre-RoPE MRL over all 10,800 tokens",
                "best_reconstruction":"largest mean per-query Pearson r of full 128D-head logits; all 10,800 queries must be valid",
                "scope":"For each axis independently: all 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 cells",
                "ties":"first in prompt ID, step, layer, head ascending order",
                "selection_bias":"Best cases are selected after inspecting all saved statistics; they are illustrative maxima, not representative performance"},
            "examples":{}}
    for axis in ("t","h","w"):
        candidates=[r for r in rows if r["axis"]==axis]
        assert len(candidates)==28800
        fixed=next(r for r in candidates if r["prompt_id"]=="p03" and r["step"]=="20" and r["layer"]=="15" and r["head"]=="0")
        pair=max(candidates,key=lambda r:min(float(r["q_dom_mrl_pre"]),float(r["k_dom_mrl_pre"])))
        valid=[r for r in candidates if r["per_query_r_mean"] and int(r["valid_queries"])==10800]
        recon=max(valid,key=lambda r:float(r["per_query_r_mean"]))
        result["examples"][axis]={}
        for name,row in (("fixed",fixed),("best_pair",pair),("best_reconstruction",recon)):
            item={k:row[k] for k in ("prompt_id","step","layer","head","axis","dominant_pair","q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","per_query_r_mean","curve_r","valid_queries","scheduler_timestep")}
            for k in ("step","layer","head","dominant_pair","valid_queries"):item[k]=int(item[k])
            for k in ("q_dom_mrl_pre","k_dom_mrl_pre","q_dom_mrl_post","k_dom_mrl_post","per_query_r_mean","curve_r","scheduler_timestep"):
                item[k]=float(item[k]) if item[k] else None
            item["joint_pair_mrl"]=min(item["q_dom_mrl_pre"],item["k_dom_mrl_pre"])
            item["source_layer_json"]=f"{item['prompt_id']}/step-{item['step']:03d}/layer-{item['layer']:03d}.json"
            item["source_layer_sha256"]=hashlib.sha256((root/item["source_layer_json"]).read_bytes()).hexdigest()
            result["examples"][axis][name]=item
    return result


def draw(root,selection,kind,axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    torch.set_num_threads(2)
    item=selection["examples"][axis][kind]
    path=root/item["source_layer_json"]
    d=json.loads(path.read_text());a=d["axes"][axis];head=item["head"];band=item["dominant_pair"]
    raw=torch.load(path.with_suffix(".pt"),map_location="cpu",weights_only=True)
    rope=torch.load(root/"rope.pt",map_location="cpu",weights_only=True)
    pair={side:raw[side][head,:,2*band:2*band+2].float() for side in ("q","k")}
    assert all(v.shape==(10800,2) for v in pair.values())
    assert a["dominant_pair"][head]==band
    c,s=rope["cos"][:,band],rope["sin"][:,band]
    scale=max(float(x.norm(dim=-1).max()) for x in pair.values())
    blue,orange="#4c72b0","#dd8452"
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":13,"axes.titlesize":13,"axes.labelsize":12,
        "xtick.labelsize":11,"ytick.labelsize":11,"axes.facecolor":"#eaeaf2","figure.facecolor":"white",
        "axes.edgecolor":"white","grid.color":"white","axes.grid":True,"legend.frameon":False,
        "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,"axes.spines.bottom":False,
        "pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,4,figsize=(14,4.15),gridspec_kw={"wspace":.42})
    fig.subplots_adjust(left=.055,right=.985,bottom=.32,top=.79)
    title={"fixed":"Fixed example","best_pair":"Best selected-pair concentration","best_reconstruction":"Best full-head reconstruction"}[kind]
    fig.suptitle(f"{title} | {item['prompt_id']} · step {item['step']:02d} · layer {item['layer']:02d} · head {head:02d} · {axis} axis · pair {band}",fontsize=13,y=.99)
    for ax,post,label in ((axs[0],False,"A"),(axs[1],True,"B")):
        for side,color in (("q",blue),("k",orange)):
            points=pair[side]
            if post:
                x,y=points.unbind(-1)
                points=torch.stack([x*c-y*s,x*s+y*c],-1).bfloat16().float()
            points=points.numpy()/scale
            ax.scatter(points[:,0],points[:,1],s=2,alpha=.20,c=color,linewidths=0,rasterized=True,label=side.upper())
        ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),xticks=[-1,0,1],yticks=[-1,0,1],xlabel="Re / shared scale",ylabel="Im / shared scale")
        ax.set_aspect("equal",adjustable="box")
        stage="post" if post else "pre"
        ax.set_title(f"({label}) {stage.capitalize()}-RoPE Q/K")
        ax.text(.97,.97,f"$R_Q$={a['q_dom_mrl_'+stage][head]:.5f}\n$R_K$={a['k_dom_mrl_'+stage][head]:.5f}",transform=ax.transAxes,ha="right",va="top",fontsize=10)
        if not post:
            legend=ax.legend(fontsize=10,loc="lower left",handletextpad=.2,markerscale=4)
            for handle in legend.legend_handles:handle.set_alpha(1)
    ax=axs[2]
    q=np.array(a["q_dom_mrl_pre"]);k=np.array(a["k_dom_mrl_pre"])
    assert len(q)==len(k)==12
    bins=np.linspace(0,1,21)
    assert np.histogram(q,bins)[0].sum()==np.histogram(k,bins)[0].sum()==12
    ax.hist([q,k],bins=bins,weights=[np.ones(12)*100/12,np.ones(12)*100/12],color=[blue,orange],label=["Q","K"],rwidth=.88)
    ax.set(title="(C) Concentration across heads",xlabel="Pre-RoPE pair MRL $R_f$",ylabel="Heads (%)",xlim=(0,1.02),ylim=(0,100),xticks=[0,.5,1])
    ax.legend(fontsize=10,loc="upper left")
    ax=axs[3];x=np.array(a["offsets"]);m=np.array(a["measured_mean"][head]);sd=np.array(a["measured_std"][head]);p=np.array(a["predicted_mean"][head])
    ax.fill_between(x,m-sd,m+sd,color=orange,alpha=.20,linewidth=0)
    ax.plot(x,m,color=orange,ls="--",lw=1.8,label="Measured mean")
    ax.plot(x,p,color=blue,ls=":",lw=2.0,label="Center prediction")
    ax.set(title="(D) Full-head logit prediction",xlabel=f"Signed {axis} offset (query − key)",ylabel="Pre-softmax logit",xlim=(x[0],x[-1]))
    rr=item["per_query_r_mean"]
    ax.text(.97,.97,f"Mean query r={rr:.6f}" if rr is not None else "Mean query r undefined",transform=ax.transAxes,ha="right",va="top",fontsize=10)
    ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.35),ncol=2,columnspacing=.6,borderaxespad=0)
    notes=[fig.text(.5,.065,"A/B: every token in one selected 2D pair. C: each of 12 heads contributes its own axis-dominant-pair MRL, over all 10,800 tokens.",ha="center",fontsize=9),
        fig.text(.5,.005,"One prompt/step/layer. D: full 128D head; centers average all tokens; curves average same-offset query–key logits; shading is population SD.",ha="center",fontsize=9)]
    dest=root/"report-figures";dest.mkdir(exist_ok=True)
    stem=f"{kind}-{axis}-abcd"
    for ext in ("png","pdf","svg"):fig.savefig(dest/f"{stem}.{ext}",dpi=220,bbox_inches="tight",pad_inches=.08)
    # Report captions already supply the same provenance; omit duplicate footnotes.
    for note in notes:note.set_visible(False)
    fig.savefig(dest/f"{stem}-report.png",dpi=220,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)
    item["figure"]=f"report-figures/{stem}.png"
    item["report_figure"]=f"report-figures/{stem}-report.png"


def gallery(root,selection):
    blocks=[]
    for kind,title in [("fixed","Fixed examples"),("best_pair","Best pair concentration"),("best_reconstruction","Best full-head reconstruction")]:
        blocks.append(f"<h2>{title}</h2>")
        for axis in ("t","h","w"):
            z=selection["examples"][axis][kind]
            blocks.append(f"<h3>{axis}: {z['prompt_id']}, step {z['step']}, layer {z['layer']}, head {z['head']}, pair {z['dominant_pair']}</h3><img src='{Path(z['figure']).name}'><p>Pair R_Q={z['q_dom_mrl_pre']:.6f}, R_K={z['k_dom_mrl_pre']:.6f}; full-head mean query r={z['per_query_r_mean']:.6f}. <a href='../{z['source_layer_json']}'>Source record</a></p>")
    text="""<!doctype html><meta charset='utf-8'><title>Paper-style ABCD examples</title><style>body{font:16px Georgia,serif;max-width:1600px;margin:25px auto;padding:0 24px;line-height:1.5}img{max-width:100%}a{color:#4c72b0}</style><h1>Paper-style A/B/C/D examples</h1><p><a href='../index.html'>Full sweep atlas</a> · <a href='selection.json'>Exact selection rules and source hashes</a></p><p>A/B use every token in one selected 2D pair. C is a distribution across 12 heads at one prompt/step/layer/axis, each head using its own dominant pair. D uses the full 128D head and same-condition all-token centers. No averaging across prompts or denoising steps enters these figures.</p><p>Best-pair score: min(R_Q,R_K). Best-reconstruction score: mean per-query full-head Pearson r. Each search covers 28,800 cells per axis. These are post-selected maxima, not representative performance.</p>"""+"\n".join(blocks)
    (root/"report-figures/index.html").write_text(text)


def report_section(selection, prefix=""):
    """Chapter 4, shared by the archive and local Markdown report."""
    def fig(z, label):return f"![{label}]({prefix}{z['figure']})"
    text=["## 4. Complete A/B/C/D examples",
        "Following TriAttention Figure 2: **A = pre-RoPE pairs, B = post-RoPE pairs, C = across-head concentration, D = logit prediction**. Each example combines all four panels at one prompt, denoising step, layer and axis.",
        "A/B plot every token's Q/K in one axis-dominant 2D pair, with the same pair, token IDs and scale. C shows the pre-RoPE pair-MRL distribution over the 12 heads at this one prompt, step, layer and axis: **each head chooses its own dominant pair by C_f**; each R uses all 10,800 tokens. Q and K each contribute 12 values, so one head is 8.33%. C does not pool layers, steps or prompts. The archive also retains post-RoPE distributions.",
        "D uses the **full 128D head, all 64 pairs**, with Q/K centers averaged over all 10,800 tokens at this same prompt/step/layer/head. Curves average all same-axis-line query-key logits per signed offset; shading is measured population standard deviation. The displayed r first correlates each query over its axis-line keys and then averages all 10,800 valid queries. It is not the correlation between the two plotted mean curves.",
        "### 4.1 Fixed examples",
        "Retained from the previous report: p03, denoising step 20, layer 15, head 0, separately for t/h/w. Layer, head and pair indices are zero-based; denoising steps are numbered 1-20."]
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["fixed"]
        text += [fig(z,f"Fixed {axis} example with A/B/C/D"),f"{axis}: pair {z['dominant_pair']}; pair R_Q={z['q_dom_mrl_pre']:.6f}, R_K={z['k_dom_mrl_pre']:.6f}; full-head mean query r={z['per_query_r_mean']:.6f}."]
    text += ["The fixed examples underestimate the zero-offset peak. Pearson r measures shape agreement, not matching logit magnitudes. Concentration in one selected 2D pair and reconstruction using the full head are different measurements.",
        "### 4.2 Additional best reconstruction examples",
        "For 'best performance', rank **full-head mean per-query r**, separately for each axis, over 4 prompts x 20 steps x 30 layers x 12 heads = **28,800 cells per axis**, requiring all 10,800 queries to have defined correlations. The displayed pair is this winning head's C_f-selected axis-dominant pair; r evaluates the full head, not that isolated pair. These are maxima selected after the sweep, not representative performance.",
        "| Axis | Prompt | Step | Layer | Head | Selected pair | Pair R_Q | Pair R_K | Full-head mean query r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["best_reconstruction"]
        text.append(f"| {axis} | {z['prompt_id']} | {z['step']} | {z['layer']} | {z['head']} | {z['dominant_pair']} | {z['q_dom_mrl_pre']:.6f} | {z['k_dom_mrl_pre']:.6f} | {z['per_query_r_mean']:.6f} |")
    # Keep adjacent table rows contiguous, while separating prose and images.
    table_start=text.index("| Axis | Prompt | Step | Layer | Head | Selected pair | Pair R_Q | Pair R_K | Full-head mean query r |")
    table="\n".join(text[table_start:]);text=text[:table_start]+[table]
    text += [fig(selection["examples"][axis]["best_reconstruction"],f"Best full-head reconstruction along {axis}") for axis in ("t","h","w")]
    text += ["### 4.3 Best concentrated pair, ranked separately",
        "If 'best pair' means both 2D Q and K are concentrated, maximize min(R_Q,R_K) among the same 28,800 **axis-dominant selected pairs** per axis. This does not search every non-dominant band. Each R is pre-RoPE and averages all tokens within one condition. The winners differ from the best full-head predictions:"]
    rows=["| Axis | Prompt | Step | Layer | Head | Pair | Pair R_Q | Pair R_K |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for axis in ("t","h","w"):
        z=selection["examples"][axis]["best_pair"]
        rows.append(f"| {axis} | {z['prompt_id']} | {z['step']} | {z['layer']} | {z['head']} | {z['dominant_pair']} | {z['q_dom_mrl_pre']:.6f} | {z['k_dom_mrl_pre']:.6f} |")
    text += ["\n".join(rows),f"[All nine A/B/C/D figures, including the strongest-pair examples]({prefix}report-figures/index.html). [Exact selection definitions, source hashes and values]({prefix}report-figures/selection.json). No denoising-step or prompt averaging enters any selected example."]
    return "\n\n".join(text)


def update_report(root, selection):
    path=root/"REPORT.md"
    if not path.exists():return
    text=path.read_text()
    start=text.find("## Fixed examples across all three axes")
    if start<0:start=text.index("## 4. Complete A/B/C/D examples")
    end=text.index("## Interpretation limits",start)
    path.write_text(text[:start]+report_section(selection)+"\n\n"+text[end:])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",default="out/full-sweep-wan-20261101")
    args=ap.parse_args();root=ROOT/args.root
    selection=select_examples(root)
    for kind in ("fixed","best_pair","best_reconstruction"):
        for axis in ("t","h","w"):
            draw(root,selection,kind,axis);print(kind,axis,"done",flush=True)
    (root/"report-figures/selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    gallery(root,selection)
    update_report(root,selection)


if __name__=="__main__":main()
