"""Render every all-token head figure, all-head sheet and distribution.

The watcher can run while GPU capture writes atomic layer JSON/PT files.
  python scripts/render_full_sweep.py --root out/full-sweep-wan-20261101 --workers 8
Use --once for a completed pilot directory. No token subsampling is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/matplotlib-full-sweep"))
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")


def atomic_json(path,value):
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value)+"\n"); tmp.replace(path)


def atomic_image(image,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    image.save(tmp,format="PNG",compress_level=1); tmp.replace(path)


def render_layer(filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    torch.set_num_threads(1)
    path=Path(filename)
    source=path.read_bytes(); digest=hashlib.sha256(source).hexdigest()
    done=path.with_suffix(".rendered.json")
    if done.exists() and json.loads(done.read_text()).get("metrics_sha256")==digest:
        return {"path":filename,"skipped":True}
    started=time.time()
    data=json.loads(source)
    root=path.parents[2]
    global _ROPE
    try: rope=_ROPE
    except NameError:
        _ROPE=torch.load(root/"rope.pt",map_location="cpu",weights_only=True); rope=_ROPE
    raw=torch.load(path.with_suffix(".pt"),map_location="cpu",weights_only=True)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":7,"axes.titlesize":8,"axes.labelsize":7,
        "axes.facecolor":"#eaeaf2","figure.facecolor":"white","axes.edgecolor":"white","grid.color":"white",
        "axes.grid":True,"axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,"axes.spines.bottom":False,
        "xtick.labelsize":6,"ytick.labelsize":6,"legend.frameon":False})
    blue,orange="#4c72b0","#dd8452"
    font=ImageFont.truetype(font_manager.findfont("DejaVu Serif"),12)
    counts={"sheets":0,"head_figures":0,"distributions":0}
    for axis,item in data["axes"].items():
        target=path.with_suffix("")/axis
        target.mkdir(parents=True,exist_ok=True)
        fig,axs=plt.subplots(data["heads"],3,figsize=(11,30),dpi=110,squeeze=False)
        fig.subplots_adjust(left=.07,right=.98,top=.955,bottom=.025,hspace=.80,wspace=.38)
        fig.suptitle(f"{data['prompt_id']}  |  denoising step {data['step']:02d}  |  layer {data['layer']:02d}  |  axis {axis}",fontsize=13,y=.994)
        fig.text(.5,.979,f"Verified conditional branch; all {data['tokens']:,} tokens. Scatters: selected 2D pair. Logits: full {data['head_dim']}D head, all axis-line query/key pairs.",ha="center",fontsize=8)
        fig.text(.5,.967,"Blue: Q / center prediction. Orange: K / measured logit. Centers average all tokens at this single prompt/step/layer/head; no denoising-step pooling.",ha="center",fontsize=8)
        for head,row in enumerate(axs):
            band=item["dominant_pair"][head]
            pair={name:raw[name][head,:,2*band:2*band+2].float() for name in ("q","k")}
            scale=max(float(pair[name].norm(dim=-1).max()) for name in pair)
            c,s=rope["cos"][:,band],rope["sin"][:,band]
            for col,post in ((0,False),(1,True)):
                ax=row[col]
                for name,color in (("q",blue),("k",orange)):
                    points=pair[name]
                    if post:
                        a,b=points.unbind(-1)
                        points=torch.stack((a*c-b*s,a*s+b*c),-1).to(torch.bfloat16).float()
                    points=points.numpy()/max(scale,1e-20)
                    ax.scatter(points[:,0],points[:,1],s=.6,alpha=.22,c=color,linewidths=0,rasterized=True)
                ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),xticks=[-1,0,1],yticks=[-1,0,1],xlabel="Re / shared scale",ylabel="Im / shared scale")
                ax.set_aspect("equal",adjustable="box")
                stage="post" if post else "pre"
                ax.set_title(f"Head {head:02d} · pair {band:02d} · {stage}-RoPE",fontsize=8)
                ax.text(.97,.96,f"Q R={item['q_dom_mrl_'+stage][head]:.3f}\nK R={item['k_dom_mrl_'+stage][head]:.3f}",transform=ax.transAxes,ha="right",va="top",fontsize=7)
            ax=row[2]
            x=np.array(item["offsets"])
            measured=np.array(item["measured_mean"][head]); sd=np.array(item["measured_std"][head]); predicted=np.array(item["predicted_mean"][head])
            ax.fill_between(x,measured-sd,measured+sd,color=orange,alpha=.20,linewidth=0)
            ax.plot(x,measured,color=orange,ls="--",lw=1.2,label="Measured mean ± SD")
            ax.plot(x,predicted,color=blue,ls=":",lw=1.5,label="Center prediction")
            rr=item["per_query_r_mean"][head]
            ax.set_title("Full-head logits · mean query r="+(f"{rr:.3f}" if rr is not None else "undefined"),fontsize=8)
            ax.set(xlabel=f"Signed {axis} offset (query − key)",ylabel="Pre-softmax logit",xlim=(x[0],x[-1]))
            ax.legend(fontsize=6,loc="best")
        fig.canvas.draw()
        canvas=np.asarray(fig.canvas.buffer_rgba())
        image=Image.fromarray(canvas).convert("RGB")
        atomic_image(image,target/"all-heads.png"); counts["sheets"]+=1
        renderer=fig.canvas.get_renderer()
        for head,row in enumerate(axs):
            boxes=[ax.get_tightbbox(renderer) for ax in row]
            x0=max(0,int(min(b.x0 for b in boxes))-6); x1=min(image.width,int(max(b.x1 for b in boxes))+7)
            y0=max(0,image.height-int(max(b.y1 for b in boxes))-7); y1=min(image.height,image.height-int(min(b.y0 for b in boxes))+7)
            tile=image.crop((x0,y0,x1,y1))
            labelled=Image.new("RGB",(tile.width,tile.height+43),"white")
            labelled.paste(tile,(0,43)); draw=ImageDraw.Draw(labelled)
            draw.text((5,2),f"{data['prompt_id']} | step {data['step']:02d} | layer {data['layer']:02d} | head {head:02d} | axis {axis} | selected pair {item['dominant_pair'][head]:02d}",fill="#111111",font=font)
            draw.text((5,20),f"Scatter blue=Q, orange=K; all {data['tokens']:,} tokens. One step; logits full {data['head_dim']}D; centers: all-token mean; curves: axis-pair means.",fill="#333333",font=font)
            atomic_image(labelled,target/"heads"/f"head-{head:02d}.png"); counts["head_figures"]+=1
        plt.close(fig)
        hist,haxs=plt.subplots(1,2,figsize=(8.6,2.9),dpi=120,sharex=True,sharey=True)
        hist.subplots_adjust(left=.09,right=.98,top=.74,bottom=.24,wspace=.25)
        for ax,stage in zip(haxs,("pre","post")):
            q=np.array(item["q_dom_mrl_"+stage]); k=np.array(item["k_dom_mrl_"+stage])
            ax.hist([q,k],bins=np.linspace(0,1,21),weights=[np.ones(len(q))*100/len(q),np.ones(len(k))*100/len(k)],color=[blue,orange],label=["Q","K"],rwidth=.86)
            ax.set(title=f"{stage}-RoPE selected-pair MRL",xlabel="2D-pair concentration R",xlim=(0,1),ylim=(0,100))
            ax.legend(fontsize=8)
        haxs[0].set_ylabel("Heads (%)")
        hist.suptitle(f"{data['prompt_id']} · step {data['step']:02d} · layer {data['layer']:02d} · axis {axis}",fontsize=11,y=.98)
        hist.text(.5,.84,f"Exactly {data['heads']} heads; each selects its strongest {axis}-axis pair by mean-token norm product. R uses all {data['tokens']:,} tokens.",ha="center",fontsize=8)
        hist.text(.5,.025,"No pooling over prompts, denoising steps or layers. Pre/post use identical selected pairs and tokens.",ha="center",fontsize=8)
        hist.canvas.draw(); atomic_image(Image.fromarray(np.asarray(hist.canvas.buffer_rgba())).convert("RGB"),target/"concentration-distribution.png")
        plt.close(hist); counts["distributions"]+=1
    elapsed=time.time()-started
    atomic_json(done,{"metrics_sha256":digest,"counts":counts,"elapsed_seconds":elapsed,"pid":os.getpid(),"tokens_subsampled":False})
    return {"path":filename,"seconds":round(elapsed,2),"counts":counts}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",required=True); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--once",action="store_true")
    ap.add_argument("--reverse",action="store_true",help="An auxiliary pool may take newest records first; atomic cached renders are idempotent")
    ap.add_argument("--progress-prefix",default="render")
    args=ap.parse_args(); root=ROOT/args.root
    root.mkdir(parents=True,exist_ok=True)
    submitted=set(); pending={}; finished=0; started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            candidates=sorted(root.glob("p*/step-*/layer-*.json"),reverse=args.reverse)
            candidates=[p for p in candidates if not p.name.endswith(".rendered.json")]
            for path in candidates:
                if str(path) in submitted or len(pending)>=args.workers*2: continue
                pending[pool.submit(render_layer,str(path))]=path; submitted.add(str(path))
            if pending:
                done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    path=pending.pop(future)
                    try: result=future.result()
                    except BaseException as error:
                        atomic_json(root/f"{args.progress_prefix}-failed.json",{"path":str(path),"error":repr(error),"traceback":traceback.format_exc()})
                        raise
                    finished+=1
                    if finished<=3 or finished%20==0:
                        print("RENDER",finished,"layer files",result,flush=True)
                    atomic_json(root/f"{args.progress_prefix}-progress.json",{"finished_layers":finished,"pending":len(pending),"elapsed_seconds":time.time()-started,"workers":args.workers})
            remaining=any(str(p) not in submitted for p in candidates)
            cfg=root/"config.json"
            capture_done=cfg.exists() and all((root/p["id"]/"complete.json").exists() for p in json.loads(cfg.read_text())["prompts"])
            if (args.once or capture_done) and not pending and not remaining:
                atomic_json(root/f"{args.progress_prefix}-complete.json",{"layer_files":finished,"elapsed_seconds":time.time()-started,"once":args.once})
                print("RENDER COMPLETE",finished,flush=True); return
            if not pending:
                if list(root.glob("p*/failed.json")): raise RuntimeError("A capture process failed; inspect its failed.json")
                time.sleep(5)


if __name__=="__main__":
    main()
