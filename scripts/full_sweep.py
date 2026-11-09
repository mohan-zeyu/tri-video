"""Run complete, real-prompt Wan trajectories with exact all-token metrics.

Run on 5090, with one GPU visible per process. See FULL_SWEEP_PLAN.md.
  CUDA_VISIBLE_DEVICES=0 /dev/shm/trivenv/bin/python scripts/full_sweep.py \
    --prompts p00 p01 --out out/full-sweep-wan-20261101
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

import torch
import torch.nn.functional as F

from full_sweep_metrics import AXES, analyze, rotate

ROOT=Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean(value),allow_nan=False,separators=(",",":"))+"\n")
    tmp.replace(path)


def save_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    torch.save(value,tmp); tmp.replace(path)


class PilotComplete(Exception):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/full_sweep_wan.json")
    ap.add_argument("--out",default="out/full-sweep-wan-20261101")
    ap.add_argument("--prompts",nargs="+",required=True)
    ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    dest=ROOT/args.out
    if args.pilot: dest=dest/"pilot"
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(cfg["seed"])
    import diffusers
    from diffusers import WanPipeline
    cache=Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))/"hub"/("models--"+cfg["model"].replace("/","--"))
    revision=(cache/"refs/main").read_text().strip()
    snapshot=cache/"snapshots"/revision
    print("Loading",snapshot,"GPU",os.environ.get("CUDA_VISIBLE_DEVICES"),flush=True)
    pipe=WanPipeline.from_pretrained(str(snapshot),torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    model=pipe.transformer
    layers=len(model.blocks); heads=model.config.num_attention_heads; dim=model.config.attention_head_dim
    assert (layers,heads,dim)==(30,12,128)
    assert (model.rope.t_dim,model.rope.h_dim,model.rope.w_dim)==(44,42,42)
    pair_axes=["t"]*22+["h"]*21+["w"]*21
    code_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"scripts/full_sweep_metrics.py"]}
    environment={"torch":torch.__version__,"diffusers":diffusers.__version__,"snapshot":revision,
                 "gpu":torch.cuda.get_device_name(0),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "hostname":socket.gethostname(),"code_sha256":code_hash}
    write_json(dest/"config.json",cfg)
    selected={p["id"]:p for p in cfg["prompts"] if p["id"] in args.prompts}
    assert len(selected)==len(args.prompts)
    for pid,prompt in selected.items():
        pdir=dest/pid
        if (pdir/"complete.json").exists():
            print(pid,"already complete; preserving",flush=True)
            continue
        pdir.mkdir(parents=True,exist_ok=True)
        state={"step":0,"active":False,"calls":[],"seen":{},"freqs":None,"qkv":{},"grid":None,"start":time.time(),"total_layers":0}
        write_json(pdir/"running.json",{"pid":os.getpid(),"environment":environment,"prompt":prompt,"start_unix":time.time()})
        pipe.text_encoder.to("cuda")
        pos,neg=pipe.encode_prompt(prompt=prompt["text"],negative_prompt=cfg["negative_prompt"],do_classifier_free_guidance=True,device="cuda",num_videos_per_prompt=1,max_sequence_length=512)
        pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
        assert not torch.equal(pos,neg)
        save_torch(pdir/"text-embeddings.pt",{"positive":pos.cpu(),"negative":neg.cpu()})
        write_json(pdir/"metadata.json",{"prompt":prompt,"seed":cfg["seed"],"steps":cfg["steps"],"environment":environment,"pilot":args.pilot,
            "branch_verification":"torch.equal(actual encoder_hidden_states, supplied positive/negative embedding), on every transformer call",
            "prediction_centers":"mean over every video token, same prompt/step/layer/head; full head; no cross-step/prompt pooling"})
        handles=[]

        def start_forward(module,inputs,kwargs):
            emb=kwargs["encoder_hidden_states"]
            if torch.equal(emb,pos): branch="conditional"
            elif torch.equal(emb,neg): branch="unconditional"
            else: raise AssertionError("Unrecognized transformer conditioning input")
            expected="conditional" if len(state["calls"])==0 else "unconditional"
            assert branch==expected,(branch,expected,state["step"])
            state["active"]=branch=="conditional"
            state["calls"].append({"branch":branch,"model_timestep":float(kwargs["timestep"][0].item()),"embedding_verified":True})
            if state["active"]:
                lat=kwargs["hidden_states"]
                patch=model.config.patch_size
                state["grid"]=tuple(int(lat.shape[i+2]//patch[i]) for i in range(3))
                save_torch(pdir/f"step-{state['step']+1:03d}"/"input-latents.pt",lat.cpu())

        def got_rope(module,inputs,output):
            if state["active"]:
                state["freqs"]=output
                if not (dest/"rope.pt").exists():
                    c,s=output
                    save_torch(dest/"rope.pt",{"cos":c[0,:,0,0::2].float().cpu(),"sin":s[0,:,0,1::2].float().cpu(),"grid":state["grid"],"pair_axes":pair_axes})

        def validate_layer(attn,box,q,k,cos,sin,output):
            qr=rotate(q,cos,sin,q.dtype).to(q.dtype)[None]
            kr=rotate(k,cos,sin,k.dtype).to(k.dtype)[None]
            v=box["v"].unflatten(2,(heads,dim)).transpose(1,2)
            reference=F.scaled_dot_product_attention(qr,kr,v).transpose(1,2).flatten(2)
            reference=attn.to_out[1](attn.to_out[0](reference))
            error=float((reference-output).abs().max())
            scale=float(output.abs().max())
            result={"max_abs_err":error,"output_scale":scale,"relative_max_err":error/max(scale,1e-12),"branch":"verified conditional","native_rope_cast":"bfloat16"}
            assert result["relative_max_err"]<=1e-3,result
            return result

        def grab(layer,name):
            def hook(module,inputs,output):
                if state["active"]:
                    if name=="v" and not (state["step"]==0 or state["step"] in (9,19) and layer in (0,15,29)): return
                    state["qkv"].setdefault(layer,{})[name]=output.detach()
            return hook

        def consume(layer):
            def hook(attn,inputs,output):
                if not state["active"]: return
                if args.pilot and layer not in (0,15,29):
                    state["qkv"].pop(layer,None)
                    return
                t0=time.time()
                box=state["qkv"].pop(layer)
                q,k=[box[name][0].unflatten(-1,(heads,dim)).permute(1,0,2) for name in ("q","k")]
                assert q.shape[1]==math.prod(state["grid"])
                c,s=state["freqs"]
                cos,sin=c[0,:,0,0::2].float(),s[0,:,0,1::2].float()
                validation=validate_layer(attn,box,q,k,cos,sin,output) if "v" in box else None
                result=analyze(q,k,cos,sin,state["grid"],pair_axes)
                result.update(prompt_id=pid,prompt=prompt["text"],step=state["step"]+1,layer=layer,
                              model_timestep=state["calls"][0]["model_timestep"],branch="conditional",grid=list(state["grid"]),validation=validation,
                              token_selection="all",centers="all video tokens at this single prompt/step/layer/head",
                              logit_dimensions="all 128 coordinates = all 64 pairs; not axis-only",code_sha256=code_hash)
                path=pdir/f"step-{state['step']+1:03d}"/f"layer-{layer:03d}"
                raw={"q":q.contiguous().cpu(),"k":k.contiguous().cpu(),"metadata":{"prompt_id":pid,"step":state["step"]+1,"layer":layer,"branch":"conditional","grid":state["grid"],"dtype":"bfloat16"}}
                save_torch(path.with_suffix(".pt"),raw)
                result["analysis_and_save_seconds"]=time.time()-t0
                write_json(path.with_suffix(".json"),result)
                state["seen"][layer]=state["seen"].get(layer,0)+1
                state["total_layers"]+=1
                write_json(pdir/"progress.json",{"prompt_id":pid,"step":state["step"]+1,"last_layer":layer,"completed_layer_captures":state["total_layers"],"elapsed_seconds":time.time()-state["start"],"gpu":environment["cuda_visible_devices"]})
                if layer in (0,15,29): print(pid,"step",state["step"]+1,"layer",layer,"seconds",round(time.time()-t0,3),flush=True)
            return hook

        def step_end(pipeline,step_idx,timestep,callback_kwargs):
            expected={0,15,29} if args.pilot else set(range(layers))
            assert set(state["seen"])==expected,state["seen"]
            assert all(v==1 for v in state["seen"].values())
            assert [c["branch"] for c in state["calls"]]==["conditional","unconditional"]
            write_json(pdir/f"step-{step_idx+1:03d}"/"step.json",{"step":step_idx+1,"scheduler_timestep":float(timestep.item()),"forwards":state["calls"],"layers":sorted(state["seen"]),"grid":state["grid"]})
            state["step"]+=1; state["calls"]=[]; state["seen"]={}; state["active"]=False
            print(pid,"finished step",step_idx+1,"elapsed",round(time.time()-state["start"],1),flush=True)
            if args.pilot: raise PilotComplete()
            return callback_kwargs

        handles.append(model.register_forward_pre_hook(start_forward,with_kwargs=True))
        handles.append(model.rope.register_forward_hook(got_rope))
        for layer,block in enumerate(model.blocks):
            if args.pilot and layer not in (0,15,29): continue
            attn=block.attn1
            for module,name in ((attn.norm_q,"q"),(attn.norm_k,"k"),(attn.to_v,"v")):
                handles.append(module.register_forward_hook(grab(layer,name)))
            handles.append(attn.register_forward_hook(consume(layer)))
        try:
            result=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,height=cfg["height"],width=cfg["width"],num_frames=cfg["frames"],
                        num_inference_steps=cfg["steps"],guidance_scale=cfg["guidance_scale"],generator=torch.Generator(device="cuda").manual_seed(cfg["seed"]),
                        output_type="latent",callback_on_step_end=step_end)
            assert state["step"]==cfg["steps"]
            save_torch(pdir/"final-latents.pt",result.frames.cpu())
        except PilotComplete:
            assert args.pilot
        except BaseException as error:
            write_json(pdir/"failed.json",{"error":repr(error),"traceback":traceback.format_exc(),"step":state["step"]+1})
            raise
        finally:
            for handle in handles: handle.remove()
        write_json(pdir/"complete.json",{"prompt_id":pid,"steps_completed":state["step"],"layer_captures":state["total_layers"],"elapsed_seconds":time.time()-state["start"],"environment":environment,"pilot":args.pilot})
        print(pid,"COMPLETE",state["step"],"steps",state["total_layers"],"layer captures",flush=True)
        del pos,neg; torch.cuda.empty_cache()


if __name__=="__main__":
    main()
