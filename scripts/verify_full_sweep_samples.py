"""Independent CPU spot checks against saved real full-token Q/K tensors."""
import argparse
import json
import math
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]


def verify(root,cases=None,output_name="real-tensor-spot-checks.json"):
    torch.set_num_threads(4)
    rope=torch.load(root/"rope.pt",map_location="cpu",weights_only=True)
    cos,sin=rope["cos"],rope["sin"]
    cases=cases or [("p00",1,0),("p01",10,15),("p02",20,29),("p03",20,15)]
    records=[]
    for pid,step,layer in cases:
        path=root/pid/f"step-{step:03d}"/f"layer-{layer:03d}"
        raw=torch.load(path.with_suffix(".pt"),map_location="cpu",weights_only=True)
        d=json.loads(path.with_suffix(".json").read_text())
        q,k=raw["q"].float(),raw["k"].float()
        assert q.shape==k.shape==(12,10800,128)
        cq,ck=q.mean(1),k.mean(1)
        torch.testing.assert_close(cq,torch.tensor(d["center_q"]),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(ck,torch.tensor(d["center_k"]),atol=2e-5,rtol=2e-5)
        for name,x in (("q",q),("k",k)):
            z=x.reshape(12,10800,64,2)
            r=(z.mean(1).norm(dim=-1)/z.norm(dim=-1).mean(1).clamp_min(1e-20)).clamp(0,1)
            torch.testing.assert_close(r,torch.tensor(d[name+"_pair_mrl_pre"]),atol=2e-5,rtol=2e-5)
        def rotated(vector,indices,round_bf16):
            z=vector.reshape(-1,64,2);c=cos[indices];s=sin[indices]
            y=torch.stack([z[...,0]*c-z[...,1]*s,z[...,0]*s+z[...,1]*c],-1).flatten(-2)
            return y.bfloat16().float() if round_bf16 else y
        for axis,n,stride in [("t",12,900),("h",30,30),("w",30,1)]:
            ids=torch.arange(n)*stride
            for head in (0,11):
                # Explicit query at (0,0,0), explicit key positions along one
                # coordinate, no reshape/axis_lines helper from production.
                q0=rotated(q[head,0:1],torch.tensor([0]),True)[0]
                kk=rotated(k[head,ids],ids,True)
                measured=torch.stack([torch.dot(q0,x)/math.sqrt(128) for x in kk])
                qc=rotated(cq[head:head+1],torch.tensor([0]),False)[0]
                kc=rotated(ck[head:head+1].expand(n,-1),ids,False)
                prediction=torch.stack([torch.dot(qc,x)/math.sqrt(128) for x in kc])
                item=d["axes"][axis]
                torch.testing.assert_close(measured,torch.tensor(item["origin_query_measured"][head]),atol=2e-4,rtol=2e-5)
                torch.testing.assert_close(prediction,torch.tensor(item["origin_query_predicted"][head]),atol=2e-4,rtol=2e-5)
                assert item["origin_query_offsets"]==[-i for i in range(n)]
                assert sum(item["pairs_per_offset"])==10800*n
        records.append({"prompt":pid,"step":step,"layer":layer,"heads_for_logits":[0,11],"axes":["t","h","w"],"origin_query":"t=0,h=0,w=0","logit_dimensions":128,"center_tokens":10800,"pair_mrl_checked":"all 12 heads x all 64 pairs, Q and K, pre-RoPE","passed":True})
    result={"passed":True,"cases":records,"method":"Independent CPU, explicit coordinate indices and per-key full-vector dot products; centers and pair MRL use every saved video token."}
    (root/output_name).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",default="out/full-sweep-wan-20261101")
    args=ap.parse_args();print(json.dumps(verify(ROOT/args.root),indent=2))
