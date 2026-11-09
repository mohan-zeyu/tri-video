"""Supervise two GPU trajectory workers, bounded CPU rendering, and audit.

Run from the authorized remote repo. This does not claim completion until
capture, every figure, and the final coverage checks have all succeeded.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"out/full-sweep-wan-20261101"


def status(value):
    tmp=DEST/f"supervisor-status.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(DEST/"supervisor-status.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",nargs=2,type=int,default=[0,2])
    args=parser.parse_args()
    assert len(set(args.gpus))==2
    logs=DEST/"logs";logs.mkdir(parents=True,exist_ok=True)
    old=DEST/"supervisor-status.json"
    if old.exists():
        previous=json.loads(old.read_text())
        proc=Path(f"/proc/{previous.get('pid',0)}/cmdline")
        if proc.exists() and b"launch_full_sweep.py" in proc.read_bytes():
            raise RuntimeError("A sweep supervisor is already running")
    usage=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    for line in usage.splitlines():
        gpu,memory,util=map(int,line.split(","))
        if gpu in args.gpus and (memory>1000 or util>10):
            raise RuntimeError(f"GPU {gpu} is no longer idle ({memory} MiB, {util}%); refusing to interfere")
    env=os.environ.copy();env.update(HF_HUB_OFFLINE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1")
    jobs=[];opened=[];started=time.time()
    for gpu,prompts in zip(args.gpus,(["p00","p01"],["p02","p03"])):
        log=(logs/f"gpu-{gpu}.log").open("a");opened.append(log)
        proc=subprocess.Popen([sys.executable,"-u","scripts/full_sweep.py","--prompts",*prompts],cwd=ROOT,env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=log,stderr=subprocess.STDOUT)
        jobs.append((f"GPU {gpu}",proc))
    log=(logs/"render.log").open("a");opened.append(log)
    renderer=subprocess.Popen([sys.executable,"-u","scripts/render_full_sweep.py","--root",str(DEST),"--workers","8"],cwd=ROOT,env={**env,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":""},stdout=log,stderr=subprocess.STDOUT)
    jobs.append(("render",renderer))
    def snapshot(stage):
        value={"pid":os.getpid(),"stage":stage,"elapsed_seconds":time.time()-started,
               "jobs":[{"name":name,"pid":proc.pid,"returncode":proc.poll()} for name,proc in jobs]}
        status(value);return value
    print(json.dumps(snapshot("running")),flush=True)
    while any(proc.poll() is None for _,proc in jobs):
        current=snapshot("running")
        bad=[j for j in current["jobs"] if j["returncode"] is not None and j["returncode"]!=0]
        if bad:
            status({**current,"stage":"failed","failures":bad})
            raise RuntimeError(f"Worker failure: {bad}; other independent workers may continue")
        time.sleep(5)
    if any(proc.returncode!=0 for _,proc in jobs):
        snapshot("failed");raise RuntimeError("A worker failed")
    snapshot("coverage_and_report")
    with (logs/"summary.log").open("w") as log:
        result=subprocess.run([sys.executable,"-u","scripts/summarize_full_sweep.py"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        snapshot("coverage_failed");raise RuntimeError("Coverage/report verification failed; see summary.log")
    snapshot("complete")
    print("SWEEP COMPLETE: captures, all figures and coverage verified",flush=True)
    for f in opened:f.close()


if __name__=="__main__":main()
