"""Claim idle lab GPUs, run an argv command, and account allocated GPU wall time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--name', required=True)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--candidates', default='1,2,6,5,0,4,3')
p.add_argument('--wall-limit', type=float, required=True)
p.add_argument('--completion-file', help='skip a queued worker if another worker completes this output')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path(a.root)
out = root / 'out/distill-noexempt-run'
out.mkdir(parents=True, exist_ok=True)
status = out / (a.name + '-status.json')
def write(data):
    tmp = status.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(status)
def query(args):
    return subprocess.check_output(['nvidia-smi'] + args, text=True)
def idle():
    rows = query(['--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    busy = {line.split(',')[0].strip() for line in query(['--query-compute-apps=gpu_uuid', '--format=csv,noheader']).splitlines()}
    result = set()
    for row in rows.splitlines():
        i, uuid, memory, utilization = [x.strip() for x in row.split(',')]
        if uuid not in busy and int(memory) < 32 and int(utilization) == 0:
            result.add(int(i))
    return result
locks = []
while True:
    if a.completion_file and (root / a.completion_file).exists():
        write({'state': 'skipped_already_complete', 'gpu_seconds': 0, 'time': time.time()})
        raise SystemExit(0)
    available = idle()
    chosen = []
    for i in map(int, a.candidates.split(',')):
        if i not in available:
            continue
        f = open('/tmp/tri-distill-pilot.EG6asH/gpu-%d.lock' % i, 'a')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            continue
        locks.append(f)
        chosen.append(i)
        if len(chosen) == a.gpus:
            break
    if len(chosen) == a.gpus and set(chosen).issubset(idle()):
        break
    for f in locks:
        f.close()
    locks = []
    write({'state': 'waiting_for_idle_gpus', 'time': time.time(), 'needed': a.gpus})
    time.sleep(30)
if a.completion_file and (root / a.completion_file).exists():
    write({'state': 'skipped_already_complete', 'gpu_seconds': 0, 'time': time.time()})
    raise SystemExit(0)
uuids = dict((int(row.split(',')[0].strip()), row.split(',')[1].strip())
             for row in query(['--query-gpu=index,uuid', '--format=csv,noheader']).splitlines())
env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids[i] for i in chosen), CUDA_DEVICE_ORDER='PCI_BUS_ID', WAN_SNAPSHOT='/dev/shm/tri-wan-snapshot',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HOME='/mnt/data/liuzeyu/hf-cache',
           VBENCH_CACHE_DIR='/tmp/tri-distill-pilot.EG6asH/evaluator-cache-static')
command = a.command[1:] if a.command[0] == '--' else a.command
start = time.time()
with open(out / (a.name + '.log'), 'a') as log:
    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write({'state': 'running', 'gpus': chosen, 'started': start, 'pid': proc.pid, 'command': command})
    try:
        code = proc.wait(timeout=a.wall_limit)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            code = proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
        code = code or 124
elapsed = time.time() - start
result = {'state': 'complete' if code == 0 else 'failed', 'exit_code': code, 'gpus': chosen,
          'started': start, 'finished': time.time(), 'wall_seconds': elapsed, 'gpu_seconds': elapsed * len(chosen)}
write(result)
print(json.dumps(result), flush=True)
raise SystemExit(code)
