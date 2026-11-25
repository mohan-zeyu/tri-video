"""Continuation of the widened-head C runner under the user's raised cap (30 GPU-hours, approved 20 Nov 2026).

Blocks on supervisor.lock until the original supervisor.py exits, so it never runs beside it. Then, only if the
update-1000 gate passed (guided validation error <= 0.10, gate-1000.json) and the full resumable checkpoint is
below 5,000, it resumes the identical recipe with --resume. A failed gate, a missing checkpoint or an exhausted
30 GPU-hour budget stops with an ASK state. Training-step logic and accounting are those of supervisor.py.
"""
import fcntl, json, math, os, subprocess, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
LOG = ROOT / 'out/distill-wide-run'
TRAIN = ROOT / 'out/distill-wide32'
APPROVAL = json.loads((LOG / 'budget-approval.json').read_text())
BUDGET_GPU_SECONDS = APPROVAL['approved_gpu_hours_total'] * 3600
RESERVE_DE_GPU_SECONDS = 3.0 * 3600
STEPS = 5000
BASE = json.loads((LOG / 'train-c01-request.json').read_text())['command']
i = BASE.index('--stop-after')
BASE = BASE[:i] + BASE[i + 2:]                      # the identical recipe without the gate stop
i = BASE.index('--budget-seconds')
BASE = BASE[:i] + BASE[i + 2:]
BASE = [x for x in BASE if x != '--resume']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    p = Path(p)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=2) + '\n')
    t.replace(p)


def used():
    total = 0.0
    for p in LOG.glob('*-status.json'):
        if not p.name.startswith(('screen-b', 'train-c')):
            continue
        s = read(p)
        total += (time.time() - s['started']) * len(s['gpus']) if s.get('state') == 'running' else s.get('gpu_seconds', 0)
    for ledger in LOG.glob('phase-*-gpu.tsv'):
        total += sum(float(x.split('\t')[2]) for x in ledger.read_text().splitlines() if x.strip())
    return total


def state(phase, **kw):
    write(LOG / 'pipeline-status.json', {'phase': phase, 'time': time.time(), 'pid': os.getpid(),
                                         'gpu_hours_used': used() / 3600, 'budget_gpu_hours': BUDGET_GPU_SECONDS / 3600, **kw})


def latest_step():
    p = TRAIN / 'checkpoint/latest.json'
    return read(p)['step'] if p.exists() else None


lock = (LOG / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)                     # wait for the original supervisor to exit
(LOG / 'supervisor-c2.pid').write_text(str(os.getpid()) + '\n')
try:
    prior = read(LOG / 'pipeline-status.json')
    if prior['phase'] == 'C-complete':
        raise SystemExit(0)
    for p in LOG.glob('train-c*-status.json'):
        assert read(p)['state'] != 'running', 'a training worker is still recorded as running: ' + p.name
    gate = read(LOG / 'gate-1000.json') if (LOG / 'gate-1000.json').exists() else None
    if not gate or not gate['passed']:
        raise RuntimeError('ASK: update-1000 gate not passed; continuation not started (%s)' % prior.get('error'))
    attempt = max(int(p.name[7:9]) for p in LOG.glob('train-c[0-9][0-9]-status.json'))
    retries = 0
    while latest_step() is not None and latest_step() < STEPS:
        attempt += 1
        remaining = BUDGET_GPU_SECONDS - RESERVE_DE_GPU_SECONDS - used()
        hard = int(remaining / 2) - 120
        if hard < 1800:
            raise RuntimeError('ASK: 30 GPU-hour cap reached before update %d (checkpoint %s)' % (STEPS, latest_step()))
        cmd = BASE + ['--resume', '--budget-seconds', str(hard - 600)]
        name = 'train-c%02d' % attempt
        write(LOG / (name + '-request.json'), {'command': cmd, 'approved_gpu_hours_total': BUDGET_GPU_SECONDS / 3600,
                                               'reserved_DE_gpu_seconds': RESERVE_DE_GPU_SECONDS, 'hard_wall_seconds': hard,
                                               'approval': APPROVAL})
        state('C-training', attempt=attempt, resume_from=latest_step(), gate=gate)
        worker = ['python3', str(LOG / 'gpu-worker.py'), '--root', str(ROOT), '--name', name, '--gpus', '2',
                  '--candidates', '2,3', '--wall-limit', str(hard), '--', *cmd]
        with (LOG / (name + '-worker.log')).open('ab') as f:
            code = subprocess.run(worker, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if (TRAIN / 'args.json').exists():
            (LOG / (name + '-args.json')).write_bytes((TRAIN / 'args.json').read_bytes())
        if code == 0 and latest_step() == STEPS:
            break
        if code != 0 and retries < 3 and 'SIGKILL' in (LOG / (name + '.log')).read_text():
            retries += 1
            continue
        if code == 0:
            raise RuntimeError('ASK: budget guard stopped training at full checkpoint step %s of %d' % (latest_step(), STEPS))
        raise RuntimeError('%s exited %s at checkpoint step %s' % (name, code, latest_step()))
    assert latest_step() == STEPS and (TRAIN / 'student/learned_center.pt').exists()
    state('C-complete', final_step=STEPS)
except SystemExit:
    raise
except Exception as e:
    state('needs-attention', error=str(e), traceback=traceback.format_exc(), checkpoint_step=latest_step())
    raise
