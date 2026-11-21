"""Widened-head study C runner (DISTILL_WIDE_PROMPT.md): 5,000-update Stage 2 fine-tune of the wide32 student.

Attempt 1 stops at update 1,000 (--stop-after keeps the full-run LR schedule) so the brief's gate can be applied
before more GPU time is spent: guided validation error (first 8 held-out prompts x 20 teacher states, both CFG
branches at w=5, whole latent, live weights) must be <= 0.10, and the measured per-update cost must fit the
approved 28 GPU-hour study budget with the D/E reserve. Otherwise it stops with an ASK state and launches nothing.
Later attempts append --resume to the identical recipe. No recurring model turns; a local watcher reads the status.
"""
import fcntl, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
LOG = ROOT / 'out/distill-wide-run'
TRAIN = ROOT / 'out/distill-wide32'
PYTHON = '/dev/shm/tri-center-runtime/bin/python'
BUDGET_GPU_SECONDS = 28 * 3600           # user-approved study allocation (A-E)
RESERVE_DE_GPU_SECONDS = 3.0 * 3600      # brief: D 1.3 + E 1.5 GPU-hours, rounded up
GATE_STEP, GATE_MAX = 1000, 0.10
STEPS = 5000
CANDIDATES = '2,3'                       # GPUs 1/6 belong to the running no-exemption study; 4 is excluded
BASE = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2', 'scripts/train_learned_center.py',
        '--stage', '2', '--out', 'out/distill-wide32', '--steps', str(STEPS), '--accum', '2', '--lr', '1e-5',
        '--lr-center', '1e-3', '--warmup', '250', '--lambda-cfg', '1.0', '--lambda-feat', '0.1', '--ema', '0.999',
        '--clip', '1.0', '--val-every', '250', '--val-prompts', '8', '--val-at-start', '--save-every', '500',
        '--ema-checkpoints', '--resume-state-with-ema', '--resume-state-every', '1000', '--seed', '20261114',
        '--selection', 'out/distill-selection-v4/selection.pt', '--variant', 'wide',
        '--calibration', 'out/distill-selection-v4/pair-statistics.pt', '--center-mode', 'mean',
        '--include-layer0', '--include-step1', '--traj', '/tmp/tri-distill-pilot.EG6asH/traj',
        '--snapshot', '/dev/shm/tri-wan-snapshot']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    p = Path(p)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=2) + '\n')
    t.replace(p)


def used():
    """Allocated GPU-seconds of finished workers (wall x GPUs) plus D/E ledgers; running workers counted live."""
    total = 0.0
    for p in LOG.glob('*-status.json'):
        if not p.name.startswith(('screen-b', 'train-c')):
            continue
        s = read(p)
        if s.get('state') == 'running':
            total += (time.time() - s['started']) * len(s['gpus'])
        else:
            total += s.get('gpu_seconds', 0)
    for ledger in LOG.glob('phase-*-gpu.tsv'):
        total += sum(float(x.split('\t')[2]) for x in ledger.read_text().splitlines() if x.strip())
    return total


def state(phase, **kw):
    write(LOG / 'pipeline-status.json', {'phase': phase, 'time': time.time(), 'pid': os.getpid(),
                                         'gpu_hours_used': used() / 3600, **kw})


def latest_step():
    p = TRAIN / 'checkpoint/latest.json'
    return read(p)['step'] if p.exists() else None


def validation(step):
    p = TRAIN / 'validation.jsonl'
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    hits = [r for r in rows if r['step'] == step]
    return hits[-1] if hits else None


def run_attempt(n, extra):
    remaining = BUDGET_GPU_SECONDS - RESERVE_DE_GPU_SECONDS - used()
    hard = int(remaining / 2) - 120
    if hard < 1800:
        raise RuntimeError('ASK: insufficient remaining C allocation (%.3f GPU-h left before the D/E reserve)' % (remaining / 3600))
    cmd = BASE + extra + ['--budget-seconds', str(hard - 600)]
    name = 'train-c%02d' % n
    write(LOG / (name + '-request.json'), {'command': cmd, 'approved_gpu_hours_total': BUDGET_GPU_SECONDS / 3600,
                                           'reserved_DE_gpu_seconds': RESERVE_DE_GPU_SECONDS, 'hard_wall_seconds': hard})
    worker = ['python3', str(LOG / 'gpu-worker.py'), '--root', str(ROOT), '--name', name, '--gpus', '2',
              '--candidates', CANDIDATES, '--wall-limit', str(hard), '--', *cmd]
    with (LOG / (name + '-worker.log')).open('ab') as f:
        code = subprocess.run(worker, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if (TRAIN / 'args.json').exists():
        (LOG / (name + '-args.json')).write_bytes((TRAIN / 'args.json').read_bytes())
    return name, code


lock = (LOG / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(LOG / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
try:
    state('B-waiting-for-screen')
    while not (LOG / 'screen-b-status.json').exists() or read(LOG / 'screen-b-status.json')['state'] in ('running', 'waiting_for_idle_gpus'):
        time.sleep(30)
    assert read(LOG / 'screen-b-status.json')['state'] == 'complete', 'B screen failed'
    screen = read(ROOT / 'out/distill-selection-v4/screen/complete.json')
    if not all(math.isfinite(v) for arm in screen['arms'].values() for v in arm['mean']):
        raise RuntimeError('ASK: non-finite B screen row')

    if latest_step() is not None:
        raise RuntimeError('ASK: out/distill-wide32 already has a checkpoint; refusing to start from teacher weights')
    attempt, retries = 0, 0
    # Phase C1: teacher weights -> update 1,000, then the gate.
    while True:
        attempt += 1
        state('C-training-to-gate', attempt=attempt)
        name, code = run_attempt(attempt, ['--stop-after', str(GATE_STEP)] + (['--resume'] if latest_step() else []))
        step = latest_step()
        if code == 0 and step == GATE_STEP:
            break
        if code != 0 and step and retries < 3 and 'SIGKILL' in (LOG / (name + '.log')).read_text():
            retries += 1
            continue
        raise RuntimeError('%s exited %s at checkpoint step %s before the update-1000 gate' % (name, code, step))
    row = validation(GATE_STEP)
    guided = row['mean'][2] if row else float('nan')
    c1_wall = sum(read(p)['wall_seconds'] for p in LOG.glob('train-c*-status.json') if 'wall_seconds' in read(p))
    projected = (STEPS - GATE_STEP) * (c1_wall / GATE_STEP) * 2
    gate = {'step': GATE_STEP, 'guided_validation_error': guided, 'threshold': GATE_MAX, 'passed': math.isfinite(guided) and guided <= GATE_MAX,
            'scope': 'Live student weights at update 1000; whole-latent relative squared velocity error vs the bf16 teacher, equal mean over the first 8 held-out prompts x 20 teacher entering states; conditional and unconditional combined at CFG 5; nothing exempt.',
            'c1_allocated_wall_seconds': c1_wall, 'projected_remaining_gpu_seconds': projected,
            'projection_basis': 'C1 allocated wall seconds per update (including model load, 5 validations and saves) x 4000 remaining updates x 2 GPUs',
            'gpu_seconds_used': used(), 'reserve_DE_gpu_seconds': RESERVE_DE_GPU_SECONDS, 'budget_gpu_seconds': BUDGET_GPU_SECONDS}
    gate['fits_budget'] = gate['gpu_seconds_used'] + projected + RESERVE_DE_GPU_SECONDS <= BUDGET_GPU_SECONDS
    write(LOG / 'gate-1000.json', gate)
    if not gate['passed']:
        raise RuntimeError('ASK: update-1000 guided validation error %.6f > %.2f' % (guided, GATE_MAX))
    if not gate['fits_budget']:
        raise RuntimeError('ASK: gate passed (%.6f) but projected study total %.2f GPU-h exceeds 28' %
                           (guided, (gate['gpu_seconds_used'] + projected + RESERVE_DE_GPU_SECONDS) / 3600))
    # Phase C2: resume the identical recipe to 5,000.
    retries = 0
    while True:
        attempt += 1
        state('C-training', attempt=attempt, gate=gate)
        name, code = run_attempt(attempt, ['--resume'])
        step = latest_step()
        if code == 0 and step == STEPS and (TRAIN / 'student/learned_center.pt').exists():
            break
        if code != 0 and step and retries < 3 and 'SIGKILL' in (LOG / (name + '.log')).read_text():
            retries += 1
            continue
        if code == 0:
            raise RuntimeError('ASK: budget guard stopped training at full checkpoint step %s of %d' % (step, STEPS))
        raise RuntimeError('%s exited %s at checkpoint step %s' % (name, code, step))
    state('C-complete', final_step=latest_step())
except Exception as e:
    state('needs-attention', error=str(e), traceback=traceback.format_exc(), checkpoint_step=latest_step())
    raise
