"""Widened-head D -> E -> CPU bias tables, after C (DISTILL_WIDE_PROMPT.md). Waits on supervisor.lock so it never
overlaps the C supervisor; launches nothing unless C completed at update 5000. Budget: benchmark.py enforces the
approved 30 GPU-hour cap from budget-approval.json. No recurring model turns; a local watcher reads the status."""
import fcntl, json, os, subprocess, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
LOG = ROOT / 'out/distill-wide-run'
PYTHON = '/dev/shm/tri-center-runtime/bin/python'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1',
       'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': '/mnt/data/liuzeyu/hf-cache', 'WAN_SNAPSHOT': '/dev/shm/tri-wan-snapshot',
       'PYTHONPATH': str(ROOT / 'scripts')}
def read(p): return json.loads(Path(p).read_text())
def write(p, d):
    t = Path(str(p) + '.tmp'); t.write_text(json.dumps(d, indent=2) + '\n'); t.replace(p)
def state(phase, **kw): write(LOG / 'pipeline-status.json', {'phase': phase, 'time': time.time(), 'pid': os.getpid(), **kw})
def run(argv, log):
    with (LOG / log).open('ab') as f:
        subprocess.run(argv, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
lock = (LOG / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
(LOG / 'de-runner.pid').write_text(str(os.getpid()) + '\n')
try:
    prior = read(LOG / 'pipeline-status.json')
    assert prior['phase'] in ('C-complete', 'D-complete', 'D-benchmark', 'E-benchmark'), 'C not complete: %s' % prior.get('phase')
    if not (LOG / 'phase-d-summary.json').exists():
        state('D-benchmark')
        run([PYTHON, '-u', str(LOG / 'benchmark.py'), 'D'], 'phase-d-driver.log')
    state('D-complete')
    if not (LOG / 'phase-e-summary.json').exists():
        state('E-benchmark')
        run([PYTHON, '-u', str(LOG / 'benchmark.py'), 'E'], 'phase-e-driver.log')
    state('E-bias-tables')
    run([PYTHON, '-u', str(LOG / 'diagnose_wide_bias.py'), '--student', 'out/distill-wide32/student',
         '--statistics', 'out/distill-selection-v4/pair-statistics.pt', '--out', 'out/distill-wide-run/bias-tables'], 'bias-tables.log')
    state('DE-complete')
except Exception as e:
    state('needs-attention', error=str(e), traceback=traceback.format_exc())
    raise
