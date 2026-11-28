#!/usr/bin/env python3
"""One-command learned-center study: choose which pairs become means (and, at inference, zeros), then run
selection -> training-free screen -> Stage 2 fine-tune -> 100-prompt means and zeros benchmarks -> evidence.

    python scripts/center_study.py configs/studies/shared32.json            # run / resume every stage
    python scripts/center_study.py configs/studies/shared32.json --dry-run  # print the plan and commands only
    python scripts/center_study.py configs/studies/shared32.json --stage screen

The study spec (JSON) is the interface. The pair choice and the bypass rules are data:
  "selection":   a selection spec (scripts/pair_selection.py): a preset name ("shared", "split", "wide", ...) or
                 {"policies": [{"criterion": damage|error|error_normalized|energy|fixed_high|all, "combine": shared|separate,
                                "pool": [steps|branches|heads...], "pairs": n, "rank": lowest|highest, "role": mean|content_nope}, ...],
                  "wide": bool}
  "bypass":      {"layers": [...], "steps": [...], "branches": [...], "heads": [...] | {"<layer>": [...]},
                  "tokens": {"kind": "frames", "native": [0]} | {"kind": "border", "width": 1}}   ({} = nothing exempt)
  legacy fields still accepted: "variant" + "means_pairs" (-> a preset) and "exempt" (-> layer 0 / step 1 bypass)
Arms: "mean" uses the trained means; "delete" zeroes the same pairs at inference on the same weights (zeros swap).
Everything else (recipe, budget, GPUs, references, lab paths, publishing) has defaults below; see
configs/studies/shared32.json for a complete example.

Run on the lab host from a checkout pinned for the study. Every stage is idempotent: completed outputs are
detected and skipped, training resumes from the latest complete checkpoint, and an external SIGKILL/SIGTERM
after a complete save is retried (bounded). GPU time is allocated wall seconds x GPUs (ledger.tsv), enforced
against "approved_gpu_hours"; idle GPUs only (no compute process, 0% utilisation, < 32 MiB, exclusive lock).
The runner stops with an "ASK:" state rather than exceeding the budget or changing the recipe.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ['overall_consistency', 'aesthetic_quality', 'imaging_quality', 'dynamic_degree', 'subject_consistency',
              'background_consistency', 'motion_smoothness']
DEFAULTS = {
    'selection': None, 'bypass': None,
    'variant': 'shared', 'means_pairs': 22, 'exempt': False, 'approved_gpu_hours': None,
    'train': {'steps': 2500, 'accum': 2, 'lr': 1e-5, 'lr_center': 2e-4, 'warmup': 250, 'lambda_cfg': 1.0,
              'lambda_feat': 0.1, 'ema': 0.999, 'clip': 1.0, 'val_every': 250, 'val_prompts': 8, 'save_every': 500,
              'resume_state_every': 1000, 'seed': 20261114, 'ranks': 2, 'max_retries': 3,
              'gate': None},   # e.g. {"step": 1000, "max_guided": 0.10}
    'screen_prompts': 32,
    'reserve_benchmark_gpu_hours': 3.0,          # kept back from training for the two benchmarks and the screen
    'gpus': {'candidates': [2, 3, 1, 6, 5, 0], 'benchmark_workers': 2},
    'arms': None,                                # default {"mean": "<name>_means", "delete": "<name>_zeros"}
    'references': {'mean': [['dense_control', 'distill-results/dense-paired/summary.json'],
                            ['shared22', 'distill-results/shared22/full100/summary.json'],
                            ['shared22_delete', 'distill-results/shared22-delete/full100/summary.json'],
                            ['s22_all_means', 'distill-results/noexempt/full100/s22_all_means/summary.json']],
                   'delete': [['s22_all_zeros', 'distill-results/noexempt/full100/s22_all_zeros/summary.json'],
                              ['s22_zeros', 'distill-results/swap/full100/summary.json']]},
    'curves': [['shared22 (layer 0 / step 1 exempt)', 'distill-results/shared22/validation.jsonl'],
               ['shared22, nothing exempt', 'distill-results/noexempt/training/validation.jsonl']],
    'lab': {'python': '/dev/shm/tri-center-runtime/bin/python', 'snapshot': '/dev/shm/tri-wan-snapshot',
            'traj': '/tmp/tri-distill-pilot.EG6asH/traj', 'lock_dir': '/tmp/tri-distill-pilot.EG6asH',
            'stats': 'out/distill-pair-stats', 'hf_home': '/mnt/data/liuzeyu/hf-cache',
            'evaluator_cache': '/tmp/tri-distill-pilot.EG6asH/evaluator-cache-static', 'vbench': '/dev/shm/tri-vbench-source',
            'frozen_teacher': '/mnt/data/liuzeyu/tri/out/center-benchmark-wan-20261109',
            'reuse_teacher_scores': '/mnt/data/liuzeyu/tri-video-distill-bf334ec/out/learned-center-wan-student',
            'benchmark_config': 'configs/learned_center_benchmark_wan.json'},
    'publish': {'enabled': True, 'branch': 'claude/magical-goodall-dmq9i1', 'as': 'claude-lab', 'to': 'claude',
                'brief': None, 'ack': None},
}


# ----------------------------------------------------------------------------------------------- spec & paths
def merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_spec(path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from pair_selection import describe, normalize_spec, spec_hash
    spec = merge(DEFAULTS, json.loads(Path(path).read_text()))
    if not spec.get('name') or not spec['name'].replace('_', '').replace('-', '').isalnum():
        raise ValueError('spec needs a simple "name"')
    if spec['selection'] is None:                      # legacy three-field choice -> preset with the given size
        if spec['variant'] not in ('shared', 'separate', 'pooled', 'split', 'wide', 'fixed_high'):
            raise ValueError('unknown variant %r' % spec['variant'])
        if spec['variant'] != 'fixed_high' and not 1 <= int(spec['means_pairs']) <= 64:
            raise ValueError('means_pairs must be 1..64')
        sel = normalize_spec(spec['variant'])
        for p in sel['policies']:
            if 'complement_of' not in p and p['criterion'] not in ('all', 'fixed_high'):
                p['pairs'] = int(spec['means_pairs'])
        spec['selection'] = sel
        spec['variant_key'] = spec['variant']
    else:
        spec['selection'] = normalize_spec(spec['selection'])
        spec['variant_key'] = 'policy'
    if spec['bypass'] is None:
        spec['bypass'] = {'layers': [0], 'steps': [0]} if spec['exempt'] else {}
    spec['exempt'] = bool(spec['bypass'])
    spec['selection_hash'] = spec_hash(spec['selection'])
    spec['choice'] = describe(spec['selection'])
    if not spec['approved_gpu_hours']:
        raise ValueError('approved_gpu_hours is required (the user-approved allocation)')
    spec['arms'] = spec['arms'] or {'mean': spec['name'] + '_means', 'delete': spec['name'] + '_zeros'}
    return spec


def bypass_text(spec):
    b = spec['bypass']
    return 'nothing exempt (all 30 layers, all 20 steps)' if not b else 'bypass ' + json.dumps(b, sort_keys=True)


class Paths:
    def __init__(self, spec, root=ROOT):
        n = spec['name']
        self.root = root
        self.work = root / 'out' / ('study-' + n)
        self.run = self.work / 'run'
        self.selection = self.work / 'selection'
        self.screen = self.work / 'screen'
        self.train = self.work / 'train'
        self.student = self.train / 'student'
        self.screen_trained = self.work / 'screen-trained'
        self.bench = {'mean': self.work / 'bench-mean', 'delete': self.work / 'bench-delete'}
        self.results = root / 'distill-results' / n
        self.ledger = self.run / 'ledger.tsv'
        self.active = self.run / 'active'
        self.status = self.run / 'status.json'


def selection_args(spec, paths):
    args = ['scripts/select_pairs.py', '--stats', spec['lab']['stats'], '--out', str(paths.selection), '--bypass', json.dumps(spec['bypass'])]
    if spec['variant_key'] == 'policy':
        args += ['--spec', json.dumps(spec['selection'])]
    else:                                              # legacy preset with its size
        v, n = spec['variant'], str(spec['means_pairs'])
        args += {'split': ['--means-pairs', n], 'wide': ['--content-pairs', n]}.get(v, ['--pairs', n])
    return args


def screen_args(spec, paths, student=False):
    a = ['scripts/screen_selections.py', '--traj', spec['lab']['traj'], '--val-prompts', str(spec['screen_prompts'])]
    if student:
        return a + ['--student', str(paths.student), '--modes', 'mean', 'delete', '--out', str(paths.screen_trained)]
    a += ['--selection', str(paths.selection / 'selection.pt'), '--statistics', str(paths.selection / 'pair-statistics.pt'),
          '--variants', spec['variant_key'], '--modes', 'mean', 'delete', '--out', str(paths.screen), '--bypass', json.dumps(spec['bypass'])]
    return a


def train_args(spec, paths, resume=False, stop_after=None, budget_seconds=None):
    t, lab = spec['train'], spec['lab']
    a = [lab['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(t['ranks']),
         'scripts/train_learned_center.py', '--stage', '2', '--out', str(paths.train), '--steps', str(t['steps']),
         '--accum', str(t['accum']), '--lr', str(t['lr']), '--lr-center', str(t['lr_center']), '--warmup', str(t['warmup']),
         '--lambda-cfg', str(t['lambda_cfg']), '--lambda-feat', str(t['lambda_feat']), '--ema', str(t['ema']),
         '--clip', str(t['clip']), '--val-every', str(t['val_every']), '--val-prompts', str(t['val_prompts']),
         '--val-at-start', '--save-every', str(t['save_every']), '--ema-checkpoints', '--resume-state-with-ema',
         '--resume-state-every', str(t['resume_state_every']), '--seed', str(t['seed']),
         '--selection', str(paths.selection / 'selection.pt'), '--variant', spec['variant_key'],
         '--calibration', str(paths.selection / 'pair-statistics.pt'), '--center-mode', 'mean',
         '--traj', lab['traj'], '--snapshot', lab['snapshot'], '--bypass', json.dumps(spec['bypass'])]
    if resume:
        a.append('--resume')
    if stop_after:
        a += ['--stop-after', str(stop_after)]
    if budget_seconds:
        a += ['--budget-seconds', str(int(budget_seconds))]
    return a


def expected_calls(layers_active, steps_active, uses_nope, branches_active=2):
    """Per video: Q and K replacement calls per active CFG branch; one processor call per layer/branch/step if NoPE/wide."""
    return {'intervention_calls': layers_active * 2 * branches_active * steps_active,
            'nope_calls': layers_active * branches_active * steps_active if uses_nope else 0}


def check_record(rec, expect, arm, mode):
    """Audit one generated video record against the trained center (expect from audit_center)."""
    env, sel = rec['environment'], rec['environment']['selection']
    problems = []
    def need(cond, what):
        if not cond:
            problems.append(what)
    need(rec['mode'] == arm, 'arm')
    need(rec['finite_step_latents'] == 20, 'finite latents')
    need(rec['intervention_calls'] == rec['expected_intervention_calls'] == expect['calls']['intervention_calls'], 'intervention calls')
    need(rec['nope_calls'] == rec['expected_nope_calls'] == expect['calls']['nope_calls'], 'nope/processor calls')
    for k in ('pairs_per_cell_q', 'pairs_per_cell_k', 'q_k_intersection_mean', 'nope_pairs_per_cell'):
        need(abs(sel[k] - expect['selection'][k]) < 1e-9, k)
    need(bool(sel.get('wide_head', False)) == expect['selection']['wide_head'], 'wide flag')
    need(not sel['default_high_masks'] or spec_is_fixed_high(expect), 'default masks')
    need(env['center_mode'] == mode and env['learned_center_sha256'] == expect['center_sha256'], 'center mode/hash')
    need(env['exempt'] == expect['exempt'], 'exemptions')
    need(env.get('bypass', expect['bypass']) == expect['bypass'], 'bypass rules')
    need(rec['frames'] == [45, 480, 480, 3], 'frames')
    return problems


def spec_is_fixed_high(expect):
    return expect.get('variant') == 'fixed_high' or 'fixed_high' in json.dumps(expect.get('selection_spec') or {})


# ----------------------------------------------------------------------------------------------- runner
class Study:
    def __init__(self, spec, dry_run=False, root=ROOT):
        self.spec, self.dry = spec, dry_run
        self.p = Paths(spec, root)
        self.budget = float(spec['approved_gpu_hours']) * 3600
        self.lab = spec['lab']
        self.env = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '', 'WAN_SNAPSHOT': self.lab['snapshot'],
                    'HF_HOME': self.lab['hf_home'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
                    'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                    'VBENCH_CACHE_DIR': self.lab['evaluator_cache'], 'PYTHONPATH': str(root / 'scripts')}

    # --- bookkeeping
    def write(self, path, data):
        path = Path(path)
        if self.dry:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(path)

    def state(self, phase, **kw):
        self.write(self.p.status, {'phase': phase, 'time': time.time(), 'pid': os.getpid(), 'study': self.spec['name'],
                                   'gpu_hours_used': self.used() / 3600, 'approved_gpu_hours': self.budget / 3600, **kw})
        print('[%s] %s %s' % (time.strftime('%H:%M:%S'), phase, json.dumps(kw)[:300]), flush=True)

    def used(self):
        total = 0.0
        if self.p.ledger.exists():
            total += sum(float(line.split('\t')[2]) * len(line.split('\t')[1].split(','))
                         for line in self.p.ledger.read_text().splitlines() if line.strip())
        if self.p.active.exists():
            for f in self.p.active.glob('*.json'):
                a = json.loads(f.read_text())
                total += (time.time() - a['started']) * len(a['gpus'])
        return total

    def remaining(self):
        return self.budget - self.used()

    def cpu(self, name, argv):
        print('CPU', name, ' '.join(map(str, argv)), flush=True)
        if self.dry:
            return
        with (self.p.run / (name + '.log')).open('ab') as log:
            subprocess.run([self.lab['python'], '-u', *argv], cwd=self.p.root, env=self.env, stdout=log, stderr=subprocess.STDOUT, check=True)

    # --- GPUs
    def smi(self, *args):
        return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=20).strip()

    def idle(self):
        rows = self.smi('--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits').splitlines()
        busy = set(self.smi('--query-compute-apps=gpu_uuid', '--format=csv,noheader').split())
        out = {}
        for row in rows:
            i, uuid, mem, util = [x.strip() for x in row.split(',')]
            if uuid not in busy and int(mem) < 32 and int(util) == 0:
                out[int(i)] = uuid
        return out

    def try_lease(self, n):
        """Lease n idle candidate GPUs now, or return None. Locks are exclusive flocks shared with other runners."""
        idle = self.idle()
        got = []
        for i in self.spec['gpus']['candidates']:
            if i not in idle:
                continue
            f = open(Path(self.lab['lock_dir']) / ('gpu-%d.lock' % i), 'a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            got.append((i, idle[i], f))
            if len(got) == n:
                break
        if len(got) == n and {g[0] for g in got} <= set(self.idle()):
            return got
        for g in got:
            g[2].close()
        return None

    def start(self, name, argv, lease):
        self.p.active.mkdir(parents=True, exist_ok=True)
        log = (self.p.run / (name + '.log')).open('ab')
        env = {**self.env, 'CUDA_VISIBLE_DEVICES': ','.join(g[1] for g in lease)}
        proc = subprocess.Popen(argv, cwd=self.p.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        self.write(self.p.active / (name + '.json'), {'name': name, 'gpus': [g[0] for g in lease], 'started': started, 'pid': proc.pid, 'argv': argv})
        print('START %s GPUs=%s pid=%d' % (name, [g[0] for g in lease], proc.pid), flush=True)
        return {'name': name, 'proc': proc, 'lease': lease, 'log': log, 'started': started}

    def finish(self, job, rc):
        seconds = time.time() - job['started']
        with self.p.ledger.open('a') as f:
            f.write('%s\t%s\t%.3f\t%d\n' % (job['name'], ','.join(str(g[0]) for g in job['lease']), seconds, rc))
        (self.p.active / (job['name'] + '.json')).unlink(missing_ok=True)
        for g in job['lease']:
            g[2].close()
        job['log'].close()
        print('EXIT %s rc=%d seconds=%.1f' % (job['name'], rc, seconds), flush=True)

    def stop(self, job):
        for sig, wait in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
            try:
                os.killpg(job['proc'].pid, sig)
            except ProcessLookupError:
                pass
            try:
                return job['proc'].wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        return job['proc'].wait()

    def wave(self, tasks, gpus_per_task=1, wall_limit=None):
        """Run tasks [(name, argv)] on leased idle GPUs, at most gpus.benchmark_workers at once, under the budget."""
        for name, argv in tasks:
            print('GPU', name, ' '.join(map(str, argv)), flush=True)
        if self.dry:
            return
        pending, active, failure = list(tasks), [], None
        while pending or active:
            for job in active[:]:
                rc = job['proc'].poll()
                if rc is None and wall_limit and time.time() - job['started'] > wall_limit:
                    rc = self.stop(job) or 124
                    # our own stop: never mistaken for an external signal by the training retry logic
                    failure = failure or 'ASK: %s hit its %.0f s wall limit' % (job['name'], wall_limit)
                if rc is not None:
                    self.finish(job, rc)
                    active.remove(job)
                    failure = failure or (rc and '%s exited %d' % (job['name'], rc))
            if self.used() + 60 >= self.budget:
                failure = 'ASK: study reached its approved %.1f GPU-hours' % (self.budget / 3600)
            if failure:
                pending.clear()
                for job in active:
                    self.finish(job, self.stop(job) or 125)
                raise RuntimeError(failure)
            while pending and len(active) < self.spec['gpus']['benchmark_workers']:
                lease = self.try_lease(gpus_per_task)
                if not lease:
                    break
                name, argv = pending.pop(0)
                active.append(self.start(name, argv, lease))
            if pending and not active:
                self.state('waiting-for-idle-gpus', tasks=[t[0] for t in pending])
            time.sleep(5 if active else 30)

    # --- stages
    def stage_select(self):
        sel = self.p.selection / 'selection.pt'
        if not sel.exists():
            self.state('select')
            self.cpu('select', selection_args(self.spec, self.p))
        if self.dry:
            return
        audit = self.audit_selection()
        self.write(self.p.run / 'selection-audit.json', audit)

    def audit_selection(self):
        import torch
        sys.path.insert(0, str(self.p.root / 'scripts'))
        from learned_center import LearnedCenter
        from pair_selection import spec_hash
        v = self.spec['variant_key']
        masks = torch.load(self.p.selection / 'selection.pt', map_location='cpu')['variants'][v]
        assert spec_hash(masks['spec']) == self.spec['selection_hash'], 'selection.pt was built from a different selection spec'
        q, k, n = masks['mask_q'].sum(-1), masks['mask_k'].sum(-1), masks['mask_nope'].sum(-1)
        for name, counts in (('mean_q', q), ('mean_k', k), ('content', n)):
            assert (counts == counts.flatten()[0]).all(), 'the number of %s pairs differs between cells' % name
        want_q, want_k, want_n = int(q.flatten()[0]), int(k.flatten()[0]), int(n.flatten()[0])
        if all(p.get('combine', 'shared') == 'shared' for p in self.spec['selection']['policies']):
            assert torch.equal(masks['mask_q'], masks['mask_k'])
        assert bool(masks['wide']) == bool(self.spec['selection']['wide'])
        static = LearnedCenter.from_checkpoint(torch.load(self.p.selection / ('static-' + v) / 'learned_center.pt', map_location='cpu'))
        b = self.spec['bypass']
        active = (int(static.layer_active.sum()), int(static.step_active.sum()))
        assert active == (static.layers - len(b.get('layers', [])), static.steps - len(b.get('steps', []))), active
        assert static.bypass_spec['branches'] == [int(x) for x in b.get('branches', [])] and static.token_rule == (b.get('tokens') or None)
        return {'variant': v, 'selection': self.spec['selection'], 'selection_hash': self.spec['selection_hash'], 'choice': self.spec['choice'],
                'bypass': static.bypass_spec, 'pairs_per_cell_q': want_q, 'pairs_per_cell_k': want_k, 'content_pairs_per_cell': want_n,
                'wide': bool(masks['wide']), 'cells': int(q.numel()), 'layers_active': active[0], 'steps_active': active[1],
                'cells_active': int(static.cell_active.sum()),
                'scope': 'Per layer-step-branch-head cell, pair-level masks over the 64 rotary pairs per head; statistics pooled over the calibration prompts; bypass rules as recorded.'}

    def stage_screen(self):
        if (self.p.screen / 'complete.json').exists():
            return
        self.state('screen')
        self.wave([('screen', [self.lab['python'], '-u', *screen_args(self.spec, self.p)])])
        if not self.dry:
            d = json.loads((self.p.screen / 'complete.json').read_text())
            if not all(math.isfinite(x) for arm in d['arms'].values() for x in arm['mean']):
                raise RuntimeError('ASK: non-finite training-free screen row')

    def latest_step(self):
        f = self.p.train / 'checkpoint' / 'latest.json'
        return json.loads(f.read_text())['step'] if f.exists() else None

    def stage_train(self):
        t = self.spec['train']
        steps = t['steps']
        if self.latest_step() == steps and (self.p.student / 'learned_center.pt').exists():
            return
        gate = t.get('gate')
        retries = 0
        attempt = len(list(self.p.run.glob('train-a*.log'))) if self.p.run.exists() else 0
        while True:
            step = self.latest_step()
            if step == steps and (self.p.student / 'learned_center.pt').exists():
                break
            if self.dry and attempt >= 1:
                break
            stop_after = gate['step'] if gate and (step or 0) < gate['step'] else None
            if gate and step == gate['step'] and not self.gate_passed():
                raise RuntimeError('ASK: update-%d gate failed (%s)' % (gate['step'], json.dumps(self.gate_record())))
            reserve = float(self.spec['reserve_benchmark_gpu_hours']) * 3600
            hard = (self.remaining() - reserve) / t['ranks'] - 120
            if hard < 1800 and not self.dry:
                raise RuntimeError('ASK: not enough approved GPU time left for training (%.2f GPU-hours remain, %.1f reserved)'
                                   % (self.remaining() / 3600, reserve / 3600))
            attempt += 1
            name = 'train-a%02d' % attempt
            argv = train_args(self.spec, self.p, resume=step is not None, stop_after=stop_after, budget_seconds=max(hard - 600, 600))
            self.write(self.p.run / (name + '-request.json'), {'argv': argv, 'hard_wall_seconds': hard, 'resume_from': step})
            self.state('train', attempt=name, resume_from=step)
            if self.dry:
                self.wave([(name, argv)], gpus_per_task=t['ranks'])
                continue
            before = step
            try:
                self.wave_single(name, argv, t['ranks'], hard)
            except RuntimeError as e:
                log = (self.p.run / (name + '.log')).read_text(errors='replace')
                external = 'SIGKILL' in log or 'SIGTERM' in log
                if str(e).startswith('ASK') or not external or retries >= t['max_retries'] or self.latest_step() is None:
                    raise
                retries += 1
                self.state('train-retry-after-external-signal', attempt=name, checkpoint=self.latest_step(), previous=before)
                continue
            if stop_after:
                self.write(self.p.run / 'gate.json', self.gate_record())
                continue
            if self.latest_step() != steps:
                raise RuntimeError('ASK: budget guard stopped training at full checkpoint %s of %d' % (self.latest_step(), steps))
        if not self.dry:
            self.audit_center(write=True)

    def wave_single(self, name, argv, gpus, wall_limit):
        self.wave([(name, argv)], gpus_per_task=gpus, wall_limit=wall_limit)

    def validation(self):
        f = self.p.train / 'validation.jsonl'
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def gate_record(self):
        g = self.spec['train']['gate']
        rows = [r for r in self.validation() if r['step'] == g['step']]
        guided = rows[-1]['mean'][2] if rows else float('nan')
        return {'step': g['step'], 'guided_validation_error': guided, 'threshold': g['max_guided'],
                'passed': math.isfinite(guided) and guided <= g['max_guided'],
                'scope': 'Live weights; whole-latent relative squared velocity error vs the bf16 teacher, first %d held-out prompts x 20 teacher states, CFG 5.' % self.spec['train']['val_prompts']}

    def gate_passed(self):
        return self.gate_record()['passed']

    def audit_center(self, write=False):
        import torch
        sys.path.insert(0, str(self.p.root / 'scripts'))
        from learned_center import LearnedCenter
        c = LearnedCenter.from_checkpoint(torch.load(self.p.student / 'learned_center.pt', map_location='cpu'))
        masks = torch.load(self.p.selection / 'selection.pt', map_location='cpu')['variants'][self.spec['variant_key']]
        for side in ('q', 'k'):
            assert torch.equal(c.selected_pairs(side), masks['mask_' + side]), 'trained masks differ from the selection'
        if masks.get('mask_nope') is not None:
            assert torch.equal(c.selected_pairs('nope'), masks['mask_nope'])
        assert c.bypass_spec['branches'] == [int(x) for x in self.spec['bypass'].get('branches', [])] and c.token_rule == (self.spec['bypass'].get('tokens') or None), 'trained bypass rules differ from the spec'
        la, sa = int(c.layer_active.sum()), int(c.step_active.sum())
        ba = int(sum(bool(c.cell_active[:, :, b_].any()) for b_ in range(c.branches)))
        expect = {'variant': self.spec['variant_key'], 'selection_spec': c.selection_spec, 'selection_hash': self.spec['selection_hash'],
                  'bypass': c.bypass_spec, 'center_sha256': sha256(self.p.student / 'learned_center.pt'),
                  'exempt': {'layers': [int(i) for i in torch.nonzero(~c.layer_active).flatten()],
                             'steps': [int(i) for i in torch.nonzero(~c.step_active).flatten()]},
                  'selection': {'pairs_per_cell_q': float(c.selected_pairs('q').sum(-1).float().mean()),
                                'pairs_per_cell_k': float(c.selected_pairs('k').sum(-1).float().mean()),
                                'q_k_intersection_mean': float((c.selected_pairs('q') & c.selected_pairs('k')).sum(-1).float().mean()),
                                'nope_pairs_per_cell': float(c.selected_pairs('nope').sum(-1).float().mean()),
                                'wide_head': bool(c.is_wide)},
                  'calls': expected_calls(la, sa, c.uses_nope, ba), 'layers_active': la, 'steps_active': sa, 'branches_active': ba,
                  'complete_resume_step': self.latest_step(),
                  'scope': 'Final EMA learned_center.pt, all layer-step-branch-head cells; masks over the 64 rotary pairs per head.'}
        if write:
            self.write(self.p.run / 'center-audit.json', expect)
        return expect

    def stage_benchmark(self, mode):
        arm = self.spec['arms'][mode]
        run = self.p.bench[mode]
        if (run / 'summary.json').exists() and (run / 'paired-comparisons.json').exists() and (run / 'agreement' / 'complete.json').exists():
            return
        self.state('benchmark-' + mode, arm=arm)
        expect = None if self.dry else self.audit_center()
        config = json.loads((self.p.root / self.lab['benchmark_config']).read_text())
        config['modes'] = ['baseline', arm]
        config['center_policy'] = ('%s EMA (%s; %s): %s.' % (
            self.spec['name'], self.spec['choice'], bypass_text(self.spec),
            'trained means on the selected pairs' if mode == 'mean' else 'the same selected pairs zeroed at inference (zeros swap)'))
        cfg = run / 'config.json'
        if not self.dry:
            run.mkdir(parents=True, exist_ok=True)
            if cfg.exists():
                assert json.loads(cfg.read_text()) == config, 'benchmark config changed since the last attempt'
            else:
                self.write(cfg, config)
        common = ['--config', str(cfg), '--out', str(run)]
        gen = 'scripts/generate_learned_center.py'
        self.cpu('%s-link-baseline' % mode, [gen, *common, '--arm', 'baseline', '--link-baseline', self.lab['frozen_teacher']])
        if mode == 'delete' and not (self.p.screen_trained / 'complete.json').exists():
            self.wave([('screen-trained', [self.lab['python'], '-u', *screen_args(self.spec, self.p, student=True)])])
        resident = self.encoder_parity(mode, arm, cfg, config['prompts'][0]['id'])
        args = [*common, '--arm', arm, '--student', str(self.p.student), '--center-mode', mode] + (['--resident-text-encoder'] if resident else [])
        if self.dry or any(not (run / 'records' / p['id'] / arm / 'complete.json').exists() for p in config['prompts']):
            # per-prompt claim directories let the workers share the manifest
            self.wave([('%s-render-%d' % (mode, i), [self.lab['python'], '-u', gen, *args]) for i in range(self.spec['gpus']['benchmark_workers'])])
        if not self.dry:
            records = []
            for prompt in config['prompts']:
                rec = json.loads((run / 'records' / prompt['id'] / arm / 'complete.json').read_text())
                assert rec['prompt'] == prompt, prompt['id']
                bad = check_record(rec, expect, arm, mode)
                if bad:
                    raise RuntimeError('record audit failed for %s: %s' % (prompt['id'], bad))
                records.append({'prompt': prompt, 'arm': arm, 'center_mode': mode, 'selection': rec['environment']['selection'],
                                'exempt': rec['environment']['exempt'], 'intervention_calls': rec['intervention_calls'],
                                'nope_calls': rec['nope_calls'], 'video_sha256': rec['video_sha256']})
            self.write(run / 'selection-records.json', records)
        tasks = [('%s-%s' % (mode, d), [self.lab['python'], '-u', 'scripts/evaluate_center_benchmark.py', '--root', str(run), '--config', str(cfg),
                  '--source', self.lab['vbench'], '--dimension', d, '--batch-prompts', '100', '--reuse-baseline-from', self.lab['reuse_teacher_scores']])
                 for d in DIMENSIONS if self.dry or not (run / 'evaluation' / d / 'complete.json').exists()]
        if tasks:
            self.wave(tasks)
        self.cpu('%s-summary' % mode, ['scripts/summarize_center_benchmark.py', '--root', str(run), '--config', str(cfg),
                                       '--baseline-config', 'configs/center_benchmark_wan.json', '--extra-dimension', 'overall_consistency'])
        refs = [['baseline', str(run / 'summary.json')]]
        if mode == 'delete':
            refs.append([self.spec['arms']['mean'], str(self.p.bench['mean'] / 'summary.json')])
        refs += [[n, str(self.p.root / p)] for n, p in self.spec['references'][mode] if (self.p.root / p).exists() or self.dry]
        cmp = ['scripts/compare_distill_arms.py', '--candidate', arm, str(run / 'summary.json')]
        for n, p in refs:
            cmp += ['--reference', n, p]
        self.cpu('%s-paired' % mode, cmp + ['--out', str(run / 'paired-comparisons.json')])
        self.cpu('%s-agreement' % mode, ['scripts/measure_distill_agreement.py', '--root', str(run), '--config', str(cfg)])

    def encoder_parity(self, mode, arm, cfg, prompt):
        run = self.p.bench[mode]
        target = run / 'encoder-parity.json'
        if self.dry:
            print('GPU encoder parity on', prompt, '(offloaded vs resident text encoder)', flush=True)
            return False
        if target.exists():
            return json.loads(target.read_text())['used_resident_text_encoder']
        import torch
        roots = {False: self.p.run / ('parity-%s-offload' % mode), True: self.p.run / ('parity-%s-resident' % mode)}
        for resident, root in roots.items():
            if not (root / 'records' / prompt / arm / 'complete.json').exists():
                self.wave([('%s-parity-%s' % (mode, 'resident' if resident else 'offload'),
                            [self.lab['python'], '-u', 'scripts/generate_learned_center.py', '--config', str(cfg), '--out', str(root),
                             '--student', str(self.p.student), '--arm', arm, '--center-mode', mode, '--prompts', prompt]
                            + (['--resident-text-encoder'] if resident else []))])
        a, b = (r / 'records' / prompt / arm for r in roots.values())
        same_latent = torch.equal(*(torch.load(x / 'final-latents.pt', map_location='cpu') for x in (a, b)))
        same_video = json.loads((a / 'complete.json').read_text())['video_sha256'] == json.loads((b / 'complete.json').read_text())['video_sha256']
        resident = same_latent and same_video
        for rel in [Path('videos') / arm / (prompt + '.mp4'), *(Path('records') / prompt / arm / f for f in ('complete.json', 'final-latents.pt'))]:
            dest = run / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(roots[resident] / rel, dest)
        self.write(target, {'prompt_id': prompt, 'identical_final_latents': same_latent, 'identical_mp4_sha256': same_video,
                            'used_resident_text_encoder': resident})
        return resident

    # --- evidence, report, publication
    def collect(self):
        if self.dry:
            print('COLLECT ->', self.p.results, flush=True)
            return
        r = self.p.results
        r.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.p.selection / 'selection-report.json', r / 'selection-report.json')
        for n in ('selection-audit.json', 'center-audit.json', 'gate.json', 'ledger.tsv'):
            if (self.p.run / n).exists():
                shutil.copyfile(self.p.run / n, r / n)
        for src, dst in ((self.p.screen, r / 'screen'), (self.p.screen_trained, r / 'screen-trained')):
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
        tr = r / 'training'
        tr.mkdir(exist_ok=True)
        for n in ('args.json', 'history.jsonl', 'validation.jsonl'):
            if (self.p.train / n).exists():
                shutil.copyfile(self.p.train / n, tr / n)
        for f in self.p.run.glob('train-a*-request.json'):
            shutil.copyfile(f, tr / f.name)
        if (self.p.student / 'learned_center.pt').exists():
            (tr / 'final-ema.sha256').write_text(''.join('%s  student/%s\n' % (sha256(self.p.student / n), n) for n in
                ('learned_center.pt', 'transformer/config.json', 'transformer/diffusion_pytorch_model.safetensors')))
            self.training_summary(tr)
        for mode, run in self.p.bench.items():
            if not (run / 'summary.json').exists():
                continue
            dest = r / 'full100' / self.spec['arms'][mode]
            dest.mkdir(parents=True, exist_ok=True)
            raw = (run / 'summary.json').read_bytes()
            d = json.loads(raw)
            d['config'].pop('model', None)
            d['source_copy'] = {'raw_sha256': hashlib.sha256(raw).hexdigest(), 'omitted_fields': ['config.model']}
            (dest / 'summary.json').write_text(json.dumps(d, indent=2) + '\n')
            cfg = json.loads((run / 'config.json').read_text())
            cfg.pop('model', None)
            (dest / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
            for n in ('paired-comparisons.json', 'per-prompt-results.csv', 'selection-records.json', 'encoder-parity.json'):
                if (run / n).exists():
                    shutil.copyfile(run / n, dest / n)
            if (run / 'agreement').exists():
                shutil.copytree(run / 'agreement', dest / 'agreement', dirs_exist_ok=True)
        for p in r.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.md', '.tsv'):
                text = p.read_text(errors='replace')
                assert 'Wan-AI/' not in text and 'models--' not in text, 'model identifier in %s' % p
        self.write(r / 'budget.json', {'approved_gpu_hours': self.budget / 3600, 'gpu_hours_used': self.used() / 3600,
                                       'accounting': 'Allocated wall seconds x GPUs of every GPU job (ledger.tsv), including loading, validation, saves, failed attempts and scorers; CPU work excluded.'})
        (r / 'README.md').write_text(self.report())

    def training_summary(self, tr):
        import statistics
        rows = [json.loads(x) for x in (tr / 'history.jsonl').read_text().splitlines()] if (tr / 'history.jsonl').exists() else []
        peaks = []
        for f in sorted(self.p.run.glob('train-a*.log')):
            for line in f.read_text(errors='replace').splitlines():
                if line.startswith('{') and 'peak_cuda_allocated_bytes' in line:
                    peaks.append({'log': f.name, **json.loads(line)})
        summary = {'updates': rows[-1]['step'] if rows else None,
                   'median_seconds_per_update': statistics.median(r['optimizer_step_seconds'] for r in rows) if rows else None,
                   'time_scope': 'Median of logged rank-mean optimizer-update wall time, excluding validation and checkpoint I/O.',
                   'peaks': peaks, 'mu_norm': {s: [rows[0]['mu_norm_' + s], rows[-1]['mu_norm_' + s]] for s in ('q', 'k')} if rows else None,
                   'mu_scope': 'Live per-cell Euclidean norm over all 128 mean coordinates, equal average over all cells; first and last logged update.'}
        self.write(tr / 'summary.json', summary)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
            for name, path in [[self.spec['name'], tr / 'validation.jsonl'], *[[n, self.p.root / p] for n, p in self.spec['curves']]]:
                if Path(path).exists():
                    v = [json.loads(x) for x in Path(path).read_text().splitlines()]
                    ax.plot([x['step'] for x in v], [x['mean'][2] for x in v], marker='.', label=name)
            ax.set_xlabel('Optimizer update')
            ax.set_ylabel('Guided relative squared velocity error')
            ax.set_title('Live weights: first %d held-out prompts x 20 teacher states, CFG 5\nWhole-latent ratios, equal prompt-state average' % self.spec['train']['val_prompts'])
            ax.grid(alpha=.2)
            ax.legend()
            fig.savefig(tr / 'validation-curve.png', dpi=150)
            plt.close(fig)
        except Exception as e:      # the curve is evidence, not a gate
            print('curve skipped:', e, flush=True)

    def report(self):
        s, r = self.spec, self.p.results
        ex = bypass_text(s)
        lines = ['# Study %s: %s; %s' % (s['name'], s['choice'], ex), '',
                 s.get('description', ''), '',
                 'Selection spec (`scripts/pair_selection.py`, hash %s): `%s`. Bypass: `%s`.' % (s['selection_hash'], json.dumps(s['selection'], sort_keys=True), json.dumps(s['bypass'], sort_keys=True)), '',
                 'Generated by `scripts/center_study.py` from `%s`. Budget: %.3f of %.1f approved GPU-hours (allocated wall time x GPUs).' % (
                     s.get('_spec_path', 'the study spec'), self.used() / 3600, self.budget / 3600), '']
        def screen_table(path, title, means_note):
            if not (path / 'complete.json').exists():
                return []
            d = json.loads((path / 'complete.json').read_text())
            out = ['## ' + title, '', 'Whole-latent relative squared velocity error vs the bf16 teacher, equal mean over %d held-out prompts x 20 teacher entering states; guided = both CFG branches combined at 5; %s; %s.' % (len(d['ids']), ex, means_note), '',
                   '| Arm | Conditional | Unconditional | Guided |', '|---|---:|---:|---:|']
            out += ['| %s | %.6f | %.6f | %.6f |' % (k, *v['mean']) for k, v in d['arms'].items()]
            return out + ['']
        lines += screen_table(r / 'screen', 'Training-free screen', 'pooled calibration means')
        v = r / 'training' / 'validation.jsonl'
        if v.exists():
            rows = [json.loads(x) for x in v.read_text().splitlines()]
            lines += ['## Fine-tune', '', 'Recipe: %s. Validation: live weights, first %d held-out prompts x 20 teacher states, whole latent, CFG 5.' % (
                json.dumps(s['train']), s['train']['val_prompts']), '', '| Update | Guided |', '|---:|---:|']
            lines += ['| %d | %.6f |' % (x['step'], x['mean'][2]) for x in rows]
            lines += ['', '![Validation curve](training/validation-curve.png)', '']
        lines += screen_table(r / 'screen-trained', 'Trained-student screen', 'trained EMA means (the screen\'s generic scope string says pooled)')
        for mode in ('mean', 'delete'):
            arm = s['arms'][mode]
            d = r / 'full100' / arm
            if not (d / 'summary.json').exists():
                continue
            summ = json.loads((d / 'summary.json').read_text())['dimensions']
            comp = json.loads((d / 'paired-comparisons.json').read_text())['dimensions']
            keys = list(next(iter(comp.values())).keys())
            lines += ['## 100-prompt benchmark: `%s` (%s)' % (arm, 'trained means' if mode == 'mean' else 'same pairs zeroed at inference'), '',
                      'Frozen protocol: one original seed for each of 100 original prompts, 20 UniPC steps, shift 3, CFG 5, empty negative, 45 frames at 480x480; frozen teacher reused. VBench equal prompt means of VBench\'s within-video aggregation (aesthetics/imaging all 45 frames, text-video eight midpoint frames; imaging /100). Paired mean [pointwise 95%%] from 10,000 paired prompt resamples, NumPy seed 20261109; one fixed seed per prompt. Every record audited against the trained center (calls, masks, exemptions, hash).', '',
                      '| Dimension | Teacher | %s | %s |' % (arm, ' | '.join(k.replace(arm + '_minus_', '− ') for k in keys)),
                      '|---|---:|---:|%s' % ('---|' * len(keys))]
            for dim in DIMENSIONS:
                if dim in summ:
                    lines.append('| %s | %.4f | %.4f | %s |' % (dim, summ[dim]['means']['baseline'], summ[dim]['means'][arm],
                                 ' | '.join('%+.4f [%+.4f, %+.4f]' % (comp[dim][k]['mean'], *comp[dim][k]['ci95']) for k in keys)))
            ag = d / 'agreement' / 'complete.json'
            if ag.exists():
                a = json.loads(ag.read_text())['arms'][arm]
                lines += ['', 'Teacher agreement: final-latent relative L2 %.4f (step-20 state, all coordinates, prompt ratios averaged); frame PSNR %.3f dB over all aligned decoded frames.' % (
                    a['mean_final_latent_relative_l2'], a['mean_finite_frame_psnr_db'])]
            lines.append('')
        return '\n'.join(lines) + '\n'

    def publish(self, subject, body, terminal=None):
        pub = self.spec['publish']
        if self.dry or not pub['enabled']:
            print('PUBLISH', subject, '(terminal=%s)' % terminal, flush=True)
            return
        genv = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_AUTHOR_NAME': 'Claude', 'GIT_AUTHOR_EMAIL': 'noreply@anthropic.com',
                'GIT_COMMITTER_NAME': 'Claude', 'GIT_COMMITTER_EMAIL': 'noreply@anthropic.com',
                'GIT_SSH_COMMAND': 'ssh -o HostName=ssh.github.com -p443 -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15'}
        git = lambda *a: subprocess.run(['git', *a], cwd=self.p.root, env=genv, text=True, capture_output=True, check=True).stdout
        git('add', str(self.p.results.relative_to(self.p.root)))
        if not git('diff', '--cached', '--name-only').strip() and not terminal:
            return
        trailer = 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
        if terminal:
            cmd = ['python3', 'scripts/handoff.py', 'send', '--as', pub['as'], '--to', pub['to'], '--branch', pub['branch'],
                   '--status', terminal, '-m', subject, '-m', body, '--trailer', trailer, '--allow-empty']
            if pub.get('brief'):
                cmd += ['--brief', pub['brief']]
            if pub.get('ack'):
                cmd += ['--ack', pub['ack']]
            subprocess.run(cmd, cwd=self.p.root, env=genv, check=True)
            return
        git('commit', '-q', '-m', subject, '-m', body, '-m', trailer)
        try:
            git('push', '-q', 'origin', 'HEAD:' + pub['branch'])
        except subprocess.CalledProcessError:
            git('fetch', '-q', 'origin', pub['branch'])
            git('merge', '-q', '--no-edit', 'origin/' + pub['branch'])
            git('push', '-q', 'origin', 'HEAD:' + pub['branch'])

    def run(self, only=None):
        stages = ['select', 'screen', 'train', 'bench-mean', 'bench-delete', 'collect']
        todo = stages if not only or only == 'all' else [only]
        if not self.dry:
            self.p.run.mkdir(parents=True, exist_ok=True)
            self.write(self.p.run / 'spec.json', self.spec)
        name = self.spec['name']
        try:
            if 'select' in todo:
                self.stage_select()
            if 'screen' in todo:
                self.stage_screen()
                self.collect()
                self.publish('%s: selection and training-free screen' % name, 'center_study.py stages select + screen; see distill-results/%s.' % name)
            if 'train' in todo:
                self.stage_train()
                self.collect()
                self.publish('%s: %d-update fine-tune and center audit' % (name, self.spec['train']['steps']),
                             'center_study.py stage train; validation curve, audit and final EMA hash in distill-results/%s.' % name)
            for mode in ('mean', 'delete'):
                if 'bench-' + mode in todo:
                    self.stage_benchmark(mode)
                    self.collect()
                    if mode == 'mean':
                        self.publish('%s: 100-prompt %s benchmark' % (name, self.spec['arms'][mode]), 'center_study.py stage bench-mean.')
            if 'collect' in todo or todo == stages:
                self.collect()
            if todo == stages:
                self.state('complete')
                self.publish('%s complete: means and zeros benchmarks' % name,
                             'Study %s (%s; %s) run by scripts/center_study.py. Results with scopes: distill-results/%s/README.md. %.3f of %.1f approved GPU-hours.' % (
                                 name, self.spec['choice'], bypass_text(self.spec), name, self.used() / 3600, self.budget / 3600), terminal='done')
        except Exception as e:
            self.state('needs-attention', error=str(e), traceback=traceback.format_exc())
            if not self.dry:
                try:
                    self.collect()
                    self.publish('%s stopped: %s' % (name, str(e)[:90]), 'See distill-results/%s and out/study-%s/run/status.json. %s' % (name, name, e),
                                 terminal='question' if str(e).startswith('ASK') else 'interrupted')
                except Exception:
                    self.state('needs-attention-publication-failed', error=str(e), traceback=traceback.format_exc())
            raise


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spec')
    ap.add_argument('--stage', choices=('all', 'select', 'screen', 'train', 'bench-mean', 'bench-delete', 'collect'), default='all')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and every command; touch nothing')
    a = ap.parse_args()
    spec = load_spec(a.spec)
    spec['_spec_path'] = a.spec
    study = Study(spec, dry_run=a.dry_run)
    if not a.dry_run:
        study.p.run.mkdir(parents=True, exist_ok=True)
        lock = (study.p.run / 'runner.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('another center_study runner holds %s' % (study.p.run / 'runner.lock'))
    study.run(a.stage)


if __name__ == '__main__':
    main()
