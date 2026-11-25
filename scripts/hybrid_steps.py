#!/usr/bin/env python3
"""Inference-only hybrid benchmark: the frozen teacher on some denoising steps, the shared32 EMA student on the rest.

    python scripts/hybrid_steps.py --stage verify            # shared32 EMA checksums against the audited hashes
    python scripts/hybrid_steps.py --stage parity            # the two parity checks on two prompts (must pass first)
    python scripts/hybrid_steps.py --stage arm --arm s32_teacher3
    python scripts/hybrid_steps.py --stage arm --arm s32_student3
    python scripts/hybrid_steps.py --stage collect           # paired comparisons and evidence -> distill-results/hybrid-steps

Arms (1-based steps; generate_learned_center.py --teacher-steps): s32_teacher3 = teacher on steps 1-3, student on
4-20; s32_student3 = student on 1-3, teacher on 4-20. Parity checks on the first two benchmark prompts: teacher on all
20 steps through the hybrid path must reproduce the frozen baseline final latents, and teacher on no step must
reproduce the s32_means final latents (torch.equal, as the encoder-parity probes; MP4 hashes reported as well).

Frozen protocol, GPU leasing (idle GPUs only, exclusive locks shared with the other runners, at most two at a
time), the allocation ledger and the benchmark scripts are those of scripts/center_study.py. Work directory
out/study-hybrid-steps/ in the lab checkout; nothing outside it is written. Publication is done by hand.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from center_study import DEFAULTS, DIMENSIONS, Study, check_record, expected_calls, merge, sha256

ROOT = Path(__file__).resolve().parents[1]
S32 = Path('/mnt/data/liuzeyu/tri-video-s32-31ef4d2/out/study-shared32')
S32_RESULTS = ROOT / 'distill-results' / 'shared32'
STEPS = 20
ARMS = {'s32_teacher3': [1, 2, 3], 's32_student3': list(range(4, STEPS + 1))}
PARITY = {'parity_teacher20': list(range(1, STEPS + 1)), 'parity_teacher0': []}
PARITY_PROMPTS = 2


def spec(budget_hours):
    s = merge(DEFAULTS, {'name': 'hybrid-steps', 'approved_gpu_hours': budget_hours,
                         'gpus': {'candidates': [0, 1, 2, 3, 4, 5, 6], 'benchmark_workers': 2},
                         'publish': {'enabled': False}})
    s['arms'] = {}
    return s


def student_dir():
    return S32 / 'train' / 'student'


def center_audit():
    return json.loads((S32_RESULTS / 'center-audit.json').read_text())


def mp4_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Hybrid(Study):
    def __init__(self, budget_hours):
        super().__init__(spec(budget_hours))
        self.base = json.loads((self.p.root / self.lab['benchmark_config']).read_text())

    def config_for(self, arm, teacher_steps, root):
        config = dict(self.base)
        config['modes'] = ['baseline', arm]
        config['center_policy'] = ('shared32 EMA (shared, 32 means pairs per head, nothing exempt), trained means; inference-only hybrid: '
                                   'denoising steps %s (1-based) run by the frozen teacher with no replacement, the other steps by the '
                                   'student, one sampler over all 20 steps' % (teacher_steps or 'none'))
        cfg = root / 'config.json'
        root.mkdir(parents=True, exist_ok=True)
        if cfg.exists():
            assert json.loads(cfg.read_text()) == config, 'benchmark config changed since the last attempt'
        else:
            self.write(cfg, config)
        return cfg, config

    def gen_args(self, cfg, root, arm, teacher_steps):
        return [self.lab['python'], '-u', 'scripts/generate_learned_center.py', '--config', str(cfg), '--out', str(root), '--arm', arm,
                '--student', str(student_dir()), '--center-mode', 'mean', '--resident-text-encoder', '--teacher-steps', *map(str, teacher_steps)]

    def expect(self, teacher_steps):
        e = center_audit()
        assert e['center_sha256'] == sha256(student_dir() / 'learned_center.pt')
        exempt_steps = set(e['exempt']['steps'])
        student_active = sum(1 for i in range(STEPS) if i + 1 not in teacher_steps and i not in exempt_steps)
        e['calls'] = expected_calls(e['layers_active'], student_active, False)
        return e

    def audit(self, rec, arm, teacher_steps):
        bad = check_record(rec, self.expect(teacher_steps), arm, 'mean')
        if rec.get('teacher_steps') != sorted(teacher_steps) or rec['environment'].get('teacher_steps') != sorted(teacher_steps):
            bad.append('teacher steps')
        if not rec.get('teacher_forwards') == rec.get('expected_teacher_forwards') == 2 * len(teacher_steps):
            bad.append('teacher forwards')
        if rec.get('student_forwards') != 2 * (STEPS - len(teacher_steps)):
            bad.append('student forwards')
        if rec.get('denoiser_by_step') != ['teacher' if i + 1 in teacher_steps else 'student' for i in range(STEPS)]:
            bad.append('denoiser by step')
        return bad

    # --- stages
    def verify(self):
        lines = (S32_RESULTS / 'training' / 'final-ema.sha256').read_text().split('\n')
        rows = []
        for line in filter(None, lines):
            want, rel = line.split('  ')
            got = sha256(S32 / 'train' / rel)
            rows.append({'file': rel, 'audited_sha256': want, 'sha256': got, 'match': got == want})
        center = center_audit()['center_sha256'] == sha256(student_dir() / 'learned_center.pt')
        result = {'files': rows, 'center_audit_match': center, 'ok': all(r['match'] for r in rows) and center,
                  'scope': 'SHA-256 of the shared32 final EMA (learned_center.pt and bf16 transformer) on the lab host against '
                           'distill-results/shared32/training/final-ema.sha256 and center-audit.json'}
        self.write(self.p.run / 'ema-verify.json', result)
        if not result['ok']:
            raise RuntimeError('ASK: shared32 EMA checksum mismatch %s' % rows)
        print('EMA verified', flush=True)

    def parity(self):
        prompts = [p['id'] for p in self.base['prompts'][:PARITY_PROMPTS]]
        tasks, roots = [], {}
        for name, steps in PARITY.items():
            root = self.p.work / name
            cfg, _ = self.config_for(name, steps, root)
            roots[name] = root
            if any(not (root / 'records' / pid / name / 'complete.json').exists() for pid in prompts):
                tasks.append((name, self.gen_args(cfg, root, name, steps) + ['--prompts', *prompts]))
        if tasks:
            self.wave(tasks)
        import torch
        refs = {'parity_teacher20': (Path(self.lab['frozen_teacher']), 'baseline', 'frozen teacher baseline (out/center-benchmark-wan-20261109)'),
                'parity_teacher0': (S32 / 'bench-mean', 's32_means', 's32_means benchmark (shared32 bench-mean)')}
        out = {'prompts': prompts, 'tolerance': 'torch.equal on the final latent (all coordinates of the step-20 state), as the encoder-parity probes; MP4 SHA-256 reported',
               'checks': {}}
        ok = True
        for name, steps in PARITY.items():
            ref_root, ref_arm, ref_label = refs[name]
            rows = []
            for pid in prompts:
                mine = roots[name] / 'records' / pid / name
                theirs = ref_root / 'records' / pid / ref_arm
                rec, ref = (json.loads((d / 'complete.json').read_text()) for d in (mine, theirs))
                a, b = (torch.load(d / 'final-latents.pt', map_location='cpu') for d in (mine, theirs))
                assert rec['prompt'] == ref['prompt'], pid
                bad = self.audit(rec, name, steps)
                same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
                diff = float((a.float() - b.float()).norm() / b.float().norm()) if a.shape == b.shape else None
                video = mp4_sha(roots[name] / 'videos' / name / (pid + '.mp4')) == mp4_sha(ref_root / 'videos' / ref_arm / (pid + '.mp4'))
                rows.append({'prompt_id': pid, 'seed': rec['prompt']['seed'], 'identical_final_latents': same, 'relative_l2': diff,
                             'dtype': [str(a.dtype), str(b.dtype)], 'identical_mp4_sha256': video, 'record_audit_problems': bad,
                             'teacher_forwards': rec['teacher_forwards'], 'intervention_calls': rec['intervention_calls']})
                ok = ok and same and not bad
            out['checks'][name] = {'teacher_steps': steps, 'reference': ref_label, 'prompts': rows,
                                   'passed': all(r['identical_final_latents'] and not r['record_audit_problems'] for r in rows)}
        out['passed'] = ok
        self.write(self.p.work / 'parity.json', out)
        print(json.dumps(out, indent=1), flush=True)
        if not ok:
            raise RuntimeError('ASK: parity check failed, see %s' % (self.p.work / 'parity.json'))

    def arm(self, arm):
        parity = self.p.work / 'parity.json'
        if not parity.exists() or not json.loads(parity.read_text())['passed']:
            raise RuntimeError('parity checks have not passed')
        steps = ARMS[arm]
        root = self.p.work / ('bench-' + arm)
        cfg, config = self.config_for(arm, steps, root)
        gen = 'scripts/generate_learned_center.py'
        self.cpu('%s-link-baseline' % arm, [gen, '--config', str(cfg), '--out', str(root), '--arm', 'baseline', '--link-baseline', self.lab['frozen_teacher']])
        if any(not (root / 'records' / p['id'] / arm / 'complete.json').exists() for p in config['prompts']):
            self.wave([('%s-render-%d' % (arm, i), self.gen_args(cfg, root, arm, steps)) for i in range(self.spec['gpus']['benchmark_workers'])])
        records = []
        for prompt in config['prompts']:
            rec = json.loads((root / 'records' / prompt['id'] / arm / 'complete.json').read_text())
            assert rec['prompt'] == prompt, prompt['id']
            bad = self.audit(rec, arm, steps)
            if bad:
                raise RuntimeError('record audit failed for %s: %s' % (prompt['id'], bad))
            records.append({'prompt': prompt, 'arm': arm, 'center_mode': 'mean', 'selection': rec['environment']['selection'],
                            'exempt': rec['environment']['exempt'], 'intervention_calls': rec['intervention_calls'],
                            'nope_calls': rec['nope_calls'], 'teacher_steps': rec['teacher_steps'], 'teacher_forwards': rec['teacher_forwards'],
                            'student_forwards': rec['student_forwards'], 'denoiser_by_step': rec['denoiser_by_step'],
                            'finite_step_latents': rec['finite_step_latents'], 'frames': rec['frames'], 'video_sha256': rec['video_sha256']})
        self.write(root / 'selection-records.json', records)
        tasks = [('%s-%s' % (arm, d), [self.lab['python'], '-u', 'scripts/evaluate_center_benchmark.py', '--root', str(root), '--config', str(cfg),
                  '--source', self.lab['vbench'], '--dimension', d, '--batch-prompts', '100', '--reuse-baseline-from', self.lab['reuse_teacher_scores']])
                 for d in DIMENSIONS if not (root / 'evaluation' / d / 'complete.json').exists()]
        if tasks:
            self.wave(tasks)
        self.cpu('%s-summary' % arm, ['scripts/summarize_center_benchmark.py', '--root', str(root), '--config', str(cfg),
                                      '--baseline-config', 'configs/center_benchmark_wan.json', '--extra-dimension', 'overall_consistency'])
        self.compare(arm)
        self.cpu('%s-agreement' % arm, ['scripts/measure_distill_agreement.py', '--root', str(root), '--config', str(cfg)])

    def compare(self, arm):
        root = self.p.work / ('bench-' + arm)
        refs = [['baseline', root / 'summary.json'], ['s32_means', S32_RESULTS / 'full100' / 's32_means' / 'summary.json']]
        refs += [[other, self.p.work / ('bench-' + other) / 'summary.json'] for other in ARMS
                 if other != arm and (self.p.work / ('bench-' + other) / 'summary.json').exists()]
        cmd = ['scripts/compare_distill_arms.py', '--candidate', arm, str(root / 'summary.json')]
        for n, p in refs:
            cmd += ['--reference', n, str(p)]
        self.cpu('%s-paired' % arm, cmd + ['--out', str(root / 'paired-comparisons.json')])

    def collect(self):
        r = ROOT / 'distill-results' / 'hybrid-steps'
        r.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            if (self.p.work / ('bench-' + arm) / 'summary.json').exists():
                self.compare(arm)
        for n, src in (('parity.json', self.p.work / 'parity.json'), ('ema-verify.json', self.p.run / 'ema-verify.json'),
                       ('ledger.tsv', self.p.ledger)):
            if src.exists():
                shutil.copyfile(src, r / n)
        for name in PARITY:
            src = self.p.work / name / 'records'
            if src.exists():
                for rec in src.glob('*/%s/complete.json' % name):
                    dest = r / 'parity' / name / (rec.parent.parent.name + '.json')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(rec, dest)
        for arm in ARMS:
            run = self.p.work / ('bench-' + arm)
            if not (run / 'summary.json').exists():
                continue
            dest = r / arm
            dest.mkdir(parents=True, exist_ok=True)
            raw = (run / 'summary.json').read_bytes()
            d = json.loads(raw)
            d['config'].pop('model', None)
            d['source_copy'] = {'raw_sha256': hashlib.sha256(raw).hexdigest(), 'omitted_fields': ['config.model']}
            (dest / 'summary.json').write_text(json.dumps(d, indent=2) + '\n')
            cfg = json.loads((run / 'config.json').read_text())
            cfg.pop('model', None)
            (dest / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
            for n in ('paired-comparisons.json', 'per-prompt-results.csv', 'selection-records.json'):
                if (run / n).exists():
                    shutil.copyfile(run / n, dest / n)
            if (run / 'agreement').exists():
                shutil.copytree(run / 'agreement', dest / 'agreement', dirs_exist_ok=True)
        self.write(r / 'budget.json', {'gpu_hours_used': self.used() / 3600, 'guard_gpu_hours': self.budget / 3600,
                                       'accounting': 'Allocated wall seconds x GPUs of every GPU job (ledger.tsv), including loading, parity, '
                                                     'rendering and scorers; CPU work excluded.'})
        for p in r.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.md', '.tsv'):
                text = p.read_text(errors='replace')
                assert 'Wan-AI/' not in text and 'models--' not in text, 'model identifier in %s' % p
        print('COLLECTED', r, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=('verify', 'parity', 'arm', 'collect'), required=True)
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--budget-gpu-hours', type=float, default=3.0, help='stop with ASK when this study allocation is reached')
    a = ap.parse_args()
    h = Hybrid(a.budget_gpu_hours)
    h.p.run.mkdir(parents=True, exist_ok=True)
    lock = (h.p.run / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('another hybrid_steps runner holds %s' % (h.p.run / 'runner.lock'))
    if a.stage == 'verify':
        h.verify()
    elif a.stage == 'parity':
        h.verify()
        h.parity()
    elif a.stage == 'arm':
        if not a.arm:
            sys.exit('--stage arm needs --arm')
        h.verify()
        h.arm(a.arm)
    else:
        h.collect()
    print('USED GPU-hours %.4f' % (h.used() / 3600), flush=True)


if __name__ == '__main__':
    main()
