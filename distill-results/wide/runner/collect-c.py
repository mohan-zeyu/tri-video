"""Collect C evidence for distill-results/wide/training (CPU only)."""
import glob, hashlib, json, math, shutil, statistics, sys, torch
from pathlib import Path
sys.path.insert(0, 'scripts')
from learned_center import LearnedCenter
LOG = Path('out/distill-wide-run'); TRAIN = Path('out/distill-wide32'); OUT = LOG / 'collect-c'; OUT.mkdir(exist_ok=True)
for n in ('history.jsonl', 'validation.jsonl'):
    shutil.copyfile(TRAIN / n, OUT / n)
for p in sorted(LOG.glob('train-c0*-args.json')) + sorted(LOG.glob('train-c0*-status.json')) + sorted(LOG.glob('train-c0*-request.json')) + [LOG / 'gate-1000.json', LOG / 'budget-approval.json']:
    shutil.copyfile(p, OUT / p.name)
shutil.copyfile(TRAIN / 'args.json', OUT / 'args.json')
rows = [json.loads(x) for x in (OUT / 'history.jsonl').read_text().splitlines()]
vals = [json.loads(x) for x in (OUT / 'validation.jsonl').read_text().splitlines()]
assert rows[-1]['step'] == vals[-1]['step'] == 5000 and [v['step'] for v in vals] == list(range(0, 5001, 250))
assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float)))
assert all(math.isfinite(v) for r in vals for v in r['mean'])
steps = [r['step'] for r in rows]; assert steps == sorted(set(steps)), 'history has duplicate/unsorted steps'
# mask/width audit of the saved final EMA center and the full resume state
saved = torch.load(TRAIN / 'student/learned_center.pt', map_location='cpu'); c = LearnedCenter.from_checkpoint(saved)
sel = torch.load('out/distill-selection-v4/selection.pt', map_location='cpu')['variants']['wide']
assert saved.get('wide') is True and c.is_wide and c.content_pairs == 32
assert c.layer_active.all() and c.step_active.all()
assert all(torch.equal(c.selected_pairs(s), sel['mask_' + s]) for s in ('q', 'k')) and torch.equal(c.selected_pairs('nope'), sel['mask_nope'])
assert (c.selected_pairs('q').sum(-1) == 64).all() and (c.selected_pairs('nope').sum(-1) == 32).all()
latest = json.loads((TRAIN / 'checkpoint/latest.json').read_text()); assert latest['step'] == 5000
cp = TRAIN / 'checkpoint' / latest['directory']; cs = torch.load(cp / 'center.pt', map_location='cpu'); assert cs['step'] == 5000
assert all((cp / p).exists() for p in ('model/.metadata', 'rng-rank00.pt', 'rng-rank01.pt'))
audit = {'wide': True, 'content_pairs_per_cell': 32, 'means_pairs_per_cell': 64, 'cells': 14400, 'layers_active': 30, 'steps_active': 20,
         'exempt_layers': [], 'exempt_steps': [], 'masks_tensor_equal_to_selection_v4_wide': True, 'complete_resume_step': 5000,
         'expected_calls_per_video': {'replacement': 2400, 'processor': 1200},
         'scope': 'Final EMA learned_center.pt, all 30 x 20 x 2 x 12 layer-step-branch-head cells; masks over the 64 rotary pairs per head; no exemptions.'}
json.dump(audit, open(OUT / 'mask-width-audit.json', 'w'), indent=1)
peaks = []
for p in sorted(LOG.glob('train-c0*.log')):
    if p.name.endswith('-worker.log'): continue
    for line in p.read_text().splitlines():
        if line.startswith('{') and 'peak_cuda_allocated_bytes' in line:
            peaks.append({'log': p.name, **json.loads(line)})
gpu = {p.name: json.loads(p.read_text()).get('gpu_seconds') for p in sorted(LOG.glob('train-c0*-status.json'))}
screen_b = json.loads((LOG / 'screen-b-status.json').read_text())['gpu_seconds']
summary = {'updates': 5000, 'attempts': ['train-c01 (teacher weights -> update 1000, --stop-after 1000 for the gate)', 'train-c02 (--resume from the complete update-1000 save -> 5000)'],
  'median_logged_rank_mean_seconds_per_update': statistics.median(r['optimizer_step_seconds'] for r in rows),
  'time_scope': 'Median over logged updates (every 10th) of the rank-mean optimizer-update wall time, excluding validation/export/checkpoint I/O; two ranks x accumulation 2.',
  'reported_peaks': peaks, 'peak_scope': 'torch.cuda.max_memory_allocated per rank per invocation (allocator bytes, not reserved/nvidia-smi).',
  'mu_norm_drift': {s: {'first_logged': rows[0]['mu_norm_' + s], 'last': rows[-1]['mu_norm_' + s], 'percent': 100 * (rows[-1]['mu_norm_' + s] / rows[0]['mu_norm_' + s] - 1)} for s in ('q', 'k')},
  'mu_scope': 'Live (not EMA) per-cell Euclidean norm over all 128 mean coordinates, equal average over all 14400 cells; first value at logged update 10; no token/prompt average.',
  'allocated_gpu_seconds': {'screen_b': screen_b, **gpu}, 'allocated_gpu_hours_A_to_C': (screen_b + sum(gpu.values())) / 3600,
  'validation_scope': 'Live weights; whole-latent relative squared velocity error vs bf16 teacher, equal mean over the first 8 held-out prompts x 20 teacher entering states; columns cond, uncond, guided (CFG 5); nothing exempt.'}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
manifest = []
for n in ('learned_center.pt', 'transformer/config.json', 'transformer/diffusion_pytorch_model.safetensors'):
    h = hashlib.sha256()
    with open(TRAIN / 'student' / n, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    manifest.append(h.hexdigest() + '  student/' + n)
(OUT / 'final-ema.sha256').write_text('\n'.join(manifest) + '\n')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
for name, path in [('wide32, nothing exempt', OUT / 'validation.jsonl'), ('shared22, layer 0 / step 1 exempt', Path('distill-results/shared22/validation.jsonl')),
                   ('split42 (NoPE), layer 0 / step 1 exempt', Path('distill-results/nope/training/validation.jsonl'))]:
    r = [json.loads(x) for x in path.read_text().splitlines()]
    ax.plot([x['step'] for x in r], [x['mean'][2] for x in r], marker='.', label=name)
ax.set_yscale('log'); ax.set_xlabel('Optimizer update'); ax.set_ylabel('Guided relative squared velocity error (log)')
ax.set_title('Live weights: first 8 held-out prompts x 20 teacher states, CFG 5\nWhole-latent ratios, equal prompt-state average')
ax.grid(alpha=.2, which='both'); ax.legend(); fig.savefig(OUT / 'validation-curve.png', dpi=150); plt.close(fig)
print(json.dumps({k: summary[k] for k in ('median_logged_rank_mean_seconds_per_update', 'mu_norm_drift', 'allocated_gpu_hours_A_to_C')}, indent=1))
print([(p['log'], p['rank'], round(p['peak_cuda_allocated_bytes'] / 2**30, 2)) for p in peaks]); print(manifest)
