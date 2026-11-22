"""Copy a widened-head 100-prompt benchmark into distill-results form (CPU). Usage: collect-bench.py D|E"""
import hashlib, json, shutil, sys
from pathlib import Path
phase = sys.argv[1]; arm = 'wide32' if phase == 'D' else 'wide32_zeros'
src = Path('out/learned-center-wan-wide32' + ('-zeros' if phase == 'E' else '')); LOG = Path('out/distill-wide-run')
dest = LOG / 'collect' / 'full100' / arm; dest.mkdir(parents=True, exist_ok=True)
raw = (src / 'summary.json').read_bytes(); d = json.loads(raw); d['config'].pop('model', None)
d['source_copy'] = {'raw_sha256': hashlib.sha256(raw).hexdigest(), 'omitted_fields': ['config.model'],
    'selection_note': 'Legacy high_pairs field unused; the actual masks (64 means pairs, 32 content pairs per cell, widened head) come from the final EMA learned_center.pt and selection-records.json. No exempt layers or steps.'}
(dest / 'summary.json').write_text(json.dumps(d, indent=2) + '\n')
for name in ('paired-comparisons.json', 'per-prompt-results.csv', 'selection-records.json', 'encoder-parity.json'):
    shutil.copyfile(src / name, dest / name)
shutil.copytree(src / 'agreement', dest / 'agreement', dirs_exist_ok=True)
for name in ('phase-' + phase.lower() + '-gpu.tsv', 'phase-' + phase.lower() + '-summary.json'):
    shutil.copyfile(LOG / name, dest / name)
prov = []
for dim in d['dimensions']:
    for p in sorted((src / 'evaluation' / dim).glob('batch-*-results.json')):
        b = json.loads(p.read_text()); prov.append({'dimension': dim, **{k: b[k] for k in ('source_file', 'source_sha256', 'vbench_commit', 'reused_baseline_from') if k in b}})
(dest / 'scorer-provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
for p in dest.rglob('*'):
    if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.tsv'):
        t = p.read_text(); assert 'Wan-AI/' not in t and 'models--' not in t, p
print('collected', dest, sum(1 for _ in dest.rglob('*')))
