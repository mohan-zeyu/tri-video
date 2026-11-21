src = open('benchmark-noexempt.py').read()
def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (old, n)
    src = src.replace(old, new)
sub('"""Budgeted D/E benchmark driver for the authorized no-exemption study; run after C."""',
    '"""Budgeted D/E benchmark driver for the widened-head study (DISTILL_WIDE_PROMPT.md); run after C.\n'
    'Derived from the no-exemption benchmark.py by asserted substitutions (make-benchmark.py)."""')
sub("STUDENT=ROOT/'out/distill-s22-all/student'", "STUDENT=ROOT/'out/distill-wide32/student'")
sub("LOGS=ROOT/'out/distill-noexempt-run'", "LOGS=ROOT/'out/distill-wide-run'")
sub("RUN=ROOT/('out/learned-center-wan-s22-all' + ('-zeros' if PHASE=='E' else ''))", "RUN=ROOT/('out/learned-center-wan-wide32' + ('-zeros' if PHASE=='E' else ''))")
sub("ARM='s22_all_means' if PHASE=='D' else 's22_all_zeros'", "ARM='wide32' if PHASE=='D' else 'wide32_zeros'")
sub("ASSIGNMENT_ORDER=[1,0,6,2,3,5]", "ASSIGNMENT_ORDER=[2,3,1,6,0,5]")
sub("if p.name.startswith(('screen-a','train-b')))", "if p.name.startswith(('screen-b','train-c')))")
sub("state=read(ROOT/'out/distill-s22-all/checkpoint/latest.json')\n    assert state['step']==2500, 'C needs its complete resumable update-2500 save'",
    "state=read(ROOT/'out/distill-wide32/checkpoint/latest.json')\n    assert state['step']==5000, 'D/E need the complete resumable update-5000 save'")
sub("rows=[json.loads(line) for line in (ROOT/'out/distill-s22-all/validation.jsonl').read_text().splitlines()]\n    assert rows[-1]['step']==2500",
    "rows=[json.loads(line) for line in (ROOT/'out/distill-wide32/validation.jsonl').read_text().splitlines()]\n    assert rows[-1]['step']==5000")
sub("original=torch.load(ROOT/'out/distill-selection/selection.pt',map_location='cpu')['variants']['shared']\n    for side in ('q','k'):\n        assert torch.equal(center.selected_pairs(side),original['mask_'+side])",
    "original=torch.load(ROOT/'out/distill-selection-v4/selection.pt',map_location='cpu')['variants']['wide']\n    assert center.is_wide and center.content_pairs==32\n    for side in ('q','k'):\n        assert torch.equal(center.selected_pairs(side),original['mask_'+side])\n    assert torch.equal(center.selected_pairs('nope'),original['mask_nope'])\n    assert (center.selected_pairs('q').sum(-1)==64).all() and (center.selected_pairs('nope').sum(-1)==32).all()")
sub("config['center_policy']='Shared22 EMA with no exempt layers or denoising steps: 22 selected Q/K pairs use '+MODE+' after full-width RMSNorm and before RoPE in all 30 layers and 20 steps.'",
    "config['center_policy']='Widened-head EMA (wide32) with no exempt layers or denoising steps: per head Q/K = [RoPE(128-dim learned mean) | 32 unrotated post-QK-norm content pairs], logit scale 1/sqrt(128), in all 30 layers and 20 steps; '+('position channel active' if MODE=='mean' else 'position channel zeroed (content-only attention)')+'.'")
sub("screen=ROOT/'out/distill-noexempt/screen-trained'", "screen=ROOT/'out/distill-selection-v4/screen-trained'")
sub("assert rec['nope_calls']==rec['expected_nope_calls']==0", "assert rec['nope_calls']==rec['expected_nope_calls']==1200")
sub("assert sel['pairs_per_cell_q']==sel['pairs_per_cell_k']==sel['q_k_intersection_mean']==22.0\n        assert sel['nope_pairs_per_cell']==0.0 and not sel['default_high_masks']",
    "assert sel['pairs_per_cell_q']==sel['pairs_per_cell_k']==sel['q_k_intersection_mean']==64.0\n        assert sel['nope_pairs_per_cell']==32.0 and not sel['default_high_masks']\n        assert sel['wide_head'] is True and sel['content_pairs']==32")
sub("'intervention_calls':2400,'nope_calls':0,", "'intervention_calls':2400,'processor_calls':1200,")
sub("""                 ('shared22_delete',ROOT/'distill-results/shared22-delete/full100/summary.json')]""",
    """                 ('shared22_delete',ROOT/'distill-results/shared22-delete/full100/summary.json'),
                 ('split42',ROOT/'distill-results/nope/full100/split42/summary.json')]""")
sub("refs += [('s22_all_means',ROOT/'out/learned-center-wan-s22-all/summary.json'),('s22_zeros',ROOT/'distill-results/swap/full100/summary.json')]",
    "refs += [('wide32',ROOT/'out/learned-center-wan-wide32/summary.json')]")
sub('raise RuntimeError("ASK: no-exemption study reached its approved GPU-hour budget")', 'raise RuntimeError("ASK: widened-head study reached its approved GPU-hour budget")', 0)
sub('failure = "ASK: no-exemption study reached its approved GPU-hour budget"', 'failure = "ASK: widened-head study reached its approved GPU-hour budget"')
open('benchmark.py', 'w').write(src)
