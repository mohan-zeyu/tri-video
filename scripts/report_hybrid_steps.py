#!/usr/bin/env python3
"""Tables for distill-results/hybrid-steps/README.md from the collected evidence (scripts/hybrid_steps.py collect).

Prints markdown: the VBench table (equal prompt means of VBench's within-video aggregation) with paired differences
[pointwise 95%] from 10,000 paired prompt resamples (NumPy seed 20261109), the agreement table, and the audit lines.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'distill-results' / 'hybrid-steps'
S32 = ROOT / 'distill-results' / 'shared32' / 'full100' / 's32_means'
ARMS = ['s32_teacher3', 's32_student3']
DIMS = ['overall_consistency', 'aesthetic_quality', 'imaging_quality', 'dynamic_degree', 'subject_consistency',
        'background_consistency', 'motion_smoothness']


def load(p):
    return json.loads(Path(p).read_text())


def ci(d):
    return '%+.4f [%+.4f, %+.4f]' % (d['mean'], d['ci95'][0], d['ci95'][1])


def main():
    arms = [a for a in ARMS if (R / a / 'summary.json').exists()]
    summ = {a: load(R / a / 'summary.json') for a in arms}
    paired = {a: load(R / a / 'paired-comparisons.json') for a in arms}
    s32 = load(S32 / 'summary.json')
    cols = ['baseline', 's32_means'] + [o for o in ARMS if o in arms]
    lines = ['| Dimension | Teacher | s32_means | ' + ' | '.join(arms) + ' | ' +
             ' | '.join('%s − %s' % (a, r) for a in arms for r in ['baseline', 's32_means'] + [o for o in arms if o != a and ARMS.index(o) > ARMS.index(a)]) + ' |']
    lines.append('|' + '---|' * (lines[0].count('|') - 1))
    for d in DIMS:
        teacher = summ[arms[0]]['dimensions'][d]['means']['baseline']
        row = ['%s' % d, '%.4f' % teacher, '%.4f' % s32['dimensions'][d]['means']['s32_means']]
        row += ['%.4f' % summ[a]['dimensions'][d]['means'][a] for a in arms]
        for a in arms:
            for r in ['baseline', 's32_means'] + [o for o in arms if o != a and ARMS.index(o) > ARMS.index(a)]:
                row.append(ci(paired[a]['dimensions'][d]['%s_minus_%s' % (a, r)]))
        lines.append('| ' + ' | '.join(row) + ' |')
    print('\n'.join(lines))
    print()
    agree = {a: load(R / a / 'agreement' / 'complete.json')['arms'][a] for a in arms}
    agree['s32_means'] = load(S32 / 'agreement' / 'complete.json')['arms']['s32_means']
    print('| Arm | Final-latent relative L2 | Frame PSNR (dB) | Identical frames | Prompts | Frames |')
    print('|---|---:|---:|---:|---:|---:|')
    for a in ['s32_means'] + arms:
        g = agree[a]
        print('| %s | %.4f | %.3f | %d | %d | %d |' % (a, g['mean_final_latent_relative_l2'], g['mean_finite_frame_psnr_db'], g['identical_frames'], g['prompts'], g['frames']))
    print()
    for a in arms:
        recs = load(R / a / 'selection-records.json')
        calls = sorted({r['intervention_calls'] for r in recs})
        tf = sorted({r['teacher_forwards'] for r in recs})
        steps = sorted({tuple(r['teacher_steps']) for r in recs})
        print('%s: %d records; teacher steps %s; teacher forwards %s; replacement calls %s; finite latents %s; frames %s' % (
            a, len(recs), [list(s) for s in steps], tf, calls, sorted({r['finite_step_latents'] for r in recs}), sorted({tuple(r['frames']) for r in recs})))
        dyn = summ[a]['dimensions']['dynamic_degree']['means']
        print('%s dynamic videos: %d of 100 (teacher %d)' % (a, round(100 * dyn[a]), round(100 * dyn['baseline'])))
    json.dump({a: {'dims': {d: summ[a]['dimensions'][d]['means'][a] for d in DIMS}, 'agreement': agree[a]} for a in arms}, sys.stdout)
    print()


if __name__ == '__main__':
    main()
