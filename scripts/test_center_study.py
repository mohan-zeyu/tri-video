"""CPU tests for scripts/center_study.py (standard library only): python3 scripts/test_center_study.py"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import center_study as cs

SPEC = Path(__file__).resolve().parents[1] / 'configs' / 'studies' / 'shared32.json'


def spec(**over):
    s = json.loads(SPEC.read_text())
    s.update(over)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(s, f)
    return cs.load_spec(f.name)


class SpecTests(unittest.TestCase):
    def test_example_spec_loads_with_defaults(self):
        s = cs.load_spec(SPEC)
        self.assertEqual((s['variant'], s['means_pairs'], s['exempt'], s['approved_gpu_hours']), ('shared', 32, False, 18))
        self.assertEqual(s['train']['steps'], 2500)
        self.assertEqual(s['train']['accum'], 2)            # default kept when the spec overrides other train keys
        self.assertEqual(s['arms'], {'mean': 's32_means', 'delete': 's32_zeros'})

    def test_validation(self):
        with self.assertRaises(ValueError):
            spec(variant='nonsense')
        with self.assertRaises(ValueError):
            spec(means_pairs=65)
        with self.assertRaises(ValueError):
            spec(approved_gpu_hours=None)
        self.assertEqual(spec(arms=None, name='x32')['arms'], {'mean': 'x32_means', 'delete': 'x32_zeros'})


class CommandTests(unittest.TestCase):
    def test_pair_choice_maps_to_the_selection_flag_of_each_variant(self):
        for variant, flag in (('shared', '--pairs'), ('separate', '--pairs'), ('pooled', '--pairs'),
                              ('split', '--means-pairs'), ('wide', '--content-pairs')):
            s = spec(variant=variant, means_pairs=40)
            a = cs.selection_args(s, cs.Paths(s))
            self.assertEqual(a[a.index(flag) + 1], '40', variant)
            self.assertIn('--include-layer0', a)
            self.assertIn('--include-step1', a)
        s = spec(exempt=True)
        self.assertNotIn('--include-layer0', cs.selection_args(s, cs.Paths(s)))

    def test_train_args(self):
        s = cs.load_spec(SPEC)
        p = cs.Paths(s)
        a = cs.train_args(s, p, resume=True, stop_after=1000, budget_seconds=1234.7)
        v = lambda f: a[a.index(f) + 1]
        self.assertEqual((v('--steps'), v('--lr'), v('--lr-center'), v('--warmup'), v('--ema'), v('--seed')),
                         ('2500', '1e-05', '0.0002', '250', '0.999', '20261114'))
        self.assertEqual((v('--variant'), v('--center-mode'), v('--stop-after'), v('--budget-seconds')), ('shared', 'mean', '1000', '1234'))
        for f in ('--resume', '--include-layer0', '--include-step1', '--val-at-start', '--ema-checkpoints', '--resume-state-with-ema'):
            self.assertIn(f, a)
        self.assertNotIn('--resume', cs.train_args(s, p))
        self.assertTrue(v('--selection').endswith('out/study-shared32/selection/selection.pt'))

    def test_screen_args(self):
        s = cs.load_spec(SPEC)
        p = cs.Paths(s)
        a = cs.screen_args(s, p)
        self.assertIn('--no-exempt', a)
        self.assertEqual(a[a.index('--variants') + 1], 'shared')
        b = cs.screen_args(s, p, student=True)
        self.assertIn('--student', b)
        self.assertNotIn('--selection', b)


class AuditTests(unittest.TestCase):
    def test_expected_calls(self):
        self.assertEqual(cs.expected_calls(30, 20, False), {'intervention_calls': 2400, 'nope_calls': 0})
        self.assertEqual(cs.expected_calls(30, 20, True), {'intervention_calls': 2400, 'nope_calls': 1200})
        self.assertEqual(cs.expected_calls(29, 19, False)['intervention_calls'], 2204)

    def record(self, **over):
        sel = {'pairs_per_cell_q': 32.0, 'pairs_per_cell_k': 32.0, 'q_k_intersection_mean': 32.0, 'nope_pairs_per_cell': 0.0,
               'wide_head': False, 'default_high_masks': False}
        rec = {'mode': 's32_means', 'finite_step_latents': 20, 'intervention_calls': 2400, 'expected_intervention_calls': 2400,
               'nope_calls': 0, 'expected_nope_calls': 0, 'frames': [45, 480, 480, 3],
               'environment': {'selection': sel, 'center_mode': 'mean', 'learned_center_sha256': 'abc', 'exempt': {'layers': [], 'steps': []}}}
        rec.update(over)
        return rec

    expect = {'variant': 'shared', 'center_sha256': 'abc', 'exempt': {'layers': [], 'steps': []},
              'selection': {'pairs_per_cell_q': 32.0, 'pairs_per_cell_k': 32.0, 'q_k_intersection_mean': 32.0,
                            'nope_pairs_per_cell': 0.0, 'wide_head': False},
              'calls': {'intervention_calls': 2400, 'nope_calls': 0}}

    def test_good_record_passes(self):
        self.assertEqual(cs.check_record(self.record(), self.expect, 's32_means', 'mean'), [])

    def test_bad_records_fail(self):
        self.assertIn('intervention calls', cs.check_record(self.record(intervention_calls=2204), self.expect, 's32_means', 'mean'))
        r = self.record()
        r['environment']['selection']['pairs_per_cell_q'] = 22.0
        self.assertIn('pairs_per_cell_q', cs.check_record(r, self.expect, 's32_means', 'mean'))
        r = self.record()
        r['environment']['exempt'] = {'layers': [0], 'steps': [0]}
        self.assertIn('exemptions', cs.check_record(r, self.expect, 's32_means', 'mean'))
        self.assertIn('center mode/hash', cs.check_record(self.record(), self.expect, 's32_means', 'delete'))


class LedgerTests(unittest.TestCase):
    def test_used_counts_finished_and_live_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            s = cs.load_spec(SPEC)
            st = cs.Study(s, root=Path(d))
            st.p.active.mkdir(parents=True)
            st.p.ledger.write_text('screen\t2\t100.0\t0\ntrain-a01\t2,3\t1000.0\t0\n')
            (st.p.active / 'x.json').write_text(json.dumps({'gpus': [1, 6], 'started': time.time() - 50}))
            used = st.used()
            self.assertGreater(used, 100 + 2000 + 99)
            self.assertLess(used, 100 + 2000 + 110)
            self.assertAlmostEqual(st.remaining(), 18 * 3600 - used, delta=5)

    def test_wall_limit_stop_is_not_an_external_signal(self):
        with tempfile.TemporaryDirectory() as d:
            s = cs.load_spec(SPEC)
            st = cs.Study(s, root=Path(d))
            st.p.run.mkdir(parents=True)
            st.try_lease = lambda n: [(9, 'GPU-fake', open(Path(d) / 'lock', 'a'))]
            st.state = lambda *a, **k: None
            with self.assertRaises(RuntimeError) as e, contextlib.redirect_stdout(io.StringIO()):
                st.wave([('sleeper', [sys.executable, '-c', 'import time; time.sleep(30)'])], wall_limit=1)
            self.assertTrue(str(e.exception).startswith('ASK:'), e.exception)
            self.assertIn('sleeper', st.p.ledger.read_text())
            self.assertEqual(list(st.p.active.glob('*.json')), [])


class DryRunTest(unittest.TestCase):
    def test_dry_run_prints_every_stage_and_writes_nothing(self):
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run([sys.executable, str(root / 'scripts' / 'center_study.py'), str(SPEC), '--dry-run'],
                             cwd=root, text=True, capture_output=True, check=True).stdout
        for needle in ('scripts/select_pairs.py', '--pairs 32', 'scripts/screen_selections.py', 'train_learned_center.py',
                       'mean-render-0', 'delete-render-1', 'screen-trained', 'evaluate_center_benchmark.py',
                       'compare_distill_arms.py', 'measure_distill_agreement.py', 'COLLECT', 'PUBLISH'):
            self.assertIn(needle, out, needle)
        self.assertFalse((root / 'out' / 'study-shared32' / 'run' / 'status.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
