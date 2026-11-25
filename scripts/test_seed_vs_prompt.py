"""CPU tests of cross_text_noise.py and seed_vs_prompt.py on a tiny random Wan and fake trajectories.

  python scripts/test_seed_vs_prompt.py      # prints ALL TESTS OK

1. cross_text_noise.py: the crossed folder holds exactly the noise prompt's step-1 state and the text prompt's
   embeddings; the cache is unchanged; writing inside the cache is refused; crossing a prompt with itself and
   capturing gives exactly the Q/K of the prompt's own cell; asking a crossed folder for step 2 fails.
2. seed_vs_prompt.py on four tiny cells and their six comparisons: every summary number against an explicit
   recomputation from the comparison files; mismatching pair sets are refused.
"""

import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_learned_center import STEPS, write_fake_trajectories, write_tiny_snapshot

ROOT = Path(__file__).resolve().parents[1]
LOW = list(range(6, 22)) + list(range(29, 43)) + list(range(51, 64))


def run(script, args, fail=False):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)] + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True)
    if fail:
        assert proc.returncode != 0, f"{script} should have failed"
        return proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-6000:])
        raise SystemExit(f"{script} failed")
    return proc.stdout


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_seed_vs_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        traj = tmp / "traj"
        write_fake_trajectories(traj)
        tiny = write_tiny_snapshot(tmp / "tiny")
        torch.manual_seed(5)
        mask = torch.rand(2, STEPS, 2, 2, 64) < .3
        torch.save({"variants": {"shared": {"mask_q": mask, "mask_k": mask.clone()}}}, tmp / "selection.pt")
        before = tree_hash(traj)

        # 1. crossed inputs
        for text, noise in (("d0000", "d0001"), ("d0001", "d0000"), ("d0000", "d0000")):
            out = run("cross_text_noise.py", ["--traj", traj, "--text", text, "--noise", noise, "--out", tmp / "seed-traj"])
            assert "CROSSED" in out
        assert tree_hash(traj) == before, "the trajectory cache must not change"
        run("cross_text_noise.py", ["--traj", traj, "--text", "d0000", "--noise", "d0001", "--out", traj / "x"], fail=True)
        assert not (traj / "x").exists()
        folder = tmp / "seed-traj" / "t0000n0001"
        states = torch.load(folder / "states.pt")
        assert states.shape[0] == 1 and torch.equal(states[0], torch.load(traj / "d0001" / "states.pt")[0])
        emb, source = torch.load(folder / "embeddings.pt"), torch.load(traj / "d0000" / "embeddings.pt")
        assert sorted(emb) == sorted(source) and all(torch.equal(emb[k], source[k]) for k in source)
        assert not torch.equal(emb["positive"], torch.load(traj / "d0001" / "embeddings.pt")["positive"])

        sets = ["--set", "shared", tmp / "selection.pt", "shared"]
        cell = lambda pid, root: run("cell_pair_scores.py", ["capture", "--traj", root, "--snapshot", tiny, "--id", pid, "--layer", 1, "--step", 1,
                                                             "--branch", "conditional", "--head", 1, "--out", tmp / "cells", *sets])
        for pid in ("d0000", "d0001"):
            cell(pid, traj)
        for pid in ("t0000n0001", "t0001n0000", "t0000n0000"):
            cell(pid, tmp / "seed-traj")
        name = lambda pid: tmp / "cells" / f"{pid}-L01-S01-conditional-H01"
        own, same = np.load(name("d0000") / "cell.npz"), np.load(name("t0000n0000") / "cell.npz")
        assert np.array_equal(own["q"], same["q"]) and np.array_equal(own["k"], same["k"]), "self-crossing must reproduce the cell exactly"
        crossed = np.load(name("t0000n0001") / "cell.npz")
        assert not np.array_equal(own["q"], crossed["q"]), "a crossed cell must differ from both originals"
        run("cell_pair_scores.py", ["capture", "--traj", tmp / "seed-traj", "--snapshot", tiny, "--id", "t0000n0001", "--layer", 1, "--step", 2,
                                    "--branch", "conditional", "--head", 1, "--out", tmp / "wrong", *sets], fail=True)
        assert not (tmp / "wrong").exists(), "step 2 of a crossed folder must not produce a cell"
        print("test cross_text_noise OK")

        # 2. summary
        labels = {"T0N0": "d0000", "T6N6": "d0001", "T0N6": "t0000n0001", "T6N0": "t0001n0000"}
        for pid in labels.values():
            run("cell_pair_scores.py", ["scores", name(pid), "--cpu"])
        comparisons = (("T0N0", "T6N0"), ("T0N6", "T6N6"), ("T0N0", "T0N6"), ("T6N0", "T6N6"), ("T0N0", "T6N6"), ("T0N6", "T6N0"))
        for a, b in comparisons:
            run("compare_cell_pair_scores.py", [name(labels[a]), name(labels[b]), "--grid", 2, 2, 2, "--out", tmp / "seed" / "compare"])
        cell_args = [x for label, pid in labels.items() for x in ("--cell", label, name(pid))]
        run("seed_vs_prompt.py", [*cell_args, "--compare", tmp / "seed" / "compare", "--out", tmp / "seed"])
        for path in ("SEED-VS-PROMPT.md", "seed-vs-prompt.csv", "seed-vs-prompt.png"):
            assert (tmp / "seed" / path).exists()
        summary = list(csv.DictReader(open(tmp / "seed" / "seed-vs-prompt.csv")))
        assert len(summary) == 6 * 64
        markdown = (tmp / "seed" / "SEED-VS-PROMPT.md").read_text()
        for index, (a, b) in enumerate(comparisons, 1):
            rows = list(csv.DictReader(open(tmp / "seed" / "compare" / f"{name(labels[a]).name}--vs--{name(labels[b]).name}.csv")))
            mine = [r for r in summary if r["comparison"] == str(index)]
            assert [r["cell_a"] for r in mine[:1]] == [a] and [r["cell_b"] for r in mine[:1]] == [b] and len(mine) == 64
            sa, sb = (np.array([float(r[c]) for r in rows]) for c in ("centred_share_a", "centred_share_b"))
            median = np.array([float(r["map_correlation_median_27_queries"]) for r in rows])
            correlation, largest = np.corrcoef(sa, sb)[0, 1], np.abs(sa - sb).max()
            assert abs(float(mine[0]["comparison_share_correlation_64_pairs"]) - correlation) < 1e-6
            assert abs(float(mine[0]["comparison_largest_share_difference"]) - largest) < 1e-7
            assert int(mine[0]["comparison_largest_difference_pair"]) == int(np.abs(sa - sb).argmax())
            assert all(abs(float(m["map_correlation_median_27_queries"]) - v) < 1e-7 for m, v in zip(mine, median))
            assert [int(m["low_frequency"]) for m in mine] == [int(f in LOW) for f in range(64)] and sum(int(m["low_frequency"]) for m in mine) == 43
            strong = sa > 1 / 64
            assert [int(m["above_1_64_in_a"]) for m in mine] == strong.astype(int).tolist()
            line = f"| {index} | " + mine[0]["type"] + f" | {a} vs {b} | all 64 pairs | " + " | ".join(f"{v:+.3f}" for v in np.quantile(median, [0, .05, .25, .5, .75, .95, 1])) + " |"
            assert line in markdown, line
            line = f"| {index} | " + mine[0]["type"] + f" | {a} vs {b} | 43 low-frequency pairs | " + " | ".join(f"{v:+.3f}" for v in np.quantile(median[LOW], [0, .05, .25, .5, .75, .95, 1])) + " |"
            assert line in markdown, line
            if strong.any():
                line = (f"| {index} | " + mine[0]["type"] + f" | {a} vs {b} | pairs above 1/64 in {a} ({int(strong.sum())}) | "
                        + " | ".join(f"{v:+.3f}" for v in np.quantile(median[strong], [0, .05, .25, .5, .75, .95, 1])) + " |")
                assert line in markdown, line
        # the both-differ comparison of the same two cells in either order has the same map correlations
        assert all(abs(float(x["map_correlation_median_27_queries"]) - float(y["map_correlation_median_27_queries"])) < 1e-9
                   for x, y in zip([r for r in summary if r["comparison"] == "6"],
                                   csv.DictReader(open(tmp / "seed" / "compare" / f"{name('t0000n0001').name}--vs--{name('t0001n0000').name}.csv"))))

        # a cell with other pair sets is refused
        other = dict(np.load(name("t0001n0000") / "cell.npz"))
        other["set_shared"] = ~other["set_shared"]
        bad = tmp / "cells" / "bad"
        bad.mkdir()
        np.savez_compressed(bad / "cell.npz", **other)
        (bad / "pair-scores.csv").write_bytes((name("t0001n0000") / "pair-scores.csv").read_bytes())
        cell_args = [x for label, pid in labels.items() for x in ("--cell", label, bad if label == "T6N0" else name(pid))]
        out = run("seed_vs_prompt.py", [*cell_args, "--compare", tmp / "seed" / "compare", "--out", tmp / "seed2"], fail=True)
        assert "pair sets" in out
        print("test seed_vs_prompt OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_seed_vs_prompt()
    print("ALL TESTS OK")
