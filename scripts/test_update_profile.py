"""CPU test of the trainer's profile mode (--profile) on a tiny random Wan, fake trajectories, --no-fsdp.

1. Phase timers with per-block times: every update has the six per-sample phases for each of its samples and the
   five per-update phases; the phases sum to the trainer's own optimizer_step_seconds within 1%; block rows cover
   both passes (first and checkpoint recomputation) of every block of every sample. No student, EMA, resume state,
   validation or history is written.
2. --profile-timers none writes only update wall times; torch.profiler on one update writes profiler-top-ops.json.
3. --profile-floor (bf16 student, no optimizer, no EMA, no clipping) has the per-sample phases and no clip,
   optimizer or ema phase.
4. A non-empty --out is refused.

Run from the repository root:  python scripts/test_update_profile.py
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_learned_center import ROOT, run, write_fake_trajectories, write_tiny_snapshot
from update_profile import SAMPLE_PHASES, UPDATE_PHASES

WARMUP, UPDATES, ACCUM = 1, 2, 2


def rows(out):
    return [json.loads(line) for line in (out / "phase-times-rank0.jsonl").read_text().splitlines()]


def check_phases(out, update_phases, samples=ACCUM):
    by_update = defaultdict(list)
    for r in rows(out):
        by_update[r["update"]].append(r)
    assert sorted(by_update) == list(range(WARMUP + UPDATES)), sorted(by_update)
    for u, rs in by_update.items():
        assert sum(r["measured"] for r in rs if r["kind"] == "update_wall") == (u >= WARMUP)
        for s in range(samples):
            got = {r["phase"] for r in rs if r["kind"] == "sample_phase" and r["sample"] == s}
            assert got == set(SAMPLE_PHASES), (u, s, got)
        got = {r["phase"] for r in rs if r["kind"] == "update_phase"}
        assert got == set(update_phases), (u, got)
        total = next(r["seconds"] for r in rs if r["kind"] == "phase_sum")
        wall = next(r["seconds"] for r in rs if r["kind"] == "update_wall")
        assert abs(total - wall) <= 0.01 * wall, (u, total, wall)
        assert all(r["seconds"] >= 0 for r in rs if r["kind"] in ("sample_phase", "update_phase") and r["phase"] != "backward_rest")
        blocks = [r for r in rs if r["kind"] == "block"]
        for s in range(samples):
            for p in ("first", "recompute"):
                assert sorted(r["block"] for r in blocks if r["sample"] == s and r["pass"] == p) == [0, 1], (u, s, p)
        recompute = next(r["seconds"] for r in rs if r["kind"] == "sample_phase" and r["sample"] == 0 and r["phase"] == "recompute")
        assert abs(recompute - sum(r["seconds"] for r in blocks if r["sample"] == 0 and r["pass"] == "recompute")) < 1e-9


def test_profile_mode():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        traj = tmp / "traj"
        cfg = write_fake_trajectories(traj)
        tiny = write_tiny_snapshot(tmp / "tiny")
        base = tmp / "distill-profile"
        common = ["--stage", "2", "--traj", str(traj), "--prompts", str(cfg), "--snapshot", str(tiny), "--no-fsdp", "--steps", "20",
                  "--warmup", "2", "--accum", str(ACCUM), "--feat-layers", "0", "--ema", "0.9", "--val-every", "1", "--val-at-start",
                  "--save-every", "1", "--log-every", "1", "--profile-warmup", str(WARMUP), "--profile-updates", str(UPDATES)]

        p1 = base / "p1"
        out = run(common + ["--out", str(p1), "--profile", "p1", "--profile-timers", "blocks"], ROOT)
        assert "VAL" not in out, out[-800:]
        check_phases(p1, UPDATE_PHASES)
        produced = {p.name for p in p1.iterdir()}
        assert produced == {"args.json", "phase-times-rank0.jsonl", "peak-memory-rank0.json"}, produced
        assert json.loads((p1 / "peak-memory-rank0.json").read_text())["updates_run"] == WARMUP + UPDATES

        p0 = base / "p0"
        run(common + ["--out", str(p0), "--profile", "p0", "--profile-timers", "none"], ROOT)
        assert {r["kind"] for r in rows(p0)} == {"update_wall", "losses"}
        # same seed and sample sequence: the losses of P0 and P1 agree exactly
        loss = lambda out: [{k: v for k, v in r.items() if k not in ("arm",)} for r in rows(out) if r["kind"] == "losses"]
        assert loss(p0) == loss(p1), (loss(p0), loss(p1))

        p2 = base / "p2"
        run(common + ["--out", str(p2), "--profile", "p2", "--profile-timers", "phases", "--profile-torch-updates", "1"], ROOT)
        top = json.loads((p2 / "profiler-top-ops.json").read_text())
        assert top["profiled_updates"] == 1 and top["top_cuda_ops_by_self_time"] is not None

        p3 = base / "p3"
        run(common + ["--out", str(p3), "--profile", "p3", "--profile-timers", "blocks", "--profile-floor"], ROOT)
        check_phases(p3, ("center_sync", "bookkeeping"))

        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_learned_center.py"), *common, "--out", str(p1), "--profile", "again"],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode != 0 and "new directory" in proc.stderr, proc.stderr[-500:]
        print("test_profile_mode OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_profile_mode()
    print("ALL TESTS OK")
