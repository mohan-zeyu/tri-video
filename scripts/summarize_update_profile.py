#!/usr/bin/env python3
"""Statistics, checks, README tables and the measured figure of the update profile (scripts/profile_update.py).

Per arm and phase: per update and rank, the per-sample phases are summed over the samples of that update; the two
ranks are averaged; median, interquartile range and mean are taken across the measured updates (warm-up discarded).
Shares are phase medians over the sum of phase medians (mean-based shares alongside). The per-sample table divides
the per-update sample phases by the samples per rank per update.
"""
import csv
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_profile import SAMPLE_PHASES, UPDATE_PHASES

REFERENCE_UPDATE = 6.911     # no-exemption run: median of 250 logged updates, mean over two ranks (brief)
ESTIMATE = {"teacher_forward": 0.62, "student_forward": 0.62, "recompute": 0.62, "rest": 1.58, "total": 3.46}
ARMS = ["P0", "P1", "P2", "P3", "P4a", "P4b", "P4c", "P4d", "P5"]


def q(values, f):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * f
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(values):
    return {"median": q(values, 0.5), "q25": q(values, 0.25), "q75": q(values, 0.75), "mean": statistics.mean(values) if values else None, "n": len(values)}


def load_rows(arm_dir):
    rows = []
    for f in sorted(arm_dir.glob("phase-times-rank*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    return rows


def arm_summary(arm_dir, accum):
    rows = [r for r in load_rows(arm_dir) if r["measured"]]
    ranks = sorted({r["rank"] for r in rows})
    updates = sorted({r["update"] for r in rows})
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))    # update -> rank -> phase -> seconds
    comm = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))   # update -> rank -> key -> value
    blocks = defaultdict(list)                                            # (pass, block) -> seconds per sample
    for r in rows:
        if r["kind"] in ("sample_phase", "update_phase"):
            per[r["update"]][r["rank"]][r["phase"]] += r["seconds"]
        elif r["kind"] in ("update_wall", "phase_sum"):
            per[r["update"]][r["rank"]]["_" + r["kind"]] = r["seconds"]
        elif r["kind"] == "comm":
            for k, v in r.items():
                if k.endswith("_bytes") or k.endswith("_calls"):
                    comm[r["update"]][r["rank"]][k] += v
                    comm[r["update"]][r["rank"]][r["phase"] + ":" + k] += v
        elif r["kind"] == "block":
            blocks[(r["pass"], r["block"])].append(r["seconds"])
    phases = sorted({p for u in per.values() for rk in u.values() for p in rk if not p.startswith("_")},
                    key=lambda p: (list(SAMPLE_PHASES + UPDATE_PHASES).index(p) if p in SAMPLE_PHASES + UPDATE_PHASES else 99, p))
    def rank_mean(u, key, src=per):
        return statistics.mean(src[u][rk].get(key, 0.0) for rk in ranks)
    out = {"ranks": len(ranks), "updates_measured": len(updates), "samples_per_rank_per_update": accum,
           "update_wall": stats([rank_mean(u, "_update_wall") for u in updates])}
    if phases:
        ph = {p: stats([rank_mean(u, p) for u in updates]) for p in phases}
        med_sum = sum(v["median"] for v in ph.values())
        mean_sum = sum(v["mean"] for v in ph.values())
        for p, v in ph.items():
            v["share_median"] = v["median"] / med_sum
            v["share_mean"] = v["mean"] / mean_sum
        out["phases_per_update"] = ph
        out["sum_of_phase_medians"] = med_sum
        out["per_sample"] = {p: {"median": ph[p]["median"] / accum, "mean": ph[p]["mean"] / accum} for p in phases if p in SAMPLE_PHASES}
        rel = [abs(per[u][rk]["_phase_sum"] - per[u][rk]["_update_wall"]) / per[u][rk]["_update_wall"] for u in updates for rk in ranks]
        out["phase_sum_vs_wall_max_relative_error"] = max(rel)
    if comm:
        keys = sorted({k for u in comm.values() for rk in u.values() for k in rk})
        out["comm_per_update"] = {k: stats([rank_mean(u, k, comm) for u in updates])["median"] for k in keys}
    if blocks:
        out["blocks_per_sample_median"] = {p: [q(blocks[(p, b)], 0.5) for b in range(30) if (p, b) in blocks] for p in ("first", "recompute")}
    return out


def telemetry(arm_dir, gpus):
    path = arm_dir / "gpu-telemetry.csv"
    if not path.exists():
        return None
    rows = list(csv.reader(path.open()))
    head = [h.strip() for h in rows[0]]
    data = defaultdict(list)
    def num(x):
        try:
            return float(x.strip().split()[0])
        except (ValueError, IndexError):
            return None
    for r in rows[1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        data[int(d["index"])].append(d)
    def col(rs, name):
        return [v for v in (num(r[name]) for r in rs) if v is not None]
    out = {}
    for i, rs in data.items():
        reasons = defaultdict(int)
        for r in rs:
            reasons[r["clocks_event_reasons.active"].strip()] += 1
        util = col(rs, "utilization.gpu [%]")
        out[str(i)] = {"leased": i in gpus, "samples": len(rs), "util_mean": statistics.mean(util) if util else None,
                       "power_w_mean": statistics.mean(col(rs, "power.draw [W]") or [0]), "power_w_max": max(col(rs, "power.draw [W]") or [0]),
                       "power_limit_w": (col(rs, "power.limit [W]") or [None])[0], "temperature_c_max": max(col(rs, "temperature.gpu") or [0]),
                       "sm_clock_mhz_median": q(col(rs, "clocks.current.sm [MHz]"), 0.5), "sm_clock_mhz_min": min(col(rs, "clocks.current.sm [MHz]") or [0]),
                       "sm_clock_max_mhz": (col(rs, "clocks.max.sm [MHz]") or [None])[0], "event_reasons_counts": dict(reasons)}
    return out


def fmt(x, d=3):
    return "-" if x is None else ("%.*f" % (d, x))


def main(work, results):
    work, results = Path(work), Path(results)
    results.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in ARMS:
        d = work / arm
        if not (d / "done.json").exists():
            continue
        done = json.loads((d / "done.json").read_text())
        dest = results / arm
        dest.mkdir(exist_ok=True)
        args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        s = {"done": done, "telemetry": telemetry(d, done["gpus"])}
        if arm != "P5" and args:
            s.update(arm_summary(d, args["accum"]))
            with (dest / "phase-times.jsonl").open("w") as f:
                for r in load_rows(d):
                    f.write(json.dumps(r) + "\n")
        for f in ("args.json", "gpu-telemetry.csv", "smi-before.txt", "smi-after.txt", "profiler-top-ops.json", "forward-anatomy.json"):
            if (d / f).exists():
                shutil.copyfile(d / f, dest / f)
        if (d / "forward-anatomy.json").exists():
            fa = json.loads((d / "forward-anatomy.json").read_text())
            s["anatomy"] = {k: fa[k] for k in fa if k != "rows"}
        s["peak_memory"] = [json.loads(f.read_text()) for f in sorted(d.glob("peak-memory-rank*.json"))]
        s["out_of_memory"] = [json.loads(f.read_text()) for f in sorted(d.glob("oom-rank*.json"))]
        (dest / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        summary[arm] = s
    for f in ("host.json",):
        if (work / f).exists():
            shutil.copyfile(work / f, results / f)
    if (work / "run" / "ledger.tsv").exists():
        shutil.copyfile(work / "run" / "ledger.tsv", results / "ledger.tsv")
    checks = compute_checks(summary)
    (results / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    try:
        figure(summary, checks, results / "measured-update-breakdown.png")
    except Exception as error:
        print("FIGURE FAILED", repr(error))
    return summary, checks


def med(summary, arm, phase=None, key="median"):
    s = summary.get(arm)
    if not s:
        return None
    if phase is None:
        return s["update_wall"][key]
    return s.get("phases_per_update", {}).get(phase, {}).get(key, 0.0)


def compute_checks(summary):
    c = {}
    if "P0" in summary:
        p0 = med(summary, "P0")
        c["1_p0_vs_reference"] = {"p0_median_update_s": p0, "reference_s": REFERENCE_UPDATE, "relative_difference": p0 / REFERENCE_UPDATE - 1,
                                  "within_10_percent": abs(p0 / REFERENCE_UPDATE - 1) <= 0.10}
    if "P0" in summary and "P1" in summary:
        c["2_timer_overhead"] = {"p1_median_update_s": med(summary, "P1"), "p0_median_update_s": med(summary, "P0"),
                                 "overhead": med(summary, "P1") / med(summary, "P0") - 1, "under_3_percent": med(summary, "P1") / med(summary, "P0") - 1 < 0.03}
    if "P1" in summary:
        e = summary["P1"]["phase_sum_vs_wall_max_relative_error"]
        c["3_phases_sum_to_wall"] = {"max_relative_error_any_update_rank": e, "within_1_percent": e <= 0.01}
        fixed = sum(med(summary, "P1", p) for p in UPDATE_PHASES)
        c["fixed_cost_p1_s"] = fixed
    if "P1" in summary and "P4c" in summary:
        t2, t1 = med(summary, "P1"), med(summary, "P4c")
        intercept = 2 * t1 - t2
        c["4_fixed_cost_intercept"] = {"accum2_median_s": t2, "accum1_median_s": t1, "intercept_s": intercept, "p1_fixed_phases_s": c["fixed_cost_p1_s"],
                                       "difference_s": intercept - c["fixed_cost_p1_s"], "within_0_2_s": abs(intercept - c["fixed_cost_p1_s"]) <= 0.2,
                                       "model": "update(accum) = fixed + accum x per-sample; intercept = 2 x update(1) - update(2); P1 timers on both"}
    if "P1" in summary and "P4a" in summary:
        c["5a_feature_loss"] = {"total_change_s": med(summary, "P1") - med(summary, "P4a"), "p1_loss_phase_s": med(summary, "P1", "loss"),
                                "p4a_loss_phase_s": med(summary, "P4a", "loss"),
                                "per_phase_change_s": {p: med(summary, "P1", p) - med(summary, "P4a", p) for p in summary["P1"]["phases_per_update"]}}
    if "P1" in summary and "P4b" in summary:
        c["5b_ema"] = {"total_change_s": med(summary, "P1") - med(summary, "P4b"), "p1_ema_phase_s": med(summary, "P1", "ema")}
    if "P4d" in summary:
        s = summary["P4d"]
        c["p4d_no_checkpointing"] = {"out_of_memory": bool(s["out_of_memory"]), "median_update_s": s["update_wall"]["median"] if s.get("update_wall", {}).get("n") else None,
                                     "peak": s["out_of_memory"] or s["peak_memory"]}
    if "P1" in summary:
        total = med(summary, "P1")
        share = lambda *ps: sum(med(summary, "P1", p) for p in ps) / total
        bounds = {"teacher_forward": share("teacher_forward"), "recompute": share("recompute"),
                  "clip_optimizer_ema": share("clip", "optimizer", "ema"), "data": share("data")}
        if "P3" in summary:
            sample = lambda arm: sum(med(summary, arm, p) for p in SAMPLE_PHASES)
            bounds["sharding_sample_phase_gap_p1_minus_p3"] = (sample("P1") - sample("P3")) / total
        if "P5" in summary and summary["P5"].get("anatomy"):
            an = summary["P5"]["anatomy"]["summary"]
            kernel = an["attn1_sdpa"]["median_seconds"] / an["forward_total"]["median_seconds"]
            bounds["attention_kernel_share_of_forward_p5"] = kernel
            bounds["attention_kernel_applied_to_student_phases"] = kernel * share("student_forward", "recompute", "backward_rest")
        c["decision_bounds_share_of_p1_update"] = bounds
    return c


def figure(summary, checks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "P1" not in summary:
        return
    accum = summary["P1"]["samples_per_rank_per_update"]
    ph = summary["P1"]["phases_per_update"]
    per_sample = {p: ph[p]["median"] / accum for p in ph}
    total = sum(per_sample.values())
    segs = [("Teacher forward (frozen, no gradient)", per_sample.get("teacher_forward", 0), "#ec7fb0"),
            ("Student forward (first pass)", per_sample.get("student_forward", 0), "#038a12"),
            ("Student forward repeated for gradient checkpointing", per_sample.get("recompute", 0), "#6b4ce0"),
            ("Rest of backward (incl. FSDP unshard / reduce-scatter)", per_sample.get("backward_rest", 0), "#e84a4a"),
            ("Loss terms and feature loss", per_sample.get("loss", 0), "#f2a900"),
            ("Trajectory loading", per_sample.get("data", 0), "#00b894"),
            ("Per-update fixed cost (centre sync, clipping, optimizer, EMA, bookkeeping) / %d" % accum,
             sum(per_sample.get(p, 0) for p in UPDATE_PHASES), "#9a9a9a")]
    fig = plt.figure(figsize=(10, 8.6), dpi=200)
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.96])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 0.975, "Where the GPU time goes in the mean-vector distillation", fontsize=17, weight="bold", va="top")
    ax.text(0, 0.935, "Each bar is one whole. The next bar opens up the largest part of the one above it.", fontsize=10.5, color="#555", va="top")
    def bar(y, parts, h=0.035):
        x = 0
        tot = sum(v for _, v, _ in parts)
        for _, v, col in parts:
            w = v / tot
            ax.add_patch(plt.Rectangle((x, y), w - 0.002, h, color=col))
            if w > 0.05:
                ax.text(x + 0.008, y + h / 2, "%d%%" % round(100 * w), color="white", fontsize=10, weight="bold", va="center")
            x += w
    def legend(y, parts, unit):
        for i, (name, v, col) in enumerate(parts):
            ax.add_patch(plt.Rectangle((0, y - i * 0.03 - 0.009), 0.012, 0.018, color=col))
            ax.text(0.025, y - i * 0.03, name, fontsize=10, va="center")
            ax.text(0.7, y - i * 0.03, unit % v, fontsize=10, va="center", weight="bold", ha="right")
    ax.text(0, 0.88, "1 · The first full study   71.5 GPU-hours   (measured)", fontsize=12.5, weight="bold", va="top")
    ax.text(0, 0.85, "Allocation ledger of the original Stage 0-3 study plus dense control: worker elapsed time x GPUs.", fontsize=9, color="#555", va="top")
    p1 = [("Training runs (Stages 1, 2, 3 and the dense control)", 48.7, "#2a7de1"), ("Teacher trajectories, calibration and references", 15.8, "#f07c32"),
          ("Benchmarks: video rendering and VBench scoring", 5.9, "#00b894"), ("Pilot allowance and scheduling margin", 1.1, "#f2a900")]
    bar(0.79, p1)
    legend(0.765, p1, "%.1f GPU-h")
    ax.text(0, 0.62, "2 · Inside one training run   10.06 GPU-hours   (measured)", fontsize=12.5, weight="bold", va="top")
    ax.text(0, 0.59, "No-exemption shared22 fine-tune: 2,500 updates on 2 GPUs, mean of 253 logged updates (7.00 s) x 2,500 x 2 GPUs.", fontsize=9, color="#555", va="top")
    p2 = [("Optimizer updates", 9.7, "#2a7de1"), ("Validation, checkpoints, EMA saves and loading", 0.3, "#9a9a9a")]
    bar(0.53, p2)
    legend(0.505, p2, "%.1f GPU-h")
    ax.text(0, 0.43, "3 · Inside one training sample   %.2f GPU-seconds   (measured, P1)" % total, fontsize=12.5, weight="bold", va="top")
    import textwrap
    argv = summary["P1"]["done"].get("argv", [])
    warm = argv[argv.index("--profile-warmup") + 1] if "--profile-warmup" in argv else None
    ax.text(0, 0.40, textwrap.fill("Recipe on 2 ranks, %d measured updates after %s discarded; one sample = one prompt at one step, both CFG "
                                   "branches as a batch of 2, 10,800 tokens. Phase medians per update, mean of the two ranks, divided by "
                                   "%d samples per rank." % (summary["P1"]["updates_measured"], warm or "the", accum), 150),
            fontsize=8.6, color="#555", va="top")
    bar(0.33, segs)
    legend(0.305, segs, "%.3f s")
    ax.text(0, 0.06, textwrap.fill("Measured on the lab host, GPUs %s; synchronised phase boundaries; recompute from per-block CUDA events "
                                   "during .backward(). The fixed per-update cost is spread over the samples of one rank. Estimate before "
                                   "profiling (bar 3 of estimate-before-profiling.png): 0.62 / 0.62 / 0.62 / 1.58 s."
                                   % summary["P1"]["done"]["gpus"], 160), fontsize=8.2, color="#555", va="top")
    fig.savefig(path)
