"""CPU tests for the per-pair attention-score attribution.

1. pair_score_attribution.attribute against brute force on explicit [query, key, pair] contributions: the
   variance decomposition, the attention-advantage and top-1 fields, the center model, the across-query
   quantiles, the bf16 rounding path and query_stride.
2. collect_pair_score_attribution.py end to end on a tiny random Wan and fake trajectories (replay source) and
   on fake saved captures (sweep source): record shapes, one cell recomputed independently from directly
   hooked Q/K, the attention-output validation, no claims left behind.
3. analyze_pair_score_attribution.py on those records with a selection file: report structure, quantiles and
   the selection overlay recomputed for the whole population, the --cell lookup, an undefined center_pair_r
   (center-model variance at rounding noise) counted instead of overflowing, and a non-finite share stopping
   the analysis.
4. step_similarity_pair_score_attribution.py: the step-to-step similarity of each trajectory's 64 shares,
   recomputed here for one pair of steps over all trajectories, per layer and per branch.
6. selection_score_share.py: shares of named pair sets and of a set difference, recomputed here and compared with
   the analyzer's selection overlay; the plain attention-score percentage; and a trained student (--student):
   replaced pairs constant over tokens, exempt cells identical to the teacher, one active cell recomputed with the
   replacement attached, and the replaced pairs' shares in the student and in the teacher on the same cells.
5. plot_step_trajectories_pair_score_attribution.py: every plotted number is one cell-pair value, and the drawn
   heads and pairs follow the stated selection rule.

Run from the repository root:  python scripts/test_pair_score_attribution.py
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_sweep_metrics import rotate
import analyze_pair_score_attribution as A
from pair_score_attribution import QUANTILES, attribute, quantiles
from test_learned_center import STEPS, write_fake_trajectories, write_tiny_snapshot

ROOT = Path(__file__).resolve().parents[1]


def close(a, b, tol=1e-5):
    torch.testing.assert_close(torch.as_tensor(a).double(), torch.as_tensor(b).double(), atol=tol, rtol=tol)


def pairwise(a, b):
    """contributions[h, query, key, pair] of rotated [H, N, D] tensors."""
    h, n, d = a.shape
    return (a.reshape(h, n, 1, d // 2, 2) * b.reshape(h, 1, n, d // 2, 2)).sum(-1) / math.sqrt(d)


def brute(qr, kr):
    """Reference fields from the explicit contribution tensor."""
    contrib = pairwise(qr.double(), kr.double())
    logits = contrib.sum(-1)
    n = logits.shape[1]
    centred = contrib - contrib.mean(2, keepdim=True)
    g = torch.einsum("hijf,hijg->hfg", centred, centred) / n ** 2
    p = torch.softmax(logits, -1)
    adv = (p[..., None] * centred).sum(2)
    top = logits.argmax(2)
    margin = torch.gather(centred, 2, top[:, :, None, None].expand(-1, -1, 1, contrib.shape[-1]))[:, :, 0]
    row_cov = (centred * (logits - logits.mean(2, keepdim=True))[..., None]).mean(2)
    entropy = -(p * p.log()).sum(2)
    return {"contrib": contrib, "logits": logits, "centred": centred, "g": g, "p": p, "adv": adv, "top": top,
            "margin": margin, "row_cov": row_cov, "row_var": logits.var(2, unbiased=False), "entropy": entropy}


def setup():
    torch.manual_seed(5)
    grid, h, d = (3, 4, 5), 2, 12
    n = math.prod(grid)
    q = torch.randn(h, n, d) + .8
    k = torch.randn(h, n, d) + .5
    coords = torch.stack(torch.meshgrid(*(torch.arange(s) for s in grid), indexing="ij"), -1).reshape(-1, 3)
    phase = torch.cat([coords[:, i:i + 1] * torch.tensor([[.9, .2]]) for i in range(3)], -1)
    return q, k, phase.cos(), phase.sin()


def test_attribute_against_brute_force():
    q, k, cos, sin = setup()
    h, n, d = q.shape
    result = attribute(q, k, cos, sin, chunk=7)
    ref = brute(rotate(q, cos, sin), rotate(k, cos, sin))
    close(ref["logits"], rotate(q, cos, sin).double() @ rotate(k, cos, sin).double().transpose(1, 2) / math.sqrt(d), 1e-10)
    # A. exact variance decomposition over every query x key
    row_var = ref["row_var"].mean(1)
    close(result["row_var"], row_var)
    close(result["var_cov"], ref["g"].sum(2))
    close(result["var_cov"].sum(1), row_var)
    close(result["var_own"], ref["g"].diagonal(dim1=1, dim2=2))
    close(result["pair_cov"], ref["g"] / row_var[:, None, None], 2e-3)             # stored as float16
    close(result["mean_contribution"], ref["contrib"].mean((1, 2)))
    # B. across-query quantiles of the per-query shares
    levels = torch.tensor(QUANTILES, dtype=torch.float64)
    close(result["var_share_quantiles"], torch.quantile(ref["row_cov"] / ref["row_var"][..., None], levels, dim=1).permute(1, 2, 0))
    close(result["row_var_quantiles"], torch.quantile(ref["row_var"], levels, dim=1).T)
    # C. attention-weighted advantage
    total = (ref["p"] * ref["logits"]).sum(2) - ref["logits"].mean(2)
    assert (total > 0).all()
    close(ref["adv"].sum(-1), total, 1e-10)
    close(result["adv"], ref["adv"].mean(1))
    close(result["adv_total"], total.mean(1))
    close(result["adv"] + result["mean_contribution"], (ref["p"][..., None] * ref["contrib"]).sum(2).mean(1))
    close(result["adv_share_quantiles"], torch.quantile(ref["adv"] / total[..., None], levels, dim=1).permute(1, 2, 0), 1e-4)
    close(result["adv_total_quantiles"], torch.quantile(total, levels, dim=1).T)
    close(result["entropy"], ref["entropy"].mean(1))
    close(result["effective_keys"], ref["entropy"].exp().mean(1))
    close(result["pmax"], ref["p"].max(2).values.mean(1))
    # D. top-1 margin
    close(ref["margin"].sum(-1), ref["logits"].max(2).values - ref["logits"].mean(2), 1e-10)
    close(result["top1"], ref["margin"].mean(1))
    close(result["top1_total"], ref["margin"].sum(-1).mean(1))
    close(result["top1_self"], (ref["top"] == torch.arange(n)).double().mean(1))
    # E. center model: token means rotated at the real positions
    cq, ck = q.mean(1, keepdim=True).expand_as(q), k.mean(1, keepdim=True).expand_as(k)
    center = brute(rotate(cq, cos, sin), rotate(ck, cos, sin))
    a, b, scale = ref["centred"], center["centred"], n ** 2
    close(result["center_var_own"], (b * b).sum((1, 2)) / scale)
    close(result["center_var_cov"], (b * b.sum(-1, keepdim=True)).sum((1, 2)) / scale)
    close(result["center_cross_own"], (a * b).sum((1, 2)) / scale)
    close(result["center_cross_cov"], (b * a.sum(-1, keepdim=True)).sum((1, 2)) / scale)
    # F. pre-RoPE pair statistics
    zq, zk = q.double().reshape(h, n, d // 2, 2), k.double().reshape(h, n, d // 2, 2)
    close(result["energy_q"], zq.square().sum(-1).mean(1))
    close(result["norm_k"], zk.norm(dim=-1).mean(1))
    close(result["center_norm_q"], zq.mean(1).norm(dim=-1))
    assert all(v.shape[0] == h for v in result.values())
    print("test_attribute_against_brute_force OK")


def test_constant_tokens_are_the_center_model():
    q, k, cos, sin = setup()
    q, k = q.mean(1, keepdim=True).expand_as(q).contiguous(), k.mean(1, keepdim=True).expand_as(k).contiguous()
    result = attribute(q, k, cos, sin, chunk=11)
    close(result["var_cov"], result["center_var_cov"])
    close(result["center_cross_own"], result["var_own"])
    close(result["center_norm_q"], result["norm_q"])
    print("test_constant_tokens_are_the_center_model OK")


def test_bf16_and_query_stride():
    q, k, cos, sin = setup()
    qb, kb = q.bfloat16(), k.bfloat16()
    result = attribute(qb, kb, cos, sin, chunk=13)
    ref = brute(rotate(qb.float(), cos, sin, torch.bfloat16), rotate(kb.float(), cos, sin, torch.bfloat16))
    close(result["var_cov"], ref["g"].sum(2))
    close(result["adv"], ref["adv"].mean(1))
    full = attribute(q, k, cos, sin, chunk=7)
    strided = attribute(q, k, cos, sin, chunk=7, query_stride=3, pair_cov=False)
    ref = brute(rotate(q, cos, sin), rotate(k, cos, sin))
    close(strided["adv"], ref["adv"][:, ::3].mean(1))
    close(strided["top1_self"], (ref["top"][:, ::3] == torch.arange(0, q.shape[1], 3)).double().mean(1))
    close(strided["var_cov"], full["var_cov"])
    assert "pair_cov" not in strided
    x = torch.randn(3, 50, 4, dtype=torch.float64)
    close(quantiles(x, 1), torch.quantile(x, torch.tensor(QUANTILES, dtype=torch.float64), dim=1).permute(1, 2, 0), 1e-12)
    print("test_bf16_and_query_stride OK")


def run(script, args):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)] + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-6000:])
        raise SystemExit(f"{script} failed")
    return proc.stdout


def test_collector_and_analysis():
    from diffusers import WanTransformer3DModel
    from train_learned_center import TrajectoryStore
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = write_fake_trajectories(tmp / "traj")
        tiny = write_tiny_snapshot(tmp / "tiny")
        stats = tmp / "attribution"
        run("collect_pair_score_attribution.py", ["--traj", tmp / "traj", "--prompts", cfg, "--snapshot", tiny, "--out", stats, "--chunk", 3])
        files = sorted(stats.glob("*.pt"))
        assert [f.stem for f in files] == ["d0000", "d0001", "d0002", "d0003"] and not list(stats.glob("*.claim"))
        record = torch.load(files[1])
        assert tuple(record["var_cov"].shape) == (2, STEPS, 2, 2, 64) and tuple(record["pair_cov"].shape) == (2, STEPS, 2, 2, 64, 64)
        assert tuple(record["adv_share_quantiles"].shape) == (2, STEPS, 2, 2, 64, 5) and tuple(record["row_var"].shape) == (2, STEPS, 2, 2)
        assert record["filled"].all() and record["branches"] == ["conditional", "unconditional"] and record["pair_cov"].dtype == torch.float16
        validation = record["attention_output_validation"]
        assert len(validation) == 4 and all(v["relative_max_err"] <= 1e-3 for v in validation.values()), validation     # 2 layers x 2 branches
        # one cell recomputed from directly hooked tensors: prompt d0001, layer 1, step 2, unconditional branch
        model = WanTransformer3DModel.from_pretrained(tiny, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        item = TrajectoryStore(tmp / "traj", ["d0001"]).item("d0001", 2)
        seen = {}
        handles = [getattr(model.blocks[1].attn1, "norm_" + s).register_forward_hook(lambda m, i, o, s=s: seen.__setitem__(s, o.detach().clone())) for s in ("q", "k")]
        handles.append(model.rope.register_forward_hook(lambda m, i, o: seen.__setitem__("rope", o)))
        with torch.inference_mode():
            x = item["x"].bfloat16().unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
            model(hidden_states=x, timestep=torch.full((2,), float(item["timestep"])),
                  encoder_hidden_states=torch.cat([item["positive"], item["negative"]]).bfloat16(), return_dict=False)
        [h.remove() for h in handles]
        cos, sin = seen["rope"][0][0, :, 0, 0::2].float(), seen["rope"][1][0, :, 0, 1::2].float()
        q, k = [seen[s][1].float().view(-1, 2, 128).transpose(0, 1) for s in ("q", "k")]                                 # [heads, tokens, 128]
        ref = brute(rotate(q, cos, sin, torch.bfloat16), rotate(k, cos, sin, torch.bfloat16))
        close(record["var_cov"][1, 2, 1], ref["g"].sum(2), 1e-4)
        close(record["adv"][1, 2, 1], ref["adv"].mean(1), 1e-4)
        close(record["top1"][1, 2, 1], ref["margin"].mean(1), 1e-4)
        close(record["energy_k"][1, 2, 1], k.double().reshape(2, -1, 64, 2).square().sum(-1).mean(1), 1e-4)
        assert not torch.allclose(record["var_cov"][1, 2, 0], record["var_cov"][1, 2, 1])                                # branches differ

        # partial steps leave the other cells marked unfilled
        partial = tmp / "partial"
        run("collect_pair_score_attribution.py", ["--traj", tmp / "traj", "--prompts", cfg, "--snapshot", tiny, "--out", partial,
                                                    "--ids", "d0000", "--steps", 1, 3, "--no-pair-cov", "--query-stride", 2])
        part = torch.load(partial / "d0000.pt")
        assert part["filled"][:, [1, 3]].all() and not part["filled"][:, [0, 2]].any() and "pair_cov" not in part
        close(part["var_cov"][:, [1, 3]], torch.load(files[0])["var_cov"][:, [1, 3]], 1e-5)

        # saved-capture source: conditional branch only
        sweep = tmp / "sweep"
        torch.manual_seed(9)
        phase = torch.rand(8, 64) * 6
        (sweep / "p00").mkdir(parents=True)
        (sweep / "p00" / "complete.json").write_text("{}")
        torch.save({"cos": phase.cos(), "sin": phase.sin(), "grid": (2, 2, 2), "pair_axes": ["t"] * 22 + ["h"] * 21 + ["w"] * 21}, sweep / "rope.pt")
        raw = {}
        for step in (1, 2):
            for layer in (0, 1):
                raw[step, layer] = {"q": (torch.randn(2, 8, 128) + .5).bfloat16(), "k": (torch.randn(2, 8, 128) + .5).bfloat16()}
                path = sweep / "p00" / f"step-{step:03d}"
                path.mkdir(exist_ok=True)
                torch.save(raw[step, layer], path / f"layer-{layer:03d}.pt")
        run("collect_pair_score_attribution.py", ["--sweep-root", sweep, "--out", tmp / "sweep-attribution"])
        saved = torch.load(tmp / "sweep-attribution" / "p00.pt")
        assert tuple(saved["var_cov"].shape) == (2, 2, 1, 2, 64) and saved["branches"] == ["conditional"]
        direct = attribute(raw[2, 1]["q"], raw[2, 1]["k"], phase.cos(), phase.sin())
        close(saved["adv"][1, 1, 0], direct["adv"], 1e-5)

        # analysis: distributions over cells and the selection overlay, recomputed here for the whole population
        torch.manual_seed(4)
        scores = torch.rand(2, STEPS, 2, 2, 64)
        mask_q = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(22).indices, True)
        mask_k = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, torch.rand_like(scores).topk(22).indices, True)
        torch.save({"variants": {"separate": {"mask_q": mask_q, "mask_k": mask_k}, "shared": {"mask_q": mask_q, "mask_k": mask_q.clone()}},
                    "pairs": 22, "damage_q": torch.rand_like(scores), "damage_k": torch.rand_like(scores)}, tmp / "selection.pt")
        out = tmp / "analysis"
        run("analyze_pair_score_attribution.py", ["--stats", stats, "--selection", tmp / "selection.pt", "--out", out, "--atlas-prompts", 1])
        report = json.loads((out / "report.json").read_text())
        assert report["cells"] == 4 * 2 * STEPS * 2 * 2 and report["shape"]["prompts"] == 4
        records = [torch.load(f) for f in files]
        share = np.stack([(r["var_cov"].double() / r["row_var"].double()[..., None]).numpy() for r in records])        # [prompt, layer, step, branch, head, pair]
        levels = [0, .05, .25, .5, .75, .95, 1]
        assert np.allclose(report["pair_share"]["var_cov_share"]["all"], np.quantile(share.reshape(-1, 64), levels, axis=0).T, atol=1e-6)
        assert np.allclose(report["pair_share"]["var_cov_share"]["by_step"][2], np.quantile(share[:, :, 2].reshape(-1, 64), levels, axis=0).T, atol=1e-6)
        assert np.allclose(report["pair_share"]["var_cov_share"]["by_branch"][1], np.quantile(share[:, :, :, 1].reshape(-1, 64), levels, axis=0).T, atol=1e-6)
        union = (mask_q | mask_k).numpy()
        assert np.allclose(report["selection"]["separate"]["union_var_cov_share"]["all"], np.quantile((share * union).sum(-1), levels), atol=1e-6)
        deleted = np.stack([torch.einsum("lsbhf,lsbhfg,lsbhg->lsbh", torch.from_numpy(union).float(), r["pair_cov"].float(), torch.from_numpy(union).float()).numpy() for r in records])
        assert np.allclose(report["selection"]["separate"]["union_deleted_logit_variance"]["all"], np.quantile(deleted, levels), atol=1e-6)
        assert report["selection"]["shared"]["pairs_in_union"][3] == 22 and report["selection"]["separate"]["pairs_in_union"][0] >= 22
        own = np.stack([(r["var_own"].double() / r["var_own"].double().sum(-1, keepdim=True)).numpy() for r in records])
        k50 = (np.sort(own, -1)[..., ::-1].cumsum(-1) < .5).sum(-1) + 1
        assert report["concentration"]["pairs_for_50_percent_counts"] == np.bincount(k50.reshape(-1), minlength=65).tolist()
        assert sum(report["largest_pair_counts"]["adv_share"]["all"]) == report["cells"]
        assert "cross_prompt_stability" in report and (out / "REPORT.md").read_text().count("| 63 | w |") == 5
        assert (out / "figures" / "pair-share-distribution-adv_share.png").exists()
        assert (out / "figures" / "atlas-d0000-unconditional-var_own_share.png").exists()
        printed = run("analyze_pair_score_attribution.py", ["--stats", stats, "--cell", "d0001", 1, 2, "unconditional", 0])
        line = next(l for l in printed.splitlines() if l.startswith("  17 "))
        assert abs(float(line.split()[2]) - share[1, 1, 2, 1, 0, 17]) < 1e-5
        # a pair whose center-model contribution is constant over keys to rounding noise (as in the lab records: stored
        # row variance about -5e-17) has an undefined center_pair_r; it is counted, never turned into a huge number
        noisy = tmp / "noisy"
        noisy.mkdir()
        for f in files:
            r = torch.load(f)
            if f.stem == "d0000":
                r["center_var_own"][1, 2, 1, 0, 21] = -5.2293e-17
                r["center_cross_own"][1, 2, 1, 0, 21] = 1.1937e-10
            torch.save(r, noisy / f.name)
        run("analyze_pair_score_attribution.py", ["--stats", noisy, "--out", tmp / "noisy-analysis", "--no-figures"])
        text = (tmp / "noisy-analysis" / "report.json").read_text()
        assert "NaN" not in text and "Infinity" not in text
        clean, dirty = report["pair_center_r"], json.loads(text)["pair_center_r"]
        assert dirty["undefined_entries"] == clean["undefined_entries"] + 1
        assert dirty["undefined_entries_per_pair"][21] == clean["undefined_entries_per_pair"][21] + 1
        assert dirty["entries"] == report["cells"] * 64 and dirty["defined_entries_outside_unit_interval"] == 0
        assert all(-1 - 1e-4 <= v <= 1 + 1e-4 for pair in dirty["all"] for v in pair if v is not None)
        assert A.shares(torch.load(noisy / "d0000.pt"))["center_pair_r"][1, 2, 1, 0, 21] != A.shares(torch.load(noisy / "d0000.pt"))["center_pair_r"][1, 2, 1, 0, 21]   # NaN
        broken = torch.load(files[0])
        broken["row_var"][0, 0, 0, 0] = 0
        try:
            A.build_report([broken], list(range(STEPS)))
            raise AssertionError("a non-finite share must stop the analysis")
        except FloatingPointError as error:
            assert "var_cov_share" in str(error)
        # score share of named pair sets (shared22-like, a nested shared32-like set, and what the larger set adds)
        mask_32 = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(32).indices, True)
        torch.save({"variants": {"shared": {"mask_q": mask_32, "mask_k": mask_32.clone()}}, "pairs": 32}, tmp / "selection32.pt")
        run("selection_score_share.py", ["--stats", stats, "--set", "s22", tmp / "selection.pt", "shared", "--set", "s32", tmp / "selection32.pt", "shared",
                                         "--set", "sep", tmp / "selection.pt", "separate", "--difference", "added", "s32", "s22", "--out", out, "--tag", "test"])
        sets = json.loads((out / "selection-shares-test.json").read_text())
        adv = np.stack([(r["adv"].double() / r["adv_total"].double()[..., None]).numpy() for r in records])
        m22, m32 = mask_q.numpy(), mask_32.numpy()
        assert sets["entries"] == report["cells"] and sets["sets"]["s32"]["pairs_per_active_cell"] == [32.0] * 7 and sets["sets"]["added"]["pairs_per_active_cell"] == [10.0] * 7
        assert sets["set_sources"]["added"]["pairs_of_b_outside_a"] == 0 and sets["sets"]["sep"]["pairs_per_active_cell"][0] >= 22
        assert np.allclose(sets["sets"]["s32"]["adv_share"]["all"], np.quantile((adv * m32).sum(-1), levels), atol=1e-6)
        assert np.allclose(sets["sets"]["added"]["var_cov_share"]["all"], np.quantile((share * (m32 & ~m22)).sum(-1), levels), atol=1e-6)
        assert np.allclose(sets["sets"]["s22"]["var_cov_share"]["by_layer"][1], np.quantile((share * m22).sum(-1)[:, 1], levels), atol=1e-6)
        assert np.allclose(sets["sets"]["s22"]["var_cov_share"]["all"], report["selection"]["shared"]["union_var_cov_share"]["all"], atol=1e-9)
        assert np.allclose(sets["sets"]["sep"]["deleted_logit_variance"]["all"], report["selection"]["separate"]["union_deleted_logit_variance"]["all"], atol=1e-9)
        assert np.allclose(sets["sets"]["s32"]["reference"]["smallest_possible_var_own_share"]["all"], np.quantile(np.sort(own, -1)[..., :32].sum(-1), levels), atol=1e-6)
        assert "reference" not in sets["sets"]["sep"] and f"| s32 | 32 | {report['cells']:,} | adv_share |" in (out / "SELECTION-SHARES-test.md").read_text()

        def score_share(recs, mask):
            """Plain percentage of the attention-weighted raw logit carried by the masked pairs, per prompt x cell."""
            raw = np.stack([(r["adv"].double() + r["mean_contribution"].double()).numpy() for r in recs])
            total = raw.sum(-1)
            value = (raw * mask).sum(-1) / total
            value[~(total > .01 * np.abs(raw).sum(-1))] = np.nan
            return value
        expected = score_share(records, m32)
        assert sets["sets"]["s32"]["entries"] == report["cells"] and np.allclose(sets["sets"]["s32"]["attn_score_share"]["all"], np.nanquantile(expected, levels), atol=1e-6)
        assert sets["sets"]["s32"]["attn_score_share_undefined_entries"] == int(np.isnan(expected).sum())

        # a trained student: its own masks, means and exemptions; replayed on the same stored teacher states
        from learned_center import LearnedCenter
        student = write_tiny_snapshot(tmp / "student")
        center = LearnedCenter(layers=2, steps=STEPS, heads=2, exempt_layers=(0,), exempt_steps=(0,), mode="mean", selection={"mask_q": mask_q, "mask_k": mask_q.clone()})
        torch.manual_seed(8)
        with torch.no_grad():
            center.init_from_calibration({"mean_q": torch.randn(2, STEPS, 2, 2, 128), "mean_k": torch.randn(2, STEPS, 2, 2, 128)})
        torch.save({"state_dict": center.state_dict(), "mode": "mean"}, student / "learned_center.pt")
        student_stats = tmp / "student-attribution"
        run("collect_pair_score_attribution.py", ["--traj", tmp / "traj", "--prompts", cfg, "--snapshot", tiny, "--student", student, "--out", student_stats, "--chunk", 3])
        srecs = [torch.load(f) for f in sorted(student_stats.glob("*.pt"))]
        srec = srecs[1]
        assert len(srecs) == 4 and srec["student_mode"] == "mean" and "student" in srec["source"]
        assert not srec["replaced_q"][0].any() and not srec["replaced_q"][:, 0].any() and (srec["replaced_q"][1, 1:].sum(-1) == 22).all()
        assert torch.equal(srec["replaced_q"][1, 2], mask_q[1, 2]) and max(srec["replaced_pairs_constant_max_deviation"].values()) < 1e-3
        on, off = mask_q[1, 2, 1, 0], ~mask_q[1, 2, 1, 0]
        assert torch.allclose(srec["norm_q"][1, 2, 1, 0][on], srec["center_norm_q"][1, 2, 1, 0][on], atol=1e-4)                   # constants: |mean| = mean |.|
        assert (srec["norm_q"][1, 2, 1, 0][off] - srec["center_norm_q"][1, 2, 1, 0][off]).min() > 1e-3                            # untouched pairs still vary
        assert torch.equal(srec["var_cov"][0], records[1]["var_cov"][0]) and torch.equal(srec["adv"][1, 0], records[1]["adv"][1, 0])   # exempt layer and step = teacher
        assert not torch.allclose(srec["var_cov"][1, 2], records[1]["var_cov"][1, 2])
        assert all(v["relative_max_err"] <= 1e-3 for v in srec["attention_output_validation"].values())
        # the active cell (layer 1, step 2, unconditional) recomputed with the same replacement attached first
        model = WanTransformer3DModel.from_pretrained(student, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        live = LearnedCenter.from_checkpoint(torch.load(student / "learned_center.pt")).attach(model)
        live.set_context(torch.tensor([2, 2]), torch.tensor([0, 1]))
        seen = {}
        handles = [getattr(model.blocks[1].attn1, "norm_" + s_).register_forward_hook(lambda m, i, o, s_=s_: seen.__setitem__(s_, o.detach().clone())) for s_ in ("q", "k")]
        handles.append(model.rope.register_forward_hook(lambda m, i, o: seen.__setitem__("rope", o)))
        with torch.inference_mode():
            model(hidden_states=item["x"].bfloat16().unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous(), timestep=torch.full((2,), float(item["timestep"])),
                  encoder_hidden_states=torch.cat([item["positive"], item["negative"]]).bfloat16(), return_dict=False)
        [h.remove() for h in handles]
        live.detach()
        cos, sin = seen["rope"][0][0, :, 0, 0::2].float(), seen["rope"][1][0, :, 0, 1::2].float()
        q, k = [seen[s_][1].float().view(-1, 2, 128).transpose(0, 1) for s_ in ("q", "k")]
        ref = brute(rotate(q, cos, sin, torch.bfloat16), rotate(k, cos, sin, torch.bfloat16))
        close(srec["var_cov"][1, 2, 1], ref["g"].sum(2), 1e-4)
        close(srec["adv"][1, 2, 1], ref["adv"].mean(1), 1e-4)
        close(srec["adv"][1, 2, 1] + srec["mean_contribution"][1, 2, 1], (ref["p"][..., None] * ref["contrib"]).sum(2).mean(1), 1e-4)
        # shares of the replaced pairs in the student, and of the same pairs and cells in the teacher
        run("selection_score_share.py", ["--stats", student_stats, "--record-masks", "replaced", "--out", out, "--tag", "student"])
        run("selection_score_share.py", ["--stats", stats, "--masks-from-record", "replaced", student_stats / "d0000.pt", "--out", out, "--tag", "teacher-same-pairs"])
        in_student = json.loads((out / "selection-shares-student.json").read_text())["sets"]["replaced"]
        in_teacher = json.loads((out / "selection-shares-teacher-same-pairs.json").read_text())["sets"]["replaced"]
        replaced = (srec["replaced_q"] | srec["replaced_k"]).numpy()
        cells = replaced.any(-1)
        assert in_student["active_cells_per_prompt"] == int(cells.sum()) == 1 * (STEPS - 1) * 2 * 2 and in_student["entries"] == 4 * int(cells.sum()) == in_teacher["entries"]
        assert in_student["pairs_per_active_cell"] == [22.0] * 7 and in_student["attn_score_share"]["by_layer"][0][3] is None
        for got, recs in ((in_student, srecs), (in_teacher, records)):
            value = score_share(recs, replaced)[:, cells]
            assert np.allclose(got["attn_score_share"]["all"], np.nanquantile(value, levels), atol=1e-6)
            assert got["attn_score_share_undefined_entries"] == int(np.isnan(value).sum())
            adv_ = np.stack([(r["adv"].double() / r["adv_total"].double()[..., None]).numpy() for r in recs])
            assert np.allclose(got["adv_share"]["all"], np.quantile((adv_ * replaced).sum(-1)[:, cells], levels), atol=1e-6)
        assert "no active cells" in (out / "SELECTION-SHARES-student.md").read_text()

        # step similarity: each trajectory (prompt x layer x branch x head) compared with itself at two steps
        run("step_similarity_pair_score_attribution.py", ["--stats", stats, "--out", out])
        sim = json.loads((out / "step-similarity.json").read_text())
        assert sim["trajectories"] == 4 * 2 * 2 * 2 and sim["step_pairs"] == [[0, 1], [1, 2], [2, 3], [1, 3], [0, 3]]
        a, b = share[:, :, 0], share[:, :, 3]                                                                          # steps 0 and 3: [prompt, layer, branch, head, pair]
        ca, cb = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
        pearson = (ca * cb).sum(-1) / np.sqrt((ca ** 2).sum(-1) * (cb ** 2).sum(-1))
        table = sim["measures"]["var_cov_share"]
        assert np.allclose(table["pearson"]["all"][0][3], np.quantile(pearson, levels), atol=1e-5)
        assert np.allclose(table["moved_share"]["all"][0][3], np.quantile(.5 * np.abs(a - b).sum(-1), levels), atol=1e-5)
        assert np.allclose(table["pearson"]["by_layer"][1][4], np.quantile(pearson[:, 1], levels), atol=1e-5)          # layer 1, step pair (0, 3)
        assert np.allclose(table["pearson"]["by_branch"][1][4], np.quantile(pearson[:, :, 1], levels), atol=1e-5)
        top = lambda x: set(np.argsort(x)[-5:].tolist())
        common = [len(top(a[i]) & top(b[i])) for i in np.ndindex(a.shape[:-1])]
        assert np.allclose(table["top5"]["all"][0][3], np.quantile(common, levels))
        assert np.allclose(table["pearson"]["all"][2][2], 1, atol=1e-9) and np.allclose(table["top22"]["all"][1][1], 22) and np.allclose(table["moved_share"]["all"][3][3], 0)
        assert np.allclose(table["spearman"]["all"][0][3], np.array(table["spearman"]["all"][3][0]))
        assert (out / "figures" / "step-similarity-adv_share.png").exists() and "1 vs 4" in (out / "STEP-SIMILARITY.md").read_text()
        # individual pairs against step for selected heads: plotted numbers are single cell-pair values
        traj = tmp / "trajectories"
        run("plot_step_trajectories_pair_score_attribution.py", ["--stats", stats, "--out", traj, "--prompts", "d0001", "--branches", "unconditional",
                                                                   "--measure", "var_cov_share", "--heads", 2, "--pairs", 4])
        rows = [l.split(",") for l in (traj / "d0001-unconditional-var_cov_share.csv").read_text().splitlines()[1:]]
        assert rows and all(abs(float(v) - share[1, int(layer), int(step) - 1, 1, int(head), int(pair)]) < 1e-5 for layer, head, _, pair, step, v in rows)
        x = share[1, 1, :, 1]                                                                                          # layer 1, unconditional: [step, head, pair]
        order = np.argsort(-np.median(x.max(-1), 0), kind="stable")
        drawn = {(int(head), int(position)) for layer, head, position, *_ in rows if layer == "1"}
        assert drawn == {(int(order[0]), 1), (int(order[1]), 2)}
        pairs = sorted({int(pair) for layer, head, _, pair, *_ in rows if layer == "1" and int(head) == int(order[0])})
        expected = {int(p) for s_ in (0, (STEPS - 1) // 2, STEPS - 1) for p in np.argsort(-x[s_, order[0]], kind="stable")[:3]}
        assert len(pairs) <= 4 and set(pairs) <= expected and int(np.argmax(x[0, order[0]])) in expected
        assert len(rows) == sum(1 for _ in rows) and all(len({r[4] for r in rows if r[:2] == [l_, h_] and r[3] == p_}) == STEPS for l_, h_, _, p_, *_ in rows[:8])
        assert sorted(p.name for p in traj.glob("*.png")) == ["d0001-unconditional-var_cov_share-layer-00.png", "d0001-unconditional-var_cov_share-layer-01.png"]
        run("analyze_pair_score_attribution.py", ["--stats", tmp / "sweep-attribution", "--out", tmp / "sweep-analysis", "--no-figures"])
        assert json.loads((tmp / "sweep-analysis" / "report.json").read_text())["shape"]["branches"] == 1
    print("test_collector_and_analysis OK")


def test_cell_pair_scores():
    """cell_pair_scores.py: one cell captured from a tiny random Wan, every column against explicit sums."""
    from diffusers import WanTransformer3DModel
    from train_learned_center import TrajectoryStore
    import cell_pair_scores as C
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fake_trajectories(tmp / "traj")
        tiny = write_tiny_snapshot(tmp / "tiny")
        torch.manual_seed(11)
        mask = torch.rand(2, STEPS, 2, 2, 64) < .3
        other = torch.rand(2, STEPS, 2, 2, 64) < .3
        torch.save({"variants": {"shared": {"mask_q": mask, "mask_k": mask.clone()}, "separate": {"mask_q": mask, "mask_k": other}}}, tmp / "selection.pt")
        layer, step, branch, head = 1, 3, 1, 1                                    # step is 1-based: the third stored state
        out = run("cell_pair_scores.py", ["capture", "--traj", tmp / "traj", "--snapshot", tiny, "--id", "d0002", "--layer", layer, "--step", step,
                                          "--branch", "unconditional", "--head", head, "--out", tmp / "cell",
                                          "--set", "a", tmp / "selection.pt", "shared", "--set", "b", tmp / "selection.pt", "separate"])
        folder = tmp / "cell" / "d0002-L01-S03-unconditional-H01"
        assert "CAPTURED" in out and (folder / "cell.npz").exists()
        run("cell_pair_scores.py", ["scores", folder, "--cpu"])
        report = json.loads((folder / "pair-scores.json").read_text())
        assert report["cell"]["attention_output_validation"]["relative_max_err"] <= 1e-3 and report["cell"]["dtype"] == "bfloat16"
        sets = {"a": mask[layer, step - 1, branch, head], "b": mask[layer, step - 1, branch, head] | other[layer, step - 1, branch, head]}
        assert all(report["sets"][name] == torch.nonzero(m).flatten().tolist() for name, m in sets.items())

        # the same cell from tensors hooked here
        model = WanTransformer3DModel.from_pretrained(tiny, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        item = TrajectoryStore(tmp / "traj", ["d0002"]).item("d0002", step - 1)
        seen = {}
        handles = [getattr(model.blocks[layer].attn1, "norm_" + s).register_forward_hook(lambda m, i, o, s=s: seen.__setitem__(s, o.detach().clone())) for s in ("q", "k")]
        handles.append(model.rope.register_forward_hook(lambda m, i, o: seen.__setitem__("rope", o)))
        with torch.inference_mode():
            x = item["x"].bfloat16().unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
            model(hidden_states=x, timestep=torch.full((2,), float(item["timestep"])),
                  encoder_hidden_states=torch.cat([item["positive"], item["negative"]]).bfloat16(), return_dict=False)
        [h.remove() for h in handles]
        cos, sin = seen["rope"][0][0, :, 0, 0::2].float(), seen["rope"][1][0, :, 0, 1::2].float()
        q, k = [seen[s][branch].float().view(-1, 2, 128).transpose(0, 1) for s in ("q", "k")]                          # [heads, tokens, 128]
        tensors, loaded_sets, meta = C.load_cell(folder)
        close(tensors["q"], q[head], 0)
        close(tensors["k"], k[head], 0)
        ref = brute(rotate(q, cos, sin, torch.bfloat16), rotate(k, cos, sin, torch.bfloat16))
        contrib, p = ref["contrib"][head], ref["p"][head]                                                               # [query, key, pair], [query, key]
        n = p.shape[0]
        levels = torch.tensor(C.LEVELS, dtype=torch.float64)

        def expected(x):
            centred = x - x.mean(1, keepdim=True)
            per_query = (p * x).sum(1)
            e = {"all_pos": x.clamp(min=0).sum() / n ** 2, "all_neg": x.clamp(max=0).sum() / n ** 2, "all_pos_frac": (x > 0).double().mean(),
                 "att_pos": (p * x.clamp(min=0)).sum() / n, "att_neg": (p * x.clamp(max=0)).sum() / n, "att_pos_mass": (p * (x > 0)).sum() / n,
                 "all_net": x.mean(), "all_abs": x.abs().mean(), "att_net": per_query.mean(), "att_abs": (p * x.abs()).sum() / n,
                 "qry_pos_frac": (per_query > 0).double().mean(), "ctr_all_abs": centred.abs().mean(),
                 "ctr_att_pos": (p * centred.clamp(min=0)).sum() / n, "ctr_att_neg": (p * centred.clamp(max=0)).sum() / n,
                 "ctr_att_net": (p * centred).sum() / n, "ctr_att_abs": (p * centred.abs()).sum() / n}
            e.update({f"qry_{name}": v for name, v in zip(C.LEVEL_NAMES, torch.quantile(per_query, levels))})
            e.update({f"all_{name}": v for name, v in zip(C.LEVEL_NAMES, torch.quantile(x.flatten(), levels))})
            return {key: float(v) for key, v in e.items()}

        def check(row, x):
            want = expected(x)
            for key, value in want.items():
                assert abs(row[key] - value) < 2e-5 * max(1, abs(value)), (row["row"], row["name"], key, row[key], value)
            return want

        rows = report["rows"]
        pairs = [r for r in rows if r["row"] == "pair"]
        assert len(pairs) == 64 and [r["name"] for r in pairs] == [str(f) for f in range(64)] and len(rows) == 64 + 1 + 2 * 2
        wanted = [check(r, contrib[:, :, f]) for f, r in enumerate(pairs)]
        assert any(w["att_pos"] > 1e-4 and w["att_neg"] < -1e-4 for w in wanted)                                        # both signs really occur
        for column in ("all_abs", "att_abs", "ctr_all_abs", "ctr_att_abs", "att_net", "ctr_att_net"):
            whole = sum(w[column] for w in wanted)
            assert all(abs(r[column + "_share"] - w[column] / whole) < 1e-4 for r, w in zip(pairs, wanted)), column
            assert abs(sum(r[column + "_share"] for r in pairs) - 1) < 1e-9
        check(next(r for r in rows if r["row"] == "logit"), ref["logits"][head])
        for name, m in sets.items():
            assert [r[f"in_{name}"] for r in pairs] == m.int().tolist()
            added = next(r for r in rows if r["row"] == "sum of pair rows" and r["name"] == name)
            assert added["pairs"] == int(m.sum())
            for column in (*C.ADDITIVE, *C.SHARES):
                assert abs(added[column] - sum(r[column] for r, inside in zip(pairs, m) if inside)) < 1e-12, column
            check(next(r for r in rows if r["row"] == "set score" and r["name"] == name), contrib[:, :, m].sum(-1))
        # the two earlier quantities are the signed columns of this table
        direct = attribute(q.bfloat16(), k.bfloat16(), cos, sin)                                                        # bf16 input: rotated tensors are rounded as in the model
        close(torch.tensor([r["ctr_att_net"] for r in pairs]), direct["adv"][head], 2e-5)
        close(torch.tensor([r["att_net"] for r in pairs]), direct["adv"][head] + direct["mean_contribution"][head], 2e-5)
        assert report["info"]["largest_abs_difference_between_logit_and_sum_of_pair_scores"] < 1e-4 and report["info"]["tokens"] == n
        table = list(csv.DictReader(open(folder / "pair-scores.csv")))
        assert len(table) == len(rows) and abs(float(table[5]["att_pos"]) - pairs[5]["att_pos"]) < 1e-7 and table[64]["row"] == "logit"
        text = (folder / "PAIR-SCORES.md").read_text()
        assert "layer 1, denoising step 3" in text and "unconditional branch, head 1" in text and text.count("\n| ") >= 64 + 5
    print("test_cell_pair_scores OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_cell_pair_scores()
    test_attribute_against_brute_force()
    test_constant_tokens_are_the_center_model()
    test_bf16_and_query_stride()
    test_collector_and_analysis()
    print("ALL TESTS OK")
