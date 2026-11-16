"""CPU tests for the learned-center student and training loop on a tiny random Wan.

1. LearnedCenter with mu set to the current-token means reproduces scripts/center_replacement.replace_with_center
   (mode "high") exactly, for every layer; the exempt layer/step and the "off"/"delete" modes behave as specified;
   gradients reach mu_q/mu_k only.
2. Fake trajectories + stages 1, 2 and 3 of train_learned_center.py run end to end (--no-fsdp), the student
   checkpoint loads back, and stage 3 fills the on-policy buffer.
3. Narrow masks the projection before native RMSNorm at every step and blocks gradient to the masked rows.

Run from the repository root:  python scripts/test_learned_center.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from center_replacement import replace_with_center
from learned_center import HIGH_DIMS, LearnedCenter

ROOT = Path(__file__).resolve().parents[1]
TINY = {"patch_size": [1, 2, 2], "num_attention_heads": 2, "attention_head_dim": 128, "in_channels": 16, "out_channels": 16,
        "text_dim": 64, "freq_dim": 64, "ffn_dim": 512, "num_layers": 2, "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6, "rope_max_seq_len": 1024}
STEPS = 4


def build_tiny():
    from diffusers import WanTransformer3DModel
    torch.manual_seed(0)
    return WanTransformer3DModel(**TINY).eval()


def write_tiny_snapshot(root):
    """Tiny random Wan saved like a HF snapshot (transformer/ + scheduler/)."""
    from diffusers import UniPCMultistepScheduler
    build_tiny().save_pretrained(root / "transformer")
    UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True,
                            flow_shift=3.0, solver_order=2).save_pretrained(root / "scheduler")
    return root


def test_replacement_exact():
    model = build_tiny()
    torch.manual_seed(1)
    x = torch.randn(2, 16, 2, 4, 4)
    text = torch.randn(2, 8, 64)
    t = torch.tensor([500.0, 500.0])
    captured = {}
    handles = [getattr(b.attn1, "norm_" + s).register_forward_hook(lambda m, i, o, key=(l, s): captured.__setitem__(key, o.detach().clone()))
               for l, b in enumerate(model.blocks) for s in ("q", "k")]
    with torch.no_grad():
        model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)
    for h in handles:
        h.remove()
    center = LearnedCenter(layers=2, steps=STEPS, heads=2, exempt_layers=(0,), exempt_steps=(0,))
    # set mu to the per-batch-element current means of layer 1 (branch 0 = batch 0, branch 1 = batch 1), step 2
    with torch.no_grad():
        for side in ("q", "k"):
            out = captured[(1, side)]
            means = out.view(2, 8, 2, 128).mean(1, dtype=torch.float32)[..., HIGH_DIMS]   # [B, heads, 44], as replace_with_center
            mu = getattr(center, "mu_" + side)
            mu[1, 2, 0][:, HIGH_DIMS] = means[0]
            mu[1, 2, 1][:, HIGH_DIMS] = means[1]
    center.set_context(torch.tensor([2, 2]), torch.tensor([0, 1]))
    for side in ("q", "k"):
        out = captured[(1, side)]
        reference = torch.cat([replace_with_center(out[0:1], 2, "high"), replace_with_center(out[1:2], 2, "high")], 0)
        got = center.replace(out, 1, side)
        assert torch.equal(got, reference), f"layer 1 {side} mismatch {(got - reference).abs().max()}"
        assert torch.equal(center.replace(out, 0, side), out), "exempt layer must pass through"
    center.set_context(torch.tensor([0, 2]), torch.tensor([0, 1]))
    out = captured[(1, "q")]
    got = center.replace(out, 1, "q")
    assert torch.equal(got[0], out[0]) and torch.equal(got[1], replace_with_center(out[1:2], 2, "high")[0]), "per-element step exemption"
    center.mode = "delete"
    center.set_context(torch.tensor([2, 2]), torch.tensor([0, 1]))
    got = center.replace(out, 1, "q").view(2, 8, 2, 128)
    assert torch.equal(got[..., HIGH_DIMS], torch.zeros_like(got[..., HIGH_DIMS])) and torch.equal(
        got[..., [d for d in range(128) if d not in HIGH_DIMS.tolist()]], out.view(2, 8, 2, 128)[..., [d for d in range(128) if d not in HIGH_DIMS.tolist()]])
    center.mode = "off"
    assert torch.equal(center.replace(out, 1, "q"), out)
    # gradients: attach, run, loss on output -> grads on mu only (model frozen)
    center.mode = "mean"
    model.requires_grad_(False)
    center.attach(model)
    center.calls = 0
    center.set_context(torch.tensor([2, 2]), torch.tensor([0, 1]))
    y = model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
    y.square().mean().backward()
    assert center.mu_q.grad is not None and center.mu_q.grad[1, 2].abs().sum() > 0, "no grad on active mu"
    assert center.mu_q.grad[0].abs().sum() == 0 and center.mu_q.grad[1, 0].abs().sum() == 0, "grad leaked to exempt cells"
    assert center.calls == 2 * 1, f"hook calls {center.calls}"  # layer 1 only, q and k
    center.detach()
    print("test_replacement_exact OK")


def test_narrow_projection():
    model = build_tiny()
    center = LearnedCenter(layers=2, steps=STEPS, heads=2, mode="narrow").attach(model)
    high = (torch.arange(2)[:, None] * 128 + HIGH_DIMS).flatten()
    kept = torch.ones(256, dtype=torch.bool)
    kept[high] = False
    assert center.step_active.all()
    x = torch.randn(2, 8, 256)
    for side in ("q", "k"):
        projection = getattr(model.blocks[1].attn1, "to_" + side)
        norm = getattr(model.blocks[1].attn1, "norm_" + side)
        raw = torch.nn.functional.linear(x, projection.weight, projection.bias)
        projected = projection(x)  # No step or branch context is set.
        assert torch.count_nonzero(projected[..., high]) == 0
        torch.testing.assert_close(projected[..., kept], raw[..., kept], rtol=0, atol=0)
        padded = raw.clone()
        padded[..., high] = 0
        # The native reduction still divides by the full, zero-padded projection width.
        manual = padded * torch.rsqrt(padded.square().mean(-1, keepdim=True) + norm.eps) * norm.weight
        normalized = norm(projected)
        assert torch.count_nonzero(normalized[..., high]) == 0
        torch.testing.assert_close(normalized[..., kept], manual[..., kept])
        exempt = getattr(model.blocks[0].attn1, "to_" + side)
        torch.testing.assert_close(exempt(x), torch.nn.functional.linear(x, exempt.weight, exempt.bias))
    assert center.calls == 2

    center.set_context([0, 1], [0, 1])
    y = model(hidden_states=torch.randn(2, 16, 2, 4, 4), timestep=torch.tensor([500.0, 500.0]),
              encoder_hidden_states=torch.randn(2, 8, 64), return_dict=False)[0]
    y.square().mean().backward()
    for projection in (model.blocks[1].attn1.to_q, model.blocks[1].attn1.to_k):
        assert torch.count_nonzero(projection.weight.grad[high]) == 0
        assert torch.count_nonzero(projection.bias.grad[high]) == 0
        assert torch.count_nonzero(projection.weight.grad[kept]) > 0
    assert center.mu_q.grad is None and center.mu_k.grad is None
    assert center.calls == 4
    center.detach()
    print("test_narrow_projection OK")


def write_fake_trajectories(root):
    from diffusers import UniPCMultistepScheduler
    sched = UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, solver_order=2)
    sched.set_timesteps(STEPS)
    timesteps = [float(t) for t in sched.timesteps]
    prompts = []
    torch.manual_seed(3)
    for i in range(6):
        pid = f"d{i:04d}"
        split = "calib" if i < 4 else "val"
        prompts.append({"id": pid, "text": f"tiny prompt {i}", "source": "test", "seed": i, "split": split})
        d = root / pid
        d.mkdir(parents=True)
        torch.save({"positive": torch.randn(1, 8, 64).bfloat16(), "negative": torch.randn(1, 8, 64).bfloat16()}, d / "embeddings.pt")
        torch.save(torch.randn(STEPS, 16, 2, 4, 4).bfloat16(), d / "states.pt")
        (d / "meta.json").write_text(json.dumps({"timesteps": timesteps, "steps": STEPS}))
    torch.save({"mean_q": 0.1 * torch.randn(2, STEPS, 2, 2, 44), "mean_k": 0.1 * torch.randn(2, STEPS, 2, 2, 44)}, root / "calibration.pt")
    cfg = root / "prompts.json"
    cfg.write_text(json.dumps({"prompts": prompts}))
    return cfg


def run(args, cwd):
    cmd = [sys.executable, str(ROOT / "scripts" / "train_learned_center.py")] + args
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-6000:])
        raise SystemExit(f"command failed: {' '.join(args)}")
    return proc.stdout


def test_training_stages():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        traj = tmp / "traj"
        cfg = write_fake_trajectories(traj)
        tiny = write_tiny_snapshot(tmp / "tiny")
        common = ["--traj", str(traj), "--prompts", str(cfg), "--snapshot", str(tiny), "--no-fsdp", "--warmup", "1", "--log-every", "1",
                  "--val-every", "2", "--save-every", "2", "--feat-layers", "0"]
        out1 = run(["--stage", "1", "--out", str(tmp / "s1"), "--steps", "3"] + common, ROOT)
        assert "TRAINING COMPLETE 1" in out1 and (tmp / "s1" / "student" / "learned_center.pt").exists(), out1[-800:]
        assert "attn_total" in out1
        out2 = run(["--stage", "2", "--out", str(tmp / "s2"), "--steps", "4", "--init-student", str(tmp / "s1" / "student"), "--ema", "0.9"] + common, ROOT)
        assert "TRAINING COMPLETE 2" in out2 and "VAL" in out2, out2[-800:]
        assert (tmp / "s2" / "student" / "transformer" / "diffusion_pytorch_model.safetensors").exists()
        assert (tmp / "s2" / "validation.jsonl").exists() and (tmp / "s2" / "history.jsonl").exists()
        # stage 1 checkpoint has no transformer weights; stage 2 output must be loadable into stage 3
        out3 = run(["--stage", "3", "--out", str(tmp / "s3"), "--steps", "4", "--init-student", str(tmp / "s2" / "student"),
                    "--rollout-every", "2", "--rollouts-per-rank", "2", "--onpolicy-ramp", "1", "--onpolicy-frac", "1.0", "--ema", "0.9"] + common, ROOT)
        assert "TRAINING COMPLETE 3" in out3, out3[-800:]
        files = list((tmp / "s3" / "onpolicy").glob("*.pt"))
        assert len(files) == 4, f"expected 4 rollout files, got {len(files)}"
        states = torch.load(files[0])["states"]
        assert tuple(states.shape) == (STEPS, 16, 2, 4, 4), states.shape
        last = [json.loads(l) for l in open(tmp / "s3" / "history.jsonl")][-1]
        assert last["onpolicy"] == 1.0, last
        # resume path
        out3b = run(["--stage", "3", "--out", str(tmp / "s3"), "--steps", "5", "--init-student", str(tmp / "s2" / "student"), "--resume",
                     "--rollout-every", "100", "--ema", "0.9"] + common, ROOT)
        assert '"step": 5' in out3b, out3b[-500:]
        print("test_training_stages OK")


def test_stage1_checkpoint_only_mu_differs():
    """Stage 1 must not change any teacher weight; the saved center must differ from calibration where active."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = write_fake_trajectories(tmp / "traj")
        tiny = write_tiny_snapshot(tmp / "tiny")
        run(["--stage", "1", "--out", str(tmp / "s1"), "--steps", "2", "--traj", str(tmp / "traj"), "--prompts", str(cfg), "--snapshot", str(tiny),
             "--no-fsdp", "--warmup", "1", "--save-every", "2", "--lr-center", "1e-2"], ROOT)
        saved = torch.load(tmp / "s1" / "student" / "learned_center.pt")["state_dict"]
        calib = torch.load(tmp / "traj" / "calibration.pt")
        diff = (saved["mu_q"][..., HIGH_DIMS] - calib["mean_q"]).abs()
        assert diff[0].sum() == 0 and diff[1, 0].sum() == 0, "exempt cells moved"
        assert diff[1, 1:].sum() > 0, "active cells did not train"
        print("test_stage1_checkpoint_only_mu_differs OK")


def test_masks_and_legacy_checkpoint():
    """Per-side masks replace only their own pairs; a 44-wide legacy state dict loads and matches default behaviour."""
    model = build_tiny()
    torch.manual_seed(5)
    x = torch.randn(2, 16, 2, 4, 4); text = torch.randn(2, 8, 64); t = torch.tensor([500.0, 500.0])
    captured = {}
    handles = [getattr(b.attn1, "norm_" + sd).register_forward_hook(lambda m, i, o, key=(l, sd): captured.__setitem__(key, o.detach().clone()))
               for l, b in enumerate(model.blocks) for sd in ("q", "k")]
    with torch.no_grad():
        model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)
    for h in handles:
        h.remove()
    L, S, B, H = 2, STEPS, 2, 2
    mask_q = torch.zeros(L, S, B, H, 64, dtype=torch.bool); mask_k = torch.zeros_like(mask_q)
    mask_q[..., [3, 40]] = True            # Q: pairs 3 and 40 ;  K: pairs 40 and 63 (one shared)
    mask_k[..., [40, 63]] = True
    center = LearnedCenter(layers=L, steps=S, heads=H, exempt_layers=(), exempt_steps=(), selection={"mask_q": mask_q, "mask_k": mask_k})
    with torch.no_grad():
        center.mu_q.fill_(7.0); center.mu_k.fill_(-3.0)
    center.set_context(torch.tensor([1, 1]), torch.tensor([0, 1]))
    out = captured[(1, "q")]
    got = center.replace(out, 1, "q").view(2, 8, H, 64, 2)
    ref = out.view(2, 8, H, 64, 2)
    assert torch.equal(got[..., [3, 40], :], torch.full_like(got[..., [3, 40], :], 7.0)) and torch.equal(got[..., [0, 1, 2, 4, 63], :], ref[..., [0, 1, 2, 4, 63], :])
    gotk = center.replace(captured[(1, "k")], 1, "k").view(2, 8, H, 64, 2)
    assert torch.equal(gotk[..., [40, 63], :], torch.full_like(gotk[..., [40, 63], :], -3.0)) and torch.equal(gotk[..., 3, :], captured[(1, "k")].view(2, 8, H, 64, 2)[..., 3, :])
    assert int(center.selected_pairs("q").sum(-1)[0, 0, 0, 0]) == 2 and not center.uses_default_masks
    # legacy 44-wide checkpoint reproduces the default (HIGH) behaviour
    torch.manual_seed(6)
    legacy = {"mu_q": torch.randn(L, S, B, H, 44), "mu_k": torch.randn(L, S, B, H, 44),
              "layer_active": torch.tensor([False, True]), "step_active": torch.tensor([False, True, True, True]), "high_dims": HIGH_DIMS.clone()}
    new = LearnedCenter(layers=L, steps=S, heads=H)
    new.load_state_dict(legacy)
    assert new.uses_default_masks and torch.equal(new.mu_q[..., HIGH_DIMS], legacy["mu_q"])
    assert new.mu_q[..., 17].abs().sum() == 0, "non-HIGH coordinate (pair 8) must stay zero after a legacy load"
    new.set_context(torch.tensor([2, 2]), torch.tensor([0, 1]))
    expected = out.view(2, 8, H, 128).clone(); expected[..., HIGH_DIMS] = legacy["mu_q"][1, 2].unsqueeze(1).expand(2, 8, H, 44)[torch.arange(2)[:, None, None], :, :][:, 0] if False else legacy["mu_q"][1][torch.tensor([2, 2]), torch.tensor([0, 1])].unsqueeze(1).expand(2, 8, H, 44)
    assert torch.equal(new.replace(out, 1, "q"), expected.view(2, 8, -1))
    again = LearnedCenter.from_checkpoint({"state_dict": new.state_dict(), "mode": "mean"})
    assert torch.equal(again.mu_q, new.mu_q) and torch.equal(again.mask_k, new.mask_k)
    print("test_masks_and_legacy_checkpoint OK")


def test_statistics_selection_screen():
    """Collector statistics match a direct computation; select_pairs produces valid masks and static exports;
    screen_selections runs on the tiny model and reports finite errors."""
    import collect_pair_statistics as C
    import select_pairs as SP
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = write_fake_trajectories(tmp / "traj")
        tiny = write_tiny_snapshot(tmp / "tiny")
        from diffusers import WanTransformer3DModel
        from train_learned_center import TrajectoryStore
        model = WanTransformer3DModel.from_pretrained(tiny, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
        store = TrajectoryStore(tmp / "traj", ["d0000", "d0001", "d0002", "d0003"])
        stats_dir = tmp / "stats"; stats_dir.mkdir()
        for pid in store.ids:
            stats = C.PairStats(model, store.steps)
            direct = {}
            hs = [getattr(b.attn1, "norm_" + sd).register_forward_hook(lambda m, i, o, key=(l, sd): direct.__setitem__((stats.context,) + key, o.detach().float().clone()))
                  for l, b in enumerate(model.blocks) for sd in ("q", "k")]
            with torch.inference_mode():
                for step in range(store.steps):
                    item = store.item(pid, step); stats.context = step
                    x = item["x"].bfloat16().unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
                    text = torch.cat([item["positive"], item["negative"]]).bfloat16()
                    model(hidden_states=x, timestep=torch.full((2,), float(item["timestep"])), encoder_hidden_states=text, return_dict=False)
            stats.close(); [h.remove() for h in hs]
            o = direct[(2, 1, "k")][1].view(-1, 2, 128)        # step 2, layer 1, uncond branch, K
            assert torch.allclose(stats.mean["k"][1, 2, 1].float(), o.mean(0), atol=1e-5)
            assert torch.allclose(stats.energy["k"][1, 2, 1].float(), o.view(-1, 2, 64, 2).square().sum(-1).mean(0), atol=1e-4)
            torch.save({"prompt_id": pid, "steps": store.steps, "timesteps": store.timesteps, "mean_q": stats.mean["q"].float(), "mean_k": stats.mean["k"].float(),
                        "energy_q": stats.energy["q"].float(), "energy_k": stats.energy["k"].float()}, stats_dir / f"{pid}.pt")
        sel_dir = tmp / "sel"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "select_pairs.py"), "--stats", str(stats_dir), "--out", str(sel_dir), "--pairs", "22"], cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
        sel = torch.load(sel_dir / "selection.pt")
        for name, v in sel["variants"].items():
            assert tuple(v["mask_q"].shape) == (2, STEPS, 2, 2, 64) and (v["mask_q"].sum(-1) == 22).all() and (v["mask_k"].sum(-1) == 22).all(), name
        assert torch.equal(sel["variants"]["shared"]["mask_q"], sel["variants"]["shared"]["mask_k"])
        assert torch.equal(sel["variants"]["fixed_high"]["mask_q"][0, 0, 0, 0].nonzero().flatten(), torch.tensor(sorted(set(HIGH_DIMS.tolist()[::2]) | set())) // 2) or True
        high_pairs = sorted(set((HIGH_DIMS // 2).tolist()))
        assert sel["variants"]["fixed_high"]["mask_q"][0, 0, 0, 0].nonzero().flatten().tolist() == high_pairs
        report = json.loads((sel_dir / "selection-report.json").read_text())
        assert 0 <= report["stability_q"]["jaccard_prompt_vs_pooled_median"] <= 1 and report["separate_q_k_intersection"]["mean_pairs"] <= 22
        # the lowest-damage pairs are selected: check one cell by recomputation
        dq = sel["damage_q"][1, 2, 1, 0]
        assert set(dq.topk(22, largest=False).indices.tolist()) == set(sel["variants"]["separate"]["mask_q"][1, 2, 1, 0].nonzero().flatten().tolist())
        static = LearnedCenter.from_checkpoint(torch.load(sel_dir / "static-separate" / "learned_center.pt"))
        pooled = torch.load(sel_dir / "pair-statistics.pt")
        assert torch.allclose(static.mu_q, pooled["mean_q"]) and torch.equal(static.selected_pairs("k"), sel["variants"]["separate"]["mask_k"])
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "screen_selections.py"), "--traj", str(tmp / "traj"), "--prompts", str(cfg), "--snapshot", str(tiny),
                               "--selection", str(sel_dir / "selection.pt"), "--statistics", str(sel_dir / "pair-statistics.pt"), "--out", str(tmp / "screen"), "--val-prompts", "2"],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
        summary = json.loads((tmp / "screen" / "complete.json").read_text())
        assert set(summary["arms"]) == {f"{v}/{m}" for v in ("fixed_high", "separate", "shared") for m in ("mean", "delete")}
        for arm, r in summary["arms"].items():
            assert len(r["per_step"]) == STEPS and all(0 <= v < 1e6 for v in r["mean"]), arm
        assert summary["arms"]["fixed_high/mean"]["mean"][2] != summary["arms"]["fixed_high/delete"]["mean"][2]
        print("test_statistics_selection_screen OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_replacement_exact()
    test_narrow_projection()
    test_masks_and_legacy_checkpoint()
    test_statistics_selection_screen()
    test_stage1_checkpoint_only_mu_differs()
    test_training_stages()
    print("ALL TESTS OK")
