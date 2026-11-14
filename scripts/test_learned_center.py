"""CPU tests for the learned-center student and training loop on a tiny random Wan.

1. LearnedCenter with mu set to the current-token means reproduces scripts/center_replacement.replace_with_center
   (mode "high") exactly, for every layer; the exempt layer/step and the "off"/"delete" modes behave as specified;
   gradients reach mu_q/mu_k only.
2. Fake trajectories + stages 1, 2 and 3 of train_learned_center.py run end to end (--no-fsdp), the student
   checkpoint loads back, and stage 3 fills the on-policy buffer.

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
            getattr(center, "mu_" + side)[1, 2, 0] = means[0]
            getattr(center, "mu_" + side)[1, 2, 1] = means[1]
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
        diff = (saved["mu_q"] - calib["mean_q"]).abs()
        assert diff[0].sum() == 0 and diff[1, 0].sum() == 0, "exempt cells moved"
        assert diff[1, 1:].sum() > 0, "active cells did not train"
        print("test_stage1_checkpoint_only_mu_differs OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_replacement_exact()
    test_stage1_checkpoint_only_mu_differs()
    test_training_stages()
    print("ALL TESTS OK")
