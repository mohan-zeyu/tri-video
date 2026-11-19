"""CPU tests for the learned-center student and training loop on a tiny random Wan.

1. LearnedCenter with mu set to the current-token means reproduces scripts/center_replacement.replace_with_center
   (mode "high") exactly, for every layer; the exempt layer/step and the "off"/"delete" modes behave as specified;
   gradients reach mu_q/mu_k only.
2. Fake trajectories + stages 1, 2 and 3 of train_learned_center.py run end to end (--no-fsdp), the student
   checkpoint loads back, and stage 3 fills the on-policy buffer.
3. Narrow masks the projection before native RMSNorm at every step and blocks gradient to the masked rows;
   with per-cell masks it follows the step/branch context and the step exemption like the other modes.

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
    center = LearnedCenter(layers=2, steps=STEPS, heads=2, mode="narrow", exempt_steps=()).attach(model)   # the original control
    assert center.cell_invariant()
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


def test_narrow_selected():
    """Narrow with per-cell masks: zeros before the norm on exactly the cell's pairs, exempt step and layer untouched,
    context required when the masks vary by cell; a pooled (cell-invariant) selection works without context."""
    model = build_tiny()
    L, S, B, H = 2, STEPS, 2, 2
    torch.manual_seed(8)
    mask = torch.zeros(L, S, B, H, 64, dtype=torch.bool)
    for l in range(L):
        for st in range(S):
            for b in range(B):
                for h in range(H):
                    mask[l, st, b, h, torch.randperm(64)[:22]] = True
    center = LearnedCenter(layers=L, steps=S, heads=H, mode="narrow", exempt_layers=(0,), exempt_steps=(0,), selection={"mask_q": mask, "mask_k": mask})
    assert not center.cell_invariant()
    x = torch.randn(2, 8, 256)
    raw = model.blocks[1].attn1.to_q(x)
    try:
        center.replace(raw, 1, "q"); raise AssertionError("context must be required for per-cell narrow masks")
    except RuntimeError:
        pass
    center.set_context([2, 0], [1, 0])          # element 0: step 2 uncond (active); element 1: step 0 (exempt)
    got = center.replace(raw, 1, "q").view(2, 8, H, 128)
    m0 = center.mask_q[1, 2, 1]                                  # [H, 128]
    ref0 = raw.view(2, 8, H, 128)[0].clone(); ref0[:, m0] = 0
    assert torch.equal(got[0], ref0) and torch.equal(got[1], raw.view(2, 8, H, 128)[1])
    assert int(m0[0].sum()) == 44
    assert torch.equal(center.replace(raw, 0, "q"), raw), "exempt layer"
    center.attach(model); center.calls = 0
    center.set_context([2, 2], [0, 1])
    with torch.no_grad():
        model(hidden_states=torch.randn(2, 16, 2, 4, 4), timestep=torch.tensor([500.0, 500.0]), encoder_hidden_states=torch.randn(2, 8, 64), return_dict=False)
    assert center.calls == 2, center.calls                       # layer 1 only, q and k
    center.set_context([0, 0], [0, 1]); center.calls = 0
    with torch.no_grad():
        model(hidden_states=torch.randn(2, 16, 2, 4, 4), timestep=torch.tensor([500.0, 500.0]), encoder_hidden_states=torch.randn(2, 8, 64), return_dict=False)
    assert center.calls == 0, "exempt step must not count"
    center.detach()
    # pooled: same set at every step and branch; no context needed; equals dropping the projection rows
    pooled = mask[:, 0:1, 0:1].expand_as(mask).clone()
    center = LearnedCenter(layers=L, steps=S, heads=H, mode="narrow", exempt_layers=(0,), exempt_steps=(), selection={"mask_q": pooled, "mask_k": pooled})
    assert center.cell_invariant()
    got = center.replace(raw, 1, "q")
    proj = model.blocks[1].attn1.to_q
    w = proj.weight.clone(); bb = proj.bias.clone(); rows = center.mask_q[1, 0, 0].flatten()
    w[rows] = 0; bb[rows] = 0
    torch.testing.assert_close(got, torch.nn.functional.linear(x, w, bb), rtol=0, atol=0)
    center.set_context([1, 3], [0, 1])
    torch.testing.assert_close(center.replace(raw, 1, "q"), got, rtol=0, atol=0)
    # round trip through a checkpoint keeps masks, mode-independent
    again = LearnedCenter.from_checkpoint({"state_dict": center.state_dict(), "mode": "narrow"})
    assert again.mode == "narrow" and torch.equal(again.mask_k, center.mask_k) and again.cell_invariant()
    print("test_narrow_selected OK")


def test_nope():
    """NoPE pairs: (a) exact agreement with a reference processor that rotates only the non-NoPE pairs, per head and
    batch element; (b) all-NoPE attention is permutation-equivariant in the token grid (no positional signal left);
    (c) exempt steps and the empty mask leave the model untouched; (d) processors are restored on detach."""
    from diffusers.models.transformers.transformer_wan import _get_qkv_projections, WanAttnProcessor
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    class Reference:
        """diffusers' self-attention path with the rotation skipped on a per-(batch, head) pair mask."""
        def __init__(self, center, layer):
            self.center, self.layer = center, layer
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kw):
            q, k, v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            q, k, v = (t.unflatten(2, (attn.heads, -1)) for t in (attn.norm_q(q), attn.norm_k(k), v))
            cos, sin = rotary_emb
            B, T, H, D = q.shape
            step, branch = self.center.step, self.center.branch
            def rot(x):
                out = x.clone()
                for b in range(B):
                    if not bool(self.center.step_active[step[b]]):
                        keep = torch.ones(H, D, dtype=torch.bool)
                    else:
                        keep = ~self.center.mask_nope[self.layer, step[b], branch[b]]
                    xb = x[b]                                   # native dtypes, as diffusers computes it
                    x1, x2 = xb.unflatten(-1, (-1, 2)).unbind(-1)
                    c = cos[0, :, 0, 0::2][:, None, :]; sn = sin[0, :, 0, 1::2][:, None, :]
                    r = torch.empty_like(xb); r[..., 0::2] = x1 * c - x2 * sn; r[..., 1::2] = x1 * sn + x2 * c
                    out[b] = torch.where(keep[None], r.type_as(xb), xb)
                return out
            q, k = rot(q), rot(k)
            h = dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, parallel_config=None)
            h = h.flatten(2, 3).type_as(q)
            return attn.to_out[1](attn.to_out[0](h))

    model = build_tiny()
    L, S, B, H = 2, STEPS, 2, 2
    torch.manual_seed(12)
    means = torch.zeros(L, S, B, H, 64, dtype=torch.bool)
    for idx in torch.cartesian_prod(*[torch.arange(n) for n in (L, S, B, H)]):
        means[tuple(idx.tolist())][torch.randperm(64)[:42]] = True
    x = torch.randn(2, 16, 2, 4, 4); text = torch.randn(2, 8, 64); t = torch.tensor([500.0, 500.0])
    calib = {"mean_q": 0.3 * torch.randn(L, S, B, H, 128), "mean_k": 0.3 * torch.randn(L, S, B, H, 128)}
    def run(center, ctx):
        center.attach(model); center.set_context(*ctx)
        with torch.no_grad():
            y = model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
        center.detach(); return y
    # (a) wrapper == reference, including an exempt step in one batch element
    sel = {"mask_q": means, "mask_k": means, "mask_nope": ~means}
    center = LearnedCenter(layers=L, steps=S, heads=H, exempt_layers=(0,), exempt_steps=(0,), selection=sel); center.init_from_calibration(calib)
    assert center.uses_nope and int(center.selected_pairs("nope").sum(-1)[1, 1, 0, 0]) == 22
    for ctx in (([2, 3], [0, 1]), ([2, 0], [1, 0])):
        got = run(center, ctx)
        # reference: same means (norm hooks of a NoPE-free twin) + hand-rotated processor
        twin = LearnedCenter(layers=L, steps=S, heads=H, exempt_layers=(0,), exempt_steps=(0,), selection={"mask_q": means, "mask_k": means}); twin.init_from_calibration(calib)
        twin.attach(model); twin.set_context(*ctx); center.set_context(*ctx)
        originals = [(blk.attn1, blk.attn1.processor) for blk in model.blocks]
        model.blocks[1].attn1.set_processor(Reference(center, 1))
        with torch.no_grad():
            ref = model(hidden_states=x, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
        for attn, proc in originals: attn.set_processor(proc)
        twin.detach()
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        assert all(type(blk.attn1.processor) is WanAttnProcessor for blk in model.blocks), "processors not restored"
    assert center.nope_calls == 2, center.nope_calls            # layer 1 once per forward (exempt-step element is masked, not skipped)
    # (c) exempt step everywhere -> identical to no NoPE; empty mask -> no wrapper at all
    center.nope_calls = 0
    plain = LearnedCenter(layers=L, steps=S, heads=H, exempt_layers=(0,), exempt_steps=(0,), selection={"mask_q": means, "mask_k": means}); plain.init_from_calibration(calib)
    torch.testing.assert_close(run(center, ([0, 0], [0, 1])), run(plain, ([0, 0], [0, 1])), rtol=0, atol=0)
    assert center.nope_calls == 0
    plain.attach(model); assert plain.wrapped == []; plain.detach()
    # (b) all-NoPE, no replacement, no exemptions: permuting the token grid permutes the output
    allnope = LearnedCenter(layers=L, steps=S, heads=H, exempt_layers=(), exempt_steps=(),
                            selection={"mask_q": torch.zeros_like(means), "mask_k": torch.zeros_like(means), "mask_nope": torch.ones_like(means)})
    def tokens(lat):                      # [B,16,2,4,4] -> [B, 8 tokens, 16, 1, 2, 2] patches (1,2,2)
        return lat.view(2, 16, 2, 2, 2, 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(2, 8, 16, 1, 2, 2)
    def untokens(tok):
        return tok.view(2, 2, 2, 2, 16, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(2, 16, 2, 4, 4)
    perm = torch.randperm(8)
    xp = untokens(tokens(x)[:, perm])
    y = run(allnope, ([2, 2], [0, 1]))
    allnope.attach(model); allnope.set_context([2, 2], [0, 1])
    with torch.no_grad():
        yp = model(hidden_states=xp, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
    allnope.detach()
    torch.testing.assert_close(tokens(yp), tokens(y)[:, perm], rtol=1e-4, atol=1e-4)
    # sanity: with RoPE in place the same permutation is not an equivariance
    plain_off = LearnedCenter(layers=L, steps=S, heads=H, mode="off")
    y0 = run(plain_off, ([2, 2], [0, 1])); plain_off.attach(model); plain_off.set_context([2, 2], [0, 1])
    with torch.no_grad():
        y0p = model(hidden_states=xp, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
    plain_off.detach()
    assert (tokens(y0p) - tokens(y0)[:, perm]).abs().max() > 1e-3, "RoPE should break permutation equivariance"
    # legacy state dict (no mask_nope) loads as no-NoPE; disjointness enforced
    legacy = {k: v for k, v in plain.state_dict().items() if k != "mask_nope"}
    again = LearnedCenter(layers=L, steps=S, heads=H); again.load_state_dict(legacy); assert not again.uses_nope
    try:
        LearnedCenter(layers=L, steps=S, heads=H, selection={"mask_q": means, "mask_k": means, "mask_nope": means}); raise AssertionError("overlap accepted")
    except ValueError:
        pass
    print("test_nope OK")


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
        # fine-tune-then-drop screen: the saved student with its own masks, means (mean) or zeros (delete)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "screen_selections.py"), "--traj", str(traj), "--prompts", str(cfg), "--snapshot", str(tiny),
                               "--student", str(tmp / "s2" / "student"), "--modes", "mean", "delete", "--out", str(tmp / "screen-student"), "--val-prompts", "2"],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
        swap = json.loads((tmp / "screen-student" / "complete.json").read_text())
        assert set(swap["arms"]) == {"student/mean", "student/delete"} and all(0 <= v < 1e6 for a in swap["arms"].values() for v in a["mean"])
        assert json.loads((tmp / "screen-student" / "meta.json").read_text())["student_center_mode"] == "mean"
        from safetensors.torch import load_file

        resumed = tmp / "s2-resume"
        saved_ema = resumed / "ema-checkpoints" / "step-000002" / "learned_center.pt"
        flags = ["--stage", "2", "--out", str(resumed), "--steps", "4", "--init-student", str(tmp / "s1" / "student"),
                 "--ema", "0.9", "--ema-checkpoints", "--resume-state-with-ema", "--skip-final-resume-state",
                 "--resume-state-every", "4"] + common
        assert "TRAINING PAUSED 2 2" in run(flags + ["--stop-after", "2"], ROOT)
        early_ema = saved_ema.read_bytes()
        with (resumed / "validation.jsonl").open("a") as stream:
            stream.write('{"step": 4}\n{"step":')
        assert "START_OPTIMIZER_STEP 3" in run(flags + ["--resume"], ROOT)
        assert saved_ema.read_bytes() == early_ema
        assert json.loads((resumed / "checkpoint" / "latest.json").read_text())["step"] == 2
        for name in ("history", "validation"):
            def scores(path):
                return [{k: v for k, v in json.loads(line).items() if k not in {"elapsed_s", "optimizer_step_seconds"}}
                        for line in path.read_text().splitlines()]
            assert scores(resumed / f"{name}.jsonl") == scores(tmp / "s2" / f"{name}.jsonl")
        assert list((resumed / "interrupted-attempts").glob("*/validation.jsonl"))
        weights = "student/transformer/diffusion_pytorch_model.safetensors"
        actual, expected = load_file(resumed / weights), load_file(tmp / "s2" / weights)
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in actual)
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
            n = 42 if name == "split" else 22
            assert tuple(v["mask_q"].shape) == (2, STEPS, 2, 2, 64) and (v["mask_q"].sum(-1) == n).all() and (v["mask_k"].sum(-1) == n).all(), name
        assert torch.equal(sel["variants"]["shared"]["mask_q"], sel["variants"]["shared"]["mask_k"])
        pooled_m = sel["variants"]["pooled"]["mask_q"]
        assert torch.equal(pooled_m, sel["variants"]["pooled"]["mask_k"]) and torch.equal(pooled_m, pooled_m[:, 0:1, 0:1].expand_as(pooled_m))
        split = sel["variants"]["split"]
        assert (split["mask_q"].sum(-1) == 42).all() and (split["mask_nope"].sum(-1) == 22).all() and not (split["mask_q"] & split["mask_nope"]).any() and (split["mask_q"] | split["mask_nope"]).all()
        assert torch.equal(split["mask_q"], split["mask_k"]) and (split["mask_q"] & sel["variants"]["shared"]["mask_q"]).sum() == sel["variants"]["shared"]["mask_q"].sum(), "split means must contain the shared set"
        st = LearnedCenter.from_checkpoint(torch.load(sel_dir / "static-split" / "learned_center.pt")); assert st.uses_nope
        summed = (sel["damage_q"] + sel["damage_k"]).sum((1, 2))[1, 0]
        assert set(summed.topk(22, largest=False).indices.tolist()) == set(pooled_m[1, 0, 0, 0].nonzero().flatten().tolist())
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
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "screen_selections.py"), "--traj", str(tmp / "traj"), "--prompts", str(cfg), "--snapshot", str(tiny),
                               "--selection", str(sel_dir / "selection.pt"), "--statistics", str(sel_dir / "pair-statistics.pt"), "--out", str(tmp / "screen-narrow"),
                               "--val-prompts", "1", "--variants", "shared", "pooled", "split", "--modes", "delete", "narrow", "--all-steps"], cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
        narrow = json.loads((tmp / "screen-narrow" / "complete.json").read_text())
        assert narrow["arms"]["split/delete"]["mean"][2] != narrow["arms"]["shared/delete"]["mean"][2]
        narrow["arms"] = {k: v for k, v in narrow["arms"].items() if not k.startswith("split/")}
        meta = json.loads((tmp / "screen-narrow" / "meta.json").read_text())
        assert meta["exempt_layers"] == [0] and meta["exempt_steps"] == []
        assert set(narrow["arms"]) == {"shared/delete", "shared/narrow", "pooled/delete", "pooled/narrow"}
        assert narrow["arms"]["pooled/narrow"]["mean"][2] != narrow["arms"]["pooled/delete"]["mean"][2], "pre- and post-norm zeros must differ"
        report = json.loads((sel_dir / "selection-report.json").read_text())
        assert 0 <= report["pooled_vs_shared_intersection"]["mean_pairs"] <= 22
        print("test_statistics_selection_screen OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_replacement_exact()
    test_narrow_projection()
    test_narrow_selected()
    test_nope()
    test_masks_and_legacy_checkpoint()
    test_statistics_selection_screen()
    test_stage1_checkpoint_only_mu_differs()
    test_training_stages()
    print("ALL TESTS OK")
