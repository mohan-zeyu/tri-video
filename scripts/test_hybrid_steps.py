"""CPU test of the hybrid sampler in generate_learned_center.py (StepHandover) on a tiny random Wan.

One WanPipeline call with a 4-step UniPC schedule, CFG 5, a student (perturbed tiny Wan + LearnedCenter in mean
mode, layer 0 and step 1 exempt) and a frozen teacher (the unperturbed tiny Wan, no center):
1. teacher on every step reproduces the teacher-only pipeline exactly (torch.equal on the final latent);
2. teacher on no step reproduces the student-only pipeline without any hand-over hooks exactly;
3. teacher on steps 1-2 follows the planned denoiser per step and branch, makes 4 teacher forwards, counts the
   replacement calls of the student steps only, and equals a hand-written sampler loop that keeps one UniPC
   scheduler across the hand-over;
4. bad step lists are rejected.

Run from the repository root:  python scripts/test_hybrid_steps.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_learned_center import StepHandover
from learned_center import LearnedCenter
from test_learned_center import STEPS, TINY, build_tiny

GUIDANCE = 5.0
SHAPE = {"height": 32, "width": 32, "num_frames": 5}


def models():
    """bf16 teacher and perturbed student, loaded with torch_dtype like generate_learned_center.py (fp32 norms kept)."""
    from diffusers import WanTransformer3DModel
    teacher = build_tiny()
    student = build_tiny()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.05 * torch.randn_like(p))
    with tempfile.TemporaryDirectory() as tmp:
        out = []
        for name, model in (("teacher", teacher), ("student", student)):
            model.save_pretrained(Path(tmp) / name)
            out.append(WanTransformer3DModel.from_pretrained(Path(tmp) / name, torch_dtype=torch.bfloat16).eval())
    return tuple(out)


def center_for(student):
    center = LearnedCenter(layers=TINY["num_layers"], steps=STEPS, heads=TINY["num_attention_heads"],
                           head_dim=TINY["attention_head_dim"], mode="mean")
    torch.manual_seed(2)
    with torch.no_grad():
        center.mu_q.normal_()
        center.mu_k.normal_()
    return center.attach(student)


def scheduler():
    from diffusers import UniPCMultistepScheduler
    return UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True,
                                   flow_shift=3.0, solver_order=2)


def embeds():
    torch.manual_seed(3)
    return torch.randn(1, 8, TINY["text_dim"]).to(torch.bfloat16), torch.randn(1, 8, TINY["text_dim"]).to(torch.bfloat16)


def pipeline(transformer):
    from diffusers import WanPipeline
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=None, scheduler=scheduler(), transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def rollout(pipe, center, handover, positive, negative):
    if center is not None:
        center.set_rollout(positive, negative)
        center.calls = 0
    final = {}

    def end_step(p, i, t, kw):
        final["latents"] = kw["latents"].detach().clone()
        if handover is not None:
            handover.after_step(i)
        return kw
    if handover is not None:
        handover.begin()
    try:
        pipe(prompt_embeds=positive, negative_prompt_embeds=negative, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
             generator=torch.Generator().manual_seed(7), output_type="latent", callback_on_step_end=end_step, **SHAPE)
        record = handover.check() if handover is not None else {}
    finally:
        if handover is not None:
            handover.end()
    calls = center.calls if center is not None else 0
    if center is not None:
        center.clear_rollout()
    return final["latents"], calls, record


def manual(pipe, center, teacher, student, teacher_steps, positive, negative):
    """The pipeline's loop written out: one scheduler, denoiser chosen per step."""
    sched = scheduler()
    sched.set_timesteps(STEPS)
    sched.set_begin_index(0)
    latents = pipe.prepare_latents(1, TINY["in_channels"], SHAPE["height"], SHAPE["width"], SHAPE["num_frames"], torch.float32,
                                   torch.device("cpu"), torch.Generator().manual_seed(7))
    center.set_rollout(positive, negative)
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            model = teacher if i + 1 in teacher_steps else student
            x = latents.to(torch.bfloat16)
            ts = t.expand(1)
            cond = model(hidden_states=x, timestep=ts, encoder_hidden_states=positive, attention_kwargs=None, return_dict=False)[0]
            uncond = model(hidden_states=x, timestep=ts, encoder_hidden_states=negative, attention_kwargs=None, return_dict=False)[0]
            latents = sched.step(uncond + GUIDANCE * (cond - uncond), t, latents, return_dict=False)[0]
    center.clear_rollout()
    return latents


def test_step_handover():
    torch.set_grad_enabled(False)
    positive, negative = embeds()

    # references without any hand-over hooks
    teacher, student = models()
    teacher_only, _, _ = rollout(pipeline(teacher), None, None, positive, negative)
    center = center_for(student)
    student_only, student_calls, _ = rollout(pipeline(student), center, None, positive, negative)
    active_layers = int(center.layer_active.sum())
    assert student_calls == active_layers * 2 * 2 * int(center.step_active.sum()), student_calls

    pipe = pipeline(student)
    for steps in ([0], [STEPS + 1], [1, 1]):
        try:
            StepHandover(pipe, student, teacher, center, steps, STEPS)
        except ValueError:
            continue
        raise AssertionError(f"teacher steps {steps} accepted")

    handover = StepHandover(pipe, student, teacher, center, list(range(1, STEPS + 1)), STEPS)
    z, calls, rec = rollout(pipe, center, handover, positive, negative)
    assert torch.equal(z, teacher_only), "teacher on every step must reproduce the teacher-only pipeline"
    assert calls == 0 and rec["teacher_forwards"] == 2 * STEPS and rec["student_forwards"] == 0, (calls, rec)
    assert pipe.transformer is student
    handover.remove()

    handover = StepHandover(pipe, student, teacher, center, [], STEPS)
    z, calls, rec = rollout(pipe, center, handover, positive, negative)
    assert torch.equal(z, student_only), "teacher on no step must reproduce the student-only pipeline"
    assert calls == student_calls and rec["teacher_forwards"] == 0 and rec["denoiser_by_step"] == ["student"] * STEPS, (calls, rec)
    handover.remove()

    split = [1, 2]
    handover = StepHandover(pipe, student, teacher, center, split, STEPS)
    z, calls, rec = rollout(pipe, center, handover, positive, negative)
    student_active = int(center.step_active[handover.student_steps()].sum())
    assert calls == active_layers * 2 * 2 * student_active, (calls, student_active)
    assert rec == {"teacher_steps": split, "teacher_forwards": 4, "expected_teacher_forwards": 4, "student_forwards": 2 * (STEPS - 2),
                   "denoiser_by_step": ["teacher", "teacher", "student", "student"]}, rec
    reference = manual(pipe, center, teacher, student, split, positive, negative)
    assert torch.equal(z, reference), "hybrid pipeline must equal the hand-written one-scheduler loop"
    assert not torch.equal(z, teacher_only) and not torch.equal(z, student_only)
    assert torch.isfinite(z).all()
    handover.remove()
    # after removal the student pipeline is unchanged
    z, calls, _ = rollout(pipe, center, None, positive, negative)
    assert torch.equal(z, student_only) and calls == student_calls
    print("test_step_handover OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_step_handover()
    print("ALL TESTS OK")
