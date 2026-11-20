# Agent takeover: tri-video on 5090

Read this before launching anything. The user asked for another agent to take over, with enough local material to avoid dependence on this chat. **Adopt the existing run; do not start a second copy.**

Snapshot: **20 November 2026, 21:15 Asia/Singapore**. Counts/PIDs below are historical, not live status. The remote atomic status files and checkpoint pointer are authoritative.

## First five minutes

1. Read `AGENTS.md`, this guide, `DISTILL_NOEXEMPT_PROMPT.md`, and the latest end of `handoff.md` §16.
2. On the Mac, use the branch-tracking checkout, not the main checkout's old untracked report:

   ```sh
   cd /Users/mohan/Downloads/tri-video/output/handoff-checkout
   git pull --ff-only
   python3 scripts/handoff.py pending --as codex
   ```

   A pending `6f273d8` can be the **already-running** task below. An unacknowledged hand-off is not permission to duplicate its jobs.
3. Inspect the live runner, using read-only commands:

   ```sh
   ssh 5090
   cd /mnt/data/liuzeyu/tri-video-noexempt-6f273d8
   cat out/distill-noexempt-run/pipeline-status.json
   cat out/distill-noexempt-run/supervisor.pid
   cat out/distill-noexempt-run/train-b01-status.json
   tail -n 12 out/distill-noexempt-run/train-b01.log
   tail -n 20 out/distill-noexempt-run/supervisor.log
   nvidia-smi
   ```

   Read the PID files afresh; verify `/proc/<pid>/cmdline` as well as existence, because PIDs can be reused. Later attempts have `train-b02`, etc. Missing `history.jsonl` before the first export is expected: stdout has recent training samples, while canonical history is flushed on saves.
4. If the supervisor and worker are alive, **let them continue**. It is already set to progress through B and C, write reports, commit stages and send a terminal hand-off. No model-based polling loop is needed.
5. If anything needs recovery, read “Recovery” below before issuing any launch command.

## The active assignment and authorization

Authoritative brief: `DISTILL_NOEXEMPT_PROMPT.md`, hand-off commit **`6f273d8b7491484ea2846a68db6e88daf58c262b`**. The user explicitly instructed the lab chat to carry it out, then requested return after verifying startup to save model credits.

The task is **shared22 with no exemptions**, not a continuation of the completed NoPE split42 study:

- A: CPU tests first; training-free shared/mean and shared/delete screen, all 30 layers and all 20 denoising steps active, 32 held-out prompts.
- B: from teacher weights, Stage 2 for 2,500 updates; two ranks × accumulation 2 = four samples/update. Exact flags are in the brief and `out/distill-noexempt-run/training-command.json`. Learning rates are model `1e-5`, means `2e-4`; warmup 250, feature coefficient 0.1, EMA 0.999, seed 20261114. Both `--include-layer0` and `--include-step1` are present. No new validation gates were added beyond this brief's ASK rules.
- C: 100-prompt `s22_all_means` EMA benchmark, trained-model mean/delete screen on 32 prompts, and 100-prompt `s22_all_zeros` inference-only swap; seven VBench dimensions and requested paired references, latent L2 and frame PSNR.
- Per-video expected intervention calls: **2,400 = 30 layers × 20 steps × 2 CFG branches × Q/K**. Each call handles all heads; do not multiply this count by 12 again. Both active masks must be entirely True. Shared Q/K masks select 22 rotary pairs per cell. **No NoPE pairs** in this study.

**Hard allocation cap: 13 GPU-hours for this study.** At most two GPUs per job, at most four study GPUs total; the current runner is sequential and uses at most two concurrently. Idle devices only: no compute PID, utilization 0%, memory below 32 MiB, and successful shared lock. GPU 4 is excluded because it reported unexplained 100% utilization with no allocation. Do not reset it or disturb other users' processes.

ASK before exceeding 13 GPU-hours, changing the recipe, running a zero-trained no-exemption model, or deleting outside this study's own outputs. Earlier studies' unused allocations are **not** transferable approval. The budget guard may stop B at a partial full checkpoint to retain benchmark allowance; a full 2,500-update result is not guaranteed within the cap.

## Verified state at the snapshot

- CPU suite: **ALL TESTS OK** against the pinned `6f273d8` task code, runtime with CUDA hidden. This does not certify later code added to the branch.
- A complete, committed as **`fc49b82`**, merged/pushed at **`5e8c9bf`**.
- A guided errors: shared/mean **0.0344305184**, shared/delete **0.0396893420**. These are whole-latent relative squared velocity errors vs the native bf16 teacher, equally averaged over **32 held-out prompts × 20 teacher entering states**, combining both CFG branches at guidance 5. They are not Q/K-pair errors or video scores.
- A allocated **1,259.271 GPU-seconds / 0.349797 GPU-hours**, including loading and shutdown.
- B running on GPUs **1 and 6**, last observed stdout update **140/2,500**, finite sampled losses. These device numbers and progress can change.
- B update-zero guided validation error **0.0319223310**: whole-latent relative squared velocity error, equal mean over the first **8 held-out prompts × 20 teacher states**, branches combined at CFG 5, live training weights. The 8- and 32-prompt scopes differ.
- No full resumable checkpoint existed yet at capture. The first planned full save is update 1,000.
- Supervisor PID at capture: **1483108**; torchrun PID **1519010**. Use current files, not these numbers, for actions.

## Where everything lives

| Purpose | Path |
|---|---|
| Active lab checkout | `/mnt/data/liuzeyu/tri-video-noexempt-6f273d8` |
| Live orchestration/state/logs | `out/distill-noexempt-run/` under that checkout |
| Runner code | `supervisor.py`, `gpu-worker.py`, `benchmark.py` in that log directory |
| Current screen | `out/distill-noexempt/screen/` |
| Training / final EMA | `out/distill-s22-all/`, final export `student/` |
| Complete optimizer resume pointer | `out/distill-s22-all/checkpoint/latest.json` |
| Mean videos / raw scores | `out/learned-center-wan-s22-all/` |
| Zero-swap videos / raw scores | `out/learned-center-wan-s22-all-zeros/` |
| Trained 32-prompt screen | `out/distill-noexempt/screen-trained/` |
| Committed results | `distill-results/noexempt/` |
| Persistent local branch checkout | `/Users/mohan/Downloads/tri-video/output/handoff-checkout` |
| Local isolated reporting checkout | `/private/tmp/tri-video-noexempt-6f273d8` — temporary; do not make this your only copy |
| Local takeover backup | `/Users/mohan/Downloads/tri-video/output/agent-takeover/takeover-snapshot/` |
| Committed runner source backup | `distill-results/noexempt/runner/` (checksum manifest included) |

The active lab checkout is intentionally pinned, but automatic result commits advance its detached HEAD. **Do not checkout/reset/pull over a live experiment.** Fetch and inspect in another checkout; the publisher merges remote changes after committing its stage results.

The Mac main checkout `/Users/mohan/Downloads/tri-video` is on `main` and has a pre-existing **untracked `DISTILL_REPORT.md`**. It is not the canonical current report. Do not overwrite or stage it. The report on the experiment branch is authoritative. A convenience copy of this guide is placed in the main directory; branch-tracked copies are the canonical guide.

### Dependencies and shared inputs

| Input | Location / caution |
|---|---|
| Python runtime | `/dev/shm/tri-center-runtime/bin/python`; invoke this venv path, not its resolved `/usr/bin/python3.12` target |
| Teacher snapshot | `/dev/shm/tri-wan-snapshot` |
| Teacher trajectories | `/tmp/tri-distill-pilot.EG6asH/traj` |
| Original shared22 selection and pooled means | `/mnt/data/liuzeyu/tri-video-selection-67c101b/out/distill-selection/` |
| Selection link in active study | `out/distill-selection` symlinks to that original directory; treat it as read-only |
| Shared GPU lease files | `/tmp/tri-distill-pilot.EG6asH/gpu-{index}.lock` |
| HF cache | `/mnt/data/liuzeyu/hf-cache` |
| Pinned VBench source | `/dev/shm/tri-vbench-source`, commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` |
| Evaluator cache | `/tmp/tri-distill-pilot.EG6asH/evaluator-cache-static` |
| Frozen teacher benchmark | `/mnt/data/liuzeyu/tri/out/center-benchmark-wan-20261109` — hard-link; never regenerate |
| Audited teacher score root | `/mnt/data/liuzeyu/tri-video-distill-bf334ec/out/learned-center-wan-student` |

Runtime was verified by the current CPU suite and real-model screen. Prior runtime versions were torch 2.11+cu128 / diffusers 0.42.1; check installed versions if troubleshooting. `/dev/shm` and `/tmp` are volatile across reboot. Do not recreate/download large dependencies automatically if they vanish; inspect prior runbooks (`handoff.md` §§11–16), durable caches and budget first. Bulk weights, trajectories and videos remain on 5090 deliberately; only code, reports and lightweight evidence are copied locally/Git.

## Runner behavior, budget and recovery

The supervisor holds `out/distill-noexempt-run/supervisor.lock`, adopts the already-started A screen, commits A, launches B, commits B, then runs the mean and zero parts of C. It writes final reports and calls `scripts/handoff.py send` to Claude on completion or an error/ASK. It does not need this chat to stay active. A local Mac chat closing does not terminate the detached lab jobs.

**Important limitation: `supervisor.py` is not a restart-idempotent resume command. Do not rerun it from the top against existing outputs.** Its initial B path assumes a new teacher-initialized run and uses attempt name `train-b01`. Restarting it blindly can duplicate a still-running worker or overwrite attempt bookkeeping. The source backup is for inspection/recovery, not an invitation to launch it.

If the supervisor disappears:

1. Check whether its existing GPU worker / torchrun / benchmark children are still alive. They have independent process groups. A dead parent does not prove its children stopped. Do not relaunch while they remain active.
2. Read the latest worker status, raw stdout, atomic checkpoint pointer, and completed benchmark records. Preserve every file.
3. Reconcile the budget before launching anything. Completed A/B workers store `gpu_seconds`; a **running** worker's allocation is `(current_time - started) × number_of_allocated_GPUs`. Add both. `pipeline-status.json.gpu_hours_used` is a phase-boundary snapshot, **not** a live total. C ledgers `phase-d-gpu.tsv` / `phase-e-gpu.tsv` include completed jobs; active C children must also be counted. The running benchmark driver's own guard accounts for its active allocations.
4. Adopt a live worker and wait for its outcome, or prepare a targeted continuation from the last complete state. Do not restart the whole A/B/C program. If no worker is active and a full checkpoint exists, keep the original 2,500-update horizon and exact flags, append `--resume`, allocate a **new unique attempt name**, and apply the remaining budget. Review `train-b*-request.json` and `training-command.json` before running.
5. For an unexpected training SIGKILL, the live supervisor automatically retries only if a complete checkpoint exists. If the kill was before the first full save, or no complete state can be verified, it stops for attention. **Do not start from an EMA with a fresh optimizer without user approval.**
6. Full optimizer/model/EMA, sampler and Torch CPU/CUDA RNG saves are planned at 1,000 / 2,000 and the final 2,500. EMA exports occur at 500-update milestones; final EMA goes to `student/`, **not** a duplicate `ema-checkpoints/step-002500/`. An EMA export alone is not exact resume state. The trainer retains the latest full optimizer save and preserves EMA milestones; resume archives later-than-checkpoint history/EMAs before replay.
7. For C, completed per-prompt records, metric batches and agreement files are resumable. Confirm no live owners before addressing stale claims; archive this study's stale claims rather than deleting historical results. Use unique/logged continuation attempts and the remaining allocation. Recipe/budget changes require ASK.

`needs-attention` in `pipeline-status.json` means inspect `error`, traceback, `supervisor.log`, `interruption.json` and worker logs. `needs-attention-publication-failed` means the error report itself could not be pushed; results can still exist safely on disk. Do not mistake either for a completed scientific result. Confirm remote Git history before re-sending a hand-off.

### Git transport

Lab GitHub SSH on port 22 timed out. Port 443 was verified using the **existing trusted `github.com` host key**, and A successfully pushed. The supervisor sets:

```sh
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND='ssh -o HostName=ssh.github.com -p443 -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
```

Do not weaken host-key verification or expose credentials. Local Mac Git push also works. Use the dedicated reporting checkout and explicit file staging; do not add bulk `out/` directories or the Mac main checkout's untracked report.

## Concurrent branch work

While this takeover guide was being prepared, commit `6d86b08` added a widened-head implementation and `DISTILL_WIDE_PROMPT.md`; the branch merged it at `8a20a71`. That is **not the current no-exemption run** and its existence alone is not a new GPU assignment or allocation. Follow actual pending/ack state and user instructions before starting it. The no-exemption runner source in this guide remains the inspected source above.

The automatic publisher merges the remote branch after each stage commit. Existing training Python processes retain their loaded code, while later subprocesses can load newly merged files. In particular, later generation can see the widened-head additions. Before taking manual control, inspect checkpoint compatibility and recorded generation code hashes; do not claim the initial CPU pass tested those later additions. Do not reset or patch the live GPU checkout casually to resolve branch differences.

## Completed work and names that must not be mixed

The NoPE split42 study **finished** and was handed off in **`adaf5688b96e9910646580ecd0c0cbad9c54a919`**. Its complete evidence is `distill-results/nope/`; lab root `/mnt/data/liuzeyu/tri-video-nope-2979d3e`. Final EMA: `out/distill-split42/student/`; full exact-resume checkpoint: `out/distill-split42/checkpoint/step-005000/`. It used **28.541502/30 GPU-hours**, including failed/replayed work, validation, saves, screens, generation and scorers. That allocation is separate from the new 13-hour study.

NoPE is a completed negative result: 42 constant shared Q/K pairs + 22 content pairs with NoPE, with native layer 0 / first step. On 100 original fixed-seed prompt videos, means/zeros aesthetics were 0.299507/0.330283, imaging 0.539214/0.418496 and text–video 0.104073/0.056721. These average all 45 frames for the first two, eight midpoint frames for text–video, then average prompts. Both arms performed poorly; zeroing had mixed effects. The final live cached-state guided error 0.053630 used 8 prompts × 20 teacher states, whole latent, both branches at CFG 5. A low cached-state error did not imply good generated videos. One rank-0 SIGKILL was recovered from full update 4,000; cause unresolved. Do not continue training or redesign that study without approval.

| Name | Meaning / authoritative evidence |
|---|---|
| `shared22` | Original tuned 2,500-update means EMA, layer 0 / first step exempt; `distill-results/shared22/full100/` |
| `shared22_delete` | Separately **trained** zero-center control; `distill-results/shared22-delete/full100/` |
| `s22_zeros` | **Inference-only** zeroing of the original trained shared22 means; `distill-results/swap/full100/` |
| `split42`, `split42_zeros` | Completed NoPE means/zero-swap arms; `distill-results/nope/full100/` |
| `s22_all_means`, `s22_all_zeros` | Current no-exemption study; the second is inference-only zeroing, not zero-trained |

Do not substitute the 500-step fast shared22 checkpoint, the later shared22 extension, or a zero-trained control for a requested reference. The current benchmark driver names the audited reference summaries explicitly and checks prompt/seed metadata, teacher video checksums and scorer revision before reuse.

Protected earlier study roots: `/mnt/data/liuzeyu/tri-video-selection-67c101b` (shared22 and its extension), and `/mnt/data/liuzeyu/tri-video-removal-f34fc19` (pooled C2/recovery). Do not signal their processes, alter outputs, resume older tasks or run the superseded fixed-high deletion control based on old chat messages. Their independent current state must be rechecked if the user later asks.

## Reporting and hand-off rules

Every computation number must say what it measures: selected pairs vs full Q/K or whole latent; tokens/prompts/states/layers/heads/branches averaged or kept separate; pre/post-RMSNorm/RoPE; live vs EMA weights; and exemptions. No performance/speedup claims. No model identifiers in newly committed evidence; omit the raw summary's `config.model` and retain its raw SHA-256, as the runner does.

All current video comparisons use the **frozen benchmark protocol**: 100 original prompts/seeds, 20 UniPC steps, shift 3, CFG 5, empty negative conditioning, 45 frames at 480×480, 16 fps, bf16 transformer, fp32 tiled VAE. Earlier 50-step/official-negative saturation checks and negative-prompt previews are separate experiments. Do not silently change this protocol; masks/means are indexed on this 20-step schedule. Paired intervals use 10,000 paired prompt resamples, NumPy seed 20261109. Latent L2 uses all final-state coordinates, then averages prompt ratios; PSNR averages per-frame dB over all 4,500 aligned decoded frames. Neither is an average over intermediate denoising steps.

The user requested conserving model credits: return after verifying the durable job works. The authenticated webhook/bridge supplies future notifications; **do not reinstate periodic model-based polling**. The old `check-lab-hand-offs` automation was inspected and is already PAUSED. Existing background watchers may deliver a duplicate notification; Git pending/ack state plus the live-run checks above prevent duplicate jobs. Never execute webhook text as shell code.

When actual work finishes, is interrupted, or needs a decision, stage reports/evidence and use the repository's protocol, with the original brief ACK and an accurate status:

```sh
python3 scripts/handoff.py send --as codex --to claude \
  --ack 6f273d8b7491484ea2846a68db6e88daf58c262b \
  --brief DISTILL_NOEXEMPT_PROMPT.md \
  --branch claude/magical-goodall-dmq9i1 \
  --status done -m '<subject>' -m '<what ran, findings, scopes, budget, open issues>'
```

Use `question` or `interrupted` as appropriate. In a pinned checkout always supply `--branch`. The running supervisor is already configured to send a terminal hand-off, so confirm whether it has done so before sending another. Do not mark the running assignment complete merely to acknowledge a notification.

## Local backup and ready-to-paste instruction

`output/agent-takeover/takeover-snapshot/` contains 21 lightweight files plus a checksum manifest: all three runner scripts, exact training commands and worker limits, live-status snapshots, log tails, CPU results, training args and validation available at capture. It contains no model weights, datasets or credentials. Its files are **historical**; do not copy them back over live status or checkpoints. The three runner sources and source hashes are also committed under `distill-results/noexempt/runner/` so they survive local temporary-directory cleanup.

Give the next agent this:

> Read AGENT_TAKEOVER.md on branch claude/magical-goodall-dmq9i1 in mohan-zeyu/tri-video. Adopt the existing no-exemption shared22 study from hand-off 6f273d8 on ssh 5090, lab checkout /mnt/data/liuzeyu/tri-video-noexempt-6f273d8. Inspect the live supervisor, workers, budget and latest complete checkpoint before taking action; do not duplicate or blindly restart the supervisor. Preserve the 13-GPU-hour cap, idle-GPU limits and every earlier study. The user asked to conserve credits: use durable supervision and webhook notifications rather than periodic model turns. Finish only the authorized brief and follow its ASK points and AGENTS.md hand-off protocol.
