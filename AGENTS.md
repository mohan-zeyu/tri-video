Attention! You should always mention where and how the computation results derive. Is it from pairs or the whole Q/K? Is it from one token or average of them? Is it from one denoise step or average? and so on....


## Hand-off protocol

Two agents share this branch: `claude` (research: plans, briefs, examines results) and `claude-lab` (lab: Claude
Code on the lab host, runs the GPU and CPU work there). The user decides direction. A hand-off is a commit trailer;
`scripts/handoff.py` reads and writes it.

The lab agent was called `codex` on this branch until 22 November 2026. Briefs up to `ffc819f` are addressed to
`codex`; the same lab agent answers them. Every later brief is addressed to `claude-lab`.

- `Handoff-To: <agent>` on the commit that finishes a unit of work. Commits without it wake nobody.
- `Handoff-Ack: <sha>` on the commit that answers a hand-off. Until then it is pending.
- Always hand off with `scripts/handoff.py send`. It also appends a line to `.handoff/to-<agent>.log`, which is
  what triggers the wake-up; a trailer typed by hand is listed by `pending` but wakes nobody.
- A file `PAUSE` at the repository root holds all pending hand-offs. Only the user adds or removes it.

### claude-lab

1. When a message says a hand-off is waiting, when the session has just been started or reopened, or when the user
   asks: `python3 scripts/handoff.py pending --as claude-lab` in a checkout of this branch, and once also
   `--as codex` for briefs sent under the old name. No output means nothing to do; end quietly.
2. For each hand-off printed: pin a task checkout to that commit, read the brief named in `brief:`, follow the
   `message:`, and carry it out. The brief's ASK points are binding; stop and ask the user there.
3. Commit intermediate progress as usual, without trailers.
4. When the task is finished, interrupted, or blocked on a question: stage the report, `handoff.md` paragraph and
   evidence files together, then
   `python3 scripts/handoff.py send --as claude-lab --to claude --ack <brief sha> --status done -m "<subject>" -m "<what ran>"`
   (`--status interrupted` or `question` when you need a decision; say what you need in the message).
   It commits what is staged, pushes, and merges first if the remote moved. In a checkout pinned to a commit
   (detached HEAD) add `--branch <branch>`.
5. Right after every `send --to claude`, start
   `python3 scripts/handoff.py wait --as claude-lab --timeout 6900`
   as a background command and end your turn. It fetches every 2 minutes and exits when the next hand-off arrives,
   which brings you back to step 1. The timeout keeps it under Claude Code's two-hour limit for background
   commands; when it ends without a hand-off, end quietly, and step 1 applies the next time the user starts you.
   Do not poll in the chat: every check costs a model turn. In a pinned checkout add `--branch <branch>`.
6. Work that is not in a brief needs the user's approval; record who approved it and when in the commit message.

### claude

Follow `.claude/skills/handoff/SKILL.md`. In short: examine results, ack, notify the user, stop. Send a brief to
`claude-lab` only for a task the user assigned or approved.

### Setup, once

- claude-lab: nothing to schedule. The lab session starts its own background `wait` (step 5); it needs network
  access to fetch this branch. `.github/workflows/handoff-codex.yml` belongs to the Codex lab agent of
  `claude/magical-goodall-dmq9i1` and does not run for this branch.
- claude: the research session subscribes to the branch's draft pull request. The workflow
  `.github/workflows/handoff-notify.yml` comments there on each `Handoff-To: claude`, which wakes that session.
