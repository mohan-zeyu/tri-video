Attention! You should always mention where and how the computation results derive. Is it from pairs or the whole Q/K? Is it from one token or average of them? Is it from one denoise step or average? and so on....


## Hand-off protocol

Two agents share this branch: `claude` (research: plans, briefs, examines results) and `codex` (lab: runs the GPU
work). The user decides direction. A hand-off is a commit trailer; `scripts/handoff.py` reads and writes it.

- `Handoff-To: <agent>` on the commit that finishes a unit of work. Commits without it wake nobody.
- `Handoff-Ack: <sha>` on the commit that answers a hand-off. Until then it is pending.
- Always hand off with `scripts/handoff.py send`. It also appends a line to `.handoff/to-<agent>.log`, which is
  what triggers the wake-up; a trailer typed by hand is listed by `pending` but wakes nobody.
- A file `PAUSE` at the repository root holds all pending hand-offs. Only the user adds or removes it.

### codex

1. Check: `python3 scripts/handoff.py pending --as codex` in a checkout of this branch. No output means nothing
   to do; end quietly.
2. For each hand-off printed: pin a task checkout to that commit, read the brief named in `brief:`, follow the
   `message:`, and carry it out. The brief's ASK points are binding; stop and ask the user there.
3. Commit intermediate progress as usual, without trailers.
4. When the task is finished, interrupted, or blocked on a question: stage the report, `handoff.md` paragraph and
   evidence files together, then
   `python3 scripts/handoff.py send --as codex --to claude --ack <brief sha> --status done -m "<subject>" -m "<what ran>"`
   (`--status interrupted` or `question` when you need a decision; say what you need in the message).
   It commits what is staged, pushes, and merges first if the remote moved. In a checkout pinned to a commit
   (detached HEAD) add `--branch <branch>`; merge the branch tip first if the checkout predates this script.
5. Work that is not in a brief needs the user's approval; record who approved it and when in the commit message.

### claude

Follow `.claude/skills/handoff/SKILL.md`. In short: examine results, ack, notify the user, stop. Send a brief to
`codex` only for a task the user assigned or approved.

### Setup, once

- codex: schedule the check of step 1 every 10 minutes in the lab chat, in a checkout that tracks this branch.
- claude: the research session subscribes to the branch's draft pull request. The workflow
  `.github/workflows/handoff-notify.yml` comments there on each `Handoff-To: claude`, which wakes that session.
