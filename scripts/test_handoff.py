"""Tests for scripts/handoff.py on throwaway repositories: one bare remote and two clones ("claude", "codex").

1. A hand-off is pending only for its addressee, intermediate commits wake nobody, and an ack clears it.
2. `send` merges and pushes again when the remote moved; a PAUSE file holds hand-offs back.
3. `announce` lists exactly the hand-offs of one push range; trailers are read from the last paragraph only.

Run from the repository root:  python3 scripts/test_handoff.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff

SCRIPT = str(Path(__file__).resolve().parent / "handoff.py")
ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
ENV.pop("HANDOFF_AGENT", None)
ENV.pop("HANDOFF_BRANCH", None)


def run(cwd, *cmd, ok=True):
    proc = subprocess.run(cmd, cwd=cwd, env=ENV, text=True, capture_output=True)
    if ok and proc.returncode:
        raise AssertionError(f"{cmd} failed in {cwd}:\n{proc.stdout}\n{proc.stderr}")
    return proc


def tool(cwd, *args, ok=True):
    return run(cwd, sys.executable, SCRIPT, *args, ok=ok)


def write(cwd, name, text):
    (Path(cwd) / name).write_text(text)
    run(cwd, "git", "add", name)


def head(cwd):
    return run(cwd, "git", "rev-parse", "HEAD").stdout.strip()


def pending(cwd, name):
    return json.loads(tool(cwd, "pending", "--as", name, "--json").stdout or "[]")


def main():
    assert handoff.parse_trailers("Fix: a thing") == {}
    assert handoff.parse_trailers("S\n\nHandoff-To: codex\n\nlast paragraph, no trailer here") == {}
    assert handoff.parse_trailers("S\n\nbody\n\nHandoff-To: codex\nCo-Authored-By: X <x@y>")["handoff-to"] == ["codex"]

    with tempfile.TemporaryDirectory() as tmp:
        remote, claude, codex = (os.path.join(tmp, d) for d in ("remote.git", "claude", "codex"))
        run(tmp, "git", "init", "-q", "--bare", "-b", "work", remote)
        run(tmp, "git", "clone", "-q", remote, claude)
        run(claude, "git", "checkout", "-q", "-b", "work")
        write(claude, "README", "start\n")
        run(claude, "git", "commit", "-q", "-m", "Start")
        start = head(claude)
        run(claude, "git", "push", "-q", "-u", "origin", "work")
        run(tmp, "git", "clone", "-q", remote, codex)

        # 1. claude sends a brief; only codex sees it; nothing staged is refused.
        assert tool(claude, "send", "--as", "claude", "--to", "codex", "-m", "x", ok=False).returncode != 0
        write(claude, "BRIEF.md", "do A then B\n")
        tool(claude, "send", "--as", "claude", "--to", "codex", "--brief", "BRIEF.md", "-m", "Brief: test",
             "-m", "Carry out A then B.", "--trailer", "Co-Authored-By: Someone <s@example.invalid>")
        brief = head(claude)
        waiting = pending(codex, "codex")
        assert [c["sha"] for c in waiting] == [brief] and waiting[0]["files"] == ["BRIEF.md"]
        assert waiting[0]["trailers"]["handoff-brief"] == ["BRIEF.md"] and "Carry out A" in waiting[0]["body"]
        assert pending(claude, "claude") == [] and tool(claude, "pending", "--as", "claude").stdout == ""
        assert "PENDING hand-off for codex" in tool(codex, "pending", "--as", "codex").stdout

        # Intermediate commits carry no trailer and change nothing.
        run(codex, "git", "pull", "-q", "--ff-only")
        write(codex, "validation.jsonl", "{}\n")
        run(codex, "git", "commit", "-q", "-m", "Record validation at update 250")
        run(codex, "git", "push", "-q")
        assert pending(claude, "claude") == [] and len(pending(codex, "codex")) == 1

        # 2. The remote moves before codex reports; send merges and pushes. The report acks the brief.
        run(claude, "git", "pull", "-q", "--ff-only")
        write(claude, "note.md", "meanwhile\n")
        run(claude, "git", "commit", "-q", "-m", "Unrelated note")
        run(claude, "git", "push", "-q")
        write(codex, "REPORT.md", "numbers\n")
        out = tool(codex, "send", "--as", "codex", "--to", "claude", "--ack", brief[:10], "--status", "done",
                   "-m", "Report test").stdout
        assert "after merging the remote 1x" in out, out
        assert pending(codex, "codex") == []
        waiting = pending(claude, "claude")
        assert len(waiting) == 1 and waiting[0]["subject"] == "Report test"
        report = waiting[0]["sha"]
        assert waiting[0]["trailers"]["handoff-ack"] == [brief]

        # announce: the push range holds one hand-off to claude; an unknown or zero "before" looks at the tip only.
        run(claude, "git", "pull", "-q", "--ff-only")
        text = tool(claude, "announce", "--to", "claude", "--before", start, "--after", head(claude)).stdout
        assert text.count("**Hand-off to claude**") == 1 and report in text and "Status: `done`" in text
        assert tool(claude, "announce", "--to", "codex", "--before", brief, "--after", head(claude)).stdout == ""
        assert tool(claude, "announce", "--to", "claude", "--before", "0" * 40, "--after", report).stdout.count("**") == 2
        assert tool(claude, "announce", "--to", "claude", "--before", "0" * 40, "--after", start).stdout == ""

        # PAUSE holds it back; --ignore-pause shows it; removing PAUSE releases it; ack clears it.
        write(claude, "PAUSE", "hold\n")
        run(claude, "git", "commit", "-q", "-m", "Pause")
        run(claude, "git", "push", "-q")
        held = tool(claude, "pending", "--as", "claude")
        assert held.stdout == "" and "held" in held.stderr
        assert "PENDING" in tool(claude, "pending", "--as", "claude", "--ignore-pause").stdout
        run(claude, "git", "rm", "-q", "PAUSE")
        run(claude, "git", "commit", "-q", "-m", "Resume")
        run(claude, "git", "push", "-q")
        assert len(pending(claude, "claude")) == 1
        tool(claude, "ack", "--as", "claude", report, "-m", "Examined.")
        assert pending(claude, "claude") == [] and pending(codex, "claude") == []

        # A conflicting remote change: the commit stays local, nothing is pushed, the merge is aborted.
        run(codex, "git", "pull", "-q", "--ff-only")
        write(codex, "note.md", "codex edit\n")
        run(codex, "git", "commit", "-q", "-m", "Edit note")
        run(codex, "git", "push", "-q")
        write(claude, "note.md", "claude edit\n")
        failed = tool(claude, "send", "--as", "claude", "--to", "codex", "-m", "Conflicting", ok=False)
        assert failed.returncode != 0 and "NOT pushed" in failed.stderr
        assert run(claude, "git", "status", "--porcelain").stdout == "" and pending(codex, "codex") == []

    print("handoff tests passed")


if __name__ == "__main__":
    main()
