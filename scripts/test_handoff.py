"""Tests for scripts/handoff.py on throwaway repositories: one bare remote and two clones ("claude", "codex").

1. A hand-off is pending only for its addressee, intermediate commits wake nobody, an ack clears it; `wait`
   returns or times out; `watch` runs its command once per new hand-off and retries a failed command.
2. `send` merges and pushes again when the remote moved (the .handoff logs merge by union); PAUSE holds hand-offs.
3. `announce` lists exactly the hand-offs of one push range; trailers are read from the last paragraph only.

Run:  python3 scripts/test_handoff.py
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
        write(claude, ".gitattributes", ".handoff/*.log merge=union\n")
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
        assert [c["sha"] for c in waiting] == [brief] and waiting[0]["files"] == [".handoff/to-codex.log", "BRIEF.md"]
        assert waiting[0]["trailers"]["handoff-brief"] == ["BRIEF.md"] and "Carry out A" in waiting[0]["body"]
        assert pending(claude, "claude") == [] and tool(claude, "pending", "--as", "claude").stdout == ""
        assert "PENDING hand-off for codex" in tool(codex, "pending", "--as", "codex").stdout

        # wait returns at once when something is pending, and gives up at --timeout when nothing is.
        assert "PENDING hand-off for codex" in tool(codex, "wait", "--as", "codex", "--interval", "1").stdout
        idle = tool(claude, "wait", "--as", "claude", "--interval", "1", "--timeout", "1", ok=False)
        assert idle.returncode == 1 and idle.stdout == ""

        # watch runs its command once per new hand-off, retries a failed command, and stays silent when idle.
        out_file = os.path.join(tmp, "woken.txt")
        record = f'printf "%s %s\\n" "$HANDOFF_AGENT" "$HANDOFF_SHA" >> "{out_file}"; echo "$HANDOFF_MESSAGE" >> "{out_file}"'
        failed = tool(codex, "watch", "--as", "codex", "--once", "--run", "exit 3")
        assert "exited 3" in failed.stderr and not os.path.exists(out_file)
        tool(codex, "watch", "--as", "codex", "--once", "--run", record)
        tool(codex, "watch", "--as", "codex", "--once", "--run", record)
        woken = Path(out_file).read_text().splitlines()
        assert woken[0] == f"codex {brief}" and len(woken) == 2, woken
        assert f"at commit {brief}" in woken[1] and "pending --as codex" in woken[1]
        tool(claude, "watch", "--as", "claude", "--once", "--run", record)
        assert len(Path(out_file).read_text().splitlines()) == 2
        # With nothing new it stops at --timeout; a live watcher (here: this test's own pid) blocks a second one.
        idle = tool(codex, "watch", "--as", "codex", "--interval", "1", "--timeout", "1", "--run", record, ok=False)
        assert idle.returncode == 1 and "watcher stopped" in idle.stdout
        pidfile = Path(codex) / ".git" / "handoff-watch-codex.pid"
        assert not pidfile.exists()
        pidfile.write_text(str(os.getpid()))
        assert "already running" in tool(codex, "watch", "--as", "codex", "--once", "--run", record).stdout
        pidfile.unlink()

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

        # --until-delivered: the loop ends as soon as one hand-off has been delivered.
        done = tool(claude, "watch", "--as", "claude", "--interval", "30", "--until-delivered", "--run", record)
        assert f"notified claude of {report}" in done.stdout
        assert Path(out_file).read_text().splitlines()[2] == f"claude {report}"

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

        # Two hand-offs to claude pushed from different checkouts: the log lines merge without conflict.
        run(codex, "git", "pull", "-q", "--ff-only")
        run(claude, "git", "pull", "-q", "--ff-only")
        tool(codex, "send", "--as", "codex", "--to", "claude", "--allow-empty", "-m", "Report one")
        out = tool(claude, "send", "--as", "other", "--to", "claude", "--allow-empty", "-m", "Report two").stdout
        assert "after merging the remote 1x" in out, out
        log = (Path(claude) / ".handoff" / "to-claude.log").read_text().splitlines()
        assert len(log) == 3 and {line.split("\t")[3] for line in log} == {"Report test", "Report one", "Report two"}
        assert len(pending(codex, "claude")) == 2
        tool(claude, "ack", "--as", "claude", *[c["sha"] for c in pending(claude, "claude")])

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
