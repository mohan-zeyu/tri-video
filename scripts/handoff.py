#!/usr/bin/env python3
"""Hand-offs between coding agents that share a git branch, carried by commit trailers.

A commit that finishes a unit of work names who is next:      Handoff-To: executor
                                                              Handoff-From: researcher
The commit that answers it (or an empty ack commit) carries:  Handoff-Ack: <sha of the hand-off commit>
A hand-off is pending for NAME while no commit on the branch acks it. Commits without trailers wake nobody.
A file named PAUSE at the repository root holds every pending hand-off back until it is removed.
Agent names are free-form lowercase words (claude, codex, researcher, gpu ...).

  python3 scripts/handoff.py pending --as codex            # fetch, list what waits for codex (prints nothing if none)
  python3 scripts/handoff.py wait --as codex               # block until something waits for codex, print it, exit
  python3 scripts/handoff.py watch --as codex --run CMD    # run CMD once per new hand-off; no agent runs while idle
  python3 scripts/handoff.py send --as claude --to codex --brief BRIEF.md -m "Subject" -m "Body"
  python3 scripts/handoff.py send --as codex --to claude --ack <sha> --status done -m "Report"
  python3 scripts/handoff.py ack --as claude <sha>         # empty commit, when the answer needs no other commit
  python3 scripts/handoff.py announce --to claude --before <sha> --after <sha>   # used by the GitHub workflow

`send` commits what is already staged (stage it yourself with git add) and pushes; if the remote moved it merges and
pushes again. It also appends a line to .handoff/to-<agent>.log. The GitHub workflow runs only on pushes that change
the woken agent's log, so a trailer written by hand is still listed by `pending` but wakes nobody. Use `send`.
Standard library only. Tests: python3 scripts/test_handoff.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

TRAILER = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PAUSE_FILE = "PAUSE"
LOG_DIR = ".handoff"  # send appends one line per hand-off to .handoff/to-<agent>.log; the workflow triggers on it


def git(*args, check=True, stdin=None):
    env = {**os.environ, "LC_ALL": "C"}  # push() matches git's English rejection text
    proc = subprocess.run(["git", *args], text=True, capture_output=True, input=stdin, env=env)
    if check and proc.returncode:
        sys.exit(f"handoff: git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc


def parse_trailers(message):
    """Key/value lines of the last paragraph; a one-paragraph message is a subject and has none."""
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", message.strip()) if p.strip()]
    found = {}
    if len(paragraphs) < 2:
        return found
    for line in paragraphs[-1].splitlines():
        match = TRAILER.match(line.strip())
        if match:
            found.setdefault(match.group(1).lower(), []).append(match.group(2).strip())
    return found


def read_commits(*rev_args):
    """Newest first. Each: sha, author, date, subject, body, trailers."""
    out = git("log", "--format=%H%x1f%an%x1f%aI%x1f%B%x1e", *rev_args).stdout
    commits = []
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        sha, author, date, message = record.split("\x1f", 3)
        trailers = parse_trailers(message)
        paragraphs = [p.strip() for p in re.split(r"\n[ \t]*\n", message.strip()) if p.strip()] or [""]
        body = paragraphs[1:-1] if trailers else paragraphs[1:]  # the trailer paragraph is not part of the body
        commits.append({"sha": sha, "author": author, "date": date, "subject": paragraphs[0].splitlines()[0],
                        "body": "\n\n".join(body), "trailers": trailers})
    return commits


def first(commit, key):
    values = commit["trailers"].get(key)
    return values[0] if values else ""


def unacked(commits, name):
    """Hand-offs addressed to name that no commit acks, oldest first."""
    acks = {value.lower() for c in commits for value in c["trailers"].get("handoff-ack", []) if len(value) >= 7}
    waiting = [c for c in commits
               if name in (v.lower() for v in c["trailers"].get("handoff-to", []))
               and not any(c["sha"].startswith(a) for a in acks)]
    return waiting[::-1]


def changed_files(sha):
    return [f for f in git("show", "--name-only", "--format=", sha).stdout.splitlines() if f]


def resolve_branch(explicit):
    branch = explicit or os.environ.get("HANDOFF_BRANCH") or git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch == "HEAD":
        sys.exit("handoff: detached HEAD; pass --branch or set HANDOFF_BRANCH")
    return branch


def agent_name(value, flag="--as"):
    value = (value or os.environ.get("HANDOFF_AGENT") or "").lower()
    if not NAME.match(value):
        sys.exit(f"handoff: {flag} needs an agent name such as claude or codex (or set HANDOFF_AGENT)")
    return value


def fetch(remote, branch):
    delay = 2
    for attempt in range(5):
        proc = git("fetch", "-q", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", check=False)
        if proc.returncode == 0:
            return
        if attempt < 4:
            time.sleep(delay)
            delay *= 2
    sys.exit(f"handoff: could not fetch {remote}/{branch}:\n{proc.stderr.strip()}")


def push(remote, branch):
    """Push HEAD to the branch; merge once per rejection, retry network errors with backoff."""
    delay, merges = 2, 0
    for attempt in range(8):
        proc = git("push", "-q", "-u", remote, f"HEAD:refs/heads/{branch}", check=False)
        if proc.returncode == 0:
            return merges
        if re.search(r"rejected|non-fast-forward|fetch first", proc.stderr) and merges < 3:
            fetch(remote, branch)
            merge = git("merge", "--no-edit", f"{remote}/{branch}", check=False)
            if merge.returncode:
                git("merge", "--abort", check=False)
                sys.exit(f"handoff: the remote moved and merging it conflicts. Your commit is kept locally and "
                         f"was NOT pushed. Merge {remote}/{branch} by hand, then git push.\n{merge.stdout.strip()}")
            merges += 1
            continue
        time.sleep(delay)
        delay = min(delay * 2, 16)
    sys.exit(f"handoff: push failed, commit kept locally:\n{proc.stderr.strip()}")


def full_shas(values):
    return [git("rev-parse", "--verify", f"{v}^{{commit}}").stdout.strip() for v in values or []]


def commit_and_push(args, subject, bodies, trailers, allow_empty):
    branch = resolve_branch(args.branch)
    for extra in args.trailer or []:
        if not TRAILER.match(extra):
            sys.exit(f"handoff: --trailer must look like 'Key: value', got {extra!r}")
        trailers.append(extra)
    message = "\n\n".join([subject, *[b.strip() for b in bodies if b.strip()], "\n".join(trailers)]) + "\n"
    git("commit", "-q", "-F", "-", *(["--allow-empty"] if allow_empty else []), stdin=message)
    sha = git("rev-parse", "HEAD").stdout.strip()
    if args.no_push:
        print(f"committed {sha} (not pushed)")
        return
    merges = push(args.remote, branch)
    print(f"pushed {sha} to {args.remote}/{branch}" + (f" (after merging the remote {merges}x)" if merges else ""))


def look(args, name):
    """Fetch and return (hand-offs waiting for name, ref, held by PAUSE)."""
    branch = resolve_branch(args.branch)
    if not getattr(args, "no_fetch", False):
        fetch(args.remote, branch)
    ref = f"{args.remote}/{branch}"
    waiting = unacked(read_commits(ref), name)
    paused = git("cat-file", "-e", f"{ref}:{PAUSE_FILE}", check=False).returncode == 0
    if paused and not args.ignore_pause:
        return [], ref, len(waiting)
    for c in waiting:
        c["files"] = changed_files(c["sha"])
    return waiting, ref, 0


def show(waiting, name, ref, as_json):
    if as_json:
        print(json.dumps(waiting, indent=1))
        return
    for c in waiting:
        print(f"PENDING hand-off for {name}: {c['sha']}  (on {ref})")
        print(f"  from:    {first(c, 'handoff-from') or c['author']}")
        print(f"  date:    {c['date']}")
        print(f"  subject: {c['subject']}")
        for key, label in (("handoff-brief", "brief"), ("handoff-status", "status")):
            if first(c, key):
                print(f"  {label + ':':8} {first(c, key)}")
        shown = c["files"][:15]
        more = len(c["files"]) - len(shown)
        print("  files:   " + ", ".join(shown) + (f" (+{more} more)" if more > 0 else ""))
        if c["body"]:
            print("  message:")
            for line in c["body"].splitlines():
                print("    " + line)
        print()


def cmd_pending(args):
    name = agent_name(args.as_name)
    waiting, ref, held = look(args, name)
    if held:
        print(f"handoff: {PAUSE_FILE} is present on {ref}; {held} hand-off(s) for {name} are held.", file=sys.stderr)
    if waiting or args.json:
        show(waiting, name, ref, args.json)


def cmd_wait(args):
    """Poll until a hand-off waits for the agent, print it and exit 0; exit 1 when --timeout passes first."""
    name = agent_name(args.as_name)
    deadline = time.time() + args.timeout if args.timeout else None
    while True:
        waiting, ref, _ = look(args, name)
        if waiting:
            show(waiting, name, ref, args.json)
            return
        if deadline and time.time() + args.interval > deadline:
            sys.exit(1)
        time.sleep(args.interval)


def cmd_watch(args):
    """Run a command once for each new hand-off to the agent. Nothing but git runs while nothing is pending."""
    name = agent_name(args.as_name)
    shared = git("rev-parse", "--git-common-dir").stdout.strip()  # shared by the worktrees of one clone
    state = os.path.join(shared, f"handoff-notified-{name}")
    lock = open(os.path.join(shared, f"handoff-watch-{name}.lock"), "w")
    try:  # the lock is released by the system when this process ends, however it ends
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass  # no fcntl on this platform: no guard against a second watcher
    except OSError:
        print(f"handoff: a watcher for {name} is already running in this clone; not starting another.")
        return
    deadline = time.time() + args.timeout if args.timeout else None
    while True:
        try:
            waiting, _, _ = look(args, name)
        except SystemExit as error:  # e.g. the network is down: keep watching unless this is a single pass
            if args.once:
                raise
            print(error, file=sys.stderr, flush=True)
            waiting = []
        seen = set(open(state).read().split()) if os.path.exists(state) else set()
        delivered = 0
        for c in waiting:
            if c["sha"] in seen:
                continue
            message = (f"A hand-off is waiting for {name} at commit {c['sha']}. Run "
                       f"`python3 scripts/handoff.py pending --as {name}` and carry out what it prints, "
                       f"following the Hand-off protocol in AGENTS.md.")
            env = {**os.environ, "HANDOFF_AGENT": name, "HANDOFF_SHA": c["sha"], "HANDOFF_MESSAGE": message}
            status = subprocess.run(args.run, shell=True, env=env).returncode
            if status == 0:
                with open(state, "a") as handle:
                    handle.write(c["sha"] + "\n")
                delivered += 1
                print(f"handoff: notified {name} of {c['sha']}", flush=True)
            else:
                print(f"handoff: the --run command exited {status} for {c['sha']}; it will be tried again",
                      file=sys.stderr, flush=True)
        if args.once or (delivered and args.until_delivered):
            return
        if deadline and time.time() + args.interval > deadline:
            print(f"handoff: no hand-off for {name} within {args.timeout} s; watcher stopped.", flush=True)
            sys.exit(1)
        time.sleep(args.interval)


def cmd_send(args):
    sender, to = agent_name(args.as_name), agent_name(args.to, "--to")
    if not args.message:
        sys.exit("handoff: send needs -m SUBJECT")
    staged = git("diff", "--cached", "--quiet", check=False).returncode != 0
    if not staged and not args.allow_empty:
        sys.exit("handoff: nothing is staged. git add the files of this hand-off first (or pass --allow-empty).")
    log = os.path.join(git("rev-parse", "--show-toplevel").stdout.strip(), LOG_DIR, f"to-{to}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(log, "a") as handle:
        handle.write("\t".join([stamp, sender, args.status or "-", args.message[0].replace("\t", " ")]) + "\n")
    git("add", log)
    trailers = [f"Handoff-To: {to}", f"Handoff-From: {sender}"]
    if args.brief:
        trailers.append(f"Handoff-Brief: {args.brief}")
    if args.status:
        trailers.append(f"Handoff-Status: {args.status}")
    trailers += [f"Handoff-Ack: {sha}" for sha in full_shas(args.ack)]
    commit_and_push(args, args.message[0], args.message[1:], trailers, allow_empty=False)


def cmd_ack(args):
    sender = agent_name(args.as_name)
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        sys.exit("handoff: files are staged. Commit them yourself with a 'Handoff-Ack: <sha>' trailer, or unstage.")
    shas = full_shas(args.sha)
    subject = "Ack hand-off " + ", ".join(s[:7] for s in shas)
    trailers = [f"Handoff-From: {sender}"] + [f"Handoff-Ack: {s}" for s in shas]
    commit_and_push(args, subject, args.message or [], trailers, allow_empty=True)


def code(text):
    return "`" + text.replace("`", "'").replace("\n", " ").strip() + "`"


def cmd_announce(args):
    """Markdown for a pull-request comment: the hand-offs to NAME among the commits of one push."""
    name = agent_name(args.to, "--to")
    known = set(args.before) != {"0"} and git("cat-file", "-e", f"{args.before}^{{commit}}", check=False).returncode == 0
    commits = read_commits(f"{args.before}..{args.after}") if known else read_commits("-1", args.after)
    blocks = []
    for c in commits[::-1]:
        if name not in (v.lower() for v in c["trailers"].get("handoff-to", [])):
            continue
        lines = [f"**Hand-off to {name}** at {code(c['sha'])}, from {code(first(c, 'handoff-from') or c['author'])}",
                 "", f"Subject: {code(c['subject'])}"]
        for key, label in (("handoff-status", "Status"), ("handoff-brief", "Brief")):
            if first(c, key):
                lines.append(f"{label}: {code(first(c, key))}")
        blocks.append("\n".join(lines))
    if blocks:
        print("\n\n---\n\n".join(blocks))
        print(f"\nTo process: `python3 scripts/handoff.py pending --as {name}`, then follow the hand-off skill.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--remote", default="origin")
    common.add_argument("--branch", help="default: the checked-out branch, or $HANDOFF_BRANCH")
    writing = argparse.ArgumentParser(add_help=False)
    writing.add_argument("-m", "--message", action="append", help="first -m is the subject, later ones are paragraphs")
    writing.add_argument("--trailer", action="append", help="extra 'Key: value' trailer, e.g. attribution lines")
    writing.add_argument("--no-push", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pending", parents=[common], help="list hand-offs waiting for an agent")
    p.add_argument("--as", dest="as_name", help="who is asking (default $HANDOFF_AGENT)")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--ignore-pause", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pending)

    p = sub.add_parser("wait", parents=[common], help="block until a hand-off waits for an agent")
    p.add_argument("--as", dest="as_name", help="who is waiting (default $HANDOFF_AGENT)")
    p.add_argument("--interval", type=int, default=120, help="seconds between fetches (default 120)")
    p.add_argument("--timeout", type=int, default=0, help="give up after this many seconds (default: never)")
    p.add_argument("--ignore-pause", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("watch", parents=[common], help="run a command once per new hand-off (no agent runs while idle)")
    p.add_argument("--as", dest="as_name", help="whose hand-offs to watch (default $HANDOFF_AGENT)")
    p.add_argument("--run", required=True, help="shell command to run for each new hand-off; it gets "
                   "$HANDOFF_MESSAGE (a fixed instruction naming the commit), $HANDOFF_SHA and $HANDOFF_AGENT")
    p.add_argument("--interval", type=int, default=300, help="seconds between fetches (default 300)")
    p.add_argument("--once", action="store_true", help="one pass and exit, for cron or launchd")
    p.add_argument("--until-delivered", action="store_true", help="exit once a hand-off has been delivered")
    p.add_argument("--timeout", type=int, default=0, help="stop after this many seconds without one (default: never)")
    p.add_argument("--ignore-pause", action="store_true")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("send", parents=[common, writing], help="commit the staged files as a hand-off and push")
    p.add_argument("--as", dest="as_name")
    p.add_argument("--to", required=True)
    p.add_argument("--ack", action="append", help="sha of the hand-off this one answers")
    p.add_argument("--brief", help="path of the brief the receiver should read")
    p.add_argument("--status", help="e.g. done, interrupted, question")
    p.add_argument("--allow-empty", action="store_true")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("ack", parents=[common, writing], help="push an empty commit that acks hand-offs")
    p.add_argument("--as", dest="as_name")
    p.add_argument("sha", nargs="+")
    p.set_defaults(func=cmd_ack)

    p = sub.add_parser("announce", help="markdown for the hand-offs in one push (GitHub workflow)")
    p.add_argument("--to", required=True)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.set_defaults(func=cmd_announce)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
