#!/usr/bin/env python3
"""Publish only signed hand-off metadata to an outbound notification relay.

The Mac receives the signal through an outbound HTTPS stream. No desktop URL,
SSH access, workflow code, instructions or credentials are sent to the relay.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def handoff_to_codex(message):
    paragraphs = [p for p in re.split(r'\n[ \t]*\n', message.strip()) if p.strip()]
    return len(paragraphs) > 1 and any(re.fullmatch(r'Handoff-To:[ \t]*codex', line.strip(), re.I)
                                     for line in paragraphs[-1].splitlines())


def main():
    repository = os.environ['GITHUB_REPOSITORY']
    ref = os.environ['GITHUB_REF']
    if repository != 'mohan-zeyu/tri-video' or ref != 'refs/heads/claude/magical-goodall-dmq9i1':
        raise SystemExit('Ignored unrelated repository or branch')
    after = os.environ['AFTER']
    before = os.environ.get('BEFORE', '')
    if not re.fullmatch(r'[a-f0-9]{40}', after):
        raise ValueError('Invalid commit')
    if re.fullmatch(r'[a-f0-9]{40}', before) and set(before) != {'0'}:
        commits = git('rev-list', '--reverse', before + '..' + after).splitlines()
    else:
        commits = [after]
    signals = []
    for sha in commits:
        changed = git('diff-tree', '--root', '--no-commit-id', '--name-only', '-r', sha).splitlines()
        message = git('show', '-s', '--format=%B', sha)
        if '.handoff/to-codex.log' in changed and handoff_to_codex(message):
            signals.append((sha, 'handoff'))
        elif '.handoff/codex-bridge-test' in changed:
            signals.append((sha, 'test'))
    if not signals:
        print('No Codex hand-off signals in this push')
        return
    key = os.environ.get('CODEX_HANDOFF_SIGNING_KEY', '')
    relay = os.environ.get('CODEX_HANDOFF_RELAY_URL', '')
    if len(key) < 32 or not re.fullmatch(r'https://ntfy\.sh/[A-Za-z0-9_-]{32,100}', relay):
        raise ValueError('Hand-off relay secrets are not configured')
    for sha, kind in signals:
        payload = {'version': 1, 'repository': repository, 'ref': ref, 'sha': sha,
                   'kind': kind, 'time': int(time.time())}
        signature = hmac.new(key.encode(), canonical(payload), hashlib.sha256).hexdigest()
        wire = json.dumps({'payload': payload, 'signature': signature}, separators=(',', ':')).encode()
        request = urllib.request.Request(relay, data=wire, method='POST', headers={
            'Content-Type': 'text/plain', 'Title': 'Codex hand-off signal',
            'User-Agent': 'CodexHandoffSignal/1.0'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError('Relay rejected signal')
                    response.read()
                print('Delivered signed', kind, 'signal for', sha)
                break
            except Exception:
                if attempt == 3:
                    raise RuntimeError('Relay publish failed; inspect connection without logging secret URL') from None
                time.sleep(2 ** attempt)


if __name__ == '__main__':
    main()
