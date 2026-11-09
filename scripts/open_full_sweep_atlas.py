"""Restore/reuse the remote read-only atlas and its local SSH tunnel.

Run locally with --host and --repo set to the remote checkout.
No packages required. The URL is local to this computer, not public hosting.
"""
import argparse
import json
import shlex
import socket
import subprocess
import urllib.request


REMOTE = r'''
import json,subprocess,sys,time,urllib.request
from pathlib import Path
repo=Path(sys.argv[1]).expanduser().resolve()
root=repo/'out/full-sweep-wan-20261101'
viewer=root/'viewer.json'
def healthy():
    try:
        info=json.loads(viewer.read_text())
        with urllib.request.urlopen('http://127.0.0.1:%d/config.json'%info['port'],timeout=2) as r:
            assert json.load(r)['experiment']=='wan-full-20261101'
        return info
    except (OSError,ValueError,AssertionError,KeyError):return None
info=healthy()
if info is None:
    logs=root/'logs';logs.mkdir(exist_ok=True)
    with (logs/'atlas-server.log').open('ab') as out:
        process=subprocess.Popen([sys.executable,str(repo/'scripts/serve_full_sweep.py'),'--root',str(root)],cwd=repo,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        time.sleep(.2)
        info=healthy()
        if info:break
        if process.poll() is not None:raise RuntimeError('Atlas server exited; inspect logs/atlas-server.log')
if info is None:raise RuntimeError('Atlas did not become healthy')
print(json.dumps(info))
'''


def healthy(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/config.json',timeout=3) as response:
            return json.load(response).get('experiment')=='wan-full-20261101'
    except (OSError,ValueError):return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',required=True);ap.add_argument('--repo',required=True)
    ap.add_argument('--port',default=17841,type=int)
    args=ap.parse_args()
    if not 1<=args.port<=65535:ap.error('Port must be between 1 and 65535')
    url=f'http://127.0.0.1:{args.port}/index.html'
    if healthy(args.port):
        print(f'Atlas and SSH tunnel are already healthy.\n{url}');return
    # Never take over an unrelated service on the requested local port.
    with socket.socket() as sock:
        try:sock.bind(('127.0.0.1',args.port))
        except OSError:raise SystemExit(f'Local port {args.port} is occupied but not serving this atlas. Choose --port with a free port.')
    result=subprocess.run(['ssh','-o','ConnectTimeout=10',args.host,shlex.join(['python3','-c',REMOTE,args.repo])],capture_output=True,text=True,check=True)
    info=json.loads(result.stdout)
    subprocess.run(['ssh','-f','-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
        '-L',f"127.0.0.1:{args.port}:127.0.0.1:{info['port']}",args.host],check=True)
    if not healthy(args.port):raise SystemExit('The SSH tunnel started, but the atlas health check failed.')
    print(f'Atlas restored. Remote port {info["port"]}; local port {args.port}.\n{url}')


if __name__=='__main__':main()
