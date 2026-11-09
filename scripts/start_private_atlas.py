"""Start/reuse the URL-key-protected atlas on an explicitly configured private interface."""
import argparse
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor, ProxyHandler

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bind',required=True,help='Private or loopback interface address for this deployment');ap.add_argument('--port',type=int,default=17841)
    args=ap.parse_args();archive=ROOT/'out/full-sweep-wan-20261101'
    source=ROOT/'.atlas-private-key'
    if not source.exists():
        with os.fdopen(os.open(source,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(secrets.token_urlsafe(32)+'\n')
    if source.stat().st_mode&0o077:raise RuntimeError('Access-key file must have mode 0600')
    secret=source.read_text().strip()
    plain=build_opener(ProxyHandler({}))
    authorized=build_opener(ProxyHandler({}),HTTPCookieProcessor(CookieJar()))
    base=f'http://{args.bind}:{args.port}'
    def status(path='/index.html'):
        try:
            with plain.open(base+path,timeout=2) as response:return response.status
        except HTTPError as error:
            code=error.code;error.close();return code
        except (URLError,OSError):return None
    initial=status()
    if initial not in (None,403):raise RuntimeError(f'Port is occupied by a service without the expected access gate (HTTP {initial})')
    if initial is None:
        with (archive/'logs/private-atlas.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/serve_full_sweep.py'),'--root',str(archive),
                '--bind',args.bind,'--port',str(args.port),'--viewer-file','private-viewer.json','--secret-file',str(source)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(30):
            time.sleep(.2)
            if process.poll() is not None:raise RuntimeError('Private atlas exited; inspect logs/private-atlas.log')
            if status()==403:break
        else:raise RuntimeError('Private atlas did not become ready')
    shared=base+'/index.html?'+urlencode({'key':secret})
    with authorized.open(shared,timeout=5) as response:
        assert response.status==200 and 'key=' not in response.url
        assert b'Complete experiment atlas' in response.read()
    for path in ('/config.json','/p03/step-020/layer-029.pt'):
        assert status(path)==403,path
    with authorized.open(base+'/config.json',timeout=5) as response:assert json.load(response)['experiment']=='wan-full-20261101'
    target=ROOT/'.atlas-private-link.txt'
    with os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:f.write(shared+'\n')
    os.chmod(target,0o600)
    print(json.dumps({'status':'ready','address':base,'authentication':'secret URL, then one-day HttpOnly cookie; all archive routes protected',
        'shared_link_file':str(target),'secret_file_served':False,'unauthorized_page_and_tensor_status':403},indent=2))


if __name__=='__main__':main()
