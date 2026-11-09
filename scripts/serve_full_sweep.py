"""Read-only experiment atlas server, loopback by default.

Use an SSH local forward to view the complete atlas without copying the
large archive, or explicitly bind one private interface for sharing.
The chosen port is recorded in viewer.json unless --viewer-file overrides it.
"""
import argparse
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import ipaddress
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
import os
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

ROOT=Path(__file__).resolve().parents[1]


def make_handler(root, secret=None):
    """Protect every archive route, including images, JSON, HEAD and downloads."""
    root=root.resolve()
    class AtlasHandler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)

        def signature(self,expiry):
            return hmac.new(secret.encode(),f"atlas-session-v1|{root}|{expiry}".encode(),hashlib.sha256).hexdigest()

        def authorized_cookie(self):
            try:
                cookie=SimpleCookie();cookie.load(self.headers.get("Cookie",""))
                expiry,signature=cookie["tri_atlas_session"].value.split(".",1)
                return int(expiry)>=int(time.time()) and hmac.compare_digest(signature,self.signature(expiry))
            except (CookieError,KeyError,ValueError,TypeError):return False

        def send_head(self):
            if secret:
                parsed=urlsplit(self.path);pairs=parse_qsl(parsed.query,keep_blank_values=True)
                keys=[v for k,v in pairs if k=="key"]
                if keys:
                    if len(keys)!=1 or not hmac.compare_digest(keys[0].encode(),secret.encode()):
                        self.send_error(403,"Private atlas: valid shared link required");return None
                    expiry=str(int(time.time())+86400)
                    cookie=SimpleCookie();cookie["tri_atlas_session"]=expiry+"."+self.signature(expiry)
                    cookie["tri_atlas_session"]["path"]="/";cookie["tri_atlas_session"]["httponly"]=True
                    cookie["tri_atlas_session"]["samesite"]="Lax";cookie["tri_atlas_session"]["max-age"]=86400
                    target="/"+parsed.path.lstrip("/")
                    query=urlencode([(k,v) for k,v in pairs if k!="key"])
                    if query:target+="?"+query
                    self.send_response(303);self.send_header("Location",target)
                    self.send_header("Set-Cookie",cookie["tri_atlas_session"].OutputString())
                    self.send_header("Content-Length","0");self.end_headers();return None
                if not self.authorized_cookie():
                    self.send_error(403,"Private atlas: valid shared link required");return None
            path=Path(self.translate_path(self.path)).resolve()
            if path!=root and root not in path.parents:
                self.send_error(403,"Outside experiment archive");return None
            return super().send_head()

        def end_headers(self):
            self.send_header("Referrer-Policy","no-referrer")
            if secret:self.send_header("Cache-Control","private, no-store")
            super().end_headers()

        def log_request(self,code="-",size="-"):
            # Query parameters (including access keys) never enter access logs.
            self.log_message('"%s %s %s" %s %s',self.command,urlsplit(self.path).path,self.request_version,str(code),str(size))

        def log_message(self,format,*args):
            message=format%args
            if secret:message=message.replace(secret,"[redacted]")
            super().log_message("%s",message)
    return AtlasHandler


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="out/full-sweep-wan-20261101")
    ap.add_argument("--bind",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=0)
    ap.add_argument("--viewer-file",default="viewer.json")
    ap.add_argument("--secret-file",help="External 0600 text file with a random shared URL key; never served")
    args=ap.parse_args();root=ROOT/args.root
    try:address=ipaddress.IPv4Address(args.bind)
    except ipaddress.AddressValueError:ap.error("--bind must be a literal loopback or private IPv4 address")
    networks=("127.0.0.0/8","10.0.0.0/8","172.16.0.0/12","192.168.0.0/16","100.64.0.0/10")
    if not any(address in ipaddress.IPv4Network(n) for n in networks):ap.error("Only loopback or private IPv4 bindings are supported")
    if not 0<=args.port<=65535:ap.error("Invalid port")
    if Path(args.viewer_file).name!=args.viewer_file or not args.viewer_file.endswith("viewer.json"):
        ap.error("--viewer-file must be a filename ending in viewer.json")
    secret=None
    if args.secret_file:
        source=Path(args.secret_file).resolve()
        if source==root.resolve() or root.resolve() in source.parents:ap.error("Secret file must be outside the served archive")
        if source.stat().st_mode&0o077:ap.error("Secret file must have mode 0600")
        secret=source.read_text().strip()
        if len(secret)<32:ap.error("Secret must contain at least 32 characters")
    if not address.is_loopback and not secret:ap.error("Private-network sharing requires --secret-file")
    server=ThreadingHTTPServer((args.bind,args.port),make_handler(root,secret))
    access="SSH-forwarded loopback" if address.is_loopback else "private-network sharing; any reachable client can read the experiment archive"
    if secret:access="secret link required; one-day HttpOnly session cookie; full experiment archive"
    info={"pid":os.getpid(),"bind":args.bind,"port":server.server_address[1],"root":str(root),"access":"read-only GET/HEAD, "+access}
    (root/args.viewer_file).write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)
    server.serve_forever()
