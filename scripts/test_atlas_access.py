"""Integration checks for the private atlas URL-key access boundary."""
import contextlib
import hashlib
import hmac
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
import io
from pathlib import Path
import secrets
import tempfile
import threading
import time
import unittest

from serve_full_sweep import make_handler


class AtlasAccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'archive';self.root.mkdir()
        (self.root/'index.html').write_text('atlas')
        (self.root/'layer.json').write_text('{"tokens":10800}')
        (self.root/'raw.pt').write_bytes(b'raw tensor')
        (self.base/'private-secret').write_text('must not be served')
        (self.root/'outside.txt').symlink_to(self.base/'private-secret')
        self.secret=secrets.token_urlsafe(32)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.root,self.secret))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self,path,method='GET',cookie=None):
        conn=HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        try:
            conn.request(method,path,headers={'Cookie':cookie} if cookie else {})
            result=conn.getresponse();return result.status,dict(result.getheaders()),result.read()
        finally:conn.close()

    def login(self,path='/index.html'):
        join='&' if '?' in path else '?'
        status,headers,_=self.request(path+join+'key='+self.secret)
        self.assertEqual(status,303)
        cookie=SimpleCookie();cookie.load(headers['Set-Cookie'])
        return headers,'tri_atlas_session='+cookie['tri_atlas_session'].value

    def test_all_resources_and_methods_require_access(self):
        for path in ('/','/index.html','/layer.json','/raw.pt','/missing'):
            for method in ('GET','HEAD'):
                self.assertEqual(self.request(path,method)[0],403,(path,method))
        self.assertEqual(self.request('/?key=wrong')[0],403)
        self.assertEqual(self.request('/?key='+self.secret+'&key=other')[0],403)

    def test_authorized_browser_can_load_page_and_downloads(self):
        headers,cookie=self.login('/index.html?prompt=p03&head=11')
        self.assertEqual(headers['Location'],'/index.html?prompt=p03&head=11')
        self.assertNotIn(self.secret,headers['Set-Cookie'])
        self.assertIn('HttpOnly',headers['Set-Cookie']);self.assertIn('SameSite=Lax',headers['Set-Cookie'])
        for path in ('/index.html','/layer.json','/raw.pt'):
            self.assertEqual(self.request(path,cookie=cookie)[0],200)
            status,h,body=self.request(path,'HEAD',cookie)
            self.assertEqual(status,200);self.assertEqual(body,b'')
            self.assertEqual(h['Referrer-Policy'],'no-referrer');self.assertEqual(h['Cache-Control'],'private, no-store')

    def test_forged_and_expired_sessions_fail(self):
        for cookie in ('tri_atlas_session=bad','tri_atlas_session=9999999999.fake'):
            self.assertEqual(self.request('/index.html',cookie=cookie)[0],403)
        expiry=str(int(time.time())-1)
        signature=hmac.new(self.secret.encode(),f'atlas-session-v1|{self.root.resolve()}|{expiry}'.encode(),hashlib.sha256).hexdigest()
        self.assertEqual(self.request('/index.html',cookie='tri_atlas_session='+expiry+'.'+signature)[0],403)

    def test_external_secret_and_traversal_are_inaccessible(self):
        _,cookie=self.login()
        self.assertEqual(self.request('/outside.txt',cookie=cookie)[0],403)
        self.assertNotIn(b'must not be served',self.request('/../private-secret',cookie=cookie)[2])

    def test_secret_is_absent_from_access_logs(self):
        log=io.StringIO()
        with contextlib.redirect_stderr(log):self.request('/index.html?key='+self.secret)
        self.assertNotIn(self.secret,log.getvalue());self.assertNotIn('?key=',log.getvalue())


if __name__=='__main__':unittest.main()
