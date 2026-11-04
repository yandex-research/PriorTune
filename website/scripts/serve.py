#!/usr/bin/env python3
"""Serve PriorTune on loopback, with optional local live reload."""

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {'index.html', 'app.js', 'styles.css', 'figures-ba.js', 'figures-ba.css', 'demo.js', 'demo.css', 'figures-hp.js', 'figures-hp.css', 'assets', 'data', 'manifest.json'}


class SiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, live_reload=False, **kwargs):
        self.live_reload = live_reload
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if self.live_reload and path == '/__preview_version':
            self.send_payload(self.preview_version().encode(), 'text/plain')
            return
        if self.is_private(path):
            self.send_error(404)
            return
        if self.live_reload and path in ('/', '/index.html'):
            index = Path(self.directory) / 'index.html'
            if index.is_file():
                script = '''<script>(()=>{let version=null;setInterval(async()=>{try{const response=await fetch('/__preview_version',{cache:'no-store'});if(!response.ok)return;const next=await response.text();if(version!==null&&next!==version)location.reload();version=next;}catch{}},1500);})();</script>'''
                content = index.read_text(encoding='utf-8').replace('</body>', script + '</body>')
                self.send_payload(content.encode(), 'text/html; charset=utf-8')
                return
        super().do_GET()

    def do_HEAD(self):
        if self.is_private(unquote(urlsplit(self.path).path)):
            self.send_error(404)
            return
        super().do_HEAD()

    def is_private(self, path):
        parts = Path(path).parts[1:]
        if not parts:
            return False
        return parts[0] not in PUBLIC or any(part.startswith('.') for part in parts)

    def preview_version(self):
        root = Path(self.directory)
        paths = [root / name for name in ('index.html', 'app.js', 'styles.css', 'figures-ba.js', 'figures-ba.css', 'demo.js', 'demo.css', 'figures-hp.js', 'figures-hp.css')]
        for name in ('assets', 'data'):
            if (root / name).is_dir():
                paths.extend(path for path in (root / name).rglob('*') if path.is_file())
        state = '|'.join('{}:{}:{}'.format(path, path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(paths) if path.is_file())
        return hashlib.sha256(state.encode()).hexdigest()

    def send_payload(self, payload, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def list_directory(self, path):
        self.send_error(404, 'Directory listing is disabled')
        return None

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, format, *args):
        if '/__preview_version' not in self.path:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=4175)
    parser.add_argument('--dist', action='store_true', help='Serve the packaged public bundle')
    parser.add_argument('--no-live-reload', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'dist' if args.dist else ROOT
    if not directory.is_dir():
        raise SystemExit('Run python3 scripts/package_site.py first.')
    reload_enabled = not args.dist and not args.no_live_reload
    handler = partial(SiteHandler, directory=str(directory), live_reload=reload_enabled)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print('Local: http://127.0.0.1:{}/'.format(args.port), flush=True)
    print('Serving {}{}'.format(directory, ' (live reload enabled)' if reload_enabled else ''), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
