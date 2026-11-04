#!/usr/bin/env python3
"""Browser smoke checks and screenshots for the locally served website.

Requires the optional Playwright Python package. It does not publish anything.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import urljoin, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def packaged_preview(directory):
    """Expose only the deployable files, beneath a repository-style URL prefix."""
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    with TemporaryDirectory(prefix="priortune-site-check-") as temporary:
        shutil.copytree(directory, Path(temporary) / "priortune")
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=temporary))
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/priortune/"
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


def check_site(args):
    args.output.mkdir(parents=True, exist_ok=True)
    checks, errors, failed_requests, asset_requests = [], [], [], []

    def check(condition, name, evidence=None):
        checks.append({'pass': bool(condition), 'check': name, 'evidence': evidence})

    with sync_playwright() as p:
        options = {'headless': True}
        if args.browser:
            options['executable_path'] = args.browser
        else:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            if chrome.is_file():
                options['executable_path'] = str(chrome)
        browser = p.chromium.launch(**options)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
        page.on('requestfailed', lambda request: failed_requests.append({'url': request.url, 'failure': request.failure}))
        page.on('request', lambda request: asset_requests.append(request.url)
                if request.resource_type in ('script', 'stylesheet', 'fetch') else None)
        response = page.goto(args.url, wait_until='networkidle')
        page.wait_for_selector(args.ready_selector)
        check(response.status == 200, 'Page responds with HTTP 200', response.status)
        check(bool(page.title()), 'Page has a document title', page.title())
        check(page.locator('h1').count() == 1, 'Page has one primary heading')
        check(bool(page.locator('html').get_attribute('lang')), 'Document language is declared')
        page.evaluate('window.priortuneDemo.pause()')
        homepage = page.locator('.hero-copy').text_content()
        check(page.locator('h1').text_content() == 'Search less.Choose better.'
              and page.locator('.hero-steps li').count() == 3,
              'Permanent homepage presents the selected headline and three usage steps')
        check(page.locator('html').get_attribute('data-copy-version') is None
              and page.locator('a[href*="version="]').count() == 0,
              'Homepage has no wording-version state or version links')
        check(page.locator('#elo-chart .ba-mark').count() == 16
              and page.locator('#random-chart svg,#scaling-chart svg,#hp-chart svg').count() == 9
              and page.locator('#heatmap-chart .heatmap-cell').count() == 300
              and page.locator('.hp-slot option').count() == 36,
              'Standalone entry initializes all current figures and parameter choices')
        page.evaluate("() => { document.activeElement.blur(); window.scrollTo({top:0,behavior:'instant'}); }")
        page.screenshot(path=str(args.output / 'desktop-hero.png'))
        page.screenshot(path=str(args.output / 'desktop-full.png'), full_page=True)
        for section in page.locator('main section[id]').all():
            identifier = section.get_attribute('id')
            section.screenshot(path=str(args.output / ('desktop-' + identifier + '.png')), style='header,.skip-link { visibility:hidden!important; }')

        unnamed_controls = page.locator('input:not([type="hidden"]),select,textarea').evaluate_all('''nodes => nodes.filter(e => !e.disabled && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby') && !e.labels?.length).map(e => e.outerHTML.slice(0,200))''')
        check(not unnamed_controls, 'Form controls have accessible labels', unnamed_controls)
        unnamed_buttons = page.locator('button').evaluate_all('''nodes => nodes.filter(e => !e.textContent.trim() && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby') && !e.getAttribute('title')).map(e => e.outerHTML.slice(0,200))''')
        check(not unnamed_buttons, 'Buttons have accessible names', unnamed_buttons)

        for width in (1440, 1024, 768, 390, 320):
            page.set_viewport_size({'width': width, 'height': 900})
            page.evaluate("() => { document.activeElement.blur(); window.scrollTo({top:0,behavior:'instant'}); }")
            page.wait_for_timeout(150)
            dimensions = page.evaluate('({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})')
            check(dimensions['scroll'] <= dimensions['width'], 'No horizontal page overflow at {} px'.format(width), dimensions)
            page.screenshot(path=str(args.output / ('viewport-' + str(width) + '.png')))
            if width in (390, 320):
                page.screenshot(path=str(args.output / ('mobile-' + str(width) + '-full.png')), full_page=True)
            if width == 390:
                for section in page.locator('main section[id]').all():
                    identifier = section.get_attribute('id')
                    section.screenshot(path=str(args.output / ('mobile-' + identifier + '.png')), style='header,.skip-link { visibility:hidden!important; }')

        links = page.locator('[href],[src]').evaluate_all("nodes => nodes.map(e => e.getAttribute('href') || e.getAttribute('src')).filter(Boolean)")
        for href in sorted(set(links)):
            if href.startswith('#'):
                found = href == '#' or page.evaluate('(id) => !!document.getElementById(id)', href[1:])
                check(found, 'Internal anchor resolves: ' + href)
            elif not urlsplit(href).scheme and not href.startswith('//'):
                resource = context.request.get(urljoin(args.url, href))
                check(resource.status == 200, 'Local resource resolves: ' + href, resource.status)
        original = urlsplit(args.url)
        legacy = urlunsplit((original.scheme, original.netloc, original.path, 'version=a&v=7', 'random-search'))
        page.goto(legacy, wait_until='networkidle')
        page.wait_for_selector(args.ready_selector)
        canonical = urlsplit(page.url)
        check(canonical.path == original.path and canonical.query == '' and canonical.fragment == 'random-search',
              'Legacy URL becomes the canonical homepage while preserving its path and anchor', page.url)
        check(page.locator('.hero-copy').text_content() == homepage
              and page.locator('html').get_attribute('data-copy-version') is None,
              'Legacy wording query cannot change the permanent homepage')
        page.locator('nav[aria-label="Main"] a[href="#results"]').click()
        check(urlsplit(page.url).query == '' and urlsplit(page.url).fragment == 'results',
              'In-page navigation stays on the canonical URL')
        local_assets = [u for u in asset_requests if urlsplit(u).netloc == original.netloc]
        check(local_assets and all(not urlsplit(u).query and not urlsplit(u).path.endswith('/copy.js') for u in local_assets),
              'Application modules, styles and data load without versioned URLs or the retired copy module', local_assets)
        if args.dist:
            manifest = json.loads((args.dist / 'manifest.json').read_text())['files']
            check(all(hashlib.sha256((args.dist / name).read_bytes()).hexdigest() == digest
                      for name, digest in manifest.items()), 'Packaged content matches its SHA-256 manifest')
            check('copy.js' not in manifest and not any(name.startswith(('.qa/', 'source-artifacts/', 'scripts/')) for name in manifest),
                  'Package excludes retired comparison code and private development inputs')
        check(not errors, 'No JavaScript or console errors', errors)
        check(not failed_requests, 'No failed browser requests', failed_requests)
        browser.close()

    report = {'url': args.url, 'passed': sum(item['pass'] for item in checks), 'total': len(checks), 'checks': checks}
    (args.output / 'browser-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('{}/{} browser checks passed; screenshots and report: {}'.format(report['passed'], report['total'], args.output))
    for item in checks:
        if not item['pass']:
            print('FAIL: {}: {}'.format(item['check'], item['evidence']))
    raise SystemExit(0 if all(item['pass'] for item in checks) else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:4175/')
    parser.add_argument('--dist', type=Path, nargs='?', const=ROOT / 'dist',
                        help='Serve a built directory under a temporary /priortune/ prefix')
    parser.add_argument('--browser', help='Path to a Chrome/Chromium executable')
    parser.add_argument('--output', type=Path, default=ROOT / '.qa')
    parser.add_argument('--ready-selector', default='html[data-ready="true"]')
    args = parser.parse_args()
    if args.dist:
        if not (args.dist / 'index.html').is_file():
            parser.error('Build the deployment directory with scripts/package_site.py first.')
        with packaged_preview(args.dist) as url:
            args.url = url
            check_site(args)
    else:
        check_site(args)


if __name__ == '__main__':
    main()
