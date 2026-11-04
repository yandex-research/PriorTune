#!/usr/bin/env python3
"""Package only reviewed public website files for static hosting."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'dist'
REQUIRED = ('index.html', 'app.js', 'styles.css', 'figures-ba.js', 'figures-ba.css', 'demo.js', 'demo.css', 'figures-hp.js', 'figures-hp.css')
PUBLIC_DIRECTORIES = ('assets', 'data')


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit('Missing site files: ' + ', '.join(missing))
    public_files = [ROOT / name for name in REQUIRED]
    for name in PUBLIC_DIRECTORIES:
        source = ROOT / name
        if source.is_symlink():
            raise SystemExit('Public asset directories cannot be symlinks: ' + name)
        if source.is_dir():
            for path in sorted(source.rglob('*')):
                relative = path.relative_to(ROOT)
                if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
                    continue
                if path.is_symlink():
                    raise SystemExit('Public files cannot be symlinks: ' + str(relative))
                if path.is_file():
                    public_files.append(path)
    for path in public_files:
        if path.is_symlink():
            raise SystemExit('Public files cannot be symlinks: ' + str(path.relative_to(ROOT)))
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    for path in public_files:
        target = OUTPUT / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (OUTPUT / '.nojekyll').write_text('', encoding='utf-8')
    checksums = {
        str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUTPUT.rglob('*')) if path.is_file()
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps({'algorithm': 'sha256', 'files': checksums}, indent=2) + '\n',
        encoding='utf-8',
    )
    size = sum(path.stat().st_size for path in OUTPUT.rglob('*') if path.is_file())
    print('Built {} public files ({:,} bytes) in {}'.format(len(checksums) + 1, size, OUTPUT))


if __name__ == '__main__':
    main()
