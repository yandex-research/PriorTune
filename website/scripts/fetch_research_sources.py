#!/usr/bin/env python3
"""Restore the pinned compact figure inputs, with SHA256 verification.

The site serves from committed data/research.json without downloads. To reproduce
that file from a fresh clone:

    python3 scripts/fetch_research_sources.py
    python3 scripts/build_research_data.py

Only the public figure CSVs, plotting definitions, two small configuration banks,
and three selected portfolios are downloaded (about 6 MB). No raw datasets,
predictions, fitting, credentials, or third-party Python packages are required.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def restore(destination: Path, manifest: Path):
    meta = json.loads(manifest.read_text())["meta"]
    origins = {
        "upstream": {"repo": "https://github.com/yandex-research/PriorTune", "commit": meta["commit"],
                     "base": "https://raw.githubusercontent.com/yandex-research/PriorTune/"},
        "huggingface": {"repo": "https://huggingface.co/datasets/mrsergazinov/PriorTune", "commit": meta["hfCommit"],
                        "base": "https://huggingface.co/datasets/mrsergazinov/PriorTune/resolve/"},
    }
    if meta["repo"] != origins["upstream"]["repo"] or meta["hfRepo"] != origins["huggingface"]["repo"]:
        raise ValueError("Unexpected source repository")
    for source in origins.values():
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise ValueError("Expected a full pinned commit")
    entries = sorted(meta["sourceHashes"].items())
    for name, expected in entries:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in origins or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid source entry: {name}")

    def fetch(item):
        name, expected = item
        target = destination / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return 0
        origin, relative = name.split("/", 1)
        source = origins[origin]
        url = source["base"] + source["commit"] + "/" + relative
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".download")
        temp.write_bytes(payload)
        temp.replace(target)
        return len(payload)

    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = sum(pool.map(fetch, entries))
    for origin, source in origins.items():
        record = {"repo": source["repo"], "commit": source["commit"], "files": [p.split("/", 1)[1] for p, _ in entries if p.startswith(origin + "/")]}
        (destination / origin / "REVISION.json").write_text(json.dumps(record, indent=2) + "\n")
    return len(entries), downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=ROOT / "source-artifacts")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/research.json")
    args = parser.parse_args()
    count, downloaded = restore(args.destination.resolve(), args.manifest.resolve())
    print(f"Verified {count} pinned inputs; downloaded {downloaded:,} bytes.")
