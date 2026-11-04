#!/usr/bin/env python3
"""Collect fits into a score matrix, or add them to an existing one.

    python bin/models/collect.py --fits local/fits/lgbm-prep --datasets local/datasets/tabicl-v2 \
        --extend artifacts/scores/synthetic-tabicl-v2/lgbm-prep.npz --out local/lgbm-prep.npz

`--fits` holds the `<dataset>/<config_id>/report.json` folders that the model scripts
write. The result has the layout of the dataset's `scores/` files, so every script that
takes `--scores` can select on it. With `--extend`, the existing cells are kept and the
new datasets and configs are added (rows and columns are sorted by name and id).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASK = {'rmse': 'regression', 'roc-auc': 'binclass', 'cross-entropy': 'multiclass'}


def read_fits(fits: Path, datasets: Path):
    """{(dataset, config_id): (validation score, test score)} of the fits that finished."""
    cells, metric = {}, {}
    for report in sorted(fits.glob('*/*/report.json')):
        folder = report.parent
        if not (folder / 'DONE').exists():
            continue
        name, config_id = folder.parent.name, int(folder.name)
        if name not in metric:
            metric[name] = json.loads((datasets / name / 'info.json').read_text())['task']['score']
        scores = json.loads(report.read_text())['metrics']
        cells[(name, config_id)] = (scores['val'][metric[name]], scores['test'][metric[name]])
    return cells, metric


def build(cells, metric, base=None):
    names = sorted({d for d, _ in cells} | set(base['datasets'].tolist() if base else []))
    ids = sorted({c for _, c in cells} | set(range(base['validation'].shape[1]) if base else []))
    row, col = {n: i for i, n in enumerate(names)}, {c: j for j, c in enumerate(ids)}
    val = np.full((len(names), len(ids)), np.nan)
    test = np.full_like(val, np.nan)
    if base:
        r = [row[n] for n in base['datasets'].tolist()]
        val[np.ix_(r, range(base['validation'].shape[1]))] = np.where(base['available'], base['validation'], np.nan)
        test[np.ix_(r, range(base['validation'].shape[1]))] = np.where(base['available'], base['test'], np.nan)
        metric = dict(zip(base['datasets'].tolist(), base['metrics'].tolist())) | metric
    for (name, config_id), (v, t) in cells.items():
        val[row[name], col[config_id]], test[row[name], col[config_id]] = v, t
    metrics = np.asarray([metric[n] for n in names])
    return dict(validation=val, test=test, available=np.isfinite(val) & np.isfinite(test),
                datasets=np.asarray(names), metrics=metrics,
                task_types=np.asarray([TASK[m] for m in metrics]), config_ids=np.asarray(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fits', type=Path, required=True, help='<output>/<model> of a model script')
    parser.add_argument('--datasets', type=Path, required=True, help='the dataset folders that were fitted')
    parser.add_argument('--extend', type=Path, help='an existing score matrix to add to')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cells, metric = read_fits(args.fits, args.datasets)
    base = None
    if args.extend:
        with np.load(args.extend, allow_pickle=False) as z:
            base = {k: z[k] for k in z.files}
    out = build(cells, metric, base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'wrote {args.out}: {out["validation"].shape[0]} datasets x {out["validation"].shape[1]} configs, '
          f'{len(cells)} new fits, {int(out["available"].sum())} cells')


if __name__ == '__main__':
    main()
