"""A template for adding a model: a random forest, its own config bank, and fits that
bin/models/collect.py turns into a score matrix.

    python examples/new_model.py bank --n 100 --out local/rf-100.jsonl
    python examples/new_model.py fit --bank local/rf-100.jsonl --dataset local/datasets/tabicl-v2/reg-0000 \
        --config-id 0 --output local/fits

A model needs two things:
  a bank     JSON lines, one config per line, each with a `config_id`
  a fit      for one (dataset, config): <output>/<model>/<dataset>/<config_id>/ with
             report.json ({"metrics": {"val": {<metric>: ...}, "test": {<metric>: ...}}}),
             predictions.npz (val, test) and an empty DONE file
where <metric> is the dataset's native metric from info.json (rmse, roc-auc, cross-entropy).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import ensemble, metrics

MODEL = 'rf'


def bank(n, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(config_id=i, n_estimators=int(rng.integers(50, 500)),
                 max_features=float(rng.uniform(0.2, 1.0)),
                 min_samples_leaf=int(np.exp(rng.uniform(0, np.log(64)))),
                 random_state=i) for i in range(n)]


def load(folder: Path):
    """A dataset folder of bin/priors: features, labels and the three splits."""
    parts = [np.load(folder / f'x_{k}.npy') for k in ('num', 'bin', 'cat') if (folder / f'x_{k}.npy').exists()]
    x, y = np.concatenate(parts, axis=1).astype(np.float32), np.load(folder / 'y.npy')
    split = {p: np.load(folder / f'splits/default/{p}.npy') for p in ('train', 'val', 'test')}
    return x, y, split, json.loads((folder / 'info.json').read_text())['task']


def fit(config, folder: Path, output: Path):
    x, y, split, task = load(folder)
    params = {k: v for k, v in config.items() if k != 'config_id'}
    if task['type'] == 'regression':
        model = ensemble.RandomForestRegressor(**params).fit(x[split['train']], y[split['train']])
        predict = model.predict
    else:
        model = ensemble.RandomForestClassifier(**params).fit(x[split['train']], y[split['train']])
        predict = (lambda a: model.predict_proba(a)[:, 1]) if task['type'] == 'binclass' else model.predict_proba
    predictions = {p: predict(x[split[p]]) for p in ('val', 'test')}
    score = {'rmse': lambda t, p: metrics.root_mean_squared_error(t, p),
             'roc-auc': metrics.roc_auc_score,
             'cross-entropy': lambda t, p: metrics.log_loss(t, p, labels=np.arange(p.shape[1]))}[task['score']]
    report = {'metrics': {p: {task['score']: float(score(y[split[p]], predictions[p]))} for p in predictions},
              'config': config}
    dest = output / MODEL / folder.name / str(config['config_id'])
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'predictions.npz', **predictions)
    (dest / 'report.json').write_text(json.dumps(report, indent=1))
    (dest / 'DONE').touch()
    print(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bank')
    b.add_argument('--n', type=int, default=100)
    b.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('fit')
    f.add_argument('--bank', type=Path, required=True)
    f.add_argument('--dataset', type=Path, required=True)
    f.add_argument('--config-id', type=int, required=True)
    f.add_argument('--output', type=Path, default=Path('local/fits'))
    args = parser.parse_args()
    if args.command == 'bank':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(''.join(json.dumps(c) + '\n' for c in bank(args.n)))
        print(f'wrote {args.out}')
    else:
        configs = {c['config_id']: c for c in map(json.loads, args.bank.read_text().splitlines())}
        fit(configs[args.config_id], args.dataset, args.output)


if __name__ == '__main__':
    main()
