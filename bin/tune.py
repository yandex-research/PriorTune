#!/usr/bin/env python3
"""Fit a PriorTune subset on your own data and predict.

    python bin/tune.py --model lgbm-prep --rule TE --k 25 \
        --train train.csv --test test.csv --target price --out predictions.csv

Takes 20% of the training rows (stratified for classification) as the validation split,
fits the K selected configurations with the training code of the paper, and deploys
them by validation. T keeps the best configuration. TE builds a 40-step greedy
ensemble (Caruana et al., 2004). The task is inferred from the target (`--task`
overrides it). Non-numeric columns are treated as categorical.

lgbm and lgbm-prep run on CPU. mlp (K = 25) and tabpack (K = 64) train the members as one
pack on a CUDA GPU with bfloat16. All need the environment of
bin/requirements-experiments.txt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

BIN = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN))
import artifacts                                                    # noqa: E402

# torch is imported only after LightGBM has trained: on macOS the two OpenMP runtimes
# crash when both are loaded first.
MODELS = ('lgbm', 'lgbm-prep', 'mlp', 'tabpack')

METRIC = {'regression': 'rmse', 'binclass': 'roc-auc', 'multiclass': 'cross-entropy'}


def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return 'binclass' if y.nunique() == 2 else 'multiclass'
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
        return 'binclass' if y.nunique() == 2 else 'multiclass'
    return 'regression'


def write_dataset(folder: Path, train: pd.DataFrame, test: pd.DataFrame, target: str,
                  task: str, val_fraction: float):
    """Your two tables in the dataset layout of bin/priors (one row per sample)."""
    rows = pd.concat([train, test], ignore_index=True)
    features = rows.drop(columns=[target], errors='ignore')
    kinds = {'num': [], 'bin': [], 'cat': []}
    for column in features:
        values = features[column]
        if pd.api.types.is_bool_dtype(values):
            kinds['bin'].append(values.astype(np.float32))
        elif pd.api.types.is_numeric_dtype(values):
            kinds['num'].append(values.astype(np.float32))
        else:
            kinds['cat'].append(pd.Series(pd.factorize(values.astype(str))[0], dtype=np.int64))
    folder.mkdir(parents=True)
    for kind, columns in kinds.items():
        if columns:
            np.save(folder / f'x_{kind}.npy', np.stack([c.to_numpy() for c in columns], axis=1))
    classes = None
    if task == 'regression':
        y = train[target].to_numpy(np.float32)
        y_test = (test[target].to_numpy(np.float32) if target in test
                  else np.zeros(len(test), np.float32))
    else:
        classes = np.sort(train[target].unique())
        code = {c: i for i, c in enumerate(classes)}
        y = train[target].map(code).to_numpy(np.int64)
        y_test = (test[target].map(code).to_numpy(np.int64) if target in test
                  else np.zeros(len(test), np.int64))
    np.save(folder / 'y.npy', np.concatenate([y, y_test]))
    rng = np.random.default_rng(0)
    groups = [np.arange(len(y))] if task == 'regression' else [np.flatnonzero(y == c) for c in np.unique(y)]
    val = np.sort(np.concatenate([rng.permutation(g)[:max(1, round(val_fraction * len(g)))] for g in groups]))
    split = folder / 'splits/default'
    split.mkdir(parents=True)
    np.save(split / 'train.npy', np.setdiff1d(np.arange(len(y)), val))
    np.save(split / 'val.npy', val)
    np.save(split / 'test.npy', np.arange(len(y), len(y) + len(test)))
    (folder / 'info.json').write_text(json.dumps({'task': {'type': task, 'score': METRIC[task]}}))
    return y[val], y_test, classes


def fit_lightgbm(folder: Path, configs: list[dict], work: Path, cpus: int):
    """{config_id: {'val': ..., 'test': ...}} for the configs that trained."""
    runtime = BIN / 'models/_runtime'
    sys.path.insert(0, str(runtime))
    import autogluon
    autogluon.__path__ = [str(runtime / 'ag_overlay/autogluon'), *autogluon.__path__]
    import autogluon.common
    autogluon.common.__path__ = [str(runtime / 'ag_overlay/autogluon/common'), *autogluon.common.__path__]
    import cpu
    cpu.OUTPUT_ROOT = work
    out = {}
    for i, config in enumerate(configs, 1):
        result = cpu.run_config(benchmark='user', dataset=folder.name, dataset_root=folder,
                                config=config, num_cpus=cpus, prediction_parts=('val', 'test'))
        if result['state'] == 'done':
            with np.load(Path(result['path']) / 'predictions.npz') as z:
                out[config['config_id']] = {'val': z['val'], 'test': z['test']}
        print(f'fitted {i}/{len(configs)} (config {config["config_id"]}: {result["state"]})', flush=True)
    return out


def fit_mlp(folder: Path, configs: list[dict], work: Path):
    """Train the configs as one TabPack pack; {config_id: {'val': ..., 'test': ...}}."""
    runtime = BIN / 'models/_runtime'
    sys.path.insert(0, str(runtime))
    os.environ['MODULARML_SUPPRESS_SUMMARY_PRINT'] = '1'
    import torch
    import lib.experiment
    import lib.util
    from bin.tabpack_independent_mlp import train_one
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('mlp and tabpack need a CUDA GPU with bfloat16 support')
    lib.util.configure_logging()
    lib.util.configure_torch()
    pack = runpy.run_path(str(BIN / 'models/mlp/sample_hp.py'))['make_pack_config']
    batch = runpy.run_path(str(BIN / 'priors/tabicl-v2.py'))['nn_batch_size_for_train_size']
    ids = [c['config_id'] for c in configs]
    context = pack(dataset=folder.name, dataset_path=folder, config_ids=ids,
                   batch_size=batch(len(np.load(folder / 'splits/default/train.npy'))),
                   physical_pack_id='priortune')
    context['data']['path'] = str(folder)
    run = work / 'pack'
    run.mkdir(parents=True)
    lib.experiment._run(run)
    train_one(context, run)
    rows = np.load(run / 'prediction_config_ids.npy').tolist()
    with np.load(run / 'member_predictions.npz') as z:
        out = {config_id: {'val': z['val'][r], 'test': z['test'][r]} for r, config_id in enumerate(rows)}
    return {i: p for i, p in out.items() if np.isfinite(p['val']).all()}


def deploy(predictions: dict, y_val, y_test, task: str, steps: int):
    """Greedy ensemble selection on validation (steps = 1: the best single config)."""
    sys.path.insert(0, str(BIN / 'hillclimb'))
    import ensemble as te
    ids = sorted(predictions)
    squeeze = lambda a: a[..., 0] if a.ndim == 2 and a.shape[1] == 1 else a
    val = np.stack([squeeze(predictions[i]['val']) for i in ids])
    test = np.stack([squeeze(predictions[i]['test']) for i in ids])
    ds = te.prepare(val, test, y_val, y_test, METRIC[task], np.ones(len(ids), bool), np.zeros(len(ids)))
    counts = te.caruana(ds, np.arange(len(ids)), steps)
    weights = counts / steps
    def blend(p):
        mixed = np.tensordot(weights, p.astype(np.float64), axes=1)
        return mixed / mixed.sum(axis=-1, keepdims=True) if task == 'multiclass' else mixed
    return blend(val), blend(test), {ids[j]: int(c) for j, c in enumerate(counts) if c}


def score(y, p, task):
    from sklearn import metrics
    if task == 'regression':
        return 'rmse', metrics.root_mean_squared_error(y, p)
    if task == 'binclass':
        return 'roc-auc', metrics.roc_auc_score(y, p)
    return 'log-loss', metrics.log_loss(y, p, labels=np.arange(p.shape[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--rule', choices=['T', 'TE'], default='TE')
    parser.add_argument('--k', type=int, choices=(25, 64, 200), default=None,
                        help='subset size (default 25, tabpack 64)')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--task', choices=list(METRIC))
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--cpus', type=int, default=os.cpu_count())
    parser.add_argument('--out', type=Path, default=Path('predictions.csv'))
    args = parser.parse_args()
    k = args.k or (64 if args.model == 'tabpack' else 25)
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    task = args.task or infer_task(train[args.target])
    record = artifacts.portfolio(args.model, args.rule, k)
    configs = [next(c for c in record['configs'] if c['config_id'] == i) for i in record['add_order']]
    print(f'{args.model}, {args.rule}, K = {k}, task: {task}, '
          f'{len(train)} training rows, {len(test)} test rows', flush=True)
    work = Path(tempfile.mkdtemp(prefix='priortune-'))
    try:
        folder = work / 'data'
        y_val, y_test, classes = write_dataset(folder, train, test, args.target, task, args.val_fraction)
        fitted = (fit_mlp(folder, configs, work) if args.model in ('mlp', 'tabpack')
                  else fit_lightgbm(folder, configs, work / 'fits', args.cpus))
        if not fitted:
            raise SystemExit('no configuration trained')
        p_val, p_test, used = deploy(fitted, y_val, y_test, task, 40 if args.rule == 'TE' else 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    name, value = score(y_val, p_val, task)
    print(f'deployed {len(used)} of {len(fitted)} configurations, validation {name} {value:.4f}')
    if args.target in test:
        name, value = score(y_test, p_test, task)
        print(f'test {name} {value:.4f}')
    if task == 'regression':
        out = pd.DataFrame({'prediction': p_test})
    elif task == 'binclass':
        out = pd.DataFrame({f'proba_{classes[1]}': p_test})
    else:
        out = pd.DataFrame(p_test, columns=[f'proba_{c}' for c in classes])
    out.to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
