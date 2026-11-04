#!/usr/bin/env python3
"""Select a portfolio of K configurations with Algorithm 1, under T or T+E.

T needs the synthetic score matrix and the validation worlds (worlds.py); T+E also the
synthetic predictions and labels. The published portfolios are repetition 0 of each
model's random streams with R = 16 paired draws, on all 4,000 configs and datasets:

    python bin/hillclimb/portfolio.py t  --model lgbm-prep --k 25 --worlds worlds/lgbm-prep_R16.npz
    python bin/hillclimb/portfolio.py te --model mlp --k 25
    python bin/hillclimb/portfolio.py te --model mlp --k 200 --reference-k 25 \
        --start TE_K64.json --universe 2000 --schedule artifacts/portfolios/mlp/TE_K200.json

T portfolios are separate climbs, each against random-K references. T+E portfolios
are greedy trajectories against random-25 references that later K continue: K = 25 on
every dataset with every world, K = 64 continues it the same way; K = 200 continues
K = 25 (LightGBM) or K = 64 (MLP) on the first 2,000 datasets of repetition 0 under a
per-step budget (a few of the 16 worlds and a rotating slice of the datasets per step),
which the published portfolio records in its `steps` and `--schedule` replays.

This is the reference implementation of the rule. T runs on one GPU in minutes. T+E
holds every synthetic dataset's reduced predictions in memory and refits ensembles for
every candidate that forks the incumbent's trajectory; at 4,000 x 4,000 the published
runs sharded that work over many GPUs for days.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifacts                                                    # noqa: E402
import core                                                         # noqa: E402
import ensemble as te                                               # noqa: E402
from worlds import resample_counts                                  # noqa: E402


def select_t(synth: core.Cache, delta: np.ndarray, prefix: tuple, k: int, draws: int = 16,
             seed: int = 0, device='cpu'):
    """The T portfolio of repetition 0: add order and the climb's own half-win rate.

    The full-pool cell of the scaling grid: every dataset and the first min(C, D) configs
    of repetition 0's order, which is every config when C <= D (4,000 x 4,000)."""
    n_datasets, n_configs = synth.val.shape
    cfg, _ = core.shuffles(seed, prefix, n_configs, n_datasets, 1)
    refs = core.references(seed, prefix, cfg, k, draws)[0]
    ranks = core.world_ranks(synth.val, None if delta is None else delta[..., :draws], synth.available)
    ranks, credit, empty = core.credit_tensors(synth, ranks, refs, device)
    return core.climb(ranks, credit, empty, cfg[0, :min(n_configs, n_datasets)],
                      np.arange(n_datasets), k)


def bootstrap_rows(y_val, metric, worlds, seed, prefix, dataset):
    """The validation rows of each world, as index arrays: the resamples of worlds.py."""
    positive = (np.asarray(y_val) > 0.5) if metric == 'roc-auc' else None
    counts = resample_counts(len(y_val), worlds, seed, prefix, dataset, positive)
    return [torch.as_tensor(np.repeat(np.arange(len(y_val)), row)) for row in counts]


def select_te(datasets, resamples, bank: np.ndarray, k: int, start=(), schedule=None, log=print):
    """Greedy T+E-objective climb.

    datasets   list of ensemble.Dataset (None: no config ran there)
    resamples  per dataset, the R world index arrays
    bank       (R, K') paired random references
    start      an add order to continue from
    schedule   step -> (worlds, dataset positions) credited at that step; default all
    Returns the add order and, per step, the credit of every remaining candidate.
    """
    draws = len(bank)
    bar = [None if ds is None else
           np.asarray([te.deploy(ds.resample(rows[r]), bank[r]) for r in range(draws)])
           for ds, rows in zip(datasets, resamples)]
    n_configs = len(next(ds for ds in datasets if ds is not None).available)
    selected, history = list(start), []
    while len(selected) < k:
        step = len(selected) + 1
        worlds, active = schedule(step) if schedule else (range(draws), range(len(datasets)))
        remaining = np.setdiff1d(np.arange(n_configs), selected)
        objective = np.zeros(len(remaining), dtype=np.int64)
        for i in active:
            ds = datasets[i]
            if ds is None:
                objective += len(worlds)    # an empty portfolio ties every reference
            elif not selected:
                objective += te.candidate_credit(ds, bar[i], [], remaining)
            else:
                for r in worlds:
                    objective += te.candidate_credit(ds.resample(resamples[i][r]), bar[i][r],
                                                     selected, remaining)
        history.append(objective)
        selected.append(int(remaining[int(np.argmax(objective))]))
        log(f'step {step}: config {selected[-1]}, '
            f'half-win {objective.max() / (2 * len(active) * len(worlds)):.4f}')
    return np.asarray(selected), history


def recorded_schedule(steps: list[dict], universe: int):
    """The budget of the K = 200 continuations: at step t, the recorded worlds and a
    rotating slice of `datasets` positions of the universe starting at t * datasets."""
    by_step = {s['step']: s for s in steps}

    def schedule(step):
        if step not in by_step:
            return range(16), range(universe)
        n = by_step[step]['datasets']
        return by_step[step]['worlds'], [(step * n + j) % universe for j in range(n)]
    return schedule


def load_te_inputs(model, corpus, worlds, seed, universe):
    """Prepared ensemble.Dataset per dataset of `universe` (ids, in climb order), with
    its world resamples."""
    cache = core.load(artifacts.scores(corpus, model))
    labels = artifacts.labels(corpus)
    prefix = core.prefix(model)
    position = {int(d): i for i, d in enumerate(universe)}
    datasets, resamples = [None] * len(universe), [None] * len(universe)
    for d, preds in artifacts.predictions(corpus, model):
        if d not in position or not cache.available[d].any():
            continue
        i = position[d]
        metric = str(cache.metrics[d])
        y_val, y_test = labels[(d, 'val')], labels[(d, 'test')]
        datasets[i] = te.prepare(preds['val'], preds['test'], y_val, y_test, metric,
                                 cache.available[d], cache.test[d])
        resamples[i] = bootstrap_rows(y_val, metric, worlds, seed, prefix, d)
    return cache, datasets, resamples


def export(model, objective, k, order, protocol):
    record = dict(model=model, objective=objective, K=k, config_ids=sorted(order.tolist()),
                  add_order=order.tolist(), protocol=protocol)
    try:
        bank = {c['config_id']: c for c in artifacts.configs(model)}
        record['configs'] = [bank[i] for i in record['config_ids']]
    except (FileNotFoundError, KeyError):
        pass                                # a model or config ids outside the published banks
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('objective', choices=['t', 'te'])
    parser.add_argument('--model', required=True, help='lgbm, lgbm-prep, mlp, or your own')
    parser.add_argument('--scores', type=Path,
                        help='a score matrix of your own (bin/models/collect.py); default: the dataset')
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--draws', type=int, default=16, help='R: paired worlds and references')
    parser.add_argument('--reference-k', type=int, help='T+E reference size (published: 25)')
    parser.add_argument('--worlds', type=Path,
                        help='T: worlds file from worlds.py (without it, the original validation split)')
    parser.add_argument('--start', type=Path, help='T+E: continue from this portfolio JSON')
    parser.add_argument('--universe', type=int,
                        help='T+E: climb on the first N datasets of repetition 0 (K = 200: 2,000)')
    parser.add_argument('--schedule', type=Path,
                        help="T+E: replay the per-step budget recorded in this portfolio's 'steps'")
    parser.add_argument('--corpus', default='synthetic-tabicl-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=Path('portfolios'))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    prefix = core.prefix(args.model)
    arrays = dict(np.load(args.scores)) if args.scores else artifacts.scores(args.corpus, args.model)
    synth = core.load(arrays)
    ids = arrays.get('config_ids', np.arange(synth.val.shape[1]))   # column -> config id
    protocol = dict(seed=args.seed, repetition=0, draws=args.draws,
                    configs=synth.val.shape[1], datasets=synth.val.shape[0])
    if args.objective == 't':
        delta = np.load(args.worlds)['delta'] if args.worlds else None
        order, halfwin = select_t(synth, delta, prefix, args.k, args.draws, args.seed, args.device)
        protocol.update(rule='T', reference_k=args.k, objective_halfwin=halfwin)
    else:
        reference_k = args.reference_k or args.k
        cfg, data = core.shuffles(args.seed, prefix, *synth.val.shape[::-1], 1)
        bank = core.references(args.seed, prefix, cfg, reference_k, args.draws)[0]
        start = json.loads(args.start.read_text())['add_order'] if args.start else ()
        universe = data[0][:args.universe or len(data[0])]
        _, datasets, resamples = load_te_inputs(args.model, args.corpus, args.draws, args.seed, universe)
        schedule = (recorded_schedule(json.loads(args.schedule.read_text())['steps'], len(universe))
                    if args.schedule else None)
        order, _ = select_te(datasets, resamples, bank, args.k, start, schedule)
        protocol.update(rule='T+E', reference_k=reference_k, caruana_steps=te.STEPS,
                        datasets=len(universe))
    order = np.asarray(ids)[order]
    name = 'T' if args.objective == 't' else 'TE'
    out = args.out / args.model / f'{name}_K{args.k}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export(args.model, name, args.k, order, protocol), indent=1) + '\n')
    print(f'wrote {out}: {order.tolist()}')


if __name__ == '__main__':
    main()
