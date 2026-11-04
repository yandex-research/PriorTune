#!/usr/bin/env python3
"""Validation bootstrap worlds of a pool, from its raw predictions.

Each synthetic dataset's validation rows are resampled R times with replacement; the
resample is shared by all configs of the dataset, so the worlds disagree about which
config wins, not about how hard the dataset is. Every config is rescored on every
resample with the dataset's native metric under integer row weights, and the deviation
from the full-sample score is stored:

    delta[d, c, r] = loss(c, resample r) - loss(c, full validation)     float32

so that world r's validation loss is the score cache's plus delta[..., r], and a world
that redraws the full sample ties exactly where the cache ties. A ROC-AUC resample must
hold both classes; a draw that does not is redrawn under an extended key.

    python bin/hillclimb/worlds.py --model mlp --out worlds/mlp_R16.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifacts                                                    # noqa: E402
import core                                                         # noqa: E402

MAX_REDRAWS = 64
MULTI_FLOOR = 1e-45


def resample_counts(n_rows: int, worlds: int, seed: int, prefix: tuple, dataset: int,
                    positive: np.ndarray | None = None) -> np.ndarray:
    """(worlds, n_rows) multinomial counts; with `positive`, each resample holds both classes."""
    counts = np.empty((worlds, n_rows), dtype=np.int64)
    p = np.full(n_rows, 1.0 / n_rows)
    for r in range(worlds):
        for attempt in range(MAX_REDRAWS):
            key = (*prefix, dataset, core.WORLD, r) + ((attempt,) if attempt else ())
            draw = core.rng(seed, *key).multinomial(n_rows, p)
            if positive is None or (draw[positive].sum() and draw[~positive].sum()):
                counts[r] = draw
                break
        else:
            raise RuntimeError(f'no two-class resample in {MAX_REDRAWS} draws (dataset {dataset})')
    return counts


def weighted_auc(y: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """(configs, worlds) ROC-AUC under integer row weights, ties counted as half:
    sum over positives i of w_i (W_neg below s_i + W_neg tied with s_i / 2) / (W_pos W_neg)."""
    positive = (y > 0.5).to(torch.float64)
    w_pos, w_neg = positive @ weights, (1.0 - positive) @ weights
    order = scores.argsort(dim=-1, stable=True)
    ordered = scores.gather(-1, order)
    lo = torch.searchsorted(ordered, ordered, right=False)
    hi = torch.searchsorted(ordered, ordered, right=True)
    n_configs, n_rows = scores.shape
    worlds = weights.shape[1]
    out = torch.empty((n_configs, worlds), dtype=torch.float64)
    step = max(1, int(2e7 // max(1, worlds * (n_rows + 1))))
    for start in range(0, n_configs, step):
        block = slice(start, start + step)
        o = order[block]
        w = weights.T[None].expand(o.shape[0], -1, -1).gather(2, o[:, None, :].expand(-1, worlds, -1))
        pos = positive[o][:, None, :]
        negatives = torch.nn.functional.pad((w * (1.0 - pos)).cumsum(dim=-1), (1, 0))
        below = negatives.gather(2, lo[block][:, None, :].expand(-1, worlds, -1))
        upto = negatives.gather(2, hi[block][:, None, :].expand(-1, worlds, -1))
        out[block] = (w * pos * 0.5 * (below + upto)).sum(dim=-1)
    return out / (w_pos * w_neg)[None]


def weighted_losses(y: np.ndarray, preds: torch.Tensor, weights: torch.Tensor, metric: str):
    """(configs, worlds) loss-oriented native score of float64 predictions."""
    if metric == 'roc-auc':
        return -weighted_auc(torch.as_tensor(y, dtype=torch.float64), preds, weights)
    if metric == 'rmse':
        rows = (preds - torch.as_tensor(y, dtype=torch.float64)) ** 2
        return ((rows @ weights) / weights.sum(dim=0)).sqrt()
    labels = torch.as_tensor(y, dtype=torch.int64)
    index = labels.view(1, -1, 1).expand(*preds.shape[:-1], 1)
    true = preds.gather(-1, index).squeeze(-1)
    rows = -(true / preds.sum(dim=-1)).clamp(MULTI_FLOOR, 1.0).log()
    return (rows @ weights) / weights.sum(dim=0)


def dataset_delta(val: np.ndarray, y: np.ndarray, metric: str, available: np.ndarray,
                  worlds: int, seed: int, prefix: tuple, dataset: int):
    """(configs, worlds) float32 deviations (NaN where the config failed), and the
    full-sample loss the deviations are relative to."""
    positive = (y > 0.5) if metric == 'roc-auc' else None
    counts = resample_counts(len(y), worlds, seed, prefix, dataset, positive)
    weights = torch.as_tensor(np.concatenate([np.ones((1, len(y))), counts]).T, dtype=torch.float64)
    delta = np.full((len(available), worlds), np.nan, dtype=np.float32)
    full = np.full(len(available), np.nan)
    ids = np.flatnonzero(available)
    step = max(1, int(2e7 // max(1, 2 * int(np.prod(val.shape[1:])) + worlds + 1)))
    for lo in range(0, len(ids), step):
        block = ids[lo:lo + step]
        scores = weighted_losses(y, torch.as_tensor(val[block], dtype=torch.float64), weights, metric)
        delta[block] = (scores[:, 1:] - scores[:, :1]).to(torch.float32).numpy()
        full[block] = scores[:, 0].numpy()
    return delta, full


def build(model: str, corpus: str, worlds: int, seed: int = 0):
    cache = core.load(artifacts.scores(corpus, model))
    labels = artifacts.labels(corpus)
    delta = np.full((*cache.val.shape, worlds), np.nan, dtype=np.float32)
    worst = 0.0
    for d, preds in artifacts.predictions(corpus, model):
        val = preds['val']
        available = cache.available[d]
        assert np.array_equal(available, ~np.isnan(val.reshape(len(val), -1)).any(axis=1))
        delta[d], full = dataset_delta(val, labels[(d, 'val')], str(cache.metrics[d]),
                                       available, worlds, seed, core.PREFIX[model], d)
        # The full-sample rebuild must agree with the score cache.
        worst = max(worst, float(np.nanmax(np.abs(full - cache.val[d]) / (2e-5 * np.abs(cache.val[d]) + 2e-7))))
        if d % 200 == 0:
            print(f'{model}: dataset {d}, worst rebuild {worst:.2f}x tolerance', flush=True)
    return delta, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=artifacts.MODELS, required=True)
    parser.add_argument('--corpus', default='synthetic-tabicl-v2')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    delta, worst = build(args.model, args.corpus, args.worlds, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, delta=delta)
    print(f'wrote {args.out}: {delta.shape}, worst full-sample rebuild {worst:.2f}x tolerance')


if __name__ == '__main__':
    main()
