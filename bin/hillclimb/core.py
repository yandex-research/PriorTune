"""Algorithm 1 of the paper under the tuned rule (T), and the T deployment it mimics.

A portfolio S is deployed on a dataset by its validation-best member; its test loss is
compared with that of a random reference subset deployed the same way. The greedy
climb adds, K times, the configuration whose addition wins most often:

    U(c) = sum_i sum_r h(L_i^r(S + c), L_i^r(R_r)),   h(a, b) = 2[a < b] + [a = b]

over synthetic datasets i and R paired draws r: validation bootstrap r is scored
against random reference r (the diagonal, never the R x R cross product). Credits are
integers, so the objective is exact; ties go to the lowest config id.

Losses are oriented so that lower is better everywhere (ROC-AUC is negated). A config
that failed on a dataset is never deployed there; a portfolio with no member left on a
dataset scores the dataset's worst observed test loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

HIGHER_IS_BETTER = ('roc-auc',)
NATIVE = {'regression': 'rmse', 'binclass': 'roc-auc', 'multiclass': 'cross-entropy'}

# Random streams: SeedSequence(seed, spawn_key=(*prefix, index, kind[, extra])).
# lgbm and lgbm-prep share a prefix, so the two pools get identical config orders,
# dataset orders, references and worlds, and their difference is a paired comparison.
PREFIX = {'lgbm': (827,), 'lgbm-prep': (827,), 'mlp': (500, 0)}


def prefix(model: str) -> tuple:
    """The stream prefix of a model; a new model gets one derived from its name."""
    import zlib
    return PREFIX.get(model, (zlib.crc32(model.encode()) % 2**31,))
CONFIG_ORDER, DATASET_ORDER, WORLD, REFERENCE = 0, 1, 2, 3
SENTINEL = np.iinfo(np.int16).max     # world rank of a config that failed


@dataclass
class Cache:
    """One score matrix, datasets x configs, loss-oriented (lower is better)."""
    val: np.ndarray           # float64, NaN where unavailable
    test: np.ndarray          # float64, NaN where unavailable
    available: np.ndarray     # bool
    raw_test: np.ndarray      # native test score, NaN where unavailable
    higher: np.ndarray        # (datasets,) bool: the native metric is higher-is-better
    metrics: np.ndarray
    task_types: np.ndarray
    datasets: np.ndarray

    def rows(self, index) -> 'Cache':
        return Cache(**{k: v[index] for k, v in vars(self).items()})

    def columns(self, index) -> 'Cache':
        return Cache(**{k: v[:, index] if v.ndim == 2 else v for k, v in vars(self).items()})

    @property
    def worst(self) -> np.ndarray:
        """Each dataset's worst observed test loss: the score of an empty portfolio."""
        return np.where(self.available.any(axis=1), np.fmax.reduce(self.test, axis=1), np.inf)


def load(arrays: dict, fold: int = 0) -> Cache:
    """A Cache from the arrays of `artifacts.scores`. A (datasets, folds, configs) cache
    is reduced to one inner fold; the paper's TabArena numbers use fold 0."""
    pick = (lambda a: a[:, fold]) if arrays['validation'].ndim == 3 else (lambda a: a)
    available = pick(arrays['available']).astype(bool)
    metrics = arrays['metrics'].astype(str)
    task_types = (arrays['task_types'].astype(str) if 'task_types' in arrays else
                  np.asarray([{v: k for k, v in NATIVE.items()}[m] for m in metrics]))
    assert all(NATIVE[t] == m for t, m in zip(task_types, metrics)), 'native metrics only'
    higher = np.isin(metrics, HIGHER_IS_BETTER)
    sign = np.where(higher, -1.0, 1.0)[:, None]
    val = np.where(available, pick(arrays['validation']).astype(np.float64), np.nan)
    test = np.where(available, pick(arrays['test']).astype(np.float64), np.nan)
    return Cache(val=sign * val, test=sign * test, available=available, raw_test=test,
                 higher=higher, metrics=metrics, task_types=task_types,
                 datasets=arrays['datasets'].astype(str))


# --- random streams ------------------------------------------------------------------

def rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def shuffles(seed: int, prefix: tuple, n_configs: int, n_datasets: int, count: int):
    """Config and dataset orders of repetitions 0..count-1, as int16 (count, n)."""
    cfg = np.stack([rng(seed, *prefix, s, CONFIG_ORDER).permutation(n_configs)
                    for s in range(count)]).astype(np.int16)
    data = np.stack([rng(seed, *prefix, s, DATASET_ORDER).permutation(n_datasets)
                     for s in range(count)]).astype(np.int16)
    return cfg, data


def references(seed: int, prefix: tuple, config_orders: np.ndarray, k: int, draws: int,
               n_configs: int | None = None) -> np.ndarray:
    """(repetitions, draws, k) random reference subsets. Draw 0 of repetition s is the
    first k configs of its config order; the others are uniform k-subsets."""
    n_configs = n_configs or config_orders.shape[1]
    bank = np.empty((len(config_orders), draws, k), dtype=np.int64)
    bank[:, 0] = config_orders[:, :k]
    for s, order in enumerate(config_orders):
        stream = rng(seed, *prefix, s, REFERENCE, n_configs)
        for j in range(1, draws):
            bank[s, j] = order[stream.choice(n_configs, k, replace=False)]
    return bank


# --- worlds and credit ---------------------------------------------------------------

def world_ranks(val: np.ndarray, delta: np.ndarray | None, available: np.ndarray) -> np.ndarray:
    """(datasets, configs, worlds) int16: each config's validation rank in each world.

    World r's validation loss is val + delta[..., r] (delta = None: the one original
    world). A stable sort makes the rank break exact ties by config id, so the argmin of
    ranks over any subset is the deployment, ties to the lowest id.
    """
    rows, configs = val.shape
    assert configs < SENTINEL
    worlds = 1 if delta is None else delta.shape[-1]
    rank = np.empty((rows, configs, worlds), dtype=np.int16)
    positions = np.arange(configs, dtype=np.int16)[None, :, None]
    for lo in range(0, rows, 64):
        block = slice(lo, lo + 64)
        values = val[block, :, None] if delta is None else val[block, :, None] + delta[block]
        values = np.where(available[block, :, None], values, np.inf)
        np.put_along_axis(rank[block], np.argsort(values, axis=1, kind='stable'), positions, axis=1)
        rank[block][~available[block]] = SENTINEL
    return rank


def paired_credit(ranks: torch.Tensor, refs: torch.Tensor, test: torch.Tensor,
                  worst: torch.Tensor):
    """Doubled half-win credit of every config against the paired references.

    ranks (D, C, Rw), refs (Rb, K), test (D, C) loss, worst (D,). Rw and Rb are equal, or
    one of them is 1 and broadcasts. Returns credit (D, C, R) int16 and the credit of an
    empty portfolio (D, R), with R = max(Rw, Rb).
    """
    n = max(ranks.shape[-1], len(refs))
    assert ranks.shape[-1] in (1, n) and len(refs) in (1, n)
    ranks, refs = ranks.expand(-1, -1, n), refs.expand(n, -1)
    rows = torch.arange(len(test), device=test.device)[:, None, None]
    draw = torch.arange(n, device=test.device)
    values = ranks[rows, refs[None], draw[None, :, None]]                # (D, R, K)
    deployed = refs[draw[None], values.argmin(dim=-1)]                   # (D, R)
    reference = test[torch.arange(len(test), device=test.device)[:, None], deployed]
    reference = torch.where(values.min(dim=-1).values == SENTINEL, worst[:, None], reference)
    credit = (2 * (test[..., None] < reference[:, None]).to(torch.int16)
              + (test[..., None] == reference[:, None]).to(torch.int16))
    empty = 2 * (worst[:, None] < reference).to(torch.int16) + (worst[:, None] == reference).to(torch.int16)
    return ranks, credit, empty


def greedy(ranks: torch.Tensor, credit: torch.Tensor, empty: torch.Tensor, k: int):
    """The exact integer climb over the given candidate columns.

    ranks, credit (D, C, R) and empty (D, R) restricted to the candidates, which must be
    ordered by config id so that argmax (first maximum) breaks ties to the lowest id.
    Returns the add order as column positions (k,) and the total credit of the result.
    """
    d, c, r = ranks.shape
    incumbent_rank = torch.full((d, r), SENTINEL, dtype=ranks.dtype, device=ranks.device)
    incumbent_credit = empty.clone()
    remaining = torch.ones(c, dtype=torch.bool, device=ranks.device)
    order = []
    chunk = max(1, int(2e7 // max(1, d * r)))
    for _ in range(k):
        # Total credit of S + c: the incumbent's, plus what c changes where it deploys.
        objective = torch.full((c,), int(incumbent_credit.sum(dtype=torch.int64)),
                               dtype=torch.int64, device=ranks.device)
        for lo in range(0, c, chunk):
            better = ranks[:, lo:lo + chunk] < incumbent_rank[:, None]
            gain = (credit[:, lo:lo + chunk] - incumbent_credit[:, None]) * better
            objective[lo:lo + chunk] += gain.sum(dim=(0, 2), dtype=torch.int64)
        objective = objective.masked_fill(~remaining, -1)
        pick = int(objective.argmax())
        better = ranks[:, pick] < incumbent_rank
        incumbent_rank = torch.where(better, ranks[:, pick], incumbent_rank)
        incumbent_credit = torch.where(better, credit[:, pick], incumbent_credit)
        remaining[pick] = False
        order.append(pick)
    return np.asarray(order), int(incumbent_credit.sum(dtype=torch.int64))


def climb(ranks, credit, empty, candidates: np.ndarray, datasets: np.ndarray, k: int):
    """Select k of `candidates` (global ids) on `datasets`; returns ids in add order and
    the half-win rate the climb reached on its own objective."""
    members = np.sort(candidates).astype(np.int64)
    device = ranks.device
    cols, rows = torch.as_tensor(members, device=device), torch.as_tensor(
        np.asarray(datasets, dtype=np.int64), device=device)
    order, total = greedy(ranks[rows][:, cols], credit[rows][:, cols], empty[rows], k)
    return members[order], total / (2 * len(datasets) * ranks.shape[-1])


def credit_tensors(cache: Cache, ranks: np.ndarray, refs: np.ndarray, device='cpu'):
    """ranks (D, C, Rw) as built by `world_ranks`, refs (Rb, K) global ids."""
    return paired_credit(torch.as_tensor(ranks, device=device),
                         torch.as_tensor(refs, dtype=torch.int64, device=device),
                         torch.as_tensor(cache.test, device=device),
                         torch.as_tensor(cache.worst, device=device))


# --- deployment and comparison -------------------------------------------------------

def deploy_t(portfolios: np.ndarray, cache: Cache) -> np.ndarray:
    """Native test score (P, D) of T-deploying each (P, K) portfolio on every dataset."""
    portfolios = np.sort(np.atleast_2d(portfolios), axis=-1)
    out = np.empty((len(portfolios), len(cache.datasets)))
    chunk = max(1, int(1e7 // (len(cache.datasets) * portfolios.shape[-1])))
    worst_raw = np.where(cache.higher, -cache.worst, cache.worst)
    rows = np.arange(len(cache.datasets))[:, None]
    for lo in range(0, len(portfolios), chunk):
        p = portfolios[lo:lo + chunk]
        values = np.nan_to_num(cache.val[:, p], nan=np.inf)              # (D, P, K)
        chosen = p[np.arange(len(p))[None], values.argmin(axis=2)]       # (D, P)
        score = cache.raw_test[rows, chosen]
        out[lo:lo + chunk] = np.where(np.isfinite(values.min(axis=2)), score, worst_raw[:, None]).T
    return out


def compare(candidate: np.ndarray, baseline: np.ndarray, higher: np.ndarray) -> np.ndarray:
    """Per-row (win rate with ties as half, mean %, median %) of candidate vs baseline.

    The improvement on a dataset is (candidate - baseline) / max(candidate, baseline),
    sign-flipped for lower-is-better metrics, so it lies in [-1, 1].
    """
    numerator = np.where(higher, candidate - baseline, baseline - candidate)
    denominator = np.maximum(candidate, baseline)
    gain = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    wins, ties = (gain > 0).sum(axis=-1), (gain == 0).sum(axis=-1)
    return np.stack(((wins + 0.5 * ties) / gain.shape[-1], 100 * gain.mean(axis=-1),
                     100 * np.median(gain, axis=-1)), axis=-1)


def summarize(per_repetition: np.ndarray):
    """Mean over repetitions and its 95% Monte Carlo half-width."""
    n = len(per_repetition)
    mean = per_repetition.mean(axis=0)
    ci = 1.96 * per_repetition.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(mean, np.nan)
    return mean, ci
