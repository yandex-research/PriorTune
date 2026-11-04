"""Tuned + ensembled (T+E): greedy ensemble selection and the T+E objective.

A portfolio is deployed by 40 steps of greedy ensemble selection with replacement
(Caruana et al., 2004) on the validation predictions; the blend (members weighted by
how often they were picked) is scored on test. Each step adds the member that most
lowers the validation loss of the blend, ties to the lowest config id. The step scores
are the dataset's native metric in a form that is cheap to update:

    regression  2 a.(s - t y) + |a|^2      the change in the summed squared error
    binary      -ROC-AUC(s + a)            rank-based; the divisor t is irrelevant
    multiclass  mean -log((s_y + a_y) / (s_row + a_row))

where s is the running sum of picked members and t the step.

The T+E objective (Algorithm 1 under T+E) credits a candidate c by the test loss of the
blend fitted on S + c in each validation bootstrap. A candidate leaves the incumbent's
greedy trajectory only at the first step where it would itself be picked; before that
its blend is the incumbent's. So the incumbent's trajectory is run once, every
candidate's option is scored against all of its states, and only the candidates that
fork are refitted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

STEPS = 40
MULTI_FLOOR = 1e-45
KIND = {'rmse': 'rmse', 'roc-auc': 'auc', 'cross-entropy': 'multi'}


@dataclass
class Dataset:
    """One dataset's predictions, reduced to what the blends need.

    `val`/`test` are (configs, rows): the prediction (regression), the positive-class
    probability (binary) or the true-class probability (multiclass, with the row sums
    in `val_row`/`test_row`). Rows of failed configs are zero and never picked.
    """
    kind: str
    available: np.ndarray        # (C,) bool
    y_val: torch.Tensor          # float32 (int64 multiclass)
    y_test: torch.Tensor         # float64 (int64 multiclass)
    val: torch.Tensor            # (C, n_val) float32
    val_row: torch.Tensor | None
    test: torch.Tensor           # (C, n_test) float64
    test_row: torch.Tensor | None
    single: np.ndarray           # (C,) test loss of each config alone, inf where failed
    worst: float                 # the score of an empty portfolio

    def resample(self, rows: torch.Tensor) -> 'Dataset':
        """The same dataset with its validation rows replaced by a bootstrap draw."""
        return replace(self, y_val=self.y_val[rows], val=self.val[:, rows],
                       val_row=None if self.val_row is None else self.val_row[:, rows])


def prepare(val, test, y_val, y_test, metric: str, available, single_loss) -> Dataset:
    """val/test (C, n[, classes]) raw predictions; `single_loss` the score-cache test
    loss of each config (loss-oriented), which is what a one-member blend scores."""
    kind = KIND[metric]
    ok = torch.as_tensor(available)
    multi = kind == 'multi'
    y_val = torch.as_tensor(y_val, dtype=torch.int64 if multi else torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.int64 if multi else torch.float64)
    val = torch.as_tensor(np.nan_to_num(val), dtype=torch.float32)
    test = torch.as_tensor(np.nan_to_num(test), dtype=torch.float32)
    val_row = test_row = None
    if multi:
        val_row, test_row = val.sum(dim=-1), test.to(torch.float64).sum(dim=-1)
        val = val.gather(-1, y_val[None, :, None].expand(len(val), -1, 1))[..., 0]
        test = test.gather(-1, y_test[None, :, None].expand(len(test), -1, 1))[..., 0]
        val_row[~ok] = 0
    val[~ok] = 0
    single = np.where(available, single_loss, np.inf)
    return Dataset(kind, np.asarray(available, dtype=bool), y_val, y_test, val, val_row,
                   test.to(torch.float64), test_row, single, float(single[available].max()))


# --- metrics -------------------------------------------------------------------------

def _avg_rank(x: torch.Tensor) -> torch.Tensor:
    """Ranks 1..n along the last axis, ties averaged (scipy's rankdata 'average')."""
    n = x.shape[-1]
    order = x.argsort(dim=-1)
    xs = x.gather(-1, order)
    starts = torch.ones(xs.shape, dtype=torch.bool, device=x.device)
    starts[..., 1:] = xs[..., 1:] != xs[..., :-1]
    ends = torch.ones(xs.shape, dtype=torch.bool, device=x.device)
    ends[..., :-1] = xs[..., :-1] != xs[..., 1:]
    position = torch.arange(n, device=x.device).expand(xs.shape)
    first = torch.where(starts, position, -1).cummax(dim=-1).values
    last = torch.where(ends, position, n).flip(-1).cummin(dim=-1).values.flip(-1)
    return torch.empty_like(x).scatter_(-1, order, (first + last).to(x.dtype) * 0.5 + 1.0)


def auc(y: torch.Tensor, scores: torch.Tensor, acc) -> torch.Tensor:
    positive = (y > 0.5).nonzero(as_tuple=True)[0]
    n_pos = positive.numel()
    n_neg = y.numel() - n_pos
    total = _avg_rank(scores).index_select(-1, positive).sum(dim=-1, dtype=acc)
    return (total - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _mce(true: torch.Tensor, row: torch.Tensor, acc) -> torch.Tensor:
    return -(true / row).clamp(MULTI_FLOOR, 1.0).log().mean(dim=-1, dtype=acc)


def options(ds: Dataset, state, state_row, members, members_row, step: int) -> torch.Tensor:
    """Validation step score of adding each member to the running sum (lower wins)."""
    if ds.kind == 'rmse':
        return 2.0 * (members @ (state - float(step) * ds.y_val)) + (members * members).sum(dim=-1)
    if ds.kind == 'auc':
        return -auc(ds.y_val, state + members, torch.float32)
    return _mce(state + members, state_row + members_row, torch.float32)


def _argmin(scores: torch.Tensor) -> int:
    scores = torch.where(scores.isnan(), torch.inf, scores)
    return int((scores == scores.min()).nonzero()[0, 0])


# --- deployment ----------------------------------------------------------------------

def caruana(ds: Dataset, members: np.ndarray, steps: int = STEPS, trace: bool = False):
    """Pick counts (len(members),) of greedy ensemble selection over `members`, which
    must be sorted by config id. None when no member ran on this dataset. With `trace`,
    also the running sums before every step and each step's winning score and pick."""
    ok = torch.as_tensor(ds.available[members])
    if not bool(ok.any()):
        return None
    a = ds.val[members]
    b = None if ds.val_row is None else ds.val_row[members]
    state = torch.zeros(a.shape[1], dtype=a.dtype)
    state_row = None if b is None else torch.zeros_like(state)
    counts = np.zeros(len(members))
    states, rows, best, picks = [], [], [], []
    for step in range(1, steps + 1):
        scores = options(ds, state, state_row, a, b, step).masked_fill(~ok, torch.inf)
        pick = _argmin(scores)
        if trace:
            states.append(state); rows.append(state_row)
            best.append(scores[pick]); picks.append(pick)
        counts[pick] += 1
        state = state + a[pick]
        if b is not None:
            state_row = state_row + b[pick]
    if trace:
        return counts, (states, rows, torch.stack(best), np.asarray(picks))
    return counts


def blend_loss(ds: Dataset, members: np.ndarray, counts: np.ndarray, steps: int = STEPS) -> float:
    """Test loss of the blend that picks member i counts[i] times out of `steps`."""
    used = np.flatnonzero(counts)
    weights = torch.as_tensor(counts[used], dtype=torch.float64)[None]
    sums = (weights @ ds.test[members[used]])[0]
    if ds.kind == 'rmse':
        return float(((sums / steps - ds.y_test) ** 2).mean().sqrt())
    if ds.kind == 'auc':
        return float(-auc(ds.y_test, sums, torch.float64))
    return float(_mce(sums, (weights @ ds.test_row[members[used]])[0], torch.float64))


def deploy(ds: Dataset, members: np.ndarray, steps: int = STEPS) -> float:
    """T+E test loss of a portfolio; the empty-portfolio score if none of it ran."""
    members = np.sort(np.asarray(members))
    counts = caruana(ds, members, steps)
    return ds.worst if counts is None else blend_loss(ds, members, counts, steps)


# --- the T+E objective ---------------------------------------------------------------

def credit(loss, bar) -> int | np.ndarray:
    """Doubled half-win credit of loss(es) against the reference losses `bar`."""
    loss = np.asarray(loss, dtype=np.float64)[..., None]
    return (2 * (loss < bar) + (loss == bar)).sum(axis=-1)


def candidate_credit(ds: Dataset, bar, selected: np.ndarray, candidates: np.ndarray,
                     steps: int = STEPS) -> np.ndarray:
    """Credit of S + c against `bar` for every candidate c, on one dataset and world."""
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    bar = np.atleast_1d(np.asarray(bar, dtype=np.float64))
    ok = ds.available[candidates]
    if not len(selected) or not ds.available[selected].any():
        # A candidate blends only with itself: its own test loss.
        return credit(np.where(ok, ds.single[candidates], ds.worst), bar)
    counts, (states, rows, best, picks) = caruana(ds, selected, steps, trace=True)
    out = np.full(len(candidates), credit(blend_loss(ds, selected, counts, steps), bar))
    # Score every candidate's own option against each incumbent state.
    a = ds.val[candidates]
    b = None if ds.val_row is None else ds.val_row[candidates]
    if ds.kind == 'rmse':
        residual = torch.stack(states) - torch.arange(1, steps + 1, dtype=a.dtype)[:, None] * ds.y_val
        own = 2.0 * (a @ residual.T) + (a * a).sum(dim=-1)[:, None]
    else:
        own = torch.stack([options(ds, states[t], rows[t], a, b, t + 1) for t in range(steps)], dim=1)
    own = torch.where(own.isnan(), torch.inf, own).masked_fill(~torch.as_tensor(ok)[:, None], torch.inf)
    best_id = torch.as_tensor(selected[picks])
    ids = torch.as_tensor(candidates)[:, None]
    forks = ((own < best) | ((own == best) & (ids < best_id))).any(dim=1).numpy()
    for i in np.flatnonzero(forks):
        out[i] = credit(deploy(ds, np.append(selected, candidates[i]), steps), bar)
    return out
