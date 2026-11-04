"""PriorTune in a few lines: the selected subsets, the score matrices, and an evaluator.

    import sys; sys.path.insert(0, 'bin')
    import priortune

    configs = priortune.subset('lgbm', rule='T', k=25)        # 25 hyperparameter dicts
    hps = priortune.autogluon('lgbm', rule='TE', k=25)        # for AutoGluon's TabularPredictor
    synth = priortune.scores('mlp')                           # 4,000 datasets x 4,000 configs
    priortune.win_rate('mlp', [3, 14, 15, 92, 653])           # vs random subsets on TabArena
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'hillclimb'))
import artifacts                                                    # noqa: E402

MODELS = ('lgbm', 'lgbm-prep', 'mlp', 'tabpack')
RULES = {'T': 'T', 'TE': 'TE', 'T+E': 'TE'}


def subset(model: str, rule: str = 'T', k: int = 25) -> list[dict]:
    """The K configurations selected for `model` under `rule` ('T' or 'TE'), with their
    hyperparameters, in the order the greedy selection added them."""
    record = artifacts.portfolio(model, RULES[rule], k)
    by_id = {c['config_id']: c for c in record['configs']}
    return [by_id[i] for i in record['add_order']]


def autogluon(model: str, rule: str = 'T', k: int = 25) -> dict:
    """AutoGluon `hyperparameters` that fit the selected LightGBM configurations exactly
    as in the paper (AutoGluon's LightGBM with its default early stopping).

        TabularPredictor(label=...).fit(train, tuning_data=val, hyperparameters=hps)

    `predictor.predict` then deploys T+E (AutoGluon's weighted ensemble). For T, pass
    `model=` the best single model of `predictor.leaderboard()`. LightGBM-TabPrep needs
    TabPrep's feature generation, which `bin/tune.py` runs for you.
    """
    if model != 'lgbm':
        raise ValueError('autogluon() covers lgbm; use bin/tune.py for lgbm-prep, mlp and tabpack')
    return {'GBM': [dict(c['lightgbm'], random_state=c['model_seed'],
                         ag_args={'name_suffix': f'_c{c["config_id"]}'})
                    for c in subset(model, rule, k)]}


def scores(model: str, corpus: str = 'synthetic-tabicl-v2'):
    """A score matrix (datasets x configs), oriented so that lower is better.

    `.val` / `.test` hold validation / test losses (RMSE, log-loss, 1 - ROC-AUC up to
    sign), NaN where a fit failed; `.available` marks the fits that ran.
    """
    import core
    return core.load(artifacts.scores(corpus, 'mlp' if model == 'tabpack' else model))


def win_rate(model: str, config_ids, repetitions: int = 2000, seed: int = 0):
    """Win rate (ties as half) of a subset against random subsets of the same size,
    deployed under T on TabArena, as in the paper; returns (mean, 95% half-width)."""
    import core
    real = scores(model, 'tabarena-51')
    ids = np.asarray(sorted(config_ids), dtype=np.int64)
    orders, _ = core.shuffles(seed, core.prefix('mlp' if model == 'tabpack' else model),
                              real.val.shape[1], 1, repetitions)
    references = core.deploy_t(orders[:, :len(ids)], real)
    candidate = core.deploy_t(ids[None], real)
    per_repetition = core.compare(np.broadcast_to(candidate, references.shape), references,
                                  real.higher)[:, 0]
    mean, ci = core.summarize(per_repetition)
    return float(mean), float(ci)
