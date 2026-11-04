"""Design a selection rule on the synthetic score matrices and compare it with PriorTune.

    python examples/new_selection.py --model lgbm-prep --k 25

Two simple rules, each scored like the paper: the win rate on TabArena against random
subsets of the same size (0.5 = no better than random).
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import priortune  # noqa: E402


def top_k_by_mean_rank(synth, k):
    """The K configurations with the best average validation rank."""
    loss = np.where(synth.available, synth.val, np.inf)
    ranks = loss.argsort(axis=1).argsort(axis=1)
    return np.argsort(ranks.mean(axis=0))[:k]


def greedy_best_member(synth, k):
    """Greedy: add the config that most lowers the mean rank of the subset's best member."""
    loss = np.where(synth.available, synth.val, np.inf)
    ranks = loss.argsort(axis=1).argsort(axis=1).astype(float)
    chosen, best = [], np.full(len(ranks), np.inf)
    for _ in range(k):
        gain = np.minimum(best[:, None], ranks).mean(axis=0)
        gain[chosen] = np.inf
        chosen.append(int(gain.argmin()))
        best = np.minimum(best, ranks[:, chosen[-1]])
    return np.asarray(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lgbm-prep', choices=['lgbm', 'lgbm-prep', 'mlp'])
    parser.add_argument('--k', type=int, default=25)
    args = parser.parse_args()

    synth = priortune.scores(args.model)                  # synthetic datasets x configs
    print(f'{args.model}: {synth.val.shape[0]} synthetic datasets x {synth.val.shape[1]} configs')
    rules = {
        'top-K by mean rank': top_k_by_mean_rank(synth, args.k),
        'greedy best member': greedy_best_member(synth, args.k),
        'PriorTune (T)': [c['config_id'] for c in priortune.subset(args.model, 'T', args.k)],
    }
    for name, ids in rules.items():
        mean, ci = priortune.win_rate(args.model, ids)
        print(f'{name:20s} win rate vs random {args.k}: {mean:.3f} ± {ci:.3f}')


if __name__ == '__main__':
    main()
