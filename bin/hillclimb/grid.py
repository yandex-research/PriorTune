#!/usr/bin/env python3
"""Scaling surfaces and the random-K' ladder (Section 5 of the paper), under T.

surfaces  For every repetition s (an independent config order and dataset order), and
          every cell (N datasets, C configs) of the prefix grid, climb on the first C
          configs and first N datasets of repetition s, T-deploy the result on
          TabArena-51 and compare it with repetition s's first K configs. The `base` arm
          is the same climb without resampling: one original world, one reference.
          Writes the paper's surfaces.csv rows (tuned study) and climbed.csv (the
          full-pool cell), plus everything per repetition in an .npz.

ladder    Random-K' (the first K' configs of repetition s) against random-K (its first
          K), T-deployed on TabArena-51: how many random configs match K tuned ones.

    python bin/hillclimb/grid.py surfaces --model mlp --k 25 --worlds worlds/mlp_R16.npz
    python bin/hillclimb/grid.py ladder --model mlp --k 25

The published surfaces use 2,000 repetitions and R = 16: GPU-days per model and K. `--shuffles` and `--full-pool` scale it down.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifacts                                                    # noqa: E402
import core                                                         # noqa: E402

AXIS = (25, 50, 75, 100, 125, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000)
LADDER_EXTRA = (1, 5, 10, 16)


def prefixes(k: int, top: int) -> tuple[int, ...]:
    """K, the axis values above it up to `top`, and `top` itself."""
    grid = (k,) + tuple(v for v in AXIS if k < v <= top)
    return grid if grid[-1] == top else grid + (top,)


def surfaces(synth: core.Cache, real: core.Cache, delta, prefix, k, shuffles, draws=16,
             seed=0, device='cpu', full_pool=False, log=print):
    n_datasets, n_configs = synth.val.shape
    grid_c = (min(n_configs, n_datasets),) if full_pool else prefixes(k, min(n_configs, n_datasets))
    grid_d = (n_datasets,) if full_pool else prefixes(k, n_datasets)
    cfg, data = core.shuffles(seed, prefix, n_configs, n_datasets, shuffles)
    bank = core.references(seed, prefix, cfg, k, draws)
    ranks = {'base': core.world_ranks(synth.val, None, synth.available),
             'val': core.world_ranks(synth.val, None if delta is None else delta[..., :draws],
                                     synth.available)}
    out = {'grid_c': np.asarray(grid_c), 'grid_d': np.asarray(grid_d)}
    for arm in ('base', 'val'):
        orders = np.empty((shuffles, len(grid_d), len(grid_c), k), dtype=np.int16)
        objective = np.empty((shuffles, len(grid_d), len(grid_c)))
        for s in range(shuffles):
            refs = bank[s] if arm == 'val' else bank[s, :1]
            r, credit, empty = core.credit_tensors(synth, ranks[arm], refs, device)
            for j, width in enumerate(grid_c):
                for i, count in enumerate(grid_d):
                    orders[s, i, j], objective[s, i, j] = core.climb(
                        r, credit, empty, cfg[s, :width], data[s, :count], k)
            log(f'{arm} repetition {s + 1}/{shuffles}')
        scores = core.deploy_t(orders.reshape(-1, k), real).reshape(*orders.shape[:3], -1)
        out |= {f'orders_{arm}': orders, f'objective_{arm}': objective, f'scores_{arm}': scores}
    out['reference_scores'] = core.deploy_t(cfg[:, :k], real)
    for arm in ('base', 'val'):
        out[f'metrics_{arm}'] = core.compare(out[f'scores_{arm}'],
                                             out['reference_scores'][:, None, None], real.higher)
    return out


def ladder(real: core.Cache, prefix, k, shuffles, seed=0):
    """(K', win rate, 95% half-width) of random-K' against random-K."""
    n_configs = real.val.shape[1]
    cfg, _ = core.shuffles(seed, prefix, n_configs, 1, shuffles)
    reference = core.deploy_t(cfg[:, :k], real)
    rows = []
    for size in sorted(set(LADDER_EXTRA) | {k} | {v for v in AXIS if v <= n_configs}):
        wr = core.compare(core.deploy_t(cfg[:, :size], real), reference, real.higher)[:, 0]
        mean, ci = core.summarize(wr)
        rows.append((size, mean, ci))
    return rows


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows([repr(float(v)) if isinstance(v, float) else v for v in row] for row in rows)
    print(f'wrote {path} ({len(rows)} rows)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['surfaces', 'ladder'])
    parser.add_argument('--model', required=True, help='lgbm, lgbm-prep, mlp, or your own')
    parser.add_argument('--scores', type=Path, help='synthetic score matrix of your own')
    parser.add_argument('--real-scores', type=Path, help='TabArena score matrix of your own')
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--worlds', type=Path,
                        help='surfaces: worlds file from worlds.py (without it, no resampling)')
    parser.add_argument('--shuffles', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=16)
    parser.add_argument('--full-pool', action='store_true', help='only the full-pool cell')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    prefix = core.prefix(args.model)
    real = core.load(dict(np.load(args.real_scores)) if args.real_scores
                     else artifacts.scores('tabarena-51', args.model))
    tag = f'{args.model}_K{args.k}'
    if args.command == 'ladder':
        rows = [(args.model, args.k, *row) for row in ladder(real, prefix, args.k, args.shuffles, args.seed)]
        write_csv(args.out / f'ladder_{tag}.csv', ('model', 'k', 'kprime', 'pool_winrate', 'pool_ci95'), rows)
        return
    synth = core.load(dict(np.load(args.scores)) if args.scores
                      else artifacts.scores('synthetic-tabicl-v2', args.model))
    delta = np.load(args.worlds)['delta'] if args.worlds else None
    out = surfaces(synth, real, delta, prefix, args.k, args.shuffles, args.draws, args.seed,
                   args.device, args.full_pool)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f'surfaces_{tag}.npz', **out)
    mean, ci = core.summarize(out['metrics_val'][..., 0])
    rows = [('tuned', args.model, args.k, int(n), int(c), mean[i, j], ci[i, j])
            for i, n in enumerate(out['grid_d']) for j, c in enumerate(out['grid_c'])]
    write_csv(args.out / f'surfaces_{tag}.csv', ('study', 'model', 'k', 'n', 'c', 'winrate', 'ci95'), rows)
    write_csv(args.out / f'climbed_{tag}.csv', ('model', 'k', 'winrate', 'ci95'),
              [(args.model, args.k, mean[-1, -1], ci[-1, -1])])


if __name__ == '__main__':
    main()
