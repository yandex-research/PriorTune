"""The LightGBM bank (artifacts configs/lgbm-4000.jsonl): TabArena's LightGBM search space,
plus TabPrep settings on the ids divisible by four. lgbm-prep runs those ids with TabPrep,
lgbm runs every id without it."""
from __future__ import annotations

from typing import Any
import numpy as np

N_CONFIGS = 4000
CHOICES = {'oafe_max_feats': [2000, 1000], 'oafe_generation_seed': [42, 84, 168, 336, 672], 'cat_combine_max_feats': [100, 500], 'cat_combine_generation_seed': [42, 84, 168, 336, 672], 'rsfc_n_subsets': [50, 1, 100], 'rsfc_generation_seed': [42, 84, 168, 336, 672], 'oofte_generation_seed': [42, 84, 168, 336, 672], 'groupby_max_feats': [500, 100, 1000]}

def _official_lightgbm_stream(seed: int = 1234, count: int = N_CONFIGS) -> list[dict[str, Any]]:
    """Exact body of TabArena's official generator, without AG imports."""
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
    search_space = ConfigurationSpace(
        space=[
            Float('learning_rate', (5e-3, 1e-1), log=True),
            Float('feature_fraction', (0.4, 1.0)),
            Float('bagging_fraction', (0.7, 1.0)),
            Categorical('bagging_freq', [1]),
            Integer('num_leaves', (2, 200), log=True),
            Integer('min_data_in_leaf', (1, 64), log=True),
            Categorical('extra_trees', [False, True]),
            Integer('min_data_per_group', (2, 100), log=True),
            Float('cat_l2', (5e-3, 2), log=True),
            Float('cat_smooth', (1e-3, 100), log=True),
            Integer('max_cat_to_onehot', (8, 100), log=True),
            Float('lambda_l1', (1e-4, 1.0)),
            Float('lambda_l2', (1e-4, 2.0)),
        ],
        seed=seed,
    )
    sampled = search_space.sample_configuration(count)
    result = []
    for config in sampled:
        result.append(
            {
                key: value.item() if hasattr(value, 'item') else value
                for key, value in dict(config).items()
            }
        )
    return result

def generate():
    rng = np.random.default_rng(2606023842)
    configs = []
    for i, hp in enumerate(_official_lightgbm_stream()):
        enabled = i % 4 == 0
        prep = None
        if enabled:
            prep = {k: v[0] if i == 0 else v[int(rng.integers(0,len(v)))] for k,v in CHOICES.items()}
        configs.append({'config_id': i, 'model_seed': 8*i, 'lightgbm': hp,
                        'tabprep_enabled': enabled, 'tabprep': prep})
    return configs


def extend(n):
    """The published bank followed by new configs up to id n - 1, from a second stream of
    the same search space (ids below 4,000 are unchanged)."""
    configs = generate()
    rng = np.random.default_rng(2606023843)
    for i, hp in enumerate(_official_lightgbm_stream(seed=1235, count=n - N_CONFIGS), N_CONFIGS):
        prep = ({k: v[int(rng.integers(0, len(v)))] for k, v in CHOICES.items()} if i % 4 == 0 else None)
        configs.append({'config_id': i, 'model_seed': 8 * i, 'lightgbm': hp,
                        'tabprep_enabled': prep is not None, 'tabprep': prep})
    return configs


def save_or_check(model, configs, output=None):
    """Write the bank as JSON lines, or compare it with the published one."""
    import json
    import sys
    from pathlib import Path
    lines = ''.join(json.dumps(c, sort_keys=True) + '\n' for c in configs)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(lines)
        print(f'wrote {output}')
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import artifacts
    published = artifacts.path('configs', f'{artifacts.BANK[model]}.jsonl').read_text()
    pairs = [(json.loads(a), json.loads(b)) for a, b in zip(lines.splitlines(), published.splitlines())]
    exact = sum(a == b for a, b in pairs)
    close = sum(_close(a, b) for a, b in pairs)
    # The MLP bank's log-uniform draws go through exp/log, whose last bit varies across
    # numpy builds; the LightGBM bank regenerates exactly.
    print(f'{model}: {exact} of {len(configs)} configs equal the published bank, '
          f'{close} to a relative 1e-12')


def _close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(abs(a), abs(b))
    return a == b
