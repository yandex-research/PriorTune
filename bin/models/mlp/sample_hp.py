"""Regenerate the 4,000-config MLP bank (artifacts configs/mlp-4000.jsonl), or extend it.

The keyed RNG, distributions, seeds, and full member construction are defined
here. make_pack_config also supplies the unchanged physical training context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

N_CONFIGS = 4000
CONFIG_SEED = 20260808
PROTOCOL = 'tabpack-independent-cosine-mlp-sfa500-stage3-500x500-v1'
EXPERIMENT = 'provisional-example'
D_EMBEDDINGS = tuple(range(8,33,4))

def stable_seed(*parts: object) -> int:
    payload = '\x1f'.join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**32)

def _rng(config_id: int, name: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(PROTOCOL, CONFIG_SEED, config_id, name))

def _loguniform(config_id: int, name: str, low: float, high: float) -> float:
    return float(math.exp(_rng(config_id, name).uniform(math.log(low), math.log(high))))

def make_member(config_id: int) -> dict[str, Any]:
    if config_id < 0:
        raise ValueError('config_id must be non-negative.')
    dropout_rng = _rng(config_id, 'dropout')
    # This is the official TabPack ``?uniform`` rule: the default value with
    # probability 1/2, otherwise a uniform draw from the stated interval.
    dropout = (
        0.0
        if not bool(dropout_rng.integers(0, 2))
        else float(dropout_rng.uniform(0.0, 0.5))
    )
    return {
        'config_id': config_id,
        'seed': stable_seed(PROTOCOL, CONFIG_SEED, config_id, 'training'),
        'model': {
            'num_embeddings': {
                'd_embedding': int(_rng(config_id, 'd_embedding').choice(D_EMBEDDINGS)),
                'init_scale': _loguniform(config_id, 'init_scale', 1e-2, 1e1),
            },
            'n_blocks': int(_rng(config_id, 'n_blocks').integers(1, 5)),
            'dropout': dropout,
        },
        'optimizer': {
            'lr': _loguniform(config_id, 'lr', 1e-4, 5e-3),
            'weight_decay': _loguniform(config_id, 'weight_decay', 1e-3, 1.0),
            'muon_lr': _loguniform(config_id, 'muon_lr', 1e-3, 1e-1),
        },
    }

def make_members() -> list[dict[str, Any]]:
    return [make_member(config_id) for config_id in range(N_CONFIGS)]

def members_sha256(members: list[dict[str, Any]] | None = None) -> str:
    payload = json.dumps(
        make_members() if members is None else members,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    return hashlib.sha256(payload).hexdigest()

def make_pack_config(
    *,
    dataset: str,
    dataset_path: Path,
    config_ids: list[int],
    batch_size: int,
    physical_pack_id: str,
) -> dict[str, Any]:
    members = [make_member(config_id) for config_id in config_ids]
    return {
        'seed': stable_seed(PROTOCOL, dataset, physical_pack_id, 'pack-runtime'),
        'device': 'auto',
        'data': {
            'path': str(dataset_path),
            'split_id': ['default'],
            'num_policy': 'noisy-quantile',
            'extract_bin_from_num': True,
            'bin_policy': 'convert-to-cat',
            'cat_policy': 'ordinal',
            'cache': False,
        },
        'model': {
            'num_embeddings': {'type': 'CosineEmbeddingsPack'},
            'activation': 'ReLU',
            'd_block': 384,
            'loop_linear': False,
        },
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': False},
        'training': {
            'batch_size': batch_size,
            'eval_batch_size': 32768,
            'max_epochs': 256,
            'patience': 16,
            'patience_stop_condition': 'consecutive_bad_validations > patience',
            'validation_frequency_epochs': 1,
            'amp_dtype': 'bfloat16',
            'individual_member_early_stopping': True,
        },
        'members': members,
        'execution': {
            'native_padded': True,
            'physical_early_stop_removal': True,
            'pack_invariant_dropout': False,
            'pack_invariant_minibatches': False,
        },
        'experiment': {
            'name': EXPERIMENT,
            'protocol': PROTOCOL,
            'dataset': dataset,
            'split': 'default',
            'randomization_mode': 'same-for-all',
            'physical_pack_id': physical_pack_id,
            'config_ids': config_ids,
            'members_sha256': members_sha256(),
            'failed_members_count_against_budget': True,
            'aggregation': 'none',
        },
        'output': {
            'save_predictions': True,
            'save_member_predictions': True,
            'compact_member_results': True,
        },
    }

def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models._sample_hp import save_or_check
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write the bank here instead of checking it')
    parser.add_argument('--extend', type=int, metavar='N',
                        help='write a bank of N configs: the published 4,000 and new ids after them')
    args = parser.parse_args()
    members = make_members() if not args.extend else [make_member(i) for i in range(args.extend)]
    save_or_check('mlp', members, args.output)


if __name__ == '__main__':
    main()
