"""Generate and save datasets from the frozen TabICL-v2 Stage-3 GraphSCM prior.

This file contains corpus indexing, seed derivation, generation/filtering,
categorical handling, train/validation/test splitting, and NumPy serialization.
Native GraphSCM components are bundled alongside it in tabicl-v2/tabicl/prior/.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

# Hash randomization must be set before interpreter startup. Do not change the
# caller's process settings when importing the batch-size helper from this file.
if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, 'PYTHONHASHSEED': '0'})
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '8'

import numpy as np
import torch
from sklearn.model_selection import train_test_split


TABICL_V2_PRIOR_ROOT = Path(__file__).resolve().parent / 'tabicl-v2'
DEFAULT_COLLECTION = 'tabicl-v2-stage3-independent-200-v1'
DEFAULT_OUTPUT_ROOT = Path('data') / DEFAULT_COLLECTION
DEFAULT_BASE_SEED = 20260804
DEFAULT_N_REGRESSION = 100
DEFAULT_N_BINARY = 70
DEFAULT_N_MULTICLASS = 30
APPEND_500_N_REGRESSION = 250
APPEND_500_N_BINARY = 175
APPEND_500_N_MULTICLASS = 75
APPEND_500_N_DATASETS = (
    APPEND_500_N_REGRESSION + APPEND_500_N_BINARY + APPEND_500_N_MULTICLASS
)
APPEND_1000_N_REGRESSION = 500
APPEND_1000_N_BINARY = 350
APPEND_1000_N_MULTICLASS = 150
APPEND_1000_N_DATASETS = (
    APPEND_1000_N_REGRESSION + APPEND_1000_N_BINARY + APPEND_1000_N_MULTICLASS
)
APPEND_2000_N_REGRESSION = 1000
APPEND_2000_N_BINARY = 700
APPEND_2000_N_MULTICLASS = 300
APPEND_2000_N_DATASETS = (
    APPEND_2000_N_REGRESSION + APPEND_2000_N_BINARY + APPEND_2000_N_MULTICLASS
)
APPEND_4000_N_REGRESSION = 2000
APPEND_4000_N_BINARY = 1400
APPEND_4000_N_MULTICLASS = 600
APPEND_4000_N_DATASETS = (
    APPEND_4000_N_REGRESSION + APPEND_4000_N_BINARY + APPEND_4000_N_MULTICLASS
)
APPEND_8000_N_REGRESSION = 4000
APPEND_8000_N_BINARY = 2800
APPEND_8000_N_MULTICLASS = 1200
APPEND_8000_N_DATASETS = (
    APPEND_8000_N_REGRESSION + APPEND_8000_N_BINARY + APPEND_8000_N_MULTICLASS
)
DEFAULT_MIN_ROWS = 400
DEFAULT_MAX_ROWS = 60_000
DEFAULT_MIN_FEATURES = 1
DEFAULT_MAX_FEATURES = 100
DEFAULT_MIN_MULTICLASS_CLASSES = 3
DEFAULT_MAX_MULTICLASS_CLASSES = 10
SOURCE_PRIOR_COMMIT = '67eab6b0065a17ab90dbf51633c7c88a680480fe'
OFFICIAL_STAGE3_RECIPE = (
    'https://github.com/soda-inria/tabicl/blob/main/scripts/train_v2_clf_stage3.sh'
)

TaskKind = Literal['regression', 'binclass', 'multiclass']


@dataclasses.dataclass(frozen=True)
class Stage3DatasetSpec:
    dataset: str
    index: int
    task_type: TaskKind
    n_rows: int
    n_features: int
    n_classes: int | None
    prior_train_size: int
    seed: int

    @property
    def train_size(self) -> int:
        return evaluation_train_size(self.n_rows)


def stable_seed(*parts: object) -> int:
    payload = '\x1f'.join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**32)


def evaluation_train_size(n_rows: int) -> int:
    """Exact train size produced by the nested sklearn 80/20 then 80/20 split."""
    test_size = math.ceil(0.20 * n_rows)
    train_val_size = n_rows - test_size
    val_size = math.ceil(0.20 * train_val_size)
    return train_val_size - val_size


def stage3_max_features(n_rows: int, max_features: int = 100) -> int:
    """The row-dependent feature cap used with Stage-3 ``seq_len_per_gp``."""
    if n_rows <= 10_240:
        return max_features
    if n_rows <= 20_000:
        return min(80, max_features)
    if n_rows <= 30_000:
        return min(60, max_features)
    if n_rows <= 40_000:
        return min(40, max_features)
    if n_rows <= 50_000:
        return min(30, max_features)
    if n_rows <= 60_000:
        return min(20, max_features)
    if n_rows <= 65_000:
        return min(15, max_features)
    return min(10, max_features)


def make_specs(
    *,
    n_regression: int = DEFAULT_N_REGRESSION,
    n_binary: int = DEFAULT_N_BINARY,
    n_multiclass: int = DEFAULT_N_MULTICLASS,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Sample deterministic specs with the exact Stage-3 shape marginals.

    Stage 3 samples rows log-uniformly.  With ``seq_len_per_gp=True`` it then
    applies a row-dependent maximum-feature cap and samples the number of
    columns continuously followed by ``round``.  Classification composition is
    fixed by the experiment protocol rather than by Stage 3's native 2..10 draw.
    """
    if min_rows < 2 or min_rows > max_rows:
        raise ValueError('Invalid row range.')
    if min_features < 1 or min_features > max_features:
        raise ValueError('Invalid feature range.')
    if min_features > stage3_max_features(max_rows, max_features):
        raise ValueError('Minimum features exceed the Stage-3 cap at max_rows.')
    counts = {
        'regression': n_regression,
        'binclass': n_binary,
        'multiclass': n_multiclass,
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError('Task counts must be non-negative.')

    specs: list[Stage3DatasetSpec] = []
    index = 0
    for task_type, prefix in (
        ('regression', 'reg'),
        ('binclass', 'bin'),
        ('multiclass', 'multi'),
    ):
        for task_index in range(counts[task_type]):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            # This mirrors round(np.random.uniform(min_features, gp_max_features)).
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    return specs


def make_append_only_specs_500(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Extend the frozen 200-dataset collection without changing its prefix.

    Calling :func:`make_specs` with the final per-task counts is *not* an
    extension: it moves the old binary and multiclass datasets to different
    global indices and therefore changes their shapes and GraphSCM seeds.  This
    function keeps all 200 original specs exactly as they were and appends the
    additional 150 regression, 105 binary, and 45 multiclass specs at global
    indices 200..499.
    """
    specs = make_specs(
        base_seed=base_seed,
        min_rows=min_rows,
        max_rows=max_rows,
        min_features=min_features,
        max_features=max_features,
    )
    index = len(specs)
    extensions = (
        (
            'regression',
            'reg',
            DEFAULT_N_REGRESSION,
            APPEND_500_N_REGRESSION,
        ),
        ('binclass', 'bin', DEFAULT_N_BINARY, APPEND_500_N_BINARY),
        (
            'multiclass',
            'multi',
            DEFAULT_N_MULTICLASS,
            APPEND_500_N_MULTICLASS,
        ),
    )
    for task_type, prefix, first_task_index, stop_task_index in extensions:
        for task_index in range(first_task_index, stop_task_index):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    if len(specs) != APPEND_500_N_DATASETS:
        raise RuntimeError(f'Expected 500 append-only specs, got {len(specs)}.')
    return specs


def make_append_only_specs_1000(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Extend the frozen 500-dataset collection without changing its prefix."""
    specs = make_append_only_specs_500(
        base_seed=base_seed,
        min_rows=min_rows,
        max_rows=max_rows,
        min_features=min_features,
        max_features=max_features,
    )
    index = len(specs)
    extensions = (
        (
            'regression',
            'reg',
            APPEND_500_N_REGRESSION,
            APPEND_1000_N_REGRESSION,
        ),
        ('binclass', 'bin', APPEND_500_N_BINARY, APPEND_1000_N_BINARY),
        (
            'multiclass',
            'multi',
            APPEND_500_N_MULTICLASS,
            APPEND_1000_N_MULTICLASS,
        ),
    )
    for task_type, prefix, first_task_index, stop_task_index in extensions:
        for task_index in range(first_task_index, stop_task_index):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    if len(specs) != APPEND_1000_N_DATASETS:
        raise RuntimeError(f'Expected 1000 append-only specs, got {len(specs)}.')
    return specs


def make_append_only_specs_2000(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Extend the frozen 1000-dataset collection without changing its prefix."""
    specs = make_append_only_specs_1000(
        base_seed=base_seed,
        min_rows=min_rows,
        max_rows=max_rows,
        min_features=min_features,
        max_features=max_features,
    )
    index = len(specs)
    extensions = (
        (
            'regression',
            'reg',
            APPEND_1000_N_REGRESSION,
            APPEND_2000_N_REGRESSION,
        ),
        (
            'binclass',
            'bin',
            APPEND_1000_N_BINARY,
            APPEND_2000_N_BINARY,
        ),
        (
            'multiclass',
            'multi',
            APPEND_1000_N_MULTICLASS,
            APPEND_2000_N_MULTICLASS,
        ),
    )
    for task_type, prefix, first_task_index, stop_task_index in extensions:
        for task_index in range(first_task_index, stop_task_index):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    if len(specs) != APPEND_2000_N_DATASETS:
        raise RuntimeError(f'Expected 2000 append-only specs, got {len(specs)}.')
    return specs


def make_append_only_specs_4000(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Extend the frozen 2000-dataset collection without changing its prefix."""
    specs = make_append_only_specs_2000(
        base_seed=base_seed,
        min_rows=min_rows,
        max_rows=max_rows,
        min_features=min_features,
        max_features=max_features,
    )
    index = len(specs)
    extensions = (
        (
            'regression',
            'reg',
            APPEND_2000_N_REGRESSION,
            APPEND_4000_N_REGRESSION,
        ),
        (
            'binclass',
            'bin',
            APPEND_2000_N_BINARY,
            APPEND_4000_N_BINARY,
        ),
        (
            'multiclass',
            'multi',
            APPEND_2000_N_MULTICLASS,
            APPEND_4000_N_MULTICLASS,
        ),
    )
    for task_type, prefix, first_task_index, stop_task_index in extensions:
        for task_index in range(first_task_index, stop_task_index):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    if len(specs) != APPEND_4000_N_DATASETS:
        raise RuntimeError(f'Expected 4000 append-only specs, got {len(specs)}.')
    return specs


def make_append_only_specs_8000(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[Stage3DatasetSpec]:
    """Extend the frozen 4000-dataset collection without changing its prefix."""
    specs = make_append_only_specs_4000(
        base_seed=base_seed,
        min_rows=min_rows,
        max_rows=max_rows,
        min_features=min_features,
        max_features=max_features,
    )
    index = len(specs)
    extensions = (
        (
            'regression',
            'reg',
            APPEND_4000_N_REGRESSION,
            APPEND_8000_N_REGRESSION,
        ),
        (
            'binclass',
            'bin',
            APPEND_4000_N_BINARY,
            APPEND_8000_N_BINARY,
        ),
        (
            'multiclass',
            'multi',
            APPEND_4000_N_MULTICLASS,
            APPEND_8000_N_MULTICLASS,
        ),
    )
    for task_type, prefix, first_task_index, stop_task_index in extensions:
        for task_index in range(first_task_index, stop_task_index):
            rng = np.random.default_rng(stable_seed(base_seed, index, 'spec'))
            n_rows = int(math.exp(rng.uniform(math.log(min_rows), math.log(max_rows))))
            n_rows = min(max(n_rows, min_rows), max_rows)
            feature_cap = stage3_max_features(n_rows, max_features)
            n_features = round(float(rng.uniform(min_features, feature_cap)))
            n_features = min(max(n_features, min_features), feature_cap)
            if task_type == 'regression':
                n_classes = None
            elif task_type == 'binclass':
                n_classes = 2
            else:
                n_classes = int(
                    rng.integers(
                        DEFAULT_MIN_MULTICLASS_CLASSES,
                        DEFAULT_MAX_MULTICLASS_CLASSES + 1,
                    )
                )
            prior_train_fraction = float(rng.uniform(0.79, 0.81))
            specs.append(
                Stage3DatasetSpec(
                    dataset=f'{prefix}-{task_index:04d}',
                    index=index,
                    task_type=task_type,  # type: ignore[arg-type]
                    n_rows=n_rows,
                    n_features=n_features,
                    n_classes=n_classes,
                    prior_train_size=int(n_rows * prior_train_fraction),
                    seed=stable_seed(base_seed, index, 'graph-scm'),
                )
            )
            index += 1
    if len(specs) != APPEND_8000_N_DATASETS:
        raise RuntimeError(f'Expected 8000 append-only specs, got {len(specs)}.')
    return specs


def spec_by_dataset(specs: list[Stage3DatasetSpec]) -> dict[str, Stage3DatasetSpec]:
    return {spec.dataset: spec for spec in specs}


def nn_batch_size_for_train_size(train_size: int) -> int:
    """TabArena NN batch-size rule, applied to the 64% evaluation train split."""
    if train_size < 1_400:
        batch_size = 16
    elif train_size < 2_800:
        batch_size = 32
    elif train_size < 4_500:
        batch_size = 64
    elif train_size < 6_400:
        batch_size = 128
    elif train_size < 32_000:
        batch_size = 256
    elif train_size < 108_000:
        batch_size = 512
    else:
        batch_size = 1024
    if math.ceil(train_size / batch_size) < 50:
        batch_size = int(2 ** math.trunc(math.log2(max(1, train_size // 50))))
        return max(16, batch_size)
    return batch_size


def _load_prior_symbols() -> dict[str, Any]:
    root = str(TABICL_V2_PRIOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tabicl.prior._dataset import should_filter
    from tabicl.prior._reg2cls import outlier_removing, standard_scaling
    from tabicl.prior.graph_lib._base import Context, DatasetProperties
    from tabicl.prior.graph_lib._config import PriorConfig
    from tabicl.prior.graph_lib._dataset import RandomDataset
    from tabicl.prior.graph_lib._properties import sample_categorical_sizes

    return {
        'Context': Context,
        'DatasetProperties': DatasetProperties,
        'PriorConfig': PriorConfig,
        'RandomDataset': RandomDataset,
        'sample_categorical_sizes': sample_categorical_sizes,
        'outlier_removing': outlier_removing,
        'standard_scaling': standard_scaling,
        'should_filter': should_filter,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _remap_categorical_columns(x: np.ndarray) -> np.ndarray:
    result = np.empty(x.shape, dtype=np.int64)
    for column in range(x.shape[1]):
        _, inverse = np.unique(x[:, column], return_inverse=True)
        result[:, column] = inverse
    return result


def _remove_constant_columns(
    x: torch.Tensor, categorical_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.tensor(
        [torch.unique(x[:, j]).numel() > 1 for j in range(x.shape[1])],
        dtype=torch.bool,
        device=x.device,
    )
    return x[:, keep], categorical_mask[keep]


def _internal_classification_sanity(
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    *,
    n_attempts: int = 10,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    def valid(labels: torch.Tensor) -> bool:
        if train_size <= 0 or train_size >= labels.shape[0]:
            return False
        train_classes = set(torch.unique(labels[:train_size]).tolist())
        test_classes = set(torch.unique(labels[train_size:]).tolist())
        return train_classes == test_classes and len(train_classes) >= 2

    if valid(y):
        return x, y
    for _ in range(n_attempts):
        permutation = torch.randperm(y.shape[0], device=y.device)
        if valid(y[permutation]):
            return x[permutation], y[permutation]
    return None


def _generate_attempt(
    spec: Stage3DatasetSpec,
    *,
    device: str,
    attempt: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, dict[str, Any]]:
    symbols = _load_prior_symbols()
    seed = stable_seed(spec.seed, attempt)
    _seed_everything(seed)
    config = symbols['PriorConfig'](
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
    )
    context = symbols['Context'](config=config, device=device, seed=seed)
    cat_sizes = symbols['sample_categorical_sizes'](
        spec.n_features, context, max_cat_size=200
    )
    properties = symbols['DatasetProperties'](
        n_train=spec.n_rows,
        n_test=0,
        cat_sizes={
            'x': cat_sizes,
            'y': [0 if spec.task_type == 'regression' else int(spec.n_classes)],
        },
    )
    dataset = symbols['RandomDataset'](context).sample(properties)
    data = dataset.get_concat_tensors()

    blocks: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    if 'x_cat' in data:
        blocks.append(data['x_cat'])
        mask_parts.append(
            torch.ones(data['x_cat'].shape[1], dtype=torch.bool, device=device)
        )
    if 'x_num' in data:
        blocks.append(data['x_num'])
        mask_parts.append(
            torch.zeros(data['x_num'].shape[1], dtype=torch.bool, device=device)
        )
    if not blocks:
        raise RuntimeError('GraphSCM generated no feature columns.')
    x = torch.cat(blocks, dim=-1).float()
    categorical_mask = torch.cat(mask_parts)
    x = symbols['outlier_removing'](x, threshold=4)
    x = symbols['standard_scaling'](x)
    permutation = torch.randperm(x.shape[1], device=x.device)
    x = x[:, permutation]
    categorical_mask = categorical_mask[permutation]

    if spec.task_type == 'regression':
        y = symbols['outlier_removing'](data['y_num'].float(), threshold=4)
        y = symbols['standard_scaling'](y).reshape(-1)
    else:
        y = data['y_cat'].reshape(-1).long()
        class_permutation = torch.randperm(int(spec.n_classes), device=y.device)
        y = class_permutation[y]

    if torch.isnan(x).any() or torch.isnan(y).any():
        raise RuntimeError('GraphSCM produced NaN values.')
    x, categorical_mask = _remove_constant_columns(x, categorical_mask)
    if x.shape[1] == 0:
        raise RuntimeError('GraphSCM produced only constant columns.')

    if spec.task_type != 'regression':
        if torch.unique(y).numel() != spec.n_classes:
            raise RuntimeError(
                f'Expected {spec.n_classes} classes but generated '
                f'{torch.unique(y).numel()}.'
            )
        checked = _internal_classification_sanity(x, y, spec.prior_train_size)
        if checked is None:
            raise RuntimeError('Dataset failed the Stage-3 class split sanity check.')
        x, y = checked

    if symbols['should_filter'](
        x, y, config, is_classif=spec.task_type != 'regression'
    ):
        raise RuntimeError('Dataset failed the Stage-3 predictability filter.')

    x_np = x.detach().cpu().numpy().astype(np.float32)
    mask_np = categorical_mask.detach().cpu().numpy()
    x_cat = _remap_categorical_columns(x_np[:, mask_np]) if mask_np.any() else None
    x_num = x_np[:, ~mask_np] if (~mask_np).any() else None
    if spec.task_type == 'regression':
        y_np = y.detach().cpu().numpy().astype(np.float32)
    else:
        y_np = y.detach().cpu().numpy().astype(np.int64)
    metadata = {
        'attempt': attempt,
        'generation_seed': seed,
        'requested_n_features': spec.n_features,
        'actual_n_features': int(x.shape[1]),
        'n_num_features': 0 if x_num is None else int(x_num.shape[1]),
        'n_cat_features': 0 if x_cat is None else int(x_cat.shape[1]),
        'categorical_cardinalities': (
            []
            if x_cat is None
            else [int(np.unique(x_cat[:, j]).size) for j in range(x_cat.shape[1])]
        ),
    }
    return x_num, x_cat, y_np, metadata


def _make_split(
    y: np.ndarray, *, task_type: TaskKind, seed: int
) -> dict[str, np.ndarray]:
    indices = np.arange(len(y), dtype=np.int64)
    stratify = None if task_type == 'regression' else y
    train_val, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.20,
        random_state=stable_seed(seed, 'validation'),
        shuffle=True,
        stratify=None if task_type == 'regression' else y[train_val],
    )
    return {
        'train': np.sort(train).astype(np.int64),
        'val': np.sort(val).astype(np.int64),
        'test': np.sort(test).astype(np.int64),
    }


def _validate_split(
    y: np.ndarray, split: dict[str, np.ndarray], task_type: TaskKind
) -> None:
    expected = {'train', 'val', 'test'}
    if set(split) != expected:
        raise ValueError(f'Invalid split keys: {set(split)}')
    joined = np.concatenate([split[name] for name in ('train', 'val', 'test')])
    if not np.array_equal(np.sort(joined), np.arange(len(y))):
        raise ValueError('Evaluation split is not an exact partition.')
    expected_sizes = {
        'train': math.floor(0.64 * len(y)),
        'val': math.ceil(0.16 * len(y)),
        'test': math.ceil(0.20 * len(y)),
    }
    # sklearn rounding can move one row between train/val for some sizes.
    if any(abs(len(split[name]) - expected_sizes[name]) > 1 for name in expected):
        raise ValueError(f'Split does not match 64/16/20: {expected_sizes=}')
    if task_type != 'regression':
        classes = np.unique(y)
        for name, indices in split.items():
            if not np.array_equal(np.unique(y[indices]), classes):
                raise ValueError(f'{name} split is missing a class.')


def generate_dataset(
    output_root: Path,
    spec: Stage3DatasetSpec,
    *,
    device: str = 'cpu',
    force: bool = False,
    max_attempts: int = 100,
) -> dict[str, Any]:
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError(
            'Set PYTHONHASHSEED=0 before generation. The vendored prior iterates '
            'over a set while assigning graph feature groups.'
        )
    dataset_dir = output_root / spec.dataset
    if dataset_dir.exists() and not force:
        return {'dataset': spec.dataset, 'path': str(dataset_dir), 'status': 'exists'}

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            x_num, x_cat, y, metadata = _generate_attempt(
                spec, device=device, attempt=attempt
            )
            split = _make_split(
                y, task_type=spec.task_type, seed=stable_seed(spec.seed, 'split')
            )
            _validate_split(y, split, spec.task_type)
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(
            f'Failed to generate a valid Stage-3 dataset for {spec.dataset} '
            f'after {max_attempts} attempts.'
        ) from last_error

    score = {
        'regression': 'rmse',
        'binclass': 'roc-auc',
        'multiclass': 'cross-entropy',
    }[spec.task_type]
    info = {
        'task': {'type': spec.task_type, 'score': score},
        'synthetic': {
            'collection': DEFAULT_COLLECTION,
            'source': 'TabICL-v2 Stage-3 GraphSCM',
            'source_prior_commit': SOURCE_PRIOR_COMMIT,
            'official_stage3_recipe': OFFICIAL_STAGE3_RECIPE,
            'requested_n_rows': spec.n_rows,
            'requested_n_features': spec.n_features,
            'requested_n_classes': spec.n_classes,
            'prior_train_size': spec.prior_train_size,
            'evaluation_split': [0.64, 0.16, 0.20],
            'base_spec_seed': spec.seed,
            **metadata,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    with tempfile.TemporaryDirectory(dir=output_root) as tmp_name:
        tmp = Path(tmp_name)
        split_dir = tmp / 'splits' / 'default'
        split_dir.mkdir(parents=True)
        (tmp / 'info.json').write_text(
            json.dumps(info, indent=2, sort_keys=True) + '\n'
        )
        if x_num is not None:
            np.save(tmp / 'x_num.npy', x_num)
        if x_cat is not None:
            np.save(tmp / 'x_cat.npy', x_cat)
        np.save(tmp / 'y.npy', y)
        for name, indices in split.items():
            np.save(split_dir / f'{name}.npy', indices)
        tmp.rename(dataset_dir)

    return {
        'dataset': spec.dataset,
        'status': 'generated',
        'task_type': spec.task_type,
        'n_rows': len(y),
        'n_features': int(metadata['actual_n_features']),
        'n_num_features': int(metadata['n_num_features']),
        'n_cat_features': int(metadata['n_cat_features']),
        'batch_size': nn_batch_size_for_train_size(len(split['train'])),
        'attempt': int(metadata['attempt']),
        'path': str(dataset_dir),
    }


def manifest_dict(
    specs: list[Stage3DatasetSpec], *, base_seed: int = DEFAULT_BASE_SEED
) -> dict[str, Any]:
    return {
        'collection': DEFAULT_COLLECTION,
        'source': 'TabICL-v2 Stage-3 GraphSCM',
        'source_prior_commit': SOURCE_PRIOR_COMMIT,
        'official_stage3_recipe': OFFICIAL_STAGE3_RECIPE,
        'base_seed': base_seed,
        'n_datasets': len(specs),
        'task_counts': {
            kind: sum(spec.task_type == kind for spec in specs)
            for kind in ('regression', 'binclass', 'multiclass')
        },
        'shape_protocol': {
            'n_rows': 'int(LogUniform(400, 60000))',
            'n_features': ('round(Uniform(1, stage3_max_features(n_rows, 100)))'),
            'seq_len_per_gp': True,
        },
        'classification_protocol': {
            'binary_n_classes': 2,
            'multiclass_n_classes': 'UniformInt[3, 10]',
        },
        'graph_config': {
            'prior_type': 'graph_scm',
            'filter_unpredictable_graphs': True,
            'filter_unpredictable_datasets': True,
            'graph_noise': False,
            'allow_act_warping': False,
            'min_n_nodes': 2,
            'max_n_nodes': 32,
            'cauchy_dag_offset': 0.0,
            'use_corrected_cat_sizes': False,
        },
        'evaluation_split': [0.64, 0.16, 0.20],
        'categorical_policy': 'preserve native GraphSCM categorical columns',
        'datasets': [
            {
                **dataclasses.asdict(spec),
                # TOML has no null literal; zero unambiguously means regression.
                'n_classes': spec.n_classes if spec.n_classes is not None else 0,
            }
            for spec in specs
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', type=Path,
                        default=Path('local/datasets/tabicl-v2'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.index < APPEND_8000_N_DATASETS:
        parser.error(f'--index must be in [0,{APPEND_8000_N_DATASETS}): the 4,000 published datasets '
                     'and an append-only extension that keeps them unchanged')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    specs = make_append_only_specs_4000() if args.index < 4000 else make_append_only_specs_8000()
    spec = specs[args.index]
    print(json.dumps(generate_dataset(args.output, spec, force=args.force), indent=2))


if __name__ == '__main__':
    main()
