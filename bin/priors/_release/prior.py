"""Synthetic prior releases: the source-agnostic core.

This module owns everything between a vendored, unmodified prior and the release
data contract (`lib/data.py`):

* release and dataset specs with deterministic realization,
* per-dataset seeding derived from `(release seed, dataset ID, attempt)`
  and never from worker order or scheduling,
* bounded filtering: the prior is sampled once per attempt and the NanoTabICL
  acceptance check is applied here, with attempt and wall-clock caps
  (the upstream `rand_dataset_filtered` retries forever),
* serialization of accepted datasets into the release data contract
  (`x_num.npy`, `x_cat.npy`, `y.npy`, `info.json`, `splits/default/*.npy`)
  plus a JSONL manifest of realized specs, checksums, and sizes.

Only the "produce candidate tensors for this spec and attempt" step depends on
the prior; it is delegated to `lib.prior_sources.<source>.sample(spec)`
(`ReleaseSpec.source`; the default is the NanoTabICL approximation of the
TabICLv2 prior, whose releases are byte-identical to those generated before
the sources existed). The per-index shape draws (`realize_dataset_spec`) do not
depend on the source, so releases with the same seed and ranges realize the
same `(task, n_samples, n_features, n_classes)` for dataset `i` whatever the
source — the "shape matching" of `bin/priors/tabpfn-v1/DESIGN.md`.
"""

import dataclasses
import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import sys as _sys

# Release layout: bin/priors/_release/ (upstream: lib/prior.py, imported as
# `lib.prior`). The helper modules `lib/` (env, experiment, util, data, types,
# metrics) and `vendor/nanotabicl/` sit next to this file.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor

from vendor.nanotabicl import prior as nanoprior

import prior_sources_init as prior_sources
from lib import util
from lib.data import (
    _SPLIT_DTYPE,
    _X_CAT_INT_DTYPE,
    _X_NUM_DTYPE,
    _Y_CLF_DTYPE,
    _Y_REG_DTYPE,
)
from prior_sources_init import Candidate

# The commit of the vendored NanoTabICL (see vendor/nanotabicl/UPSTREAM.md).
NANOTABICL_COMMIT = prior_sources.get_source('nanotabicl').COMMIT
DEFAULT_SOURCE = prior_sources.DEFAULT_SOURCE

MANIFEST_FILE_NAME = 'manifest.jsonl'
RELEASE_FILE_NAME = 'release.json'

# Stream tags separating the RNG streams derived from (release seed, dataset index).
_SPEC_STREAM = 0
_ATTEMPT_STREAM = 1
_SPLIT_STREAM = 2

_TASK_TYPES = ('regression', 'binclass', 'multiclass')


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReleaseSpec:
    """The full specification of a dataset release.

    The release content is a pure function of this spec: dataset `i` of the
    release depends only on `(seed, i)` and the remaining fields.
    """

    seed: int
    n_datasets: int
    # Weights (not necessarily normalized) of 'regression'/'binclass'/'multiclass'.
    task_types: dict[str, float]
    # Inclusive log-uniform integer ranges.
    n_samples: tuple[int, int]
    n_features: tuple[int, int]
    # Upper bound for categorical cardinalities (see nanoprior.rand_cat_sizes).
    max_cat_size: int = 100
    # Multiclass class count is sampled uniformly from [3, max_classes].
    max_classes: int = 10
    split_fractions: tuple[float, float, float] = (0.64, 0.16, 0.2)
    # Filter limits (per dataset).
    max_attempts: int = 64
    max_seconds_per_dataset: float = 600.0
    regression_score: str = 'rmse'
    binclass_score: str = 'accuracy'
    multiclass_score: str = 'accuracy'
    # The generative mechanism (see lib/prior_sources). Only the shape draws
    # above are shared; everything the prior does with them is the source's.
    source: str = DEFAULT_SOURCE

    def __post_init__(self) -> None:
        assert self.source in prior_sources.SOURCE_NAMES, self.source
        assert self.n_datasets > 0
        assert set(self.task_types) <= set(_TASK_TYPES), self.task_types
        assert sum(self.task_types.values()) > 0
        assert all(v >= 0 for v in self.task_types.values())
        assert 2 <= self.n_samples[0] <= self.n_samples[1]
        assert 1 <= self.n_features[0] <= self.n_features[1]
        assert self.max_classes >= 3
        assert abs(sum(self.split_fractions) - 1.0) < 1e-9, self.split_fractions
        assert self.max_attempts > 0
        assert self.max_seconds_per_dataset > 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    """The realized specification of one dataset of a release."""

    dataset_id: str
    index: int
    task_type: str
    score: str
    n_samples: int
    n_features: int
    # Per-feature categorical sizes: 0 for numerical features.
    x_cat_sizes: tuple[int, ...]
    # 0 for regression, 2 for binclass, >2 for multiclass.
    n_classes: int


def make_release_spec(config: dict[str, Any]) -> ReleaseSpec:
    # TOML writes integral floats as ints; coerce them for the strict parser.
    config = dict(config)
    if 'task_types' in config:
        config['task_types'] = {k: float(v) for k, v in config['task_types'].items()}
    if 'split_fractions' in config:
        config['split_fractions'] = [float(x) for x in config['split_fractions']]
    if 'max_seconds_per_dataset' in config:
        config['max_seconds_per_dataset'] = float(config['max_seconds_per_dataset'])
    return util.dataclass_from_dict(ReleaseSpec, config)


def spec_hash(release: ReleaseSpec) -> str:
    payload = dataclasses.asdict(release)
    if payload['source'] == DEFAULT_SOURCE:
        # Keep the recorded `spec_hash` of pre-existing releases byte-stable.
        del payload['source']
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_release_root(source: str) -> str:
    """The directory under `data/` that holds the releases of a source."""
    return prior_sources.get_release_root(source)


def make_dataset_id(index: int) -> str:
    return f'{index:05d}'


def seed_global_rngs(*entropy: int) -> None:
    """Seed the global NumPy, Torch and Python RNGs used by the vendored priors.

    NumPy and Torch are seeded from the first two words of the seed sequence
    exactly as before; Python's `random` (used by TabPFN and TabICL, not by
    NanoTabICL) takes the third, so NanoTabICL releases are unaffected.
    """
    state = np.random.SeedSequence(list(entropy)).generate_state(3, dtype=np.uint64)
    np.random.seed(int(state[0] % (2**32)))
    torch.manual_seed(int(state[1] % (2**63)))
    random.seed(int(state[2]))


def realize_dataset_spec(release: ReleaseSpec, index: int) -> DatasetSpec:
    """Realize the spec of dataset `index` (shape, task type, cardinalities).

    The realized spec is fixed across filtering attempts, matching the
    upstream semantics where `rand_dataset_filtered` retries the *data* for
    fixed `(x_cat_sizes, y_cat_sizes, n_samples)`.
    """
    seed_global_rngs(release.seed, index, _SPEC_STREAM)

    names, weights = zip(*sorted(release.task_types.items()))
    probabilities = np.asarray(weights, dtype=np.float64)
    task_type = str(
        np.random.choice(list(names), p=probabilities / probabilities.sum())
    )

    n_samples = nanoprior.randlogint(release.n_samples[0], release.n_samples[1] + 1)
    n_features = nanoprior.randlogint(release.n_features[0], release.n_features[1] + 1)
    x_cat_sizes = tuple(nanoprior.rand_cat_sizes(n_features, release.max_cat_size))

    if task_type == 'regression':
        n_classes = 0
        score = release.regression_score
    elif task_type == 'binclass':
        n_classes = 2
        score = release.binclass_score
    elif task_type == 'multiclass':
        n_classes = int(np.random.randint(3, release.max_classes + 1))
        score = release.multiclass_score
    else:
        raise ValueError(f'Unknown {task_type=}')

    return DatasetSpec(
        dataset_id=make_dataset_id(index),
        index=index,
        task_type=task_type,
        score=score,
        n_samples=n_samples,
        n_features=n_features,
        x_cat_sizes=x_cat_sizes,
        n_classes=n_classes,
    )


def acceptance_check(
    tensors: dict[str, torch.Tensor], n_classes: int
) -> tuple[bool, float]:
    """The upstream acceptance check applied to one candidate dataset.

    This mirrors the check inside `rand_dataset_filtered` from
    `vendor/nanotabicl/prior.py` @ ead24e3 line by line (only reformatted).
    It consumes no global RNG state: the forest uses a fixed `random_state`
    and the bootstrap uses a fixed `np.random.default_rng(0)`, so splitting
    the upstream retry loop into per-attempt calls is exact.
    """
    X_np = torch.cat(
        [t.float() for name, t in tensors.items() if name.startswith('x')], dim=-1
    ).numpy()
    y = tensors['y_0']

    if n_classes > 0:  # classification
        y = y.long().squeeze(-1)
        y = F.one_hot(y, num_classes=int(y.max().item() + 1)).float()
        y = y[:, :1] if y.shape[1] == 2 else y  # drop one dimension for binary

    Y_np = y.float().cpu().numpy()

    # random_state=0 doesn't give OOB scores for all samples for some 133 <= n <= 257
    et = ExtraTreesRegressor(
        n_estimators=25,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,
        max_depth=6,
    ).fit(X_np, Y_np[:, 0] if Y_np.shape[1] == 1 else Y_np)

    # compute improvement in MSE over mean prediction baseline per sample
    Yhat = (
        et.oob_prediction_[:, None]
        if len(et.oob_prediction_.shape) == 1
        else et.oob_prediction_
    )
    mask = ~np.isnan(Yhat).any(axis=1)  # keep only valid out-of-bag rows
    imp = (
        (Y_np[mask] - Y_np.mean(axis=0, keepdims=True)) ** 2
        - (Y_np[mask] - Yhat[mask]) ** 2
    ).sum(axis=1)
    idx = np.random.default_rng(0).integers(0, len(imp), size=(200, len(imp)))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))
    return pval < 0.05, pval


def _build_arrays(
    spec: DatasetSpec, candidate: Candidate
) -> None | dict[str, np.ndarray]:
    """Convert prior tensors into contract arrays; None on a structural reject."""
    tensors = candidate.tensors
    if candidate.n_features != spec.n_features:
        return None
    num_idx = [i for i in range(spec.n_features) if i not in candidate.cat_columns]
    cat_idx = sorted(candidate.cat_columns)

    data: dict[str, np.ndarray] = {}
    if num_idx:
        data['x_num'] = np.concatenate(
            [tensors[f'x_{i}'].numpy() for i in num_idx], axis=1
        ).astype(_X_NUM_DTYPE)
    if cat_idx:
        data['x_cat'] = np.concatenate(
            [tensors[f'x_{i}'].numpy() for i in cat_idx], axis=1
        ).astype(_X_CAT_INT_DTYPE)

    y = tensors['y_0'].squeeze(-1).numpy()
    if spec.n_classes == 0:
        data['y'] = y.astype(_Y_REG_DTYPE)
        if data['y'].std() == 0:
            return None  # constant target
    else:
        labels, y = np.unique(y.astype(np.int64), return_inverse=True)
        # Require every spec'd class to actually occur, so that the recorded
        # task type (binclass/multiclass) is faithful to the data.
        if len(labels) != spec.n_classes:
            return None
        data['y'] = y.astype(_Y_CLF_DTYPE)

    if not all(np.isfinite(v).all() for v in data.values()):
        return None
    return data


def _make_split(
    spec: DatasetSpec, y: np.ndarray, fractions: tuple[float, float, float], seed: Any
) -> None | dict[str, np.ndarray]:
    """A deterministic train/val/test split; stratified for classification.

    Returns None on a structural reject (a part is empty or misses classes).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        return None

    if spec.n_classes == 0:
        permutation = rng.permutation(n)
        parts = {
            'train': permutation[:n_train],
            'val': permutation[n_train : n_train + n_val],
            'test': permutation[n_train + n_val :],
        }
    else:
        # Largest-remainder proportional allocation per class.
        buckets: dict[str, list[np.ndarray]] = {'train': [], 'val': [], 'test': []}
        quotas = {'train': n_train, 'val': n_val, 'test': n_test}
        taken = {'train': 0, 'val': 0, 'test': 0}
        class_indices = [
            rng.permutation(np.where(y == c)[0]) for c in range(spec.n_classes)
        ]
        for indices in class_indices:
            n_c = len(indices)
            exact = {part: n_c * quotas[part] / n for part in buckets}
            counts = {part: int(np.floor(exact[part])) for part in buckets}
            remainders = sorted(
                buckets, key=lambda part: exact[part] - counts[part], reverse=True
            )
            for part in remainders:
                if sum(counts.values()) == n_c:
                    break
                counts[part] += 1
            offset = 0
            for part in ('train', 'val', 'test'):
                buckets[part].append(indices[offset : offset + counts[part]])
                taken[part] += counts[part]
                offset += counts[part]
        parts = {
            part: rng.permutation(np.concatenate(indices))
            for part, indices in buckets.items()
        }
        # Structural checks: train must contain every class; every part must
        # be non-degenerate for classification metrics.
        if len(np.unique(y[parts['train']])) != spec.n_classes:
            return None
        if any(len(np.unique(y[parts[part]])) < 2 for part in ('val', 'test')):
            return None

    if any(len(indices) == 0 for indices in parts.values()):
        return None
    return {part: indices.astype(_SPLIT_DTYPE) for part, indices in parts.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_files(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(dataset_dir.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(dataset_dir))] = {
                'sha256': _sha256(path),
                'n_bytes': path.stat().st_size,
            }
    return files


def _serialize_dataset(
    release_dir: Path,
    spec: DatasetSpec,
    data: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Atomically write one dataset directory; the rename is the completion marker."""
    dataset_dir = release_dir / spec.dataset_id
    assert not dataset_dir.exists()
    tmp_dir = Path(tempfile.mkdtemp(dir=release_dir, prefix=f'.tmp-{spec.dataset_id}-'))
    try:
        info = {
            'task': {
                'type': spec.task_type,
                'score': spec.score,
            },
        }
        tmp_dir.joinpath('info.json').write_text(json.dumps(info, indent=4))
        for key, value in data.items():
            np.save(tmp_dir / f'{key}.npy', value)
        split_dir = tmp_dir / 'splits' / 'default'
        split_dir.mkdir(parents=True)
        for part, indices in split.items():
            np.save(split_dir / f'{part}.npy', indices)
        files = _describe_files(tmp_dir)
        tmp_dir.rename(dataset_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return files


def _spec_entry(spec: DatasetSpec, release: ReleaseSpec) -> dict[str, Any]:
    """The manifest fields known before the data exists.

    The categorical structure is the spec's own only for NanoTabICL; other
    sources decide it themselves, and the entry is completed once a candidate
    is accepted (`_describe_features`).
    """
    entry: dict[str, Any] = {
        'dataset_id': spec.dataset_id,
        'index': spec.index,
        'task': spec.task_type,
        'score': spec.score,
        'n_samples': spec.n_samples,
        'n_features': spec.n_features,
    }
    if release.source == DEFAULT_SOURCE:
        entry.update(
            n_num_features=sum(1 for s in spec.x_cat_sizes if s == 0),
            n_cat_features=sum(1 for s in spec.x_cat_sizes if s > 0),
            cat_sizes=[s for s in spec.x_cat_sizes if s > 0],
        )
    entry['n_classes'] = spec.n_classes
    if release.source != DEFAULT_SOURCE:
        entry['source'] = release.source
    return entry


def _describe_features(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """`n_num_features` / `n_cat_features` / `cat_sizes` read off the arrays."""
    x_cat = data.get('x_cat')
    return {
        'n_num_features': 0 if 'x_num' not in data else int(data['x_num'].shape[1]),
        'n_cat_features': 0 if x_cat is None else int(x_cat.shape[1]),
        'cat_sizes': (
            []
            if x_cat is None
            else [len(np.unique(x_cat[:, j])) for j in range(x_cat.shape[1])]
        ),
    }


def generate_dataset(
    release_dir: str | Path, release: ReleaseSpec, index: int
) -> dict[str, Any]:
    """Generate one dataset of a release. Returns its manifest entry.

    The result is a pure function of `(release, index)`: seeds are derived
    from the release seed, the dataset index, and the attempt number.
    """
    release_dir = Path(release_dir)
    source = prior_sources.get_source(release.source)
    spec = realize_dataset_spec(release, index)
    entry = _spec_entry(spec, release)

    start_time = time.perf_counter()
    n_filter_rejects = 0
    n_structural_rejects = 0
    for attempt in range(release.max_attempts):
        if (
            attempt > 0
            and time.perf_counter() - start_time > release.max_seconds_per_dataset
        ):
            entry.update(
                status='failed',
                reason='wall_clock_cap',
                attempts=attempt,
                n_filter_rejects=n_filter_rejects,
                n_structural_rejects=n_structural_rejects,
                seconds=round(time.perf_counter() - start_time, 3),
            )
            return entry

        seed_global_rngs(release.seed, index, _ATTEMPT_STREAM, attempt)
        candidate = source.sample(spec)
        if candidate.reject is not None:
            n_structural_rejects += 1
            continue
        accepted, pval = acceptance_check(candidate.tensors, spec.n_classes)
        if not accepted:
            n_filter_rejects += 1
            continue

        data = _build_arrays(spec, candidate)
        if data is None:
            n_structural_rejects += 1
            continue
        split = _make_split(
            spec,
            data['y'],
            release.split_fractions,
            np.random.SeedSequence([release.seed, index, _SPLIT_STREAM, attempt]),
        )
        if split is None:
            n_structural_rejects += 1
            continue

        files = _serialize_dataset(release_dir, spec, data, split)
        if release.source != DEFAULT_SOURCE:
            entry.update(_describe_features(data))
            entry['prior_hparams'] = candidate.hparams
        entry.update(
            status='ok',
            attempts=attempt + 1,
            pval=pval,
            n_filter_rejects=n_filter_rejects,
            n_structural_rejects=n_structural_rejects,
            seconds=round(time.perf_counter() - start_time, 3),
            files=files,
            n_bytes=sum(f['n_bytes'] for f in files.values()),
        )
        return entry

    entry.update(
        status='failed',
        reason='attempt_cap',
        attempts=release.max_attempts,
        n_filter_rejects=n_filter_rejects,
        n_structural_rejects=n_structural_rejects,
        seconds=round(time.perf_counter() - start_time, 3),
    )
    return entry


def generate_dataset_worker(args: tuple[str, dict[str, Any], int]) -> dict[str, Any]:
    """A top-level adapter for multiprocessing (spawn) workers."""
    release_dir, release_dict, index = args
    torch.set_num_threads(1)
    release = util.dataclass_from_dict(ReleaseSpec, release_dict)
    return generate_dataset(release_dir, release, index)


# ==================================================================================
# Manifest and release summary
# ==================================================================================
def get_manifest_path(release_dir: str | Path) -> Path:
    return Path(release_dir) / MANIFEST_FILE_NAME


def read_manifest(release_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read the JSONL manifest; the last entry per dataset ID wins."""
    manifest_path = get_manifest_path(release_dir)
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries[entry['dataset_id']] = entry
    return entries


def append_manifest(release_dir: str | Path, entry: dict[str, Any]) -> None:
    with open(get_manifest_path(release_dir), 'a') as f:
        f.write(json.dumps(entry) + '\n')


def remove_stale_tmp_dirs(release_dir: str | Path) -> None:
    for path in Path(release_dir).glob('.tmp-*'):
        shutil.rmtree(path, ignore_errors=True)


def reconcile_manifest(
    release_dir: str | Path, release: ReleaseSpec
) -> dict[str, dict[str, Any]]:
    """Recover manifest entries for dataset dirs written without one (crash)."""
    release_dir = Path(release_dir)
    entries = read_manifest(release_dir)
    for index in range(release.n_datasets):
        dataset_id = make_dataset_id(index)
        dataset_dir = release_dir / dataset_id
        if dataset_dir.exists() and entries.get(dataset_id, {}).get('status') != 'ok':
            spec = realize_dataset_spec(release, index)
            files = _describe_files(dataset_dir)
            entry = _spec_entry(spec, release)
            if release.source != DEFAULT_SOURCE:
                # The categorical structure is the prior's; read it off the
                # arrays (the realized hyperparameters are lost with the crash).
                data = {
                    name: np.load(dataset_dir / f'{name}.npy', mmap_mode='r')
                    for name in ('x_num', 'x_cat')
                    if (dataset_dir / f'{name}.npy').is_file()
                }
                entry.update(_describe_features(data))
            entry.update(
                status='ok',
                recovered=True,
                files=files,
                n_bytes=sum(f['n_bytes'] for f in files.values()),
            )
            append_manifest(release_dir, entry)
            entries[dataset_id] = entry
    return entries


def release_hash(entries: dict[str, dict[str, Any]]) -> str:
    """A stable hash over the per-file checksums of all 'ok' datasets."""
    digest = hashlib.sha256()
    for dataset_id in sorted(entries):
        entry = entries[dataset_id]
        if entry.get('status') != 'ok':
            continue
        for name in sorted(entry['files']):
            digest.update(
                f'{dataset_id}/{name}:{entry["files"][name]["sha256"]}\n'.encode()
            )
    return digest.hexdigest()


def write_release_summary(
    release_dir: str | Path, release: ReleaseSpec, name: str
) -> dict[str, Any]:
    """Write release.json summarizing the release; returns the summary."""
    release_dir = Path(release_dir)
    entries = read_manifest(release_dir)
    ok = [e for e in entries.values() if e.get('status') == 'ok']
    failed = [e for e in entries.values() if e.get('status') == 'failed']
    summary = {
        'name': name,
        'spec': dataclasses.asdict(release),
        'spec_hash': spec_hash(release),
        'source': release.source,
        'source_commit': prior_sources.get_source(release.source).COMMIT,
        **(
            {'nanotabicl_commit': NANOTABICL_COMMIT}
            if release.source == DEFAULT_SOURCE
            else {}
        ),
        'release_hash': release_hash(entries),
        'n_datasets_ok': len(ok),
        'n_datasets_failed': len(failed),
        'failed_dataset_ids': sorted(e['dataset_id'] for e in failed),
        'n_bytes_total': sum(e.get('n_bytes', 0) for e in ok),
        'task_counts': {
            task: sum(1 for e in ok if e['task'] == task) for task in _TASK_TYPES
        },
        # Filled in by bin/upload_release.py.
        'archives': [],
    }
    release_dir.joinpath(RELEASE_FILE_NAME).write_text(json.dumps(summary, indent=4))
    return summary


def list_ok_dataset_ids(release_dir: str | Path) -> list[str]:
    entries = read_manifest(release_dir)
    return sorted(
        dataset_id
        for dataset_id, entry in entries.items()
        if entry.get('status') == 'ok'
    )
