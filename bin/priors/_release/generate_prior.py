"""Generate a synthetic prior dataset release into data/<source root>/<release>.

The experiment directory name is the release name; its config.toml is the
release spec (see `prior.ReleaseSpec`). Datasets are written atomically
(one directory per dataset, the rename being the completion marker), so an
interrupted run resumes with `--resume` and skips completed datasets.

Usage:

    python bin/priors/_release/generate_prior.py <exp dir with config.toml> --resume

The release is a pure function of the spec: per-dataset seeds derive from
`(seed, dataset index, attempt)`, never from the worker count or scheduling.
The optional `source` key of the spec selects the generative mechanism
(`lib.prior_sources`; the default is NanoTabICL) and the data root:
`data/tabicl-prior` (default), `data/tabpfn1-prior`, `data/tabicl1-prior`.
"""

import os
import sys
from pathlib import Path

# Release layout: bin/priors/_release/. `prior.py` (upstream `lib.prior`) sits
# next to this file; the prior sources are bin/priors/<prior>/prior_source.py.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _set_thread_limits() -> None:
    # Must run before numpy/torch are imported (transitively via lib).
    for name in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(name, '1')


_set_thread_limits()

import concurrent.futures  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import multiprocessing  # noqa: E402
from typing import NotRequired, TypedDict  # noqa: E402

from loguru import logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib.env  # noqa: E402
import lib.experiment  # noqa: E402
import prior  # noqa: E402  (upstream: lib.prior)
import lib.util  # noqa: E402


class Config(TypedDict):
    seed: int
    n_datasets: int
    task_types: dict[str, float]
    n_samples: list[int]
    n_features: list[int]
    max_cat_size: NotRequired[int]
    max_classes: NotRequired[int]
    split_fractions: NotRequired[list[float]]
    max_attempts: NotRequired[int]
    max_seconds_per_dataset: NotRequired[float]
    regression_score: NotRequired[str]
    binclass_score: NotRequired[str]
    multiclass_score: NotRequired[str]
    source: NotRequired[str]
    # Execution-only options (not part of the release spec).
    n_workers: NotRequired[int]
    data_dir: NotRequired[str]


_EXECUTION_KEYS = ('n_workers', 'data_dir')


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=False)

    release = prior.make_release_spec(
        {k: v for k, v in config.items() if k not in _EXECUTION_KEYS}
    )
    release_name = exp.name
    release_dir = Path(
        config.get(
            'data_dir',
            lib.env.get_data_dir()
            / prior.get_release_root(release.source)
            / release_name,
        )
    )
    release_dir.mkdir(parents=True, exist_ok=True)
    n_workers = config.get('n_workers', 2)

    prior.remove_stale_tmp_dirs(release_dir)
    entries = prior.reconcile_manifest(release_dir, release)
    done = {
        entry['dataset_id']
        for entry in entries.values()
        if entry.get('status') in ('ok', 'failed')
    }
    todo = [
        index
        for index in range(release.n_datasets)
        if prior.make_dataset_id(index) not in done
    ]
    logger.info(
        f'Release "{release_name}": {release.n_datasets} datasets,'
        f' {len(done)} already done, {len(todo)} to generate,'
        f' {n_workers} worker(s)'
    )

    release_dict = dataclasses.asdict(release)
    tasks = [(str(release_dir), release_dict, index) for index in todo]
    n_new_ok = 0
    n_new_failed = 0
    with contextlib.ExitStack() as stack:
        if n_workers <= 1:
            results = map(prior.generate_dataset_worker, tasks)
        else:
            executor = stack.enter_context(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                )
            )
            results = executor.map(prior.generate_dataset_worker, tasks)

        for entry in tqdm(results, total=len(todo), desc='Generating datasets'):
            prior.append_manifest(release_dir, entry)
            if entry['status'] == 'ok':
                n_new_ok += 1
            else:
                n_new_failed += 1
                logger.warning(
                    f'Dataset {entry["dataset_id"]} failed: {entry.get("reason")}'
                    f' (attempts: {entry.get("attempts")})'
                )

    summary = prior.write_release_summary(release_dir, release, release_name)
    report['release_dir'] = str(lib.util.try_get_relative_path(release_dir))
    report['n_new_ok'] = n_new_ok
    report['n_new_failed'] = n_new_failed
    for key in (
        'spec_hash',
        'release_hash',
        'n_datasets_ok',
        'n_datasets_failed',
        'failed_dataset_ids',
        'n_bytes_total',
        'task_counts',
    ):
        report[key] = summary[key]
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init(torch_=False)
    lib.experiment.run_cli(main, resumable=True)
