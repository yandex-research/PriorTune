#!/usr/bin/env python3
"""The published artifacts: where they live, how to fetch them, how to read them.

Every script in this repository reaches data through this module. The artifacts are a
Hugging Face dataset repository with this layout (formats in README.md, "Artifacts"):

    configs/{lgbm-4000,mlp-4000}.jsonl                  the config banks
    scores/<corpus>/<model>.npz                         score matrices, datasets x configs
    predictions/<corpus>/<model>/part-*.parquet         raw validation and test predictions
    labels/<corpus>.parquet                             validation and test labels
    datasets/tabicl-v2/part-*.parquet                   the 4,000 synthetic datasets with their splits
    splits/{tabarena-51,tabrepo-180}.parquet            split indices of the real corpora (no data)
    portfolios/<model>/{T,TE}_K<k>.json                 the selected subsets

    corpus in synthetic-tabicl-v2, tabarena-51, tabrepo-180
    model  in lgbm, lgbm-prep, mlp  (portfolios also: tabpack)

Row i of every score matrix is `dataset_index` i in labels/, predictions/, datasets/ and
splits/; join on it, not on order. predictions/<corpus>/lgbm/ holds only the TabPrep-off
reruns of the config ids divisible by four: the other ids are identical to lgbm-prep and
are not stored twice. predictions() rebuilds the full lgbm arrays from both folders.

Download (a pattern per part; scores, configs and portfolios are about 1 GB together):

    python bin/artifacts.py download configs scores portfolios
    python bin/artifacts.py download 'predictions/synthetic-tabicl-v2/mlp/*' labels

Files land under $ARTIFACTS (default: artifacts/ at the repository root).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HF_REPO = 'mrsergazinov/PriorTune'
ROOT = Path(os.environ.get('ARTIFACTS', Path(__file__).resolve().parents[1] / 'artifacts'))

MODELS = ('lgbm', 'lgbm-prep', 'mlp')
CORPORA = ('synthetic-tabicl-v2', 'tabarena-51', 'tabrepo-180')
KS = (25, 64, 200)
BANK = {'lgbm': 'lgbm-4000', 'lgbm-prep': 'lgbm-4000', 'mlp': 'mlp-4000', 'tabpack': 'mlp-4000'}
PARTS = ('configs', 'scores', 'portfolios', 'labels', 'predictions', 'datasets', 'splits')


def path(*parts: str) -> Path:
    """A file under the artifact root; says how to fetch it when it is missing."""
    p = ROOT.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f'{p} is missing. Fetch it with: python bin/artifacts.py '
                                f'download {"/".join(parts[:2])}')
    return p


def configs(model: str) -> list[dict]:
    """The model's 4,000 configurations, indexed by config_id.

    lgbm and lgbm-prep share one bank: lgbm-prep runs TabPrep on the ids divisible by
    four (the entries whose `tabprep` is set), lgbm runs every id without it.
    """
    with path('configs', f'{BANK[model]}.jsonl').open() as f:
        bank = [json.loads(line) for line in f if line.strip()]
    assert [c['config_id'] for c in bank] == list(range(len(bank)))
    if model == 'lgbm':
        bank = [dict(c, tabprep=None, tabprep_enabled=False) for c in bank]
    elif model == 'lgbm-prep':
        bank = [dict(c, tabprep_enabled=c['tabprep'] is not None) for c in bank]
    return bank


def scores(corpus: str, model: str) -> dict[str, np.ndarray]:
    """Raw score matrices: `validation`, `test` (datasets x configs, native metric, so
    ROC-AUC is higher-is-better), `available`, `datasets`, `metrics`, `task_types`."""
    with np.load(path('scores', corpus, f'{model}.npz'), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def portfolio(model: str, objective: str, k: int) -> dict:
    """A selected subset: `config_ids`, the greedy `add_order`, resolved `configs`."""
    try:
        p = path('portfolios', model, f'{objective}_K{k}.json')
    except FileNotFoundError as error:
        raise FileNotFoundError(f'{error} If portfolios/{model}/{objective}_K{k}.json is also absent '
                                f'from {HF_REPO}, that subset is not published yet.') from None
    return json.loads(p.read_text())


def _arrays(table, key_columns):
    """Rows of (key..., shape, dtype, values) -> {key: array}."""
    out = {}
    columns = table.to_pydict()
    for i in range(table.num_rows):
        key = tuple(columns[c][i] for c in key_columns)
        out[key] = np.frombuffer(columns['values'][i], dtype=columns['dtype'][i]).reshape(
            columns['shape'][i])
    return out


def labels(corpus: str) -> dict[tuple[int, str], np.ndarray]:
    """{(dataset_index, split): y} with split in val, test."""
    import pyarrow.parquet as pq
    return _arrays(pq.read_table(path('labels', f'{corpus}.parquet')), ('dataset_index', 'split'))


def _prediction_blocks(corpus: str, model: str):
    """Yield (dataset_index, {'val': array, 'test': array}, config_ids) as stored, in dataset order.

    A dataset's rows are split into blocks of configs whose size varies (each block stays under
    64 MB); they are put back in config_id_start order. config_ids is None when the blocks cover
    the ids config_id_start..config_id_stop, else the explicit global ids of the rows.
    """
    import pyarrow.parquet as pq
    folder = path('predictions', corpus, model)
    current, parts = None, {}

    def assemble():
        out, ids = {}, None
        for split in ('val', 'test'):
            blocks = [parts[split][start] for start in sorted(parts[split])]
            out[split] = np.concatenate([values for values, _ in blocks])
            if any(block_ids is not None for _, block_ids in blocks):
                ids = np.concatenate([block_ids for _, block_ids in blocks])
        return out, ids

    for shard in sorted(folder.glob('part-*.parquet')):
        table = pq.read_table(shard)
        columns = table.to_pydict()
        explicit = columns.get('config_ids', [None] * table.num_rows)
        for i in range(table.num_rows):
            index = columns['dataset_index'][i]
            if index != current:
                if current is not None:
                    yield (current, *assemble())
                assert current is None or index > current, 'shards must be in dataset order'
                current, parts = index, {'val': {}, 'test': {}}
            values = np.frombuffer(columns['values'][i], dtype=columns['dtype'][i]).reshape(
                columns['shape'][i])
            ids = None if explicit[i] is None else np.asarray(explicit[i], dtype=np.int64)
            assert ids is None or len(ids) == len(values), 'config_ids must match the block rows'
            parts[columns['split'][i]][columns['config_id_start'][i]] = (values, ids)
    if current is not None:
        yield (current, *assemble())


def predictions(corpus: str, model: str):
    """Yield (dataset_index, {'val': array, 'test': array}) in dataset order, all 4,000 configs.

    Each array is (configs, rows) for regression and binary classification (the
    positive-class probability) and (configs, rows, classes) for multiclass, float32,
    NaN where the config failed. For `lgbm` this needs both predictions/<corpus>/lgbm and
    predictions/<corpus>/lgbm-prep: the lgbm folder stores only the TabPrep-off reruns, at
    their explicit config ids, and every other row is the lgbm-prep row.
    """
    if model != 'lgbm':
        for index, arrays, ids in _prediction_blocks(corpus, model):
            assert ids is None, f'{model}: unexpected explicit config_ids'
            yield index, arrays
        return
    reruns = _prediction_blocks(corpus, 'lgbm')
    for index, arrays in predictions(corpus, 'lgbm-prep'):
        rerun_index, rerun, ids = next(reruns)
        assert rerun_index == index, f'lgbm and lgbm-prep disagree on dataset order at {index}'
        assert ids is not None, 'lgbm blocks must carry config_ids'
        for split in ('val', 'test'):
            assert rerun[split].shape[1:] == arrays[split].shape[1:], (index, split)
            arrays[split] = arrays[split].copy()
            arrays[split][ids] = rerun[split]
        yield index, arrays
    assert next(reruns, None) is None, 'lgbm has datasets that lgbm-prep does not'


def synthetic_datasets():
    """Yield one dict per synthetic dataset (datasets/tabicl-v2) in dataset order: dataset_index,
    dataset, x_num, x_cat (None if no categorical features), y, the train / val / test row indices
    the configurations were fit on, and info (generator seeds and shapes)."""
    import io
    import pyarrow.parquet as pq
    for shard in sorted(path('datasets', 'tabicl-v2').glob('part-*.parquet')):
        columns = pq.read_table(shard).to_pydict()
        for i in range(len(columns['dataset_index'])):
            row = {key: columns[key][i] for key in columns}
            for key in ('x_num', 'x_cat', 'y'):
                row[key] = None if row.get(key) is None else np.load(io.BytesIO(row[key]))
            for key in ('train', 'val', 'test'):
                row[key] = np.asarray(row[key], dtype=np.int64)
            row['info'] = json.loads(row['info']) if isinstance(row.get('info'), str) else row.get('info')
            yield row


def splits(corpus: str) -> dict[tuple[int, str, str], np.ndarray]:
    """{(dataset_index, split, part): row indices} for tabarena-51 and tabrepo-180, with part in
    train, val, test. Only indices are published: fetch the data from OpenML or TabArena by the
    `source_name` and `info` columns (read them with pyarrow from splits/<corpus>.parquet)."""
    import pyarrow.parquet as pq
    columns = pq.read_table(path('splits', f'{corpus}.parquet')).to_pydict()
    return {(columns['dataset_index'][i], columns['split'][i], columns['part'][i]):
            np.asarray(columns['indices'][i], dtype=np.int64) for i in range(len(columns['dataset_index']))}


def download(patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download
    allow = [p if any(ch in p for ch in '*?.') else f'{p}/**' for p in patterns]
    snapshot_download(HF_REPO, repo_type='dataset', local_dir=ROOT, allow_patterns=allow)
    print(f'downloaded {", ".join(patterns)} to {ROOT}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('download', help='fetch parts of the artifact repository')
    get.add_argument('patterns', nargs='+',
                     help=f'a part ({", ".join(PARTS)}), a folder or a glob inside it')
    sub.add_parser('list', help='what is already under $ARTIFACTS')
    args = parser.parse_args()
    if args.command == 'download':
        download(args.patterns)
    else:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():
                print(f'{p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
