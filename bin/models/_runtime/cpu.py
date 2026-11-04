from __future__ import annotations
import hashlib, json, math, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
PROTOCOL = 'same-for-all-lightgbm'
EXPERIMENT = 'provisional-example'
FEATURE_CAP = 2000
TABPREP_SOURCE_COMMIT = 'fee7f189764070836b59e8cf29207050468589f0'
AUTOGLOUON_TABPREP_SOURCE_COMMIT = '554b5c986fc797acdd60b15eff10bd76f6ae1fed'
OUTPUT_ROOT = Path('.')
def experiment_path(benchmark, dataset, config_id):
    return OUTPUT_ROOT / dataset / str(config_id)

def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()

def terminal_state(path: Path) -> str | None:
    if path.joinpath('DONE').is_file():
        return 'done'
    if path.joinpath('FAILED').is_file():
        return 'failed'
    return None

def _frames(
    dataset_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    parts = ('train', 'val', 'test')
    split_root = dataset_root / 'splits' / 'default'
    split = {part: np.load(split_root / f'{part}.npy') for part in parts}
    y_all = np.load(dataset_root / 'y.npy')
    targets = {part: np.asarray(y_all[split[part]]) for part in parts}
    frames = {part: pd.DataFrame(index=range(len(targets[part]))) for part in parts}
    for key, prefix in (('x_num', 'num'), ('x_bin', 'bin')):
        path = dataset_root / f'{key}.npy'
        if not path.is_file():
            continue
        values = np.load(path)
        for column in range(values.shape[1]):
            name = f'{prefix}_{column:04d}'
            for part in parts:
                frames[part][name] = values[split[part], column]
    cat_path = dataset_root / 'x_cat.npy'
    if cat_path.is_file():
        cat_values = np.load(cat_path)
        train_values = cat_values[split['train']]
        for column in range(train_values.shape[1]):
            name = f'cat_{column:04d}'
            categories = pd.unique(train_values[:, column]).tolist()
            for part in parts:
                frames[part][name] = pd.Categorical(
                    cat_values[split[part], column], categories=categories
                )
    if not all(len(frames[part]) == len(targets[part]) for part in parts):
        raise RuntimeError('Feature/target row mismatch.')
    if any(frame.shape[1] == 0 for frame in frames.values()):
        raise RuntimeError('Dataset has no feature columns.')
    return frames, targets

def _tabprep_frames(
    frames: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    task_type: str,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    from dev.external.tabprep_fee7f189 import TabPrepFeatureGenerator

    hp = config['tabprep']
    target_type = {'binclass': 'binary'}.get(task_type, task_type)
    prep = TabPrepFeatureGenerator(
        target_type=target_type,
        # Axes without an explicit generation-seed HPO retain the paper default.
        random_state=42,
        groupby_kwargs={'max_features': int(hp['groupby_max_feats'])},
        rsfc_kwargs={
            'n_subsets': int(hp['rsfc_n_subsets']),
            'random_state': int(hp['rsfc_generation_seed']),
        },
        arithmetic_kwargs={
            'max_new_feats': int(hp['oafe_max_feats']),
            'random_state': int(hp['oafe_generation_seed']),
        },
        cat_interact_kwargs={
            'max_new_feats': int(hp['cat_combine_max_feats']),
            'random_state': int(hp['cat_combine_generation_seed']),
        },
        oofte_kwargs={'random_state': int(hp['oofte_generation_seed'])},
    )
    output = {
        'train': prep.fit_transform(frames['train'], pd.Series(y['train'])),
        'val': prep.transform(frames['val']),
        'test': prep.transform(frames['test']),
    }
    before_cap = output['train'].shape[1]
    selected = list(output['train'].columns)
    if before_cap > FEATURE_CAP:
        correlations: list[tuple[float, int, Any]] = []
        target = np.asarray(y['train'])
        for index, column in enumerate(selected):
            values = output['train'][column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                numeric = values.cat.codes.to_numpy(dtype=np.float64)
                numeric[numeric < 0] = np.nan
            else:
                numeric = pd.to_numeric(values, errors='coerce').to_numpy(
                    dtype=np.float64
                )
            corr = scipy.stats.spearmanr(numeric, target, nan_policy='omit').statistic
            strength = -math.inf if not np.isfinite(corr) else abs(float(corr))
            correlations.append((strength, -index, column))
        correlations.sort(reverse=True)
        keep = {column for _, _, column in correlations[:FEATURE_CAP]}
        selected = [column for column in selected if column in keep]
        output = {part: frame.loc[:, selected] for part, frame in output.items()}
    return output, {
        'features_before_tabprep': frames['train'].shape[1],
        'features_after_tabprep_before_cap': before_cap,
        'features_after_cap': output['train'].shape[1],
        'feature_cap_applied': before_cap > FEATURE_CAP,
        'feature_cap': FEATURE_CAP,
        'feature_cap_ranking': 'absolute Spearman on training rows only',
    }

def _score_metrics(
    y: np.ndarray, prediction: np.ndarray, task_type: str
) -> dict[str, Any]:
    if task_type == 'regression':
        return {
            'rmse': float(sklearn.metrics.root_mean_squared_error(y, prediction)),
            'mae': float(sklearn.metrics.mean_absolute_error(y, prediction)),
            'r2': float(sklearn.metrics.r2_score(y, prediction)),
        }
    if task_type == 'binclass':
        probability = np.asarray(prediction).reshape(-1)
        labels = np.rint(probability).astype(np.int64)
        return {
            'accuracy': float(sklearn.metrics.accuracy_score(y, labels)),
            'cross-entropy': float(
                sklearn.metrics.log_loss(y, probability, labels=[0, 1])
            ),
            'roc-auc': float(sklearn.metrics.roc_auc_score(y, probability)),
        }
    probability = np.asarray(prediction)
    labels = np.argmax(probability, axis=1)
    return {
        'accuracy': float(sklearn.metrics.accuracy_score(y, labels)),
        'cross-entropy': float(
            sklearn.metrics.log_loss(
                y, probability, labels=np.arange(probability.shape[1])
            )
        ),
    }

def run_config(
    *,
    benchmark: str,
    dataset: str,
    dataset_root: Path,
    config: dict[str, Any],
    num_cpus: int,
    prediction_parts: tuple[str, ...] = ('train', 'val', 'test'),
) -> dict[str, Any]:
    """Run one config, recording a technical failure without stopping its shard."""
    from autogluon.tabular.models import LGBModel

    allowed_prediction_parts = {'train', 'val', 'test'}
    if (
        not prediction_parts
        or len(prediction_parts) != len(set(prediction_parts))
        or not set(prediction_parts) <= allowed_prediction_parts
        or 'val' not in prediction_parts
    ):
        raise ValueError(
            'prediction_parts must be a unique nonempty subset of '
            "('train', 'val', 'test') containing validation."
        )

    config_id = int(config['config_id'])
    output = experiment_path(benchmark, dataset, config_id)
    state = terminal_state(output)
    if state is not None:
        return {'state': state, 'path': str(output), 'skipped': True}
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath('_RUNNING').write_text('RUNNING\n')
    started = time.perf_counter()
    try:
        info = json.loads(dataset_root.joinpath('info.json').read_text())['task']
        task_type = str(info['type'])
        score_name = str(info['score'])
        frames, targets = _frames(dataset_root)
        prep_report = {
            'features_before_tabprep': frames['train'].shape[1],
            'features_after_tabprep_before_cap': frames['train'].shape[1],
            'features_after_cap': frames['train'].shape[1],
            'feature_cap_applied': False,
            'feature_cap': FEATURE_CAP,
        }
        prep_seconds = 0.0
        if config['tabprep_enabled']:
            prep_started = time.perf_counter()
            frames, prep_report = _tabprep_frames(frames, targets, task_type, config)
            prep_seconds = time.perf_counter() - prep_started
        problem_type = {'binclass': 'binary'}.get(task_type, task_type)
        eval_metric = {
            'regression': 'root_mean_squared_error',
            'binclass': 'roc_auc',
            'multiclass': 'log_loss',
        }[task_type]
        hyperparameters = dict(config['lightgbm'])
        hyperparameters['random_state'] = int(config['model_seed'])
        model = LGBModel(
            path=str(output / 'model-cache'),
            name='LightGBM',
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
        )
        fit_started = time.perf_counter()
        model.fit(
            X=frames['train'],
            y=pd.Series(targets['train']),
            X_val=frames['val'],
            y_val=pd.Series(targets['val']),
            num_cpus=num_cpus,
            num_gpus=0,
            verbosity=0,
        )
        fit_seconds = time.perf_counter() - fit_started
        predictions = {
            part: np.asarray(model.predict_proba(frames[part], num_cpus=num_cpus))
            for part in prediction_parts
        }
        metrics = {
            part: _score_metrics(targets[part], predictions[part], task_type)
            for part in prediction_parts
        }
        for part in metrics:
            metrics[part]['score'] = float(metrics[part][score_name])
        np.savez_compressed(output / 'predictions.npz', **predictions)
        report = {
            'protocol': PROTOCOL,
            'experiment': EXPERIMENT,
            'benchmark': benchmark,
            'dataset': dataset,
            'dataset_root': str(dataset_root),
            'split': 'default',
            'config_id': config_id,
            'config': config,
            'task_type': task_type,
            'score_name': score_name,
            'score': metrics['val']['score'],
            'metrics': metrics,
            'tabprep_source_commit': TABPREP_SOURCE_COMMIT,
            'autogluon_tabprep_source_commit': (AUTOGLOUON_TABPREP_SOURCE_COMMIT),
            'tabprep_fit_on': 'training split only',
            'validation_test_policy': 'unseen by TabPrep fit',
            'trainer': 'AutoGluon LGBModel official defaults and callbacks',
            'num_cpus': num_cpus,
            'prediction_parts': list(prediction_parts),
            'preprocessing_seconds': prep_seconds,
            'fit_seconds': fit_seconds,
            'time_seconds': time.perf_counter() - started,
            **prep_report,
        }
        _atomic_json(output / 'report.json', report)
        _atomic_json(
            output / 'status.json',
            {
                'status': 'done',
                'score': metrics['val']['score'],
                'metric': score_name,
                'time_seconds': report['time_seconds'],
            },
        )
        output.joinpath('_RUNNING').unlink(missing_ok=True)
        output.joinpath('FAILED').unlink(missing_ok=True)
        output.joinpath('DONE').touch()
        return {'state': 'done', 'path': str(output), 'report': report}
    except BaseException as error:
        failure = {
            'status': 'failed',
            'protocol': PROTOCOL,
            'benchmark': benchmark,
            'dataset': dataset,
            'config_id': config_id,
            'config': config,
            'error': repr(error),
            'traceback': ''.join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            'time_seconds': time.perf_counter() - started,
        }
        _atomic_json(output / 'status.json', failure)
        output.joinpath('_RUNNING').unlink(missing_ok=True)
        output.joinpath('DONE').unlink(missing_ok=True)
        output.joinpath('FAILED').touch()
        return {'state': 'failed', 'path': str(output), 'failure': failure}
