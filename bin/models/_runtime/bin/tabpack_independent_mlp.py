"""Train independent MLP configurations through TabPack acceleration.

This entry point deliberately does not build a TabPack ensemble.  Each list item in
``members`` is an independent logical experiment with its own seed, optimizer,
minibatch order, dropout stream and validation-selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
from torch import Tensor, nn

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib.data
import lib.experiment
import lib.optim.utils
import lib.util
from bin.core import capture_gpu_stats, maybe_sync, reset_gpu_stats, resolve_device
from bin.tabpack_independent_mlp_lib.model import (
    make_independent_batch_generators,
    make_independent_model_pack,
    native_training_batches,
    remove_independent_dropout_members,
    training_batches,
)
from bin.tabpack_lib import project as tabpack_project
from bin.tabpack_lib import vendor as tabpack_vendor


sys.modules.setdefault('project', tabpack_project)
sys.modules.setdefault('vendor', tabpack_vendor)

import project.nn  # noqa: E402
import project.optim  # noqa: E402
import project.metrics  # noqa: E402
import project.tabpack  # noqa: E402
from lib.types import PredictionType  # noqa: E402


PARTS = ('train', 'val', 'test')
UPSTREAM_TABPACK_COMMIT = '3a47379f063f566f0cb93fa059c0cb038d6606d5'
VENDORED_TABPACK_COMMIT = 'eeb42cdc5280608d95b2aa6b978ef1f396720f9f'


class Config(TypedDict):
    seed: int
    data: dict[str, Any]
    model: dict[str, Any]
    optimizer: dict[str, Any]
    training: dict[str, Any]
    members: list[dict[str, Any]]
    experiment: NotRequired[dict[str, Any]]
    execution: NotRequired[dict[str, Any]]
    device: NotRequired[str]
    output: NotRequired[dict[str, Any]]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_status(exp: Path, status: str, **extra: Any) -> None:
    _write_json(exp / 'status.json', {'status': status, **extra})


def mark_failed(exp: str | Path, err: BaseException) -> None:
    exp = Path(exp)
    if not exp.is_dir():
        return
    exp.joinpath('_RUNNING').unlink(missing_ok=True)
    exp.joinpath('DONE').unlink(missing_ok=True)
    exp.joinpath('FAILED').touch()
    _write_status(
        exp,
        'failed',
        error=repr(err),
        traceback=''.join(
            traceback.format_exception(type(err), err, err.__traceback__, limit=60)
        ),
    )
    lib.experiment.backup(exp)


def _validate_config(config: Config) -> None:
    members = config['members']
    if not members:
        raise ValueError('members must not be empty.')
    ids = [int(member['config_id']) for member in members]
    if len(ids) != len(set(ids)):
        raise ValueError('member config_id values must be unique within a pack.')
    widths = {
        int(member['model']['num_embeddings']['d_embedding']) for member in members
    }
    native_padded = bool(config.get('execution', {}).get('native_padded', False))
    if len(widths) != 1 and not native_padded:
        raise ValueError(
            'Scientific-correctness guard: configs with different d_embedding values '
            'must be put in different packs, otherwise padding changes the standalone '
            'first-layer initialization and scaling.'
        )
    if native_padded and not bool(
        config.get('execution', {}).get('physical_early_stop_removal', False)
    ):
        raise ValueError(
            'native_padded production packs require physical early-stop removal.'
        )
    execution = config.get('execution', {})
    fixed_width = execution.get('fixed_max_d_embedding')
    fixed_depth = execution.get('fixed_max_n_blocks')
    if fixed_width is not None and int(fixed_width) < max(widths):
        raise ValueError('fixed_max_d_embedding is below an active member width.')
    depths = {int(member['model']['n_blocks']) for member in members}
    if fixed_depth is not None and int(fixed_depth) < max(depths):
        raise ValueError('fixed_max_n_blocks is below an active member depth.')
    if (fixed_width is not None or fixed_depth is not None) and not native_padded:
        raise ValueError('A fixed padded envelope requires native_padded=true.')
    model = config['model']
    if model.get('num_embeddings', {}).get('type') != 'CosineEmbeddingsPack':
        raise ValueError(
            'This executor currently implements native CosineEmbeddingsPack.'
        )
    if model.get('activation', 'ReLU') != 'ReLU':
        raise ValueError('The validated native recipe uses ReLU.')
    if config['optimizer'].get('type') != 'MuonAdamWPack':
        raise ValueError('The validated native recipe uses MuonAdamWPack.')
    if bool(config['optimizer'].get('shared_step', False)):
        raise ValueError('shared_step must be false for independent optimizers.')
    training = config['training']
    if int(training['batch_size']) < 1:
        raise ValueError('training.batch_size must be positive.')
    if int(training['max_epochs']) < 1 and int(training['patience']) < 0:
        raise ValueError('At least one stopping condition must be enabled.')


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(
    config: Config,
    model: project.tabpack.ModelPack,
) -> project.optim.MuonAdamWPack:
    members = config['members']
    custom_groups = [
        {
            'params': [block.linear.weight],
            'muon': True,
            'muon_compute_dtype': {
                'bfloat16': torch.bfloat16, 'float32': torch.float32,
            }[config['optimizer'].get('muon_compute_dtype', 'bfloat16')],
            'muon_update_scale': project.tabpack._make_muon_scale(block.linear),
        }
        for block in model.backbone._iter_blocks()
    ]
    groups = lib.optim.utils.make_parameter_groups(
        model,
        project.tabpack._default_zero_weight_decay_condition,
        custom_groups=custom_groups,
    )
    optimizer = project.optim.MuonAdamWPack(
        groups,
        lr=[float(member['optimizer']['lr']) for member in members],
        weight_decay=[float(member['optimizer']['weight_decay']) for member in members],
        muon_lr=[float(member['optimizer']['muon_lr']) for member in members],
        pack_size=len(members),
        shared_step=False,
    )
    return optimizer


def _take(dataset: lib.data.Dataset[Tensor], key: str, part: str, idx: Tensor):
    values = dataset.data.get(key)
    if values is None:
        return None
    part_values = values[part]
    # ``extract_bin_from_num`` may legitimately leave an explicit [N, 0]
    # tensor.  TabPack represents the matching absent branch as ``None``.
    if part_values.ndim >= 2 and part_values.shape[-1] == 0:
        return None
    return part_values[idx]


def _forward(
    model: project.tabpack.ModelPack,
    dataset: lib.data.Dataset[Tensor],
    part: str,
    idx: Tensor,
    *,
    use_amp: bool,
    device: torch.device,
) -> Tensor:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        return (
            model(
                _take(dataset, 'x_num', part, idx),
                _take(dataset, 'x_cat', part, idx),
            )
            .squeeze(-1)
            .float()
        )


def _prediction_type(dataset: lib.data.Dataset) -> PredictionType:
    return PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS


def _native_predictions(
    logits: Tensor,
    dataset: lib.data.Dataset,
    regression_stats: lib.data.RegressionLabelStats | None,
) -> Tensor:
    if dataset.task.is_regression:
        assert regression_stats is not None
        return logits * regression_stats.std + regression_stats.mean
    if dataset.task.is_binclass:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def _predict_once(
    model: project.tabpack.ModelPack,
    dataset: lib.data.Dataset[Tensor],
    part: str,
    *,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
    regression_stats: lib.data.RegressionLabelStats | None,
) -> np.ndarray:
    model.eval()
    values = []
    for idx in torch.arange(dataset.size(part), device=device).split(batch_size):
        values.append(
            _forward(model, dataset, part, idx, use_amp=use_amp, device=device)
        )
    logits = torch.cat(values, dim=1)
    return _native_predictions(logits, dataset, regression_stats).cpu().numpy()


def _predict(
    model: project.tabpack.ModelPack,
    dataset: lib.data.Dataset[Tensor],
    part: str,
    *,
    initial_batch_size: int,
    use_amp: bool,
    device: torch.device,
    regression_stats: lib.data.RegressionLabelStats | None,
) -> tuple[np.ndarray, int]:
    batch_size = initial_batch_size
    while True:
        try:
            return (
                _predict_once(
                    model,
                    dataset,
                    part,
                    batch_size=batch_size,
                    use_amp=use_amp,
                    device=device,
                    regression_stats=regression_stats,
                ),
                batch_size,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)


def _member_metrics(
    raw_dataset: lib.data.Dataset,
    predictions: dict[str, np.ndarray],
    member_index: int,
) -> dict[str, Any]:
    return raw_dataset.task.calculate_metrics(
        {part: values[member_index] for part, values in predictions.items()},
        _prediction_type(raw_dataset),
    )


def _copy_improved_state(
    model: nn.Module,
    best_state: dict[str, Tensor],
    improved: Tensor,
) -> None:
    indices = torch.nonzero(improved, as_tuple=False).squeeze(1).cpu().tolist()
    if not indices:
        return
    for name, value in model.state_dict().items():
        current = value.detach().cpu()
        for index in indices:
            best_state[name][index].copy_(current[index])


def _load_best_state_slices(
    model: nn.Module,
    best_state: dict[str, Tensor],
    pack_indices: Tensor,
) -> None:
    """Load only selected CPU checkpoints without duplicating the full pack on GPU."""
    cpu_indices = pack_indices.detach().cpu()
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    with torch.inference_mode():
        for name, source in best_state.items():
            target = tensors[name]
            if not isinstance(target, project.nn.ParameterPack | project.nn.BufferPack):
                raise RuntimeError(f'Unexpected non-packed state tensor: {name}')
            target.data[pack_indices].copy_(
                source[cpu_indices].to(target.device, non_blocking=False)
            )


def _compact_best_state(
    best_state: dict[str, Tensor], keep_indices: Tensor
) -> dict[str, Tensor]:
    cpu_indices = keep_indices.detach().cpu()
    return {name: value[cpu_indices].clone() for name, value in best_state.items()}


def _validation_scores(
    raw_dataset: lib.data.Dataset,
    predictions: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize metric computation while isolating invalid member predictions."""
    finite = np.isfinite(predictions).reshape(len(predictions), -1).all(axis=1)
    valid = valid_mask & finite
    scores = np.full(len(predictions), -np.inf, dtype=np.float64)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        try:
            metrics = project.metrics.calculate_metrics_pack(
                y_true=raw_dataset.task.labels['val'],
                y_pred=predictions[valid_indices],
                task_type=raw_dataset.task.type_,
                prediction_type=_prediction_type(raw_dataset),
                score=raw_dataset.task.score,
            )
            scores[valid_indices] = np.asarray(metrics['score'], dtype=np.float64)
        except Exception:
            # A pathological label split or member-specific metric error must not
            # turn one bad configuration into a failed physical pack.
            for index in valid_indices:
                try:
                    metrics = raw_dataset.task.calculate_metrics(
                        {'val': predictions[index]}, _prediction_type(raw_dataset)
                    )
                    scores[index] = float(metrics['val']['score'])
                except Exception:
                    scores[index] = -math.inf
    valid &= np.isfinite(scores)
    return scores, valid


def _metrics_by_member(
    raw_dataset: lib.data.Dataset,
    predictions: dict[str, np.ndarray],
    success_mask: np.ndarray,
) -> list[dict[str, Any] | None]:
    result: list[dict[str, Any] | None] = [None] * len(success_mask)
    indices = np.flatnonzero(success_mask)
    if not len(indices):
        return result
    packed: dict[str, dict[str, np.ndarray]] = {}
    for part, values in predictions.items():
        packed[part] = project.metrics.calculate_metrics_pack(
            y_true=raw_dataset.task.labels[part],
            y_pred=values[indices],
            task_type=raw_dataset.task.type_,
            prediction_type=_prediction_type(raw_dataset),
            score=raw_dataset.task.score,
        )
    for packed_index, member_index in enumerate(indices):
        result[int(member_index)] = {
            part: {
                key: float(np.asarray(value)[packed_index])
                for key, value in metrics.items()
            }
            for part, metrics in packed.items()
        }
    return result


def _materialize_member_results(
    exp: Path,
    config: Config,
    raw_dataset: lib.data.Dataset,
    predictions: dict[str, np.ndarray],
    statuses: list[str],
    best_epochs: Tensor,
    best_val_scores: Tensor,
    wall_seconds: float,
) -> list[dict[str, Any]]:
    records = []
    compact = bool(config.get('output', {}).get('compact_member_results', False))
    members_root = exp / 'members'
    if not compact:
        members_root.mkdir(exist_ok=True)
    success_mask = np.asarray(
        [status.startswith('done_') for status in statuses], dtype=bool
    )
    metrics_by_member = _metrics_by_member(raw_dataset, predictions, success_mask)
    for index, member in enumerate(config['members']):
        config_id = int(member['config_id'])
        success = statuses[index].startswith('done_')
        metrics = metrics_by_member[index]
        member_exp = members_root / f'cfg-{config_id:04d}'
        if not compact:
            member_exp.mkdir(exist_ok=True)
        if success and not compact:
            member_predictions = {
                part: np.asarray(values[index], dtype=np.float32)
                for part, values in predictions.items()
            }
            np.savez(member_exp / 'predictions.npz', **member_predictions)
            (member_exp / 'FAILED').unlink(missing_ok=True)
            (member_exp / 'DONE').touch()
        elif not success and not compact:
            (member_exp / 'DONE').unlink(missing_ok=True)
            (member_exp / 'FAILED').touch()
        report = {
            'config_id': config_id,
            'seed': int(member['seed']),
            'member_config': member,
            'status': statuses[index],
            'best_epoch': int(best_epochs[index]),
            'best_val_score': (
                None
                if not math.isfinite(float(best_val_scores[index]))
                else float(best_val_scores[index])
            ),
            'metrics': metrics,
            'prediction_type': _prediction_type(raw_dataset).value,
            'test_labels_accessed_after_selection_only': True,
            'packed_execution_only': True,
            'wall_time_seconds_shared_pack': wall_seconds,
            'prediction_artifact': 'member_predictions.npz',
            'prediction_row': index,
        }
        if not compact:
            _write_json(member_exp / 'report.json', report)
            _write_json(
                member_exp / 'status.json',
                {'status': 'done' if success else 'failed', 'report': report},
            )
        records.append(report)
    return records


def train_one(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    _validate_config(config)
    report = lib.experiment.create_report(train_one, add_gpu_info=True)
    _seed_everything(int(config['seed']))
    torch.set_float32_matmul_precision('highest')
    device = resolve_device(str(config.get('device', 'auto')))
    reset_gpu_stats(device)
    started = time.perf_counter()

    raw_dataset = lib.data.build_dataset(**config['data'])
    if raw_dataset.n_bin_features != 0:
        raise RuntimeError(
            'TabPack data policy must convert extracted binary features to categorical.'
        )
    regression_stats = raw_dataset.try_standardize_labels_()
    dataset = raw_dataset.to_torch(device)
    members = config['members']
    execution = config.get('execution', {})
    native_padded = bool(execution.get('native_padded', False))
    fixed_max_d_embedding = execution.get('fixed_max_d_embedding')
    fixed_max_n_blocks = execution.get('fixed_max_n_blocks')
    pack_invariant_dropout = bool(
        execution.get('pack_invariant_dropout', not native_padded)
    )
    pack_invariant_batches = bool(
        execution.get('pack_invariant_minibatches', not native_padded)
    )
    model = make_independent_model_pack(
        members,
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        d_block=int(config['model']['d_block']),
        activation=str(config['model'].get('activation', 'ReLU')),
        device=device,
        loop_linear=(
            bool(config['model'].get('loop_linear', False))
            or os.environ.get('TABPACK_MLP_DEBUG_LOOP_LINEAR') == '1'
        ),
        native_padded=native_padded,
        pack_invariant_dropout=pack_invariant_dropout,
        max_d_embedding=(
            None if fixed_max_d_embedding is None else int(fixed_max_d_embedding)
        ),
        max_n_blocks=(None if fixed_max_n_blocks is None else int(fixed_max_n_blocks)),
    )
    optimizer = _make_optimizer(config, model)
    loss_fn = project.tabpack._make_loss_fn_pack(dataset.task.type_)
    training = config['training']
    batch_size = int(training['batch_size'])
    patience = int(training['patience'])
    max_epochs = int(training['max_epochs'])
    debug_patience = os.environ.get('TABPACK_MLP_DEBUG_PATIENCE')
    if debug_patience is not None:
        patience = int(debug_patience)
    debug_max_epochs = os.environ.get('TABPACK_MLP_DEBUG_MAX_EPOCHS')
    if debug_max_epochs is not None:
        max_epochs = int(debug_max_epochs)
    if max_epochs < 1:
        max_epochs = 1_000_000_000
    eval_batch_size = int(training.get('eval_batch_size', 32768))
    use_amp = str(training.get('amp_dtype', 'bfloat16')) == 'bfloat16'
    if os.environ.get('TABPACK_MLP_DEBUG_DISABLE_AMP') == '1':
        use_amp = False
    if device.type == 'cpu':
        use_amp = False

    n_members = len(members)
    batch_generators = (
        make_independent_batch_generators(members, device)
        if pack_invariant_batches
        else None
    )
    native_batch_generator = (
        None
        if pack_invariant_batches
        else torch.Generator(device=device).manual_seed(
            int(config['seed']) ^ 0x5A17A9E3
        )
    )
    local_to_global = torch.arange(n_members, dtype=torch.long, device=device)
    stale = torch.zeros(n_members, dtype=torch.long, device=device)
    best_epochs = np.full(n_members, -1, dtype=np.int64)
    best_val_scores = np.full(n_members, -np.inf, dtype=np.float64)
    statuses = ['running'] * n_members
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    n_classes = dataset.task.try_compute_n_classes()
    prediction_tail = (
        (int(n_classes),)
        if dataset.task.is_multiclass and n_classes is not None
        else ()
    )
    prediction_parts = tuple(
        str(part)
        for part in config.get('output', {}).get('prediction_parts', PARTS)
    )
    if (
        not prediction_parts
        or len(prediction_parts) != len(set(prediction_parts))
        or not set(prediction_parts) <= set(PARTS)
        or 'val' not in prediction_parts
        or 'test' not in prediction_parts
    ):
        raise ValueError(
            'output.prediction_parts must be a unique subset of train/val/test '
            'that includes val and test.'
        )
    all_predictions = {
        part: np.full(
            (n_members, dataset.size(part), *prediction_tail),
            np.nan,
            dtype=np.float32,
        )
        for part in prediction_parts
    }
    y_train = project.tabpack._make_Y_train(dataset)
    train_size = dataset.size('train')
    train_seconds = 0.0
    epoch = 0
    debug_fail_config = os.environ.get('TABPACK_MLP_DEBUG_FAIL_CONFIG_ID')
    debug_fail_member_index = (
        None
        if debug_fail_config is None
        else next(
            (
                index
                for index, member in enumerate(members)
                if int(member['config_id']) == int(debug_fail_config)
            ),
            None,
        )
    )
    debug_fail_epoch = int(os.environ.get('TABPACK_MLP_DEBUG_FAIL_EPOCH', '2'))

    while len(local_to_global):
        epoch += 1
        epoch_started = time.perf_counter()
        model.train()
        if pack_invariant_batches:
            assert batch_generators is not None
            batches = training_batches(train_size, batch_size, batch_generators, device)
        else:
            assert native_batch_generator is not None
            batches = native_training_batches(
                train_size, batch_size, len(local_to_global), native_batch_generator
            )
        locally_alive = torch.ones(
            len(local_to_global), dtype=torch.bool, device=device
        )
        for idx in batches:
            outputs = _forward(
                model, dataset, 'train', idx, use_amp=use_amp, device=device
            )
            losses = loss_fn(outputs, y_train[idx])
            if debug_fail_member_index is not None and epoch == debug_fail_epoch:
                matching = torch.nonzero(
                    local_to_global == debug_fail_member_index, as_tuple=False
                ).squeeze(1)
                if len(matching):
                    losses = losses.clone()
                    losses[matching] = torch.nan
            finite = torch.isfinite(losses)
            newly_failed = locally_alive & ~finite
            for local_index in (
                torch.nonzero(newly_failed, as_tuple=False).squeeze(1).tolist()
            ):
                global_index = int(local_to_global[local_index])
                statuses[global_index] = 'failed_nonfinite_training'
            locally_alive &= finite
            if bool(locally_alive.any()):
                optimizer.zero_grad(set_to_none=True)
                losses[locally_alive].sum().backward()
                optimizer.step()
            if not bool(locally_alive.any()):
                break
        maybe_sync(device)
        train_seconds += time.perf_counter() - epoch_started
        if bool(locally_alive.any()):
            val_predictions, eval_batch_size = _predict(
                model,
                dataset,
                'val',
                initial_batch_size=eval_batch_size,
                use_amp=use_amp,
                device=device,
                regression_stats=regression_stats,
            )
            prediction_finite = (
                np.isfinite(val_predictions)
                .reshape(len(val_predictions), -1)
                .all(axis=1)
            )
            scores, valid_scores = _validation_scores(
                raw_dataset,
                val_predictions,
                locally_alive.detach().cpu().numpy(),
            )
            invalid = locally_alive & torch.as_tensor(~valid_scores, device=device)
            for local_index in (
                torch.nonzero(invalid, as_tuple=False).squeeze(1).tolist()
            ):
                global_index = int(local_to_global[local_index])
                statuses[global_index] = (
                    'failed_nonfinite_validation'
                    if not prediction_finite[local_index]
                    else 'failed_invalid_validation_metric'
                )
            locally_alive &= torch.as_tensor(valid_scores, device=device)
        else:
            scores = np.full(len(local_to_global), -np.inf, dtype=np.float64)

        global_indices = local_to_global.detach().cpu().numpy()
        improved_np = locally_alive.detach().cpu().numpy() & (
            scores > best_val_scores[global_indices]
        )
        improved = torch.as_tensor(improved_np, device=device)
        _copy_improved_state(model, best_state, improved)
        improved_global = global_indices[improved_np]
        best_val_scores[improved_global] = scores[improved_np]
        best_epochs[improved_global] = epoch
        stale[improved] = 0
        stale[locally_alive & ~improved] += 1
        stopping = locally_alive & ((stale > patience) | (epoch >= max_epochs))
        for local_index in torch.nonzero(stopping, as_tuple=False).squeeze(1).tolist():
            global_index = int(local_to_global[local_index])
            statuses[global_index] = (
                'done_max_epochs' if epoch >= max_epochs else 'done_early_stop'
            )
        removing = ~locally_alive | stopping
        done_indices = torch.nonzero(stopping, as_tuple=False).squeeze(1)
        if len(done_indices):
            _load_best_state_slices(model, best_state, done_indices)
            with project.nn.module_pack_select(model, done_indices):
                for part in prediction_parts:
                    values, eval_batch_size = _predict(
                        model,
                        dataset,
                        part,
                        initial_batch_size=eval_batch_size,
                        use_amp=use_amp,
                        device=device,
                        regression_stats=regression_stats,
                    )
                    rows = local_to_global[done_indices].detach().cpu().numpy()
                    all_predictions[part][rows] = np.asarray(values, dtype=np.float32)

        keep_indices = torch.nonzero(~removing, as_tuple=False).squeeze(1)
        print(
            f'epoch={epoch} active={len(keep_indices)}/{n_members} '
            f'pack_before={len(local_to_global)} improved={int(improved.sum())} '
            f'stopped={int(stopping.sum())} failed={int((~locally_alive).sum())}',
            flush=True,
        )
        if not len(keep_indices):
            break
        remove_indices = torch.nonzero(removing, as_tuple=False).squeeze(1)
        if len(remove_indices):
            remove_independent_dropout_members(model, remove_indices)
            old_to_new = project.nn.module_pack_remove(model, remove_indices)
            project.optim.optimizer_pack_remove(optimizer, remove_indices, old_to_new)
            best_state = _compact_best_state(best_state, keep_indices)
            stale = stale[keep_indices].clone()
            local_to_global = local_to_global[keep_indices].clone()
            if batch_generators is not None:
                keep_list = keep_indices.detach().cpu().tolist()
                batch_generators = [batch_generators[index] for index in keep_list]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    for member_index, status in enumerate(statuses):
        if status == 'running':
            statuses[member_index] = 'failed_no_finite_checkpoint'
    success_mask_np = np.asarray(
        [status.startswith('done_') for status in statuses], dtype=bool
    )
    for values in all_predictions.values():
        values[~success_mask_np] = np.nan
    np.savez(exp / 'member_predictions.npz', **all_predictions)
    np.save(
        exp / 'prediction_config_ids.npy',
        np.asarray([int(member['config_id']) for member in members], dtype=np.int64),
    )
    wall_seconds = time.perf_counter() - started
    member_reports = _materialize_member_results(
        exp,
        config,
        raw_dataset,
        all_predictions,
        statuses,
        torch.from_numpy(best_epochs),
        torch.from_numpy(best_val_scores),
        wall_seconds,
    )
    _write_json(exp / 'experiments.json', member_reports)

    finite_val_scores = [
        float(value)
        for value in best_val_scores.tolist()
        if math.isfinite(float(value))
    ]
    report.update(
        {
            'score': (
                float(np.mean(finite_val_scores)) if finite_val_scores else -math.inf
            ),
            'score_semantics': 'diagnostic mean member validation score; no ensemble',
            'n_members_requested': n_members,
            'n_members_successful': int(success_mask_np.sum()),
            'n_members_failed': int((~success_mask_np).sum()),
            'member_statuses': statuses,
            'best_epochs': best_epochs.tolist(),
            'best_val_scores': best_val_scores.tolist(),
            'prediction_type': _prediction_type(raw_dataset).value,
            'train_time_seconds': train_seconds,
            'wall_time_seconds': wall_seconds,
            'eval_batch_size': eval_batch_size,
            'gpu_stats': capture_gpu_stats(device),
            'packed_execution_only': True,
            'aggregation': None,
            'independent_initialization': True,
            'independent_minibatch_order': True,
            'pack_invariant_minibatch_streams': pack_invariant_batches,
            'independent_dropout_draws': True,
            'pack_invariant_dropout_streams': pack_invariant_dropout,
            'independent_optimizer_state': True,
            'independent_early_stopping': True,
            'test_labels_accessed_after_selection_only': True,
            'embedding': 'TabPack CosineEmbeddingsPack',
            'native_padded': native_padded,
            'physical_early_stop_removal': True,
            'prediction_config_ids': 'prediction_config_ids.npy',
            'upstream_tabpack_commit_checked': UPSTREAM_TABPACK_COMMIT,
            'vendored_tabpack_commit': VENDORED_TABPACK_COMMIT,
        }
    )
    exp.joinpath('FAILED').unlink(missing_ok=True)
    exp.joinpath('DONE').touch()
    _write_status(
        exp,
        'done',
        n_members_requested=n_members,
        n_members_successful=int(success_mask_np.sum()),
        n_members_failed=int((~success_mask_np).sum()),
        member_statuses=statuses,
        best_epochs=best_epochs.tolist(),
        wall_time_seconds=wall_seconds,
    )
    lib.experiment.finish(exp, report)
    return report


def _main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', metavar='PATH')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    try:
        lib.experiment.run(train_one, None, args.exp, force=args.force)
    except BaseException as err:
        mark_failed(args.exp, err)
        raise


if __name__ == '__main__':
    lib.util.init()
    _main_cli()
