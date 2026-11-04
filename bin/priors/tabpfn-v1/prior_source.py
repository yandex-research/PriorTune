"""The TabPFN v1 prior (Hollmann et al., ICLR 2023) as a source.

Design: `bin/priors/tabpfn-v1/DESIGN.md`. The vendored prior
(`vendor/tabpfn1`, automl/TabPFN branch `tabpfn_v1` @ bef0cf25) is driven exactly
as `tabpfn/scripts/model_builder.get_model` drives it for the released model —
`prior_type = 'prior_bag'`, `differentiable = True`, `flexible = True`, with the
hyperparameters resolved by `reload_config` and the notebook overrides
(`RESOLVED_HYPERPARAMETERS`) — with these deliberate differences:

* one differentiable-hyperparameter draw per dataset (upstream: one per group of
  `batch_size_per_gp_sample` datasets; same marginal distribution);
* `seq_len`, `num_features_used`, the task type and the class count come from the
  shared per-index dataset spec (shape matching across sources) instead of
  TabPFN's own samplers;
* regression keeps the z-normalized continuous target (upstream never trained on
  regression; its `RegressionNormalized` min-max scaling is an affine map away);
* no zero padding to 100 columns and no `check_is_compatible` (our stratified
  split guarantees every class in train); the `normalize_by_used_features`
  scale `x / (n_features / 100)` is kept;
* columns the `FlexibleCategorical` wrapper converts to categorical are exposed
  as integer codes (`x_cat`) instead of being z-scored into the numeric matrix.

Everything runs on CPU, one dataset per call (`batch_size = 1`).
"""

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

# Release layout: this file lives in bin/priors/<prior>/; the shared core
# (prior.py, prior_sources_init.py) lives in bin/priors/_release/ and the
# vendored upstream prior in ./vendor/. Upstream these were importable as
# `lib.prior_sources.<x>` and `vendor.tabpfn1` from the repository root. The
# vendored package is loaded from ./vendor/ under the name `vendor_tabpfn1` so
# that it does not clash with `_release/vendor/` (NanoTabICL).
_HERE = _Path(__file__).resolve().parent
_RELEASE = str(_HERE.parent / '_release')
if _RELEASE not in _sys.path:
    _sys.path.insert(0, _RELEASE)


def _load_vendored(name: str) -> None:
    if name in _sys.modules:
        return
    path = _HERE / 'vendor'
    spec = _ilu.spec_from_file_location(
        name, path / '__init__.py', submodule_search_locations=[str(path)]
    )
    assert spec is not None and spec.loader is not None, path
    module = _ilu.module_from_spec(spec)
    _sys.modules[name] = module
    spec.loader.exec_module(module)


_load_vendored('vendor_tabpfn1')

import random
from typing import TYPE_CHECKING, Any

import torch

from vendor_tabpfn1 import model_configs
from vendor_tabpfn1.priors import differentiable_prior, flexible_categorical, mlp
from vendor_tabpfn1.utils import (
    normalize_by_used_features_f,
    normalize_data,
    remove_outliers,
    to_ranking_low_mem,
)

from prior_sources_init import Candidate

if TYPE_CHECKING:
    from prior import DatasetSpec

# The commit of the vendored TabPFN (see vendor/tabpfn1/UPSTREAM.md).
COMMIT = 'bef0cf257f3c4c519a2f8e301e0b81a26ae73aff'

# TabPFN's padded feature width. It enters only through `normalize_by_used_features_f`.
NUM_FEATURES = 100
# `DifferentiableHyperparameterList` takes the transformer embedding size; unused
# by the samplers (kept for fidelity with `reload_config`).
EMSIZE = 512
# Index order of `prior_bag_get_batch = (get_batch_gp, get_batch_mlp)` in
# `model_builder.get_model`: the bag draws index 0 for the GP, 1 for the MLP.
PRIOR_NAMES = ('gp', 'mlp')

# `get_prior_config_causal` -> `reload_config` -> `PriorFittingCustomPrior.ipynb`
# (cells 8-10) -> `get_model`, restricted to what the prior reads. Values that the
# per-dataset spec overrides (`num_features_used`, `num_classes`) are omitted.
RESOLVED_HYPERPARAMETERS: dict[str, Any] = {
    # get_general_config / notebook
    'max_num_classes': 10,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,
    'balanced': False,
    'sampling': 'normal',
    'pre_sample_causes': True,
    'mix_activations': False,
    'new_mlp_per_example': True,
    'prior_mlp_scale_weights_sqrt': True,
    'random_feature_rotation': True,
    'rotate_normalized_labels': True,
    'differentiable_hps_as_style': False,
    'verbose': False,
    # get_flexible_categorical_config / notebook
    'categorical_feature_p': 0.2,
    'nan_prob_no_reason': 0.0,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'nan_prob_unknown_reason_reason_prior': 0.5,
    'set_value_to_nan': 0.1,
    'normalize_to_ranking': False,
    'normalize_by_used_features': True,
    'normalize_with_sqrt': False,
    'normalize_ignore_label_too': False,
    # set by get_model when `flexible`
    'normalize_labels': True,
    # `check_is_compatible` is True upstream; it is deliberately not applied here
    # (see the module docstring), so it is not part of the resolved table.
}

# Differentiable hyperparameters the notebook deletes after fixing their values.
DELETED_DIFFERENTIABLE_HYPERPARAMETERS = (
    'output_multiclass_ordered_p',
    'multiclass_type',
    'sampling',
)


def get_differentiable_hyperparameters() -> dict[str, dict[str, Any]]:
    """The `differentiable_hyperparameters` distributions of the released model."""
    distributions = model_configs.get_diff_config()['differentiable_hyperparameters']
    for key in DELETED_DIFFERENTIABLE_HYPERPARAMETERS:
        del distributions[key]
    return distributions


def sample_hyperparameters() -> tuple[dict[str, Any], dict[str, Any]]:
    """One draw of the "meta" hyperparameters, exactly as `DifferentiablePrior`.

    Returns `(passed, indicators)`: `passed` is what `get_batch` receives
    (several entries are still samplers, resolved by `FlexibleCategorical`);
    `indicators` are the numeric meta draws.
    """
    hyperparameters = differentiable_prior.DifferentiableHyperparameterList(
        get_differentiable_hyperparameters(), embedding_dim=EMSIZE, device='cpu'
    )
    passed, indicators = hyperparameters.sample_parameter_object()
    return passed, indicators


def choose_prior(prior_bag_exp_weights_1: float) -> int:
    """`prior_bag.get_batch`'s draw for one model: 0 = GP, 1 = MLP."""
    weights = torch.tensor([1.0, float(prior_bag_exp_weights_1)], dtype=torch.float)
    return int(torch.multinomial(torch.softmax(weights, 0), 1, replacement=True)[0])


def gp_get_batch():
    try:
        from vendor_tabpfn1.priors import fast_gp
    except ImportError as error:
        raise ImportError(
            'The TabPFN GP prior needs gpytorch: run with `uv run --extra tabpfn-prior`'
        ) from error
    return fast_gp.get_batch


class FlexibleCategoricalRecording(flexible_categorical.FlexibleCategorical):
    """`FlexibleCategorical` that records the categorical conversion.

    `forward` runs the identical sequence of upstream calls (same RNG
    consumption, same tensors) up to the padding step, except that (1) the
    integer codes of the columns converted by `MulticlassRank` are returned
    alongside, (2) zero padding and `check_is_compatible` are not applied and
    (3) for regression (`num_classes == 0`) the class assigner and the label
    normalization are skipped so `y` stays the z-normalized continuous target.
    A parity test (dev/tests/test_prior_tabpfn1.py) pins (1) against upstream.
    """

    def forward(  # type: ignore[override]
        self, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        assert batch_size == self.args['batch_size'] == 1
        x, y, _ = self.get_batch(hyperparameters=self.h, **self.args_passed)

        # Upstream NaN injection: every `nan_prob_*` is 0 in the released config,
        # so its guard is false and, thanks to `and` short-circuiting, no RNG is
        # consumed. Nothing to replicate.
        assert (
            self.h['nan_prob_no_reason']
            + self.h['nan_prob_a_reason']
            + self.h['nan_prob_unknown_reason']
            == 0
        )

        # Categorical features (upstream lines 164-171), recording the codes.
        codes: dict[int, torch.Tensor] = {}
        if (
            'categorical_feature_p' in self.h
            and random.random() < self.h['categorical_feature_p']
        ):
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = flexible_categorical.MulticlassRank(
                    num_unique_features, ordered_p=0.3
                )
                if random.random() < p:
                    d = m(x[:, :, col])
                    codes[col] = d[:, 0].clone()
                    x[:, :, col] = d

        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)
        x, y = normalize_data(x), normalize_data(y)

        classification = self.h['num_classes'] > 0
        if classification:
            y = self.class_assigner(y).float()

        if self.h['normalize_by_used_features']:
            x = normalize_by_used_features_f(
                x,
                self.h['num_features_used'],
                self.args['num_features'],
                normalize_with_sqrt=self.h.get('normalize_with_sqrt', False),
            )

        # No zero padding to `args['num_features']`, no `check_is_compatible`.

        if self.h['normalize_labels'] and classification:
            for b in range(y.shape[1]):
                valid_labels = y[:, b] != -100
                if self.h.get('normalize_ignore_label_too', False):
                    valid_labels[:] = True
                y[valid_labels, b] = (
                    (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1))
                    .sum(axis=0)
                    .unsqueeze(0)
                    .float()
                )
                if y[valid_labels, b].numel() != 0 and self.h.get(
                    'rotate_normalized_labels', True
                ):
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(
                        0, num_classes, (1,), device=self.args['device']
                    )
                    y[valid_labels, b] = (
                        y[valid_labels, b] + random_shift
                    ) % num_classes

        return x, y, codes


def build_hyperparameters(
    spec: 'DatasetSpec', passed: dict[str, Any]
) -> dict[str, Any]:
    return {
        **RESOLVED_HYPERPARAMETERS,
        **passed,
        'num_features_used': spec.n_features,
        'num_classes': spec.n_classes,
    }


def make_model(
    spec: 'DatasetSpec',
    hyperparameters: dict[str, Any],
    prior: str,
    *,
    recording: bool = True,
) -> flexible_categorical.FlexibleCategorical:
    """The `FlexibleCategorical` wrapper around the chosen base prior.

    Constructing it resolves the sampler-valued hyperparameters (one RNG draw
    each, as upstream) and, for classification, draws `MulticlassRank`'s own
    class count, which is then overridden by the spec's.
    """
    get_batch = gp_get_batch() if prior == 'gp' else mlp.get_batch
    args = {
        'device': 'cpu',
        'seq_len': spec.n_samples,
        'num_features': NUM_FEATURES,
        'batch_size': 1,
    }
    cls = (
        FlexibleCategoricalRecording
        if recording
        else flexible_categorical.FlexibleCategorical
    )
    model = cls(get_batch, hyperparameters, args)
    if spec.n_classes > 0:
        assert isinstance(model.class_assigner, flexible_categorical.MulticlassRank)
        model.class_assigner.num_classes = spec.n_classes
    return model


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if hasattr(value, 'item'):  # numpy scalars
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    return str(value)


def sample(spec: 'DatasetSpec') -> Candidate:
    passed, indicators = sample_hyperparameters()
    prior = PRIOR_NAMES[choose_prior(passed['prior_bag_exp_weights_1'])]
    model = make_model(spec, build_hyperparameters(spec, passed), prior)
    x, y, codes = model(batch_size=1)
    assert x.shape == (spec.n_samples, 1, spec.n_features), x.shape
    assert y.shape == (spec.n_samples, 1), y.shape

    tensors: dict[str, torch.Tensor] = {}
    for col in range(x.shape[2]):
        tensors[f'x_{col}'] = (
            codes[col].unsqueeze(1) if col in codes else x[:, 0, col].unsqueeze(1)
        )
    tensors['y_0'] = y[:, 0].unsqueeze(1)

    sampled = {
        key: value
        for key, value in model.h.items()
        if key in passed and not callable(value)
    }
    activation = model.h['prior_mlp_activations']
    if prior == 'mlp':
        # `mlp.get_batch` fixed the activation for this dataset (`lambda: s`).
        sampled['prior_mlp_activations'] = type(activation()).__name__
    return Candidate(
        tensors=tensors,
        cat_columns=tuple(sorted(codes)),
        hparams={
            'prior': prior,
            'sampled': _jsonable(sampled),
            'meta': _jsonable(indicators),
        },
    )
