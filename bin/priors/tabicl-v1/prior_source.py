"""The TabICL v1 prior (soda-inria/tabicl v0.1.4) as a source.

Design: `bin/priors/tabicl-v1/DESIGN.md`. The vendored prior
(`vendor/tabicl1`, tag `v0.1.4` @ 1badfa7) is the MLP-SCM / Tree-SCM mixture
that trained the TabICL v1 checkpoint. One dataset is produced exactly as
`SCMPrior.get_batch` + `SCMPrior.generate_dataset` produce it for a group of
size 1 — group-level hyperparameter samplers, the per-subgroup resolution, the
`mix_scm` prior choice, `MLPSCM`/`TreeSCM`, `Reg2Cls`, `delete_unique_features`
— with these deliberate differences:

* `num_features`, `num_classes` and `seq_len` come from the shared per-index
  dataset spec (shape matching across sources) instead of TabICL's own draws;
  `num_classes = 0` (regression, 40 % of the study) is what v1's `Reg2Cls`
  already does for a zero class count (no class assigner, `standard_scaling(y)`);
* one hyperparameter draw per dataset (upstream: one per group of 4; same
  marginal distribution);
* `permute_features = False` and `max_features = num_features` (no zero
  padding): column order carries no information for TabPack or the filter,
  and the contract has no padding;
* the columns `Reg2Cls._num2cat` converts to categorical are exposed as
  integer codes (`x_cat`) instead of being z-scored into the numeric matrix
  (`Reg2ClsRecording`);
* a column that `delete_unique_features` would drop is a structural reject
  (the attempt is redone) rather than a narrower dataset, so that dataset `i`
  has exactly the `n_features` of dataset `i` of every other release. The
  upstream NaN guard (`X = 0`, `y = -100`) is a structural reject too;
* upstream's prefix-split `sanity_check` is replaced by the shared stratified
  split and the "every spec'd class occurs" check of `lib.prior`; the
  ExtraTrees acceptance filter of `lib.prior` (which TabICLv2 later adopted
  upstream) is applied like for every source.

Everything runs on CPU under `torch.no_grad()`, one dataset per call.
"""

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

# Release layout: this file lives in bin/priors/<prior>/; the shared core
# (prior.py, prior_sources_init.py) lives in bin/priors/_release/ and the
# vendored upstream prior in ./vendor/. Upstream these were importable as
# `lib.prior_sources.<x>` and `vendor.tabicl1` from the repository root. The
# vendored package is loaded from ./vendor/ under the name `vendor_tabicl1` so
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


_load_vendored('vendor_tabicl1')

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from vendor_tabicl1 import activations as tabicl_activations
from vendor_tabicl1 import reg2cls
from vendor_tabicl1.dataset import Prior
from vendor_tabicl1.hp_sampling import HpSamplerList
from vendor_tabicl1.mlp_scm import MLPSCM
from vendor_tabicl1.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from vendor_tabicl1.tree_scm import TreeLayer, TreeSCM

from prior_sources_init import Candidate

if TYPE_CHECKING:
    from prior import DatasetSpec

# The commit of the vendored TabICL (see vendor/tabicl1/UPSTREAM.md).
COMMIT = '1badfa744f3098926acae9fabf63e0cec5dd7237'

# `SCMPrior.get_prior` for `prior_type = 'mix_scm'` (the v1 training recipe):
# `np.random.choice(["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas",
# [0.7, 0.3]))`. `DEFAULT_FIXED_HP` spells the key `mix_probs`, so upstream
# always falls back to the literal `[0.7, 0.3]`; the same numbers either way.
PRIOR_TYPES = ('mlp_scm', 'tree_scm')
MIX_PROBS = (0.7, 0.3)

# The value the upstream NaN guard writes into `y` (`MLPSCM.forward`).
_NAN_GUARD_TARGET = -100.0


def sample_hyperparameters() -> tuple[dict[str, Any], str]:
    """The two upstream sampling levels and the prior choice, in upstream order.

    `SCMPrior.get_batch`: `group_sampled_hp = self.hp_sampling()` (samplers,
    one per group), `subgp_prior_type = self.get_prior()`, then the per-subgroup
    resolution `{k: v() if callable(v) else v}`. (Upstream draws the subgroup's
    `num_features` between the last two steps; here it comes from the spec.)
    """
    group = HpSamplerList(DEFAULT_SAMPLED_HP, device='cpu').sample()
    prior_type = str(np.random.choice(PRIOR_TYPES, p=MIX_PROBS))
    resolved = {
        key: value() if callable(value) else value for key, value in group.items()
    }
    return resolved, prior_type


def build_params(
    spec: 'DatasetSpec', resolved: dict[str, Any], prior_type: str
) -> dict[str, Any]:
    """The per-dataset parameter dict of `SCMPrior.get_batch`, key by key."""
    return {
        **DEFAULT_FIXED_HP,
        'seq_len': spec.n_samples,
        # `train_size` (only read by upstream's `sanity_check`) is not needed.
        'max_features': spec.n_features,  # == num_features: no zero padding
        **resolved,
        'prior_type': prior_type,
        'num_features': spec.n_features,
        'num_classes': spec.n_classes,
        'device': 'cpu',
        # Deviation (see the module docstring): keep the column order.
        'permute_features': False,
    }


class Reg2ClsRecording(reg2cls.Reg2Cls):
    """`Reg2Cls` that records which columns became categorical, and their codes.

    `_num2cat` is upstream's line by line (same RNG consumption, same tensor
    writes); the only addition is `self.codes[col]`, the integer codes before
    upstream casts them to float and z-scores them with the numeric columns.
    A parity test (not shipped) pins this against the
    vendored class.
    """

    def __init__(self, hp: dict[str, Any]):
        super().__init__(hp)
        self.codes: dict[int, Tensor] = {}

    def _num2cat(self, X: Tensor) -> Tensor:
        if random.random() < self.hp.get('cat_prob', 0.2):
            col_prob = random.random()  # Probability for converting a specific column
            max_categories = self.hp.get('max_categories', 10)
            for col in range(X.shape[1]):
                if random.random() < col_prob:
                    # Determine number of categories for this feature
                    num_cats = min(
                        max(round(random.gammavariate(1, 10)), 2), max_categories
                    )
                    # Use MulticlassAssigner to convert this column
                    assigner = reg2cls.MulticlassAssigner(
                        num_cats, mode='rank', ordered_prob=0.3
                    )
                    codes = assigner(X[:, col])
                    self.codes[col] = codes.clone()
                    X[:, col] = codes.float()
        return X


def make_model(params: dict[str, Any]) -> nn.Module:
    """`SCMPrior.generate_dataset`'s `prior_cls(**params)`."""
    if params['prior_type'] == 'mlp_scm':
        return MLPSCM(**params)
    if params['prior_type'] == 'tree_scm':
        return TreeSCM(**params)
    raise ValueError(f'Unknown prior type {params["prior_type"]}')


# ==================================================================================
# Manifest record
# ==================================================================================
def _activation_name(module: nn.Module) -> str:
    """The activation class behind `get_activations`' wrappers."""
    if isinstance(module, tabicl_activations.RandomChoiceActivation):
        return _activation_name(module.act)
    if isinstance(module, nn.Sequential):
        # StdRandomScaleFactory: StdScaleLayer -> RandomScaleLayer -> activation
        return _activation_name(module[-1])
    return type(module).__name__


def describe_model(model: nn.Module) -> dict[str, Any]:
    """What the constructed SCM actually looks like (TreeSCM re-samples inside)."""
    record: dict[str, Any] = {
        'is_causal': bool(model.is_causal),
        'num_causes': int(model.num_causes),
        'num_layers': int(model.num_layers),
        'hidden_dim': int(model.hidden_dim),
    }
    if isinstance(model, MLPSCM):
        record['activations'] = [
            _activation_name(layer[0])
            for layer in model.layers
            if isinstance(layer, nn.Sequential)
        ]
    else:
        trees = [
            layer if isinstance(layer, TreeLayer) else layer[0]
            for layer in model.layers
        ]
        record['tree_layers'] = [
            {'max_depth': int(tree.max_depth), 'n_estimators': int(tree.n_estimators)}
            for tree in trees
        ]
    return record


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
    if callable(value):  # the resolved `mlp_activations` is a sampler closure
        return f'<{type(value).__name__}>'
    return str(value)


# ==================================================================================
# The source
# ==================================================================================
def _reject(reason: str, hparams: dict[str, Any]) -> Candidate:
    return Candidate(tensors={}, cat_columns=(), hparams=hparams, reject=reason)


@torch.no_grad()
def sample(spec: 'DatasetSpec') -> Candidate:
    resolved, prior_type = sample_hyperparameters()
    params = build_params(spec, resolved, prior_type)
    model = make_model(params)
    x, y = model()
    hparams: dict[str, Any] = {
        'prior_type': prior_type,
        'sampled': _jsonable(resolved),
        'model': describe_model(model),
    }
    assert x.shape == (spec.n_samples, spec.n_features), x.shape
    assert y.shape == (spec.n_samples,), y.shape
    if bool((y == _NAN_GUARD_TARGET).all()) and bool((x == 0.0).all()):
        return _reject('nan_guard', hparams)

    converter = Reg2ClsRecording(params)
    x, y = converter(x, y)
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        return _reject('non_finite', hparams)

    # Upstream drops constant columns and narrows the dataset; here that is a
    # reject (the realized width must be the spec's, see the module docstring).
    _, kept = Prior.delete_unique_features(
        x.unsqueeze(0), torch.tensor([spec.n_features], dtype=torch.long)
    )
    n_realized = int(kept.item())
    hparams['n_features_realized'] = n_realized
    if n_realized != spec.n_features:
        return _reject('constant_column', hparams)

    codes = converter.codes
    tensors: dict[str, Tensor] = {}
    for col in range(spec.n_features):
        tensors[f'x_{col}'] = (
            codes[col].unsqueeze(1) if col in codes else x[:, col].unsqueeze(1)
        )
    tensors['y_0'] = y.unsqueeze(1)
    hparams['cat_columns'] = sorted(codes)
    hparams['cat_sizes'] = [len(torch.unique(codes[col])) for col in sorted(codes)]
    return Candidate(tensors=tensors, cat_columns=tuple(sorted(codes)), hparams=hparams)
