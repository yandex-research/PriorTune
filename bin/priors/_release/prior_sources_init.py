"""Prior sources: the generative mechanism behind one dataset of a release.

`lib.prior.generate_dataset` owns everything that is common to all sources — the
per-index spec (task type, `n_samples`, `n_features`, `n_classes`), seeding, the
attempt loop and its caps, the acceptance filter, array building, the split and
the serialization — and delegates exactly one step to a source module:

    candidate = <source>.sample(spec)   # after the global RNGs were seeded

A source returns a `Candidate`: the raw feature/target tensors in the layout the
acceptance check expects, which feature columns are categorical (integer codes),
and a JSON-safe record of the realized prior hyperparameters (stored in the
manifest entry). Sources never touch the file system and never seed anything.

The registry maps the `source` field of a release spec to its module (imported
lazily: the TabPFN source needs optional dependencies) and to the data root the
release lands in (`data/<root>/<release>`). The default source keeps the
original root so nothing on disk moves.
"""

import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

DEFAULT_SOURCE = 'nanotabicl'

# source name -> (module, data root under `data/`)
# Upstream the modules were `lib.prior_sources.<name>`; in the release layout
# the v1 sources are `bin/priors/<prior>/prior_source.py` and the default
# NanoTabICL source is `_release/prior_source_nanotabicl.py`; all are loaded by
# file path (the directory names contain hyphens).
_PRIORS_DIR = Path(__file__).resolve().parent.parent
_REGISTRY: dict[str, tuple[str, str]] = {
    'nanotabicl': (str(_PRIORS_DIR / '_release' / 'prior_source_nanotabicl.py'), 'tabicl-prior'),
    'tabpfn1': (str(_PRIORS_DIR / 'tabpfn-v1' / 'prior_source.py'), 'tabpfn1-prior'),
    'tabicl1': (str(_PRIORS_DIR / 'tabicl-v1' / 'prior_source.py'), 'tabicl1-prior'),
}
SOURCE_NAMES = tuple(_REGISTRY)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One attempt at one dataset, before the acceptance check.

    `tensors` follows the NanoTabICL convention consumed by
    `lib.prior.acceptance_check` and `lib.prior._build_arrays`: one `(n, 1)`
    tensor per feature column under the key `x_<column>` (in column order) and
    the target under `y_0` (`(n, 1)`, integer labels for classification).
    Columns listed in `cat_columns` hold integer codes.
    """

    tensors: dict[str, torch.Tensor]
    cat_columns: tuple[int, ...]
    hparams: dict[str, Any]
    # A source that already knows the attempt is unusable (a NaN guard fired,
    # a column came out constant, ...) names the reason here; the attempt is
    # counted as a structural reject without running the acceptance filter.
    reject: None | str = None

    @property
    def n_features(self) -> int:
        return sum(1 for key in self.tensors if key.startswith('x_'))


def get_release_root(source: str) -> str:
    _check_source(source)
    return _REGISTRY[source][1]


def get_source(source: str) -> ModuleType:
    _check_source(source)
    target = _REGISTRY[source][0]
    if not target.endswith('.py'):
        return importlib.import_module(target)
    name = f'prior_source_{source}'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None, target
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check_source(source: str) -> None:
    if source not in _REGISTRY:
        raise ValueError(f'Unknown prior source {source!r}; known: {SOURCE_NAMES}')
