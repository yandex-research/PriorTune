"""The NanoTabICL prior (the approximation of the TabICLv2 prior) as a source.

This is the body of the original `lib.prior.generate_dataset` attempt step,
moved verbatim: one call of the vendored `rand_dataset_plain` for the realized
spec. The categorical structure is the spec's own `x_cat_sizes` (see
`lib.prior.realize_dataset_spec`); the tensors are returned untouched so the
release produced through this module is byte-identical to the releases produced
before the multi-source refactor.

Release layout: upstream this module was `lib.prior_sources.nanotabicl`; it now
sits in bin/priors/_release/ next to `prior.py` and `vendor/nanotabicl/`.
"""

from typing import TYPE_CHECKING

from vendor.nanotabicl import prior as nanoprior

from prior_sources_init import Candidate

if TYPE_CHECKING:
    from prior import DatasetSpec

# The commit of the vendored NanoTabICL (see vendor/nanotabicl/UPSTREAM.md).
COMMIT = 'ead24e3a1bf2dcda3ec46041a5189c034b6c0a62'


def sample(spec: 'DatasetSpec') -> Candidate:
    tensors = nanoprior.rand_dataset_plain(
        list(spec.x_cat_sizes), [spec.n_classes], spec.n_samples
    )
    return Candidate(
        tensors=tensors,
        cat_columns=tuple(i for i, size in enumerate(spec.x_cat_sizes) if size > 0),
        hparams={},
    )
