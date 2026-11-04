# TabPFN v1 prior (vendored namespace port)

- Upstream: <https://github.com/automl/TabPFN>, branch `tabpfn_v1`
- Commit: `bef0cf257f3c4c519a2f8e301e0b81a26ae73aff` ("fix: switch to raw", 2025-01-09),
  package version 0.1.9
- Reference (user-provided): <https://github.com/tabpfn-anonym/TabPFNAnonym> @ `a9c1948`
  (2022-11-25, the ICLR-2023 submission code; **no LICENSE file**). The prior modules
  are the same code modulo import paths; the licensed branch above is what is vendored.
- Vendored on: 2026-09-04
- Paper: Hollmann, Müller, Eggensperger, Hutter. "TabPFN: A Transformer That Solves
  Small Tabular Classification Problems in a Second." ICLR 2023.

## License

The upstream README states Apache-2.0 ("Copyright 2022 Noah Hollmann, Samuel Müller,
Katharina Eggensperger, Frank Hutter ... Licensed under the Apache License, Version
2.0"); `LICENSE.txt` on that branch is an **empty file** and `pyproject.toml` carries an
`OSI Approved :: MIT License` classifier. `LICENSE` here is the Apache-2.0 text with the
README's copyright line. Confirm with the authors before redistributing.

## What is vendored

Only the prior (data-generating) code, never the transformer, training loop or
evaluation scripts:

| vendor/tabpfn1/ | upstream `tabpfn/` | status |
| --- | --- | --- |
| `priors/prior.py` | `priors/prior.py` | identical (header only) |
| `priors/utils.py` | `priors/utils.py` | import path; `plot_features`/`plot_prior` (seaborn, matplotlib) removed |
| `priors/mlp.py` | `priors/mlp.py` | import path |
| `priors/fast_gp.py` | `priors/fast_gp.py` | import path (needs `gpytorch`, optional extra `tabpfn-prior`) |
| `priors/flexible_categorical.py` | `priors/flexible_categorical.py` | import path |
| `priors/differentiable_prior.py` | `priors/differentiable_prior.py` | import path |
| `priors/prior_bag.py` | `priors/prior_bag.py` | import path |
| `priors/__init__.py` | `priors/__init__.py` | `fast_gp` no longer imported eagerly |
| `utils.py` | `utils.py` | the subset of helpers the priors import; `default_device = 'cpu'` |
| `model_configs.py` | `scripts/model_configs.py` | only the ConfigSpace-free `get_diff_*` distributions |
| `__init__.py`, `LICENSE`, `UPSTREAM.md`, `PATCHES.md` | — | added here |

The functions kept in `utils.py`: `set_locals_in_self`, `default_device`,
`get_nan_value`, `to_ranking`, `to_ranking_low_mem`, `nan_handling_missing_for_*`,
`torch_masked_mean`, `torch_masked_std`, `torch_nanmean`, `torch_nanstd`,
`normalize_data`, `remove_outliers`, `normalize_by_used_features_f`. Every function body
is byte-identical to upstream.

The resolved hyperparameters of the released model (`reload_config` +
`PriorFittingCustomPrior.ipynb` overrides) are frozen as a constant in
`lib/prior_sources/tabpfn1.py`; ConfigSpace is not needed because after the overrides
no prior-relevant hyperparameter is a `CSH` object.

## How to update

```
git clone --branch tabpfn_v1 https://github.com/automl/TabPFN /tmp/tabpfn-upstream/TabPFN
git -C /tmp/tabpfn-upstream/TabPFN checkout <commit>
```

then re-apply `PATCHES.md` (the port is mechanical: headers, relative imports, the two
subsets), update the commit above and run `uv run pytest dev/tests/test_prior_tabpfn1.py`.
