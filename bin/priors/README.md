# TabICL-v2 dataset generation

[`tabicl-v2.py`](tabicl-v2.py) contains the full dataset-generation implementation:

- `make_append_only_specs_4000`: immutable dataset indexing and shape sampling.
- `_generate_attempt`: GraphSCM sampling, filtering, categorical handling.
- `_make_split`: stored train/validation/test row assignments.
- `generate_dataset`: NumPy arrays and metadata serialization.
- `main`: the command-line entry point.

The native prior is bundled in [`tabicl-v2/tabicl/prior/`](tabicl-v2/tabicl/prior/).
Its main graph sampler is `graph_lib/_dataset.py` (`RandomDataset`), and
`graph_lib/_config.py` defines `PriorConfig`.

Run from the repository root:

```bash
python bin/priors/tabicl-v2.py --index 0 --output local/datasets/tabicl-v2
```

The paper's 4,000 datasets are indices 0-3999; they are also published under
`datasets/tabicl-v2/` of the artifacts.

## TabICL-v1 and TabPFN-v1

The two other priors share one source-agnostic core in [`_release/`](_release/):
`prior.py` (specs, seeding, attempt loop, acceptance filter, split,
serialization), `generate_prior.py` (the command-line entry point) and
`prior_sources_init.py` (the `Candidate` record and the `source` registry).
`config-example.toml` is the release spec used for the 1000-dataset releases;
set `source = "tabpfn1"` or `source = "tabicl1"` to pick the prior.

- [`tabpfn-v1/`](tabpfn-v1/README.md): the TabPFN v1 prior bag (MLP + GP SCMs),
  vendored from `automl/TabPFN` under `tabpfn-v1/vendor/`.
- [`tabicl-v1/`](tabicl-v1/README.md): the TabICL v1 MLP-SCM / Tree-SCM mixture,
  vendored from `soda-inria/tabicl` v0.1.4 under `tabicl-v1/vendor/`.

```bash
python bin/priors/_release/generate_prior.py <exp dir> --resume
```

`_release/` also carries the helper modules the core needs: `lib/` (`env.py`
locates the repository root, `experiment.py` is the config.toml/report CLI,
`util.py`, `data.py` with the dataset contract, `types.py`, `metrics.py`) and
`vendor/nanotabicl/` (BSD 3-Clause; its shape samplers realize every dataset's
`(n_samples, n_features, cat sizes)` for all sources), plus
`prior_source_nanotabicl.py`, the default `source = "nanotabicl"`. Runtime
dependencies (Python 3.12+): numpy, torch, scipy, scikit-learn, loguru, tqdm,
tomli_w, and xgboost for the TabICL-v1 tree SCM.
