# TabPFN-v1 prior

The synthetic-dataset prior of TabPFN v1 (Hollmann et al., ICLR 2023): a
prior bag of Bayesian-neural-network (MLP) and Gaussian-process structural
causal models with differentiable hyperparameters and the `FlexibleCategorical`
wrapper. [`DESIGN.md`](DESIGN.md) records how the released model's settings
are reproduced and every deliberate deviation.

Upstream: `automl/TabPFN`, branch `tabpfn_v1` @ `bef0cf25`, vendored unmodified
except for the patches listed in [`vendor/PATCHES.md`](vendor/PATCHES.md)
(see [`vendor/UPSTREAM.md`](vendor/UPSTREAM.md) and [`vendor/LICENSE`](vendor/LICENSE), Apache-2.0).

[`prior_source.py`](prior_source.py) exposes one function, `sample(spec) -> Candidate`,
which the source-agnostic core in [`../_release/prior.py`](../_release/prior.py)
(`generate_dataset`) calls once per attempt after seeding the global RNGs; the
core applies the acceptance filter, the split and the serialization. The
registry in `../_release/prior_sources_init.py` maps `source = "tabpfn1"` to this
file and to the data root `data/tabpfn1-prior/<release>`. Generate a release with

```bash
python bin/priors/_release/generate_prior.py <exp dir> --resume
```

where `<exp dir>/config.toml` is a copy of `../_release/config-example.toml`
with `source = "tabpfn1"` added. Upstream the module was `lib.prior_sources.tabpfn1`
and the vendored code `vendor.tabpfn1`; here `prior_source.py` inserts
`../_release` into `sys.path` and loads `./vendor/` under the package name
`vendor_tabpfn1` (so it cannot clash with `_release/vendor/`), so
`vendor.tabpfn1.*` became `vendor_tabpfn1.*` and `lib.prior_sources` became
`prior_sources_init`.
