# TabICL-v1 prior

The synthetic-dataset prior of TabICL v1 (Qu et al., 2025): the MLP-SCM /
Tree-SCM mixture (`mix_scm`) with group-level hyperparameter samplers and the
`Reg2Cls` target transform that trained the v1 checkpoint. [`DESIGN.md`](DESIGN.md)
records the reproduction of `SCMPrior.get_batch` for a group of size one and
every deliberate deviation (shape matching, structural rejects, no padding).

Upstream: `soda-inria/tabicl`, tag `v0.1.4` @ `1badfa7`, `src/tabicl/prior`
vendored unmodified (see [`vendor/UPSTREAM.md`](vendor/UPSTREAM.md) and
[`vendor/LICENSE`](vendor/LICENSE), BSD 3-Clause).

[`prior_source.py`](prior_source.py) exposes `sample(spec) -> Candidate`, called
once per attempt by `generate_dataset` in [`../_release/prior.py`](../_release/prior.py)
after the global RNGs are seeded; a `Candidate` with `reject` set is counted as a
structural reject without running the acceptance filter. The registry in
`../_release/prior_sources_init.py` maps `source = "tabicl1"` to this file and to
the data root `data/tabicl1-prior/<release>`. Generate a release with

```bash
python bin/priors/_release/generate_prior.py <exp dir> --resume
```

where `<exp dir>/config.toml` is a copy of `../_release/config-example.toml`
with `source = "tabicl1"` added. Upstream the module was `lib.prior_sources.tabicl1`
and the vendored code `vendor.tabicl1`; here `prior_source.py` inserts
`../_release` into `sys.path` and loads `./vendor/` under the package name
`vendor_tabicl1` (so it cannot clash with `_release/vendor/`), so
`vendor.tabicl1.*` became `vendor_tabicl1.*` and `lib.prior_sources` became
`prior_sources_init`.
