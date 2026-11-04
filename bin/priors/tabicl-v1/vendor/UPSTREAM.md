# TabICL v1 prior (vendored)

- Upstream: https://github.com/soda-inria/tabicl
- Tag: `v0.1.4`, commit `1badfa744f3098926acae9fabf63e0cec5dd7237`
  ("bump: Update version to 0.1.4", 2025-11-27) — the last release of the
  v1 line; `main` is the TabICLv2 prior (graph SCM, fixed `is_causal = False`,
  outlier-clipped targets) and is deliberately not what is vendored here.
  See `bin/priors/tabicl-v1/DESIGN.md` for the v1/v2 delta.
- Vendored on: 2026-09-04
- License: BSD 3-Clause (see `LICENSE`), Copyright (c) 2025, Soda team @ Inria.

## Contents

`src/tabicl/prior/` of the tag, byte-identical (`cmp` against the checkout):

| File | Status |
| --- | --- |
| `__init__.py` | byte-identical (empty upstream) |
| `activations.py` | byte-identical |
| `dataset.py` | byte-identical (only `Prior.delete_unique_features` is executed; `SCMPrior.get_batch`, joblib batching and `PriorDataset` are not) |
| `hp_sampling.py` | byte-identical |
| `mlp_scm.py` | byte-identical |
| `prior_config.py` | byte-identical |
| `reg2cls.py` | byte-identical |
| `tree_scm.py` | byte-identical |
| `utils.py` | byte-identical |
| `LICENSE` | byte-identical to the repository root `LICENSE` |
| `UPSTREAM.md` | added here |

`genload.py` (batch files / DataLoader) is not vendored: it is not needed.
No upstream file is modified in any way. Package imports are relative, so no
import rewrite is needed. `dataset.py` imports `joblib` at module top; joblib
is a dependency of scikit-learn and therefore always installed, so no patch is
needed either. `tree_scm.py` imports `xgboost`, which is the
`tabicl-prior` optional extra of `pyproject.toml` (generation host only).

The project-side wrapper lives in `lib/prior_sources/tabicl1.py`: it mirrors
the per-dataset parameter assembly of `SCMPrior.get_batch` with a group size
of 1 and replaces upstream's `sanity_check` with the shared acceptance filter
and split of `lib/prior.py`.

## How to update

```
git clone --branch v0.1.4 --depth 1 https://github.com/soda-inria/tabicl.git /tmp/tabicl-upstream
cp /tmp/tabicl-upstream/src/tabicl/prior/{__init__,activations,dataset,hp_sampling,mlp_scm,prior_config,reg2cls,tree_scm,utils}.py vendor/tabicl1/
cp /tmp/tabicl-upstream/LICENSE vendor/tabicl1/
```

Then update the commit hash above and re-run the tests
(a parity test in the generator's development repository, not shipped).
