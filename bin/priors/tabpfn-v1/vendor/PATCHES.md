# Patches (mechanical namespace port)

Every deviation from upstream `tabpfn/` @ `bef0cf257f3c4c519a2f8e301e0b81a26ae73aff`,
besides the provenance header prepended to every file:

- `from tabpfn.utils import ...` becomes `from ..utils import ...` in
  `priors/{utils,mlp,fast_gp,flexible_categorical,differentiable_prior,prior_bag}.py`
  (one line each; the imported names are unchanged).
- `priors/utils.py`: `plot_features` and `plot_prior` (upstream lines 49–101) are
  removed. They import `seaborn`/`matplotlib`, which are dev-only here, and nothing in
  the priors calls them.
- `priors/__init__.py`: `fast_gp` is not imported eagerly because it imports
  `gpytorch`, an optional dependency (`uv run --extra tabpfn-prior`). Import
  `vendor.tabpfn1.priors.fast_gp` explicitly where the GP branch is needed.
- `utils.py`: only the helpers listed in `UPSTREAM.md` are kept (byte-identical
  bodies); `default_device` is the constant `'cpu'` instead of
  `'cuda:0' if torch.cuda.is_available() else 'cpu:0'` — dataset generation runs on
  CPU and must not depend on the host's GPU.
- `model_configs.py`: only `get_diff_flex`, `get_diff_gp`, `get_diff_causal`,
  `get_diff_prior_bag` and `get_diff_config` are kept (byte-identical bodies). The
  other functions need `ConfigSpace` and only produce transformer-training settings
  once the notebook overrides are applied.

No algorithmic line differs. Deliberate *usage* deviations (shape matching, task type
from the spec, no zero-padding, no `check_is_compatible`, categorical codes exposed as
`x_cat`, one hyperparameter draw per dataset) are implemented in
`lib/prior_sources/tabpfn1.py` and documented in `bin/priors/tabpfn-v1/DESIGN.md`.
