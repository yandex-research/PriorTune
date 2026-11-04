# ====================================================================================
# Source: https://github.com/automl/TabPFN/blob/bef0cf257f3c4c519a2f8e301e0b81a26ae73aff/tabpfn/priors/__init__.py
# Mechanical namespace port of the TabPFN v1 prior (branch tabpfn_v1): only the
# lines listed in vendor/tabpfn1/PATCHES.md were changed. No algorithmic changes.
# fmt: off
# isort: off
# ====================================================================================
# `fast_gp` (gpytorch) is imported on demand: gpytorch is an optional dependency here.
from . import mlp, flexible_categorical, differentiable_prior, prior_bag



