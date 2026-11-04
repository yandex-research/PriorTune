# ====================================================================================
# Source: https://github.com/automl/TabPFN/blob/bef0cf257f3c4c519a2f8e301e0b81a26ae73aff/tabpfn/priors/prior.py
# Mechanical namespace port of the TabPFN v1 prior (branch tabpfn_v1): only the
# lines listed in vendor/tabpfn1/PATCHES.md were changed. No algorithmic changes.
# fmt: off
# isort: off
# ====================================================================================
from torch.utils.data import DataLoader


class PriorDataLoader(DataLoader):
    pass
    # init accepts num_steps as first argument

    # has two attributes set on class or object level:
    # num_features: int and
    # num_outputs: int
    # fuse_x_y: bool
    # Optional: validate function that accepts a transformer model
