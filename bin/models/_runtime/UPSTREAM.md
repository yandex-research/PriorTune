# Source provenance

The numerical kernels, data preprocessing, and training loops are frozen
copies of the implementations used to produce the example reference results.
Local adapters change filesystem layout, terminal-result export, and disable
cluster storage hooks. They do not change numerical algorithms.

- TabICL-v2 GraphSCM: `soda-inria/tabicl`, commit
  `67eab6b0065a17ab90dbf51633c7c88a680480fe`, prior package only.
- TabPrep wrapper: `atschalz/tabprep`, commit
  `fee7f189764070836b59e8cf29207050468589f0`.
- AutoGluon trainer: release 1.5.0. The feature-generator and common-feature
  source overlay is from `autogluon/autogluon`, commit
  `554b5c986fc797acdd60b15eff10bd76f6ae1fed`. The remaining common utilities
  are from release 1.5.0. A release wheel alone lacks the required generators.
- LightGBM: release 4.6.0.
- TabPack: the authors' own project, commit
  `eeb42cdc5280608d95b2aa6b978ef1f396720f9f`; distributed here under this
  repository's Apache-2.0 license.
- Muon: upstream `KellerJordan/Muon`, commit
  `f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd`.

Upstream license notices are preserved in `licenses/`.
