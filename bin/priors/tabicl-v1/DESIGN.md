# TabICL v1 prior: 1,000 configs × 1,000 datasets

## Purpose

Re-run the `exp/tabpack-prior/full` study (1,000 TabPack MLP + `CosineEmbeddingsPack`
+ Muon configs, bank hash `9f3681d9…`) on 1,000 datasets drawn from the **TabICL v1
prior** (Qu et al., ICML 2025: the MLP-SCM / Tree-SCM mixture) instead of the
NanoTabICL approximation of the TabICLv2 prior. As in the TabPFN v1 sibling design,
only the generative mechanism changes:

- the same config bank, by value (`freeze --bank-from exp/tabpack-prior/full`);
- the same pack size 200, batch 256, trainer settings and `noisy-quantile` num policy;
- the same per-index dataset shape as `full-1000` (task, `n_samples`, `n_features`,
  `n_classes`), the same acceptance filter, split, data contract, upload and launch.

Sibling design: `bin/priors/tabpfn-v1/DESIGN.md`. The "multi-source release
plumbing" and "shape matching" sections there apply verbatim and are implemented once;
this document only restates what is TabICL-specific.

## Which code is "the v1 prior"

The provided link (`soda-inria/tabicl` `main`, `src/tabicl/prior/`) is the **TabICLv2**
prior: `main` is tagged `v2.2.0`, the "Code for TabICLv2" commit landed 2026-02-12 and
the graph-SCM prior (`_graph_scm.py`, `graph_lib/`, default `--prior_type graph_scm`)
on 2026-07-07. The v1 prior is the `v0.1.x` line; the last prior change on that line is
`201fa34` ("sample more binary datasets", 2025-05-05) and the last release is tag
**`v0.1.4`** (`1badfa7`, 2025-11-27), which is what gets vendored.

Delta between `v0.1.4` and `main` in the SCM path (the files are otherwise identical):

| Aspect | `v0.1.4` (v1, vendored) | `main` (v2) |
| --- | --- | --- |
| Prior types | `mlp_scm`, `tree_scm`, `mix_scm` (0.7 / 0.3) | + `graph_scm` (default) |
| `is_causal` | **sampled** (meta_choice True/False) per group | fixed `False` |
| Target preprocessing | `standard_scaling(y)` | `outlier_removing(y)` + `standard_scaling(y)` |
| Regression | `Reg2Cls` supports `num_classes == 0` but `SCMPrior` never emits it | `regression` flag through the stack |
| Acceptance | none; `while True` until `delete_unique_features`/`sanity_check` pass | ExtraTrees p-value + trivial-dataset filters, 100-attempt cap |
| Feature count | `U_int(min, max_features)` | + `log_n_features` |
| Kwarg names | `max_depth_lambda`, `n_estimators_lambda`; `mix_probas` typo (falls back to 0.7/0.3) | `tree_depth_lambda`, `tree_n_estimators_lambda`; `mix_probs` |
| Module names | `prior/mlp_scm.py` … (public) | `prior/_mlp_scm.py` … (private) |

The upstream training recipe for the v1 checkpoint (`scripts/train_stage{1,2,3}.sh`,
unchanged since v0.1.4 apart from an `--ignore_d` flag): `--prior_type mix_scm`,
`--min_features 2 --max_features 100 --max_classes 10`, stage 1 `max_seq_len 1024`
(fixed), stage 2 `min/max_seq_len 1000/40000 --log_seq_len --seq_len_per_gp`, stage 3
`40000/60000` with `--replay_small`; `batch_size_per_gp` 4 / 2 / 1. Our shared spec
(`n_samples` log-uniform in `[1024, 32768]`, `n_features` in `[4, 64]`) sits inside the
stage-2 regime, so no dataset in the release is outside what the v1 prior produced.

## Source pointers

| Resource | Location | Use |
| --- | --- | --- |
| Upstream | <https://github.com/soda-inria/tabicl> tag `v0.1.4` @ `1badfa7`, BSD-3-Clause (Soda team @ Inria) | Vendored unmodified |
| Prior package | `src/tabicl/prior/{activations,dataset,hp_sampling,mlp_scm,prior_config,reg2cls,tree_scm,utils}.py` | `genload.py` (batch files / DataLoader) is not needed |
| Reference for v2 delta | `main` @ `0dbff3e`, `src/tabicl/prior/_dataset.py` `should_filter` | Same ExtraTrees test as ours; supports applying our filter to this prior |
| NanoTabICL wrapper | `lib/prior.py`, `vendor/nanotabicl/UPSTREAM.md` | Core reused; vendoring conventions |
| Existing sweep / launch | `exp/tabpack-prior/full`, the synth-1000-launch LOG note (not included) | Space, trainer, bank; resource shape and steps |

## What the TabICL v1 prior is

`SCMPrior.get_batch` (v0.1.4 `dataset.py`) builds one parameter dict per dataset and
`generate_dataset` runs it:

1. **Group-level hyperparameters**: `HpSamplerList(DEFAULT_SAMPLED_HP).sample()` returns
   *samplers* (meta-distributions); per subgroup they are resolved by
   `{k: v() if callable(v) else v}`. Two levels of randomness per dataset when the group
   size is 1.
2. **Per subgroup**: `prior_type` (`mix_scm` → `mlp_scm` w.p. 0.7, `tree_scm` 0.3),
   `num_features ~ U_int(2, max_features)` (max reduced by `adjust_max_features` for
   long sequences: 100 → 80 → 60 → 40 at 10k/20k/30k rows).
3. **Per dataset**: `num_classes = 2` w.p. ½ else `U_int(2, 10)`.
4. **MLPSCM** (`mlp_scm.py`): causes from `XSampler(sampling ∈ {normal, mixed,
   uniform}, pre_stats)`; `num_layers` layers of (activation → Linear → GaussianNoise);
   block-wise-dropout or Bernoulli-dropout init; causal mode samples X/y from hidden
   activations (`is_causal`, `y_is_effect`, `in_clique`, `sort_features`), non-causal
   mode uses causes as X and the last layer as y. Activations from
   `get_activations(random=True, scale=True, diverse=True)`: 18 simple activations +
   10× `RandomFunctionActivation` (256 random Fourier features), each wrapped in
   `StdScaleLayer → RandomScaleLayer → act`, plus per-layer random choice.
5. **TreeSCM** (`tree_scm.py`): forces `is_causal = False`, `num_layers ∈ {1, 2}`,
   `hidden_dim ∈ [3, 9]`; each layer fits an `XGBRegressor(tree_method='hist',
   multi_strategy='multi_output_tree', max_depth ≤ 4, n_estimators ≤ 4)` on random
   targets and predicts; Gaussian noise after each layer.
6. **Reg2Cls** (`reg2cls.py`): `_num2cat` (w.p. `cat_prob = 0.2` a dataset gets
   categorical columns; each column w.p. `U(0,1)`; `min(max(round(Gamma(1,10)), 2),
   max_categories = ∞)` categories via `MulticlassAssigner('rank', ordered_prob 0.3)`);
   `outlier_removing` (4σ clamp) + `standard_scaling` (clip ±100) on X; optional
   feature permutation; padding to `max_features`; `standard_scaling(y)`; class
   assignment (`MulticlassAssigner(K, mode ∈ {value, rank}, ordered_prob 0)` →
   random label permutation / reversal) or `BalancedBinarize` (`balanced = False`, so
   never); `permute_classes` (`permute_labels = True`).
7. `delete_unique_features` (drop constant columns), `sanity_check` (train prefix and
   test suffix contain the same ≥ 2 classes; up to 10 row permutations), else retry.

Fixed and sampled hyperparameters (`prior_config.py` @ v0.1.4):

| Key | Value / distribution |
| --- | --- |
| `mix_probs` | (0.7, 0.3) |
| `tree_model`, `tree_depth_lambda`, `tree_n_estimators_lambda` | `xgboost`, 0.5, 0.5 (depth `2 + Exp(2)`, trees `1 + Exp(2)`, both capped at 4) |
| `balanced`, `multiclass_ordered_prob`, `cat_prob`, `max_categories` | False, 0.0, 0.2, ∞ |
| `scale_by_max_features`, `permute_features`, `permute_labels` | False, True, True |
| `multiclass_type` | meta_choice(value, rank) |
| `mlp_activations` | meta_choice_mixed over `get_activations(True, True, True)` |
| `block_wise_dropout` | meta_choice(True, False) |
| `mlp_dropout_prob` | meta_beta(scale 0.9, a,b ~ U(0.1, 5)) |
| `is_causal`, `y_is_effect`, `in_clique`, `sort_features` | meta_choice(True, False) |
| `num_causes` | meta_trunc_norm_log_scaled(mean ∈ [1, 12]), round, ≥ 1 |
| `num_layers` | meta_trunc_norm_log_scaled(mean ∈ [1, 6]), round, ≥ 2 |
| `hidden_dim` | meta_trunc_norm_log_scaled(mean ∈ [5, 130]), round, ≥ 4; raised to `1 + 2·num_features` when causal |
| `init_std` | meta_trunc_norm_log_scaled(mean ∈ [0.01, 10]) |
| `noise_std` | meta_trunc_norm_log_scaled(mean ∈ [1e-4, 0.3]) |
| `sampling` | meta_choice(normal, mixed, uniform) |
| `pre_sample_cause_stats`, `pre_sample_noise_std` | meta_choice(True, False) |

## Layout

    vendor/tabicl1/                  # src/tabicl/prior @ v0.1.4, byte-identical
      UPSTREAM.md LICENSE __init__.py
      activations.py dataset.py hp_sampling.py mlp_scm.py prior_config.py
      reg2cls.py tree_scm.py utils.py
    lib/prior_sources/tabicl1.py     # this design (core + plumbing: see sibling design)
    exp/tabicl1-prior/{pilot-50,full-1000}/config.toml   # source = "tabicl1"
    data/tabicl1-prior/<release>/
    exp/tabpack-tabicl1/full/        # sweep.toml + bank.json (+ cells)
    dev/tests/test_prior_tabicl1.py

`dataset.py` is vendored for provenance and for `Prior.delete_unique_features`, but its
batching (`joblib`, nested tensors, `SavePriorDataset`) is not executed; `joblib` is
therefore not a dependency (it is imported at module top, so the port either vendors
`dataset.py` with that import behind a guard — recorded in `PATCHES.md` — or copies
`delete_unique_features` into the wrapper with a line-by-line parity test; the former
is preferred, it is a one-line patch).

## Per-dataset generation (`lib/prior_sources/tabicl1.py`)

CPU only, one dataset per call; mirrors `SCMPrior.get_batch` + `generate_dataset` with
group size 1.

1. Seed `random`, NumPy, Torch from `(release.seed, index, attempt)`. XGBoost is
   deterministic under `tree_method='hist'` with one thread and its default seed.
2. `group = HpSamplerList(DEFAULT_SAMPLED_HP, device='cpu').sample()`;
   `hp = {k: v() if callable(v) else v for k, v in group.items()}` (both upstream levels).
3. `prior_type = np.random.choice(['mlp_scm', 'tree_scm'], p=(0.7, 0.3))`.
4. `params = {**DEFAULT_FIXED_HP, 'seq_len': n_samples, 'train_size': n_train,
   'max_features': n_features, **hp, 'prior_type': …, 'num_features': n_features,
   'num_classes': K, 'device': 'cpu'}` with, from the shared spec, `n_features`,
   `K ∈ {0 (regression), 2, 3..10}`, and `permute_features = False` (see deviations).
5. `X, y = (MLPSCM | TreeSCM)(**params)()`; the NaN guard (`X = 0, y = -100`) →
   structural reject.
6. `X, y = Reg2ClsRecording(params)(X, y)` — a `lib/` subclass overriding `_num2cat`
   with the same eight lines plus a record of converted columns and their codes.
   Parity test: with cat exposure off, identical output to the vendored `Reg2Cls`
   under the same seed.
7. `delete_unique_features` (upstream): constant columns are dropped, the realized
   width is recorded as `n_features_realized`; `0` → structural reject. Upstream's
   `sanity_check` is replaced by our stratified `_make_split` + the "every spec'd class
   occurs" check in `_build_arrays` (missing classes → structural reject).
8. Arrays: `x_num` = surviving non-converted columns after upstream scaling (float32);
   `x_cat` = recorded integer codes (int64) of converted columns; `y` = standardized
   continuous target (regression) or the permuted class labels.
9. `lib.prior.acceptance_check` (ExtraTrees OOB bootstrap p < 0.05), split, serialize,
   manifest (with the realized `hp`, `prior_type`, `n_features_realized`, cat columns).

Deviations from upstream, all deliberate:

| Upstream | Here | Why |
| --- | --- | --- |
| `num_features ~ U_int(2, max)`, `num_classes` 50 % binary else `U_int(2, 10)`, seq_len per stage | from the shared per-index spec | Shape matching with `full-1000` and the TabPFN release |
| classification only (v1 training) | `num_classes = 0` for regression datasets, which v1's `Reg2Cls` already supports (`class_assigner = None`, `standard_scaling(y)`) | 40 % of the study is regression |
| no acceptance filter | NanoTabICL's ExtraTrees filter (the same test TabICLv2 later adopted upstream) | Comparability of "predictable" datasets across sources |
| group of 4 datasets shares hyperparameters | one draw per dataset | Same marginal |
| `permute_features = True` | False | Column order carries no information for TabPack or the filter; keeps the categorical mask trivially aligned |
| padding to `max_features` | `max_features = num_features` → no zero columns | The contract has no padding |
| categorical codes z-scored into X | exposed as `x_cat` | Real datasets carry this information |
| `sanity_check` on a prefix split | our stratified split | Split semantics differ |

## Vendoring and dependencies

- `vendor/tabicl1/` is byte-identical to `src/tabicl/prior/` at `v0.1.4` (plus
  `__init__.py`, `UPSTREAM.md`, `LICENSE`), following `vendor/nanotabicl`. Package
   imports are already relative, so no import rewrite is needed.
- **xgboost** (pinned; ≥ 2.0 for `multi_strategy='multi_output_tree'`, e.g.
  `xgboost==3.0.x`) as a `[project.optional-dependencies] tabicl-prior` extra, used
  only on the generation host. The existing `OMP_NUM_THREADS=1` in
  `bin/generate_prior.py` bounds XGBoost's `n_jobs=-1` and `MultiOutputRegressor`'s
  `n_jobs=-1` to one thread per worker.
- scipy (`loguniform`, `truncnorm`), scikit-learn, torch: already pinned.

## Cost and risk

- **Memory, not time, is the hazard.** `RandomFunctionActivation.forward` forms
  `sin(freqs · x[..., None] + bias)` of shape `(n_samples, hidden_dim, 256)`: at
  32,768 rows and `hidden_dim ≥ 129` (forced by `1 + 2·64` in causal mode) that is a
  4.3 GB float32 transient per activation call, several per forward. It is drawn ~10×
  more often than any single simple activation. Upstream ran this at up to 60k rows.
  Mitigation without touching vendor code: the pilot measures peak RSS per worker on
  the largest specs and `n_workers` is chosen so `n_workers × peak` fits the host; a
  size-aware worker count (fewer workers for `n_samples > 16k`) is the fallback.
- TreeSCM: XGBoost with ≤ 4 trees, depth ≤ 4, ≤ 9 outputs on 32k × 64: well under a
  second per layer. MLPSCM forward: trivial. ExtraTrees filter: 1–3 s at 32k rows.
  Expect the mean well under 5 s/dataset; the pilot gives the number.
- Acceptance and structural reject rates are unknown; `SignActivation`/`UnitIntervalIndicator`/`RBF`-type
  activations produce constant columns (dropped upstream) and can produce constant
  targets (rejected here). The 64-attempt / 600 s caps stay; `failed_dataset_ids` is
  the escape hatch.
- `max_categories = ∞` with `Gamma(1, 10)` categories: P(> 100) ≈ 4.5e-5 per column;
  the ordinal `cat_policy` handles any cardinality, so no cap is added.

## Sweep and launch

    exp/tabpack-tabicl1/full/sweep.toml
      seed = 0, n_configs = 1000, pack_size = 200
      release = "full-1000", release_root = "data/tabicl1-prior"
      [space] / [trainer] verbatim from exp/tabpack-prior/full,
      except trainer.save_final_predictions = true

    python bin/plots/submit_sweep.py freeze exp/tabpack-tabicl1/full \
        --bank-from exp/tabpack-prior/full          # bank_hash must equal 9f3681d9…
    python bin/plots/submit_sweep.py plan  exp/tabpack-tabicl1/full --num-jobs 4 --task-minutes 1.5
    python bin/plots/submit_sweep.py emit  exp/tabpack-tabicl1/full --num-jobs 4 \
        --jobs-per-instance 1 --priority low --n-cpu-cores 16 --memory-gb 64 \
        --disk-gb 100 --ttl-hours 12

5,000 cells, 4 × 8 A100-80G at low priority, ~4 h/job, predictions on (≈17.5 GiB total
as measured for identical shapes in `full-pred`), consolidated to one file per dataset
as the sweep runs (see "Outputs and storage"). Download, `check`, and the T+E study
follow the synth-1000-launch LOG note (not included). The two new sweeps can be launched
independently or together; each is 32 GPUs at low priority.

## Outputs and storage

Identical to the TabPFN v1 design (shared implementation in `lib/sweep_artifacts.py`);
the requirements and format are restated here so this document stands alone:

- **Caruana ensembles** need val/test predictions and the true val/test labels: both
  are in the per-dataset file (`pred_val`, `pred_test` fp16, `y_val`, `y_test`,
  global `config_ids`, task type, score name, `prediction_type`).
- **Meta-tuning** needs per-dataset, per-config scores: `score_{train,val,test}` and
  `best_step` per config in the same file, merged into one `scores.npz`
  (`[K=1000, N=1000]` per part, ≈ 12 MB) by `bin/sweep_scores.py` after download.
- **File and space limits**: the worker consolidates each dataset's 5 packs into one
  `np.savez_compressed` file as soon as the last pack finishes and deletes the cell
  directories, so a 250-dataset job ships ~250 files / ~4.4 GiB instead of 3,750
  files, and nothing per cell is committed to Git (the previous layout put 10,002
  files and 1.5 GiB of JSON per sweep into the repository). Git tracks `sweep.toml`,
  `bank.json` and a ~1,000-row `summary.json` only.
- Completeness, retry and the T+E reader work on the consolidated layout;
  `exp/tabpack-prior/full-pred` is converted to it so all three sources are analyzed
  by the same code.

## Rejected alternatives

- **Vendoring `main` and disabling the graph prior.** The `mlp_scm`/`tree_scm` branch
  on `main` is *not* the v1 prior: `is_causal` is fixed to False and the target is
  outlier-clipped. Tag `v0.1.4` is the v1 code.
- **Reproducing the v1 curriculum shapes (1024-row stage 1, 40–60k stage 3).** Confounds
  the mechanism with the shape; the shared spec is inside stage 2's regime anyway.
- **Skipping the acceptance filter to stay faithful to v1.** Would compare filtered
  (NanoTabICL) against unfiltered datasets; TabICLv2 itself adopted this exact test,
  and the manifest records the p-value so the effect of filtering can be studied.
- **Chunking `RandomFunctionActivation` to cut memory.** Touches vendored code; the
  worker count achieves the same without a patch.
- **Using upstream `SCMPrior.get_batch` with `batch_size = 1`.** Drags in joblib,
  nested tensors and the prefix-split sanity check for no benefit; the per-dataset
  parameter assembly is ten lines and is mirrored explicitly with a comment per line.

## Milestones

1. **M0** — shared multi-source plumbing and output consolidation (see the TabPFN v1
   design); skip if done.
2. **M1** — vendor `vendor/tabicl1` @ `v0.1.4`; import smoke incl. xgboost;
   `UPSTREAM.md`.
3. **M2** — `lib/prior_sources/tabicl1.py` + `Reg2ClsRecording`; tests: contract
   datasets, determinism, `_num2cat` parity, shape alignment with `full-1000`, a forced
   `RandomFunctionActivation` draw at 32,768 × 64 (peak RSS).
4. **M3** — `exp/tabicl1-prior/pilot-50`: reject rates, s/dataset, RSS,
   MLP/tree and categorical shares; set `n_workers`.
5. **M4** — `exp/tabicl1-prior/full-1000`, upload, data ID in `release.json`; one-cell
   local smoke.
6. **M5** — freeze `exp/tabpack-tabicl1/full` with the reused bank, emit, launch 4
   low-priority instances, download, `check`, `LOG.md`; then the T+E study.
