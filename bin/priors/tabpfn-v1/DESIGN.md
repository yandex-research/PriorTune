# TabPFN v1 prior: 1,000 configs × 1,000 datasets

## Purpose

Re-run the `exp/tabpack-prior/full` study (1,000 TabPack MLP + `CosineEmbeddingsPack`
+ Muon configs, bank hash `9f3681d9…`) on 1,000 datasets drawn from the **TabPFN v1
prior** (Hollmann et al., ICLR 2023) instead of the NanoTabICL approximation of the
TabICLv2 prior. The question is whether meta-tuning (top-k / T+E portfolios) depends on
the synthetic source, so everything except the generative mechanism is held fixed:

- the same config bank, by value (`freeze --bank-from exp/tabpack-prior/full`);
- the same pack size 200, batch 256, trainer settings and `noisy-quantile` num policy;
- the same **per-index dataset shape**: dataset `i` has the same task type, `n_samples`,
  `n_features` and `n_classes` as dataset `i` of `full-1000` (see "Shape matching");
- the same acceptance filter, split, data contract, upload and launch machinery.

Sibling design: `bin/priors/tabicl-v1/DESIGN.md` (TabICLv1 prior). Both share the
"multi-source release" refactor described below; implement it once.

## Source pointers

| Resource | Location | Use |
| --- | --- | --- |
| User-provided reference | <https://github.com/tabpfn-anonym/TabPFNAnonym/tree/main> @ `a9c1948` (2022-11-25) | ICLR-submission code; **no LICENSE file** |
| Vendoring source | <https://github.com/automl/TabPFN> branch `tabpfn_v1` @ `bef0cf25` (2025-01-09), package version 0.1.9 | Same prior code, Apache-2.0 per README (`LICENSE.txt` on that branch is an empty file; `pyproject.toml` carries an MIT classifier — record both in UPSTREAM.md and confirm before pushing) |
| Prior modules | `tabpfn/priors/{mlp,fast_gp,flexible_categorical,differentiable_prior,prior_bag,utils,prior}.py` | Verified byte-identical to the anonymous repo modulo import paths |
| Helpers | `tabpfn/utils.py`: `normalize_data`, `remove_outliers`, `torch_nanmean/std`, `to_ranking_low_mem`, `nan_handling_*`, `normalize_by_used_features_f`, `set_locals_in_self`, `default_device` | The only functions the priors import from the package root |
| HP space | `tabpfn/scripts/model_configs.py` (`get_prior_config_causal`, `get_diff_*`) | The exact TabPFN v1 prior distribution |
| Training recipe | `PriorFittingCustomPrior.ipynb` (anon repo) cells 8–10: `reload_config` + overrides | The concrete overrides applied for the released model |
| Existing wrapper | `lib/prior.py`, `bin/generate_prior.py`, `bin/upload_release.py` | Spec/seeding/filter/split/serialize/manifest — reused |
| Existing sweep | `exp/tabpack-prior/full/{sweep.toml,bank.json}` | Space, trainer, bank to reuse |
| Launch log | the synth-1000-launch LOG note (not included) | Measured pack times, resource shape, download/check steps |

## What the TabPFN v1 prior is (as run for the released model)

`prior_type = 'prior_bag'`, `differentiable = True`, `flexible = True`. Per group of
datasets the pipeline is:

1. **Differentiable hyperparameters** (`differentiable_prior.py`): one draw of the
   "meta" hyperparameters per `DifferentiablePrior` instance (one instance per
   `batch_size_per_gp_sample` datasets; in the released run that is 8 datasets).
2. **Prior bag** (`prior_bag.py`): GP vs MLP chosen with `softmax([1, w])`,
   `w = prior_bag_exp_weights_1 ~ U(2, 10)` → P(GP) = 1/(1+e^w), ≈1.6 % on average.
3. **Base prior**:
   - `mlp.py`: random MLP / causal graph. `new_mlp_per_example = True` → a fresh MLP per
     dataset with the group's hyperparameters. Causes `~ N(μ_j, σ_j)` with pre-sampled
     per-cause stats (`pre_sample_causes = True`, `sampling = 'normal'`); optional
     block-wise dropout init; Gaussian layer noise; causal mode samples X and y from
     hidden activations (`is_causal`, `y_is_effect`, `in_clique`, `sort_features`);
     `random_feature_rotation = True`.
   - `fast_gp.py`: RBF GP via gpytorch in `prior_mode`, `x ~ U(0,1)^d`, sample of
     `K + noise·I` with `fast_computations(True, True, True)` — for `n > 800` gpytorch
     draws the sample from a **rank-100 Lanczos root decomposition**, i.e. TabPFN's GP
     datasets are low-rank approximate GP samples (upstream behaviour, kept as is;
     verify the thresholds against the pinned gpytorch in M1).
4. **FlexibleCategorical** (`flexible_categorical.py`): NaN injection (all probabilities
   0 in the released config → off); with p = `categorical_feature_p = 0.2` a dataset
   gets categorical columns, each column converted w.p. `p_col ~ U(0,1)` by
   `MulticlassRank(max(round(Gamma(1,10)), 2), ordered_p=0.3)`; `remove_outliers`
   (4σ log-clamp) and z-`normalize_data` on X and y; class assignment
   (`MulticlassRank`, boundaries sampled from the data, random label permutation /
   reversal); `normalize_by_used_features` (`x / (n_used / 100)`); zero padding to 100
   features; `check_is_compatible` (train/eval class sets equal, else relabel −100);
   `normalize_labels` (dense 0..C−1 + random rotation).

Resolved hyperparameters of the released model (`reload_config` + notebook overrides,
`evaluate_hypers` no longer draws anything prior-relevant after the overrides):

| Key | Value |
| --- | --- |
| `num_features` / `num_features_used` | 100 / `U_int(1, 100)` |
| `max_num_classes`, `num_classes` | 10, `U_int(2, 10)`; `MulticlassRank` then keeps 2 w.p. ½ else `U_int(2, K)` |
| `multiclass_type`, `output_multiclass_ordered_p`, `balanced` | `'rank'`, 0.0, False |
| `categorical_feature_p`; `nan_prob_*`; `set_value_to_nan` | 0.2; 0, 0, 0; 0.1 (unused) |
| `normalize_to_ranking`, `normalize_by_used_features`, `normalize_with_sqrt` | False, True, False |
| `sampling`, `pre_sample_causes`, `mix_activations`, `new_mlp_per_example` | `'normal'`, True, False, True |
| `prior_mlp_scale_weights_sqrt`, `random_feature_rotation`, `rotate_normalized_labels` | True, True, True |
| `normalize_labels`, `check_is_compatible` | True, True (set by `get_model` when `flexible`) |
| `differentiable_hps_as_style`, `emsize` | False, 512 (only feeds `DifferentiableHyperparameterList`) |
| `bptt` (seq_len) | 1152 in training; **we pass `seq_len = n_samples`** |

Differentiable ("meta") distributions (`get_diff_causal`, `get_diff_gp`,
`get_diff_prior_bag`; `get_diff_flex` entries were deleted by the overrides):

| Hyperparameter | Distribution |
| --- | --- |
| `num_layers` | meta_gamma(max_alpha 2, max_scale 3), round, ≥ 2 |
| `prior_mlp_hidden_dim` | meta_gamma(max_alpha 3, max_scale 100), round, ≥ 4 |
| `prior_mlp_dropout_prob` | meta_beta(scale 0.6, a,b ~ U(0.1, 5)) |
| `noise_std` | meta_trunc_norm_log_scaled(mean ∈ [1e-4, 0.3]) |
| `init_std` | meta_trunc_norm_log_scaled(mean ∈ [0.01, 10]) |
| `num_causes` | meta_gamma(max_alpha 3, max_scale 7), round, ≥ 2 |
| `is_causal`, `pre_sample_weights`, `y_is_effect`, `block_wise_dropout`, `sort_features`, `in_clique` | meta_choice(True, False) |
| `prior_mlp_activations` | meta_choice_mixed(Tanh, Identity, ReLU) — one activation per group |
| GP `outputscale`, `lengthscale` | meta_trunc_norm_log_scaled(mean ∈ [1e-5, 10]) |
| GP `noise` | meta_choice(1e-5, 1e-4, 1e-2) |
| `prior_bag_exp_weights_1` | U(2, 10) |

## Layout

    vendor/tabpfn1/                  # namespace port of tabpfn/priors + utils subset
      UPSTREAM.md PATCHES.md LICENSE
      priors/{__init__,prior,utils,mlp,fast_gp,flexible_categorical,
              differentiable_prior,prior_bag}.py
      utils.py                       # only the helpers listed above
      model_configs.py               # get_diff_* / get_prior_config_causal (HP space)
    lib/prior.py                     # source-agnostic core (unchanged API) + source registry
    lib/prior_sources/
      nanotabicl.py                  # today's generate_dataset body, moved verbatim
      tabpfn1.py                     # this design
      tabicl1.py                     # sibling design
    exp/tabpfn1-prior/{pilot-50,full-1000}/config.toml   # release specs (source = "tabpfn1")
    data/tabpfn1-prior/<release>/    # generated release (ignored by Git)
    exp/tabpack-tabpfn1/full/        # sweep.toml + bank.json (+ cells)
    dev/tests/test_prior_tabpfn1.py

## Multi-source release plumbing (shared with the TabICLv1 design)

- `ReleaseSpec.source: str = 'nanotabicl'`. Excluded from `spec_hash` when it holds the
  default so `full-1000`'s recorded `spec_hash` stays byte-stable. `make_release_spec`
  and `bin/generate_prior.py` pass it through; the release root becomes
  `data/<root>/<release>` with `root = {'nanotabicl': 'tabicl-prior', 'tabpfn1':
  'tabpfn1-prior', 'tabicl1': 'tabicl1-prior'}[source]` (the old name is kept for the
  old source; nothing on disk or in cluster scheduler moves).
- `lib/prior.generate_dataset` keeps the attempt loop, caps, acceptance check, array
  building, split and serialization. Only the "produce candidate tensors for this
  spec and attempt" step dispatches to `lib/prior_sources/<source>.sample(spec, seed)`.
  A source returns `x_num`, `x_cat` (int codes or None), `y` and a JSON-safe dict of
  realized prior hyperparameters that is stored in the manifest entry.
- `bin/upload_release.py`: `DATA_PREFIX` derives from `release_dir.parent.name`, so a
  tar unpacks to `data/tabpfn1-prior/<release>/…`; the launcher's default
  `mv $INPUT_PATH/data $PROJECT_DIR` needs no change.
- `lib/sweep.SweepSpec.release_root: None | str = None`; `get_release_dir` uses it when
  set; `sweep_hash` includes it **only when set** (old sweep hashes unchanged).
  `sweep_worker`, `submit_sweep`, `check` and `smoke_tabpack` go through
  `get_release_dir`, so they inherit it.
- `seed_global_rngs` additionally seeds Python's `random` (TabPFN and TabICL use it;
  NanoTabICL does not, so its releases are unaffected).

## Shape matching

`realize_dataset_spec(release, index)` is **not changed**: it still draws task type,
`n_samples`, `n_features`, `x_cat_sizes` and `n_classes` in that order from the
`(seed, index)` spec stream. With the same seed 0 and the same ranges as
`exp/tabicl-prior/full-1000/config.toml` (n_samples `[1024, 32768]`, n_features
`[4, 64]`, `max_classes` 10, task mix 0.4/0.3/0.3) every release therefore realizes the
same `(task, n_samples, n_features, n_classes)` for dataset `i`. Non-NanoTabICL sources
ignore `x_cat_sizes` (categorical structure is the prior's own business) but the draw is
kept so the downstream `n_classes` draw stays aligned. The manifest test asserts this
alignment against `data/tabicl-prior/full-1000/manifest.jsonl` for the pilot indices.

Consequences worth stating: the eval-heavy and widest datasets (`00980`, `00466`,
`00191`'s shape) exist in every release, so the GPU sizing from the smoke study carries
over, and per-dataset paired comparisons across sources are possible.

## Per-dataset generation (`lib/prior_sources/tabpfn1.py`)

Everything runs on CPU, one dataset per call, `batch_size = 1`, `seq_len = n_samples`,
`num_features = 100`, `num_features_used = n_features` (from the spec).

1. Seed `random`, NumPy, Torch from `(release.seed, index, attempt)`.
2. Build the hyperparameter dict = resolved table above + `DifferentiableHyperparameterList(get_diff_config()['differentiable_hyperparameters']).sample_parameter_object()`.
   Group size is 1: the released run shared one draw across 8 datasets, which only
   correlates datasets inside a training batch; the marginal per-dataset distribution is
   identical. Record the sampled values (activation as class name) in the manifest.
3. Prior bag choice exactly as `prior_bag.get_batch` (`torch.multinomial(softmax([1, w]))`).
4. Call the vendored `mlp.get_batch` or `fast_gp.get_batch` with
   `num_features = n_features` (that is what `FlexibleCategorical` passes as
   `args_passed['num_features']`).
5. Post-process with `FlexibleCategoricalRecording(FlexibleCategorical)`, a subclass in
   `lib/` that overrides `forward` to run the identical sequence of upstream calls
   (`drop_*` are skipped because all NaN probabilities are 0; categorical conversion
   via the vendored `MulticlassRank`; `remove_outliers`; `normalize_data`; class
   assigner; `normalize_by_used_features_f`; label normalization/rotation) while
   **recording which columns were converted to categorical and their integer codes**.
   Padding to 100 features and `check_is_compatible` are not applied (see below).
   A parity test runs the vendored `FlexibleCategorical.forward` and the subclass with
   the same seed and asserts identical `x[:, :, :n_features]` and `y`.
6. Task type from the spec, replacing TabPFN's per-batch class sampling:
   - `binclass` / `multiclass`: `MulticlassRank(K)` with `num_classes` forced to the
     spec's `K` (set the attribute after construction; `MulticlassRank.__init__`
     otherwise re-samples it via `class_sampler_f`), `ordered_p = 0`, then the upstream
     label normalization. Missing classes (coincident boundaries) → structural reject.
   - `regression`: skip the class assigner; `y` is the z-normalized continuous target
     after `normalize_data` (upstream's `RegressionNormalized` min-max is what
     `num_classes == 0` would do; z-scoring is the same up to an affine map and TabPack
     standardizes the target anyway). Constant `y` → structural reject.
7. Arrays: `x_num` = non-converted columns after upstream normalization (float32);
   `x_cat` = recorded integer codes (int64) of converted columns, **before**
   `remove_outliers`/`normalize_data` (the codes are what the column *is*; upstream
   z-scores them only because a transformer sees numbers). NaN in X/y (the MLP's
   `x[:] = 0, y[:] = -100` NaN guard) → structural reject.
8. `lib.prior.acceptance_check` (ExtraTrees OOB bootstrap p < 0.05), `_make_split`
   (0.64/0.16/0.2, stratified), serialize, manifest — unchanged.

Deviations from upstream, all deliberate and listed in `PATCHES.md`/manifest:

| Upstream | Here | Why |
| --- | --- | --- |
| seq_len 1152, `num_features_used ~ U(1,100)` | seq_len and width from the shared spec | Shape matching |
| `num_classes` sampled per batch, 50 % binary | task/K from the spec | Shape matching; task mix identical across sources |
| classification only | regression = continuous normalized target | 40 % of the study is regression |
| `check_is_compatible` on a train prefix | our stratified split | Split semantics differ; ours guarantees every class in train, ≥ 2 classes in val/test |
| categorical codes z-scored into the numeric matrix | exposed as `x_cat` | Real datasets carry this information; `cat_policy = 'ordinal'` path already exists in the sweep |
| zero-padding to 100 columns | none | The contract has no padding; `normalize_by_used_features` (a per-dataset constant scale) is kept and is invariant under noisy-quantile normalization anyway |
| one HP draw per 8 datasets | per dataset | Same marginal; removes batch-level correlation |

## Vendoring and dependencies

- Port rule as for `vendor/tabpack`: provenance header, relative imports
  (`from ..utils import …`), `default_device = 'cpu'`, plotting imports
  (`seaborn`, `matplotlib`) and `plot_*` helpers removed from `priors/utils.py`
  (they are dev-only dependencies and unused). No algorithmic edits. `UPSTREAM.md`
  records both repositories and the identity check; `PATCHES.md` records each change.
- **gpytorch** (pinned, e.g. `gpytorch==1.14`) as a `[project.optional-dependencies]
  tabpfn-prior` extra: needed only for the ≈1.6 % GP datasets and only on the
  generation host (cluster scheduler cubes never import the prior). Generation runs with
  `uv run --extra tabpfn-prior`.
- **ConfigSpace is not needed.** `evaluate_hypers` samples only `CSH` hyperparameters,
  and after the notebook overrides none of the prior-relevant ones remain (`lr`,
  `emsize`, … are transformer settings). The resolved table is frozen as a constant in
  `lib/prior_sources/tabpfn1.py`; an optional dev-only test with ConfigSpace installed
  checks it against `evaluate_hypers(config)`.
- Compatibility with torch 2.7.1 / Python 3.12.9 is expected (plain `nn` modules,
  `torch.multinomial`, `torch.Tensor()`); the pilot is the check.

## Cost and risk

- MLP branch: forward of `n_samples × hidden (≤ ~300) × layers (≤ ~8)`, then
  `remove_outliers`/`normalize_data` and the ExtraTrees filter (25 trees, depth 6, 1
  thread): comparable to NanoTabICL's measured 1.6 s/dataset mean (full-1000 took ~110 s
  wall with 16 workers).
- GP branch at `n = 32,768`: gpytorch's Lanczos root decomposition materializes kernel
  matmuls of size `n²` (4.3 GB in float32) and may retry on `torch.linalg.eigh` failures
  (upstream loop). With ~16 GP datasets total this is acceptable even at minutes each,
  but the pilot must measure peak RSS and time for a forced GP draw at the largest
  spec, and `n_workers` is set from that number.
- `RandomFunctionActivation` is *not* in TabPFN's activation set (Tanh/Identity/ReLU
  only), so no large transient tensors in the MLP branch.
- Acceptance rate is unknown for this prior (TabPFN never filtered). The 64-attempt /
  600 s caps stay; the pilot reports the reject rate, and `failed_dataset_ids` in
  `release.json` is the escape hatch if some specs cannot be met.

## Sweep and launch

    exp/tabpack-tabpfn1/full/sweep.toml
      seed = 0, n_configs = 1000, pack_size = 200
      release = "full-1000", release_root = "data/tabpfn1-prior"
      [space] / [trainer] copied verbatim from exp/tabpack-prior/full,
      except trainer.save_final_predictions = true

    python bin/plots/submit_sweep.py freeze exp/tabpack-tabpfn1/full \
        --bank-from exp/tabpack-prior/full          # bank_hash must equal 9f3681d9…
    python bin/plots/submit_sweep.py plan  exp/tabpack-tabpfn1/full --num-jobs 4 --task-minutes 1.5
    python bin/plots/submit_sweep.py emit  exp/tabpack-tabpfn1/full --num-jobs 4 \
        --jobs-per-instance 1 --priority low --n-cpu-cores 16 --memory-gb 64 \
        --disk-gb 100 --ttl-hours 12

Predictions are saved from the start (`full-pred` measured 17.5 GiB total, 4.4 GiB per
job for identical shapes) so one sweep serves both the metrics study and the T+E
portfolio study; no metrics-only sibling. Cells are consolidated into one file per
dataset as the sweep runs (see "Outputs and storage"). Resource shape, `--jobs-per-instance 1`,
low priority and the download/`check` steps are exactly those of
the synth-1000-launch LOG note (not included) (2026-08-11/12 entries). Cells: 5,000; ~4 h/job at
~1.5 min/task. Post-launch: `check`, then `bin/te_portfolio.py` on a new
`exp/tabpack-tabpfn1/te-portfolio` spec pointing at this sweep.

## Outputs and storage

Three requirements drive the output format: (1) a Caruana ensemble over the 1,000
configs needs val/test predictions **and** the true val/test labels; (2) meta-tuning
needs per-dataset, per-config scores; (3) file count and space are limited, so results
are compressed while the sweep runs, not after download.

What the current layout costs (measured on `exp/tabpack-prior/full` and `full-pred`):
per cell a 60 KB `config.toml` (it embeds all 200 member configs, redundant with
`bank.json`), a 250 KB `report.json` and, with predictions, one `predictions.npz`.
That is 10,002 tracked files and 1.5 GiB of JSON/TOML **in Git** per sweep, plus 5,000
prediction files (24 GiB unpacked). Two more sweeps in that layout is not acceptable.

**Unit of storage = dataset, not cell.** `bin/tabpack.py` still trains one pack per
cell and writes its per-cell artifact (unchanged; the compact `predictions.npz` already
holds fp16 val/test predictions, `y_val`, `y_test`, global `member_ids`, task type,
score name and `prediction_type`). The new step is in `bin/sweep_worker.py`: the
dataset-local queue already finishes all 5 packs of a dataset on one worker, so when
the last pack succeeds the worker **consolidates** the dataset and deletes the cell
directories:

    exp/tabpack-tabpfn1/full/<dataset>.npz        # one file per dataset, format_version 2
      config_ids        int32   [1000]           global config IDs, sorted
      pred_val/pred_test float16 [1000, n, C|1]  probabilities (classification) or values
      y_val/y_test      int64|float32 [n]        true labels in the dataset's own encoding
      score_train/val/test float32 [1000]        the trainer's per-config scores
      best_step, seconds int32/float32 [1000]
      task_type, score, prediction_type, dataset_id            scalars
      sweep_hash, bank_hash, release_hash, dataset_source_hash  scalars (provenance)
      n_packs_merged    int16                    must equal sweep.n_packs

Written with `np.savez_compressed` to a temp name and renamed (atomic); the pilot
measures the compression ratio on fp16 probabilities, and the trainer's raw
`predictions.npz` is never kept. Expected size ≈ the measured 17.5 GiB / 1,000 ≈ 18 MB
per dataset (worst dataset `00980` ≈ 230 MB), so a 250-dataset job ships ~4.4 GiB and
**250 files** instead of 3,750. Per-cell `config.toml` is not needed for provenance:
`make_cell_config` regenerates it deterministically from `sweep.toml` + `bank.json`.

**Scores table.** At the end of its slice the worker writes
`scores-job-<j>.npz` from the dataset files it produced; after download
`bin/sweep_scores.py` merges the jobs into `exp/tabpack-tabpfn1/full/scores.npz`:

    dataset_ids [K], config_ids [N], score_{train,val,test} float32 [K, N],
    best_step int32 [K, N], task_type/n_samples/n_features/n_classes per dataset
    (from the release manifest, for meta-features)

`K × N = 10⁶` entries ≈ 12 MB; it is rebuildable from the dataset files, so it is a
convenience, not a source of truth. Git tracks only `sweep.toml`, `bank.json` and a
small `summary.json` (completeness result, per-dataset best/mean val and test score,
≈ 1,000 rows). Prediction and score files are ignored by the existing `*.npz` rule and
live in the run archives.

**Plumbing changes** (one shared implementation, `lib/sweep_artifacts.py`):

- `consolidate_dataset(sweep_dir, dataset_id)`: validates the 5 packs with
  `cell_output_issue`, merges, writes the dataset file, removes the cell directories.
  Idempotent; a retried job skips datasets whose file exists with matching provenance
  and finishes only the missing packs of partially done datasets (those still have
  their cell directories).
- `check_sweep_completeness` accepts either layout per dataset (consolidated file with
  `n_packs_merged == n_packs` and matching provenance, or 5 valid cells) so the check
  works mid-sweep and after consolidation.
- `lib/te_portfolio.load_prediction_dataset` reads the dataset file; a one-off
  `bin/sweep_scores.py consolidate exp/tabpack-prior/full-pred` converts the existing
  NanoTabICL sweep to the same layout so the three sources are analyzed by identical
  code (its per-cell copies can then be deleted from the working tree; Git history
  keeps the old reports).
- `sweep_worker --parallel-packs` (TabArena mode) is unaffected: consolidation is only
  attached to the dataset-local queue.

cluster scheduler and RAM: the cube's output tar is its `exp/` workspace, so it now holds ~250
dataset files plus one scores file; merging one dataset touches at most ~250 MB of
fp16 arrays. Download and the resume-on-retry semantics are unchanged.

## Rejected alternatives

- **Vendoring the anonymous repository.** Identical prior code but no license file;
  the official `tabpfn_v1` branch is the same code with a stated license.
- **Re-implementing the GP with a plain-torch Cholesky.** Exact sampling would actually
  *differ* from upstream (Lanczos, rank 100) and costs O(n³) at 32k rows; keeping
  gpytorch preserves upstream behaviour for a 1.6 % branch.
- **Faithful TabPFN shapes (seq_len 1152, U(1,100) features, classification only).**
  Would confound "prior mechanism" with "dataset shape and task mix"; the shape-matched
  release isolates the mechanism and keeps GPU sizing valid.
- **Treating converted columns as numeric** (bit-faithful to what the transformer
  saw). Loses categorical structure that real benchmarks expose; retained as a
  release-spec switch only if the pilot shows the `x_cat` path misbehaves.
- **Keeping `check_is_compatible` with `single_eval_pos = n_train`.** It permutes rows
  and writes `-100` labels for a contiguous-prefix split we do not use.
- **Sampling `num_classes`/`num_features_used` upstream-style.** Breaks per-index
  alignment with `full-1000`.
- **Keeping the per-cell layout and compressing after download.** 15,000 files and
  1.5 GiB of Git-tracked JSON per sweep, and the download itself is the bottleneck
  (`wget` stalled at 3.8 GiB once already); consolidating in the worker costs nothing.
- **Storing predictions in float32 or with `--parallel-packs`.** Doubles the size for
  no gain in Caruana weights; pack parallelism breaks the dataset-local consolidation
  point and is only needed when a few datasets dominate makespan.

## Milestones

1. **M0** — multi-source plumbing: `ReleaseSpec.source`, `lib/prior_sources/`,
   `release_root`, upload prefix; NanoTabICL tests and `full-1000` hashes unchanged.
   Plus `lib/sweep_artifacts.py`: dataset consolidation in the worker, layout-agnostic
   completeness check, `bin/sweep_scores.py`, `te_portfolio` reader; tested by
   converting `exp/tabpack-prior/full-pred` and re-running the T+E study on it.
2. **M1** — vendor `vendor/tabpfn1` from `automl/TabPFN@bef0cf25`; import smoke under
   torch 2.7.1; `UPSTREAM.md`/`PATCHES.md`.
3. **M2** — `lib/prior_sources/tabpfn1.py` + `FlexibleCategoricalRecording`; tests:
   contract datasets, determinism, parity with the vendored `forward`, shape alignment
   with `full-1000`, forced GP draw at 32,768 rows (time and RSS).
4. **M3** — `exp/tabpfn1-prior/pilot-50`: reject rate, s/dataset, bytes,
   categorical share; adjust `n_workers`.
5. **M4** — `exp/tabpfn1-prior/full-1000`, upload (TTL ≥ 30 d), record the data ID in
   `release.json`; one-cell local smoke against the new release.
6. **M5** — freeze `exp/tabpack-tabpfn1/full` with the reused bank, emit, launch 4
   low-priority instances, download, `check`, log in `LOG.md`; then the T+E study.
