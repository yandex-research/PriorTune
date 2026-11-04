# What is in data/

These CSV files are the inputs of the paper's figures. Model ids follow the dataset
(`lgbm`, `lgbm-prep`, `mlp`). A win rate compares a selected subset with a random subset of
the same size on TabArena (ties count as half), averaged over repeated random orders.
`ci95` is its 95% confidence half-width.

## ablations/ (Section 5)

| file | content | how to recompute it |
|---|---|---|
| `surfaces.csv` | Win rate for every number of configurations C and synthetic datasets N. Study `tuned` uses the 4,000 x 4,000 pools. Study `priors` uses 2,000 configurations on 2,000 datasets from each of three priors (TabICLv2, TabICLv1, TabPFNv1). | `tuned`: `grid.py surfaces`. `priors`: the same command on datasets from `bin/priors/` |
| `climbed.csv` | Win rate at the full pool (C = N = 4,000). | `grid.py surfaces` |
| `ladder.csv` | Random K' configurations against random K. | `grid.py ladder` |
| `budget.csv` | The `tuned` win rates by budget B = C x N. | `grid.py surfaces`, with the dataset ladder extended below K |
| `random_draws.csv` | Random K' configurations drawn from a larger bank of 16,000. | needs the 16,000-config bank, which is not in the dataset |
| `tabrepo_ladder.csv` | Selection on TabRepo (182 datasets that do not overlap TabArena) instead of synthetic data. | needs the TabRepo caches of the 16,000-config bank, which are not in the dataset |

## selection/ (selection rule)

`arms.csv` compares variants of the selection rule. Each run selects on 3,000 synthetic
datasets and is scored on the other 1,000 against fresh random subsets. The variants are
PriorTune with R = 16, 8, 4, 2 or 1 bootstraps, PriorTune without validation bootstraps,
PriorTune with a single reference, plain greedy selection, and two baselines (TabRepo's
greedy selection and a greedy count of datasets won).

## main/ (Section 4 and Figure 1)

These come from deploying the selected subsets (K = 25, and 64 for TabPack) and random
subsets of the same size on BeyondArena.

| file | content |
|---|---|
| `elo_full_ba.csv` | Elo on the full BeyondArena (142 datasets, 58 methods) |
| `elo_groups.csv` | the same Elo computed on each group of datasets |
| `groups.csv` | which datasets belong to which group |
| `full_ba_per_task.csv` | the error of every method on every task |
| `latency.csv` | inference latency for a single row, on 10 BeyondArena datasets |
