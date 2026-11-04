#!/usr/bin/env python3
"""Export website figures from pinned published artifacts; no fits or simulations.

Restore ignored inputs with `python3 scripts/fetch_research_sources.py`, then run
this script. All transformations are presentation aggregations of existing data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "source-artifacts"
UPSTREAM = SOURCES / "upstream"
HF = SOURCES / "huggingface"
OUT = ROOT / "data"
MODELS = {"lgbm": "LightGBM", "lgbm-prep": "LightGBM-TabPrep", "mlp": "MLP"}
KS = (25, 64, 200)
REGIMES = {"default": "D", "tuned": "T", "tuned_ensemble": "TE"}


def scalar(s):
    if s == "":
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        n = float(s)
        return (int(n) if n.is_integer() else n) if math.isfinite(n) else None
    except (TypeError, ValueError):
        return s


def read_csv(relative):
    with (UPSTREAM / "data" / relative).open(newline="") as f:
        return [{k: scalar(v) for k, v in r.items()} for r in csv.DictReader(f)]


def identity(row):
    bank = row["config_type"]
    if bank.startswith("LGBM-P"):
        family = "lgbm-prep"
    elif bank.startswith("LightGBM") or bank == "GBM":
        family = "lgbm"
    elif bank.startswith("MLP-Cos"):
        family = "mlp"
    elif bank.startswith("TabPack"):
        family = "tabpack"
    else:
        return row["family"].lower().replace(" ", "-"), "reference", row["family"]
    arm = "FTE" if "FTE" in bank else "FT" if "FT" in bank else "R"
    name = MODELS.get(family, "TabPack") + (" " + arm if arm != "R" else "")
    return family, arm, name


def elo_data():
    out = []
    for r in read_csv("main/elo_full_ba.csv"):
        family, arm, name = identity(r)
        regime = REGIMES[r["method_subtype"]]
        out.append({"method": r["method"], "label": name + " (" + ("T+E" if regime == "TE" else regime) + ")",
                    "name": name, "family": family, "arm": arm, "regime": regime,
                    "elo": r["elo"], "ciLow": r["elo"] - r["elo-"], "ciHigh": r["elo"] + r["elo+"],
                    "datasetCount": r["n"], "isSelected": arm in ("FT", "FTE"), "group": r["group"]})
    assert len(out) == 58 and all(x["datasetCount"] == 142 for x in out)
    return out


def latency_data(elo):
    # Mapping and aggregation reproduce bin/figures/elo_latency.py, without its visual dodge.
    measured = {("lgbm", "R"): "lightgbm", ("lgbm", "FT"): "lightgbm-ft", ("lgbm", "FTE"): "lightgbm-fte",
                ("lgbm-prep", "R"): "lgbm-prep-r", ("lgbm-prep", "FT"): "lgbm-prep-ft", ("lgbm-prep", "FTE"): "lgbm-prep-fte",
                ("mlp", "R"): "mlp-cos-r", ("mlp", "FT"): "mlp-cos-ft", ("mlp", "FTE"): "mlp-cos-fte",
                ("tabpack", "R"): "tabpack-r64", ("tabpack", "FTE"): "tabpack-fte64"}
    refs = {"catboost": "catboost", "realmlp": "realmlp", "tabm": "tabm", "tabpfn-3": "tabpfn3", "tabiclv2": "tabicl2",
            "kumo-tabular": "kumo-tabular", "kumo-tabular-medium": "kumo-tabular-medium", "kumo-tabular-small": "kumo-tabular-small"}
    fm = {"tabpfn3", "tabicl2", "kumo-tabular", "kumo-tabular-medium", "kumo-tabular-small"}
    modes = {"public": "recompute", "cache_gpu": "cache_gpu", "cache_cpu": "cache_cpu"}
    groups = defaultdict(list)
    for r in read_csv("main/latency.csv"):
        is_fm = r["family"] in fm
        if r["family"] not in set(measured.values()) | set(refs.values()):
            continue
        if r["source"] != ("ba-real-icl" if is_fm else "ba-real-r1"):
            continue
        if not is_fm and r["variant"] != "public":
            continue
        kind = "D" if r["config_name"] in ("default", "pack") else "TE" if r["config_name"].startswith("te") else "T"
        mode = modes[r["variant"]] if is_fm else ""
        groups[(r["family"], kind, mode)].append(r)
    out = []
    for r in elo:
        m = measured.get((r["family"], r["arm"])) or refs.get(r["family"])
        if not m:
            continue
        if r["regime"] == "D" and r["family"] != "tabpack" and r["arm"] in ("FT", "FTE"):
            continue
        for mode in (("recompute", "cache_gpu", "cache_cpu") if m in fm else ("",)):
            rows = groups.get((m, r["regime"], mode), [])
            if not rows:
                continue
            out.append({**r, "mode": mode, "latencyMs": statistics.median(x["median_ms"] for x in rows),
                        "latencyDatasetCount": len({x["dataset"] for x in rows}), "device": rows[0]["device"],
                        "hardware": rows[0]["gpu_name"] or "CPU, 16 threads", "latencySource": rows[0]["source"]})
    assert all(x["latencyDatasetCount"] == 10 for x in out)
    return out


def random_data():
    ladder = read_csv("ablations/ladder.csv")
    draws = read_csv("ablations/random_draws.csv")
    climbed = read_csv("ablations/climbed.csv")
    real = read_csv("ablations/tabrepo_ladder.csv")
    out = []
    for model in MODELS:
        for k in KS:
            selected = next(x for x in climbed if x["model"] == model and x["k"] == k)
            real_corner = [x for x in real if x["model"] == model and x["k"] == k and x["width"] == 4000 and x["n"] == 182 and x["c"] == 4000]
            assert len(real_corner) == 1, (model, k, "expected one full TabRepo-182 × 4,000 corner")
            selected_real = {key: real_corner[0][key] for key in ("winrate", "ci95", "n", "c")}
            pool = [{"kprime": x["kprime"], "winrate": x["pool_winrate"], "ci95": x["pool_ci95"]} for x in ladder if x["model"] == model and x["k"] == k]
            samples = [{"kprime": x["kprime"], "mean": x["mean"], "lo": x["p2_5"], "hi": x["p97_5"], "min": x["min"], "max": x["max"], "shareAboveSelected": x["share_ge_meta_tuned"], "nDraws": x["n_draws"]} for x in draws if x["model"] == model and x["k"] == k]
            assert next(x["winrate"] for x in pool if x["kprime"] == k) == 0.5
            assert max(x["kprime"] for x in pool) == 4000
            out.append({"model": model, "k": k, "selected": {"winrate": selected["winrate"], "ci95": selected["ci95"]},
                        "selectedReal": selected_real,
                        "pool": sorted(pool, key=lambda x: x["kprime"]), "draws": sorted(samples, key=lambda x: x["kprime"])})
    return out


def budget_cells():
    """The original paper's scatter points, including the synthetic N<K extension.

    Synthetic budget.csv does not retain separate C and N, so these fields remain
    null. Real cells use exactly the width=16,000, C>K filter in budget_series().
    """
    synthetic = [{"model": r["model"], "k": r["k"], "source": "synthetic", "budget": r["budget"],
                  "winrate": r["winrate"], "n": None, "c": None} for r in read_csv("ablations/budget.csv")]
    real = [{"model": r["model"], "k": r["k"], "source": "tabrepo", "budget": r["n"] * r["c"],
             "winrate": r["winrate"], "n": r["n"], "c": r["c"]} for r in read_csv("ablations/tabrepo_ladder.csv")
            if r["width"] == 16000 and r["c"] > r["k"]]
    return synthetic + real


def budget_data():
    best = {}
    for r in budget_cells():
        exponent = round(math.log2(r["budget"]))
        key = r["model"], r["k"], r["source"], exponent
        if key not in best or r["winrate"] > best[key]["winrate"]:
            best[key] = {"model": r["model"], "k": r["k"], "source": r["source"], "budget": 2 ** exponent,
                         "sourceBudget": r["budget"], "winrate": r["winrate"], "binExponent": exponent,
                         "c": r.get("c"), "n": r.get("n")}
    return sorted(best.values(), key=lambda r: (r["model"], r["k"], r["source"], r["budget"]))


LGBM_HP = [("learning_rate", "Learning rate", "log", [0.005, 0.1]), ("feature_fraction", "Feature fraction", "lin", [0.4, 1]),
           ("bagging_fraction", "Bagging fraction", "lin", [0.7, 1]), ("num_leaves", "Leaves", "log", [2, 200]),
           ("min_data_in_leaf", "Minimum data in a leaf", "log", [1, 64]), ("extra_trees", "Extra trees", "cat", [False, True]),
           ("min_data_per_group", "Minimum data per group", "log", [2, 100]), ("cat_l2", "Categorical L2", "log", [0.005, 2]),
           ("cat_smooth", "Categorical smoothing", "log", [0.001, 100]), ("max_cat_to_onehot", "Maximum category one-hot", "log", [8, 100]),
           ("lambda_l1", "L1 regularization", "lin", [0.0001, 1]), ("lambda_l2", "L2 regularization", "lin", [0.0001, 2])]
HP = {"lgbm": LGBM_HP, "lgbm-prep": LGBM_HP + [
    ("tabprep_enabled", "TabPrep enabled", "cat", [False, True]), ("oafe_max_feats", "OpenFE maximum features", "prep", [2000, 1000]),
    ("cat_combine_max_feats", "Categorical combination features", "prep", [100, 500]), ("rsfc_n_subsets", "Random subsets", "prep", [50, 1, 100]),
    ("groupby_max_feats", "Group-by features", "prep", [500, 100, 1000])],
    "mlp": [("lr", "Learning rate (AdamW)", "log", [0.0001, 0.005]), ("weight_decay", "Weight decay", "log", [0.001, 1]),
            ("muon_lr", "Learning rate (Muon)", "log", [0.001, 0.1]), ("init_scale", "Embedding initialization scale", "log", [0.01, 10]),
            ("dropout", "Dropout rate", "dropout", [0, 0.5]), ("d_embedding", "Embedding dimension", "cat", list(range(8, 33, 4))),
            ("n_blocks", "Blocks", "cat", [1, 2, 3, 4])]}
MAIN_HP = {"lgbm": ["learning_rate", "min_data_in_leaf", "lambda_l1"], "lgbm-prep": ["learning_rate", "min_data_in_leaf", "lambda_l1", "tabprep_enabled"], "mlp": ["muon_lr", "lr", "d_embedding"]}


def hp_flat(model, config):
    if model == "mlp":
        o, m = config["optimizer"], config["model"]
        return {**o, "init_scale": m["num_embeddings"]["init_scale"], "dropout": m["dropout"],
                "d_embedding": m["num_embeddings"]["d_embedding"], "n_blocks": m["n_blocks"]}
    out = dict(config["lightgbm"])
    if model == "lgbm-prep":
        out["tabprep_enabled"] = config["tabprep"] is not None
        for key in ("oafe_max_feats", "cat_combine_max_feats", "rsfc_n_subsets", "groupby_max_feats"):
            out[key] = config["tabprep"][key] if out["tabprep_enabled"] else None
    return out


def histogram(values, kind, domain, key):
    if kind in ("cat", "prep"):
        probs = [0.75, 0.25] if key == "tabprep_enabled" else [1 / len(domain)] * len(domain)
        return [{"label": str(o).lower(), "value": o, "prior": p, "selected": values.count(o) / len(values), "count": values.count(o)} for o, p in zip(domain, probs)]
    lo, hi = domain
    edges = [lo * (hi / lo) ** (j / 12) if kind == "log" else lo + (hi - lo) * j / 12 for j in range(13)]
    counts = [0] * 12
    atom = values.count(0) if kind == "dropout" else 0
    for v in values:
        if kind == "dropout" and v == 0:
            continue
        u = math.log(v / lo) / math.log(hi / lo) if kind == "log" else (v - lo) / (hi - lo)
        counts[min(11, max(0, int(u * 12)))] += 1
    bins = [{"lo": edges[i], "hi": edges[i + 1], "prior": (0.5 if kind == "dropout" else 1) / 12,
             "selected": counts[i] / len(values), "count": counts[i]} for i in range(12)]
    if kind == "dropout":
        bins.insert(0, {"label": "Exactly zero", "value": 0, "prior": 0.5, "selected": atom / len(values), "count": atom})
    return bins


def density_samples(values, kind, domain):
    """Reproduce the paper's unit()/reflected_kde()/draw_hp() using stdlib math.

    This is density in the sampling-CDF coordinate, not density per raw HP unit.
    Reflection is evaluated at u, -u and 2-u exactly as in the paper source.
    The dropout atom is separate; its positive component is scaled relative to
    the prior's 0.5 continuous mass, matching draw_hp(). No density is invented
    for categorical choices or a degenerate continuous sample.
    """
    if kind not in ("lin", "log", "dropout"):
        return None
    lo, hi = domain
    atoms = []
    if kind == "dropout":
        positive = [v for v in values if v > 0]
        unit = [v / hi for v in positive]
        selected_mass = len(positive) / len(values)
        prior_mass = 0.5
        atoms = [{"x": 0, "prior": 0.5, "selected": values.count(0) / len(values)}]
    else:
        unit = [math.log10(v / lo) / math.log10(hi / lo) if kind == "log" else (v - lo) / (hi - lo) for v in values]
        selected_mass = prior_mass = 1.0
    if len(unit) < 2 or statistics.stdev(unit) == 0:
        raise ValueError("Cannot reproduce a continuous KDE for a degenerate sample")
    bandwidth = statistics.stdev(unit) * len(unit) ** (-1 / 5)
    normalizer = len(unit) * bandwidth * math.sqrt(2 * math.pi)
    mass_ratio = selected_mass / prior_mass

    def kernel(x):
        return math.fsum(math.exp(-0.5 * ((x - v) / bandwidth) ** 2) for v in unit) / normalizer

    points = []
    for i in range(300):
        u = i / 299
        x = lo * (hi / lo) ** u if kind == "log" else lo + u * (hi - lo)
        ratio = (kernel(u) + kernel(-u) + kernel(2 - u)) * mass_ratio
        points.append({"x": x, "u": u, "ratio": ratio})
    integral = math.fsum((a["ratio"] + b["ratio"]) * (b["u"] - a["u"]) / 2 for a, b in zip(points, points[1:]))
    ordered = sorted(unit)
    ks = max(max((i + 1) / len(unit) - u, u - i / len(unit)) for i, u in enumerate(ordered))
    return {"points": points, "domain": list(domain), "xScale": "log" if kind == "log" else "linear",
            "coordinate": "sampling-cdf", "bandwidth": bandwidth, "bandwidthRule": "Scott: sample standard deviation × n^(−1/5)",
            "nKernelSamples": len(unit), "reflectionBounds": [0, 1], "referenceRatio": 1,
            "selectedContinuousMass": selected_mass, "priorContinuousMass": prior_mass,
            "ksDistance": ks if kind != "dropout" else None, "integral": integral, "expectedIntegral": mass_ratio,
            "atoms": atoms, "source": "bin/figures/ablations.py:unit,reflected_kde,draw_hp"}


def hp_data():
    banks = {name: [json.loads(line) for line in (HF / "configs" / (name + "-4000.jsonl")).read_text().splitlines() if line] for name in ("lgbm", "mlp")}
    out = []
    for model, specs in HP.items():
        portfolio = json.loads((HF / "portfolios" / model / "T_K200.json").read_text())
        bank = banks["mlp" if model == "mlp" else "lgbm"]
        ids = portfolio["config_ids"]
        assert len(ids) == len(set(ids)) == 200
        rows = [hp_flat(model, bank[i]) for i in ids]
        for key, label, kind, domain in specs:
            values = [r[key] for r in rows if r[key] is not None]
            bins = histogram(values, kind, domain, key)
            assert sum(x["count"] for x in bins) == len(values)
            out.append({"model": model, "k": 200, "key": key, "label": label, "kind": kind, "domain": domain,
                        "selected": values, "nSelected": len(values), "conditional": kind == "prep", "bins": bins,
                        "main": key in MAIN_HP[model], "selectionProtocol": portfolio["protocol"],
                        "density": density_samples(values, kind, domain)})
    return out


def main():
    if not (UPSTREAM / "REVISION.json").exists():
        raise SystemExit("Restore sources first: python3 scripts/fetch_research_sources.py")
    rev = json.loads((UPSTREAM / "REVISION.json").read_text())
    hf_rev = json.loads((HF / "REVISION.json").read_text())
    hashes = {str(p.relative_to(SOURCES)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES.rglob("*")) if p.is_file() and p.name != "REVISION.json" and p.relative_to(SOURCES).parts[0] in ("upstream", "huggingface")}
    elo = elo_data()
    surfaces = [r for r in read_csv("ablations/surfaces.csv") if r["study"] == "tuned"]
    for row in surfaces:
        if row["c"] == row["k"]:
            assert row["winrate"] == 0.5
    # Rounded values transcribed from the repository README's explicitly labelled direct comparison.
    direct = {25: [0.61, 0.55, 0.51], 64: [0.71, 0.68, 0.58], 200: [0.59, 0.62, 0.62]}
    data = {"meta": {"title": "PriorTune", "paper": "https://arxiv.org/abs/2611.05141v1", "repo": rev["repo"], "commit": rev["commit"],
                     "hfRepo": hf_rev["repo"], "hfCommit": hf_rev["commit"], "sourceHashes": hashes, "license": "Apache-2.0; data/SOURCE-LICENSE.txt",
                     "protocolClarifications": {"randomSearch": {"source": "User confirmation of paper Figure 6, 2026-11-04", "statement": "The displayed random-search comparison uses 1-fold T. It does not join eight validation folds or average eight test predictions. The source CSV numbers are unchanged."}},
                     "protocols": {
                         "elo": "Full BeyondArena: 142 datasets, 497 tasks, 58-method field. Elo remains as published when filtering the display. D, T and T+E are distinct deployment regimes; FT/FTE name the objective used to select the portfolio. K=25, except TabPack K=64.",
                         "latency": "Separate single-row inference measurements on 10 BeyondArena datasets: median of per-dataset median milliseconds, native serving paths. CPU trees use 16 threads; neural and in-context methods use an H100. KV cache modes are separate measurements. Timing and Elo have different evaluation scopes.",
                         "randomSearch": "TabArena-51, 1-fold T deployment: select one configuration on validation and score on test. Published split metadata: outer repeat 0 / fold 0; validation = inner fold 0. All curves and horizontal references use a common random-K baseline. Synthetic reference: selected at C=N=4,000. Real-data reference: the exact TabRepo N=182, C=4,000 corner, not the best budget-envelope point. Pool curve: subsets from the original 4,000 bank. Draw curve: 200 subsets from a 16,000 bank; its 2.5–97.5% band is variation across draws, not a confidence interval. Neither curve is a direct selected-K versus random-K′ win rate.",
                         "budget": "T deployment on TabArena. Envelope follows the original plotting code: maximum observed win rate in each rounded log2(C×N) budget bin, independently per bin. This is not a trajectory from a single selected run. Synthetic source uses up to C=N=4,000; TabRepo source uses up to 182 real datasets and 16,000 configurations, with C>K.",
                         "surfaces": "Tuned study only: TabICLv2 prior, C×N up to 4,000×4,000; T deployment evaluated on TabArena-51. Each cell uses its selected K-subset against random K. C=K is 0.5 by construction. ci95 is the published 95% half-width.",
                         "hyperparameters": "Exact K=200, T-selected portfolio configurations from the pinned Hugging Face release. Smooth curves reproduce the paper's Gaussian KDE with Scott bandwidth, reflected at both bounds in the sampling-CDF coordinate, evaluated at 300 points. Their y-axis is selected density divided by sampling density; the sampling-prior reference is 1. Native hyperparameter values are shown on a logarithmic axis for log-uniform parameters. The original 12-bin empirical histograms remain in the downloadable data. Categorical choices retain probability masses. Dropout's zero atom is separate from its continuous component. TabPrep-only parameters condition on TabPrep being enabled.",
                     }},
            "models": [{"id": k, "label": v} for k, v in MODELS.items()], "ks": list(KS),
            "elo": elo, "latency": latency_data(elo), "randomSearch": random_data(), "budget": budget_data(), "budgetCells": budget_cells(), "surfaces": surfaces,
            "headToHead4000": [{"model": model, "k": k, "randomK": 4000, "winrate": vals[i], "rounded": True, "source": "README.md"} for k, vals in direct.items() for i, model in enumerate(MODELS)],
            "hyperparameters": hp_data()}
    OUT.mkdir(exist_ok=True)
    (OUT / "research.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    shutil.copyfile(UPSTREAM / "LICENSE", OUT / "SOURCE-LICENSE.txt")
    for path in (UPSTREAM / "data").rglob("*.csv"):
        if path.name in ("full_ba_per_task.csv", "groups.csv", "elo_groups.csv", "arms.csv"):
            continue
        target = OUT / "downloads" / path.relative_to(UPSTREAM / "data")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(f"Exported {len(elo)} Elo methods, {len(data['latency'])} latency points, 9 random-search comparisons, {len(data['budget'])} envelope points, {len(data['budgetCells'])} budget cells, {len(surfaces)} grid cells, {len(data['hyperparameters'])} HP distributions.")


if __name__ == "__main__":
    main()
