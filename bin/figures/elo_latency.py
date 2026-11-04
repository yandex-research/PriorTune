"""Paper figure: BeyondArena Elo (left) next to Elo vs batch-1 inference latency (right).

Elo comes from the full BeyondArena core (142 datasets, 497 tasks; data/main/elo_full_ba.csv).
Full-BA bank labels are mapped onto the names below
(``FULL_BA_BANKS``). Both panels show the same methods: a method without a latency measurement
(TabPFN-3.5, TabPFN-3.5-Fast, Causilo, XGBoost, RandomForest, ExtraTrees, Linear) is left out of
both. Its Elo stays as fitted on the full 58-method field; nothing is refitted.

    python bin/figures/elo_latency.py

(with pandas < 3: pandas 3's default string dtype turns the ``None`` family of reference rows into
``NaN`` and breaks ``display_name``; requirements.txt pins it).

Writes ``--out`` (default ``figures/elo-latency``) as pdf + png. Latency input: data/main/latency.csv;
every row is taken from exactly one (file, source) pair in ``LATENCY_INPUTS``, all other rows are ignored:

  latency.csv                    the real-BeyondArena timing on one host: 10 BeyondArena datasets,
                                 one fixed outer/inner split,
                                 T / T+E selected on that split, models called natively (no AutoGluon).
                                 Source ``ba-real-r1``: GBDTs (16 CPU threads; LightGBM-TabPrep through the
                                 compiled TabPrep path), DNNs and TabPack (one H100, exact fast quantile step).
                                 Source ``ba-real-icl``: TabPFN-3, TabICLv2 and Kumo-Tabular large / medium /
                                 small on one H100, each in three serving modes (``variant``): ``public`` =
                                 recompute, ``cache_gpu`` / ``cache_cpu`` = KV cache on the GPU / CPU.

Every GPU point comes from these two sources (asserted in ``latency_table``).

Left panel: the official TabArena "tuning impact" bars (Default / Tuned / Tuned + Ensembled drawn
over each other from Elo 800, error bars in the darker shade, hatch = partially imputed), redrawn
on a shared axis because the official function owns its figure. Pairing follows the TabPrep paper:
each improved family's label is coloured, its baseline in regular weight and our filtered
portfolios (FT / FTE) in bold. Baselines: the published LightGBM for pure LightGBM, and the
random 25-config portfolio (64 for TabPack) for LGBM-P (shown as LightGBM-TabPrep), MLP-dagger (shown as MLP) and
TabPack. Rows are sorted
by their best Elo. ``EXCLUDED_BANKS`` (TabPack FT64) is left out of both panels; it stays in the
field the Elo was fitted on.

Right panel: T and T+E Elo against measured latency (median over 10 datasets, ms per row), plus
the default of TabPack and of the in-context models (no tuning). The D points of tunable methods
are left out: they sit 150-300 Elo lower and would compress everything else. Colour = our family
(key top left; grey = published reference), fill = portfolio (hollow = baseline, half = FT, full =
FTE), shape = regime, diamond = in-context model, dashed = Pareto front over every point drawn
(cache modes included). Our points carry no names -- colour and fill identify them; the published
references are named. LGBM-P is plotted at its own measured (compiled) latency, TabPrep feature
generation included; ``--lgbm-prep-as-lightgbm`` substitutes LightGBM's latency instead.

The in-context models are drawn in their three serving modes, at one Elo (``--fm-style``; the
paper uses ``shade``):
  shape   diamond fill = mode (hollow recompute, full GPU cache, right half CPU cache), joined by a line
  shade   diamond shade = mode (light recompute, medium CPU cache, dark GPU cache), joined by a line
  range   one diamond at the library default mode (TabPFN-3, TabICLv2: recompute; Kumo: CPU cache, the
          TabArena serving default) on a bar from the fastest to the slowest mode, ticks at the
          other modes; only the default-mode points enter the Pareto front
  arrow   arrow from recompute (hollow) to GPU cache (filled), CPU cache as a small hollow dot at its
          measured latency (on the arrow, or inside the GPU diamond when the two coincide)
A near-coincident FT / FTE pair, or two near-coincident modes of one model (shape, shade), are
dodged apart along x by a few points (``dodge``); only the slower point ever moves right, and the
front uses measured latencies. The figure height follows from the right panel's aspect ratio
(``--aspect``, default 1.7).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parents[1]
#: batch-1 latency rows used by the figure (see the module docstring)
LATENCY_DIR = PAPER_DIR / 'data/main'
DNN_FAMILIES = {'realmlp', 'tabm', 'mlp-cos-r', 'mlp-cos-ft', 'mlp-cos-fte', 'tabpack-r64', 'tabpack-fte64'}
FM_FAMILIES = {'tabpfn3', 'tabicl2', 'kumo-tabular', 'kumo-tabular-medium', 'kumo-tabular-small'}
GBDT_FAMILIES = {'lightgbm', 'lightgbm-ft', 'lightgbm-fte', 'lgbm-prep-r', 'lgbm-prep-ft', 'lgbm-prep-fte'}
#: (file, source run, variants) -> latency families taken from it; every other row is ignored
#: real BeyondArena timing: 10 datasets, one host, native paths
LATENCY_INPUTS = {
    ('latency.csv', 'ba-real-r1', ('public',)): DNN_FAMILIES | GBDT_FAMILIES | {'catboost'},
    ('latency.csv', 'ba-real-icl', ('public', 'cache_gpu', 'cache_cpu')): FM_FAMILIES,
}
#: the only runs a GPU (cuda) row may come from
GPU_SOURCES = {'ba-real-r1', 'ba-real-icl'}
#: serving mode of an in-context model, by ``variant`` of the fm run
MODE_OF = {'public': 'recompute', 'cache_gpu': 'cache_gpu', 'cache_cpu': 'cache_cpu'}
MODES = ('recompute', 'cache_gpu', 'cache_cpu')
MODE_LABEL = {'recompute': 'recompute', 'cache_gpu': 'KV cache, GPU', 'cache_cpu': 'KV cache, CPU'}
FM_STYLES = ('shape', 'shade', 'range', 'arrow')
#: Elo: the full BeyondArena core (142 datasets, 497 tasks)
FULL_BA_ELO = PAPER_DIR / 'data/main/elo_full_ba.csv'

# ---- identities -------------------------------------------------------------------------------
MLP_TEX = r'$\mathregular{MLP}^{\dagger}$'
#: family -> label / marker colour (validated together, dataviz validate_palette.js, light surface)
FAMILY_COLOR = {'LightGBM': '#2a78d6', 'LGBM-P': '#008300', 'MLP†': '#4a3aa7', 'TabPack': '#c2185b'}
#: family -> label as displayed (paper terminology): LGBM-P reads LightGBM-TabPrep, MLP-dagger reads MLP
FAMILY_TEX = {'LightGBM': 'LightGBM', 'LGBM-P': 'LightGBM-TabPrep', 'MLP†': 'MLP', 'TabPack': 'TabPack'}
FAMILY_ORDER = ['LGBM-P', 'LightGBM', 'MLP†', 'TabPack']
#: leaderboard bank (the part of ``method`` before " (regime)") -> (family, arm)
BANKS = {
    'GBM': ('LightGBM', 'R'), 'LGBM FT': ('LightGBM', 'FT'), 'LGBM FTE': ('LightGBM', 'FTE'),
    'LGBM-P': ('LGBM-P', 'R'), 'LGBM-P FT': ('LGBM-P', 'FT'), 'LGBM-P FTE': ('LGBM-P', 'FTE'),
    MLP_TEX: ('MLP†', 'R'), f'{MLP_TEX} FT': ('MLP†', 'FT'), 'MLP-COS-F25TE': ('MLP†', 'FTE'),
    'TabPack R64': ('TabPack', 'R'), 'TabPack FT64': ('TabPack', 'FT'), 'TabPack FTE64': ('TabPack', 'FTE'),
}
REFERENCE_NAMES = {
    'LR': 'Linear', 'RF': 'RandomForest', 'XT': 'ExtraTrees', 'XGB': 'XGBoost', 'CAT': 'CatBoost',
    'TA-TABM': 'TabM', 'TA-REALMLP': 'RealMLP', 'TA-TABICLv2': 'TabICLv2', 'TA-TABPFN-3': 'TabPFN-3',
    'TA-TABPFN-2.6': 'TabPFN-2.6', 'TA-TABDPT': 'TabDPT',
    'TA-TABPFN-3.5': 'TabPFN-3.5', 'TA-TABPFN-3.5-FAST': 'TabPFN-3.5-Fast', 'TA-CAUSILO': 'Causilo',
    'TA-KUMO-TABULAR': 'Kumo-Tabular', 'TA-KUMO-TABULAR-MEDIUM': 'Kumo-Tabular-Medium',
    'TA-KUMO-TABULAR-SMALL': 'Kumo-Tabular-Small',
}
#: full-BeyondArena leaderboard (render_full_ba_arms.py) bank labels -> the bank names above
FULL_BA_BANKS = {
    'LGBM-P R25·25%': 'LGBM-P', 'LGBM-P FT-25': 'LGBM-P FT', 'LGBM-P FTE-25': 'LGBM-P FTE',
    'LightGBM FT-25': 'LGBM FT', 'LightGBM FTE-25': 'LGBM FTE',
    'MLP-Cos R25': MLP_TEX, 'MLP-Cos FT-25': f'{MLP_TEX} FT', 'MLP-Cos FTE-25': 'MLP-COS-F25TE',
}
REGIMES = ('default', 'tuned', 'tuned_ensemble')
#: banks left out of both panels (still part of the field the Elo was fitted on)
EXCLUDED_BANKS = {'TabPack FT64'}
REGIME_OF = {'default': 'default', 'tuned': 'tuned', 'tuned + ensemble': 'tuned_ensemble'}

# ---- the official TabArena look --------------------------------------------------------------
PASTEL = sns.color_palette('pastel').as_hex()
DEEP = sns.color_palette('deep').as_hex()
BAR_COLOR = {'tuned_ensemble': PASTEL[2], 'tuned': PASTEL[1], 'default': PASTEL[0]}
ERR_COLOR = {'tuned_ensemble': DEEP[2], 'tuned': DEEP[1], 'default': DEEP[0]}
ERR_WIDTH = {'tuned_ensemble': 0.9, 'tuned': 0.75, 'default': 0.6}
BAR_LABEL = {'default': 'Default', 'tuned': 'Tuned', 'tuned_ensemble': 'Tuned + Ensembled'}
ELO_MIN = 800
INK, INK_2, GREY = '#1a1a1a', '#52514e', '#a3a29d'


def display_name(family: str | None, arm: str | None, bank: str) -> str:
    if family is None:
        return REFERENCE_NAMES.get(bank, bank)
    if family == 'LightGBM' and arm == 'R':
        return 'LightGBM'
    head = FAMILY_TEX[family]
    if arm == 'R':
        return head
    return f'{head} {arm}'


def load_leaderboard(path: Path) -> pd.DataFrame:
    lb = pd.read_csv(path)
    parts = lb['method'].str.extract(r'^(?P<bank>.*?)\s*\((?P<regime>default|tuned|tuned \+ ensemble)\)$')
    lb['bank'] = parts['bank'].replace(FULL_BA_BANKS)
    if 'imputed' not in lb:   # the full-BA leaderboard is built only from complete arms (no imputation)
        lb['imputed'] = 0.0
    lb['regime'] = parts['regime'].map(REGIME_OF)
    lb['family'] = lb['bank'].map(lambda b: BANKS.get(b, (None, None))[0])
    lb['arm'] = lb['bank'].map(lambda b: BANKS.get(b, (None, None))[1])
    lb['name'] = [display_name(f, a, b) for f, a, b in zip(lb['family'], lb['arm'], lb['bank'])]
    return lb


def wide(lb: pd.DataFrame) -> pd.DataFrame:
    """One row per method name: Elo and CI per regime, family, arm, imputed flag."""
    rows = []
    for name, g in lb.groupby('name', sort=False):
        r = {'name': name, 'family': g['family'].iloc[0], 'arm': g['arm'].iloc[0],
             'imputed': bool((g['imputed'] > 0).any())}
        for _, x in g.iterrows():
            r[x['regime']] = x['elo']
            r[x['regime'] + '+'] = x['elo+']
            r[x['regime'] + '-'] = x['elo-']
        rows.append(r)
    w = pd.DataFrame(rows)
    w['best'] = w[[c for c in REGIMES if c in w]].max(axis=1)
    return w


# ---- left panel: official tuning-impact bars --------------------------------------------------
def draw_elo(ax, w: pd.DataFrame) -> None:
    w = w.sort_values('best', ascending=False).reset_index(drop=True)
    y = np.arange(len(w))
    height = 0.7
    for reg in ('tuned_ensemble', 'tuned', 'default'):  # widest first, like the official plot
        vals = w[reg] if reg in w else pd.Series(np.nan, index=w.index)
        ok = vals.notna()
        bars = ax.barh(y[ok], vals[ok] - ELO_MIN, left=ELO_MIN, height=height, color=BAR_COLOR[reg],
                       linewidth=0, label=BAR_LABEL[reg], zorder=2)
        for bar, imp in zip(bars, w.loc[ok, 'imputed']):
            if imp:
                bar.set_hatch('xx')
                bar.set_edgecolor('white')
    for reg in REGIMES:
        if reg not in w:
            continue
        ok = w[reg].notna()
        ax.errorbar(w.loc[ok, reg], y[ok], xerr=[w.loc[ok, reg + '-'], w.loc[ok, reg + '+']], fmt='none',
                    ecolor=ERR_COLOR[reg], elinewidth=ERR_WIDTH[reg], alpha=0.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(w['name'])
    for t, (_, r) in zip(ax.get_yticklabels(), w.iterrows()):
        if isinstance(r['family'], str):
            t.set_color(FAMILY_COLOR[r['family']])
            if r['arm'] in ('FT', 'FTE'):
                t.set_fontweight('bold')
    ax.set_ylim(len(w) - 0.35, -0.65)
    right = max(np.nanmax((w[r] + w[r + '+']).to_numpy()) for r in REGIMES if r in w) + 8
    ax.set_xlim(ELO_MIN, right)
    ax.set_xlabel('Elo')
    ax.tick_params(axis='y', pad=0)
    ax.grid(False, axis='y')
    handles = [Patch(color=BAR_COLOR[r], label=BAR_LABEL[r]) for r in REGIMES]
    if w['imputed'].any():
        handles.insert(0, Patch(facecolor='gray', edgecolor='white', hatch='xx', label='Partially imputed'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.42, 1.005), ncol=2,
              borderaxespad=0.0, borderpad=0.25, handletextpad=0.4, labelspacing=0.2, columnspacing=0.8,
              fontsize=5.8, frameon=True)


# ---- right panel: Elo vs latency --------------------------------------------------------------
#: plot key -> measured latency family (LGBM-P: its own compiled latency, TabPrep included).
MEASURED = {
    ('LightGBM', 'R'): 'lightgbm', ('LightGBM', 'FT'): 'lightgbm-ft', ('LightGBM', 'FTE'): 'lightgbm-fte',
    ('LGBM-P', 'R'): 'lgbm-prep-r', ('LGBM-P', 'FT'): 'lgbm-prep-ft', ('LGBM-P', 'FTE'): 'lgbm-prep-fte',
    ('MLP†', 'R'): 'mlp-cos-r', ('MLP†', 'FT'): 'mlp-cos-ft', ('MLP†', 'FTE'): 'mlp-cos-fte',
    ('TabPack', 'R'): 'tabpack-r64', ('TabPack', 'FTE'): 'tabpack-fte64',
}
#: TabICLv2 is timed as ``tabicl2`` in the in-context run (the leaderboard's TabICLv2)
REF_MEASURED = {'CatBoost': 'catboost', 'RealMLP': 'realmlp', 'TabM': 'tabm', 'TabPFN-3': 'tabpfn3',
                'TabICLv2': 'tabicl2', 'Kumo-Tabular': 'kumo-tabular',
                'Kumo-Tabular-Medium': 'kumo-tabular-medium', 'Kumo-Tabular-Small': 'kumo-tabular-small'}
IN_CONTEXT = {'TabPFN-3', 'TabICLv2', 'Kumo-Tabular', 'Kumo-Tabular-Medium', 'Kumo-Tabular-Small'}
#: serving mode each library uses by default (``--fm-style range``); Kumo: TabArena serves from a CPU KV cache
DEFAULT_MODE = {'TabPFN-3': 'recompute', 'TabICLv2': 'recompute', 'Kumo-Tabular': 'cache_cpu',
                'Kumo-Tabular-Medium': 'cache_cpu', 'Kumo-Tabular-Small': 'cache_cpu'}
#: horizontal dodge of near-coincident FT / FTE pairs, in decades of latency and Elo
DODGE_X, DODGE_ELO, DODGE_STEP = 0.06, 12.0, 0.07
REGIME_MARKER = {'default': 'o', 'tuned': 's', 'tuned_ensemble': 'X'}
FILL = {'R': 'none', 'FT': 'bottom', 'FTE': 'full'}
#: in-context look per mode: fill (shape, arrow) and shade (shade)
MODE_FILL = {'recompute': 'none', 'cache_gpu': 'full', 'cache_cpu': 'right'}
MODE_SHADE = {'recompute': '#e2e1dc', 'cache_cpu': '#a3a29d', 'cache_gpu': '#4f4e4a'}
SHADE_EDGE = '#6f6e69'
IC_SIZE = 3.9
#: arrow style: the CPU-cache point, a small hollow dot drawn over the arrow (and over the GPU diamond
#: when the two coincide)
CPU_DOT = dict(marker='o', linestyle='none', markersize=2.3, markerfacecolor='white', markeredgecolor=INK_2,
               markeredgewidth=0.7, zorder=2.7)
#: dodge of two near-coincident modes of one in-context model: a diamond's width, in decades
IC_DODGE_STEP = 0.09


def latency_table(directory: Path) -> pd.Series:
    """Median latency (ms per row) by (family, kind, mode); mode is '' outside the in-context models."""
    files, parts = {}, []
    for (fname, source, variants), fams in LATENCY_INPUTS.items():
        if fname not in files:
            files[fname] = pd.read_csv(directory / fname)
        df = files[fname]
        sel = df[df['source'].eq(source) & df['family'].isin(fams)]
        if 'status' in sel:
            sel = sel[sel['status'].fillna('ok').eq('ok')]
        assert set(sel['variant']) == set(variants), (fname, source, sorted(set(sel['variant'])))
        missing = fams - set(sel['family'])
        assert not missing, f'{fname} / {source}: no rows for {sorted(missing)}'
        parts.append(sel)
    df = pd.concat(parts, ignore_index=True)
    runs = df.groupby('family')['source'].nunique()
    assert (runs == 1).all(), f'family timed in more than one run: {runs[runs > 1].to_dict()}'
    # every GPU point from the H100 runs: no GPU family may fall back to run-1 rows
    gpu = df['device'].eq('cuda')
    assert df.loc[gpu, 'source'].isin(GPU_SOURCES).all()
    assert df.loc[gpu, 'gpu_name'].str.contains('H100').all()
    assert set(df.loc[gpu, 'family']) == DNN_FAMILIES | FM_FAMILIES, sorted(set(df.loc[gpu, 'family']))
    assert df.loc[df['family'].isin(DNN_FAMILIES | FM_FAMILIES), 'device'].eq('cuda').all()
    df['mode'] = np.where(df['family'].isin(FM_FAMILIES), df['variant'].map(MODE_OF), '')
    name = df['config_name'].astype(str)
    df['kind'] = np.where(name.isin(['default', 'pack']), 'default',
                          np.where(name.str.startswith('te'), 'tuned_ensemble', 'tuned'))
    return df.groupby(['family', 'kind', 'mode'])['median_ms'].median()


def points(lb: pd.DataFrame, lat: pd.Series) -> pd.DataFrame:
    """Every plottable (method, regime[, mode]) with its Elo and measured latency."""
    out = []
    for _, r in lb.iterrows():
        fam, arm, reg = r['family'], r['arm'], r['regime']
        if isinstance(fam, str):
            measured = MEASURED[(fam, arm)]
            kind = 'default' if fam == 'TabPack' else reg
            if fam == 'TabPack' and reg != 'default':
                continue
            # D is the same default config in every arm of a family: one point, the baseline's
            if reg == 'default' and fam != 'TabPack' and arm != 'R':
                continue
        elif r['name'] in REF_MEASURED:
            measured = REF_MEASURED[r['name']]
            kind = 'default' if r['name'] in IN_CONTEXT else reg
            if r['name'] in IN_CONTEXT and reg != 'default':
                continue
        else:
            continue
        modes = MODES if r['name'] in IN_CONTEXT else ('',)
        for mode in modes:
            if (measured, kind, mode) not in lat.index:
                assert mode == '', f'{r["name"]}: no {mode} timing'
                continue
            out.append({'name': r['name'], 'family': fam, 'arm': arm, 'regime': reg, 'mode': mode,
                        'elo': r['elo'], 'latency_ms': float(lat.loc[(measured, kind, mode)])})
    return pd.DataFrame(out)


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    best, keep = -np.inf, []
    for i in np.argsort(x):
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return np.array(keep)


def dodge(pts: pd.DataFrame, modes: bool = True) -> pd.DataFrame:
    """Pull near-coincident points apart along x, moving only the slower one, and only to the right.

    An FT / FTE pair of one family and regime within ``DODGE_X`` decades and ``DODGE_ELO`` Elo: FTE
    moves to ``DODGE_STEP`` decades (a few points on the page) past FT. Two serving modes of one
    in-context model (same Elo) within ``DODGE_X`` decades (``modes``): the slower moves to
    ``IC_DODGE_STEP`` past the faster, so the two diamonds just touch. Every other point stays at its measured latency; moving a point only to the right can
    never make it look better than it is."""
    pts = pts.copy()
    ours = pts[pts['family'].notna() & pts['arm'].isin(['FT', 'FTE'])]
    for _, g in ours.groupby(['family', 'regime']):
        if len(g) != 2:
            continue
        lx = np.log10(g['latency_ms'].to_numpy())
        if abs(lx[0] - lx[1]) < DODGE_X and abs(g['elo'].iloc[0] - g['elo'].iloc[1]) < DODGE_ELO:
            ft, fte = g.index[g['arm'] == 'FT'][0], g.index[g['arm'] == 'FTE'][0]
            pts.loc[fte, 'latency_ms'] = max(pts.loc[fte, 'latency_ms'], pts.loc[ft, 'latency_ms'] * 10 ** DODGE_STEP)
    for _, g in pts[pts['name'].isin(IN_CONTEXT) & modes].groupby('name'):
        g = g.sort_values('latency_ms')
        prev = None
        for i in g.index:
            if prev is not None and np.log10(pts.loc[i, 'latency_ms'] / prev) < DODGE_X:
                pts.loc[i, 'latency_ms'] = prev * 10 ** IC_DODGE_STEP
            prev = pts.loc[i, 'latency_ms']
    return pts


HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
#: published references named on the plot (name, regime) -> (dx, dy, ha) in points; unlisted points
#: (the references' T) stay unnamed
REF_LABEL = {('RealMLP', 'tuned_ensemble'): (6, 0, 'left'), ('CatBoost', 'tuned'): (-5, 0, 'right'),
             ('TabM', 'tuned_ensemble'): (6, 0, 'left')}
#: in-context models, named once below their span of modes: name -> (anchor, dx, dy, ha) in points;
#: anchor 'mid' = centred on the span, 'min' = starting just right of the fastest mode (clear of the
#: front's vertical steps, which rise at the Kumo cache points)
IC_LABEL = {'TabPFN-3': ('mid', 0, -7, 'center'), 'TabICLv2': ('mid', 0, -7, 'center'),
            'Kumo-Tabular': ('min', 3, -7, 'left'), 'Kumo-Tabular-Medium': ('min', 3, -7, 'left'),
            'Kumo-Tabular-Small': ('min', 3, -7, 'left')}

def mark(ax, r) -> None:
    mine = isinstance(r['family'], str)
    color = FAMILY_COLOR[r['family']] if mine else GREY
    marker = REGIME_MARKER[r['regime']]
    size = ((5.5 if marker == 'X' else 4.7) - (0 if mine else 0.6))
    ax.plot(r['latency_ms'], r['elo'], marker=marker, linestyle='none', markersize=size,
            fillstyle=FILL[r['arm']] if mine else 'none', markerfacecolor=color, markerfacecoloralt='white',
            markeredgecolor=color, markeredgewidth=1.1, zorder=3 if mine else 2.5)


def diamond(ax, x, y, mode, style, size=IC_SIZE) -> None:
    """One in-context point in the look of ``style``."""
    kw = dict(marker='D', linestyle='none', markersize=size, markeredgewidth=1.0, zorder=2.6)
    if style == 'shade':
        kw.update(markerfacecolor=MODE_SHADE[mode], markeredgecolor=SHADE_EDGE, markeredgewidth=0.8)
    elif style == 'range':
        kw.update(markerfacecolor='white', markeredgecolor=GREY)
    else:
        kw.update(fillstyle=MODE_FILL[mode], markerfacecolor=GREY, markerfacecoloralt='white', markeredgecolor=GREY)
    ax.plot(x, y, **kw)


def draw_in_context(ax, ic: pd.DataFrame, shown: pd.DataFrame, style: str) -> None:
    """``ic``: all modes at measured latency; ``shown``: the drawn points (dodged)."""
    for name, g in shown.groupby('name'):
        g = g.set_index('mode')
        y = g['elo'].iloc[0]
        if style in ('shape', 'shade'):
            ax.plot(np.sort(g['latency_ms']), [y] * len(g), color=GREY, lw=0.6, zorder=2.2)
            for mode in ('recompute', 'cache_cpu', 'cache_gpu'):
                diamond(ax, g.loc[mode, 'latency_ms'], y, mode, style)
        elif style == 'range':
            meas = ic[ic['name'] == name].set_index('mode')['latency_ms']
            ax.plot([meas.min(), meas.max()], [y, y], color=GREY, lw=0.9, solid_capstyle='butt', zorder=2.2)
            others = meas.drop(DEFAULT_MODE[name])
            ax.plot(others, [y] * len(others), marker='|', linestyle='none', markersize=5.0,
                    markeredgewidth=0.9, color=GREY, zorder=2.3)
            diamond(ax, g.loc[DEFAULT_MODE[name], 'latency_ms'], y, DEFAULT_MODE[name], style)
        else:  # arrow
            rec, gpu = g.loc['recompute', 'latency_ms'], g.loc['cache_gpu', 'latency_ms']
            ax.annotate('', xy=(gpu, y), xytext=(rec, y), zorder=2.4,
                        arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.8, mutation_scale=6,
                                        shrinkA=3.2, shrinkB=3.2))
            diamond(ax, rec, y, 'recompute', style)
            diamond(ax, gpu, y, 'cache_gpu', style)
            ax.plot(g.loc['cache_cpu', 'latency_ms'], y, **CPU_DOT)


class _ArrowHandle:
    """Legend stand-in for the recompute -> GPU cache arrow."""


class _ArrowHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        y = -yd + height / 2
        x0, x1 = -xd, -xd + width
        arrow = FancyArrowPatch((x0, y), (x1 - 5.6, y), arrowstyle='-|>', color=GREY, lw=0.8,
                                mutation_scale=5, shrinkA=0, shrinkB=0, transform=trans)
        head = Line2D([x1 - 3.4], [y], marker='D', linestyle='none', markersize=3.6, markerfacecolor=GREY,
                      markeredgecolor=GREY, transform=trans)
        return [arrow, head]


class _RangeHandle:
    """Legend stand-in for the mode range bar."""


class _RangeHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        y = -yd + height / 2
        x0, x1 = -xd, -xd + width
        bar = Line2D([x0, x1], [y, y], color=GREY, lw=0.9, solid_capstyle='butt', transform=trans)
        ticks = Line2D([x0, x1], [y, y], marker='|', linestyle='none', markersize=5.0, markeredgewidth=0.9,
                       color=GREY, transform=trans)
        return [bar, ticks]


def mode_key(style: str) -> tuple[list, list[str], dict]:
    """Legend handles, labels and handler map for the in-context modes."""
    def d(fill='none', face=GREY, edge=GREY, size=4.0, mew=1.0):
        return Line2D([], [], marker='D', linestyle='none', markersize=size, fillstyle=fill, markerfacecolor=face,
                      markerfacecoloralt='white', markeredgecolor=edge, markeredgewidth=mew)
    if style == 'shape':
        return [d(MODE_FILL[m]) for m in MODES], [MODE_LABEL[m] for m in MODES], {}
    if style == 'shade':
        return ([d('full', MODE_SHADE[m], SHADE_EDGE, mew=0.8) for m in MODES], [MODE_LABEL[m] for m in MODES], {})
    if style == 'range':
        # a blank first slot keeps the two in-context entries in one legend column
        return ([Line2D([], [], linestyle='none'), d('full', 'white'), _RangeHandle()],
                ['', 'default mode', 'range of modes'], {_RangeHandle: _RangeHandler()})
    return ([d(), _ArrowHandle(), Line2D([], [], **CPU_DOT)], [MODE_LABEL['recompute'], MODE_LABEL['cache_gpu'],
                                                 MODE_LABEL['cache_cpu']], {_ArrowHandle: _ArrowHandler()})


def no_tuning(pts: pd.DataFrame) -> pd.Series:
    """TabPack and the in-context models have a single (default) configuration."""
    return (pts['family'] == 'TabPack') | pts['name'].isin(IN_CONTEXT)


def draw_latency(ax, pts: pd.DataFrame, style: str) -> pd.DataFrame:
    """Draw the right panel; returns the Pareto front (measured latencies)."""
    # T and T+E only: the D points of tunable methods sit 150-300 Elo lower and would stretch the axis
    pts = pts[no_tuning(pts) | (pts['regime'] != 'default')]
    ic_all = pts[pts['name'].isin(IN_CONTEXT)]
    if style == 'range':  # one point per in-context model, at its library default mode
        pts = pts[~pts['name'].isin(IN_CONTEXT) | pts['mode'].eq(pts['name'].map(DEFAULT_MODE))]
    # arrow: the small CPU-cache marker sits at its measured latency, on the arrow (no dodge needed)
    shown = dodge(pts, modes=style != 'arrow')
    ic = shown['name'].isin(IN_CONTEXT)
    for _, r in shown[~ic].iterrows():
        mark(ax, r)
    draw_in_context(ax, ic_all, shown[ic], style)
    # one Pareto front over every point drawn, on the measured latencies (never the dodged ones)
    f = pts.iloc[pareto_front(pts['latency_ms'].to_numpy(), pts['elo'].to_numpy())].sort_values('latency_ms')
    ax.step(np.append(f['latency_ms'], 1e5), np.append(f['elo'], f['elo'].iloc[-1]), where='post',
            color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1.5)
    for _, r in shown[shown['family'].isna() & ~ic].iterrows():
        if (r['name'], r['regime']) in REF_LABEL:
            dx, dy, ha = REF_LABEL[(r['name'], r['regime'])]
            ax.annotate(r['name'], (r['latency_ms'], r['elo']), xytext=(dx, dy), textcoords='offset points',
                        fontsize=6.0, color=INK_2, ha=ha, va='center', zorder=5, path_effects=HALO)
    for name, (anchor, dx, dy, ha) in IC_LABEL.items():
        span = shown.loc[shown['name'].eq(name), 'latency_ms']
        if style == 'range':  # the bar spans every mode, the diamond only the default
            span = ic_all.loc[ic_all['name'].eq(name), 'latency_ms']
        if anchor == 'mid':
            x = np.sqrt(span.min() * span.max())
        else:  # range: start right of the default diamond, where the front's step rises
            x = shown.loc[shown['name'].eq(name), 'latency_ms'].max() if style == 'range' else span.min()
        ax.annotate(name, (x, shown.loc[shown['name'].eq(name), 'elo'].iloc[0]), xytext=(dx, dy),
                    textcoords='offset points', fontsize=6.0, color=INK_2, ha=ha, va='center', zorder=5,
                    path_effects=HALO)

    xs = pd.concat([shown['latency_ms'], ic_all['latency_ms']])
    ax.set_xscale('log')
    ax.set_xlim(xs.min() / 1.45, xs.max() * 2.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    # above 1 s only the decades are labelled: 1000 / 2000 / 5000 would collide
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{v:g}' if v < 1000 or round(v / 10 ** np.floor(np.log10(v))) == 1 else ''))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(pts['elo'].min() - 12, pts['elo'].max() + 14)
    ax.set_xlabel('Inference latency, batch size 1 (ms per row)')
    ax.set_ylabel('Elo')

    # marker key above the panel: regime = shape, in-context mode, portfolio = fill
    key = [Line2D([], [], marker=REGIME_MARKER[r], linestyle='none', markersize=4.6, markerfacecolor='white',
                  markeredgecolor=INK_2) for r in ('default', 'tuned', 'tuned_ensemble')]
    labels = ['D', 'T', 'T+E']
    mh, ml, hmap = mode_key(style)
    key += mh
    labels += ml
    key += [Line2D([], [], marker='s', linestyle='none', markersize=4.6, fillstyle=fs, markerfacecolor=INK_2,
                   markerfacecoloralt='white', markeredgecolor=INK_2) for fs in ('none', 'bottom', 'full')]
    labels += ['baseline', 'FT', 'FTE']
    key.append(Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2))))
    labels.append('Pareto front')
    ax.legend(handles=key, labels=labels, handler_map=hmap, loc='lower center', bbox_to_anchor=(0.5, 1.005),
              ncol=int(np.ceil(len(key) / 2)), fontsize=5.8, frameon=True, borderaxespad=0.0, borderpad=0.25,
              handletextpad=0.3, labelspacing=0.2, columnspacing=0.8)
    # family key, top left: colour = our family, grey = published reference
    fam = [Patch(facecolor=FAMILY_COLOR[f], edgecolor='none', label=FAMILY_TEX[f]) for f in FAMILY_ORDER]
    fam.append(Patch(facecolor=GREY, edgecolor='none', label='Published'))
    ax.add_artist(ax.get_legend())
    ax.legend(handles=fam, loc='upper left', fontsize=5.8, ncol=1, frameon=True, borderpad=0.3,
              handlelength=0.9, handleheight=0.9, handletextpad=0.4, labelspacing=0.25)
    return f


# ---- the figure ------------------------------------------------------------------------------
#: fixed margins, inches: room for the legends on top and the x labels at the bottom
#: left margin widened from 0.78 in so the widest row label (bold "LightGBM-TabPrep FTE") stays on the canvas
TOP_IN, BOTTOM_IN, LEFT_IN, RIGHT_IN = 0.36, 0.36, 0.90, 0.06


def render(lb: pd.DataFrame, lat: pd.Series, out: Path, *, width: float, aspect: float, style: str) -> None:
    """``aspect`` = width / height of the right panel's axes; the figure height follows from it."""
    pts = points(lb, lat)
    w = wide(lb)
    # one method set for both panels: a method without a latency measurement is not drawn
    dropped = sorted(set(w['name']) - set(pts['name']))
    if dropped:
        print('left out of both panels (no latency):', ', '.join(dropped))
    w = w[w['name'].isin(set(pts['name']))].reset_index(drop=True)
    print(f'{len(w)} methods in both panels')
    ratios, wspace = (1.0, 1.3), 0.24
    axes_w = width - LEFT_IN - RIGHT_IN
    unit = axes_w / (sum(ratios) + wspace * sum(ratios) / 2)
    axes_h = ratios[1] * unit / aspect
    height = axes_h + TOP_IN + BOTTOM_IN
    # left-panel row labels: as large as the row pitch allows (one row per method)
    tick_pt = min(5.8, 0.88 * axes_h * 72 / len(w))
    with sns.axes_style('whitegrid'), plt.rc_context({'font.size': 6.5, 'axes.labelsize': 7.5,
                                                        'xtick.labelsize': 6.3, 'ytick.labelsize': tick_pt,
                                                        'axes.edgecolor': '#cccccc'}):
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(1, 2, width_ratios=ratios, wspace=wspace, left=LEFT_IN / width,
                                right=1 - RIGHT_IN / width, top=1 - TOP_IN / height, bottom=BOTTOM_IN / height)
        ax_l = fig.add_subplot(grid[0, 0])
        ax_r = fig.add_subplot(grid[0, 1])
        draw_elo(ax_l, w)
        with plt.rc_context({'ytick.labelsize': 6.3}):
            ax_r.tick_params(axis='y', labelsize=6.3)
            front = draw_latency(ax_r, pts, style)
        ax_r.grid(True, which='major', color='#e4e4e4', lw=0.6)
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            fig.savefig(out.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
        bb = ax_r.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        print(f'figure {width:.2f} x {height:.2f} in; right panel {bb.width:.2f} x {bb.height:.2f} in '
              f'(aspect {bb.width / bb.height:.2f})')
        plt.close(fig)
    shown = pts[no_tuning(pts) | (pts['regime'] != 'default')].sort_values(['elo', 'latency_ms'], ascending=[False, True])
    print('right-panel points (measured latency, ms per row):')
    for _, r in shown.iterrows():
        print(f'  {r["name"]:<22} {r["regime"]:<15} {r["mode"] or "-":<10} {r["latency_ms"]:9.3f} ms  Elo {r["elo"]:.1f}')
    scope = 'default-mode in-context points only' if style == 'range' else 'every point drawn'
    print(f'Pareto front ({scope}):')
    for _, r in front.iterrows():
        print(f'  {r["name"]:<22} {r["regime"]:<15} {r["mode"] or "-":<10} {r["latency_ms"]:9.3f} ms  Elo {r["elo"]:.1f}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--leaderboard', type=Path, default=FULL_BA_ELO,
                   help='Elo leaderboard; default the full BeyondArena core')
    p.add_argument('--latency-dir', type=Path, default=LATENCY_DIR,
                   help='directory holding the batch-1 latency CSVs of LATENCY_INPUTS')
    p.add_argument('--lgbm-prep-as-lightgbm', action='store_true',
                   help='LGBM-P at LightGBM latency (TabPrep feature generation not timed)')
    p.add_argument('--fm-style', choices=FM_STYLES, default='shade',
                   help='how the three serving modes of the in-context models are drawn (module docstring)')
    p.add_argument('--out', type=Path, default=PAPER_DIR / 'figures' / 'elo-latency')
    p.add_argument('--width', type=float, default=7.0, help='inches; a full A4 text width')
    p.add_argument('--aspect', type=float, default=1.7,
                   help='right panel width / height (the axes of pareto-elo-vs-latency are 1.16); '
                        '1.7 keeps the figure short on an A4 page')
    args = p.parse_args()
    lb = load_leaderboard(args.leaderboard)
    lb = lb[~lb['bank'].isin(EXCLUDED_BANKS)]
    lat = latency_table(args.latency_dir)
    if args.lgbm_prep_as_lightgbm:
        for arm in ('R', 'FT', 'FTE'):
            MEASURED[('LGBM-P', arm)] = 'lightgbm'
    missing = {v for v in [*MEASURED.values(), *REF_MEASURED.values()] if not any(v == f for f, _, _ in lat.index)}
    assert not missing, f'no latency rows for {sorted(missing)}'
    render(lb, lat, args.out, width=args.width, aspect=args.aspect, style=args.fm_style)
    print(f'wrote {args.out}.pdf/.png')


if __name__ == '__main__':
    main()
