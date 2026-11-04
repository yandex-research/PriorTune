#!/usr/bin/env python3
"""Section 5 and its appendix: every ablation figure and the selection-rule table.

Reads data/ablations/ and data/selection/ (see data/SOURCES.md for what produced each
file) and, for the selected-hyperparameter figures, the K = 200 T portfolios and the
config banks from the artifacts (python bin/artifacts.py download portfolios configs).
Writes figures/ablations/*.pdf and figures/tables/selection_ablation.tex, and prints the
numbers the text quotes (scaling fits, ladder peaks, budget crossovers, HP shifts).

    python bin/figures/ablations.py [--preview DIR]

  budget_K200, budget_grid                 iso-budget curves, synthetic vs TabRepo
  ladder_K200, ladder_grid                 selected K subset vs random-K'
  priors_K200, priors_grid_{configs,datasets}   generator priors, vs C and vs N
  scaling_K200_heatmaps, scaling_heatmaps_grid  C x N win-rate surfaces
  scaling_grid_{datasets,configs}          full-pool curves with 1/sqrt(N) and log2 C fits
  selection_oof                            selection rules, synthetic out-of-fold
  hp_main, hp_<model>                      selected hyperparameters vs their sampling distribution
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

# --- sources (see data/SOURCES.md) ---------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bin'))
import artifacts  # noqa: E402
ABL = ROOT / 'data/ablations'
SEL = ROOT / 'data/selection'
OUT = ROOT / 'figures/ablations'

KS = (25, 64, 200)
POOLS = ('lgbm', 'lgbm-prep', 'mlp')
MODEL = {'lgbm': 'LightGBM', 'lgbm-prep': 'LightGBM-TabPrep',
         'mlp': 'MLP'}
# The priors study: 2,000 configs of each bank x 2,000 datasets of each prior.
FAMILIES = ('lgbm', 'lgbm-prep', 'mlp')
FAMILY = {'lgbm': 'LightGBM', 'lgbm-prep': 'LightGBM-TabPrep', 'mlp': 'MLP'}
PRIOR_ARMS = ('tabicl-v2', 'tabicl-v1', 'tabpfn-v1')
PRIOR = {'tabicl-v2': 'TabICLv2', 'tabicl-v1': 'TabICLv1', 'tabpfn-v1': 'TabPFNv1'}

# --- style (one rcParams dict, one axis helper) ------------------------------

INK, SPINE, GRIDC = '#1a1a1a', '#8a8983', '#e4e4e4'
COLOR = {'lgbm': '#2a78d6', 'lgbm-prep': '#008300', 'mlp': '#4a3aa7'}
TABREPO = '#eb6834'
PRIOR_COLOR = {'tabicl-v2': '#1baf7a', 'tabicl-v1': '#eb6834', 'tabpfn-v1': '#2a78d6'}
FAMILY_COLOR = {'lgbm': COLOR['lgbm'], 'lgbm-prep': COLOR['lgbm-prep'],
                'mlp': COLOR['mlp']}
LW, BAND = 1.3, 0.18
Y_WR = 'Win rate vs. random-$K$'


def y_wr(k):
    """Y label naming the subset size, e.g. 'Win rate vs. random-200'."""
    return f'Win rate vs. random-{k}'

X_BUDGET = r'Budget $B = C \cdot N$'
X_C, X_N, X_KP = 'Configurations $C$', 'Synthetic datasets $N$', "Random subset size $K'$"

RC = {
    'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'axes.titlesize': 8, 'axes.titleweight': 'normal',
    'axes.titlelocation': 'center', 'axes.titlepad': 4, 'axes.titlecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': SPINE,
    'xtick.color': SPINE, 'ytick.color': SPINE, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'lines.linewidth': LW, 'lines.markersize': 2.4, 'axes.linewidth': 0.8,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5,
    'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3, 'legend.columnspacing': 1.2,
    'mathtext.fontset': 'dejavusans', 'savefig.dpi': 300, 'figure.dpi': 100,
    'axes.axisbelow': True,
}


def style_axis(ax, grid=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(which='both', color=SPINE, labelcolor=INK)
    ax.grid(grid, which='major', axis='both', color=GRIDC, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def reference_line(ax):
    ax.axhline(0.5, color=SPINE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)


def doubling_ticks(ax, lo, hi, n_max=5):
    """Log2 x axis with plain-number labels lo, 2^s lo, 4^s lo, ... (s chosen for <= n_max)."""
    ax.set_xscale('log', base=2)
    if lo == 1:
        ticks = [t for t in (1, 10, 100, 1000) if t <= hi * 1.02]
    else:
        for step in (1, 2, 3, 4):
            ticks = [lo * 2 ** (step * j) for j in range(20) if lo * 2 ** (step * j) <= hi * 1.02]
            if len(ticks) <= n_max:
                break
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lo / 1.3, hi * 1.3)


def power2_ticks(ax, lo, hi, every=3):
    """Log2 x axis labelled 2^k at multiples of `every`, unlabelled minor ticks at every k."""
    ax.set_xscale('log', base=2)
    klo, khi = int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi)))
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** k for k in range(klo, khi + 1) if k % every == 0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'$2^{{{int(round(np.log2(v)))}}}$'))
    ax.xaxis.set_minor_locator(FixedLocator([2.0 ** k for k in range(klo, khi + 1)]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo / 1.5, hi * 1.5)


def tint(color, t):
    """Mix `color` with white; t=1 is the colour itself, t=0 white."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * t + (1 - t))


def new_figure(nrows, ncols, figsize, width_ratios=None):
    fig = plt.figure(figsize=figsize, layout='constrained')
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.03, hspace=0.04)
    gs = fig.add_gridspec(nrows, ncols, width_ratios=width_ratios)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])
    return fig, axes


def row_label(ax, k, ylabel):
    ax.set_ylabel(f'$K = {k}$\n{ylabel}')


def save(fig, name, preview):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f'wrote {path} ({path.stat().st_size:,} bytes)', flush=True)
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f'{name}.png', bbox_inches='tight', dpi=220)
    plt.close(fig)


# --- data ------------------------------------------------------------------------

def read_csv(path):
    """Rows of a CSV as dicts, numeric fields converted."""
    def conv(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(path, newline='') as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]


def as_grid(rows, row_key, col_key):
    """{'rows', 'cols', 'wr', 'ci'} from long rows with winrate/ci95 columns."""
    rs = sorted({r[row_key] for r in rows})
    cs = sorted({r[col_key] for r in rows})
    wr, ci = np.full((len(rs), len(cs)), np.nan), np.full((len(rs), len(cs)), np.nan)
    for r in rows:
        i, j = rs.index(r[row_key]), cs.index(r[col_key])
        wr[i, j], ci[i, j] = r['winrate'], r['ci95']
    assert not np.isnan(wr).any(), 'incomplete grid'
    return dict(rows=np.asarray(rs, float), cols=np.asarray(cs, float), wr=wr, ci=ci)


def surface(rec, k):
    """(grid, wr[N_idx, C_idx], ci[N_idx, C_idx]) of a square C x N surface; checks the C=K column."""
    grid, wr, ci = rec['grid'], rec['wr'], rec['ci']
    assert grid[0] == k, (grid[0], k)
    assert np.allclose(wr[:, 0], 0.5), 'C=K column must read 0.500'
    return grid, wr, ci


def load_surfaces():
    """{(study, model, k): {'grid', 'wr', 'ci'}} with N on rows and C on columns."""
    groups = defaultdict(list)
    for r in read_csv(ABL / 'surfaces.csv'):
        groups[(r['study'], r['model'], r['k'])].append(r)
    out = {}
    for key, rows in groups.items():
        g = as_grid(rows, 'n', 'c')
        assert np.array_equal(g['rows'], g['cols'])
        out[key] = dict(grid=g['rows'], wr=g['wr'], ci=g['ci'])
    return out


def load_tuned():
    surf = load_surfaces()
    repo = defaultdict(list)
    for r in read_csv(ABL / 'tabrepo_ladder.csv'):
        repo[(r['model'], r['k'], r['width'])].append(r)
    ladder = defaultdict(list)
    for r in read_csv(ABL / 'ladder.csv'):
        ladder[(r['model'], r['k'])].append(r)
    climbed = {(r['model'], r['k']): r for r in read_csv(ABL / 'climbed.csv')}
    draws = defaultdict(dict)
    for r in read_csv(ABL / 'random_draws.csv'):
        draws[(r['model'], r['k'])][r['kprime']] = r
    data = {}
    for pool in POOLS:
        for k in KS:
            ladders = {}
            for width in (4000, 16000):
                g = as_grid(repo[(pool, k, width)], 'n', 'c')
                assert g['rows'][-1] == 182 and g['cols'][-1] == width
                ladders[width] = dict(grid_d=g['rows'], grid_c=g['cols'], wr=g['wr'], ci=g['ci'],
                                      n_meta=182, width=width)
            rows = sorted(ladder[(pool, k)], key=lambda r: r['kprime'])
            data[(pool, k)] = dict(T=surf[('tuned', pool, k)], ladder=rows, draws=draws[(pool, k)],
                                   climbed=climbed[(pool, k)], tabrepo=ladders[4000],
                                   tabrepo16000=ladders[16000])
    return data


def load_priors():
    surf = load_surfaces()
    return {(family, prior, k): surf[('priors', f'{family}-{prior}', k)]
            for family in FAMILIES for prior in PRIOR_ARMS for k in KS}


def load_budget():
    data = defaultdict(list)
    for r in read_csv(ABL / 'budget.csv'):
        data[(r['model'], r['k'])].append((r['budget'], r['winrate']))
    return {key: tuple(np.asarray(v, float) for v in zip(*pts)) for key, pts in data.items()}


def full_pool(rec, k, axis):
    """Full-pool curve along `axis`: 'N' at the widest C, 'C' (C > K) at the largest N."""
    grid, wr, ci = surface(rec, k)
    if axis == 'N':
        return grid, wr[:, -1], ci[:, -1]
    return grid[1:], wr[-1, 1:], ci[-1, 1:]


def family_curves(rec, k, axis):
    """{level: (y, ci)} along `axis`, one curve per level of the other axis; x first."""
    grid, wr, ci = surface(rec, k)
    if axis == 'N':
        return grid, {int(c): (wr[:, j], ci[:, j]) for j, c in enumerate(grid) if j > 0}
    return grid[1:], {int(n): (wr[i, 1:], ci[i, 1:]) for i, n in enumerate(grid)}


def ladder_series(entry, k):
    """Random-K' against the fixed random-K baseline, plus the two meta-selected K subsets.

    Returns dict: x; the random-K' curve on the original 4,000 pool (pool, pool_ci; 0.5 at
    K'=K by construction); single random draws of K' from the 16,000 bank (mean, and the
    2.5-97.5 percentile band over draws, share of draws at or above the meta-tuned subset);
    (win rate, ci95) of the synthetic-selected K subset and of the TabRepo-182 selected
    K subset (full corner: all 4,000 configs, 182 datasets).
    """
    rows = entry['ladder']
    x = np.asarray([r['kprime'] for r in rows], float)
    y = np.asarray([r['pool_winrate'] for r in rows], float)
    assert abs(y[list(x).index(k)] - 0.5) < 1e-9, 'random-K vs random-K must read 0.5'
    draws = [entry['draws'][int(kp)] for kp in x]
    col = lambda name: np.asarray([r[name] for r in draws], float)
    repo = entry['tabrepo']
    return dict(x=x, pool=y, pool_ci=np.asarray([r['pool_ci95'] for r in rows], float),
                mean=col('mean'), lo=col('p2_5'), hi=col('p97_5'), dmax=col('max'),
                share=col('share_ge_meta_tuned'), n_draws=int(draws[0]['n_draws']),
                synth=(float(entry['climbed']['winrate']), float(entry['climbed']['ci95'])),
                repo=(float(repo['wr'][-1, -1]), float(repo['ci'][-1, -1])))


def envelope(xs, ys):
    """Best cell per log2 budget bin -> (2^e, wr) in budget order."""
    bins = {}
    for x, y in zip(xs, ys):
        e = int(round(np.log2(x)))
        bins[e] = max(bins.get(e, -np.inf), y)
    es = np.array(sorted(bins))
    return 2.0 ** es, np.array([bins[e] for e in es])


def cells(grid_d, grid_c, wr, k):
    """(budget, wr) of every C x N cell with C > K, flattened."""
    D = np.asarray(grid_d, float)[:, None]
    C = np.asarray(grid_c, float)[None, :]
    keep = np.broadcast_to(C > k, wr.shape)
    return (D * C)[keep], wr[keep]


def budget_series(points, ext, k):
    """Synthetic cells + envelope (incl. the N < K extension); TabRepo cells + envelope
    from the 16,000-config bank ladder (C > K, N up to 182)."""
    sx, sy = points
    sex, sey = envelope(sx, sy)
    # Scored against the synthetic baselines, so the C=K column is not 0.5 by construction;
    # cells() drops it anyway.
    wr = ext['wr']
    tx, ty = cells(ext['grid_d'], ext['grid_c'], wr, k)
    tex, tey = envelope(tx, ty)
    corner = float(wr[-1, -1])
    return {'synthetic': (sx, sy, sex, sey), 'tabrepo': (tx, ty, tex, tey)}, ext['n_meta'], ext['width'], corner


# --- fits --------------------------------------------------------------------

def basis(x, kind):
    x = np.asarray(x, float)
    return np.log2(x) if kind == 'log' else 1.0 / np.sqrt(x)


def fit(x, y, kind):
    """Least squares y = a + b f(x), f = log2 x or 1/sqrt x. Returns a, b, R²."""
    f, y = basis(x, kind), np.asarray(y, float)
    A = np.stack([np.ones_like(f), f], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * f
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return dict(a=float(a), b=float(b), r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def predict(x, kind, p):
    return p['a'] + p['b'] * basis(x, kind)


def both_fits(x, y):
    return {'log': fit(x, y, 'log'), 'isqrt': fit(x, y, 'isqrt')}


def theory_kind(axis):
    """The paper's prediction: 1/sqrt(N) over datasets, log2 C over configurations."""
    return 'isqrt' if axis == 'N' else 'log'


def eq_text(fits, axis):
    kind = theory_kind(axis)
    p = fits[kind]
    if kind == 'isqrt':
        b = -p['b']
        eq = f"${p['a']:.3f} {'-' if b >= 0 else '+'} {abs(b):.2f}/\\sqrt{{N}}$"
    else:
        eq = f"${p['a']:.3f} {'+' if p['b'] >= 0 else '-'} {abs(p['b']):.4f}\\,\\log_2 C$"
    return f"{eq}\n$R^2 = {p['r2']:.2f}$"


def draw_fit(ax, x, fits, axis):
    kind = theory_kind(axis)
    xx = np.geomspace(x.min(), x.max(), 200)
    line, = ax.plot(xx, predict(xx, kind, fits[kind]), color=INK, linewidth=0.9,
                    linestyle=(0, (4, 2)), zorder=5)
    ax.text(0.04, 0.96, eq_text(fits, axis), transform=ax.transAxes, fontsize=6,
            color=INK, ha='left', va='top', linespacing=1.25, zorder=6)
    return line


# --- figure 1: budget ----------------------------------------------------------

def draw_budget(ax, series, pool, k, n_meta, width):
    handles = []
    for side, colour, label in (('synthetic', COLOR[pool], 'Synthetic'),
                                ('tabrepo', TABREPO, f'TabRepo ({n_meta} datasets, {width:,} configs)')):
        xs, ys, ex, ey = series[side]
        ax.scatter(xs, ys, s=4, color=colour, alpha=0.25, linewidths=0, zorder=2)
        h, = ax.plot(ex, ey, color=colour, linewidth=LW, marker='o', markersize=2.4,
                     markeredgewidth=0, zorder=4 if side == 'synthetic' else 5, label=label)
        handles.append(h)
    reference_line(ax)
    return handles


def budget_limits(entries):
    lo = min(e[0][side][1].min() for e in entries for side in ('synthetic', 'tabrepo'))
    hi = max(e[0][side][1].max() for e in entries for side in ('synthetic', 'tabrepo'))
    blo = min(e[0][side][0].min() for e in entries for side in ('synthetic', 'tabrepo'))
    bhi = max(e[0][side][0].max() for e in entries for side in ('synthetic', 'tabrepo'))
    return (min(lo, 0.49) - 0.01, hi + 0.015), (blo, bhi)


def figure_budget(budget, tuned, ks, name, figsize, preview):
    entries = {(pool, k): budget_series(budget[(pool, k)], tuned[(pool, k)]['tabrepo16000'], k)
               for pool in POOLS for k in ks}
    ylim, (blo, bhi) = budget_limits(list(entries.values()))
    fig, axes = new_figure(len(ks), len(POOLS), figsize)
    for r, k in enumerate(ks):
        for c, pool in enumerate(POOLS):
            ax = axes[r, c]
            series, n_meta, width, _ = entries[(pool, k)]
            handles = draw_budget(ax, series, pool, k, n_meta, width)
            if r == 0 and c == 0:
                legend = handles
            style_axis(ax)
            power2_ticks(ax, blo, bhi)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(MODEL[pool])
            if r == len(ks) - 1:
                ax.set_xlabel(X_BUDGET)
            if c == 0:
                row_label(ax, k, y_wr(k)) if len(ks) > 1 else ax.set_ylabel(y_wr(k))
            else:
                ax.tick_params(labelleft=False)
    # One shared legend under the figure: the long TabRepo label does not fit inside a panel.
    fig.legend(handles=legend, loc='outside lower center', ncol=2)
    print(f'{name}: ylim = ({ylim[0]:.4f}, {ylim[1]:.4f}), budgets 2^{np.log2(blo):.1f} .. 2^{np.log2(bhi):.1f}',
          flush=True)
    save(fig, name, preview)


# --- figure 2: ladder -----------------------------------------------------------

def draw_ladder(ax, entry, pool, k, ylim):
    """Random-K' against random-K: single draws from the 16,000 bank (mean, 95% band over draws),
    with the two selected K subsets as horizontal lines."""
    L = ladder_series(entry, k)
    x, colour = L['x'], COLOR[pool]
    band = ax.fill_between(x, L['lo'], L['hi'], color=colour, alpha=BAND, linewidth=0, zorder=2)
    curve, = ax.plot(x, L['mean'], color=colour, linewidth=LW, marker='o', markersize=2.4,
                     markeredgewidth=0, zorder=4)
    s_wr, s_ci = L['synth']
    r_wr, r_ci = L['repo']
    synth = ax.axhline(s_wr, color=INK, linewidth=1.2, zorder=5)
    repo = ax.axhline(r_wr, color=TABREPO, linewidth=1.2, linestyle=(0, (5, 2)), zorder=5)
    reference_line(ax)
    ax.axvline(k, color=SPINE, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.text(k * 1.18, ylim[1] - 0.006, "$K' = K$", fontsize=6, color=SPINE, ha='left', va='top')
    return ([(band, curve), synth, repo],
            [f"Random $K'$ of 16,000 (mean, 95% of {L['n_draws']} draws)",
             'Meta-tuned $K$ (synthetic)', 'Meta-tuned $K$ (TabRepo)'])


def ladder_limits(series):
    """Random-K' bands: their minimum minus a margin, up to the highest band or line plus margin."""
    lo, hi = 0.5, 0.5
    for L in series:
        lo = min(lo, float(L['lo'].min()))
        hi = max(hi, float(L['hi'].max()), L['synth'][0] + L['synth'][1], L['repo'][0] + L['repo'][1])
    return (lo - 0.02, hi + 0.04)


def figure_ladder(tuned, ks, name, figsize, preview):
    series = [ladder_series(tuned[(pool, k)], k) for pool in POOLS for k in ks]
    ylim = ladder_limits(series)
    xlo = min(L['x'].min() for L in series)
    xhi = max(L['x'].max() for L in series)
    print(f'{name}: ylim = ({ylim[0]:.4f}, {ylim[1]:.4f})', flush=True)
    fig, axes = new_figure(len(ks), len(POOLS), figsize)
    for r, k in enumerate(ks):
        for c, pool in enumerate(POOLS):
            ax = axes[r, c]
            handles, labels = draw_ladder(ax, tuned[(pool, k)], pool, k, ylim)
            if r == 0 and c == 0:
                legend = (handles, labels)
            style_axis(ax)
            doubling_ticks(ax, xlo, xhi)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(MODEL[pool])
            if r == len(ks) - 1:
                ax.set_xlabel(X_KP)
            if c == 0:
                row_label(ax, k, y_wr(k)) if len(ks) > 1 else ax.set_ylabel(y_wr(k))
            else:
                ax.tick_params(labelleft=False)
    # The LightGBM panel's artists stand in for all panels; the colours differ per model only.
    grey = [(Patch(color=tint(INK, 0.25), linewidth=0),
             Line2D([], [], color=SPINE, linewidth=LW, marker='o', markersize=2.4, markeredgewidth=0))]
    fig.legend(handles=grey + legend[0][1:], labels=legend[1], loc='outside lower center', ncol=3)
    save(fig, name, preview)


# --- figure 3: priors -----------------------------------------------------------



def draw_priors(ax, records, k, axis):
    handles = []
    xs = None
    for prior in PRIOR_ARMS:
        x, y, c = full_pool(records[prior], k, axis)
        colour = PRIOR_COLOR[prior]
        h, = ax.plot(x, y, color=colour, linewidth=LW, marker='o', markersize=2.4,
                     markeredgewidth=0, zorder=4, label=PRIOR[prior])
        handles.append(h)
        xs = x
    reference_line(ax)
    return handles, xs


def priors_limits(priors, ks):
    lo, hi = 0.49, 0.5
    for (family, prior, k), rec in priors.items():
        if k not in ks:
            continue
        for axis in ('C', 'N'):
            x, y, c = full_pool(rec, k, axis)
            lo, hi = min(lo, float((y - c).min())), max(hi, float((y + c).max()))
    return (lo - 0.01, hi + 0.015)


def figure_priors(priors, rows, name, figsize, preview):
    """One column per model; one row per (K, axis): win rate of the three priors vs C or vs N."""
    ks = sorted({k for k, _ in rows})
    ylim = priors_limits(priors, ks)
    full = int(max(next(iter(priors.values()))['grid']))
    fig, axes = new_figure(len(rows), len(FAMILIES), figsize)
    handles = None
    for r, (k, axis) in enumerate(rows):
        other = 'N' if axis == 'C' else 'C'
        for c, family in enumerate(FAMILIES):
            ax = axes[r, c]
            records = {p: priors[(family, p, k)] for p in PRIOR_ARMS}
            handles, x = draw_priors(ax, records, k, axis)
            style_axis(ax)
            doubling_ticks(ax, x.min(), x.max(), n_max=4)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(FAMILY[family])
            if r == len(rows) - 1 or rows[r + 1][1] != axis:
                ax.set_xlabel(f"{X_C if axis == 'C' else X_N} (${other} = {full}$)")
            if c == 0 and len(ks) > 1:
                ax.set_ylabel(row_label_text(k))
            elif c > 0:
                ax.tick_params(labelleft=False)
    if len(ks) == 1:                   # one K: one shared label instead of a label per row
        fig.supylabel(y_wr(ks[0]), fontsize=RC['axes.labelsize'])
    fig.legend(handles=handles, loc='outside upper center', ncol=3)
    save(fig, name, preview)


def row_label_text(k):
    return f'$K = {k}$\n{y_wr(k)}'


# --- figure 4: scaling ----------------------------------------------------------

# Red = panel minimum, light grey, green = maximum, as in the analysis repo's heatmap.py.
HEAT = LinearSegmentedColormap.from_list('wr', ['#e0442b', '#f0efec', '#299653'])


def relative_luminance(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(float(c)) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def cell_ink(value, vmin, vmax):
    """White on the darkest (saturated red/green) cells, dark ink everywhere else."""
    return 'white' if relative_luminance(HEAT((value - vmin) / (vmax - vmin))) < 0.25 else INK


def draw_heatmap(ax, rec, k, vmin, vmax):
    """Win-rate surface over (C, N) with the C=K column dropped; returns (image, matrix)."""
    grid, wr, ci = surface(rec, k)
    M = wr[:, 1:]                       # drop the C=K column (0.5 by construction)
    cs, ns = grid[1:], grid
    im = ax.imshow(M, origin='lower', aspect='auto', cmap=HEAT, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    step = 1 if len(cs) <= 5 else 2 if len(cs) <= 10 else 3       # labels counted from the right
    ax.set_xticks(np.arange(len(cs)))
    ax.set_xticklabels([f'{int(v)}' if (len(cs) - 1 - j) % step == 0 else ''
                        for j, v in enumerate(cs)], fontsize=6)
    ax.set_yticks(np.arange(len(ns)))
    ax.set_yticklabels([f'{int(v)}' for v in ns], fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(cs)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ns)), minor=True)
    ax.grid(True, which='minor', color='white', linewidth=0.35)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=2, color=SPINE)
    for side in ax.spines.values():
        side.set_visible(False)
    return im, M


def annotate_heatmap(fig, ax, M, vmin, vmax, fontsize, pad_pt=1.0):
    """Write 3-decimal values (no leading zero) into the cells if they fit at `fontsize`.

    The figure must have been drawn once so the axes have their final size. Returns
    (annotated, cell width pt, label width pt).
    """
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    cell_pt = bbox.width / fig.dpi * 72 / M.shape[1]
    probe = ax.text(0, 0, '.888', fontsize=fontsize)
    label_pt = probe.get_window_extent(renderer).width / fig.dpi * 72
    probe.remove()
    fits = cell_pt >= label_pt + pad_pt
    if fits:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f'{v:.3f}'[1:], ha='center', va='center', fontsize=fontsize,
                        color=cell_ink(v, vmin, vmax))
    return fits, cell_pt, label_pt


def figure_scaling_heatmaps(tuned, k, name, figsize, preview):
    """One heatmap per model at one K, a shared colour scale and colorbar."""
    recs = [(tuned[(pool, k)]['T'], k) for pool in POOLS]
    grids = [surface(rec, k)[1][:, 1:] for rec, _ in recs]
    vmin, vmax = min(g.min() for g in grids), max(g.max() for g in grids)
    fig, axes = new_figure(1, 4, figsize, width_ratios=[1, 1, 1, 0.045])
    axes = axes.reshape(1, 4)
    for c, pool in enumerate(POOLS):
        ax = axes[0, c]
        im, _ = draw_heatmap(ax, tuned[(pool, k)]['T'], k, vmin, vmax)
        ax.set_title(MODEL[pool])
        ax.set_xlabel(X_C)
        if c == 0:
            ax.set_ylabel(X_N)
        else:
            ax.tick_params(labelleft=False)
    cb = fig.colorbar(im, cax=axes[0, 3])
    cb.set_label(y_wr(k))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2, color=SPINE)
    save(fig, name, preview)


def figure_heatmap_grid(tuned, ks, name, figsize, preview, fontsize=5.5):
    """3 x 3 heatmaps (rows K, columns pools), one shared colour scale and colorbar."""
    recs = {(pool, k): tuned[(pool, k)]['T'] for pool in POOLS for k in ks}
    grids = [surface(rec, k)[1][:, 1:] for (pool, k), rec in recs.items()]
    vmin, vmax = min(g.min() for g in grids), max(g.max() for g in grids)
    fig, axes = new_figure(len(ks), len(POOLS), figsize)
    mats = {}
    for r, k in enumerate(ks):
        for c, pool in enumerate(POOLS):
            ax = axes[r, c]
            im, mats[(r, c)] = draw_heatmap(ax, recs[(pool, k)], k, vmin, vmax)
            if r == 0:
                ax.set_title(MODEL[pool])
            if r == len(ks) - 1:
                ax.set_xlabel(X_C)
            if c == 0:
                row_label(ax, k, X_N)
            else:
                ax.tick_params(labelleft=False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, aspect=35, pad=0.015)
    cb.set_label(Y_WR)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2, color=SPINE)
    fig.canvas.draw()                   # final axes sizes, so the cell fit-check is exact
    for (r, c), M in mats.items():
        fits, cell_pt, label_pt = annotate_heatmap(fig, axes[r, c], M, vmin, vmax, fontsize)
        print(f'{name}: K={ks[r]} {MODEL[POOLS[c]]:17s} cells {M.shape[0]}x{M.shape[1]}, '
              f'cell {cell_pt:.1f} pt vs label {label_pt:.1f} pt at {fontsize} pt -> '
              f'{"annotated" if fits else "colour only"}', flush=True)
    save(fig, name, preview)


def draw_family(ax, rec, pool, k, axis):
    x, per_level = family_curves(rec, k, axis)
    levels = sorted(per_level)
    top = levels[-1]
    colour = COLOR[pool]
    for i, level in enumerate(levels[:-1]):
        y, _ = per_level[level]
        t = 0.28 + 0.55 * i / max(1, len(levels) - 2)
        ax.plot(x, y, color=tint(colour, t), linewidth=0.7, zorder=2 + i * 0.01)
    y, c = per_level[top]
    ax.plot(x, y, color=colour, linewidth=LW, marker='o', markersize=2.4, markeredgewidth=0, zorder=4)
    fits = both_fits(x, y)
    draw_fit(ax, x, fits, axis)
    reference_line(ax)
    return fits, x


def family_limits(tuned, ks, axis):
    lo, hi = 0.49, 0.5
    for pool in POOLS:
        for k in ks:
            x, per_level = family_curves(tuned[(pool, k)]['T'], k, axis)
            for y, c in per_level.values():
                lo, hi = min(lo, float((y - c).min())), max(hi, float((y + c).max()))
    return (lo - 0.01, hi + 0.05)


def figure_scaling_grid(tuned, ks, axis, name, figsize, preview):
    ylim = family_limits(tuned, ks, axis)
    other = 'C' if axis == 'N' else 'N'
    full = int(max(tuned[(POOLS[0], ks[0])]['T']['grid']))
    fig, axes = new_figure(len(ks), len(POOLS), figsize)
    fits_out = {}
    for r, k in enumerate(ks):
        for c, pool in enumerate(POOLS):
            ax = axes[r, c]
            fits, x = draw_family(ax, tuned[(pool, k)]['T'], pool, k, axis)
            fits_out[(pool, k)] = fits
            style_axis(ax)
            doubling_ticks(ax, x.min(), x.max())
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(MODEL[pool])
            if r == len(ks) - 1:
                ax.set_xlabel(X_N if axis == 'N' else X_C)
            if c == 0:
                row_label(ax, k, y_wr(k))
            else:
                ax.tick_params(labelleft=False)
    handles = [
        Line2D([], [], color=INK, linewidth=LW, marker='o', markersize=2.4, markeredgewidth=0,
               label=f'${other} = {full}$'),
        Line2D([], [], color=tint(INK, 0.45), linewidth=0.7, label=f'Smaller ${other}$ (lighter = fewer)'),
        Line2D([], [], color=INK, linewidth=0.9, linestyle=(0, (4, 2)),
               label='Fit $a - b/\\sqrt{N}$' if axis == 'N' else 'Fit $a + b\\,\\log_2 C$'),
    ]
    fig.legend(handles=handles, loc='outside upper center', ncol=3)
    save(fig, name, preview)
    return fits_out


# --- figure 5: selection rules ----------------------------------------------------

OURS = 'paired-R16'
SEL_GROUPS = (
    ('Draws $R$', ((OURS, '$R = 16$ (ours)'), ('paired-R8', '$R = 8$'), ('paired-R4', '$R = 4$'),
                   ('paired-R2', '$R = 2$'), ('paired-R1', '$R = 1$'))),
    ('Resampling kept', (('val-only', 'Validation only'), ('ref-only', 'Reference only'),
                         ('plain', 'Neither'))),
    ('TabRepo algorithms', (('tabrepo', 'Normalized error'), ('count', 'Counts'))),
)
SEL_COLUMNS = ((OURS, '16'), ('paired-R8', '8'), ('paired-R4', '4'), ('paired-R2', '2'), ('paired-R1', '1'),
               ('val-only', 'Val.'), ('ref-only', 'Ref.'), ('plain', 'Neither'),
               ('tabrepo', 'Norm.'), ('count', 'Count'))
K_MARKER = {25: 'o', 64: 's', 200: 'D'}
K_TINT = {25: 0.40, 64: 0.70, 200: 1.0}


def load_selection():
    """{('oof', pool, k): {arm: row}} from data/selection/arms.csv (synthetic out-of-fold)."""
    data = defaultdict(dict)
    for r in read_csv(SEL / 'arms.csv'):
        data[('oof', r['model'], r['k'])][r['arm']] = r
    return data


def figure_selection(sel, setting, name, figsize, preview):
    """Win rate against fresh random-K baselines, one row per selection rule, one marker per K,
    +-1.96 standard errors over repetitions; our rule's row is shaded."""
    rows, y, pos, spans = [], [], 0.0, []
    for title, arms in SEL_GROUPS:
        spans.append((title, pos))
        for arm, label in arms:
            rows.append((arm, label)); y.append(pos); pos += 1
        pos += 0.7
    y = -np.asarray(y)
    def xlim_of(pool):
        vals = [(r['winrate'], r['ci95']) for (st, pl, _), arms in sel.items() if st == setting and pl == pool
                for r in arms.values()]
        return (min(m - c for m, c in vals) - 0.004, max(m + c for m, c in vals) + 0.004)
    fig, axes = new_figure(1, len(POOLS), figsize)
    offsets = dict(zip(KS, (0.24, 0.0, -0.24)))
    for c, pool in enumerate(POOLS):
        ax = axes[0, c]
        ax.axhspan(y[0] - 0.45, y[0] + 0.45, color=GRIDC, alpha=0.6, linewidth=0, zorder=0)
        for k in KS:
            arms = sel[(setting, pool, k)]
            m = np.array([arms[a]['winrate'] for a, _ in rows])
            e = np.array([arms[a]['ci95'] for a, _ in rows])
            colour = tint(COLOR[pool], K_TINT[k])
            ax.errorbar(m, y + offsets[k], xerr=e, fmt=K_MARKER[k], markersize=2.6, markeredgewidth=0,
                        color=colour, ecolor=colour, elinewidth=0.8, capsize=0, zorder=3)
        style_axis(ax)
        ax.grid(False, axis='y')
        ax.set_yticks(y)
        ax.set_ylim(y.min() - 0.6, y.max() + 1.0)
        ax.set_xlim(*xlim_of(pool))
        ax.set_title(MODEL[pool])
        if c == 1:
            ax.set_xlabel('Win rate vs. random-$K$ on held-out synthetic tasks ($\\pm$1.96 SE)')
        if c == 0:
            ax.set_yticklabels([label for _, label in rows])
            for title, top in spans:
                ax.annotate(title, xy=(0, -top + 0.55), xycoords=('axes fraction', 'data'),
                            xytext=(-2, 0), textcoords='offset points', fontsize=6.5, color=SPINE,
                            ha='right', va='bottom', style='italic', annotation_clip=False)
        else:
            ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=tint(INK, K_TINT[k] * 0.8), marker=K_MARKER[k], markersize=2.6,
                      markeredgewidth=0, linewidth=0.8, label=f'$K = {k}$') for k in KS]
    fig.legend(handles=handles, loc='outside lower center', ncol=3)
    save(fig, name, preview)


def write_selection_table(sel, path):
    """Win rate vs. random-K of every selection rule, synthetic out-of-fold."""
    cols = 'll' + 'c' * len(SEL_COLUMNS)
    lines = [rf'\begin{{tabular}}{{{cols}}}', r'\toprule',
             r' & & \multicolumn{5}{c}{Draws $R$} & \multicolumn{3}{c}{Resampling kept} & '
             r'\multicolumn{2}{c}{TabRepo} \\',
             r'\cmidrule(lr){3-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}',
             'Model & $K$ & ' + ' & '.join(label for _, label in SEL_COLUMNS) + r' \\']
    lines.append(r'\midrule')
    for i, pool in enumerate(POOLS):
        if i:
            lines.append(r'\addlinespace')
        for j, k in enumerate(KS):
            arms = sel[('oof', pool, k)]
            vals = [f"{arms[a]['winrate']:.3f}" for a, _ in SEL_COLUMNS]
            best = max(vals)
            cells = [rf'\textbf{{{v}}}' if v == best else v for v in vals]
            lines.append(f"{MODEL[pool] if j == 0 else ''} & {k} & " + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {path} ({path.stat().st_size:,} bytes)', flush=True)


def report_selection(sel):
    print('\n=== Selection rules, synthetic OOF, vs fresh random-K: win rate (change vs R_1 baseline) ===')
    arms = [a for a, _ in SEL_COLUMNS]
    for pool in POOLS:
        for k in KS:
            r = sel[('oof', pool, k)]
            cells = ' '.join(f"{a}={r[a]['winrate']:.3f}±{r[a]['ci95']:.3f}({r[a]['winrate'] - r[a]['winrate_vs_R1']:+.3f})"
                             for a in arms)
            print(f'{MODEL[pool]:17s} K={k:<4d} {cells}')
    print('-- monotone in R:', all(
        np.all(np.diff([sel[('oof', p, k)][a]['winrate'] for a in arms[4::-1]]) > 0)
        for p in POOLS for k in KS))


# --- figure 6: selected hyperparameters vs the sampling distribution -------------------

_LGBM = (
    ('learning_rate', 'Learning rate', 'log', (5e-3, 1e-1)),
    ('feature_fraction', 'Feature fraction', 'lin', (0.4, 1.0)),
    ('bagging_fraction', 'Bagging fraction', 'lin', (0.7, 1.0)),
    ('num_leaves', '# leaves', 'log', (2, 200)),
    ('min_data_in_leaf', 'Min. data in leaf', 'log', (1, 64)),
    ('extra_trees', 'Extra trees', 'cat', (False, True)),
    ('min_data_per_group', 'Min. data per group', 'log', (2, 100)),
    ('cat_l2', 'Categorical L2', 'log', (5e-3, 2.0)),
    ('cat_smooth', 'Categorical smoothing', 'log', (1e-3, 100)),
    ('max_cat_to_onehot', 'Max. cat. to one-hot', 'log', (8, 100)),
    ('lambda_l1', 'L1 regularization', 'lin', (1e-4, 1.0)),
    ('lambda_l2', 'L2 regularization', 'lin', (1e-4, 2.0)),
)
# Sampling distributions of the banks (Appendix search-space tables; the samplers are
# bin/models/_sample_hp.py and bin/models/mlp/sample_hp.py). TabPrep choices are drawn only for the one-in-four configs
# with TabPrep enabled; their generator seeds are not hyperparameters and are not shown.
HP_SPEC = {
    'lgbm': _LGBM,
    'lgbm-prep': _LGBM + (
        ('tabprep_enabled', 'TabPrep enabled', 'cat', (False, True)),
        ('oafe_max_feats', 'OpenFE max. feat.', 'prep', (2000, 1000)),
        ('cat_combine_max_feats', 'Cat. comb. max. feat.', 'prep', (100, 500)),
        ('rsfc_n_subsets', 'Random subsets', 'prep', (50, 1, 100)),
        ('groupby_max_feats', 'Group-by max. feat.', 'prep', (500, 100, 1000))),
    'mlp': (
        ('lr', 'Learning rate (AdamW)', 'log', (1e-4, 5e-3)),
        ('weight_decay', 'Weight decay', 'log', (1e-3, 1.0)),
        ('muon_lr', 'Learning rate (Muon)', 'log', (1e-3, 1e-1)),
        ('init_scale', 'Embedding init. scale', 'log', (1e-2, 1e1)),
        ('dropout', 'Dropout rate', 'dropout', (0.0, 0.5)),
        ('d_embedding', 'Embedding dim', 'cat', tuple(range(8, 33, 4))),
        ('n_blocks', '# blocks', 'cat', (1, 2, 3, 4))),
}
CAT_PRIOR = {'tabprep_enabled': (0.75, 0.25)}
# Main-text panels (LightGBM-TabPrep only in the appendix): for LightGBM the three continuous
# HPs farthest from their sampling distribution by Kolmogorov-Smirnov distance (report_hp);
# for the MLP the two farthest and the embedding dimension, whose share grows with its value.
HP_MAIN = {'lgbm': ('learning_rate', 'min_data_in_leaf', 'lambda_l1'),
           'mlp': ('muon_lr', 'lr', 'd_embedding')}
PRIOR_GREY = '#b9b8b3'


def flatten(model, config):
    """The sampled hyperparameters of one bank entry, by the names of HP_SPEC."""
    if model == 'mlp':
        o, m = config['optimizer'], config['model']
        return dict(lr=o['lr'], weight_decay=o['weight_decay'], muon_lr=o['muon_lr'],
                    init_scale=m['num_embeddings']['init_scale'], dropout=m['dropout'],
                    d_embedding=m['num_embeddings']['d_embedding'], n_blocks=m['n_blocks'])
    row = dict(config['lightgbm'])
    if model == 'lgbm-prep':
        row['tabprep_enabled'] = bool(config['tabprep_enabled'])
        for key in ('oafe_max_feats', 'cat_combine_max_feats', 'rsfc_n_subsets', 'groupby_max_feats'):
            row[key] = config['tabprep'][key] if config['tabprep_enabled'] else ''
    return row


def load_hp():
    """{model: {hp: the 200 values selected under T}}; '' marks TabPrep fields of configs
    without TabPrep. None when the artifacts are not downloaded."""
    out = {}
    for model in HP_SPEC:
        try:
            ids = artifacts.portfolio(model, 'T', 200)['config_ids']
            bank = artifacts.configs(model)
        except FileNotFoundError as missing:
            print(f'skipping the hyperparameter figures: {missing}')
            return None
        rows = [flatten(model, bank[i]) for i in ids]
        assert len(rows) == 200
        out[model] = {key: [r[key] for r in rows] for key in rows[0]}
    return out


def unit(values, kind, lo, hi):
    """Selected values mapped to [0, 1] under the sampling CDF (uniform in log space for 'log')."""
    x = np.asarray(values, float)
    return (np.log10(x / lo) / np.log10(hi / lo)) if kind == 'log' else (x - lo) / (hi - lo)


def ks_distance(u):
    """Kolmogorov-Smirnov distance of [0, 1]-mapped values from the uniform distribution."""
    u = np.sort(u)
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


def reflected_kde(data, grid, lo, hi):
    """Gaussian KDE (Scott bandwidth) reflected at both bounds, so no mass leaks out of [lo, hi]."""
    data = np.asarray(data, float)
    bw = data.std(ddof=1) * len(data) ** (-1 / 5)
    def k(x):
        return np.exp(-0.5 * ((x[:, None] - data[None, :]) / bw) ** 2).sum(1) / (len(data) * bw * np.sqrt(2 * np.pi))
    return k(grid) + k(2 * lo - grid) + k(2 * hi - grid)


def as_bool(v):
    return v if isinstance(v, bool) else str(v) == 'True'


def draw_hp(ax, values, kind, arg, colour, key):
    """One hyperparameter: sampling distribution (grey) against the 200 selected values (colour).

    Continuous: the KDE of F(x) over the selected values, F the sampling CDF, which is the density
    ratio f_sel(x) / f(x); the sampling distribution is flat at 1. KS is invariant to F."""
    style_axis(ax)
    ax.grid(False, axis='x')
    if kind in ('log', 'lin'):
        lo, hi = arg
        u = unit(values, kind, lo, hi)
        grid = np.linspace(0, 1, 300)
        dens = reflected_kde(u, grid, 0.0, 1.0)
        x = lo * (hi / lo) ** grid if kind == 'log' else lo + grid * (hi - lo)
        ax.fill_between(x, 1.0, color=PRIOR_GREY, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(x, dens, color=colour, linewidth=LW, zorder=3)
        top = max(1.0, float(dens.max()))
        xs = lo * (hi / lo) ** u if kind == 'log' else lo + u * (hi - lo)
        ax.plot(xs, np.full_like(xs, -0.06 * top), '|', color=colour, alpha=0.5, markersize=3, zorder=2)
        ax.set_ylim(-0.12 * top, top * 1.12)
        if kind == 'log':
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(lo, hi)
        ax.text(0.97, 0.95, f'KS {ks_distance(u):.2f}', transform=ax.transAxes, fontsize=5.5,
                color=SPINE, ha='right', va='top', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.6))
    elif kind == 'dropout':
        lo, hi = arg
        v = np.asarray(values, float)
        pos = v[v > 0]
        grid = np.linspace(0, 1, 300)
        dens = reflected_kde(pos / hi, grid, 0.0, 1.0) * len(pos) / len(v) * 2   # continuous half; flat prior = 1
        ax.fill_between(grid * hi, 1.0, color=PRIOR_GREY, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(grid * hi, dens, color=colour, linewidth=LW, zorder=3)
        ax.set_xlim(lo, hi)
        ax.text(0.97, 0.95, f'P(0): 0.50 / {np.mean(v == 0):.2f}', transform=ax.transAxes, fontsize=5.5,
                color=SPINE, ha='right', va='top', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.6))
    else:
        options = list(arg)
        if kind == 'prep':
            keep = [v for v in values if v != '']
        else:
            keep = [as_bool(v) for v in values] if isinstance(options[0], bool) else list(values)
        prior = np.asarray(CAT_PRIOR.get(key, [1 / len(options)] * len(options)))
        share = np.array([np.mean([v == o for v in keep]) for o in options])
        xi = np.arange(len(options))
        ax.bar(xi - 0.2, prior, width=0.4, color=PRIOR_GREY, alpha=0.8, linewidth=0, zorder=2)
        ax.bar(xi + 0.2, share, width=0.4, color=colour, linewidth=0, zorder=3)
        ax.set_xticks(xi, [str(o) for o in options])
        ax.set_xlim(-0.6, len(options) - 0.4)
        return len(keep) if kind == 'prep' else None


def hp_legend(fig, colour_note='model colour'):
    handles = [Patch(color=PRIOR_GREY, alpha=0.8, linewidth=0),
               Line2D([], [], color=INK, linewidth=LW)]
    fig.legend(handles=handles, labels=['Sampling distribution',
                                        f'200 selected configurations ({colour_note})'],
               loc='outside lower center', ncol=2)


def figure_hp_main(hp, name, figsize, preview):
    fig, axes = new_figure(len(HP_MAIN), 3, figsize)
    for r, pool in enumerate(HP_MAIN):
        spec = {k: (label, kind, arg) for k, label, kind, arg in HP_SPEC[pool]}
        for c, key in enumerate(HP_MAIN[pool]):
            label, kind, arg = spec[key]
            ax = axes[r, c]
            draw_hp(ax, hp[pool][key], kind, arg, COLOR[pool], key)
            ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f'{MODEL[pool]}\nDensity ratio')
            elif kind == 'cat':
                ax.set_ylabel('Share')
    hp_legend(fig)
    save(fig, name, preview)


def figure_hp_pool(hp, pool, name, preview, ncols=4):
    spec = HP_SPEC[pool]
    nrows = int(np.ceil(len(spec) / ncols))
    fig, axes = new_figure(nrows, ncols, (5.5, 1.3 * nrows + 0.3))
    for i, (ax, (key, label, kind, arg)) in enumerate(zip(axes.flat, spec)):
        n = draw_hp(ax, hp[pool][key], kind, arg, COLOR[pool], key)
        ax.set_title(label if n is None else f'{label}\n($n={n}$ with TabPrep)', fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel('Share' if kind in ('cat', 'prep') else 'Density ratio')
    for ax in list(axes.flat)[len(spec):]:
        ax.set_axis_off()
    hp_legend(fig, MODEL[pool])
    save(fig, name, preview)


def report_hp(hp):
    print('\n=== Selected hyperparameters vs sampling distribution (KS; share in lowest / highest tenth) ===')
    for pool in POOLS:
        rows = []
        for key, label, kind, arg in HP_SPEC[pool]:
            if kind in ('log', 'lin'):
                u = unit(hp[pool][key], kind, *arg)
                rows.append((ks_distance(u), label, float(np.mean(u < 0.1)), float(np.mean(u > 0.9))))
        for ks, label, low, high in sorted(rows, reverse=True):
            print(f'{MODEL[pool]:17s} {label:28s} KS {ks:.3f}  low {low:.2f}  high {high:.2f}')


# --- table of fits ----------------------------------------------------------------

TABLES = ROOT / 'figures/tables'


def report(tuned, priors, budget):
    print("\n=== Ladder: random-K' vs random-K, original pool and single draws from 16,000; selected K subsets ===")
    print(f'{"model":18s} {"K":>4s} | {"pool peak (K\')":>16s} {"pool @4000":>11s} | '
          f'{"draws mean peak":>16s} {"p97.5 peak (K\')":>17s} {"max share>=line":>15s} | '
          f'{"synthetic":>16s} {"TabRepo-182":>16s}')
    for pool in POOLS:
        for k in KS:
            L = ladder_series(tuned[(pool, k)], k)
            x = L['x']
            i, j, h = int(np.argmax(L['pool'])), int(np.argmax(L['mean'])), int(np.argmax(L['hi']))
            (s_wr, s_ci), (r_wr, r_ci) = L['synth'], L['repo']
            print(f'{MODEL[pool]:18s} {k:4d} | {L["pool"][i]:.4f} ({int(x[i]):4d})   {L["pool"][-1]:.4f}     | '
                  f'{L["mean"][j]:.4f} ({int(x[j]):4d})   {L["hi"][h]:.4f} ({int(x[h]):4d})    '
                  f'{L["share"].max():.3f}        | {s_wr:.4f} ± {s_ci:.4f}  {r_wr:.4f} ± {r_ci:.4f}')

    print('\n=== Fits on the full-pool curves (least squares) ===')
    print('vs N at C=4000:   (a) wr = a - b/sqrt(N)          (b) wr = a + b log2 N')
    print('vs C at N=4000:   (c) wr = a + b log2 C           (d) wr = a - b/sqrt(C)')
    print(f'{"model":18s} {"K":>4s} | {"a":>7s} {"b":>7s} {"R2":>5s} | {"a":>7s} {"b":>7s} {"R2":>5s} | '
          f'{"a":>7s} {"b":>7s} {"R2":>5s} | {"a":>7s} {"b":>7s} {"R2":>5s}')
    for pool in POOLS:
        for k in KS:
            rec = tuned[(pool, k)]['T']
            fn = both_fits(*full_pool(rec, k, 'N')[:2])
            fc = both_fits(*full_pool(rec, k, 'C')[:2])
            a, b, c_ = fn['isqrt'], fn['log'], fc['log']
            d = fc['isqrt']
            print(f'{MODEL[pool]:18s} {k:4d} | {a["a"]:7.4f} {-a["b"]:7.4f} {a["r2"]:5.2f} | '
                  f'{b["a"]:7.4f} {b["b"]:7.4f} {b["r2"]:5.2f} | '
                  f'{c_["a"]:7.4f} {c_["b"]:7.4f} {c_["r2"]:5.2f} | '
                  f'{d["a"]:7.4f} {-d["b"]:7.4f} {d["r2"]:5.2f}')

    print('\n=== Full-pool win rates (top-right cell, ±95% CI) ===')
    for pool in POOLS:
        for k in KS:
            grid, wr, ci = surface(tuned[(pool, k)]['T'], k)
            print(f'{MODEL[pool]:18s} K={k:<4d} C=N={int(grid[-1])}: {wr[-1, -1]:.4f} ± {ci[-1, -1]:.4f}')
    for family in FAMILIES:
        for prior in PRIOR_ARMS:
            for k in KS:
                grid, wr, ci = surface(priors[(family, prior, k)], k)
                print(f'{FAMILY[family]:18s} {PRIOR[prior]:9s} K={k:<4d} C=N={int(grid[-1])}: '
                      f'{wr[-1, -1]:.4f} ± {ci[-1, -1]:.4f}')

    print('\n=== Budget: TabRepo-182 envelope from the 16,000-config bank vs the synthetic envelope ===')
    for pool in POOLS:
        for k in KS:
            series, n_meta, width, corner = budget_series(
                budget[(pool, k)], tuned[(pool, k)]['tabrepo16000'], k)
            sx, sy, sex, sey = series['synthetic']
            tx, ty, tex, tey = series['tabrepo']
            i = int(np.argmax(tey))
            e_max = int(np.log2(tex[i]))
            last = ', '.join(f'2^{int(np.log2(b))}: {w:.4f}' for b, w in zip(tex[-3:], tey[-3:]))
            s_at = sey[np.isclose(sex, tex[i])]
            reach = sex[sey >= tey[i]]
            at = f'2^{int(np.log2(reach[0]))}' if len(reach) else 'never'
            print(f'{MODEL[pool]:17s} K={k:<4d} TabRepo max {tey[i]:.4f} at 2^{e_max}; last bins {last}; '
                  f'final cell C={width:,} x N={n_meta} (B={int(tx.max()):,} = 2^{np.log2(tx.max()):.1f}): {corner:.4f} | '
                  f'synthetic at 2^{e_max}: {s_at[0] if len(s_at) else float("nan"):.4f}, at 2^24: {sey[-1]:.4f}, '
                  f'reaches TabRepo max at {at}; ends above TabRepo max: {sey[-1] > tey[i]}')


# --- main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preview', type=Path, default=None,
                        help='also write PNG previews of every figure into this directory')
    args = parser.parse_args()
    matplotlib.rcParams.update(RC)

    tuned, priors, budget = load_tuned(), load_priors(), load_budget()
    pv = args.preview

    figure_budget(budget, tuned, (200,), 'budget_K200', (5.5, 1.9), pv)
    figure_budget(budget, tuned, KS, 'budget_grid', (5.5, 5.3), pv)
    figure_ladder(tuned, (200,), 'ladder_K200', (5.5, 1.9), pv)
    figure_ladder(tuned, KS, 'ladder_grid', (5.5, 5.3), pv)
    figure_priors(priors, ((200, 'C'), (200, 'N')), 'priors_K200', (5.5, 2.9), pv)
    figure_priors(priors, tuple((k, 'C') for k in KS), 'priors_grid_configs', (5.5, 5.0), pv)
    figure_priors(priors, tuple((k, 'N') for k in KS), 'priors_grid_datasets', (5.5, 5.0), pv)
    figure_scaling_heatmaps(tuned, 200, 'scaling_K200_heatmaps', (5.5, 1.9), pv)
    figure_scaling_grid(tuned, KS, 'N', 'scaling_grid_datasets', (5.5, 5.3), pv)
    figure_scaling_grid(tuned, KS, 'C', 'scaling_grid_configs', (5.5, 5.3), pv)
    figure_heatmap_grid(tuned, KS, 'scaling_heatmaps_grid', (5.5, 5.6), pv)
    sel = load_selection()
    figure_selection(sel, 'oof', 'selection_oof', (5.5, 2.3), pv)
    write_selection_table(sel, TABLES / 'selection_ablation.tex')
    report(tuned, priors, budget)
    report_selection(sel)
    hp = load_hp()
    if hp is not None:
        figure_hp_main(hp, 'hp_main', (5.5, 2.8), pv)
        for pool in POOLS:
            figure_hp_pool(hp, pool, f'hp_{pool}', pv)
        report_hp(hp)


if __name__ == '__main__':
    main()
