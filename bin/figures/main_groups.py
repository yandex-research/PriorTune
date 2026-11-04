#!/usr/bin/env python3
"""Results by BeyondArena dataset group on the full BeyondArena core: Figure 3 and its appendix twin.

Both figures share one design: a tuned (T) panel with the FT subsets and a tuned + ensembled (T+E)
panel with the FTE subsets, each followed by a strip of bars with the gain of each meta-tuned subset
over the same model's random subset under the same rule. Groups are BeyondArena's sub-benchmarks
(data/main/groups.csv); a dataset can be in several. References: RealMLP (both panels) and
Kumo-Tabular (default, T+E panel). LightGBM's random subset is the benchmark's published LightGBM.

  figures/main-groups-elo          y = Elo of the 58-method field recomputed on each group's tasks
                                   (data/main/elo_groups.csv);
                                   gain = Elo difference.
  figures/main-groups-improvement  y = median over the group's datasets of the per-dataset improvement
                                   over the tuned MLP with its random subset, 100 (err_base - err) /
                                   err_base averaged over the dataset's splits (a split with base error 0
                                   is left out; data/main/full_ba_per_task.csv); gain = difference of
                                   the two medians.
  figures/tables/main_groups.tex   the medians behind the improvement figure, and the paired win rate
                                   of each meta-tuned subset against its random subset.

    python bin/figures/main_groups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablations import RC, COLOR, INK, SPINE, style_axis  # noqa: E402

DATA = ROOT / 'data/main'
GROUPS = (('Task', ('IID', 'Grouped', 'Temporal')),
          ('Scale', ('Tiny', 'Small', 'Medium', 'Large')),
          ('Dimensionality', ('Low-dim', 'High-dim')),
          ('Features', ('Text', 'High card.')),
          ('', ('All',)))
ORDER = [g for _, gs in GROUPS for g in gs]
TICK = {'Low-dim': 'Low', 'High-dim': 'High'}
BASELINE = 'MLP-Cos R25 (tuned)'
HIGHLIGHT = 'LightGBM-TabPrep'

#: model -> (colour, marker); Figure 1 colours (elo_latency_figure.FAMILY_COLOR)
STYLE = {
    'Kumo-Tabular': (INK, 'D'),
    'RealMLP': ('#8a8983', '^'),
    'LightGBM-TabPrep': (COLOR['lgbm-prep'], 's'),
    'TabPack': ('#c2185b', 'o'),
    'MLP': (COLOR['mlp'], 's'),
    'LightGBM': (COLOR['lgbm'], 's'),
}
#: panel -> (rule, subset, [(model, meta-tuned or reference method, random-subset method or None)])
PANELS = (
    ('T', 'FT', [('RealMLP', 'TA-REALMLP (tuned)', None),
                 ('LightGBM-TabPrep', 'LGBM-P FT-25 (tuned)', 'LGBM-P R25·25% (tuned)'),
                 ('MLP', 'MLP-Cos FT-25 (tuned)', 'MLP-Cos R25 (tuned)'),
                 ('LightGBM', 'LightGBM FT-25 (tuned)', 'GBM (tuned)')]),
    ('T+E', 'FTE', [('Kumo-Tabular', 'TA-KUMO-TABULAR (default)', None),
                    ('RealMLP', 'TA-REALMLP (tuned + ensemble)', None),
                    ('LightGBM-TabPrep', 'LGBM-P FTE-25 (tuned + ensemble)', 'LGBM-P R25·25% (tuned + ensemble)'),
                    ('TabPack', 'TabPack FTE64 (default)', 'TabPack R64 (default)'),
                    ('MLP', 'MLP-Cos FTE-25 (tuned + ensemble)', 'MLP-Cos R25 (tuned + ensemble)'),
                    ('LightGBM', 'LightGBM FTE-25 (tuned + ensemble)', 'GBM (tuned + ensemble)')]),
)
TITLE = {'T': 'Tuned (T): FT subsets', 'T+E': 'Tuned + ensembled (T+E): FTE subsets'}


# ---- data ------------------------------------------------------------------------------------
def members():
    groups = pd.read_csv(DATA / 'groups.csv')
    m = {g: sorted(groups.loc[groups.group == g, 'dataset']) for g in groups.group.unique()}
    assert set(m) == set(ORDER) and len(m['All']) == 142
    return m


def group_elo():
    """(method, group) -> Elo; group -> number of datasets."""
    d = pd.read_csv(DATA / 'elo_groups.csv')
    assert set(d.group) == set(ORDER) and d.groupby('group').method.nunique().eq(58).all()
    return d.set_index(['method', 'group']).elo.to_dict(), d.drop_duplicates('group').set_index('group').n_datasets


def errors():
    d = pd.read_csv(DATA / 'full_ba_per_task.csv', float_precision='round_trip')
    return d.pivot_table(index=['dataset', 'fold'], columns='method', values='metric_error')


def improvements(err):
    """Per (dataset, method): % improvement over the baseline, averaged over the dataset's splits.

    As in BeyondArena's render_bench_improvement.improvement_by_dataset: a split where the baseline's
    error is 0 has no relative improvement and is left out.
    """
    base = err[BASELINE]
    assert err.notna().all().all()
    imp = (100.0 * (base.to_numpy()[:, None] - err) / base.to_numpy()[:, None])[base > 0]
    imp = imp.groupby(level='dataset').mean()
    assert len(imp) == 142
    return imp


def win_rates(err):
    """Per dataset and (model, rule): paired win rate of the meta-tuned subset vs its random subset.

    Per split 1 / 1/2 / 0 for win / tie / loss, averaged over the dataset's splits.
    """
    out = {}
    for rule, _, lines in PANELS:
        for model, key, rand in lines:
            if rand:
                a, b = err[key], err[rand]
                out[(model, rule)] = ((a < b).astype(float) + 0.5 * (a == b)).groupby(level='dataset').mean()
    return pd.DataFrame(out)


# ---- figure ----------------------------------------------------------------------------------
def positions():
    """x of each group, with a half-step gap between the axes of groups; span of each axis."""
    xs, x, spans = {}, 0.0, []
    for title, gs in GROUPS:
        start = x
        for g in gs:
            xs[g] = x
            x += 1.0
        spans.append((title, start, x - 1.0))
        x += 0.5
    return xs, spans


def frame(ax, spans, ylabel, titles=False):
    style_axis(ax)
    ax.grid(False, axis='x')
    for i, (_, _, end) in enumerate(spans[:-1]):
        ax.axvline((end + spans[i + 1][1]) / 2, color=SPINE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    if titles:
        for title, start, end in spans:
            if title:
                ax.text((start + end) / 2, 1.0, title, transform=ax.get_xaxis_transform(), fontsize=6.5,
                        fontweight='bold', color=SPINE, ha='center', va='top')
    ax.set_ylabel(ylabel)


def line_kw(model):
    colour, marker = STYLE[model]
    return dict(color=colour, marker=marker, markersize=3.6, markeredgewidth=0.9, markeredgecolor=colour,
                markerfacecolor=colour, linewidth=1.7 if model == HIGHLIGHT else 1.1,
                linestyle=(0, (1, 1.2)) if model == 'RealMLP' else '-')


def broken(upper, lower):
    """Axis-break marks between `upper` and `lower` (a cut y axis)."""
    upper.spines['bottom'].set_visible(False)
    upper.tick_params(axis='x', which='both', bottom=False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=6, linestyle='none', color=SPINE, mec=SPINE, mew=0.8,
              clip_on=False)
    upper.plot([0], [0], transform=upper.transAxes, **kw)
    lower.plot([0], [1], transform=lower.transAxes, **kw)


def draw(value, size, ylabel, unit, out, zero=False, cut=None, height=5.0):
    """`value[(method, group)]`; the strips show value(meta-tuned) - value(random subset).

    ``cut=(a, b)`` cuts the T+E axis between a and b (Kumo-Tabular sits far above our models).
    """
    xs, spans = positions()
    ratios = [1.0, 0.42] + ([0.5, 1.15] if cut else [1.45]) + [0.42]
    fig = plt.figure(figsize=(7.0, height), layout='constrained')
    fig.get_layout_engine().set(h_pad=2 / 72, hspace=0.04)
    axes = list(fig.subplots(len(ratios), 1, sharex=True, height_ratios=ratios))
    layout = [(axes[:1], axes[1]), (axes[2:-1], axes[-1])]   # per panel: value axes (top first), strip
    gains = {}
    for r, ((rule, subset, lines), (vaxes, strip)) in enumerate(zip(PANELS, layout)):
        vals = [0.0] if zero else []
        for model, key, _ in lines:
            vals += [value[key, g] for g in ORDER]
        lo, hi = min(vals), max(vals)
        top_pad = 0.16 if r == 0 else 0.06   # room for the group titles in the top panel
        for i, ax in enumerate(vaxes):
            frame(ax, spans, ylabel if i == len(vaxes) - 1 else '', titles=r == 0)
            if zero:
                ax.axhline(0, color=SPINE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            for model, key, _ in lines:
                for _, gs in GROUPS:   # lines connect groups within an axis only
                    ax.plot([xs[g] for g in gs], [value[key, g] for g in gs],
                            zorder=4 if model == HIGHLIGHT else 3, **line_kw(model))
        if len(vaxes) == 1:
            vaxes[0].set_ylim(lo - 0.06 * (hi - lo), hi + top_pad * (hi - lo))
        else:
            a, b = cut
            assert not any(a < v < b for v in vals), 'a value falls into the cut'
            vaxes[0].set_ylim(b, hi + top_pad * (hi - b))
            vaxes[1].set_ylim(lo - 0.06 * (a - lo), a)
            broken(*vaxes)
        vaxes[0].set_title(TITLE[rule], loc='left', fontsize=7.5)

        frame(strip, spans, f'Gain over\nrandom ({unit})')
        strip.axhline(0, color=SPINE, linewidth=0.8, zorder=1)
        pairs = [(m, k, rd) for m, k, rd in lines if rd]
        w = 0.62 / len(pairs)
        for i, (model, key, rand) in enumerate(pairs):
            gain = [value[key, g] - value[rand, g] for g in ORDER]
            gains[(model, rule)] = gain
            strip.bar([xs[g] + (i - (len(pairs) - 1) / 2) * w for g in ORDER], gain, width=w * 0.86,
                      color=STYLE[model][0], linewidth=0, zorder=3)
        allg = [v for m, _, _ in pairs for v in gains[(m, rule)]]
        lim = max(abs(v) for v in allg)
        strip.set_ylim(min(min(allg), 0) - 0.12 * lim, max(max(allg), 0) + 0.12 * lim)
        strip.set_title(f'{subset} subset minus random subset, same model and rule', loc='left', fontsize=6.5,
                        color=SPINE)
    axes[-1].set_xticks([xs[g] for g in ORDER])
    axes[-1].set_xticklabels([f'{TICK.get(g, g)}\n$N={size[g]}$' for g in ORDER], fontsize=6.0)
    axes[-1].set_xlim(-0.55, max(xs.values()) + 0.55)
    handles = [Line2D([], [], **line_kw(m)) for m in STYLE]
    fig.legend(handles=handles, labels=list(STYLE), loc='outside upper center', ncol=len(STYLE), handlelength=1.8,
               columnspacing=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=220)
    plt.close(fig)
    print(f'wrote {out}.pdf/.png')
    return gains


def report(title, value, gains, fmt):
    print(f'\n== {title}')
    print(f'{"":30s}' + ''.join(f'{g[:9]:>10s}' for g in ORDER))
    for key, label in (('TA-KUMO-TABULAR (default)', 'Kumo-Tabular'),
                       ('LGBM-P FTE-25 (tuned + ensemble)', 'LightGBM-TabPrep FTE (T+E)')):
        print(f'{label:30s}' + ''.join(f'{value[key, g]:10{fmt}}' for g in ORDER))
    print('-- gain over own random subset')
    for (model, rule), gain in gains.items():
        print(f'{model + " (" + rule + ")":30s}' + ''.join(f'{v:+10{fmt}}' for v in gain))


# ---- table -----------------------------------------------------------------------------------
#: median rows (method, label) per rule; the random subsets next to the meta-tuned ones
TABLE_MEDIANS = (
    ('Tuned (T)', (('TA-REALMLP (tuned)', 'RealMLP'),
                   ('LGBM-P FT-25 (tuned)', 'LightGBM-TabPrep, FT'), ('LGBM-P R25·25% (tuned)', 'LightGBM-TabPrep, random'),
                   ('LightGBM FT-25 (tuned)', 'LightGBM, FT'), ('GBM (tuned)', 'LightGBM, random'),
                   ('MLP-Cos FT-25 (tuned)', 'MLP, FT'))),
    ('Tuned + ensembled (T+E)', (('TA-KUMO-TABULAR (default)', 'Kumo-Tabular (default)'),
                                 ('TA-REALMLP (tuned + ensemble)', 'RealMLP'),
                                 ('LGBM-P FTE-25 (tuned + ensemble)', 'LightGBM-TabPrep, FTE'),
                                 ('LGBM-P R25·25% (tuned + ensemble)', 'LightGBM-TabPrep, random'),
                                 ('TabPack FTE64 (default)', 'TabPack, FTE'), ('TabPack R64 (default)', 'TabPack, random'),
                                 ('LightGBM FTE-25 (tuned + ensemble)', 'LightGBM, FTE'),
                                 ('GBM (tuned + ensemble)', 'LightGBM, random'),
                                 ('MLP-Cos FTE-25 (tuned + ensemble)', 'MLP, FTE'),
                                 ('MLP-Cos R25 (tuned + ensemble)', 'MLP, random'))))


def write_table(path, med, wins, mem):
    ncol = len(ORDER)
    head = ' & '.join(TICK.get(g, g).replace('High card.', 'High card') for g in ORDER)
    sizes = ' & '.join(f'{len(mem[g])}' for g in ORDER)
    lines = [r'\begin{table}[h]',
             r'\caption{Values behind Figure~\ref{fig:main-improvement}: full BeyondArena, by BeyondArena dataset '
             r'group. Median per-dataset improvement (\%) over the tuned MLP with its random subset (the baseline, '
             r'0 by definition), and paired win rate of each meta-tuned subset against the same model with its random '
             r'subset under the same rule. The bars of Figure~\ref{fig:main-improvement} are differences of the '
             r'median rows. Win rates give a tie half credit and average each dataset over its splits.}',
             r'\label{tab:main-groups}',
             r'\begin{center}', r'\setlength{\tabcolsep}{1.5pt}', r'\scriptsize',
             # shrink only if wider than the text (11pt layouts), never enlarge
             r'\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%',
             r'\begin{tabular}{l' + 'r' * ncol + '}', r'\toprule']
    spans, col = [], 2
    for title, gs in GROUPS:
        if title:
            spans.append((title, col, col + len(gs) - 1))
        col += len(gs)
    lines += [' & ' + ' & '.join(rf'\multicolumn{{{b - a + 1}}}{{c}}{{{t}}}' for t, a, b in spans) + r' & \\',
              ' '.join(rf'\cmidrule(lr){{{a}-{b}}}' for _, a, b in spans),
              f' & {head} \\\\', f'Datasets & {sizes} \\\\']

    def signed(v):
        v = round(v, 2)
        return '$0.00$' if v == 0 else f'${v:+.2f}$'

    for title, rows in TABLE_MEDIANS:
        lines += [r'\midrule', rf'\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{Median improvement, {title}}}}} \\']
        lines += [f'{label} & ' + ' & '.join(signed(med[key, g]) for g in ORDER) + r' \\' for key, label in rows]
    lines += [r'\midrule', rf'\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{Win rate, meta-tuned vs.\ own random subset}}}} \\']
    for rule, subset, rows in PANELS[::-1]:
        for model, _, rand in rows:
            if rand:
                lines.append(f'{model}, {subset} ({rule}) & '
                             + ' & '.join(f'{wins.loc[mem[g], (model, rule)].mean():.2f}' for g in ORDER) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{center}', r'\end{table}']
    path.write_text('\n'.join(lines).replace('·', r'$\cdot$') + '\n')
    print(f'wrote {path}')


def main():
    matplotlib.rcParams.update(RC)
    mem = members()
    elo, size = group_elo()
    assert all(size[g] == len(mem[g]) for g in ORDER)
    gains = draw(elo, size, 'Elo', 'Elo', ROOT / 'figures/main-groups-elo')
    report('Elo', elo, gains, '.1f')

    err = errors()
    imp = improvements(err)
    med = {(key, g): float(np.median(imp.loc[mem[g], key])) for key in imp.columns for g in ORDER}
    gains = draw(med, size, 'Median improvement\nover tuned MLP (%)', 'pp', ROOT / 'figures/main-groups-improvement',
                 zero=True, cut=(2.6, 3.6), height=5.3)
    report('Median improvement over tuned MLP (%); gain = difference of medians (pp)', med, gains, '.2f')
    write_table(ROOT / 'figures/tables/main_groups.tex', med, win_rates(err), mem)


if __name__ == '__main__':
    main()
