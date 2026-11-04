"""Appendix figure: the full BeyondArena Elo leaderboard (all 58 methods of data/main/elo_full_ba.csv).

The left panel of Figure 1 (elo_latency.py) turned on its side and without its
exclusions: one column per method name, the Default / Tuned / Tuned + Ensembled bars drawn over each
other from Elo 800 with their bootstrap CIs, columns sorted by their best Elo. Labels of our families
are coloured as in Figure 1, our meta-tuned subsets (FT / FTE) in bold.

    python bin/figures/elo_full_field.py

Writes figures/elo-full-field.{pdf,png}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from elo_latency import (BAR_COLOR, BAR_LABEL, ELO_MIN, ERR_COLOR, ERR_WIDTH, FAMILY_COLOR, FULL_BA_ELO,
                         PAPER_DIR, REGIMES, load_leaderboard, wide)

OUT = PAPER_DIR / 'figures' / 'elo-full-field'
WIDTH, HEIGHT = 7.0, 2.9


def draw(ax, w: pd.DataFrame) -> None:
    w = w.sort_values('best', ascending=False).reset_index(drop=True)
    x = np.arange(len(w))
    for reg in ('tuned_ensemble', 'tuned', 'default'):   # widest first, like the official plot
        ok = w[reg].notna()
        ax.bar(x[ok], w.loc[ok, reg] - ELO_MIN, bottom=ELO_MIN, width=0.7, color=BAR_COLOR[reg], linewidth=0,
               zorder=2)
    for reg in REGIMES:
        ok = w[reg].notna()
        ax.errorbar(x[ok], w.loc[ok, reg], yerr=[w.loc[ok, reg + '-'], w.loc[ok, reg + '+']], fmt='none',
                    ecolor=ERR_COLOR[reg], elinewidth=ERR_WIDTH[reg], alpha=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(w['name'], rotation=60, ha='right', rotation_mode='anchor')
    for t, (_, r) in zip(ax.get_xticklabels(), w.iterrows()):
        if isinstance(r['family'], str):
            t.set_color(FAMILY_COLOR[r['family']])
            if r['arm'] in ('FT', 'FTE'):
                t.set_fontweight('bold')
    ax.set_xlim(-0.65, len(w) - 0.35)
    top = max(np.nanmax((w[r] + w[r + '+']).to_numpy()) for r in REGIMES) + 8
    ax.set_ylim(ELO_MIN, top)
    ax.set_ylabel('Elo')
    ax.tick_params(axis='x', pad=1)
    ax.grid(False, axis='x')
    ax.legend(handles=[Patch(color=BAR_COLOR[r], label=BAR_LABEL[r]) for r in REGIMES], loc='upper right', ncol=3,
              fontsize=6.3, frameon=True, borderpad=0.3, handletextpad=0.4, columnspacing=0.9)


def main() -> None:
    lb = load_leaderboard(FULL_BA_ELO)
    assert len(lb) == 58 and lb['regime'].notna().all()
    w = wide(lb)
    for reg in REGIMES:
        if reg not in w:
            w[reg] = w[reg + '+'] = w[reg + '-'] = np.nan
    n_bars = int(w[list(REGIMES)].notna().sum().sum())
    assert n_bars == len(lb), (n_bars, len(lb))   # every leaderboard row is one bar
    with sns.axes_style('whitegrid'), plt.rc_context({'font.size': 6.5, 'axes.labelsize': 7.5,
                                                        'xtick.labelsize': 6.0, 'ytick.labelsize': 6.3,
                                                        'axes.edgecolor': '#cccccc'}):
        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), layout='constrained')
        draw(ax, w)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT.with_suffix(f'.{ext}'), dpi=300, facecolor='white')
        plt.close(fig)
    print(f'{len(lb)} leaderboard rows as {n_bars} bars in {len(w)} columns; wrote {OUT}.pdf/.png')


if __name__ == '__main__':
    main()
