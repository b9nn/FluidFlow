"""
Female BCM Showcase — presentation-quality headline figure for
Male -> Female BCM transfer vs non-transfer (GP, TabPFN).

Mirrors Beam_Membrane/BM_Showcase.py:fig_headline. Differences from
BM/TBCM:
  - Female's "transfer methods" are different regressor families
    (RF, NN, PR) rather than transfer variants. All three are plotted
    as spaghetti; RF is highlighted as the canonical Female transfer
    line per Brian's 2026-05-12 decision (RF beats NN/PR at every N
    in this CSV).
  - No "full N continuation" needed — the transfer CSV already covers
    N=[25..843] in one sweep.

Story: Male->Female BCM is well aligned (same physics, demographic
shift). RF transfer holds r²_avg ≈ 0.72 across N=25..843 — extremely
strong baseline. TabPFN catches at N≈75 and dominates from N=100,
reaching 0.97 at N=500. GP alone does NOT catch — TabPFN's pretrained
prior is the load-bearing component.

Inputs:
  - VocalFoldRegression/BCM Model/Alternates/results/alternates_results.json
  - VocalFoldRegression/BCM Model/ResgressorAnalysis/figs/all_regressors_transfer_comparison.csv

Output:
  - VocalFoldRegression/BCM Model/Alternates/figs/female_showcase_headline.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir = os.path.join(script_dir, 'figs')
results_dir = os.path.join(script_dir, 'results')

TRANSFER_CSV = os.path.join(
    os.path.dirname(script_dir),
    'ResgressorAnalysis', 'figs', 'all_regressors_transfer_comparison.csv'
)


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'GP':        '#0f766e',
    'TabPFN':    '#d97706',
    'transfer':  '#9ca3af',
    'best_xfer': '#475569',
}


def load_alternates():
    with open(os.path.join(results_dir, 'alternates_results.json')) as f:
        raw = json.load(f)
    out = {}
    for method in ('GP', 'TabPFN'):
        if method not in raw:
            continue
        by_n = {}
        for n_str, rec in raw[method].items():
            if not n_str.isdigit():
                continue
            f0 = np.asarray(rec['r2_f0'])
            spl = np.asarray(rec['r2_spl'])
            by_n[int(n_str)] = (f0 + spl) / 2.0
        out[method] = by_n
    return out


def load_transfer():
    """Returns {regressor: [(n, r2_avg), ...]}."""
    df = pd.read_csv(TRANSFER_CSV)
    out = {}
    for reg in df['regressor'].unique():
        sub = df[df['regressor'] == reg].sort_values('sample_size')
        out[reg] = list(zip(sub['sample_size'].astype(int).tolist(),
                            sub['r2_avg'].astype(float).tolist()))
    return out


def fig_headline():
    alt = load_alternates()
    xfer = load_transfer()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Spaghetti: NN and PR transfer
    others = [r for r in ('NN', 'PR') if r in xfer]
    for i, r in enumerate(others):
        xs = [n for n, _ in xfer[r]]
        ys = [v for _, v in xfer[r]]
        ax.plot(xs, ys, color=COLORS['transfer'], lw=1.4, alpha=0.7,
                marker='o', markersize=4, zorder=2,
                label='NN / PR transfer (Brian)' if i == 0 else None)

    # RF transfer highlighted — the canonical Female transfer line
    if 'RF' in xfer:
        xs = [n for n, _ in xfer['RF']]
        ys = [v for _, v in xfer['RF']]
        ax.plot(xs, ys, color=COLORS['best_xfer'], lw=2.4, marker='s',
                markersize=6, label='RF Male->Female transfer (Brian)',
                zorder=3)

    # GP + TabPFN bold lines with 1-sigma bands
    for method, color in [('GP', COLORS['GP']), ('TabPFN', COLORS['TabPFN'])]:
        if method not in alt:
            continue
        ns = sorted(alt[method])
        means = np.array([alt[method][n].mean() for n in ns])
        stds = np.array([alt[method][n].std() for n in ns])
        ax.fill_between(ns, means - stds, means + stds,
                        color=color, alpha=0.15, zorder=3)
        ax.plot(ns, means, color=color, lw=2.8, marker='o', markersize=7,
                label=f'{method} (non-transfer, ours)', zorder=5)

    # Annotate the catch-up point at N=75 and the divergence at N=500
    rf_by_n = {n: v for n, v in xfer.get('RF', [])}
    if 75 in alt['TabPFN']:
        tab_75 = alt['TabPFN'][75].mean()
        # RF reference at the nearest N in transfer grid: 50 (below) and 100 (above)
        # Interpolate roughly
        if 50 in rf_by_n and 100 in rf_by_n:
            rf_75 = 0.5 * (rf_by_n[50] + rf_by_n[100])
        else:
            rf_75 = rf_by_n.get(100, np.nan)
        ax.annotate(
            f'catch-up at N≈75\nTabPFN {tab_75:.2f}  ≈  RF transfer {rf_75:.2f}',
            xy=(75, tab_75), xytext=(7, 0.92),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#92400e', lw=1),
            arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.2,
                            connectionstyle='arc3,rad=0.15'))

    if 500 in alt['TabPFN'] and 500 in rf_by_n:
        tab_500 = alt['TabPFN'][500].mean()
        rf_500 = rf_by_n[500]
        gap_500 = tab_500 - rf_500
        ax.annotate(
            f'+{gap_500:.2f} R² at N=500\n'
            f'TabPFN {tab_500:.2f}  vs  RF transfer {rf_500:.2f}',
            xy=(500, tab_500), xytext=(120, 0.45),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#d1fae5', ec='#065f46', lw=1),
            arrowprops=dict(arrowstyle='->', color='#065f46', lw=1.2,
                            connectionstyle='arc3,rad=-0.2'))

    ax.set_xscale('log')
    ax.set_xlabel('Number of Female BCM training samples (log scale)')
    ax.set_ylabel('Average R²  (F0 + SPL) / 2')
    ax.set_title('Male->Female BCM: TabPFN catches RF transfer at N≈75, dominates from N=100',
                 fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.axhline(0.5, color='#000', lw=0.4, alpha=0.15, ls=':')
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)

    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'female_showcase_headline.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('FEMALE BCM SHOWCASE')
    print('=' * 70)
    fig_headline()


if __name__ == '__main__':
    main()
