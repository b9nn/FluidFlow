"""
Cross-domain alternates figure: GP/TabPFN vs best-transfer at each N,
across target domains (BM, TBCM, Female BCM). Reads alternates JSONs
from each domain plus the matching transfer comparator source.

Three-way comparison per panel (alternates GP, alternates TabPFN, best
transfer for that domain). Transfer comparator sources:
  BM       -> hardcoded SMALL_N_TRANSFER (mirrors BM_Showcase.py)
  TBCM     -> TBCM/results/rf_transfer_small_n.json (produced by Task 5.5)
  Female   -> ResgressorAnalysis/figs/all_regressors_transfer_comparison.csv
              filtered to regressor == 'RF' (2026-05-12 scope decision)

A domain with missing data renders a "no data yet" placeholder panel.
Allows the figure to be produced before TBCM data resolves.

Outputs:
  Beam_Membrane/figs/cross_domain_alternates.png  (3 panels side-by-side)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS_DIR = os.path.join(os.path.dirname(__file__), 'figs')


# BM transfer comparator: hardcoded best-per-N from BM_Showcase.SMALL_N_TRANSFER.
BM_TRANSFER_BEST = {
    10: 0.075, 20: 0.054, 30: 0.187, 50: 0.193,
    75: 0.294, 100: 0.280, 200: 0.586, 500: 0.739,
}


DOMAINS = [
    {
        'name': 'BM',
        'title': 'Beam-Membrane (FEM)',
        'alt_json': os.path.join(ROOT, 'Beam_Membrane', 'results',
                                  'alternates_results.json'),
        'transfer_label': 'best BCM->BM transfer',
        'transfer_loader': lambda d: BM_TRANSFER_BEST,
    },
    {
        'name': 'TBCM',
        'title': 'TBCM (triangular body-cover)',
        'alt_json': os.path.join(ROOT, 'TBCM', 'results',
                                  'alternates_results.json'),
        'transfer_label': 'best BCM->TBCM transfer (TransRF)',
        'transfer_loader': 'tbcm_small_n_json',
        'transfer_path': os.path.join(ROOT, 'TBCM', 'results',
                                       'rf_transfer_small_n.json'),
    },
    {
        'name': 'FemaleBCM',
        'title': 'Female BCM (Male -> Female transfer)',
        'alt_json': os.path.join(ROOT, 'VocalFoldRegression', 'BCM Model',
                                  'Alternates', 'results',
                                  'alternates_results.json'),
        'transfer_label': 'best Male->Female RF transfer (TransRF, retrained at N)',
        'transfer_loader': 'tbcm_small_n_json',  # same loader shape — JSON schema matches
        'transfer_path': os.path.join(ROOT, 'VocalFoldRegression', 'BCM Model',
                                       'Alternates', 'results',
                                       'rf_transfer_small_n.json'),
    },
]

COLORS = {'GP': '#0f766e', 'TabPFN': '#d97706', 'transfer': '#475569'}


def load_alternates(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
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
        if by_n:
            out[method] = by_n
    return out


def load_tbcm_transfer(path):
    """Pull TransRF mean avg-R^2 per N from the small-N TBCM JSON produced
    by Task 5.5 (TBCM_SmallData.py). Returns {N: avg_R^2}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    block = data.get('transrf', {})
    out = {}
    for n_str, rec in block.items():
        if not n_str.isdigit():
            continue
        f0 = np.asarray(rec['r2_f0'])
        spl = np.asarray(rec['r2_spl'])
        out[int(n_str)] = float((f0.mean() + spl.mean()) / 2.0)
    return out


def load_female_transfer(csv_path):
    """Male->Female RF transfer comparator. Reads
    all_regressors_transfer_comparison.csv, filters to RF, returns
    {sample_size: r2_avg}. Defensive recompute of r2_avg if needed."""
    if not os.path.exists(csv_path):
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = {'sample_size', 'regressor', 'r2_avg'}
    if not required.issubset(df.columns):
        if {'sample_size', 'regressor', 'r2_f0', 'r2_spl'}.issubset(df.columns):
            df = df.copy()
            df['r2_avg'] = (df['r2_f0'] + df['r2_spl']) / 2.0
        else:
            return {}
    rf = df[df['regressor'] == 'RF']
    return {int(row['sample_size']): float(row['r2_avg'])
            for _, row in rf.iterrows()}


def resolve_transfer(dom):
    loader = dom.get('transfer_loader')
    if callable(loader):
        return loader(dom)
    if loader == 'tbcm_small_n_json':
        return load_tbcm_transfer(dom['transfer_path'])
    if loader == 'female_rf_csv':
        return load_female_transfer(dom['transfer_path'])
    return {}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        alt = load_alternates(dom['alt_json'])
        xfer = resolve_transfer(dom)
        if not alt and not xfer:
            ax.text(0.5, 0.5, f"{dom['name']}\n(no data yet)",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, alpha=0.6)
            ax.set_title(dom['title'])
            ax.set_xscale('log')
            ax.set_xlabel('Number of target-domain training samples')
            continue

        for method, color in [('GP', COLORS['GP']), ('TabPFN', COLORS['TabPFN'])]:
            if method not in alt:
                continue
            ns = sorted(alt[method])
            means = np.array([alt[method][n].mean() for n in ns])
            stds = np.array([alt[method][n].std() for n in ns])
            ax.fill_between(ns, means - stds, means + stds,
                            color=color, alpha=0.15)
            ax.plot(ns, means, color=color, lw=2.4, marker='o', markersize=6,
                    label=f'{method} (non-transfer)')

        if xfer:
            xs = sorted(xfer)
            ys = [xfer[n] for n in xs]
            ax.plot(xs, ys, color=COLORS['transfer'], lw=2.0, ls='--',
                    marker='s', markersize=5, label=dom['transfer_label'])

        ax.set_xscale('log')
        ax.set_xlabel('Number of target-domain training samples')
        ax.set_title(dom['title'], fontweight='bold')
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_ylim(-0.2, 1.0)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='lower right', fontsize=9)

    axes[0].set_ylabel('Average R²  (F0 + SPL) / 2')
    fig.suptitle('GP / TabPFN vs transfer learning -- three target domains',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    out_path = os.path.join(FIGS_DIR, 'cross_domain_alternates.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
