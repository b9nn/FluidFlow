"""
BM extended — Result 1 figure: metrics vs training size, TabPFN vs transfer.

Per output (F0, SPL, ACFL, PC, CPP), plots BOTH R2 (top row) and normalized
RMSE (bottom row) as a function of N, comparing:
  - TabPFN (single-output, no transfer)   [from bm_ext_tabpfn.json]
  - TransRF (best transfer)                [from bm_ext_transfer.json]
  - Target-only RF (no-transfer baseline)  [faint, context]
with mean +/- std bands over seeds.

Output: figs/bm_ext_metrics_vs_n.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
figs_dir = os.path.join(script_dir, 'figs')

TARGETS = ['F0', 'SPL', 'ACFL', 'PC', 'CPP']
GRID_MATCH = (10, 20, 50, 100, 200, 500)  # consistent with all other paper figures
PROVISIONAL = {'ACFL'}   # flagged: BM ACFL definition pending Parra reconciliation

STYLE = {
    'TabPFN':  dict(color='#d97706', marker='o', ls='-', lw=2.6, ms=7, label='TabPFN (no transfer)'),
    'TransRF': dict(color='#1e293b', marker='s', ls='-', lw=2.2, ms=6, label='TransRF (best transfer)'),
    'Target':  dict(color='#94a3b8', marker='x', ls=':', lw=1.5, ms=6, label='Target-only RF'),
}


def load_transfer():
    with open(os.path.join(results_dir, 'bm_ext_transfer.json')) as f:
        return json.load(f)


def load_tabpfn():
    with open(os.path.join(results_dir, 'bm_ext_tabpfn.json')) as f:
        return json.load(f)


def tabpfn_curve(tab, target, metric):
    block = tab['single'][target]
    ns = sorted(int(k) for k in block if k.isdigit() and int(k) in GRID_MATCH)
    mean = [float(np.mean(block[str(n)][metric])) for n in ns]
    std = [float(np.std(block[str(n)][metric])) for n in ns]
    return ns, np.array(mean), np.array(std)


def transfer_curve(tr, target, method, metric):
    rows = [r for r in tr[target] if r['n_samples'] in GRID_MATCH]
    ns = [r['n_samples'] for r in rows]
    mean = np.array([r[f'{method}_{metric}_mean'] for r in rows])
    std = np.array([r[f'{method}_{metric}_std'] for r in rows])
    return ns, mean, std


def main():
    tr = load_transfer()
    tab = load_tabpfn()

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'legend.fontsize': 8.5, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(4.0 * len(TARGETS), 8), sharex=True)

    for col, target in enumerate(TARGETS):
        for row, metric in enumerate(('r2', 'nrmse')):
            ax = axes[row, col]
            # TabPFN
            ns, m, s = tabpfn_curve(tab, target, metric)
            st = STYLE['TabPFN']
            ax.fill_between(ns, m - s, m + s, color=st['color'], alpha=0.15)
            ax.plot(ns, m, **{k: v for k, v in st.items() if k != 'label'},
                    label=st['label'], zorder=5)
            # TransRF + target-only
            for method, key in (('transrf', 'TransRF'), ('target_only', 'Target')):
                ns2, m2, s2 = transfer_curve(tr, target, method, metric)
                st = STYLE[key]
                if key == 'TransRF':
                    ax.fill_between(ns2, m2 - s2, m2 + s2, color=st['color'], alpha=0.12)
                ax.plot(ns2, m2, **{k: v for k, v in st.items() if k != 'label'},
                        label=st['label'], zorder=3)
            ax.set_xscale('log')
            ax.grid(True, which='both', alpha=0.25)
            if row == 0:
                title = target + ('  (ACFL provisional*)' if target in PROVISIONAL else '')
                ax.set_title(title, fontweight='bold')
                ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
                ax.set_ylim(top=1.03)
            else:
                ax.set_xlabel('N training samples (log)')
            if col == 0:
                ax.set_ylabel('R²' if row == 0 else 'normalized RMSE')

    axes[0, 0].legend(loc='lower right', framealpha=0.95)
    fig.suptitle('BCM→BM: TabPFN vs transfer — accuracy (R²) and error (nRMSE) vs training size\n'
                 '*ACFL provisional: BM ACFL ~2–4× BCM source, definition pending reconciliation',
                 fontweight='bold', fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'bm_ext_metrics_vs_n.png')
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    main()
