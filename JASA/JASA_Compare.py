"""
Component 2 figure — transfer learning vs TabPFN, per new output.

Overlays the TabPFN single-target curve (from Component 1,
results/tabpfn_multihead.json) against the RF transfer methods (Component 2,
results/rf_transfer.json) for MFDR, ACFL, PC. One panel per target.

Output: figs/jasa_transfer_vs_tabpfn.png
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

TARGETS = ['F0', 'SPL', 'MFDR', 'ACFL', 'PC']

# Transfer methods to draw (json key -> label, style). Only TransRF per request.
TRANSFER_STYLE = {
    'transrf':     dict(label='TransRF (transfer)', color='#1e293b', marker='s', ls='-',  lw=2.4, ms=7),
}
TABPFN_STYLE = dict(label='TabPFN (in-context, ours)', color='#d97706',
                    marker='o', ls='-', lw=2.8, ms=8)


def load_transfer():
    with open(os.path.join(results_dir, 'rf_transfer.json')) as f:
        return json.load(f)


def load_tabpfn_single():
    with open(os.path.join(results_dir, 'tabpfn_multihead.json')) as f:
        raw = json.load(f)
    return raw['single']


def main():
    transfer = load_transfer()
    tabpfn = load_tabpfn_single()

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'legend.fontsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10), sharey=True)
    axes = axes.ravel()

    for ax, target in zip(axes, TARGETS):
        rows = transfer[target]
        ns = [r['n_samples'] for r in rows]

        for key, st in TRANSFER_STYLE.items():
            ys = [r[f'{key}_r2_mean'] for r in rows]
            ax.plot(ns, ys, color=st['color'], marker=st['marker'], ls=st['ls'],
                    lw=st['lw'], ms=st['ms'], label=st['label'], zorder=3)

        # TabPFN single-target curve.
        tb = tabpfn[target]
        tb_ns = sorted(int(k) for k in tb.keys())
        tb_ys = [float(np.mean(tb[str(n)])) for n in tb_ns]
        ax.plot(tb_ns, tb_ys, color=TABPFN_STYLE['color'], marker=TABPFN_STYLE['marker'],
                ls=TABPFN_STYLE['ls'], lw=TABPFN_STYLE['lw'], ms=TABPFN_STYLE['ms'],
                label=TABPFN_STYLE['label'], zorder=5)

        ax.set_xscale('log')
        ax.set_xlabel('JASA data budget N (log scale)')
        ax.set_title(target, fontweight='bold')
        ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_ylim(-0.8, 1.02)

    axes[0].set_ylabel('R²')
    axes[3].set_ylabel('R²')
    axes[0].legend(loc='lower right', framealpha=0.95)
    # Hide any unused panels (5 targets in a 2x3 grid).
    for ax in axes[len(TARGETS):]:
        ax.set_visible(False)
    fig.suptitle('BCM→JASA transfer vs in-context TabPFN — all outputs',
                 fontweight='bold', fontsize=14, y=1.0)
    fig.tight_layout()

    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'jasa_transfer_vs_tabpfn.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    main()
