"""
TBCM Showcase — presentation-quality headline figure for BCM->TBCM.

Mirrors Beam_Membrane/BM_Showcase.py:fig_headline structure but for TBCM:
plots GP and TabPFN bold lines with std bands, overlays small-N transfer
spaghetti (Target/Residual/Augmented/TransRF) plus a per-N best-transfer
line, with full-N TransRF dashed continuation as reference.

Story framing differs from BM: on TBCM, transfer is well aligned (same
physics family, geometry shift only) so the alternates' lead narrows
with N and the curves essentially tie at N=500. Annotation highlights
the small-N gap at N=50.

Inputs:
  - TBCM/results/alternates_results.json  (GP + TabPFN per-replicate)
  - TBCM/results/rf_transfer_small_n.json (per-replicate transfer N=10..500)
  - TBCM/results/rf_transfer_results.json (Callum's full-N N>=1379)

Output:
  - TBCM/figs/tbcm_showcase_headline.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir = os.path.join(script_dir, 'figs')
results_dir = os.path.join(script_dir, 'results')

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


def load_small_n_transfer():
    """Per-N mean avg R^2 for each transfer method from the small-N JSON
    produced by TBCM_SmallData.py. Returns {method_label: {N: mean_R^2}}."""
    with open(os.path.join(results_dir, 'rf_transfer_small_n.json')) as f:
        raw = json.load(f)
    label_map = {
        'target':    'Target Only',
        'residual':  'Residual',
        'augmented': 'Feature Aug',
        'transrf':   'TransRF',
    }
    out = {label: {} for label in label_map.values()}
    for key, label in label_map.items():
        block = raw.get(key, {})
        for n_str, rec in block.items():
            if not n_str.isdigit():
                continue
            f0 = np.asarray(rec['r2_f0'])
            spl = np.asarray(rec['r2_spl'])
            out[label][int(n_str)] = float((f0.mean() + spl.mean()) / 2.0)
    return out


def load_full_n_transfer():
    """Callum's TBCM transfer at full N (>=1379), aggregated means.
    Returns {method_label: [(n, mean_avg_R^2), ...]}"""
    path = os.path.join(results_dir, 'rf_transfer_results.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    methods = {
        'Target Only': 'target_only',
        'Residual':    'residual',
        'Feature Aug': 'augmented',
        'Simple Ens':  'simple_ens',
    }
    out = {m: [] for m in methods}
    # Try to add TransRF if present
    if rows and any('transrf_f0_r2_mean' in r for r in rows):
        methods['TransRF'] = 'transrf'
        out['TransRF'] = []
    for row in rows:
        n = row.get('n_samples') or row.get('n')
        if n is None:
            continue
        for label, key in methods.items():
            f0 = row.get(f'{key}_f0_r2_mean')
            spl = row.get(f'{key}_spl_r2_mean')
            if f0 is not None and spl is not None:
                out[label].append((int(n), (f0 + spl) / 2.0))
    return out


def best_per_n(small_table):
    """Per-N best transfer method. Returns [(n, method, value), ...]."""
    ns = sorted({n for d in small_table.values() for n in d})
    out = []
    for n in ns:
        cands = {m: small_table[m][n] for m in small_table if n in small_table[m]}
        if not cands:
            continue
        m = max(cands, key=cands.get)
        out.append((n, m, cands[m]))
    return out


def fig_headline():
    alt = load_alternates()
    small = load_small_n_transfer()
    full = load_full_n_transfer()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    method_order = ['Target Only', 'Residual', 'Feature Aug', 'TransRF']
    small_ns = sorted({n for d in small.values() for n in d})

    # Faint spaghetti per transfer method
    for i, m in enumerate(method_order):
        if m not in small:
            continue
        ys = [small[m].get(n, np.nan) for n in small_ns]
        ax.plot(small_ns, ys, color=COLORS['transfer'], lw=1.4,
                alpha=0.7, marker='o', markersize=4, zorder=2,
                label='Individual transfer methods' if i == 0 else None)

    # Best-per-N transfer line
    best = best_per_n(small)
    if best:
        ax.plot([n for n, _, _ in best], [v for _, _, v in best],
                color=COLORS['best_xfer'], lw=2.2, marker='s', markersize=6,
                label='Best BCM->TBCM transfer (Callum)', zorder=3)

    # Full-N TransRF dashed continuation (if present)
    if full.get('TransRF'):
        full_tr = sorted(full['TransRF'])
        ns_tr = [n for n, _ in full_tr]
        vs_tr = [v for _, v in full_tr]
        label_full = 'TransRF at full N (reference)'
    else:
        # Fall back to Feature Aug if TransRF missing at full N
        full_tr = sorted(full.get('Feature Aug', []))
        ns_tr = [n for n, _ in full_tr]
        vs_tr = [v for _, v in full_tr]
        label_full = 'Feature Aug at full N (reference)'

    if ns_tr:
        ax.plot(ns_tr, vs_tr, color=COLORS['best_xfer'], lw=1.6,
                ls=(0, (4, 3)), alpha=0.65, marker='s', markersize=4,
                label=label_full, zorder=2)

    # GP + TabPFN with 1-sigma bands
    for method, color in [('GP', COLORS['GP']), ('TabPFN', COLORS['TabPFN'])]:
        if method not in alt:
            continue
        ns = sorted(alt[method])
        means = np.array([alt[method][n].mean() for n in ns])
        stds = np.array([alt[method][n].std() for n in ns])
        ax.fill_between(ns, means - stds, means + stds,
                        color=color, alpha=0.15, zorder=3)
        ax.plot(ns, means, color=color, lw=2.8,
                marker='o', markersize=7,
                label=f'{method} (non-transfer, ours)', zorder=5)

    # Annotate small-N gap at N=50 and convergence at N=500
    if 50 in alt['TabPFN'] and 50 in small['TransRF']:
        tab_at_50 = alt['TabPFN'][50].mean()
        best_xfer_50 = max(small[m][50] for m in small if 50 in small[m])
        gap_50 = tab_at_50 - best_xfer_50
        ax.annotate(
            f'+{gap_50:.2f} R² at N=50\n'
            f'TabPFN {tab_at_50:.2f}  vs  best transfer {best_xfer_50:.2f}',
            xy=(50, tab_at_50), xytext=(7, 0.05),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#92400e', lw=1),
            arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.2,
                            connectionstyle='arc3,rad=0.15'))

    if 500 in alt['TabPFN'] and 500 in small['TransRF']:
        tab_at_500 = alt['TabPFN'][500].mean()
        best_xfer_500 = max(small[m][500] for m in small if 500 in small[m])
        gap_500 = tab_at_500 - best_xfer_500
        ax.annotate(
            f'gap +{gap_500:.3f} at N=500\n'
            f'TabPFN {tab_at_500:.3f}  ≈  best transfer {best_xfer_500:.3f}',
            xy=(500, tab_at_500), xytext=(140, 0.62),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#d1fae5', ec='#065f46', lw=1),
            arrowprops=dict(arrowstyle='->', color='#065f46', lw=1.2,
                            connectionstyle='arc3,rad=-0.2'))

    ax.set_xscale('log')
    ax.set_xlabel('Number of TBCM training samples (log scale)')
    ax.set_ylabel('Average R²  (F0 + SPL) / 2')
    ax.set_title('BCM->TBCM: alternates lead at small N; well-aligned transfer ties by N=500',
                 fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.axhline(0.5, color='#000', lw=0.4, alpha=0.15, ls=':')
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)

    fig.tight_layout()
    path = os.path.join(figs_dir, 'tbcm_showcase_headline.png')
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('TBCM SHOWCASE')
    print('=' * 70)
    fig_headline()


if __name__ == '__main__':
    main()
