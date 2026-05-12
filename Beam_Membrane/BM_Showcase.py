"""
BM Showcase — presentation-quality visualizations for the headline result:
non-transfer alternates (GP, TabPFN) beat BCM->BM transfer methods at small N.

Inputs:
  - results/alternates_results.json (GP, TabPFN — 10 bootstrap replicates per N)
  - results/rf_transfer_results.json (Callum's transfer methods at larger N >= 160)
  - Hardcoded small-N transfer table from PROJECT_GUIDE.md (means only;
    per-replicate data not committed yet — TODO #13).

Outputs (Beam_Membrane/figs/):
  - bm_showcase_headline.png    Average R^2 vs N, alternates vs transfer envelope
  - bm_showcase_sim_budget.png  How many BM sims needed to hit R^2 thresholds
  - bm_showcase_bootstrap.png   Bootstrap-replicate distributions at key N
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir = os.path.join(script_dir, 'figs')
results_dir = os.path.join(script_dir, 'results')

# --- Style ---
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'GP':       '#0f766e',   # teal
    'TabPFN':   '#d97706',   # amber
    'transfer': '#9ca3af',   # neutral gray
    'best_xfer':'#475569',   # darker slate for highlighted transfer line
    'target':   '#3b82f6',   # blue (target-only baseline)
}

# Small-N transfer numbers from PROJECT_GUIDE.md (avg of F0 and SPL R^2).
# These columns in PROJECT_GUIDE are already presented as average R^2.
# Note: per-replicate values not committed (TODO #13). Means only.
SMALL_N_TRANSFER = {
    # N: {method: avg_R^2}
    10:  {'Target Only': 0.040, 'Feature Aug': 0.071, 'TransRF': 0.075},
    20:  {'Target Only': -0.040, 'Feature Aug': 0.054, 'TransRF': 0.019},
    30:  {'Target Only': 0.143, 'Feature Aug': 0.187, 'TransRF': 0.171},
    50:  {'Target Only': 0.118, 'Feature Aug': 0.193, 'TransRF': 0.188},
    75:  {'Target Only': 0.222, 'Feature Aug': 0.294, 'TransRF': 0.250},
    100: {'Target Only': 0.211, 'Feature Aug': 0.278, 'TransRF': 0.280},
    200: {'Target Only': 0.494, 'Feature Aug': 0.584, 'TransRF': 0.586},
    500: {'Target Only': 0.719, 'Feature Aug': 0.716, 'TransRF': 0.739},
}


def load_alternates():
    """Return per-N average-R^2 arrays (10 bootstrap replicates) for GP and TabPFN."""
    with open(os.path.join(results_dir, 'alternates_results.json')) as f:
        raw = json.load(f)
    out = {}
    for method in ('GP', 'TabPFN'):
        if method not in raw:
            continue
        by_n = {}
        for n_str, rec in raw[method].items():
            f0 = np.asarray(rec['r2_f0'])
            spl = np.asarray(rec['r2_spl'])
            by_n[int(n_str)] = (f0 + spl) / 2.0  # average R^2 per replicate
        out[method] = by_n
    return out


def load_full_n_transfer():
    """Callum's transfer results at full N (N>=160). Returns
       {method: [(n, mean_avg_R2), ...]} where mean_avg_R2 = (f0+spl)/2 averaged
       over the experiment's internal replicates."""
    with open(os.path.join(results_dir, 'rf_transfer_results.json')) as f:
        rows = json.load(f)
    methods = {
        'Target Only': 'target_only',
        'Residual':    'residual',
        'Feature Aug': 'augmented',
        'Simple Ens':  'simple_ens',
        'TransRF':     'transrf',
    }
    out = {m: [] for m in methods}
    for row in rows:
        n = row['n_samples']
        for label, key in methods.items():
            f0 = row.get(f'{key}_f0_r2_mean')
            spl = row.get(f'{key}_spl_r2_mean')
            if f0 is not None and spl is not None:
                out[label].append((n, (f0 + spl) / 2.0))
    return out


def best_transfer_at(n_target, small=True):
    """Return (best_method_name, best_R2) at N. Uses small-N table (N<=500)."""
    if n_target in SMALL_N_TRANSFER:
        d = SMALL_N_TRANSFER[n_target]
        method = max(d, key=d.get)
        return method, d[method]
    return None, None


# ============================================================================
# Figure 1: Headline — Average R^2 vs N
# ============================================================================
def fig_headline():
    alt = load_alternates()
    xfer_full = load_full_n_transfer()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Transfer "envelope" at small N: draw each method faintly, then the
    # per-N best in a heavier slate.
    method_order = ['Target Only', 'Feature Aug', 'TransRF']
    small_ns = sorted(SMALL_N_TRANSFER.keys())
    small_ns_in_alt_range = [n for n in small_ns if n <= 100]

    # Faint per-method lines (gray spaghetti) — individual transfer methods
    for i, m in enumerate(method_order):
        ys = [SMALL_N_TRANSFER[n].get(m, np.nan) for n in small_ns]
        ax.plot(small_ns, ys, color=COLORS['transfer'], lw=1.4,
                alpha=0.75, marker='o', markersize=4, zorder=2,
                label='Individual transfer methods' if i == 0 else None)

    # Best-transfer-per-N line
    best_xs, best_ys, best_labels = [], [], []
    for n in small_ns:
        m, v = best_transfer_at(n)
        if m is not None:
            best_xs.append(n)
            best_ys.append(v)
            best_labels.append(m)
    ax.plot(best_xs, best_ys, color=COLORS['best_xfer'],
            lw=2.2, marker='s', markersize=6,
            label='Best BCM->BM transfer (Callum)', zorder=3)

    # Full-N transfer reference (TransRF only, dashed continuation)
    xfer_tr = sorted(xfer_full['TransRF'])
    ns_tr = [n for n, _ in xfer_tr]
    vs_tr = [v for _, v in xfer_tr]
    ax.plot(ns_tr, vs_tr, color=COLORS['best_xfer'], lw=1.6,
            ls=(0, (4, 3)), alpha=0.65, marker='s', markersize=4,
            label='TransRF at full N (reference)', zorder=2)

    # GP and TabPFN: bold lines + 1-sigma bands
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

    # Annotate headline gap at N=50
    gp_at_50 = alt['GP'][50].mean()
    tab_at_50 = alt['TabPFN'][50].mean()
    best_xfer_at_50 = SMALL_N_TRANSFER[50]
    best_xfer_val_50 = max(best_xfer_at_50.values())
    gap = tab_at_50 - best_xfer_val_50

    ax.annotate(
        f'+{gap:.2f} R² at N=50\nTabPFN {tab_at_50:.2f}  vs  best transfer {best_xfer_val_50:.2f}',
        xy=(50, tab_at_50), xytext=(120, 0.30),
        fontsize=10, fontweight='bold', color='#111',
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#92400e', lw=1),
        arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.2,
                        connectionstyle='arc3,rad=0.15'))

    ax.set_xscale('log')
    ax.set_xlabel('Number of BM training samples (log scale)')
    ax.set_ylabel('Average R²  (F0 + SPL) / 2')
    ax.set_title('Headline: non-transfer baselines beat BCM->BM transfer at small N',
                 fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.axhline(0.5, color='#000', lw=0.4, alpha=0.15, ls=':')
    ax.set_ylim(-0.15, 1.0)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)

    fig.tight_layout()
    path = os.path.join(figs_dir, 'bm_showcase_headline.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 2: Sim-budget — how many BM simulations to reach R^2 thresholds
# ============================================================================
def fig_sim_budget():
    alt = load_alternates()

    # For alternates we have per-replicate; use the mean curve.
    methods_data = {}
    for m in ('GP', 'TabPFN'):
        ns = sorted(alt[m])
        means = [alt[m][n].mean() for n in ns]
        methods_data[m] = list(zip(ns, means))

    # Transfer methods: use small-N table directly.
    for m in ('Target Only', 'Feature Aug', 'TransRF'):
        ns = sorted(SMALL_N_TRANSFER)
        means = [SMALL_N_TRANSFER[n].get(m, np.nan) for n in ns]
        methods_data[m] = list(zip(ns, means))

    def n_to_reach(curve, threshold):
        """Linear-interp on log-N axis to find smallest N where R^2 >= threshold.
        Returns None if curve never reaches threshold within its range."""
        prev = None
        for n, v in curve:
            if v >= threshold:
                if prev is None:
                    return n
                n_prev, v_prev = prev
                # interpolate on log-N
                if v == v_prev:
                    return n
                t = (threshold - v_prev) / (v - v_prev)
                log_n = np.log10(n_prev) + t * (np.log10(n) - np.log10(n_prev))
                return 10 ** log_n
            prev = (n, v)
        return None  # never reaches

    thresholds = [0.5, 0.7]
    method_order = ['TabPFN', 'GP', 'TransRF', 'Feature Aug', 'Target Only']
    colors = {
        'TabPFN': COLORS['TabPFN'],
        'GP': COLORS['GP'],
        'TransRF': COLORS['best_xfer'],
        'Feature Aug': COLORS['transfer'],
        'Target Only': COLORS['target'],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, thr in zip(axes, thresholds):
        budgets = []
        labels = []
        bar_colors = []
        for m in method_order:
            n_req = n_to_reach(methods_data[m], thr)
            labels.append(m)
            bar_colors.append(colors[m])
            budgets.append(n_req if n_req is not None else np.nan)

        ys = np.arange(len(method_order))
        # Replace nans with a cap value for visualisation, marked separately.
        cap = 600
        plot_vals = [b if (b is not None and not np.isnan(b)) else cap for b in budgets]
        bars = ax.barh(ys, plot_vals, color=bar_colors, edgecolor='black', lw=0.7)
        for i, (bar, b) in enumerate(zip(bars, budgets)):
            if b is None or np.isnan(b):
                ax.text(cap * 0.5, ys[i],
                        'not reached at N≤100 (alternates not tested above 100)',
                        ha='center', va='center', fontsize=8.5,
                        color='white', fontweight='bold')
            else:
                ax.text(min(b, cap) + 8, ys[i], f'N = {int(round(b))}',
                        ha='left', va='center', fontsize=10,
                        fontweight='bold')
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlabel('BM simulations needed')
        ax.set_title(f'To reach average R² ≥ {thr}', fontweight='bold')
        ax.set_xlim(0, cap)
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle('Simulation budget: BM samples required to hit accuracy thresholds',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'bm_showcase_sim_budget.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 3: Bootstrap distributions at key N
# ============================================================================
def fig_bootstrap():
    alt = load_alternates()
    # Show progression including extended N range. Only include N's actually
    # present for both GP and TabPFN.
    candidates = [10, 50, 100, 200, 500]
    key_ns = [n for n in candidates if n in alt.get('GP', {}) and n in alt.get('TabPFN', {})]
    if not key_ns:
        print('  WARN: no overlapping N values for bootstrap figure')
        return

    fig, axes = plt.subplots(1, len(key_ns), figsize=(3.0 * len(key_ns) + 1.5, 5),
                             sharey=True)
    if len(key_ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, key_ns):
        gp_vals = alt['GP'][n]
        tab_vals = alt['TabPFN'][n]
        data = [gp_vals, tab_vals]
        labels = ['GP', 'TabPFN']
        colors_box = [COLORS['GP'], COLORS['TabPFN']]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', lw=1.5),
                        boxprops=dict(lw=1),
                        whiskerprops=dict(lw=1),
                        capprops=dict(lw=1),
                        flierprops=dict(marker='o', markersize=4,
                                        markerfacecolor='white', alpha=0.7))
        for patch, c in zip(bp['boxes'], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)

        # Scatter raw replicates
        for i, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(n + i).random(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color='black', s=14, alpha=0.55, zorder=4)

        # Reference lines for best transfer at this N
        if n in SMALL_N_TRANSFER:
            best_m, best_v = best_transfer_at(n)
            ax.axhline(best_v, color=COLORS['best_xfer'], ls='--', lw=1.6,
                       alpha=0.9, zorder=2,
                       label=f'best transfer ({best_m}): {best_v:.2f}')
            ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

        ax.set_title(f'N = {n}', fontweight='bold')
        ax.set_ylim(-0.6, 1.0)
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.grid(True, axis='y', alpha=0.25)
        if n == key_ns[0]:
            ax.set_ylabel('Average R²  (F0 + SPL) / 2')

    fig.suptitle('Bootstrap robustness: 10 replicates per N, alternates vs best transfer',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'bm_showcase_bootstrap.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('BM SHOWCASE — generating presentation-quality figures')
    print('=' * 70)
    os.makedirs(figs_dir, exist_ok=True)
    fig_headline()
    fig_sim_budget()
    fig_bootstrap()
    print('=' * 70)
    print('Done.')


if __name__ == '__main__':
    main()
