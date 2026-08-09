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
  - bm_showcase_table.png       Per-N R^2 table: GP vs TransRF vs TabPFN
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir = os.path.join(script_dir, 'figs')
results_dir = os.path.join(script_dir, 'results')

# Match Callum's fair-comparison N grid so figures line up across collaborators.
GRID_MATCH = (10, 20, 50, 100, 200, 500)

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
    """Aligned with TBCM_Showcase / Female_Showcase: only TabPFN, GP, best
    transfer (TransRF), and the dashed full-N TransRF reference. No
    annotation text boxes — figures speak for themselves; numbers live in
    the head-to-head table figure."""
    alt = load_alternates()
    xfer_full = load_full_n_transfer()

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # Best of Callum's transfer at small N (TransRF only)
    small_ns = sorted(SMALL_N_TRANSFER.keys())
    transrf_small = [SMALL_N_TRANSFER[n]['TransRF'] for n in small_ns]
    ax.plot(small_ns, transrf_small, color='#1e293b',
            marker='s', ls='-', lw=2.2, ms=7, alpha=0.95,
            label="TransRF (best of Callum's BCM->BM transfer)", zorder=3)

    # Full-N TransRF dashed continuation
    xfer_tr = sorted(xfer_full['TransRF'])
    ns_tr = [n for n, _ in xfer_tr]
    vs_tr = [v for _, v in xfer_tr]
    ax.plot(ns_tr, vs_tr, color='#1e293b',
            marker='s', ls=(0, (4, 3)), lw=1.4, ms=4, alpha=0.55,
            label="TransRF at full N (Callum's larger-N runs)", zorder=2)

    # TabPFN bold with std band
    for method, color in [('TabPFN', COLORS['TabPFN'])]:
        if method not in alt:
            continue
        ns = sorted(alt[method])
        means = np.array([alt[method][n].mean() for n in ns])
        stds = np.array([alt[method][n].std() for n in ns])
        ax.fill_between(ns, means - stds, means + stds,
                        color=color, alpha=0.15, zorder=3)
        ax.plot(ns, means, color=color, marker='o', ls='-',
                lw=2.8, ms=8,
                label=f'{method} (non-transfer, ours)', zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of BM training samples (log scale)')
    ax.set_ylabel('Average R²  (F0 + SPL) / 2')
    ax.set_title('BCM->BM: alternates dominate at every N; transfer never catches within tested range',
                 fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.axhline(0.5, color='#000', lw=0.4, alpha=0.15, ls=':')
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, which='both', alpha=0.25)

    # Reorder legend to match TBCM/Female: alternates first, then transfer
    handles, labels = ax.get_legend_handles_labels()
    order_keys = [
        'TabPFN (non-transfer, ours)',
        "TransRF (best of Callum's BCM->BM transfer)",
        "TransRF at full N (Callum's larger-N runs)",
    ]
    label_to_handle = dict(zip(labels, handles))
    ordered = [(label_to_handle[k], k) for k in order_keys if k in label_to_handle]
    ax.legend([h for h, _ in ordered], [l for _, l in ordered],
              loc='lower right', framealpha=0.95, fontsize=9)

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
    candidates = [10, 20, 50, 100, 200, 500]
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


def fig_table():
    """Per-N table figure: GP (mean ± std), TransRF, TabPFN (mean ± std).
    Winner per row highlighted. TransRF means come from PROJECT_GUIDE.md
    (small N) and rf_transfer_results.json (full N); per-replicate
    standard deviations for TransRF aren't committed yet — TODO #13."""
    alt = load_alternates()
    xfer_full = load_full_n_transfer()  # {label: [(n, mean_avg_R2), ...]}

    # Build TransRF lookup over the union of small-N table + full-N JSON.
    transrf_mean = {n: SMALL_N_TRANSFER[n]['TransRF'] for n in SMALL_N_TRANSFER}
    for n, v in xfer_full.get('TransRF', []):
        transrf_mean.setdefault(n, v)

    # Rows: all N where we have at least one of the alternates.
    rows_n = sorted({n for m in ('GP', 'TabPFN') for n in alt.get(m, {}).keys()})
    rows_n = [n for n in rows_n if n >= 5]  # drop edge cases

    headers = ['N', 'GP', 'TransRF (best transfer)', 'TabPFN']
    cell_text = []
    cell_colors = []
    bg_default = '#ffffff'
    bg_winner = '#dcfce7'   # soft green
    bg_alt_row = '#f8fafc'  # subtle row banding
    text_winner = '#065f46'

    for i, n in enumerate(rows_n):
        gp_vals = alt.get('GP', {}).get(n)
        tab_vals = alt.get('TabPFN', {}).get(n)
        trf = transrf_mean.get(n)

        def fmt_alt(arr):
            if arr is None:
                return '—', None
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            return f'{mean:+.3f} ± {std:.2f}', mean

        gp_text, gp_mean = fmt_alt(gp_vals)
        tab_text, tab_mean = fmt_alt(tab_vals)
        trf_text = f'{trf:+.3f}' if trf is not None else '—'

        candidates = [(gp_mean, 1), (trf, 2), (tab_mean, 3)]
        valid = [(v, idx) for v, idx in candidates if v is not None]
        winner_col = max(valid, key=lambda t: t[0])[1] if valid else None

        cell_text.append([f'{n}', gp_text, trf_text, tab_text])

        row_bg = bg_alt_row if i % 2 == 1 else bg_default
        row_colors = [row_bg] * 4
        if winner_col is not None:
            row_colors[winner_col] = bg_winner
        cell_colors.append(row_colors)

    # Figure sizing
    n_rows = len(rows_n)
    fig_h = 0.55 + 0.40 * n_rows
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=['#0f172a'] * 4,
        cellLoc='center',
        loc='upper center',
        colWidths=[0.10, 0.30, 0.30, 0.30],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)

    # Header text white; cell font weights
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
            cell.set_edgecolor('#0f172a')
        else:
            cell.set_edgecolor('#cbd5e1')
            # Highlight winner cells in green text
            if cell.get_facecolor()[:3] == tuple(int(bg_winner[i:i+2], 16) / 255
                                                  for i in (1, 3, 5)):
                cell.get_text().set_color(text_winner)
                cell.get_text().set_fontweight('bold')

    ax.set_title(
        'BM head-to-head: average R² (F0+SPL)/2  —  winner per N highlighted',
        fontweight='bold', fontsize=13, pad=14)
    fig.text(0.5, 0.02,
             'GP & TabPFN: mean ± std over 10 bootstrap replicates.  '
             'TransRF: means only (per-replicate values not yet committed — TODO #13).  '
             'Source: alternates_results.json, rf_transfer_results.json, PROJECT_GUIDE.md.',
             ha='center', fontsize=8.5, style='italic', color='#475569')
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(figs_dir, 'bm_showcase_table.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def fig_headline_zoom(n_max=100, logx=False, fname='bm_showcase_headline_zoom.png'):
    """F0 and SPL in separate panels, TabPFN vs TransRF.
    logx=False: linear x-axis, zoomed to N<=n_max (small-data view).
    logx=True : log x-axis, full N range (companion to the linear zoom).
    TabPFN from alternates_results.json; TransRF from rf_transfer_small_n.json
    (BCM->BM, generated with MaleBCM source). Self-contained styles."""
    cap = 10 ** 9 if logx else n_max
    STYLE = {'TabPFN': dict(color='#d97706', marker='o', ls='-', lw=2.8, ms=8, label='TabPFN (non-transfer, ours)'),
             'GP': dict(color='#0f766e', marker='o', ls='-', lw=2.8, ms=8, label='GP (non-transfer, ours)'),
             'TransRF': dict(color='#1e293b', marker='s', ls='-', lw=2.2, ms=7, label='TransRF (BCM->BM transfer)')}
    with open(os.path.join(results_dir, 'alternates_results.json')) as f:
        altraw = json.load(f)
    alt = {m: {int(n): {'r2_f0': np.asarray(r['r2_f0']), 'r2_spl': np.asarray(r['r2_spl'])}
               for n, r in altraw.get(m, {}).items() if n.isdigit()} for m in ('GP', 'TabPFN')}
    trf = {}
    trf_path = os.path.join(results_dir, 'rf_transfer_small_n.json')
    if os.path.exists(trf_path):
        with open(trf_path) as f:
            block = json.load(f).get('transrf', {})
        trf = {int(n): {'r2_f0': float(np.mean(r['r2_f0'])), 'r2_spl': float(np.mean(r['r2_spl']))}
               for n, r in block.items() if n.isdigit()}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, tkey, tlabel in [(axes[0], 'r2_f0', 'F0'), (axes[1], 'r2_spl', 'SPL')]:
        if trf:
            ns = [n for n in sorted(trf) if n <= cap and n in GRID_MATCH]
            s = STYLE['TransRF']
            ax.plot(ns, [trf[n][tkey] for n in ns], color=s['color'], marker=s['marker'],
                    ls=s['ls'], lw=s['lw'], ms=s['ms'], label=s['label'], zorder=3)
        for m in ('TabPFN',):
            ns = [n for n in sorted(alt[m]) if n <= cap and n in GRID_MATCH]
            means = np.array([alt[m][n][tkey].mean() for n in ns])
            stds = np.array([alt[m][n][tkey].std() for n in ns])
            s = STYLE[m]
            ax.fill_between(ns, means - stds, means + stds, color=s['color'], alpha=0.15, zorder=3)
            ax.plot(ns, means, color=s['color'], marker=s['marker'], ls=s['ls'],
                    lw=s['lw'], ms=s['ms'], label=s['label'], zorder=5)
        if logx:
            ax.set_xscale('log')
            ax.set_xlabel('Number of BM training samples (log scale)')
        else:
            ax.set_xlabel('Number of BM training samples (linear scale)')
            ax.set_xlim(0, n_max + 2)
        ax.set_title(tlabel, fontweight='bold')
        ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
        ax.set_ylim(-1.0, 1.05)
        ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel('R²')
    axes[1].legend(loc='lower right', framealpha=0.95, fontsize=9)
    title = ('BCM→BM (log scale, all N): weak source — TabPFN leads' if logx
             else f'BCM→BM small-data zoom (N ≤ {n_max}): weak source — TabPFN leads')
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.0)
    fig.tight_layout()
    path = os.path.join(figs_dir, fname)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('BM SHOWCASE — generating presentation-quality figures')
    print('=' * 70)
    os.makedirs(figs_dir, exist_ok=True)
    fig_headline()
    fig_headline_zoom()
    fig_headline_zoom(logx=True, fname='bm_showcase_headline_log.png')
    fig_sim_budget()
    fig_bootstrap()
    fig_table()
    print('=' * 70)
    print('Done.')


if __name__ == '__main__':
    main()
