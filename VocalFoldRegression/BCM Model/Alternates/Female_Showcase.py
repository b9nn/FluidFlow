"""
Female BCM Showcase — Male->Female BCM transfer vs non-transfer (GP, TabPFN).
Mirrors Beam_Membrane/BM_Showcase.py (4-figure set).

Female transfer comparator notes:
  - Transfer CSV has 3 regressor families: RF, NN, PR. All plotted as
    individually labeled lines (RF highlighted, NN/PR as variants).
  - CSV has means only, no per-replicate data, so bootstrap fig shows
    GP/TabPFN per-replicate boxes with RF transfer as a dashed reference
    line (not a box).
  - No "small N vs full N" split — transfer CSV covers N=[25..843] in
    one sweep.

Story: Male->Female BCM is well aligned (same physics, demographic
shift). RF transfer holds r²_avg ≈ 0.72 across N=25..843 — extremely
strong baseline. TabPFN catches at N≈75 and dominates from N=100,
reaching 0.97 at N=500. GP alone does NOT catch — TabPFN's pretrained
prior is the load-bearing component.

Inputs:
  - VocalFoldRegression/BCM Model/Alternates/results/alternates_results.json
  - VocalFoldRegression/BCM Model/ResgressorAnalysis/figs/
    all_regressors_transfer_comparison.csv

Outputs (VocalFoldRegression/BCM Model/Alternates/figs/):
  - female_showcase_headline.png    line chart, per-method legend
  - female_showcase_sim_budget.png  bar chart, samples to hit R^2 thresholds
  - female_showcase_bootstrap.png   boxplots per N (alternates only, transfer as ref)
  - female_showcase_table.png       per-N table with winner highlighted
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

METHOD_STYLE = {
    'TabPFN':   dict(color='#d97706', marker='o', ls='-',  lw=2.8, ms=8, alpha=1.0, label='TabPFN (non-transfer, ours)'),
    'GP':       dict(color='#0f766e', marker='o', ls='-',  lw=2.8, ms=8, alpha=1.0, label='GP (non-transfer, ours)'),
    'RF':       dict(color='#1e293b', marker='s', ls='-',  lw=2.4, ms=7, alpha=0.95, label='RF Male->Female transfer (Brian)'),
    'NN':       dict(color='#64748b', marker='D', ls='--', lw=1.6, ms=5, alpha=0.85, label='NN Male->Female transfer (Brian)'),
    'PR':       dict(color='#94a3b8', marker='^', ls=':',  lw=1.6, ms=5, alpha=0.85, label='PR Male->Female transfer (Brian)'),
}

LEGEND_ORDER = ['TabPFN', 'GP', 'RF', 'NN', 'PR']


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
    """Returns {regressor: {N: r2_avg}}."""
    df = pd.read_csv(TRANSFER_CSV)
    out = {}
    for reg in df['regressor'].unique():
        sub = df[df['regressor'] == reg].sort_values('sample_size')
        out[reg] = {int(n): float(v) for n, v in
                    zip(sub['sample_size'], sub['r2_avg'])}
    return out


def _ordered_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles, ordered_labels = [], []
    for key in LEGEND_ORDER:
        lbl = METHOD_STYLE[key]['label']
        if lbl in label_to_handle:
            ordered_handles.append(label_to_handle[lbl])
            ordered_labels.append(lbl)
    for lbl, h in label_to_handle.items():
        if lbl not in ordered_labels:
            ordered_handles.append(h)
            ordered_labels.append(lbl)
    ax.legend(ordered_handles, ordered_labels, loc='lower right',
              framealpha=0.95, fontsize=9)


# ============================================================================
# Figure 1: Headline
# ============================================================================
def fig_headline():
    alt = load_alternates()
    xfer = load_transfer()

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # Each transfer regressor as its OWN labeled line
    for reg in ['NN', 'PR', 'RF']:
        if reg not in xfer:
            continue
        ns = sorted(xfer[reg])
        ys = [xfer[reg][n] for n in ns]
        s = METHOD_STYLE[reg]
        ax.plot(ns, ys, color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], ms=s['ms'], alpha=s['alpha'], label=s['label'],
                zorder=3 if reg == 'RF' else 2)

    # GP + TabPFN with std bands
    for method in ('GP', 'TabPFN'):
        if method not in alt:
            continue
        ns = sorted(alt[method])
        means = np.array([alt[method][n].mean() for n in ns])
        stds = np.array([alt[method][n].std() for n in ns])
        s = METHOD_STYLE[method]
        ax.fill_between(ns, means - stds, means + stds, color=s['color'], alpha=0.15, zorder=3)
        ax.plot(ns, means, color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], ms=s['ms'], label=s['label'], zorder=5)

    rf = xfer.get('RF', {})

    # Catch-up callout at N=75
    if 75 in alt['TabPFN']:
        tab_75 = alt['TabPFN'][75].mean()
        if 50 in rf and 100 in rf:
            rf_75 = 0.5 * (rf[50] + rf[100])
        else:
            rf_75 = rf.get(100, np.nan)
        ax.annotate(
            f'catch-up at N≈75\nTabPFN {tab_75:.2f}  ≈  RF transfer {rf_75:.2f}',
            xy=(75, tab_75), xytext=(6, 0.92),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#92400e', lw=1),
            arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.2,
                            connectionstyle='arc3,rad=0.15'))

    # Divergence callout at N=500
    if 500 in alt['TabPFN'] and 500 in rf:
        tab_500 = alt['TabPFN'][500].mean()
        rf_500 = rf[500]
        gap = tab_500 - rf_500
        ax.annotate(
            f'+{gap:.2f} R² at N=500\nTabPFN {tab_500:.2f}  vs  RF transfer {rf_500:.2f}',
            xy=(500, tab_500), xytext=(110, 0.42),
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
    _ordered_legend(ax)

    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'female_showcase_headline.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 2: Sim-budget
# ============================================================================
def fig_sim_budget():
    alt = load_alternates()
    xfer = load_transfer()

    curves = {
        'TabPFN': sorted((n, alt['TabPFN'][n].mean()) for n in alt['TabPFN']),
        'GP':     sorted((n, alt['GP'][n].mean()) for n in alt['GP']),
        'RF':     sorted(xfer.get('RF', {}).items()),
        'NN':     sorted(xfer.get('NN', {}).items()),
        'PR':     sorted(xfer.get('PR', {}).items()),
    }

    def n_to_reach(curve, threshold):
        prev = None
        for n, v in curve:
            if v >= threshold:
                if prev is None:
                    return n
                n_prev, v_prev = prev
                if v == v_prev:
                    return n
                t = (threshold - v_prev) / (v - v_prev)
                log_n = np.log10(n_prev) + t * (np.log10(n) - np.log10(n_prev))
                return 10 ** log_n
            prev = (n, v)
        return None

    # Female RF transfer hovers at ~0.72-0.75 — choose thresholds that
    # discriminate methods (0.75 RF can reach, 0.90 RF cannot).
    thresholds = [0.75, 0.90]
    method_order = ['TabPFN', 'GP', 'RF', 'NN', 'PR']

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    cap = 900
    for ax, thr in zip(axes, thresholds):
        budgets, colors = [], []
        for m in method_order:
            n_req = n_to_reach(curves[m], thr)
            budgets.append(n_req)
            colors.append(METHOD_STYLE[m]['color'])

        ys = np.arange(len(method_order))
        plot_vals = [b if (b is not None and not np.isnan(b)) else cap for b in budgets]
        bars = ax.barh(ys, plot_vals, color=colors, edgecolor='black', lw=0.7,
                       alpha=0.9)
        for i, (bar, b) in enumerate(zip(bars, budgets)):
            if b is None:
                ax.text(cap * 0.5, ys[i], 'not reached in tested range',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            else:
                ax.text(min(b, cap) + 8, ys[i], f'N = {int(round(b))}',
                        ha='left', va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(ys)
        ax.set_yticklabels(method_order)
        ax.set_xlabel('Female BCM training samples needed')
        ax.set_title(f'To reach average R² ≥ {thr}', fontweight='bold')
        ax.set_xlim(0, cap)
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle('Simulation budget: Female BCM samples required to hit accuracy thresholds',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'female_showcase_sim_budget.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 3: Bootstrap (alternates per-replicate; transfer as dashed reference)
# ============================================================================
def fig_bootstrap():
    alt = load_alternates()
    xfer = load_transfer()

    # Pick N values overlapping with the transfer CSV
    rf = xfer.get('RF', {})
    candidates = [10, 20, 50, 100, 500]  # add 10/20 even where transfer absent
    key_ns = [n for n in candidates
              if n in alt.get('GP', {}) and n in alt.get('TabPFN', {})]
    if not key_ns:
        print('  WARN: no overlapping N for bootstrap')
        return

    fig, axes = plt.subplots(1, len(key_ns), figsize=(3.2 * len(key_ns) + 1, 5.5),
                             sharey=True)
    if len(key_ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, key_ns):
        data, labels, face_colors = [], [], []
        for m in ('TabPFN', 'GP'):
            data.append(alt[m][n])
            labels.append(m)
            face_colors.append(METHOD_STYLE[m]['color'])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', lw=1.5),
                        boxprops=dict(lw=1),
                        whiskerprops=dict(lw=1),
                        capprops=dict(lw=1),
                        flierprops=dict(marker='o', markersize=4,
                                        markerfacecolor='white', alpha=0.7))
        for patch, c in zip(bp['boxes'], face_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)

        for i, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(n + i).random(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color='black', s=14, alpha=0.55, zorder=4)

        # RF transfer reference line — interpolate if N not directly in CSV
        rf_val = rf.get(n)
        if rf_val is None and rf:
            # nearest-neighbor on log-N
            ns_avail = sorted(rf)
            below = [m for m in ns_avail if m <= n]
            above = [m for m in ns_avail if m >= n]
            if below and above and below[-1] != above[0]:
                n_lo, n_hi = below[-1], above[0]
                v_lo, v_hi = rf[n_lo], rf[n_hi]
                t = (np.log10(n) - np.log10(n_lo)) / (np.log10(n_hi) - np.log10(n_lo))
                rf_val = v_lo + t * (v_hi - v_lo)
                label_suffix = ' (interp)'
            else:
                rf_val = rf[below[-1] if below else above[0]]
                label_suffix = ' (nearest)'
        else:
            label_suffix = ''
        if rf_val is not None:
            s = METHOD_STYLE['RF']
            ax.axhline(rf_val, color=s['color'], ls='--', lw=1.8, alpha=0.9, zorder=2,
                       label=f'RF transfer{label_suffix}: {rf_val:.2f}')
            ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

        ax.set_title(f'N = {n}', fontweight='bold')
        ax.set_ylim(-0.6, 1.05)
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.grid(True, axis='y', alpha=0.25)
        if n == key_ns[0]:
            ax.set_ylabel('Average R²  (F0 + SPL) / 2')

    fig.suptitle('Bootstrap distributions per N — alternates (10 reps) with RF '
                 'transfer reference line (CSV means only)',
                 fontweight='bold', fontsize=12.5, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'female_showcase_bootstrap.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 4: Per-N head-to-head table
# ============================================================================
def fig_table():
    alt = load_alternates()
    xfer = load_transfer()
    rf = xfer.get('RF', {})

    rows_n = sorted({n for m in ('GP', 'TabPFN') for n in alt.get(m, {})})
    rows_n = [n for n in rows_n if n >= 5]

    headers = ['N', 'GP', 'RF transfer (Brian)', 'TabPFN']

    def fmt_alt(arr):
        if arr is None or len(arr) == 0:
            return '—', None
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return f'{mean:+.3f} ± {std:.2f}', mean

    def fmt_rf(n):
        if n in rf:
            return f'{rf[n]:+.3f}', rf[n]
        # interpolate if both neighbours exist
        ns_avail = sorted(rf)
        below = [m for m in ns_avail if m <= n]
        above = [m for m in ns_avail if m >= n]
        if below and above and below[-1] != above[0]:
            n_lo, n_hi = below[-1], above[0]
            v_lo, v_hi = rf[n_lo], rf[n_hi]
            t = (np.log10(n) - np.log10(n_lo)) / (np.log10(n_hi) - np.log10(n_lo))
            v = v_lo + t * (v_hi - v_lo)
            return f'{v:+.3f}  (interp)', v
        return '—', None

    cell_text, cell_colors = [], []
    bg_default = '#ffffff'
    bg_winner = '#dcfce7'
    bg_alt_row = '#f8fafc'

    for i, n in enumerate(rows_n):
        gp_text, gp_mean = fmt_alt(alt.get('GP', {}).get(n))
        tab_text, tab_mean = fmt_alt(alt.get('TabPFN', {}).get(n))
        rf_text, rf_mean = fmt_rf(n)

        candidates = [(gp_mean, 1), (rf_mean, 2), (tab_mean, 3)]
        valid = [(v, idx) for v, idx in candidates if v is not None]
        winner_col = max(valid, key=lambda t: t[0])[1] if valid else None

        cell_text.append([f'{n}', gp_text, rf_text, tab_text])
        row_bg = bg_alt_row if i % 2 == 1 else bg_default
        row_colors = [row_bg] * 4
        if winner_col is not None:
            row_colors[winner_col] = bg_winner
        cell_colors.append(row_colors)

    n_rows = len(rows_n)
    fig_h = 0.55 + 0.40 * n_rows
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text, colLabels=headers,
        cellColours=cell_colors, colColours=['#0f172a'] * 4,
        cellLoc='center', loc='upper center',
        colWidths=[0.10, 0.30, 0.30, 0.30],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)

    bg_winner_rgb = tuple(int(bg_winner[i:i+2], 16) / 255 for i in (1, 3, 5))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
            cell.set_edgecolor('#0f172a')
        else:
            cell.set_edgecolor('#cbd5e1')
            if cell.get_facecolor()[:3] == bg_winner_rgb:
                cell.get_text().set_color('#065f46')
                cell.get_text().set_fontweight('bold')

    ax.set_title(
        'Female BCM head-to-head: average R² (F0+SPL)/2  —  winner per N highlighted',
        fontweight='bold', fontsize=13, pad=14)
    fig.text(0.5, 0.02,
             'GP & TabPFN: mean ± std over 10 bootstrap replicates.  '
             'RF transfer: mean only (per-replicate not committed).  '
             '"interp" = log-N interpolation between adjacent CSV rows.',
             ha='center', fontsize=8.5, style='italic', color='#475569')
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(figs_dir, 'female_showcase_table.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('FEMALE BCM SHOWCASE — generating presentation-quality figures')
    print('=' * 70)
    os.makedirs(figs_dir, exist_ok=True)
    fig_headline()
    fig_sim_budget()
    fig_bootstrap()
    fig_table()
    print('=' * 70)
    print('Done.')


if __name__ == '__main__':
    main()
