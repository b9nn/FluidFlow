"""
TBCM Showcase — presentation-quality visualizations for BCM->TBCM.
Mirrors Beam_Membrane/BM_Showcase.py (4-figure set).

Story (differs from BM):
  BCM->TBCM is well aligned (same physics family, geometry shift).
  Alternates lead at small N; transfer catches up by N≈200; curves
  essentially tie at N=500. Different framing than BM's "alternates
  dominate at every N" headline.

Inputs:
  - results/alternates_results.json     (GP+TabPFN, per-replicate)
  - results/rf_transfer_small_n.json    (per-replicate transfer N=10..500)
  - results/rf_transfer_results.json    (Callum's full-N, N>=1379, means only)

Outputs (TBCM/figs/):
  - tbcm_showcase_headline.png    line chart, per-method legend
  - tbcm_showcase_sim_budget.png  bar chart, samples to hit R^2 thresholds
  - tbcm_showcase_bootstrap.png   boxplots per N (alternates + transfer)
  - tbcm_showcase_table.png       per-N table with winner highlighted
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

# Per-method colors + markers (every line gets its own identity in the legend).
METHOD_STYLE = {
    'TabPFN':      dict(color='#d97706', marker='o', ls='-',  lw=2.8, ms=8, alpha=1.0, label='TabPFN (non-transfer, ours)'),
    'GP':          dict(color='#0f766e', marker='o', ls='-',  lw=2.8, ms=8, alpha=1.0, label='GP (non-transfer, ours)'),
    'TransRF':     dict(color='#1e293b', marker='s', ls='-',  lw=2.2, ms=7, alpha=0.95, label='TransRF (best of Callum\'s transfer)'),
    'Feature Aug': dict(color='#64748b', marker='D', ls='--', lw=1.6, ms=5, alpha=0.85, label='Feature Augmentation (transfer variant)'),
    'Residual':    dict(color='#94a3b8', marker='^', ls='--', lw=1.6, ms=5, alpha=0.85, label='Residual Correction (transfer variant)'),
    'Target Only': dict(color='#cbd5e1', marker='x', ls=':',  lw=1.6, ms=6, alpha=0.95, label='Target Only (no transfer baseline)'),
    'TransRF_fullN': dict(color='#1e293b', marker='s', ls=(0, (4, 3)), lw=1.4, ms=4, alpha=0.55, label='TransRF at full N (Callum\'s larger-N runs)'),
}

# Method key in our small-N JSON -> display label
JSON_TO_LABEL = {
    'target':    'Target Only',
    'residual':  'Residual',
    'augmented': 'Feature Aug',
    'transrf':   'TransRF',
}

# Display order in legend (top to bottom)
LEGEND_ORDER = ['TabPFN', 'GP', 'TransRF', 'Feature Aug', 'Residual', 'Target Only', 'TransRF_fullN']


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


def load_small_transfer_per_replicate():
    """Returns {label: {N: per-replicate avg-R^2 array}}."""
    with open(os.path.join(results_dir, 'rf_transfer_small_n.json')) as f:
        raw = json.load(f)
    out = {label: {} for label in JSON_TO_LABEL.values()}
    for key, label in JSON_TO_LABEL.items():
        block = raw.get(key, {})
        for n_str, rec in block.items():
            if not n_str.isdigit():
                continue
            f0 = np.asarray(rec['r2_f0'])
            spl = np.asarray(rec['r2_spl'])
            out[label][int(n_str)] = (f0 + spl) / 2.0
    return out


def small_transfer_means(per_rep):
    return {label: {n: float(arr.mean()) for n, arr in by_n.items()}
            for label, by_n in per_rep.items()}


def load_full_n_transfer():
    """Callum's full-N transfer, means only. Returns {label: [(n, mean), ...]}."""
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
    if rows and any('transrf_f0_r2_mean' in r for r in rows):
        methods['TransRF'] = 'transrf'
    out = {m: [] for m in methods}
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


def _ordered_legend(ax):
    """Reorder ax legend to follow LEGEND_ORDER."""
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles, ordered_labels = [], []
    for key in LEGEND_ORDER:
        lbl = METHOD_STYLE[key]['label']
        if lbl in label_to_handle:
            ordered_handles.append(label_to_handle[lbl])
            ordered_labels.append(lbl)
    # Append any leftover labels not in our ordering
    for lbl, h in label_to_handle.items():
        if lbl not in ordered_labels:
            ordered_handles.append(h)
            ordered_labels.append(lbl)
    ax.legend(ordered_handles, ordered_labels, loc='lower right',
              framealpha=0.95, fontsize=9)


# ============================================================================
# Figure 1: Headline — Average R^2 vs N
# ============================================================================
def fig_headline():
    alt = load_alternates()
    small_rep = load_small_transfer_per_replicate()
    small_mean = small_transfer_means(small_rep)
    full = load_full_n_transfer()

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # Plot each transfer variant as its OWN labeled line.
    for label in ['Target Only', 'Residual', 'Feature Aug', 'TransRF']:
        if label not in small_mean or not small_mean[label]:
            continue
        ns = sorted(small_mean[label])
        ys = [small_mean[label][n] for n in ns]
        s = METHOD_STYLE[label]
        ax.plot(ns, ys, color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], ms=s['ms'], alpha=s['alpha'], label=s['label'],
                zorder=3 if label == 'TransRF' else 2)

    # TransRF at full N (dashed continuation)
    if full.get('TransRF'):
        full_tr = sorted(full['TransRF'])
        s = METHOD_STYLE['TransRF_fullN']
        ax.plot([n for n, _ in full_tr], [v for _, v in full_tr],
                color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], ms=s['ms'], alpha=s['alpha'], label=s['label'],
                zorder=2)

    # GP + TabPFN bold with std bands
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

    # Small-N gap callout at N=50
    if 50 in alt['TabPFN']:
        tab_50 = alt['TabPFN'][50].mean()
        best_transfer_50 = max(small_mean[m].get(50, -np.inf) for m in small_mean)
        gap = tab_50 - best_transfer_50
        ax.annotate(
            f'+{gap:.2f} R² at N=50\nTabPFN {tab_50:.2f}  vs  best transfer {best_transfer_50:.2f}',
            xy=(50, tab_50), xytext=(6, 0.05),
            fontsize=10, fontweight='bold', color='#111',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec='#92400e', lw=1),
            arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.2,
                            connectionstyle='arc3,rad=0.15'))

    # Convergence callout at N=500
    if 500 in alt['TabPFN']:
        tab_500 = alt['TabPFN'][500].mean()
        best_transfer_500 = max(small_mean[m].get(500, -np.inf) for m in small_mean)
        gap = tab_500 - best_transfer_500
        ax.annotate(
            f'gap +{gap:.3f} at N=500\nTabPFN {tab_500:.3f}  ≈  TransRF {best_transfer_500:.3f}',
            xy=(500, tab_500), xytext=(120, 0.55),
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
    _ordered_legend(ax)

    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'tbcm_showcase_headline.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 2: Sim-budget — TBCM sims needed to hit accuracy thresholds
# ============================================================================
def fig_sim_budget():
    alt = load_alternates()
    small_mean = small_transfer_means(load_small_transfer_per_replicate())

    curves = {
        'TabPFN':      sorted((n, alt['TabPFN'][n].mean()) for n in alt['TabPFN']),
        'GP':          sorted((n, alt['GP'][n].mean()) for n in alt['GP']),
        'TransRF':     sorted(small_mean['TransRF'].items()),
        'Feature Aug': sorted(small_mean['Feature Aug'].items()),
        'Residual':    sorted(small_mean['Residual'].items()),
        'Target Only': sorted(small_mean['Target Only'].items()),
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

    # TBCM is easier than BM — use higher thresholds
    thresholds = [0.7, 0.9]
    method_order = ['TabPFN', 'GP', 'TransRF', 'Feature Aug', 'Residual', 'Target Only']

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    cap = 700
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
        ax.set_xlabel('TBCM training samples needed')
        ax.set_title(f'To reach average R² ≥ {thr}', fontweight='bold')
        ax.set_xlim(0, cap)
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle('Simulation budget: TBCM samples required to hit accuracy thresholds',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'tbcm_showcase_sim_budget.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 3: Bootstrap distributions at key N (alternates + transfer per-rep)
# ============================================================================
def fig_bootstrap():
    alt = load_alternates()
    small_rep = load_small_transfer_per_replicate()

    candidates = [10, 20, 50, 100, 200, 500]
    key_ns = [n for n in candidates
              if n in alt.get('GP', {})
              and n in alt.get('TabPFN', {})
              and n in small_rep.get('TransRF', {})]
    if not key_ns:
        print('  WARN: no overlapping N for bootstrap')
        return

    fig, axes = plt.subplots(1, len(key_ns), figsize=(3.4 * len(key_ns) + 1, 5.5),
                             sharey=True)
    if len(key_ns) == 1:
        axes = [axes]

    box_methods = ['TabPFN', 'GP', 'TransRF', 'Feature Aug']

    for ax, n in zip(axes, key_ns):
        data, labels, face_colors = [], [], []
        for m in box_methods:
            if m in ('GP', 'TabPFN') and n in alt[m]:
                data.append(alt[m][n])
                labels.append(m)
                face_colors.append(METHOD_STYLE[m]['color'])
            elif m in small_rep and n in small_rep[m]:
                data.append(small_rep[m][n])
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

        # Scatter individual replicates
        for i, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(n + i).random(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color='black', s=14, alpha=0.55, zorder=4)

        ax.set_title(f'N = {n}', fontweight='bold')
        ax.set_ylim(-0.6, 1.05)
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelrotation=25)
        if n == key_ns[0]:
            ax.set_ylabel('Average R²  (F0 + SPL) / 2')

    fig.suptitle('Bootstrap distributions per N — alternates vs transfer variants '
                 '(10 reps alternates, 5 reps transfer)',
                 fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(figs_dir, 'tbcm_showcase_bootstrap.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ============================================================================
# Figure 4: Per-N head-to-head table, winner per row highlighted
# ============================================================================
def fig_table():
    alt = load_alternates()
    small_rep = load_small_transfer_per_replicate()

    rows_n = sorted({n for m in ('GP', 'TabPFN') for n in alt.get(m, {})})
    rows_n = [n for n in rows_n if n >= 5]

    headers = ['N', 'GP', 'TransRF', 'TabPFN']

    def fmt_alt(arr):
        if arr is None or len(arr) == 0:
            return '—', None
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return f'{mean:+.3f} ± {std:.2f}', mean

    def fmt_transfer(arr):
        if arr is None or len(arr) == 0:
            return '—', None
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return f'{mean:+.3f} ± {std:.2f}', mean

    cell_text, cell_colors = [], []
    bg_default = '#ffffff'
    bg_winner = '#dcfce7'
    bg_alt_row = '#f8fafc'

    for i, n in enumerate(rows_n):
        gp_text, gp_mean = fmt_alt(alt.get('GP', {}).get(n))
        tab_text, tab_mean = fmt_alt(alt.get('TabPFN', {}).get(n))
        trf_text, trf_mean = fmt_transfer(small_rep.get('TransRF', {}).get(n))

        candidates = [(gp_mean, 1), (trf_mean, 2), (tab_mean, 3)]
        valid = [(v, idx) for v, idx in candidates if v is not None]
        winner_col = max(valid, key=lambda t: t[0])[1] if valid else None

        cell_text.append([f'{n}', gp_text, trf_text, tab_text])
        row_bg = bg_alt_row if i % 2 == 1 else bg_default
        row_colors = [row_bg] * 4
        if winner_col is not None:
            row_colors[winner_col] = bg_winner
        cell_colors.append(row_colors)

    n_rows = len(rows_n)
    fig_h = 0.55 + 0.40 * n_rows
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
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
        'TBCM head-to-head: average R² (F0+SPL)/2  —  winner per N highlighted',
        fontweight='bold', fontsize=13, pad=14)
    fig.text(0.5, 0.02,
             'GP & TabPFN: mean ± std over 10 bootstrap replicates.  '
             'TransRF: mean ± std over 5 replicates from rf_transfer_small_n.json.  '
             'Source: alternates_results.json + rf_transfer_small_n.json.',
             ha='center', fontsize=8.5, style='italic', color='#475569')
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(figs_dir, 'tbcm_showcase_table.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    print('=' * 70)
    print('TBCM SHOWCASE — generating presentation-quality figures')
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
