"""
Full Domain Comparison: BCM vs TBCM vs BM

Comprehensive statistical and distributional comparison of all three
datasets to understand domain gaps, overlap regions, and why transfer
works (or doesn't) between source/target pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp, spearmanr, pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']


def overlap_fraction(a, b):
    """Fraction of range overlap between two arrays."""
    lo = max(a.min(), b.min())
    hi = min(a.max(), b.max())
    if lo >= hi:
        return 0.0
    union = max(a.max(), b.max()) - min(a.min(), b.min())
    return (hi - lo) / union if union > 0 else 0.0


def coverage_fraction(source, target):
    """What fraction of the target range is covered by the source range."""
    t_lo, t_hi = target.min(), target.max()
    s_lo, s_hi = source.min(), source.max()
    overlap_lo = max(t_lo, s_lo)
    overlap_hi = min(t_hi, s_hi)
    if overlap_lo >= overlap_hi:
        return 0.0
    return (overlap_hi - overlap_lo) / (t_hi - t_lo)


def main():
    print("=" * 70)
    print("FULL DOMAIN COMPARISON: BCM vs TBCM vs BM")
    print("=" * 70)

    # ---- Load datasets ----
    print("\nLoading datasets...")
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'), index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})

    df_tbcm = pd.read_csv(os.path.join(script_dir, '..', 'TBCM', 'dataset_TBCM.csv'),
                           index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'})
    df_tbcm = df_tbcm.drop(columns=['PL'])

    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    df_bm = pd.read_csv(bm_path)
    if 'Ps' in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})
    df_bm = df_bm.dropna(subset=['F0', 'SPL'])

    datasets = {'BCM': df_bcm, 'TBCM': df_tbcm, 'BM': df_bm}
    colors = {'BCM': '#2196F3', 'TBCM': '#FF5722', 'BM': '#4CAF50'}

    for name, df in datasets.items():
        print(f"  {name}: {len(df)} samples, columns: {list(df.columns)}")

    # ==================== 1. SUMMARY STATISTICS ====================
    print("\n\n" + "=" * 70)
    print("1. SUMMARY STATISTICS")
    print("=" * 70)

    all_cols = SHARED_FEATURES + TARGETS
    # BM has a_LCA
    if 'a_LCA' in df_bm.columns:
        bm_extra = ['a_LCA']
    else:
        bm_extra = []

    print(f"\n{'Feature':<8} {'Dataset':<7} {'Count':>7} {'Min':>10} {'Max':>10} "
          f"{'Mean':>10} {'Median':>10} {'Std':>10} {'Skew':>8} {'Kurt':>8}")
    print("-" * 100)

    for col in all_cols:
        for name, df in datasets.items():
            if col in df.columns:
                s = df[col]
                print(f"{col:<8} {name:<7} {len(s):>7} {s.min():>10.2f} {s.max():>10.2f} "
                      f"{s.mean():>10.2f} {s.median():>10.2f} {s.std():>10.2f} "
                      f"{s.skew():>8.3f} {s.kurtosis():>8.3f}")
        print()

    # BM-only features
    for col in bm_extra:
        s = df_bm[col]
        print(f"{col:<8} {'BM':<7} {len(s):>7} {s.min():>10.3f} {s.max():>10.3f} "
              f"{s.mean():>10.3f} {s.median():>10.3f} {s.std():>10.3f} "
              f"{s.skew():>8.3f} {s.kurtosis():>8.3f}")

    # ==================== 2. RANGE OVERLAP ====================
    print("\n\n" + "=" * 70)
    print("2. RANGE OVERLAP AND COVERAGE")
    print("=" * 70)

    pairs = [('BCM', 'BM'), ('TBCM', 'BM'), ('BCM', 'TBCM')]

    for src_name, tgt_name in pairs:
        df_src = datasets[src_name]
        df_tgt = datasets[tgt_name]
        print(f"\n  {src_name} -> {tgt_name}:")
        print(f"  {'Feature':<8} {'Overlap%':>10} {'SrcCoversTgt%':>15} "
              f"{'Src range':<25} {'Tgt range':<25}")
        print(f"  {'-'*85}")
        for col in all_cols:
            if col in df_src.columns and col in df_tgt.columns:
                ovr = overlap_fraction(df_src[col].values, df_tgt[col].values)
                cov = coverage_fraction(df_src[col], df_tgt[col])
                src_r = f"[{df_src[col].min():.1f}, {df_src[col].max():.1f}]"
                tgt_r = f"[{df_tgt[col].min():.1f}, {df_tgt[col].max():.1f}]"
                print(f"  {col:<8} {ovr*100:>9.1f}% {cov*100:>14.1f}% "
                      f"{src_r:<25} {tgt_r:<25}")

    # ==================== 3. DISTRIBUTIONAL DISTANCE (KS test) ====================
    print("\n\n" + "=" * 70)
    print("3. KOLMOGOROV-SMIRNOV TEST (distributional distance)")
    print("=" * 70)
    print("  KS statistic: 0 = identical distributions, 1 = completely different")

    for src_name, tgt_name in pairs:
        df_src = datasets[src_name]
        df_tgt = datasets[tgt_name]
        print(f"\n  {src_name} vs {tgt_name}:")
        print(f"  {'Feature':<8} {'KS stat':>10} {'p-value':>12} {'Verdict':<20}")
        print(f"  {'-'*55}")
        for col in all_cols:
            if col in df_src.columns and col in df_tgt.columns:
                # Subsample for efficiency
                s1 = df_src[col].sample(n=min(5000, len(df_src)), random_state=42).values
                s2 = df_tgt[col].sample(n=min(5000, len(df_tgt)), random_state=42).values
                ks, pval = ks_2samp(s1, s2)
                if ks < 0.1:
                    verdict = "Very similar"
                elif ks < 0.25:
                    verdict = "Moderate gap"
                elif ks < 0.5:
                    verdict = "Large gap"
                else:
                    verdict = "Very different"
                print(f"  {col:<8} {ks:>10.4f} {pval:>12.2e} {verdict:<20}")

    # ==================== 4. PERCENTILE COMPARISON ====================
    print("\n\n" + "=" * 70)
    print("4. PERCENTILE COMPARISON")
    print("=" * 70)

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for col in all_cols:
        print(f"\n  {col}:")
        header = f"  {'Dataset':<7}"
        for p in percentiles:
            header += f" {'P'+str(p):>8}"
        header += f" {'IQR':>10}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for name, df in datasets.items():
            if col in df.columns:
                pvals = np.percentile(df[col], percentiles)
                iqr = pvals[4] - pvals[2]  # P75 - P25
                line = f"  {name:<7}"
                for pv in pvals:
                    line += f" {pv:>8.2f}"
                line += f" {iqr:>10.2f}"
                print(line)

    # ==================== 5. CORRELATION STRUCTURE ====================
    print("\n\n" + "=" * 70)
    print("5. INTERNAL CORRELATION STRUCTURE")
    print("=" * 70)
    print("  Pearson correlation between each input and each output")

    for name, df in datasets.items():
        print(f"\n  {name}:")
        feats = [c for c in SHARED_FEATURES + bm_extra if c in df.columns]
        header = f"  {'':>8}"
        for f in feats:
            header += f" {f:>8}"
        print(header)
        for t in TARGETS:
            line = f"  {t:>8}"
            for f in feats:
                r, _ = pearsonr(df[f], df[t])
                line += f" {r:>8.3f}"
            print(line)

    # ==================== 6. CROSS-DOMAIN CORRELATION ====================
    print("\n\n" + "=" * 70)
    print("6. CROSS-DOMAIN SPEARMAN CORRELATION")
    print("=" * 70)
    print("  How well does one domain's output ranking predict another's?")
    print("  (Evaluated on shared input space via source model predictions)")

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # --- Plot 1: Marginal distributions (histograms + KDE) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Marginal Distributions: BCM vs TBCM vs BM', fontsize=16, fontweight='bold')

    plot_cols = ['a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    for idx, col in enumerate(plot_cols):
        row, c = divmod(idx, 3)
        ax = axes[row, c]
        for name, df in datasets.items():
            if col in df.columns:
                data = df[col].values
                ax.hist(data, bins=60, alpha=0.35, color=colors[name],
                        label=f'{name} (n={len(df)})', density=True, edgecolor='none')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    # Extra panel: a_LCA (BM only)
    ax = axes[1, 2]
    if 'a_LCA' in df_bm.columns:
        ax.hist(df_bm['a_LCA'].values, bins=40, alpha=0.5, color=colors['BM'],
                label='BM only', density=True, edgecolor='none')
        ax.set_xlabel('a_LCA', fontsize=12)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('a_LCA Distribution (BM only)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_marginal_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: Box plots side-by-side ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Range Comparison: BCM vs TBCM vs BM', fontsize=14, fontweight='bold')

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        data_list = []
        labels_list = []
        color_list = []
        for name, df in datasets.items():
            if col in df.columns:
                data_list.append(df[col].values)
                labels_list.append(name)
                color_list.append(colors[name])

        bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_boxplots.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 3: PS vs F0 and PS vs SPL scatter (domain overlap visualization) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Input-Output Relationships Across Domains', fontsize=14, fontweight='bold')

    # Subsample for plotting
    n_plot = 2000
    dfs_plot = {}
    for name, df in datasets.items():
        if len(df) > n_plot:
            dfs_plot[name] = df.sample(n=n_plot, random_state=42)
        else:
            dfs_plot[name] = df

    scatter_pairs = [('PS', 'F0'), ('PS', 'SPL'), ('a_CT', 'F0'), ('a_TA', 'SPL')]
    for idx, (xc, yc) in enumerate(scatter_pairs):
        row, col_idx = divmod(idx, 2)
        ax = axes[row, col_idx]
        for name in ['BCM', 'TBCM', 'BM']:
            df = dfs_plot[name]
            if xc in df.columns and yc in df.columns:
                ax.scatter(df[xc], df[yc], alpha=0.15, s=8,
                           color=colors[name], label=name)
        ax.set_xlabel(xc, fontsize=11)
        ax.set_ylabel(yc, fontsize=11)
        ax.set_title(f'{xc} vs {yc}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, markerscale=3)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_scatter_relationships.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 4: CDF comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Cumulative Distribution Functions: BCM vs TBCM vs BM',
                 fontsize=14, fontweight='bold')

    for idx, col in enumerate(plot_cols):
        row, c = divmod(idx, 3)
        ax = axes[row, c]
        for name, df in datasets.items():
            if col in df.columns:
                sorted_vals = np.sort(df[col].values)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                # Subsample for plotting efficiency
                step = max(1, len(sorted_vals) // 2000)
                ax.plot(sorted_vals[::step], cdf[::step],
                        color=colors[name], label=name, linewidth=2)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('CDF', fontsize=10)
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # a_LCA CDF
    ax = axes[1, 2]
    if 'a_LCA' in df_bm.columns:
        sorted_vals = np.sort(df_bm['a_LCA'].values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=colors['BM'], label='BM', linewidth=2)
        ax.set_xlabel('a_LCA', fontsize=11)
        ax.set_ylabel('CDF', fontsize=10)
        ax.set_title('a_LCA (BM only)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_cdfs.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 5: Correlation heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Internal Correlation Structure per Domain',
                 fontsize=14, fontweight='bold')

    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        cols = [c for c in SHARED_FEATURES + bm_extra + TARGETS if c in df.columns]
        corr = df[cols].corr()
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)
        ax.set_title(f'{name} (n={len(df)})', fontsize=12, fontweight='bold')
        # Annotate cells
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.values[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Pearson Correlation')

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_correlation_heatmaps.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 6: Domain gap summary (range overlap bar chart) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Domain Gap: Range Overlap & Coverage',
                 fontsize=14, fontweight='bold')

    # Overlap
    x = np.arange(len(all_cols))
    width = 0.25
    for i, (src, tgt) in enumerate(pairs):
        overlaps = []
        for col in all_cols:
            if col in datasets[src].columns and col in datasets[tgt].columns:
                overlaps.append(overlap_fraction(
                    datasets[src][col].values, datasets[tgt][col].values) * 100)
            else:
                overlaps.append(0)
        ax1.bar(x + i * width, overlaps, width, label=f'{src} vs {tgt}',
                alpha=0.8)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(all_cols)
    ax1.set_ylabel('Range Overlap (%)')
    ax1.set_title('Pairwise Range Overlap')
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis='y')

    # Coverage of BM target by each source
    x = np.arange(len(all_cols))
    width = 0.35
    for i, src in enumerate(['BCM', 'TBCM']):
        coverages = []
        for col in all_cols:
            if col in datasets[src].columns:
                coverages.append(coverage_fraction(
                    datasets[src][col], df_bm[col]) * 100 if col in df_bm.columns else 0)
            else:
                coverages.append(0)
        ax2.bar(x + i * width, coverages, width, label=f'{src} covers BM',
                color=colors[src], alpha=0.7)
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(all_cols)
    ax2.set_ylabel('Coverage of BM Range (%)')
    ax2.set_title('How Well Does Each Source Cover the BM Range?')
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim([0, 110])

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'domain_gap_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("DOMAIN COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
