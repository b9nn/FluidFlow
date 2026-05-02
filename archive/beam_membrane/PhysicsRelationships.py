"""
Side-by-side comparison of physical relationships:
  BCM (lumped-element) vs BM v1 (old) vs BM v2 (updated)

Shows how each model maps:
  - CT activation -> F0 (at different TA levels)
  - TA activation -> F0 (at different CT levels)
  - Subglottal pressure -> SPL
  - CT activation -> SPL
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


# ==================== LOAD ALL DATASETS ====================
print("Loading datasets...")

# Male BCM
df_bcm = pd.read_csv(os.path.join(project_root, 'MaleBCM.csv'))
df_bcm = df_bcm[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
# Subsample for speed (360k is too much for binned stats)
df_bcm = df_bcm.sample(n=20000, random_state=42).reset_index(drop=True)
print(f"  BCM (male): {len(df_bcm)} samples (subsampled)")

# BM v1
df_v1 = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df_v1.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df_v1 = df_v1[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
df_v1 = df_v1[(df_v1['F0'] >= 50) & (df_v1['F0'] <= 500)]
df_v1 = df_v1[(df_v1['SPL'] >= 40) & (df_v1['SPL'] <= 140)]
print(f"  BM v1: {len(df_v1)} samples")

# BM v2 (all)
df_v2 = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data_v2.csv'))
df_v2.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df_v2 = df_v2[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
df_v2 = df_v2[(df_v2['F0'] >= 50) & (df_v2['F0'] <= 500)]
print(f"  BM v2 (all): {len(df_v2)} samples")

# BM v2 phonating only
df_v2_phon = df_v2[df_v2['SPL'] >= 40].copy()
print(f"  BM v2 (phonating): {len(df_v2_phon)} samples")


# ==================== HELPER ====================
def plot_binned_relationship(ax, data_list, x_col, y_col, n_bins=8):
    """Plot mean y vs binned x for multiple datasets on one axis."""
    for df, color, label, marker, ls in data_list:
        if len(df) < 20:
            continue
        try:
            bins = pd.qcut(df[x_col], n_bins, duplicates='drop')
        except ValueError:
            continue
        grouped = df.groupby(bins)[y_col].agg(['mean', 'std', 'count'])
        # Only plot bins with enough samples
        grouped = grouped[grouped['count'] >= 5]
        midpoints = [(b.left + b.right) / 2 for b in grouped.index]
        ax.errorbar(midpoints, grouped['mean'], yerr=grouped['std'],
                    fmt=f'{marker}{ls}', color=color, label=label,
                    linewidth=2.5, markersize=8, capsize=4, alpha=0.85)


def plot_binned_by_third_var(ax, data_list, x_col, y_col, split_col, n_bins_x=6, n_splits=3):
    """Plot y vs x, split by terciles of a third variable."""
    split_labels = ['low', 'mid', 'high']
    alpha_levels = [0.6, 0.8, 1.0]
    lw_levels = [1.5, 2.0, 2.5]

    for df, base_color, dataset_label, base_marker, base_ls in data_list:
        if len(df) < 30:
            continue
        try:
            terciles = pd.qcut(df[split_col], n_splits, labels=split_labels, duplicates='drop')
        except ValueError:
            continue

        # Generate color shades
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(base_color)
        shades = [
            tuple(min(1, c + 0.3) for c in rgb),  # lighter
            rgb,                                     # original
            tuple(max(0, c - 0.2) for c in rgb),   # darker
        ]

        for i, sl in enumerate(split_labels):
            subset = df[terciles == sl]
            if len(subset) < 10:
                continue
            split_range = subset[split_col]
            try:
                bins = pd.qcut(subset[x_col], n_bins_x, duplicates='drop')
            except ValueError:
                continue
            grouped = subset.groupby(bins)[y_col].agg(['mean', 'std', 'count'])
            grouped = grouped[grouped['count'] >= 3]
            midpoints = [(b.left + b.right) / 2 for b in grouped.index]
            label = f'{dataset_label} ({sl} {split_col}: {split_range.min():.2f}-{split_range.max():.2f})'
            ax.errorbar(midpoints, grouped['mean'], yerr=grouped['std'],
                        fmt=f'{base_marker}{base_ls}', color=shades[i],
                        linewidth=lw_levels[i], markersize=6, capsize=3,
                        alpha=alpha_levels[i], label=label)


# ==================== DATASETS CONFIG ====================
# (df, color, label, marker, linestyle)
all_datasets = [
    (df_bcm, '#1976D2', 'BCM (male)', 'o', '-'),
    (df_v1, '#F44336', 'BM v1', 's', '--'),
    (df_v2_phon, '#FF9800', 'BM v2', '^', '-'),
]

bcm_only = [(df_bcm, '#1976D2', 'BCM', 'o', '-')]
bm_v1_only = [(df_v1, '#F44336', 'BM v1', 's', '--')]
bm_v2_only = [(df_v2_phon, '#FF9800', 'BM v2', '^', '-')]


# ==================================================================
# FIGURE 1: F0 vs CT at different TA levels — side by side
# ==================================================================
print("\nGenerating Figure 1: F0 vs CT...")
fig1, axes = plt.subplots(1, 3, figsize=(22, 7))
fig1.suptitle('F0 vs CT Activation at Different TA Levels', fontsize=15, fontweight='bold')

titles = ['BCM (Male Lumped-Element)', 'BM v1 (Old Constitutive)', 'BM v2 (Updated Constitutive)']
datasets_per_panel = [bcm_only, bm_v1_only, bm_v2_only]

for ax, title, dset in zip(axes, titles, datasets_per_panel):
    plot_binned_by_third_var(ax, dset, 'a_CT', 'F0', 'a_TA', n_bins_x=8, n_splits=3)
    ax.set_xlabel('a_CT', fontsize=12)
    ax.set_ylabel('Mean F0 (Hz)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

# Unify y-axis range
all_f0 = pd.concat([df_bcm['F0'], df_v1['F0'], df_v2_phon['F0']])
y_lo = max(50, all_f0.quantile(0.02) - 20)
y_hi = min(500, all_f0.quantile(0.98) + 20)
for ax in axes:
    ax.set_ylim([y_lo, y_hi])

plt.tight_layout()
fig1_path = os.path.join(script_dir, 'figs', 'physics_F0_vs_CT.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig1_path}")


# ==================================================================
# FIGURE 2: F0 vs TA at different CT levels — side by side
# ==================================================================
print("Generating Figure 2: F0 vs TA...")
fig2, axes = plt.subplots(1, 3, figsize=(22, 7))
fig2.suptitle('F0 vs TA Activation at Different CT Levels', fontsize=15, fontweight='bold')

for ax, title, dset in zip(axes, titles, datasets_per_panel):
    plot_binned_by_third_var(ax, dset, 'a_TA', 'F0', 'a_CT', n_bins_x=8, n_splits=3)
    ax.set_xlabel('a_TA', fontsize=12)
    ax.set_ylabel('Mean F0 (Hz)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([y_lo, y_hi])

plt.tight_layout()
fig2_path = os.path.join(script_dir, 'figs', 'physics_F0_vs_TA.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig2_path}")


# ==================================================================
# FIGURE 3: SPL vs PS at different TA levels — side by side
# ==================================================================
print("Generating Figure 3: SPL vs PS...")
fig3, axes = plt.subplots(1, 3, figsize=(22, 7))
fig3.suptitle('SPL vs Subglottal Pressure at Different TA Levels', fontsize=15, fontweight='bold')

for ax, title, dset in zip(axes, titles, datasets_per_panel):
    plot_binned_by_third_var(ax, dset, 'PS', 'SPL', 'a_TA', n_bins_x=8, n_splits=3)
    ax.set_xlabel('PS (Pa)', fontsize=12)
    ax.set_ylabel('Mean SPL (dB)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(script_dir, 'figs', 'physics_SPL_vs_PS.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig3_path}")


# ==================================================================
# FIGURE 4: All three models overlaid on same axes
# ==================================================================
print("Generating Figure 4: All models overlaid...")
fig4, axes = plt.subplots(2, 2, figsize=(18, 14))
fig4.suptitle('Physical Relationships: BCM vs BM v1 vs BM v2\n(How the models see the same physics)',
              fontsize=15, fontweight='bold')

# --- Panel 1: F0 vs CT (all models) ---
ax = axes[0, 0]
plot_binned_relationship(ax, all_datasets, 'a_CT', 'F0', n_bins=10)
ax.set_xlabel('a_CT', fontsize=12)
ax.set_ylabel('Mean F0 (Hz)', fontsize=12)
ax.set_title('F0 vs CT Activation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 2: F0 vs TA (all models) ---
ax = axes[0, 1]
plot_binned_relationship(ax, all_datasets, 'a_TA', 'F0', n_bins=10)
ax.set_xlabel('a_TA', fontsize=12)
ax.set_ylabel('Mean F0 (Hz)', fontsize=12)
ax.set_title('F0 vs TA Activation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 3: SPL vs PS (all models) ---
ax = axes[1, 0]
plot_binned_relationship(ax, all_datasets, 'PS', 'SPL', n_bins=10)
ax.set_xlabel('PS (Pa)', fontsize=12)
ax.set_ylabel('Mean SPL (dB)', fontsize=12)
ax.set_title('SPL vs Subglottal Pressure', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 4: SPL vs CT (all models) ---
ax = axes[1, 1]
plot_binned_relationship(ax, all_datasets, 'a_CT', 'SPL', n_bins=10)
ax.set_xlabel('a_CT', fontsize=12)
ax.set_ylabel('Mean SPL (dB)', fontsize=12)
ax.set_title('SPL vs CT Activation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4_path = os.path.join(script_dir, 'figs', 'physics_all_overlaid.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig4_path}")


# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "=" * 70)
print("PHYSICS RELATIONSHIP SUMMARY")
print("=" * 70)

# Compute overall correlations for each model
print(f"\n{'Relationship':<20} {'BCM':>10} {'BM v1':>10} {'BM v2':>10} {'Expected':>10}")
print("-" * 65)

for x_col, y_col, expected in [
    ('a_CT', 'F0', 'positive'),
    ('a_TA', 'F0', 'negative'),
    ('PS', 'SPL', 'positive'),
    ('a_CT', 'SPL', 'weak/none'),
]:
    corrs = []
    for df, label in [(df_bcm, 'BCM'), (df_v1, 'BM v1'), (df_v2_phon, 'BM v2')]:
        r, _ = sp_stats.spearmanr(df[x_col], df[y_col])
        corrs.append(r)
    print(f"{x_col:>5} -> {y_col:<5}       {corrs[0]:>+10.3f} {corrs[1]:>+10.3f} {corrs[2]:>+10.3f} {expected:>10}")

print("\n" + "=" * 70)
