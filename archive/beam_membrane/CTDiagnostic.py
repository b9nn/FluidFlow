"""
CT Diagnostic: Where does BM's CT -> F0 monotonicity break?

The beam-membrane continuum model has a known bug: F0 doesn't always increase
with CT activation, contradicting both lumped-element models and clinical
observation. This script diagnoses exactly where the relationship breaks.

Output:
  - Beam_Membrane/figs/ct_diagnostic.png  (4-panel diagnostic figure)
  - Beam_Membrane/ct_consistent_indices.csv  (indices that pass monotonicity filter)
  - Console summary of consistent vs broken regions
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== LOAD & CLEAN DATA ====================
print("=" * 70)
print("CT DIAGNOSTIC: Where does BM's CT -> F0 monotonicity break?")
print("=" * 70)

print("\nLoading Beam-Membrane data...")
df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
print(f"Samples after dropping NaN: {len(df)}")

# Physical bounds
n_before = len(df)
df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
print(f"Removed {n_before - len(df)} physically unreasonable rows")

# Z-score filter
z_f0 = np.abs(stats.zscore(df['F0']))
z_spl = np.abs(stats.zscore(df['SPL']))
n_before = len(df)
df = df[(z_f0 < 3) & (z_spl < 3)]
print(f"Removed {n_before - len(df)} z-score outliers (|z| > 3)")
print(f"Clean samples: {len(df)}")

# Reset index so we have a clean integer index for CSV output
df = df.reset_index(drop=True)

# ==================== BIN THE INPUT SPACE ====================
N_BINS = 5

# Create bins for a_TA and PS
df['a_TA_bin'] = pd.qcut(df['a_TA'], N_BINS, labels=False, duplicates='drop')
df['PS_bin'] = pd.qcut(df['PS'], N_BINS, labels=False, duplicates='drop')
df['a_CT_bin'] = pd.qcut(df['a_CT'], N_BINS, labels=False, duplicates='drop')

# Get bin edges for labeling
ta_edges = pd.qcut(df['a_TA'], N_BINS, duplicates='drop', retbins=True)[1]
ps_edges = pd.qcut(df['PS'], N_BINS, duplicates='drop', retbins=True)[1]
ct_edges = pd.qcut(df['a_CT'], N_BINS, duplicates='drop', retbins=True)[1]

print(f"\nBinning: {N_BINS} bins each for a_TA, PS, a_CT")
print(f"  a_CT edges: {np.round(ct_edges, 3)}")
print(f"  a_TA edges: {np.round(ta_edges, 3)}")
print(f"  PS edges:   {np.round(ps_edges, 1)}")

# ==================== CHECK MONOTONICITY ====================
print("\n" + "=" * 70)
print("MONOTONICITY CHECK: F0 should increase with CT")
print("=" * 70)

monotonicity_results = {}
consistent_indices = []
broken_indices = []

# For each (a_TA_bin, PS_bin), check if mean F0 increases with CT bin
for ta_bin in sorted(df['a_TA_bin'].unique()):
    for ps_bin in sorted(df['PS_bin'].unique()):
        mask = (df['a_TA_bin'] == ta_bin) & (df['PS_bin'] == ps_bin)
        subset = df[mask]

        if len(subset) < 5:
            continue

        # Compute mean F0 per CT bin
        grouped = subset.groupby('a_CT_bin')['F0'].mean()
        if len(grouped) < 2:
            continue

        # Check monotonicity: each step should be non-decreasing
        f0_vals = grouped.values
        ct_bins_sorted = grouped.index.values
        is_monotonic = all(f0_vals[i] <= f0_vals[i+1] for i in range(len(f0_vals)-1))

        # Count violations
        n_violations = sum(1 for i in range(len(f0_vals)-1) if f0_vals[i] > f0_vals[i+1])

        monotonicity_results[(ta_bin, ps_bin)] = {
            'is_monotonic': is_monotonic,
            'n_violations': n_violations,
            'n_steps': len(f0_vals) - 1,
            'n_samples': len(subset),
            'f0_values': f0_vals,
            'ct_bins': ct_bins_sorted,
        }

        if is_monotonic:
            consistent_indices.extend(subset.index.tolist())
        else:
            broken_indices.extend(subset.index.tolist())

n_consistent_combos = sum(1 for v in monotonicity_results.values() if v['is_monotonic'])
n_broken_combos = sum(1 for v in monotonicity_results.values() if not v['is_monotonic'])
n_total_combos = len(monotonicity_results)

# Deduplicate indices (a sample might appear in overlapping bins if qcut has ties)
consistent_indices = sorted(set(consistent_indices))
broken_indices = sorted(set(broken_indices))

print(f"\nTotal (a_TA, PS) combinations with enough data: {n_total_combos}")
print(f"  Monotonically consistent: {n_consistent_combos} ({100*n_consistent_combos/n_total_combos:.1f}%)")
print(f"  Monotonicity broken:      {n_broken_combos} ({100*n_broken_combos/n_total_combos:.1f}%)")
print(f"\nSamples in consistent regions: {len(consistent_indices)} ({100*len(consistent_indices)/len(df):.1f}%)")
print(f"Samples in broken regions:     {len(broken_indices)} ({100*len(broken_indices)/len(df):.1f}%)")

# Some samples might be in neither if they were in combos with < 5 samples
n_unclassified = len(df) - len(consistent_indices) - len(broken_indices)
# Handle overlap: samples could appear in both if bins overlap
overlap = set(consistent_indices) & set(broken_indices)
if overlap:
    print(f"Samples in overlapping bins (classified as broken): {len(overlap)}")
    consistent_indices = sorted(set(consistent_indices) - overlap)
    print(f"  -> Adjusted consistent: {len(consistent_indices)}")

if n_unclassified > 0:
    print(f"Samples not in any combo with >= 5 points: {n_unclassified}")

# Save consistent indices to CSV
consistent_df = pd.DataFrame({'index': consistent_indices})
csv_path = os.path.join(script_dir, 'ct_consistent_indices.csv')
consistent_df.to_csv(csv_path, index=False)
print(f"\nSaved {len(consistent_indices)} consistent indices to {csv_path}")

# ==================== DETAILED BREAKDOWN ====================
print("\n" + "-" * 60)
print("DETAILED BREAKDOWN BY (a_TA bin, PS bin):")
print("-" * 60)
for (ta_bin, ps_bin), info in sorted(monotonicity_results.items()):
    status = "OK" if info['is_monotonic'] else f"BROKEN ({info['n_violations']} violations)"
    f0_str = " -> ".join(f"{v:.1f}" for v in info['f0_values'])
    print(f"  a_TA={ta_bin}, PS={ps_bin}: {status:>20}  |  n={info['n_samples']:>4}  |  F0: {f0_str}")

# ==================== PLOTS ====================
print("\n\nGenerating diagnostic plots...")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('CT Diagnostic: Where Does BM\'s CT->F0 Monotonicity Break?',
             fontsize=14, fontweight='bold')

# --- Panel 1: Heatmap of monotonicity violations ---
ax = axes[0, 0]
ta_bins_unique = sorted(df['a_TA_bin'].unique())
ps_bins_unique = sorted(df['PS_bin'].unique())
violation_matrix = np.full((len(ta_bins_unique), len(ps_bins_unique)), np.nan)

for (ta_bin, ps_bin), info in monotonicity_results.items():
    ta_idx = ta_bins_unique.index(ta_bin)
    ps_idx = ps_bins_unique.index(ps_bin)
    violation_matrix[ta_idx, ps_idx] = info['n_violations']

im = ax.imshow(violation_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
               vmin=0, vmax=max(v['n_steps'] for v in monotonicity_results.values()))
ax.set_xlabel('PS bin')
ax.set_ylabel('a_TA bin')
ax.set_title('Monotonicity Violations by (a_TA, PS) Bin')
ax.set_xticks(range(len(ps_bins_unique)))
ax.set_xticklabels([f"{ps_edges[i]:.0f}-{ps_edges[i+1]:.0f}"
                     for i in range(len(ps_bins_unique))], rotation=45, fontsize=8)
ax.set_yticks(range(len(ta_bins_unique)))
ax.set_yticklabels([f"{ta_edges[i]:.2f}-{ta_edges[i+1]:.2f}"
                     for i in range(len(ta_bins_unique))], fontsize=8)

# Annotate cells
for (ta_bin, ps_bin), info in monotonicity_results.items():
    ta_idx = ta_bins_unique.index(ta_bin)
    ps_idx = ps_bins_unique.index(ps_bin)
    color = 'white' if info['n_violations'] > 1 else 'black'
    ax.text(ps_idx, ta_idx, f"{info['n_violations']}/{info['n_steps']}",
            ha='center', va='center', fontsize=9, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of violations')

# --- Panel 2: F0 vs CT for representative slices ---
ax = axes[0, 1]

# Pick a few consistent and broken examples
consistent_examples = [(k, v) for k, v in monotonicity_results.items() if v['is_monotonic']]
broken_examples = [(k, v) for k, v in monotonicity_results.items() if not v['is_monotonic']]

# Sort by sample count to get good examples
consistent_examples.sort(key=lambda x: x[1]['n_samples'], reverse=True)
broken_examples.sort(key=lambda x: x[1]['n_violations'], reverse=True)

n_examples = min(3, len(consistent_examples))
for i in range(n_examples):
    (ta_bin, ps_bin), info = consistent_examples[i]
    ct_mids = [(ct_edges[b] + ct_edges[b+1])/2 for b in info['ct_bins']]
    ax.plot(ct_mids, info['f0_values'], 'g-o', alpha=0.7,
            label=f'OK: TA={ta_bin}, PS={ps_bin} (n={info["n_samples"]})')

n_examples = min(3, len(broken_examples))
for i in range(n_examples):
    (ta_bin, ps_bin), info = broken_examples[i]
    ct_mids = [(ct_edges[b] + ct_edges[b+1])/2 for b in info['ct_bins']]
    ax.plot(ct_mids, info['f0_values'], 'r--s', alpha=0.7,
            label=f'BROKEN: TA={ta_bin}, PS={ps_bin} (n={info["n_samples"]})')

ax.set_xlabel('a_CT (bin midpoint)')
ax.set_ylabel('Mean F0 (Hz)')
ax.set_title('F0 vs CT: Consistent (green) vs Broken (red) Slices')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# --- Panel 3: Input space map coloring consistent vs broken ---
ax = axes[1, 0]

# Color each sample by whether it's in a consistent or broken region
colors = np.array(['gray'] * len(df))
for idx in consistent_indices:
    colors[idx] = 'green'
for idx in broken_indices:
    colors[idx] = 'red'

scatter_consistent = ax.scatter(
    df.loc[consistent_indices, 'a_TA'],
    df.loc[consistent_indices, 'PS'],
    c='green', alpha=0.3, s=10, label=f'Consistent (n={len(consistent_indices)})')
scatter_broken = ax.scatter(
    df.loc[broken_indices, 'a_TA'],
    df.loc[broken_indices, 'PS'],
    c='red', alpha=0.3, s=10, label=f'Broken (n={len(broken_indices)})')
ax.set_xlabel('a_TA')
ax.set_ylabel('PS (Pa)')
ax.set_title('Input Space: Consistent (green) vs Broken (red)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 4: Sample counts ---
ax = axes[1, 1]

categories = ['Total Clean', 'Consistent\n(pass filter)', 'Broken\n(fail filter)', 'Unclassified']
counts = [len(df), len(consistent_indices), len(broken_indices),
          len(df) - len(consistent_indices) - len(broken_indices)]
bar_colors = ['steelblue', 'green', 'red', 'gray']

bars = ax.bar(categories, counts, color=bar_colors, edgecolor='black', alpha=0.8)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{count}\n({100*count/len(df):.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Number of Samples')
ax.set_title('Sample Count: How Many Survive Monotonicity Filter?')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = os.path.join(script_dir, 'figs', 'ct_diagnostic.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved diagnostic plot to {fig_path}")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total clean BM samples:          {len(df)}")
print(f"Consistent (CT->F0 monotonic):   {len(consistent_indices)} ({100*len(consistent_indices)/len(df):.1f}%)")
print(f"Broken (CT->F0 non-monotonic):   {len(broken_indices)} ({100*len(broken_indices)/len(df):.1f}%)")
print(f"Consistent indices saved to:     {csv_path}")
print(f"Diagnostic figure saved to:      {fig_path}")
print("\nThis consistent subset can be used for F0 transfer learning")
print("where physics is expected to be sound.")
print("=" * 70)
