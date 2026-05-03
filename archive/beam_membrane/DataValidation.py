"""
Data Validation & Evaluation for Beam-Membrane Simulation Data

Parses the raw simulation output, identifies:
  - Failed simulations (NaN rows)
  - Physically unreasonable values
  - Statistical outliers (IQR and z-score)
  - Correlations between inputs and failure/outlier patterns

Produces visual summaries of data quality.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, 'figs')
os.makedirs(fig_dir, exist_ok=True)

# ==================== LOAD RAW DATA ====================
print("=" * 70)
print("BEAM-MEMBRANE DATA VALIDATION")
print("=" * 70)

df_raw = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df_raw.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']

print(f"\nTotal rows in raw file: {len(df_raw)}")

# Identify constant columns
varying_inputs = ['a_CT', 'a_TA', 'PS']
constant_inputs = ['a_LCA', 'a_IA', 'a_PCA']
outputs = ['F0', 'SPL']

print(f"\nConstant inputs (not varied in simulation):")
for col in constant_inputs:
    print(f"  {col} = {df_raw[col].unique()[0]}")

print(f"\nVarying inputs: {varying_inputs}")
print(f"Outputs: {outputs}")

# ==================== 1. FAILED SIMULATIONS (NaN) ====================
print("\n" + "=" * 70)
print("1. FAILED SIMULATIONS")
print("=" * 70)

nan_mask = df_raw['F0'].isna() | df_raw['SPL'].isna()
df_failed = df_raw[nan_mask]
df_valid = df_raw[~nan_mask].copy()

n_total = len(df_raw)
n_failed = len(df_failed)
n_valid = len(df_valid)

print(f"  Failed (NaN):  {n_failed:>5} ({100*n_failed/n_total:.1f}%)")
print(f"  Valid:         {n_valid:>5} ({100*n_valid/n_total:.1f}%)")

# What input conditions cause failures?
print("\n  Input statistics for FAILED vs VALID simulations:")
for col in varying_inputs:
    f_mean = df_failed[col].mean()
    v_mean = df_valid[col].mean()
    print(f"    {col:>5}: failed mean={f_mean:.3f}, valid mean={v_mean:.3f}, "
          f"diff={f_mean - v_mean:+.3f}")

# ==================== 2. VALUE RANGES ====================
print("\n" + "=" * 70)
print("2. VALUE RANGES (valid data only)")
print("=" * 70)

print(f"\n  {'Column':>8}  {'Min':>10}  {'25%':>10}  {'Median':>10}  "
      f"{'75%':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
print("  " + "-" * 90)
for col in varying_inputs + outputs:
    s = df_valid[col]
    print(f"  {col:>8}  {s.min():>10.2f}  {s.quantile(0.25):>10.2f}  "
          f"{s.median():>10.2f}  {s.quantile(0.75):>10.2f}  "
          f"{s.max():>10.2f}  {s.mean():>10.2f}  {s.std():>10.2f}")

# ==================== 3. PHYSICAL REASONABLENESS ====================
print("\n" + "=" * 70)
print("3. PHYSICAL REASONABLENESS CHECKS")
print("=" * 70)

# Define physically expected ranges
phys_bounds = {
    'F0': (50, 500),      # Hz — vocal fold F0 should be in this range
    'SPL': (40, 140),     # dB — normal phonation range
    'PS': (300, 1000),    # Pa — subglottal pressure range from simulation
    'a_CT': (0, 1),       # activation levels
    'a_TA': (0, 1),
}

df_valid['phys_flag'] = ''

for col, (lo, hi) in phys_bounds.items():
    if col not in df_valid.columns:
        continue
    below = df_valid[col] < lo
    above = df_valid[col] > hi
    n_below = below.sum()
    n_above = above.sum()
    n_bad = n_below + n_above

    if n_bad > 0:
        df_valid.loc[below | above, 'phys_flag'] += f'{col}_OOB '
        print(f"  {col}: expected [{lo}, {hi}]")
        print(f"    Below: {n_below} rows (min={df_valid.loc[below, col].min():.2f})" if n_below else f"    Below: 0")
        print(f"    Above: {n_above} rows (max={df_valid.loc[above, col].max():.2f})" if n_above else f"    Above: 0")
    else:
        print(f"  {col}: ALL within [{lo}, {hi}]")

n_phys_bad = (df_valid['phys_flag'] != '').sum()
print(f"\n  Total rows with at least one physical flag: {n_phys_bad} ({100*n_phys_bad/n_valid:.1f}%)")

# ==================== 4. STATISTICAL OUTLIERS ====================
print("\n" + "=" * 70)
print("4. STATISTICAL OUTLIER DETECTION")
print("=" * 70)

# IQR method
print("\n  --- IQR Method (1.5x IQR) ---")
df_valid['iqr_outlier'] = False
for col in outputs:
    s = df_valid[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    df_valid.loc[mask, 'iqr_outlier'] = True
    print(f"  {col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"    Bounds: [{lo:.2f}, {hi:.2f}]")
    print(f"    Outliers: {mask.sum()} ({100*mask.sum()/n_valid:.1f}%)")

n_iqr = df_valid['iqr_outlier'].sum()
print(f"\n  Total IQR outliers (either output): {n_iqr} ({100*n_iqr/n_valid:.1f}%)")

# Z-score method
print("\n  --- Z-Score Method (|z| > 3) ---")
df_valid['zscore_outlier'] = False
for col in outputs:
    z = np.abs(stats.zscore(df_valid[col]))
    mask = z > 3
    df_valid.loc[mask, 'zscore_outlier'] = True
    print(f"  {col}: {mask.sum()} outliers ({100*mask.sum()/n_valid:.1f}%)")

n_zscore = df_valid['zscore_outlier'].sum()
print(f"\n  Total z-score outliers (either output): {n_zscore} ({100*n_zscore/n_valid:.1f}%)")

# Combined
df_valid['any_outlier'] = df_valid['iqr_outlier'] | df_valid['zscore_outlier'] | (df_valid['phys_flag'] != '')
n_any = df_valid['any_outlier'].sum()
print(f"\n  Total flagged (any method): {n_any} ({100*n_any/n_valid:.1f}%)")

# ==================== 5. CLEAN DATA SUMMARY ====================
print("\n" + "=" * 70)
print("5. CLEAN DATA SUMMARY")
print("=" * 70)

df_clean = df_valid[~df_valid['any_outlier']].copy()
print(f"  Raw:     {n_total} rows")
print(f"  - NaN:   {n_failed} removed")
print(f"  - Flagged: {n_any} removed")
print(f"  = Clean: {len(df_clean)} rows ({100*len(df_clean)/n_total:.1f}% of original)")

print(f"\n  Clean data ranges:")
for col in varying_inputs + outputs:
    s = df_clean[col]
    print(f"    {col:>5}: [{s.min():.2f}, {s.max():.2f}]  mean={s.mean():.2f}  std={s.std():.2f}")

# ==================== VISUALIZATIONS ====================
print("\n\nGenerating visualizations...")

# ---------- FIGURE 1: Data Overview ----------
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Beam-Membrane Data Validation Overview', fontsize=16, fontweight='bold')
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1a. Pie chart: valid vs failed
ax = fig.add_subplot(gs[0, 0])
ax.pie([n_valid, n_failed], labels=['Valid', 'Failed (NaN)'],
       autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
ax.set_title('Simulation Success Rate')

# 1b. Histogram of F0 (all valid data)
ax = fig.add_subplot(gs[0, 1])
ax.hist(df_valid['F0'], bins=60, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(df_valid['F0'].median(), color='red', linestyle='--', label=f"Median={df_valid['F0'].median():.1f}")
ax.axvline(phys_bounds['F0'][1], color='orange', linestyle=':', linewidth=2, label=f"Upper bound={phys_bounds['F0'][1]}")
ax.set_xlabel('F0 (Hz)')
ax.set_ylabel('Count')
ax.set_title('F0 Distribution')
ax.legend(fontsize=8)

# 1c. Histogram of SPL (all valid data)
ax = fig.add_subplot(gs[0, 2])
ax.hist(df_valid['SPL'], bins=60, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(df_valid['SPL'].median(), color='red', linestyle='--', label=f"Median={df_valid['SPL'].median():.1f}")
ax.axvline(phys_bounds['SPL'][0], color='orange', linestyle=':', linewidth=2, label=f"Lower bound={phys_bounds['SPL'][0]}")
ax.set_xlabel('SPL (dB)')
ax.set_ylabel('Count')
ax.set_title('SPL Distribution')
ax.legend(fontsize=8)

# 1d. a_CT vs a_TA colored by success/failure
ax = fig.add_subplot(gs[1, 0])
ax.scatter(df_valid['a_CT'], df_valid['a_TA'], s=5, alpha=0.3, c='green', label='Valid')
ax.scatter(df_failed['a_CT'], df_failed['a_TA'], s=5, alpha=0.3, c='red', label='Failed')
ax.set_xlabel('a_CT')
ax.set_ylabel('a_TA')
ax.set_title('Input Space: Valid vs Failed')
ax.legend(markerscale=4)

# 1e. a_CT vs a_TA colored by F0
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(df_valid['a_CT'], df_valid['a_TA'], c=df_valid['F0'],
                s=8, alpha=0.5, cmap='viridis')
plt.colorbar(sc, ax=ax, label='F0 (Hz)')
ax.set_xlabel('a_CT')
ax.set_ylabel('a_TA')
ax.set_title('F0 vs Muscle Activations')

# 1f. a_CT vs a_TA colored by SPL
ax = fig.add_subplot(gs[1, 2])
sc = ax.scatter(df_valid['a_CT'], df_valid['a_TA'], c=df_valid['SPL'],
                s=8, alpha=0.5, cmap='magma')
plt.colorbar(sc, ax=ax, label='SPL (dB)')
ax.set_xlabel('a_CT')
ax.set_ylabel('a_TA')
ax.set_title('SPL vs Muscle Activations')

# 1g. Box plots of outputs
ax = fig.add_subplot(gs[2, 0])
bp = ax.boxplot([df_valid['F0'], df_clean['F0']],
                tick_labels=['All Valid', 'Clean'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel('F0 (Hz)')
ax.set_title('F0: Before vs After Cleaning')

ax = fig.add_subplot(gs[2, 1])
bp = ax.boxplot([df_valid['SPL'], df_clean['SPL']],
                tick_labels=['All Valid', 'Clean'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightsalmon')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel('SPL (dB)')
ax.set_title('SPL: Before vs After Cleaning')

# 1h. Bar chart: data pipeline
ax = fig.add_subplot(gs[2, 2])
categories = ['Raw', 'Valid\n(no NaN)', 'Clean\n(no outliers)']
counts = [n_total, n_valid, len(df_clean)]
colors = ['#90CAF9', '#66BB6A', '#43A047']
bars = ax.bar(categories, counts, color=colors, edgecolor='black')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(count), ha='center', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_title('Data Pipeline')

plt.savefig(os.path.join(fig_dir, 'data_validation_overview.png'), dpi=150, bbox_inches='tight')
print(f"Saved: figs/data_validation_overview.png")

# ---------- FIGURE 2: Outlier Deep Dive ----------
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')

# 2a. F0 with outliers highlighted
ax = axes2[0, 0]
inliers = df_valid[~df_valid['any_outlier']]
outliers = df_valid[df_valid['any_outlier']]
ax.scatter(inliers['a_CT'], inliers['F0'], s=5, alpha=0.3, c='steelblue', label='Inlier')
ax.scatter(outliers['a_CT'], outliers['F0'], s=15, alpha=0.6, c='red', label='Outlier', marker='x')
ax.set_xlabel('a_CT')
ax.set_ylabel('F0 (Hz)')
ax.set_title('F0 vs a_CT')
ax.legend(markerscale=2)

# 2b. SPL with outliers highlighted
ax = axes2[0, 1]
ax.scatter(inliers['a_CT'], inliers['SPL'], s=5, alpha=0.3, c='coral', label='Inlier')
ax.scatter(outliers['a_CT'], outliers['SPL'], s=15, alpha=0.6, c='red', label='Outlier', marker='x')
ax.set_xlabel('a_CT')
ax.set_ylabel('SPL (dB)')
ax.set_title('SPL vs a_CT')
ax.legend(markerscale=2)

# 2c. F0 vs SPL
ax = axes2[0, 2]
ax.scatter(inliers['F0'], inliers['SPL'], s=5, alpha=0.3, c='steelblue', label='Inlier')
ax.scatter(outliers['F0'], outliers['SPL'], s=15, alpha=0.6, c='red', label='Outlier', marker='x')
ax.set_xlabel('F0 (Hz)')
ax.set_ylabel('SPL (dB)')
ax.set_title('F0 vs SPL')
ax.legend(markerscale=2)

# 2d. F0 vs a_TA
ax = axes2[1, 0]
ax.scatter(inliers['a_TA'], inliers['F0'], s=5, alpha=0.3, c='steelblue', label='Inlier')
ax.scatter(outliers['a_TA'], outliers['F0'], s=15, alpha=0.6, c='red', label='Outlier', marker='x')
ax.set_xlabel('a_TA')
ax.set_ylabel('F0 (Hz)')
ax.set_title('F0 vs a_TA')
ax.legend(markerscale=2)

# 2e. SPL vs PS
ax = axes2[1, 1]
ax.scatter(inliers['PS'], inliers['SPL'], s=5, alpha=0.3, c='coral', label='Inlier')
ax.scatter(outliers['PS'], outliers['SPL'], s=15, alpha=0.6, c='red', label='Outlier', marker='x')
ax.set_xlabel('PS (Pa)')
ax.set_ylabel('SPL (dB)')
ax.set_title('SPL vs Subglottal Pressure')
ax.legend(markerscale=2)

# 2f. Failure heatmap: a_CT vs a_TA binned
ax = axes2[1, 2]
ct_bins = np.linspace(0, 1, 21)
ta_bins = np.linspace(0, 1, 21)
fail_rate, _, _ = np.histogram2d(df_failed['a_CT'], df_failed['a_TA'], bins=[ct_bins, ta_bins])
total_rate, _, _ = np.histogram2d(df_raw['a_CT'], df_raw['a_TA'], bins=[ct_bins, ta_bins])
with np.errstate(divide='ignore', invalid='ignore'):
    pct = np.where(total_rate > 0, 100 * fail_rate / total_rate, 0)
im = ax.imshow(pct.T, origin='lower', extent=[0, 1, 0, 1],
               aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label='Failure Rate (%)')
ax.set_xlabel('a_CT')
ax.set_ylabel('a_TA')
ax.set_title('Simulation Failure Rate by Input Region')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'data_validation_outliers.png'), dpi=150, bbox_inches='tight')
print(f"Saved: figs/data_validation_outliers.png")

# ---------- FIGURE 3: Input distributions ----------
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
fig3.suptitle('Input Distributions (All 5000 Samples)', fontsize=14, fontweight='bold')

for ax, col, color in zip(axes3, varying_inputs, ['#2196F3', '#FF9800', '#4CAF50']):
    ax.hist(df_raw[col], bins=40, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title(f'{col}: [{df_raw[col].min():.1f}, {df_raw[col].max():.1f}]')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'data_validation_inputs.png'), dpi=150, bbox_inches='tight')
print(f"Saved: figs/data_validation_inputs.png")

plt.show()

# ==================== FINAL REPORT ====================
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)
print(f"""
  Raw samples:           {n_total}
  Failed simulations:    {n_failed} ({100*n_failed/n_total:.1f}%) -- NaN outputs
  Valid samples:         {n_valid}
  Physically flagged:    {n_phys_bad} -- outside expected ranges
  IQR outliers:          {n_iqr}
  Z-score outliers:      {n_zscore}
  Total flagged:         {n_any} ({100*n_any/n_valid:.1f}% of valid)
  Clean samples:         {len(df_clean)} ({100*len(df_clean)/n_total:.1f}% of original)

  Key observations:
    - Simulations fail when a_TA >> a_CT (low tension -> buckling)
    - Failed sim mean a_CT={df_failed['a_CT'].mean():.2f} vs valid mean a_CT={df_valid['a_CT'].mean():.2f}
    - SPL has some extreme negative values (bad simulations that didn't NaN)
    - F0 has a long right tail (high a_CT, low a_TA -> high tension -> high F0)
""")
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
