"""
Domain Gap Analysis: Male BCM vs Beam-Membrane
Quantifies exactly how the two physics models differ in their outputs
for the same input parameters.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# ==================== LOAD DATA ====================
print("Loading datasets...")

# Male BCM
df_male = pd.read_csv(os.path.join(project_root, 'MaleBCM.csv')).dropna()
df_male = df_male[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']]

# Beam-Membrane
df_bm = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df_bm.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df_bm = df_bm[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()

# Clean BM data (same pipeline as other scripts)
df_bm = df_bm[(df_bm['F0'] >= 50) & (df_bm['F0'] <= 500)]
df_bm = df_bm[(df_bm['SPL'] >= 40) & (df_bm['SPL'] <= 140)]
z_f0 = np.abs(stats.zscore(df_bm['F0']))
z_spl = np.abs(stats.zscore(df_bm['SPL']))
df_bm = df_bm[(z_f0 < 3) & (z_spl < 3)]

print(f"Male BCM samples: {len(df_male)}")
print(f"Beam-Membrane samples: {len(df_bm)}")

# Load male model for direct comparison
rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
scaler_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))


# ==================== MATCHED COMPARISON ====================
# For each BM data point, get what the male model WOULD predict
# for the exact same inputs — this isolates the physics gap

X_bm = df_bm[['a_CT', 'a_TA', 'PS']].values
Y_bm = df_bm[['F0', 'SPL']].values

male_pred = rf_male.predict(scaler_male.transform(X_bm))

diff_f0 = male_pred[:, 0] - Y_bm[:, 0]
diff_spl = male_pred[:, 1] - Y_bm[:, 1]

print("\n" + "=" * 70)
print("POINT-BY-POINT COMPARISON")
print("Same inputs → different outputs")
print("=" * 70)

print(f"\nF0 (fundamental frequency):")
print(f"  Male BCM predicts: mean={male_pred[:,0].mean():.1f} Hz, "
      f"std={male_pred[:,0].std():.1f}")
print(f"  Beam-Membrane actual: mean={Y_bm[:,0].mean():.1f} Hz, "
      f"std={Y_bm[:,0].std():.1f}")
print(f"  Difference (male - BM): mean={diff_f0.mean():+.1f} Hz, "
      f"std={diff_f0.std():.1f}")
print(f"  Male overshoots on {(diff_f0 > 0).sum()}/{len(diff_f0)} "
      f"points ({(diff_f0 > 0).mean()*100:.1f}%)")

print(f"\nSPL (sound pressure level):")
print(f"  Male BCM predicts: mean={male_pred[:,1].mean():.1f} dB, "
      f"std={male_pred[:,1].std():.1f}")
print(f"  Beam-Membrane actual: mean={Y_bm[:,1].mean():.1f} dB, "
      f"std={Y_bm[:,1].std():.1f}")
print(f"  Difference (male - BM): mean={diff_spl.mean():+.1f} dB, "
      f"std={diff_spl.std():.1f}")


# ==================== BINNED ANALYSIS ====================
# Group by input ranges and compare outputs within each bin
print("\n\n" + "=" * 70)
print("BINNED COMPARISON: same input region → what does each model output?")
print("=" * 70)

# Filter male data to overlapping PS range
df_male_overlap = df_male[
    (df_male['PS'] >= df_bm['PS'].min()) &
    (df_male['PS'] <= df_bm['PS'].max())
].copy()
print(f"\nMale samples in overlapping PS range "
      f"({df_bm['PS'].min():.0f}-{df_bm['PS'].max():.0f} Pa): "
      f"{len(df_male_overlap)}")

# Bin by a_CT
print("\n--- F0 by a_CT bin (within overlapping PS range) ---")
print(f"{'a_CT bin':<15} {'Male F0':>10} {'BM F0':>10} {'Diff':>10}")
print("-" * 50)

ct_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for lo, hi in ct_bins:
    male_in_bin = df_male_overlap[
        (df_male_overlap['a_CT'] >= lo) & (df_male_overlap['a_CT'] < hi)
    ]
    bm_in_bin = df_bm[
        (df_bm['a_CT'] >= lo) & (df_bm['a_CT'] < hi)
    ]
    if len(male_in_bin) > 0 and len(bm_in_bin) > 0:
        m_f0 = male_in_bin['F0'].mean()
        b_f0 = bm_in_bin['F0'].mean()
        print(f"[{lo:.1f}, {hi:.1f})     {m_f0:>10.1f} {b_f0:>10.1f} "
              f"{m_f0 - b_f0:>+10.1f}")

# Bin by a_TA
print("\n--- F0 by a_TA bin (within overlapping PS range) ---")
print(f"{'a_TA bin':<15} {'Male F0':>10} {'BM F0':>10} {'Diff':>10}")
print("-" * 50)

for lo, hi in ct_bins:
    male_in_bin = df_male_overlap[
        (df_male_overlap['a_TA'] >= lo) & (df_male_overlap['a_TA'] < hi)
    ]
    bm_in_bin = df_bm[
        (df_bm['a_TA'] >= lo) & (df_bm['a_TA'] < hi)
    ]
    if len(male_in_bin) > 0 and len(bm_in_bin) > 0:
        m_f0 = male_in_bin['F0'].mean()
        b_f0 = bm_in_bin['F0'].mean()
        print(f"[{lo:.1f}, {hi:.1f})     {m_f0:>10.1f} {b_f0:>10.1f} "
              f"{m_f0 - b_f0:>+10.1f}")

# Bin by PS
print("\n--- F0 by PS bin ---")
print(f"{'PS bin (Pa)':<15} {'Male F0':>10} {'BM F0':>10} {'Diff':>10}")
print("-" * 50)

ps_bins = [(300, 500), (500, 700), (700, 900), (900, 1000)]
for lo, hi in ps_bins:
    male_in_bin = df_male[
        (df_male['PS'] >= lo) & (df_male['PS'] < hi)
    ]
    bm_in_bin = df_bm[
        (df_bm['PS'] >= lo) & (df_bm['PS'] < hi)
    ]
    if len(male_in_bin) > 0 and len(bm_in_bin) > 0:
        m_f0 = male_in_bin['F0'].mean()
        b_f0 = bm_in_bin['F0'].mean()
        print(f"[{lo}, {hi})   {m_f0:>10.1f} {b_f0:>10.1f} "
              f"{m_f0 - b_f0:>+10.1f}")

# Same for SPL
print("\n--- SPL by PS bin ---")
print(f"{'PS bin (Pa)':<15} {'Male SPL':>10} {'BM SPL':>10} {'Diff':>10}")
print("-" * 50)

for lo, hi in ps_bins:
    male_in_bin = df_male[
        (df_male['PS'] >= lo) & (df_male['PS'] < hi)
    ]
    bm_in_bin = df_bm[
        (df_bm['PS'] >= lo) & (df_bm['PS'] < hi)
    ]
    if len(male_in_bin) > 0 and len(bm_in_bin) > 0:
        m_spl = male_in_bin['SPL'].mean()
        b_spl = bm_in_bin['SPL'].mean()
        print(f"[{lo}, {hi})   {m_spl:>10.1f} {b_spl:>10.1f} "
              f"{m_spl - b_spl:>+10.1f}")


# ==================== CORRELATION ANALYSIS ====================
print("\n\n" + "=" * 70)
print("CORRELATION: are the models at least correlated?")
print("=" * 70)

corr_f0 = np.corrcoef(male_pred[:, 0], Y_bm[:, 0])[0, 1]
corr_spl = np.corrcoef(male_pred[:, 1], Y_bm[:, 1])[0, 1]

print(f"  F0:  Pearson r = {corr_f0:.3f}")
print(f"  SPL: Pearson r = {corr_spl:.3f}")
print()
if corr_f0 > 0.5:
    print("  F0 is positively correlated — the models agree on DIRECTION")
    print("  (when one goes up, the other goes up) but disagree on MAGNITUDE.")
    print("  A simple linear calibration (scale + offset) could help.")
else:
    print("  F0 correlation is weak — the models disagree on both")
    print("  direction and magnitude for many input combinations.")

# Linear calibration: what if we just rescale male predictions?
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

reg_f0 = LinearRegression()
reg_f0.fit(male_pred[:, 0].reshape(-1, 1), Y_bm[:, 0])
calibrated_f0 = reg_f0.predict(male_pred[:, 0].reshape(-1, 1))

reg_spl = LinearRegression()
reg_spl.fit(male_pred[:, 1].reshape(-1, 1), Y_bm[:, 1])
calibrated_spl = reg_spl.predict(male_pred[:, 1].reshape(-1, 1))

print(f"\n  If we linearly calibrate male predictions (y = a*pred + b):")
print(f"  F0:  y = {reg_f0.coef_[0]:.2f} * male_pred + {reg_f0.intercept_:.1f}")
print(f"       R2 after calibration: {r2_score(Y_bm[:,0], calibrated_f0):.3f}")
print(f"  SPL: y = {reg_spl.coef_[0]:.2f} * male_pred + {reg_spl.intercept_:.1f}")
print(f"       R2 after calibration: {r2_score(Y_bm[:,1], calibrated_spl):.3f}")


# ==================== VISUALIZATIONS ====================
print("\n\nGenerating plots...")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Domain Gap: Male BCM vs Beam-Membrane Model',
             fontsize=14, fontweight='bold')

# --- Plot 1: F0 distributions ---
ax = axes[0, 0]
ax.hist(df_male_overlap['F0'], bins=50, alpha=0.6, label='Male BCM',
        color='blue', density=True)
ax.hist(df_bm['F0'], bins=50, alpha=0.6, label='Beam-Membrane',
        color='orange', density=True)
ax.set_xlabel('F0 (Hz)')
ax.set_ylabel('Density')
ax.set_title('F0 Output Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 2: SPL distributions ---
ax = axes[0, 1]
ax.hist(df_male_overlap['SPL'], bins=50, alpha=0.6, label='Male BCM',
        color='blue', density=True)
ax.hist(df_bm['SPL'], bins=50, alpha=0.6, label='Beam-Membrane',
        color='orange', density=True)
ax.set_xlabel('SPL (dB)')
ax.set_ylabel('Density')
ax.set_title('SPL Output Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 3: Male prediction vs BM actual (F0) ---
ax = axes[0, 2]
ax.scatter(Y_bm[:, 0], male_pred[:, 0], alpha=0.15, s=5, color='red')
lim = [50, 400]
ax.plot(lim, lim, 'k--', linewidth=2, label='Perfect agreement')
ax.set_xlabel('Beam-Membrane F0 (Hz)')
ax.set_ylabel('Male BCM Prediction F0 (Hz)')
ax.set_title(f'Same Inputs, Different Outputs (F0)\nr = {corr_f0:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(lim)
ax.set_ylim(lim)

# --- Plot 4: Difference histogram (F0) ---
ax = axes[1, 0]
ax.hist(diff_f0, bins=60, color='red', alpha=0.7, edgecolor='darkred')
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.axvline(x=diff_f0.mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {diff_f0.mean():+.1f} Hz')
ax.set_xlabel('Male Prediction - BM Actual (Hz)')
ax.set_ylabel('Count')
ax.set_title('F0 Error Distribution\n(positive = male overshoots)')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 5: Difference histogram (SPL) ---
ax = axes[1, 1]
ax.hist(diff_spl, bins=60, color='purple', alpha=0.7, edgecolor='darkviolet')
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.axvline(x=diff_spl.mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {diff_spl.mean():+.1f} dB')
ax.set_xlabel('Male Prediction - BM Actual (dB)')
ax.set_ylabel('Count')
ax.set_title('SPL Error Distribution\n(positive = male overshoots)')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 6: F0 vs a_CT for both models ---
ax = axes[1, 2]
# Subsample male for visibility
male_sub = df_male_overlap.sample(n=min(3000, len(df_male_overlap)),
                                   random_state=42)
ax.scatter(male_sub['a_CT'], male_sub['F0'], alpha=0.15, s=5,
           color='blue', label='Male BCM')
ax.scatter(df_bm['a_CT'], df_bm['F0'], alpha=0.15, s=5,
           color='orange', label='Beam-Membrane')
ax.set_xlabel('a_CT (cricothyroid activation)')
ax.set_ylabel('F0 (Hz)')
ax.set_title('F0 vs a_CT: Two Models Compared')
ax.legend(markerscale=5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(script_dir, 'figs', 'domain_gap_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"Saved to {path}")

# --- Second figure: input-conditioned comparison ---
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('How Each Input Affects the Output Gap',
              fontsize=14, fontweight='bold')

for col_idx, input_var in enumerate(['a_CT', 'a_TA', 'PS']):
    # F0 row
    ax = axes2[0, col_idx]
    ax.scatter(df_bm[input_var], Y_bm[:, 0], alpha=0.15, s=5,
               color='orange', label='BM actual')
    ax.scatter(df_bm[input_var], male_pred[:, 0], alpha=0.15, s=5,
               color='blue', label='Male pred')
    ax.set_xlabel(input_var)
    ax.set_ylabel('F0 (Hz)')
    ax.set_title(f'F0 vs {input_var}')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # SPL row
    ax = axes2[1, col_idx]
    ax.scatter(df_bm[input_var], Y_bm[:, 1], alpha=0.15, s=5,
               color='orange', label='BM actual')
    ax.scatter(df_bm[input_var], male_pred[:, 1], alpha=0.15, s=5,
               color='blue', label='Male pred')
    ax.set_xlabel(input_var)
    ax.set_ylabel('SPL (dB)')
    ax.set_title(f'SPL vs {input_var}')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path2 = os.path.join(script_dir, 'figs', 'domain_gap_by_input.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Saved to {path2}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
