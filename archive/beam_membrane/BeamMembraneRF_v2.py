"""
Random Forest trained on Beam-Membrane v2 data (updated constitutive relations).
Trains model, evaluates at varying data fractions, saves model for transfer learning.
Also runs CT diagnostic and structural transfer test against male BCM.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats as sp_stats
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# ==================== LOAD DATA ====================
print("=" * 70)
print("BEAM-MEMBRANE v2: RF Training + CT Diagnostic + Structural Transfer")
print("=" * 70)

print("\nLoading Beam-Membrane v2 data...")
df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data_v2.csv'))
df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
print(f"Samples after dropping NaN: {len(df)}")

# ==================== DATA CLEANING ====================
# v2 data has a bimodal SPL distribution:
#   ~47% sub-threshold phonation (SPL ~ -40 to -20 dB, low PS)
#   ~49% real phonation (SPL ~ 80-130 dB)
# F0 is valid across the board (50-500 Hz). We keep all data and add a
# phonation flag so the model can learn both regimes.

n_before = len(df)
df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
print(f"Removed {n_before - len(df)} rows with F0 outside 50-500 Hz")

# Flag sub-threshold phonation instead of removing it
# The gap in SPL between -20 and 60 dB is a natural boundary
df['phonating'] = (df['SPL'] >= 40).astype(int)
n_phonating = df['phonating'].sum()
n_subthreshold = len(df) - n_phonating
print(f"Phonation flag: {n_phonating} phonating, {n_subthreshold} sub-threshold")

# Light z-score filter only — remove true outliers, not the sub-threshold regime
if len(df) > 10:
    z_f0 = np.abs(sp_stats.zscore(df['F0']))
    n_before = len(df)
    df = df[z_f0 < 3]
    print(f"Removed {n_before - len(df)} F0 z-score outliers (|z| > 3)")

df = df.reset_index(drop=True)
print(f"Clean samples: {len(df)}")
print(df.describe())

X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

X_pool, X_test, Y_pool, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(f"Test set: {len(X_test)} samples")
print(f"Training pool: {len(X_pool)} samples")

# ==================== PART 1: RF TRAINING ====================
print("\n" + "=" * 70)
print("PART 1: RF Training at Varying Data Fractions")
print("=" * 70)

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
results = []

for frac in FRACTIONS:
    if frac < 1.0:
        X_train = X_pool.sample(frac=frac, random_state=42)
        Y_train = Y_pool.loc[X_train.index]
    else:
        X_train = X_pool
        Y_train = Y_pool

    n_train = len(X_train)
    print(f"\nTraining with {frac*100:.0f}% of data ({n_train} samples)")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42))
    rf.fit(X_train_scaled, Y_train)

    y_pred = rf.predict(X_test_scaled)
    r2_f0 = r2_score(Y_test.iloc[:, 0], y_pred[:, 0])
    r2_spl = r2_score(Y_test.iloc[:, 1], y_pred[:, 1])
    mae_f0 = mean_absolute_error(Y_test.iloc[:, 0], y_pred[:, 0])
    mae_spl = mean_absolute_error(Y_test.iloc[:, 1], y_pred[:, 1])

    print(f"  F0:  R2={r2_f0:.4f}  MAE={mae_f0:.2f} Hz")
    print(f"  SPL: R2={r2_spl:.4f}  MAE={mae_spl:.2f} dB")

    results.append({
        'frac': frac, 'n_train': n_train,
        'r2_f0': r2_f0, 'r2_spl': r2_spl,
        'mae_f0': mae_f0, 'mae_spl': mae_spl,
    })

    if frac == 1.0:
        os.makedirs(os.path.join(script_dir, 'models'), exist_ok=True)
        joblib.dump(rf, os.path.join(script_dir, 'models', 'RF_BeamMembrane_v2.pkl'))
        joblib.dump(x_scaler, os.path.join(script_dir, 'models', 'x_scaler_BeamMembrane_v2.pkl'))
        print("  Saved model to models/RF_BeamMembrane_v2.pkl")

print("\n\nRF SUMMARY:")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format='%.4f'))

# ==================== PART 2: CT DIAGNOSTIC ====================
print("\n" + "=" * 70)
print("PART 2: CT -> F0 Monotonicity Diagnostic (v2 data)")
print("=" * 70)

N_BINS = 5
df['a_TA_bin'] = pd.qcut(df['a_TA'], N_BINS, labels=False, duplicates='drop')
df['PS_bin'] = pd.qcut(df['PS'], N_BINS, labels=False, duplicates='drop')
df['a_CT_bin'] = pd.qcut(df['a_CT'], N_BINS, labels=False, duplicates='drop')

ct_edges = pd.qcut(df['a_CT'], N_BINS, duplicates='drop', retbins=True)[1]
ta_edges = pd.qcut(df['a_TA'], N_BINS, duplicates='drop', retbins=True)[1]
ps_edges = pd.qcut(df['PS'], N_BINS, duplicates='drop', retbins=True)[1]

monotonicity_results = {}
consistent_count = 0
broken_count = 0

for ta_bin in sorted(df['a_TA_bin'].unique()):
    for ps_bin in sorted(df['PS_bin'].unique()):
        mask = (df['a_TA_bin'] == ta_bin) & (df['PS_bin'] == ps_bin)
        subset = df[mask]
        if len(subset) < 3:
            continue

        grouped = subset.groupby('a_CT_bin')['F0'].mean()
        if len(grouped) < 2:
            continue

        f0_vals = grouped.values
        is_monotonic = all(f0_vals[i] <= f0_vals[i+1] for i in range(len(f0_vals)-1))
        n_violations = sum(1 for i in range(len(f0_vals)-1) if f0_vals[i] > f0_vals[i+1])

        monotonicity_results[(ta_bin, ps_bin)] = {
            'is_monotonic': is_monotonic,
            'n_violations': n_violations,
            'n_steps': len(f0_vals) - 1,
            'n_samples': len(subset),
            'f0_values': f0_vals,
        }

        if is_monotonic:
            consistent_count += 1
        else:
            broken_count += 1

n_total = consistent_count + broken_count
print(f"\nTotal (a_TA, PS) combinations: {n_total}")
print(f"  Monotonically consistent: {consistent_count} ({100*consistent_count/max(n_total,1):.1f}%)")
print(f"  Monotonicity broken:      {broken_count} ({100*broken_count/max(n_total,1):.1f}%)")

# ==================== PART 3: STRUCTURAL TRANSFER TEST ====================
print("\n" + "=" * 70)
print("PART 3: Structural Transfer Test (Male BCM -> BM v2)")
print("=" * 70)

print("\nLoading male BCM model...")
rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
scaler_X_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))

pred_male_all = rf_male.predict(scaler_X_male.transform(X_test))
pred_f0 = pred_male_all[:, 0]
pred_spl = pred_male_all[:, 1]
true_f0 = Y_test['F0'].values
true_spl = Y_test['SPL'].values

# Get phonation flag for test set
test_phonating = df.loc[Y_test.index, 'phonating'].values.astype(bool)

# All data
r2_raw_f0 = r2_score(true_f0, pred_f0)
r2_raw_spl = r2_score(true_spl, pred_spl)
spearman_f0, p_f0 = sp_stats.spearmanr(pred_f0, true_f0)
spearman_spl, p_spl = sp_stats.spearmanr(pred_spl, true_spl)
pearson_f0, _ = sp_stats.pearsonr(pred_f0, true_f0)
pearson_spl, _ = sp_stats.pearsonr(pred_spl, true_spl)

print(f"\nALL DATA ({len(X_test)} samples):")
print(f"{'Metric':<25} {'F0':>10} {'SPL':>10}")
print("-" * 50)
print(f"{'R2 (raw prediction)':<25} {r2_raw_f0:>10.3f} {r2_raw_spl:>10.3f}")
print(f"{'Pearson r':<25} {pearson_f0:>10.3f} {pearson_spl:>10.3f}")
print(f"{'Spearman rho':<25} {spearman_f0:>10.3f} {spearman_spl:>10.3f}")
print(f"{'Spearman p-value':<25} {p_f0:>10.2e} {p_spl:>10.2e}")

# Phonating-only subset
n_phon_test = test_phonating.sum()
if n_phon_test >= 10:
    sp_f0_phon, _ = sp_stats.spearmanr(pred_f0[test_phonating], true_f0[test_phonating])
    sp_spl_phon, _ = sp_stats.spearmanr(pred_spl[test_phonating], true_spl[test_phonating])
    r2_f0_phon = r2_score(true_f0[test_phonating], pred_f0[test_phonating])
    r2_spl_phon = r2_score(true_spl[test_phonating], pred_spl[test_phonating])
    print(f"\nPHONATING ONLY ({n_phon_test} samples):")
    print(f"{'Spearman rho':<25} {sp_f0_phon:>10.3f} {sp_spl_phon:>10.3f}")
    print(f"{'R2 (raw)':<25} {r2_f0_phon:>10.3f} {r2_spl_phon:>10.3f}")
else:
    sp_f0_phon, sp_spl_phon = np.nan, np.nan

# Sub-threshold subset
n_sub_test = (~test_phonating).sum()
if n_sub_test >= 10:
    sp_f0_sub, _ = sp_stats.spearmanr(pred_f0[~test_phonating], true_f0[~test_phonating])
    sp_spl_sub, _ = sp_stats.spearmanr(pred_spl[~test_phonating], true_spl[~test_phonating])
    print(f"\nSUB-THRESHOLD ONLY ({n_sub_test} samples):")
    print(f"{'Spearman rho':<25} {sp_f0_sub:>10.3f} {sp_spl_sub:>10.3f}")

# Linear calibration test at varying sizes
print("\nLinear calibration test:")
pred_male_pool = rf_male.predict(scaler_X_male.transform(X_pool))
pred_male = pred_male_all  # alias for calibration section

CALIB_SIZES = [5, 10, 20, 50, 100]
N_RUNS = 10

print(f"{'N_calib':>8} | {'F0 R2':>12} | {'SPL R2':>12}")
print("-" * 40)
for n_calib in CALIB_SIZES:
    if n_calib > len(X_pool):
        continue
    f0_r2s, spl_r2s = [], []
    for run in range(N_RUNS):
        rng = np.random.RandomState(42 + run)
        idx = rng.choice(len(X_pool), size=n_calib, replace=False)
        for i, (name, r2_list) in enumerate([('F0', f0_r2s), ('SPL', spl_r2s)]):
            reg = LinearRegression()
            reg.fit(pred_male_pool[idx, i].reshape(-1, 1),
                    Y_pool.iloc[idx][name].values)
            calibrated = reg.predict(pred_male[:, i].reshape(-1, 1))
            r2_list.append(r2_score(Y_test[name].values, calibrated))
    print(f"{n_calib:>8d} | {np.mean(f0_r2s):>6.3f}+/-{np.std(f0_r2s):.3f} | "
          f"{np.mean(spl_r2s):>6.3f}+/-{np.std(spl_r2s):.3f}")

# ==================== PLOTS ====================
print("\n\nGenerating plots...")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Beam-Membrane v2: RF Performance + CT Diagnostic + Transfer Test',
             fontsize=14, fontweight='bold')

# Panel 1: R2 vs training size
ax = axes[0, 0]
n_trains = [r['n_train'] for r in results]
ax.plot(n_trains, [r['r2_f0'] for r in results], 'o-', label='F0', linewidth=2)
ax.plot(n_trains, [r['r2_spl'] for r in results], 's-', label='SPL', linewidth=2)
ax.set_xlabel('Number of Training Samples')
ax.set_ylabel('R2')
ax.set_title('RF Performance vs Training Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# Panel 2: Pred vs actual (full model)
ax = axes[0, 1]
rf_full = joblib.load(os.path.join(script_dir, 'models', 'RF_BeamMembrane_v2.pkl'))
scaler_full = joblib.load(os.path.join(script_dir, 'models', 'x_scaler_BeamMembrane_v2.pkl'))
y_pred_full = rf_full.predict(scaler_full.transform(X_test))
ax.scatter(Y_test.iloc[:, 0], y_pred_full[:, 0], alpha=0.5, s=15, c='red', label='F0')
ax.scatter(Y_test.iloc[:, 1], y_pred_full[:, 1], alpha=0.5, s=15, c='green', label='SPL')
all_vals = np.concatenate([Y_test.values.flatten(), y_pred_full.flatten()])
lim = [min(all_vals) - 5, max(all_vals) + 5]
ax.plot(lim, lim, 'k--', alpha=0.5)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title(f'Pred vs Actual (full data)\nF0 R2={results[-1]["r2_f0"]:.3f}, SPL R2={results[-1]["r2_spl"]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: CT monotonicity heatmap
ax = axes[0, 2]
ta_bins_unique = sorted(set(k[0] for k in monotonicity_results.keys()))
ps_bins_unique = sorted(set(k[1] for k in monotonicity_results.keys()))

if ta_bins_unique and ps_bins_unique:
    violation_matrix = np.full((len(ta_bins_unique), len(ps_bins_unique)), np.nan)
    for (ta_bin, ps_bin), info in monotonicity_results.items():
        ta_idx = ta_bins_unique.index(ta_bin)
        ps_idx = ps_bins_unique.index(ps_bin)
        violation_matrix[ta_idx, ps_idx] = info['n_violations']

    im = ax.imshow(violation_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                   vmin=0, vmax=max(v['n_steps'] for v in monotonicity_results.values()))
    ax.set_xlabel('PS bin')
    ax.set_ylabel('a_TA bin')
    ax.set_title(f'CT->F0 Monotonicity Violations\n{consistent_count} OK / {broken_count} broken')
    for (ta_bin, ps_bin), info in monotonicity_results.items():
        ta_idx = ta_bins_unique.index(ta_bin)
        ps_idx = ps_bins_unique.index(ps_bin)
        color = 'white' if info['n_violations'] > 1 else 'black'
        ax.text(ps_idx, ta_idx, f"{info['n_violations']}/{info['n_steps']}",
                ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax)
else:
    ax.text(0.5, 0.5, 'Not enough data\nfor binning', ha='center', va='center',
            transform=ax.transAxes, fontsize=14)
    ax.set_title('CT->F0 Monotonicity')

# Panel 4: Male BCM predictions vs BM v2 truth (scatter)
ax = axes[1, 0]
ax.scatter(pred_spl, true_spl, alpha=0.4, s=15, c='green', label='SPL')
ax.scatter(pred_f0, true_f0, alpha=0.4, s=15, c='red', label='F0')
for p, t, color, name in [(pred_spl, true_spl, 'darkgreen', 'SPL'),
                            (pred_f0, true_f0, 'darkred', 'F0')]:
    if len(p) > 2:
        slope, intercept, _, _, _ = sp_stats.linregress(p, t)
        x_line = np.array([p.min(), p.max()])
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
                label=f'{name}: y={slope:.2f}x{intercept:+.1f}')
ax.set_xlabel('Male BCM Prediction')
ax.set_ylabel('BM v2 True Value')
ax.set_title(f'Structural Transfer\nSpearman: F0={spearman_f0:.3f}, SPL={spearman_spl:.3f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: Rank agreement
ax = axes[1, 1]
for pred, true, color, name in [(pred_spl, true_spl, 'green', 'SPL'),
                                  (pred_f0, true_f0, 'red', 'F0')]:
    pred_rank = sp_stats.rankdata(pred) / len(pred)
    true_rank = sp_stats.rankdata(true) / len(true)
    ax.scatter(pred_rank, true_rank, alpha=0.3, s=10, c=color, label=name)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('Male BCM Prediction (percentile)')
ax.set_ylabel('BM v2 Truth (percentile)')
ax.set_title('Rank Agreement')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 6: Summary comparison v1 vs v2
ax = axes[1, 2]
categories = ['v1 Spearman\nF0', 'v2 Spearman\nF0', 'v1 Spearman\nSPL', 'v2 Spearman\nSPL']
values = [-0.339, spearman_f0, 0.360, spearman_spl]
colors = ['#ffcccc', '#ff4444' if spearman_f0 > -0.339 else '#cc0000',
          '#ccffcc', '#44ff44' if spearman_spl > 0.360 else '#00cc00']

bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
for bar, val in zip(bars, values):
    y_pos = val + 0.02 if val >= 0 else val - 0.05
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Spearman rho')
ax.set_title('v1 vs v2: Structural Transfer Comparison')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([-0.5, 1.0])

plt.tight_layout()
fig_path = os.path.join(script_dir, 'figs', 'beam_membrane_v2_full.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {fig_path}")

# ==================== FINAL VERDICT ====================
print("\n" + "=" * 70)
print("VERDICT: v1 vs v2 Comparison")
print("=" * 70)
print(f"\nCT -> F0 monotonicity:")
print(f"  v1: 0% consistent (100% broken)")
print(f"  v2: {100*consistent_count/max(n_total,1):.1f}% consistent ({consistent_count}/{n_total})")
print(f"\nStructural transfer (Spearman rho):")
print(f"  {'':>6} {'v1':>8} {'v2':>8} {'Change':>8}")
print(f"  {'F0':>6} {-0.339:>8.3f} {spearman_f0:>8.3f} {spearman_f0 - (-0.339):>+8.3f}")
print(f"  {'SPL':>6} {0.360:>8.3f} {spearman_spl:>8.3f} {spearman_spl - 0.360:>+8.3f}")
print(f"\nRaw male BCM R2 on BM v2:")
print(f"  F0:  {r2_raw_f0:.3f}")
print(f"  SPL: {r2_raw_spl:.3f}")
print(f"\nBM v2 within-domain RF (full data):")
print(f"  F0:  R2={results[-1]['r2_f0']:.3f}  MAE={results[-1]['mae_f0']:.1f} Hz")
print(f"  SPL: R2={results[-1]['r2_spl']:.3f}  MAE={results[-1]['mae_spl']:.1f} dB")
print("=" * 70)
