"""
Step 1: Does the male BCM model capture the right STRUCTURE even though
the absolute values are wrong?

If Spearman rank correlation is high but R2 is -5, the model gets the
relative ordering right — "sample A should have higher SPL than sample B" —
but the scale/offset is wrong. That's fixable with a trivial 2-parameter
linear calibration: y_true = a * y_pred + b.

Tests:
  1. Rank correlation (Spearman) of raw male BCM predictions vs BM truth
  2. Linear calibration with varying calibration set sizes (5, 10, 20, 50, 100)
  3. Compare calibrated R2 against full transfer learning pipeline

Output:
  - Beam_Membrane/figs/structural_transfer_test.png
  - Console summary
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# ==================== LOAD DATA ====================
print("=" * 70)
print("STRUCTURAL TRANSFER TEST")
print("Does male BCM capture the right structure for BM?")
print("=" * 70)

# Load source model
print("\nLoading male BCM model...")
rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
scaler_X_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))

# Load BM data
print("Loading Beam-Membrane data...")
df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()

# Clean
df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
z_f0 = np.abs(sp_stats.zscore(df['F0']))
z_spl = np.abs(sp_stats.zscore(df['SPL']))
df = df[(z_f0 < 3) & (z_spl < 3)].reset_index(drop=True)
print(f"Clean BM samples: {len(df)}")

X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# Fixed test set
X_pool, X_test, Y_pool, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(f"Test set: {len(X_test)}, Calibration pool: {len(X_pool)}")

# ==================== RAW PREDICTIONS ====================
pred_male = rf_male.predict(scaler_X_male.transform(X_test))
pred_f0 = pred_male[:, 0]
pred_spl = pred_male[:, 1]
true_f0 = Y_test['F0'].values
true_spl = Y_test['SPL'].values

# ==================== TEST 1: RANK CORRELATION ====================
print("\n" + "=" * 70)
print("TEST 1: RANK CORRELATION (Spearman)")
print("=" * 70)
print("If high, the model gets relative ordering right despite wrong scale.\n")

spearman_f0, p_f0 = sp_stats.spearmanr(pred_f0, true_f0)
spearman_spl, p_spl = sp_stats.spearmanr(pred_spl, true_spl)
pearson_f0, pp_f0 = sp_stats.pearsonr(pred_f0, true_f0)
pearson_spl, pp_spl = sp_stats.pearsonr(pred_spl, true_spl)
r2_raw_f0 = r2_score(true_f0, pred_f0)
r2_raw_spl = r2_score(true_spl, pred_spl)

print(f"{'Metric':<25} {'F0':>10} {'SPL':>10}")
print("-" * 50)
print(f"{'R2 (raw prediction)':<25} {r2_raw_f0:>10.3f} {r2_raw_spl:>10.3f}")
print(f"{'Pearson r':<25} {pearson_f0:>10.3f} {pearson_spl:>10.3f}")
print(f"{'Spearman rho':<25} {spearman_f0:>10.3f} {spearman_spl:>10.3f}")
print(f"{'Spearman p-value':<25} {p_f0:>10.2e} {p_spl:>10.2e}")

print(f"\nInterpretation:")
for name, spear, r2_raw in [('F0', spearman_f0, r2_raw_f0),
                              ('SPL', spearman_spl, r2_raw_spl)]:
    if abs(spear) > 0.7:
        print(f"  {name}: Strong rank correlation ({spear:.3f}) despite R2={r2_raw:.3f}")
        print(f"       -> Structure transfers! Linear calibration should recover it.")
    elif abs(spear) > 0.4:
        print(f"  {name}: Moderate rank correlation ({spear:.3f}), R2={r2_raw:.3f}")
        print(f"       -> Partial structure transfer. Calibration will help but not fully fix.")
    else:
        print(f"  {name}: Weak rank correlation ({spear:.3f}), R2={r2_raw:.3f}")
        print(f"       -> Structure does NOT transfer. Need full retraining.")

# ==================== TEST 2: LINEAR CALIBRATION ====================
print("\n" + "=" * 70)
print("TEST 2: LINEAR CALIBRATION (y_true = a * y_pred + b)")
print("How many calibration samples do we need?")
print("=" * 70)

# Also get predictions on the pool for calibration
pred_male_pool = rf_male.predict(scaler_X_male.transform(X_pool))

CALIB_SIZES = [5, 10, 20, 50, 100, 200, 500]
N_RUNS = 10  # average over random draws

calib_results = []

for n_calib in CALIB_SIZES:
    if n_calib > len(X_pool):
        continue

    run_results = []
    for run in range(N_RUNS):
        rng = np.random.RandomState(42 + run)
        idx = rng.choice(len(X_pool), size=n_calib, replace=False)

        # Calibration data: male predictions + true values
        calib_pred_f0 = pred_male_pool[idx, 0]
        calib_pred_spl = pred_male_pool[idx, 1]
        calib_true_f0 = Y_pool.iloc[idx]['F0'].values
        calib_true_spl = Y_pool.iloc[idx]['SPL'].values

        run_r2 = {}
        run_mae = {}

        for name, c_pred, c_true, t_pred, t_true in [
            ('F0', calib_pred_f0, calib_true_f0, pred_f0, true_f0),
            ('SPL', calib_pred_spl, calib_true_spl, pred_spl, true_spl),
        ]:
            # Fit linear calibration: true = a * pred + b
            reg = LinearRegression()
            reg.fit(c_pred.reshape(-1, 1), c_true)
            calibrated = reg.predict(t_pred.reshape(-1, 1))

            run_r2[name] = r2_score(t_true, calibrated)
            run_mae[name] = mean_absolute_error(t_true, calibrated)

        run_results.append(run_r2 | {f'mae_{k}': v for k, v in run_mae.items()})

    avg = {'n_calib': n_calib}
    for key in ['F0', 'SPL', 'mae_F0', 'mae_SPL']:
        vals = [r[key] for r in run_results]
        avg[f'{key}_mean'] = np.mean(vals)
        avg[f'{key}_std'] = np.std(vals)
    calib_results.append(avg)

print(f"\n{'N_calib':>8} | {'F0 R2':>12} {'F0 MAE':>12} | {'SPL R2':>12} {'SPL MAE':>12}")
print("-" * 65)
for r in calib_results:
    print(f"{r['n_calib']:>8d} | "
          f"{r['F0_mean']:>6.3f}+/-{r['F0_std']:.3f} "
          f"{r['mae_F0_mean']:>6.1f}+/-{r['mae_F0_std']:.1f} | "
          f"{r['SPL_mean']:>6.3f}+/-{r['SPL_std']:.3f} "
          f"{r['mae_SPL_mean']:>6.1f}+/-{r['mae_SPL_std']:.1f}")

# ==================== TEST 3: COMPARE WITH FULL PIPELINE ====================
print("\n" + "=" * 70)
print("TEST 3: LINEAR CALIBRATION vs FULL TRANSFER PIPELINE")
print("=" * 70)

# Quick target-only RF at same calibration sizes for comparison
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

comparison_results = []

for n_calib in CALIB_SIZES:
    if n_calib > len(X_pool):
        continue

    calib_runs = []
    rf_runs = []

    for run in range(N_RUNS):
        rng = np.random.RandomState(42 + run)
        idx = rng.choice(len(X_pool), size=n_calib, replace=False)

        X_calib = X_pool.iloc[idx]
        Y_calib = Y_pool.iloc[idx]

        # --- Linear calibration (2 params per output) ---
        calib_pred = pred_male_pool[idx]
        calib_r2 = {}
        for i, name in enumerate(['F0', 'SPL']):
            reg = LinearRegression()
            reg.fit(calib_pred[:, i].reshape(-1, 1), Y_calib[name].values)
            calibrated = reg.predict(pred_male[:, i].reshape(-1, 1))
            calib_r2[name] = r2_score(Y_test[name].values, calibrated)
        calib_runs.append(calib_r2)

        # --- Target-only RF (many params) ---
        if n_calib >= 10:
            scaler = StandardScaler()
            X_calib_sc = scaler.fit_transform(X_calib)
            X_test_sc = scaler.transform(X_test)

            # Adaptive complexity
            if n_calib < 50:
                rf_params = {'n_estimators': 20, 'max_depth': 3,
                             'min_samples_leaf': max(3, n_calib // 10)}
            elif n_calib < 200:
                rf_params = {'n_estimators': 50, 'max_depth': 5,
                             'min_samples_leaf': 3}
            else:
                rf_params = {'n_estimators': 100, 'max_depth': 10,
                             'min_samples_leaf': 2}

            rf_r2 = {}
            for i, name in enumerate(['F0', 'SPL']):
                rf = RandomForestRegressor(random_state=42 + run, **rf_params)
                rf.fit(X_calib_sc, Y_calib[name].values)
                rf_pred = rf.predict(X_test_sc)
                rf_r2[name] = r2_score(Y_test[name].values, rf_pred)
            rf_runs.append(rf_r2)

    comp = {'n_calib': n_calib}
    for name in ['F0', 'SPL']:
        comp[f'calib_{name}_mean'] = np.mean([r[name] for r in calib_runs])
        comp[f'calib_{name}_std'] = np.std([r[name] for r in calib_runs])
        if rf_runs:
            comp[f'rf_{name}_mean'] = np.mean([r[name] for r in rf_runs])
            comp[f'rf_{name}_std'] = np.std([r[name] for r in rf_runs])
        else:
            comp[f'rf_{name}_mean'] = np.nan
            comp[f'rf_{name}_std'] = np.nan
    comparison_results.append(comp)

print(f"\n{'N':>5} | {'--- F0 R2 ---':^27} | {'--- SPL R2 ---':^27}")
print(f"{'':>5} | {'Calib (2p)':>12} {'RF (target)':>12} | {'Calib (2p)':>12} {'RF (target)':>12}")
print("-" * 65)
for r in comparison_results:
    rf_f0 = f"{r['rf_F0_mean']:.3f}" if not np.isnan(r['rf_F0_mean']) else "  n/a"
    rf_spl = f"{r['rf_SPL_mean']:.3f}" if not np.isnan(r['rf_SPL_mean']) else "  n/a"
    print(f"{r['n_calib']:>5d} | "
          f"{r['calib_F0_mean']:>7.3f}+/-{r['calib_F0_std']:.3f} "
          f"{rf_f0:>7}+/-{r.get('rf_F0_std', 0):.3f} | "
          f"{r['calib_SPL_mean']:>7.3f}+/-{r['calib_SPL_std']:.3f} "
          f"{rf_spl:>7}+/-{r.get('rf_SPL_std', 0):.3f}")

# Winner summary
print("\nWinner at each calibration size:")
for r in comparison_results:
    for name in ['F0', 'SPL']:
        c = r[f'calib_{name}_mean']
        rf = r[f'rf_{name}_mean']
        if np.isnan(rf):
            winner = "Calib (RF too few samples)"
        elif c > rf:
            winner = f"Calib by {c - rf:+.3f}"
        else:
            winner = f"RF by {rf - c:+.3f}"
        if name == 'F0':
            print(f"  n={r['n_calib']:>4d}:  F0: {winner:<30}", end="")
        else:
            print(f"  SPL: {winner}")

# ==================== PLOTS ====================
print("\n\nGenerating plots...")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Structural Transfer Test: Does Male BCM Structure Transfer to BM?',
             fontsize=14, fontweight='bold')

# --- Panel 1: Scatter — male BCM predictions vs BM truth ---
ax = axes[0, 0]
ax.scatter(pred_spl, true_spl, alpha=0.3, s=10, c='green', label='SPL')
ax.scatter(pred_f0, true_f0, alpha=0.3, s=10, c='red', label='F0')

# Fit lines for visual
for p, t, color, name in [(pred_spl, true_spl, 'darkgreen', 'SPL'),
                            (pred_f0, true_f0, 'darkred', 'F0')]:
    slope, intercept, _, _, _ = sp_stats.linregress(p, t)
    x_line = np.array([p.min(), p.max()])
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
            label=f'{name} fit: y={slope:.2f}x{intercept:+.1f}')

ax.set_xlabel('Male BCM Prediction')
ax.set_ylabel('BM True Value')
ax.set_title(f'Panel 1: Raw Predictions vs Truth\n'
             f'Spearman: F0={spearman_f0:.3f}, SPL={spearman_spl:.3f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: Rank plot — percentile of prediction vs percentile of truth ---
ax = axes[0, 1]
for pred, true, color, name in [(pred_spl, true_spl, 'green', 'SPL'),
                                  (pred_f0, true_f0, 'red', 'F0')]:
    pred_rank = sp_stats.rankdata(pred) / len(pred)
    true_rank = sp_stats.rankdata(true) / len(true)
    ax.scatter(pred_rank, true_rank, alpha=0.2, s=8, c=color, label=name)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect rank agreement')
ax.set_xlabel('Male BCM Prediction (percentile)')
ax.set_ylabel('BM Truth (percentile)')
ax.set_title('Panel 2: Rank Agreement\n(closer to diagonal = better structure transfer)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- Panel 3: Calibrated R2 vs n_calib ---
ax = axes[1, 0]
n_calibs = [r['n_calib'] for r in comparison_results]

# Linear calibration
ax.errorbar(n_calibs,
            [r['calib_SPL_mean'] for r in comparison_results],
            yerr=[r['calib_SPL_std'] for r in comparison_results],
            fmt='o-', color='green', linewidth=2, markersize=7, capsize=3,
            label='SPL: Linear Calib (2 params)')
ax.errorbar(n_calibs,
            [r['calib_F0_mean'] for r in comparison_results],
            yerr=[r['calib_F0_std'] for r in comparison_results],
            fmt='s-', color='red', linewidth=2, markersize=7, capsize=3,
            label='F0: Linear Calib (2 params)')

# Target-only RF
valid = [(r['n_calib'], r['rf_SPL_mean'], r['rf_SPL_std'])
         for r in comparison_results if not np.isnan(r['rf_SPL_mean'])]
if valid:
    ax.errorbar([v[0] for v in valid], [v[1] for v in valid],
                yerr=[v[2] for v in valid],
                fmt='o--', color='green', alpha=0.5, linewidth=1.5,
                markersize=5, capsize=3, label='SPL: Target-only RF')
valid = [(r['n_calib'], r['rf_F0_mean'], r['rf_F0_std'])
         for r in comparison_results if not np.isnan(r['rf_F0_mean'])]
if valid:
    ax.errorbar([v[0] for v in valid], [v[1] for v in valid],
                yerr=[v[2] for v in valid],
                fmt='s--', color='red', alpha=0.5, linewidth=1.5,
                markersize=5, capsize=3, label='F0: Target-only RF')

ax.set_xlabel('Number of Calibration Samples')
ax.set_ylabel('R2 on Test Set')
ax.set_title('Panel 3: Linear Calibration vs Target-Only RF')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_ylim([min(-0.5, min(r['calib_F0_mean'] - r['calib_F0_std']
                            for r in comparison_results) - 0.1), 1.0])

# --- Panel 4: Calibrated predictions scatter (using 20 calibration samples) ---
ax = axes[1, 1]

# Use median run at n=20
n_demo = 20
rng = np.random.RandomState(42)
idx = rng.choice(len(X_pool), size=n_demo, replace=False)

for i, (name, color) in enumerate([('F0', 'red'), ('SPL', 'green')]):
    reg = LinearRegression()
    reg.fit(pred_male_pool[idx, i].reshape(-1, 1), Y_pool.iloc[idx][name].values)
    calibrated = reg.predict(pred_male[:, i].reshape(-1, 1))
    r2_cal = r2_score(Y_test[name].values, calibrated)
    mae_cal = mean_absolute_error(Y_test[name].values, calibrated)

    ax.scatter(Y_test[name].values, calibrated, alpha=0.3, s=10, c=color,
               label=f'{name}: R2={r2_cal:.3f}, MAE={mae_cal:.1f}')

# Perfect prediction line
all_vals = np.concatenate([true_f0, true_spl])
lims = [all_vals.min() - 10, all_vals.max() + 10]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('True Value')
ax.set_ylabel('Calibrated Prediction')
ax.set_title(f'Panel 4: Calibrated Predictions (n_calib={n_demo})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(script_dir, 'figs', 'structural_transfer_test.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {fig_path}")

# ==================== FINAL VERDICT ====================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

for name, spear in [('F0', spearman_f0), ('SPL', spearman_spl)]:
    # Find the calibration size where R2 > 0.8
    threshold_n = None
    for r in calib_results:
        if r[f'{name}_mean'] > 0.8:
            threshold_n = r['n_calib']
            break

    print(f"\n{name}:")
    print(f"  Spearman rho = {spear:.3f}")
    if threshold_n:
        print(f"  Linear calibration reaches R2 > 0.8 with just {threshold_n} samples")
        print(f"  -> STRUCTURE TRANSFERS. Simple calibration is viable.")
    else:
        print(f"  Linear calibration never reaches R2 > 0.8")
        print(f"  -> Structure transfer is WEAK. Need richer methods or more data.")

print("\n" + "=" * 70)
