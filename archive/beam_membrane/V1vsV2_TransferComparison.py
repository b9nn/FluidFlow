"""
V1 vs V2 Beam-Membrane Comparison + Transfer Learning with BCM

Figure 1: How v1 and v2 BM models differ (data distributions, F0 vs CT, SPL vs PS)
Figure 2: TransRF transfer learning results — male BCM -> BM v2

Uses the same transfer methods as TransferLearningRF.py:
  Source-only, Target-only, Residual, Feature Augmentation, TransRF Ensemble
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats as sp_stats
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


# ==================== ADAPTIVE MODEL COMPLEXITY ====================
def get_model_params(n_samples):
    if n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(10, n_samples // 10),
                'min_samples_split': max(5, n_samples // 20)}
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    elif n_samples < 500:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}
    elif n_samples < 1000:
        return {'n_estimators': 100, 'max_depth': 10,
                'min_samples_leaf': 2, 'min_samples_split': 2}
    else:
        return {'n_estimators': 200, 'max_depth': None,
                'min_samples_leaf': 1, 'min_samples_split': 2}


# ==================== TRANSFER EXPERIMENT ====================
def run_transfer_experiment(df_target, frac, rf_male, scaler_X_male, random_state=42):
    """Run all transfer methods at a given data fraction. Returns dict of R2 scores."""
    if frac < 1.0:
        df = df_target.sample(frac=frac, random_state=random_state)
    else:
        df = df_target.copy()

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    if len(df) < 20:
        return None

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=random_state)

    n_train = len(X_train)
    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_val_arr = Y_val.values
    Y_test_arr = Y_test.values

    results = {'frac': frac, 'n_total': len(df),
               'n_train': n_train, 'n_test': len(X_test)}

    # Male predictions
    pred_male_train = rf_male.predict(scaler_X_male.transform(X_train))
    pred_male_val = rf_male.predict(scaler_X_male.transform(X_val))
    pred_male_test = rf_male.predict(scaler_X_male.transform(X_test))

    # === SOURCE ONLY ===
    for i, name in enumerate(['f0', 'spl']):
        results[f'source_only_{name}_r2'] = r2_score(Y_test_arr[:, i], pred_male_test[:, i])

    # === TARGET ONLY ===
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    X_val_sc = scaler_X.transform(X_val)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    for i, name in enumerate(['f0', 'spl']):
        results[f'target_only_{name}_r2'] = r2_score(Y_test_arr[:, i], pred_target[:, i])

    # === RESIDUAL CORRECTION ===
    rf_residual = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_residual.fit(X_train.values, Y_train_arr - pred_male_train)
    pred_residual = pred_male_test + rf_residual.predict(X_test.values)

    for i, name in enumerate(['f0', 'spl']):
        results[f'residual_{name}_r2'] = r2_score(Y_test_arr[:, i], pred_residual[:, i])

    # === FEATURE AUGMENTATION ===
    X_train_aug = np.column_stack([X_train.values, pred_male_train])
    X_test_aug = np.column_stack([X_test.values, pred_male_test])

    rf_augmented = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_augmented.fit(X_train_aug, Y_train_arr)
    pred_augmented = rf_augmented.predict(X_test_aug)

    for i, name in enumerate(['f0', 'spl']):
        results[f'augmented_{name}_r2'] = r2_score(Y_test_arr[:, i], pred_augmented[:, i])

    # === TRANSRF ENSEMBLE (learned weights on validation split) ===
    pred_t_val = rf_target.predict(X_val_sc)
    pred_r_val = pred_male_val + rf_residual.predict(X_val.values)
    X_val_aug = np.column_stack([X_val.values, pred_male_val])
    pred_a_val = rf_augmented.predict(X_val_aug)

    weights = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i], pred_r_val[:, i], pred_a_val[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights[name] = w

    pred_transrf = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i], pred_residual[:, i], pred_augmented[:, i]])
        pred_transrf[:, i] = stack @ weights[name]

    for i, name in enumerate(['f0', 'spl']):
        results[f'transrf_{name}_r2'] = r2_score(Y_test_arr[:, i], pred_transrf[:, i])

    results['weights_f0'] = weights['F0']
    results['weights_spl'] = weights['SPL']

    return results


# ==================== LOAD DATA ====================
def load_and_clean(filepath, version_label):
    df = pd.read_fwf(filepath)
    df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    n_raw = len(df)
    # Only remove truly invalid rows (F0 outside physical range)
    df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
    # For v1 apply the original SPL filter since all v1 data is phonating
    if 'v1' in version_label:
        df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
    # Light z-score on F0
    if len(df) > 10:
        z_f0 = np.abs(sp_stats.zscore(df['F0']))
        df = df[z_f0 < 3]
    df = df.reset_index(drop=True)
    print(f"  {version_label}: {n_raw} raw -> {len(df)} clean")
    return df


print("=" * 70)
print("V1 vs V2 BEAM-MEMBRANE COMPARISON + TRANSFER LEARNING")
print("=" * 70)

print("\nLoading data...")
df_v1 = load_and_clean(os.path.join(script_dir, 'Beam_Membrane_Data.csv'), 'v1')
df_v2 = load_and_clean(os.path.join(script_dir, 'Beam_Membrane_Data_v2.csv'), 'v2')

# For v2, flag phonation regime
df_v2['phonating'] = (df_v2['SPL'] >= 40).astype(int)
df_v2_phon = df_v2[df_v2['phonating'] == 1].copy()
print(f"  v2 phonating subset: {len(df_v2_phon)} samples")

print("\nLoading male BCM model...")
rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
scaler_X_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))


# ==================================================================
# FIGURE 1: V1 vs V2 Comparison
# ==================================================================
print("\n" + "=" * 70)
print("FIGURE 1: V1 vs V2 Beam-Membrane Model Comparison")
print("=" * 70)

fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('Beam-Membrane: v1 (old constitutive) vs v2 (updated constitutive)',
              fontsize=14, fontweight='bold')

# --- Panel 1: F0 distributions ---
ax = axes[0, 0]
ax.hist(df_v1['F0'], bins=30, alpha=0.6, color='blue', label=f'v1 (n={len(df_v1)})', density=True)
ax.hist(df_v2['F0'], bins=30, alpha=0.6, color='red', label=f'v2 all (n={len(df_v2)})', density=True)
ax.hist(df_v2_phon['F0'], bins=30, alpha=0.4, color='orange',
        label=f'v2 phonating (n={len(df_v2_phon)})', density=True, linestyle='--')
ax.set_xlabel('F0 (Hz)')
ax.set_ylabel('Density')
ax.set_title('F0 Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: SPL distributions ---
ax = axes[0, 1]
ax.hist(df_v1['SPL'], bins=30, alpha=0.6, color='blue', label='v1', density=True)
ax.hist(df_v2['SPL'], bins=50, alpha=0.6, color='red', label='v2 all', density=True)
ax.hist(df_v2_phon['SPL'], bins=30, alpha=0.4, color='orange',
        label='v2 phonating', density=True)
ax.set_xlabel('SPL (dB)')
ax.set_ylabel('Density')
ax.set_title('SPL Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: F0 vs a_CT (the key relationship) ---
ax = axes[0, 2]
N_CT_BINS = 8
for data, color, label, marker in [
    (df_v1, 'blue', 'v1', 'o'),
    (df_v2, 'red', 'v2 (all)', 's'),
    (df_v2_phon, 'orange', 'v2 (phonating)', '^'),
]:
    bins = pd.qcut(data['a_CT'], N_CT_BINS, duplicates='drop')
    grouped = data.groupby(bins)['F0'].agg(['mean', 'std'])
    midpoints = [(b.left + b.right) / 2 for b in grouped.index]
    ax.errorbar(midpoints, grouped['mean'], yerr=grouped['std'],
                fmt=f'{marker}-', color=color, label=label, linewidth=2,
                markersize=6, capsize=3, alpha=0.8)
ax.set_xlabel('a_CT')
ax.set_ylabel('Mean F0 (Hz)')
ax.set_title('F0 vs CT Activation\n(should increase — key diagnostic)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 4: SPL vs PS ---
ax = axes[1, 0]
N_PS_BINS = 8
for data, color, label, marker in [
    (df_v1, 'blue', 'v1', 'o'),
    (df_v2_phon, 'orange', 'v2 (phonating)', '^'),
]:
    bins = pd.qcut(data['PS'], N_PS_BINS, duplicates='drop')
    grouped = data.groupby(bins)['SPL'].agg(['mean', 'std'])
    midpoints = [(b.left + b.right) / 2 for b in grouped.index]
    ax.errorbar(midpoints, grouped['mean'], yerr=grouped['std'],
                fmt=f'{marker}-', color=color, label=label, linewidth=2,
                markersize=6, capsize=3, alpha=0.8)
ax.set_xlabel('PS (Pa)')
ax.set_ylabel('Mean SPL (dB)')
ax.set_title('SPL vs Subglottal Pressure')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 5: F0 vs a_CT at different a_TA levels ---
ax = axes[1, 1]
ta_bins_labels = ['low a_TA', 'mid a_TA', 'high a_TA']
ta_quantiles = pd.qcut(df_v2['a_TA'], 3, labels=ta_bins_labels, duplicates='drop')
colors_ta = ['#2196F3', '#4CAF50', '#F44336']

for (ta_label, color) in zip(ta_bins_labels, colors_ta):
    subset = df_v2[ta_quantiles == ta_label]
    if len(subset) < 20:
        continue
    ct_bins = pd.qcut(subset['a_CT'], 6, duplicates='drop')
    grouped = subset.groupby(ct_bins)['F0'].agg(['mean', 'std'])
    midpoints = [(b.left + b.right) / 2 for b in grouped.index]
    ta_range = subset['a_TA']
    ax.errorbar(midpoints, grouped['mean'], yerr=grouped['std'],
                fmt='o-', color=color, linewidth=2, markersize=5, capsize=3,
                label=f'v2: {ta_label} ({ta_range.min():.2f}-{ta_range.max():.2f})')

# Same for v1
ta_quantiles_v1 = pd.qcut(df_v1['a_TA'], 3, labels=ta_bins_labels, duplicates='drop')
for (ta_label, color) in zip(ta_bins_labels, colors_ta):
    subset = df_v1[ta_quantiles_v1 == ta_label]
    if len(subset) < 20:
        continue
    ct_bins = pd.qcut(subset['a_CT'], 6, duplicates='drop')
    grouped = subset.groupby(ct_bins)['F0'].agg(['mean', 'std'])
    midpoints = [(b.left + b.right) / 2 for b in grouped.index]
    ax.plot(midpoints, grouped['mean'], '--', color=color, alpha=0.5,
            linewidth=1.5, label=f'v1: {ta_label}')

ax.set_xlabel('a_CT')
ax.set_ylabel('Mean F0 (Hz)')
ax.set_title('F0 vs CT by TA Level\n(solid=v2, dashed=v1)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# --- Panel 6: Summary statistics table ---
ax = axes[1, 2]
ax.axis('off')

summary_data = [
    ['Metric', 'v1', 'v2 (all)', 'v2 (phonating)'],
    ['N samples', f'{len(df_v1)}', f'{len(df_v2)}', f'{len(df_v2_phon)}'],
    ['F0 mean', f'{df_v1["F0"].mean():.1f}', f'{df_v2["F0"].mean():.1f}', f'{df_v2_phon["F0"].mean():.1f}'],
    ['F0 std', f'{df_v1["F0"].std():.1f}', f'{df_v2["F0"].std():.1f}', f'{df_v2_phon["F0"].std():.1f}'],
    ['F0 range', f'{df_v1["F0"].min():.0f}-{df_v1["F0"].max():.0f}',
     f'{df_v2["F0"].min():.0f}-{df_v2["F0"].max():.0f}',
     f'{df_v2_phon["F0"].min():.0f}-{df_v2_phon["F0"].max():.0f}'],
    ['SPL mean', f'{df_v1["SPL"].mean():.1f}', f'{df_v2["SPL"].mean():.1f}', f'{df_v2_phon["SPL"].mean():.1f}'],
    ['SPL std', f'{df_v1["SPL"].std():.1f}', f'{df_v2["SPL"].std():.1f}', f'{df_v2_phon["SPL"].std():.1f}'],
    ['PS mean', f'{df_v1["PS"].mean():.0f}', f'{df_v2["PS"].mean():.0f}', f'{df_v2_phon["PS"].mean():.0f}'],
    ['PS range', f'{df_v1["PS"].min():.0f}-{df_v1["PS"].max():.0f}',
     f'{df_v2["PS"].min():.0f}-{df_v2["PS"].max():.0f}',
     f'{df_v2_phon["PS"].min():.0f}-{df_v2_phon["PS"].max():.0f}'],
]

table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for j in range(4):
    table[0, j].set_facecolor('#E0E0E0')
    table[0, j].set_text_props(fontweight='bold')
ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
fig1_path = os.path.join(script_dir, 'figs', 'v1_vs_v2_comparison.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig1_path}")


# ==================================================================
# FIGURE 2: Transfer Learning Results — Male BCM -> BM v2
# ==================================================================
print("\n" + "=" * 70)
print("FIGURE 2: Transfer Learning — Male BCM -> BM v2")
print("=" * 70)

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 3

# Run on full v2 data
print("\nRunning transfer on v2 (all data)...")
all_results_v2 = []
for frac in FRACTIONS:
    frac_runs = []
    for run in range(N_RUNS):
        rs = 42 + run
        result = run_transfer_experiment(df_v2, frac, rf_male, scaler_X_male, rs)
        if result is not None:
            frac_runs.append(result)

    if frac_runs:
        avg = {'frac': frac,
               'n_samples': int(np.mean([r['n_train'] for r in frac_runs]))}
        for key in ['source_only', 'target_only', 'residual', 'augmented', 'transrf']:
            for t in ['f0', 'spl']:
                vals = [r[f'{key}_{t}_r2'] for r in frac_runs]
                avg[f'{key}_{t}_r2_mean'] = np.mean(vals)
                avg[f'{key}_{t}_r2_std'] = np.std(vals)
        avg['weights_f0'] = np.mean([r['weights_f0'] for r in frac_runs], axis=0)
        avg['weights_spl'] = np.mean([r['weights_spl'] for r in frac_runs], axis=0)
        all_results_v2.append(avg)
    print(f"  {frac*100:.0f}% done")

# Run on v1 data for comparison
print("\nRunning transfer on v1...")
all_results_v1 = []
for frac in FRACTIONS:
    frac_runs = []
    for run in range(N_RUNS):
        rs = 42 + run
        result = run_transfer_experiment(df_v1, frac, rf_male, scaler_X_male, rs)
        if result is not None:
            frac_runs.append(result)

    if frac_runs:
        avg = {'frac': frac,
               'n_samples': int(np.mean([r['n_train'] for r in frac_runs]))}
        for key in ['source_only', 'target_only', 'residual', 'augmented', 'transrf']:
            for t in ['f0', 'spl']:
                vals = [r[f'{key}_{t}_r2'] for r in frac_runs]
                avg[f'{key}_{t}_r2_mean'] = np.mean(vals)
                avg[f'{key}_{t}_r2_std'] = np.std(vals)
        all_results_v1.append(avg)
    print(f"  {frac*100:.0f}% done")

# Print summary tables
print("\n\nTRANSFER RESULTS — v2 (all data):")
print(f"{'Frac':>6} {'N':>5} | {'Source':>8} {'Target':>8} {'Resid':>8} {'Augment':>8} {'TransRF':>8} | "
      f"{'Source':>8} {'Target':>8} {'Resid':>8} {'Augment':>8} {'TransRF':>8}")
print(f"{'':>12} | {'--- F0 R2 ---':^44} | {'--- SPL R2 ---':^44}")
print("-" * 120)
for r in all_results_v2:
    print(f"{r['frac']*100:>5.0f}% {r['n_samples']:>5d} | "
          f"{r['source_only_f0_r2_mean']:>7.3f}  {r['target_only_f0_r2_mean']:>7.3f}  "
          f"{r['residual_f0_r2_mean']:>7.3f}  {r['augmented_f0_r2_mean']:>7.3f}  "
          f"{r['transrf_f0_r2_mean']:>7.3f}  | "
          f"{r['source_only_spl_r2_mean']:>7.3f}  {r['target_only_spl_r2_mean']:>7.3f}  "
          f"{r['residual_spl_r2_mean']:>7.3f}  {r['augmented_spl_r2_mean']:>7.3f}  "
          f"{r['transrf_spl_r2_mean']:>7.3f}")


# ==================================================================
# FIGURE 2 PLOTS
# ==================================================================
methods_to_plot = [
    ('source_only', 'Source Only (Male BCM)', 'gray', '--', 'D'),
    ('target_only', 'Target Only', 'blue', '-', 'o'),
    ('residual', 'Residual Correction', 'purple', '-', 's'),
    ('augmented', 'Feature Augmentation', 'green', '-', '^'),
    ('transrf', 'TransRF Ensemble', 'red', '-', '*'),
]

fig2, axes = plt.subplots(2, 3, figsize=(22, 13))
fig2.suptitle('Transfer Learning: Male BCM -> Beam-Membrane (v1 vs v2)',
              fontsize=14, fontweight='bold')

# --- Panel 1: F0 R2 vs samples — v2 ---
ax = axes[0, 0]
n_samples = [r['n_samples'] for r in all_results_v2]
for key, label, color, ls, marker in methods_to_plot:
    means = [r[f'{key}_f0_r2_mean'] for r in all_results_v2]
    stds = [r[f'{key}_f0_r2_std'] for r in all_results_v2]
    ax.errorbar(n_samples, means, yerr=stds, fmt=f'{marker}-',
                label=label, color=color, linestyle=ls,
                linewidth=2, markersize=7, capsize=3)
ax.set_xlabel('Training Samples')
ax.set_ylabel('R2 (F0)')
ax.set_title('F0 Transfer — BM v2')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
y_min = min(r['source_only_f0_r2_mean'] for r in all_results_v2)
ax.set_ylim([min(-0.5, y_min - 0.1), 1])

# --- Panel 2: SPL R2 vs samples — v2 ---
ax = axes[0, 1]
for key, label, color, ls, marker in methods_to_plot:
    means = [r[f'{key}_spl_r2_mean'] for r in all_results_v2]
    stds = [r[f'{key}_spl_r2_std'] for r in all_results_v2]
    ax.errorbar(n_samples, means, yerr=stds, fmt=f'{marker}-',
                label=label, color=color, linestyle=ls,
                linewidth=2, markersize=7, capsize=3)
ax.set_xlabel('Training Samples')
ax.set_ylabel('R2 (SPL)')
ax.set_title('SPL Transfer — BM v2')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
y_min = min(r['source_only_spl_r2_mean'] for r in all_results_v2)
ax.set_ylim([min(-0.5, y_min - 0.1), 1])

# --- Panel 3: Average R2 — v2 ---
ax = axes[0, 2]
for key, label, color, ls, marker in methods_to_plot:
    avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
              for r in all_results_v2]
    ax.plot(n_samples, avg_r2, f'{marker}-', label=label, color=color,
            linestyle=ls, linewidth=2, markersize=8)
ax.set_xlabel('Training Samples')
ax.set_ylabel('Average R2 (F0+SPL)/2')
ax.set_title('Combined Transfer — BM v2')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# --- Panel 4: V1 vs V2 — F0 TransRF comparison ---
ax = axes[1, 0]
n_v1 = [r['n_samples'] for r in all_results_v1]
n_v2 = [r['n_samples'] for r in all_results_v2]

for key, label_suffix, ls in [('target_only', 'Target Only', '-'), ('transrf', 'TransRF', '-')]:
    v1_vals = [r[f'{key}_f0_r2_mean'] for r in all_results_v1]
    v2_vals = [r[f'{key}_f0_r2_mean'] for r in all_results_v2]
    color = 'blue' if 'target' in key else 'red'
    ax.plot(n_v1, v1_vals, f'o--', color=color, alpha=0.5, linewidth=1.5,
            markersize=5, label=f'v1 {label_suffix}')
    ax.plot(n_v2, v2_vals, f's{ls}', color=color, linewidth=2,
            markersize=7, label=f'v2 {label_suffix}')

ax.set_xlabel('Training Samples')
ax.set_ylabel('R2 (F0)')
ax.set_title('F0: v1 (dashed) vs v2 (solid)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# --- Panel 5: V1 vs V2 — SPL TransRF comparison ---
ax = axes[1, 1]
for key, label_suffix, ls in [('target_only', 'Target Only', '-'), ('transrf', 'TransRF', '-')]:
    v1_vals = [r[f'{key}_spl_r2_mean'] for r in all_results_v1]
    v2_vals = [r[f'{key}_spl_r2_mean'] for r in all_results_v2]
    color = 'blue' if 'target' in key else 'red'
    ax.plot(n_v1, v1_vals, f'o--', color=color, alpha=0.5, linewidth=1.5,
            markersize=5, label=f'v1 {label_suffix}')
    ax.plot(n_v2, v2_vals, f's{ls}', color=color, linewidth=2,
            markersize=7, label=f'v2 {label_suffix}')

ax.set_xlabel('Training Samples')
ax.set_ylabel('R2 (SPL)')
ax.set_title('SPL: v1 (dashed) vs v2 (solid)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# --- Panel 6: TransRF learned weights (v2) ---
ax = axes[1, 2]
fracs_pct = [r['frac'] * 100 for r in all_results_v2]
w_labels = ['Target-only', 'Residual', 'Augmented']
colors_w = ['blue', 'purple', 'green']

bar_width = 2.5
for i_method in range(3):
    w_f0 = [r['weights_f0'][i_method] for r in all_results_v2]
    w_spl = [r['weights_spl'][i_method] for r in all_results_v2]
    x = np.array(fracs_pct) + (i_method - 1) * bar_width
    ax.bar(x - 5, w_f0, bar_width, color=colors_w[i_method], alpha=0.7,
           label=f'F0: {w_labels[i_method]}' if i_method == 0 else f'{w_labels[i_method]}')
    ax.bar(x + 5, w_spl, bar_width, color=colors_w[i_method], alpha=0.3,
           hatch='//')

ax.set_xlabel('Target Data Fraction (%)')
ax.set_ylabel('Weight')
ax.set_title('TransRF Weights (v2)\nsolid=F0, hatched=SPL')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

plt.tight_layout()
fig2_path = os.path.join(script_dir, 'figs', 'v1_v2_transfer_comparison.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig2_path}")


# ==================================================================
# FINAL SUMMARY
# ==================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'':>30} {'v1':>12} {'v2':>12}")
print("-" * 55)
print(f"{'N samples (clean)':<30} {len(df_v1):>12} {len(df_v2):>12}")

if all_results_v1 and all_results_v2:
    r1 = all_results_v1[-1]
    r2 = all_results_v2[-1]
    print(f"\nAt 100% target data:")
    print(f"{'Target-only F0 R2':<30} {r1['target_only_f0_r2_mean']:>12.3f} {r2['target_only_f0_r2_mean']:>12.3f}")
    print(f"{'Target-only SPL R2':<30} {r1['target_only_spl_r2_mean']:>12.3f} {r2['target_only_spl_r2_mean']:>12.3f}")
    print(f"{'TransRF F0 R2':<30} {r1['transrf_f0_r2_mean']:>12.3f} {r2['transrf_f0_r2_mean']:>12.3f}")
    print(f"{'TransRF SPL R2':<30} {r1['transrf_spl_r2_mean']:>12.3f} {r2['transrf_spl_r2_mean']:>12.3f}")

    # Transfer gain
    for ver, results in [('v1', all_results_v1), ('v2', all_results_v2)]:
        r = results[-1]
        gain_f0 = r['transrf_f0_r2_mean'] - r['target_only_f0_r2_mean']
        gain_spl = r['transrf_spl_r2_mean'] - r['target_only_spl_r2_mean']
        print(f"\n{ver} Transfer gain (TransRF - Target-only):")
        print(f"  F0:  {gain_f0:+.3f}")
        print(f"  SPL: {gain_spl:+.3f}")

print("\n" + "=" * 70)
