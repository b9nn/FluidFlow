"""
SPL Transfer Comparison: Male BCM -> BM vs Male BCM -> Female BCM

Part A: Male BCM baseline R2 on BM test set (SPL and F0)
Part B: SPL-only transfer learning (male BCM -> BM) at varying fractions
Part C: F0 transfer on CT-consistent BM subset vs full dataset
Part D: Transfer gain comparison: BM vs female BCM (SPL)

Requires: Run CTDiagnostic.py first to generate ct_consistent_indices.csv

Output:
  - Beam_Membrane/figs/spl_transfer_comparison.png  (4-panel figure)
  - Console summary tables
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


# ==================== ADAPTIVE MODEL COMPLEXITY ====================
def get_model_params(n_samples):
    """Adjust RF complexity based on available training samples."""
    if n_samples < 100:
        return {
            'n_estimators': 20, 'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20),
        }
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


# ==================== HELPERS ====================
def get_male_predictions(X, scaler, model):
    """Get predictions from the pre-trained male BCM model."""
    return model.predict(scaler.transform(X))


def run_single_output_transfer(X_pool, y_pool, X_test, y_test,
                                rf_male, scaler_X_male, target_col_idx,
                                frac, random_state=42):
    """
    Run transfer learning for a single output variable at a given fraction.
    target_col_idx: 0 for F0, 1 for SPL (index into male model's 2-output predictions).
    Returns dict of R2 scores for each method.
    """
    # Subsample the training pool
    if frac < 1.0:
        n_sample = max(10, int(len(X_pool) * frac))
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_pool), size=n_sample, replace=False)
        X_train = X_pool.iloc[idx]
        y_train = y_pool.iloc[idx]
    else:
        X_train = X_pool
        y_train = y_pool

    n_train = len(X_train)
    if n_train < 10:
        return None

    model_params = get_model_params(n_train)
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    results = {'n_train': n_train}

    # Male model predictions (2-output model, pick the right column)
    pred_male_train_full = get_male_predictions(X_train, scaler_X_male, rf_male)
    pred_male_test_full = get_male_predictions(X_test, scaler_X_male, rf_male)
    pred_male_train = pred_male_train_full[:, target_col_idx]
    pred_male_test = pred_male_test_full[:, target_col_idx]

    # === SOURCE ONLY ===
    results['source_only_r2'] = r2_score(y_test_arr, pred_male_test)

    # === TARGET ONLY ===
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    rf_target = RandomForestRegressor(random_state=random_state, **model_params)
    rf_target.fit(X_train_sc, y_train_arr)
    pred_target = rf_target.predict(X_test_sc)
    results['target_only_r2'] = r2_score(y_test_arr, pred_target)

    # === RESIDUAL CORRECTION ===
    rf_residual = RandomForestRegressor(random_state=random_state, **model_params)
    rf_residual.fit(X_train.values, y_train_arr - pred_male_train)
    pred_residual = pred_male_test + rf_residual.predict(X_test.values)
    results['residual_r2'] = r2_score(y_test_arr, pred_residual)

    # === FEATURE AUGMENTATION ===
    X_train_aug = np.column_stack([X_train.values, pred_male_train])
    X_test_aug = np.column_stack([X_test.values, pred_male_test])

    rf_augmented = RandomForestRegressor(random_state=random_state, **model_params)
    rf_augmented.fit(X_train_aug, y_train_arr)
    pred_augmented = rf_augmented.predict(X_test_aug)
    results['augmented_r2'] = r2_score(y_test_arr, pred_augmented)

    # === TRANSRF ENSEMBLE (learned weights on hold-out from training) ===
    # Split training into sub-train / val for weight learning
    if n_train >= 20:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state)

        mp_sub = get_model_params(len(X_tr))
        y_tr_arr = y_tr.values
        y_va_arr = y_va.values

        # Retrain sub-models on sub-train
        sc_sub = StandardScaler()
        X_tr_sc = sc_sub.fit_transform(X_tr)
        X_va_sc = sc_sub.transform(X_va)

        rf_t = RandomForestRegressor(random_state=random_state, **mp_sub)
        rf_t.fit(X_tr_sc, y_tr_arr)
        pred_t_va = rf_t.predict(X_va_sc)

        pred_m_tr = get_male_predictions(X_tr, scaler_X_male, rf_male)[:, target_col_idx]
        pred_m_va = get_male_predictions(X_va, scaler_X_male, rf_male)[:, target_col_idx]

        rf_r = RandomForestRegressor(random_state=random_state, **mp_sub)
        rf_r.fit(X_tr.values, y_tr_arr - pred_m_tr)
        pred_r_va = pred_m_va + rf_r.predict(X_va.values)

        X_tr_aug = np.column_stack([X_tr.values, pred_m_tr])
        X_va_aug = np.column_stack([X_va.values, pred_m_va])
        rf_a = RandomForestRegressor(random_state=random_state, **mp_sub)
        rf_a.fit(X_tr_aug, y_tr_arr)
        pred_a_va = rf_a.predict(X_va_aug)

        # Learn weights
        stack_va = np.column_stack([pred_t_va, pred_r_va, pred_a_va])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack_va, y_va_arr)
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])

        # Apply weights to test predictions
        stack_test = np.column_stack([pred_target, pred_residual, pred_augmented])
        pred_transrf = stack_test @ w
    else:
        # Too few samples for weight learning, equal weights
        pred_transrf = (pred_target + pred_residual + pred_augmented) / 3.0

    results['transrf_r2'] = r2_score(y_test_arr, pred_transrf)

    return results


def run_transfer_sweep(X_pool, y_pool, X_test, y_test,
                       rf_male, scaler_X_male, target_col_idx,
                       fractions, n_runs=3, label=""):
    """Run transfer learning sweep over fractions with multiple runs."""
    all_results = []

    for frac in fractions:
        frac_runs = []
        for run in range(n_runs):
            rs = 42 + run
            result = run_single_output_transfer(
                X_pool, y_pool, X_test, y_test,
                rf_male, scaler_X_male, target_col_idx,
                frac, random_state=rs)
            if result is not None:
                frac_runs.append(result)

        if frac_runs:
            avg = {'frac': frac, 'n_train': int(np.mean([r['n_train'] for r in frac_runs]))}
            for method in ['source_only', 'target_only', 'residual', 'augmented', 'transrf']:
                vals = [r[f'{method}_r2'] for r in frac_runs]
                avg[f'{method}_r2_mean'] = np.mean(vals)
                avg[f'{method}_r2_std'] = np.std(vals)
            all_results.append(avg)

    return all_results


# ==================== LOAD DATA ====================
def load_bm_data():
    """Load and clean Beam-Membrane data."""
    df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
    df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()

    df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
    df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
    z_f0 = np.abs(stats.zscore(df['F0']))
    z_spl = np.abs(stats.zscore(df['SPL']))
    df = df[(z_f0 < 3) & (z_spl < 3)]
    df = df.reset_index(drop=True)
    return df


def load_female_bcm_data():
    """Load and clean female BCM data."""
    df = pd.read_csv(os.path.join(project_root, 'FemaleBCM.csv'))
    df = df[df['ACFL'] > 30]
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    df = df.reset_index(drop=True)
    return df


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("SPL TRANSFER COMPARISON")
    print("Male BCM -> BM  vs  Male BCM -> Female BCM")
    print("=" * 70)

    # Load source model
    print("\nLoading pre-trained male BCM model...")
    rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
    scaler_X_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))

    # Load BM data
    print("Loading Beam-Membrane data...")
    df_bm = load_bm_data()
    print(f"  BM samples: {len(df_bm)}")

    # Load female BCM data
    print("Loading female BCM data...")
    df_fem = load_female_bcm_data()
    print(f"  Female BCM samples: {len(df_fem)}")

    # Load consistent indices from CTDiagnostic
    csv_path = os.path.join(script_dir, 'ct_consistent_indices.csv')
    if os.path.exists(csv_path):
        consistent_idx = pd.read_csv(csv_path)['index'].values
        df_bm_consistent = df_bm.loc[df_bm.index.isin(consistent_idx)].copy()
        print(f"  BM consistent subset: {len(df_bm_consistent)} samples")
    else:
        print("  WARNING: ct_consistent_indices.csv not found. Run CTDiagnostic.py first.")
        print("  Skipping Part C (F0 on consistent subset).")
        df_bm_consistent = None

    FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    N_RUNS = 3

    # ==================================================================
    # PART A: Male BCM baseline on BM
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART A: Male BCM Baseline on BM Test Set")
    print("=" * 70)

    X_bm = df_bm[['a_CT', 'a_TA', 'PS']]
    Y_bm = df_bm[['F0', 'SPL']]

    X_bm_pool, X_bm_test, Y_bm_pool, Y_bm_test = train_test_split(
        X_bm, Y_bm, test_size=0.2, random_state=42)

    pred_male_bm = get_male_predictions(X_bm_test, scaler_X_male, rf_male)
    baseline_f0_r2 = r2_score(Y_bm_test['F0'], pred_male_bm[:, 0])
    baseline_spl_r2 = r2_score(Y_bm_test['SPL'], pred_male_bm[:, 1])

    print(f"  Male BCM -> BM test set:")
    print(f"    F0  R2 = {baseline_f0_r2:.4f}")
    print(f"    SPL R2 = {baseline_spl_r2:.4f}")

    # ==================================================================
    # PART B: SPL-only transfer learning (male BCM -> BM)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART B: SPL Transfer Learning (Male BCM -> BM)")
    print("=" * 70)

    spl_bm_results = run_transfer_sweep(
        X_bm_pool, Y_bm_pool['SPL'], X_bm_test, Y_bm_test['SPL'],
        rf_male, scaler_X_male, target_col_idx=1,
        fractions=FRACTIONS, n_runs=N_RUNS, label="SPL BM")

    print("\nSPL Transfer Results (Male BCM -> BM):")
    print(f"{'Frac':>6} {'N':>5} {'Source':>10} {'Target':>10} {'Residual':>10} "
          f"{'Augment':>10} {'TransRF':>10}")
    print("-" * 65)
    for r in spl_bm_results:
        print(f"{r['frac']*100:>5.0f}% {r['n_train']:>5d} "
              f"{r['source_only_r2_mean']:>9.3f}  {r['target_only_r2_mean']:>9.3f}  "
              f"{r['residual_r2_mean']:>9.3f}  {r['augmented_r2_mean']:>9.3f}  "
              f"{r['transrf_r2_mean']:>9.3f}")

    # ==================================================================
    # PART C: F0 transfer on consistent BM subset vs full
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART C: F0 Transfer — Consistent Subset vs Full BM")
    print("=" * 70)

    # Full BM F0 transfer
    f0_full_results = run_transfer_sweep(
        X_bm_pool, Y_bm_pool['F0'], X_bm_test, Y_bm_test['F0'],
        rf_male, scaler_X_male, target_col_idx=0,
        fractions=FRACTIONS, n_runs=N_RUNS, label="F0 Full BM")

    print("\nF0 Transfer on FULL BM dataset:")
    print(f"{'Frac':>6} {'N':>5} {'Source':>10} {'Target':>10} {'Residual':>10} "
          f"{'Augment':>10} {'TransRF':>10}")
    print("-" * 65)
    for r in f0_full_results:
        print(f"{r['frac']*100:>5.0f}% {r['n_train']:>5d} "
              f"{r['source_only_r2_mean']:>9.3f}  {r['target_only_r2_mean']:>9.3f}  "
              f"{r['residual_r2_mean']:>9.3f}  {r['augmented_r2_mean']:>9.3f}  "
              f"{r['transrf_r2_mean']:>9.3f}")

    # Consistent subset F0 transfer
    f0_consistent_results = None
    if df_bm_consistent is not None and len(df_bm_consistent) >= 50:
        X_con = df_bm_consistent[['a_CT', 'a_TA', 'PS']]
        Y_con_f0 = df_bm_consistent['F0']

        X_con_pool, X_con_test, y_con_pool, y_con_test = train_test_split(
            X_con, Y_con_f0, test_size=0.2, random_state=42)

        f0_consistent_results = run_transfer_sweep(
            X_con_pool, y_con_pool, X_con_test, y_con_test,
            rf_male, scaler_X_male, target_col_idx=0,
            fractions=FRACTIONS, n_runs=N_RUNS, label="F0 Consistent BM")

        print("\nF0 Transfer on CONSISTENT BM subset:")
        print(f"{'Frac':>6} {'N':>5} {'Source':>10} {'Target':>10} {'Residual':>10} "
              f"{'Augment':>10} {'TransRF':>10}")
        print("-" * 65)
        for r in f0_consistent_results:
            print(f"{r['frac']*100:>5.0f}% {r['n_train']:>5d} "
                  f"{r['source_only_r2_mean']:>9.3f}  {r['target_only_r2_mean']:>9.3f}  "
                  f"{r['residual_r2_mean']:>9.3f}  {r['augmented_r2_mean']:>9.3f}  "
                  f"{r['transrf_r2_mean']:>9.3f}")
    else:
        print("  Skipping consistent subset (not enough data or CSV not found)")

    # ==================================================================
    # PART D: SPL gain comparison — BM vs female BCM
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART D: Transfer Gain Comparison — BM vs Female BCM (SPL)")
    print("=" * 70)

    # Female BCM SPL transfer
    X_fem = df_fem[['a_CT', 'a_TA', 'PS']]
    Y_fem_spl = df_fem['SPL']

    X_fem_pool, X_fem_test, y_fem_pool, y_fem_test = train_test_split(
        X_fem, Y_fem_spl, test_size=0.2, random_state=42)

    spl_fem_results = run_transfer_sweep(
        X_fem_pool, y_fem_pool, X_fem_test, y_fem_test,
        rf_male, scaler_X_male, target_col_idx=1,
        fractions=FRACTIONS, n_runs=N_RUNS, label="SPL Female BCM")

    print("\nSPL Transfer Results (Male BCM -> Female BCM):")
    print(f"{'Frac':>6} {'N':>5} {'Source':>10} {'Target':>10} {'Residual':>10} "
          f"{'Augment':>10} {'TransRF':>10}")
    print("-" * 65)
    for r in spl_fem_results:
        print(f"{r['frac']*100:>5.0f}% {r['n_train']:>5d} "
              f"{r['source_only_r2_mean']:>9.3f}  {r['target_only_r2_mean']:>9.3f}  "
              f"{r['residual_r2_mean']:>9.3f}  {r['augmented_r2_mean']:>9.3f}  "
              f"{r['transrf_r2_mean']:>9.3f}")

    # Compute transfer gains
    print("\n\nTRANSFER GAIN COMPARISON (best_transfer_R2 - target_only_R2):")
    print("-" * 70)
    print(f"{'Frac':>6} | {'BM Target':>10} {'BM Best':>10} {'BM Gain':>10} | "
          f"{'Fem Target':>10} {'Fem Best':>10} {'Fem Gain':>10}")
    print("-" * 70)

    bm_gains = []
    fem_gains = []
    for r_bm, r_fem in zip(spl_bm_results, spl_fem_results):
        bm_target = r_bm['target_only_r2_mean']
        bm_best = max(r_bm[f'{m}_r2_mean'] for m in
                      ['source_only', 'residual', 'augmented', 'transrf'])
        bm_gain = bm_best - bm_target

        fem_target = r_fem['target_only_r2_mean']
        fem_best = max(r_fem[f'{m}_r2_mean'] for m in
                       ['source_only', 'residual', 'augmented', 'transrf'])
        fem_gain = fem_best - fem_target

        bm_gains.append(bm_gain)
        fem_gains.append(fem_gain)

        print(f"{r_bm['frac']*100:>5.0f}% | {bm_target:>9.3f}  {bm_best:>9.3f}  "
              f"{bm_gain:>+9.3f}  | {fem_target:>9.3f}  {fem_best:>9.3f}  {fem_gain:>+9.3f}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    print("\n\nGenerating plots...")
    os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('SPL Transfer Learning Comparison', fontsize=14, fontweight='bold')

    methods_to_plot = [
        ('source_only', 'Source Only (Male BCM)', 'gray', '--', 'D'),
        ('target_only', 'Target Only', 'blue', '-', 'o'),
        ('residual', 'Residual Correction', 'purple', '-', 's'),
        ('augmented', 'Feature Augmentation', 'green', '-', '^'),
        ('transrf', 'TransRF Ensemble', 'red', '-', '*'),
    ]

    # --- Panel 1: SPL R2 vs sample count (BM target) ---
    ax = axes[0, 0]
    for key, label, color, ls, marker in methods_to_plot:
        n_samples = [r['n_train'] for r in spl_bm_results]
        means = [r[f'{key}_r2_mean'] for r in spl_bm_results]
        stds = [r[f'{key}_r2_std'] for r in spl_bm_results]
        ax.errorbar(n_samples, means, yerr=stds, fmt=f'{marker}-',
                     label=label, color=color, linestyle=ls,
                     linewidth=2, markersize=7, capsize=3)
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('R2 (SPL)')
    ax.set_title('Panel 1: SPL Transfer Learning (Male BCM -> BM)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0, min(r['source_only_r2_mean'] for r in spl_bm_results) - 0.1), 1.0])

    # --- Panel 2: F0 R2 — consistent subset vs full dataset ---
    ax = axes[0, 1]
    n_full = [r['n_train'] for r in f0_full_results]
    for key, label, color, ls, marker in methods_to_plot:
        vals = [r[f'{key}_r2_mean'] for r in f0_full_results]
        ax.plot(n_full, vals, f'{marker}--', color=color, alpha=0.4,
                linewidth=1, markersize=5)

    if f0_consistent_results is not None:
        for key, label, color, ls, marker in methods_to_plot:
            n_con = [r['n_train'] for r in f0_consistent_results]
            vals = [r[f'{key}_r2_mean'] for r in f0_consistent_results]
            ax.plot(n_con, vals, f'{marker}-', label=f'{label} (consistent)',
                    color=color, linewidth=2, markersize=7)

    # Add a note about dashed = full, solid = consistent
    ax.plot([], [], 'k--', alpha=0.4, label='Full dataset (dashed)')
    ax.plot([], [], 'k-', label='Consistent subset (solid)')

    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('R2 (F0)')
    ax.set_title('Panel 2: F0 Transfer — Consistent Subset vs Full BM')
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)
    y_min = min(r['source_only_r2_mean'] for r in f0_full_results)
    if f0_consistent_results:
        y_min = min(y_min, min(r['source_only_r2_mean'] for r in f0_consistent_results))
    ax.set_ylim([min(-0.5, y_min - 0.1), 1.0])

    # --- Panel 3: Transfer gain comparison — BM vs female BCM (SPL) ---
    ax = axes[1, 0]
    fracs_pct = [r['frac'] * 100 for r in spl_bm_results]
    bar_width = 3.0
    x_pos = np.array(fracs_pct)

    ax.bar(x_pos - bar_width/2, bm_gains, bar_width, label='BM gain',
           color='steelblue', edgecolor='black', alpha=0.8)
    ax.bar(x_pos + bar_width/2, fem_gains, bar_width, label='Female BCM gain',
           color='coral', edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Target Data Fraction (%)')
    ax.set_ylabel('Transfer Gain (R2)')
    ax.set_title('Panel 3: SPL Transfer Gain — BM vs Female BCM')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(fracs_pct)

    # --- Panel 4: Male BCM baseline summary ---
    ax = axes[1, 1]
    categories = ['F0 on BM', 'SPL on BM']
    baseline_vals = [baseline_f0_r2, baseline_spl_r2]
    bar_colors = ['#d62728', '#2ca02c']

    bars = ax.bar(categories, baseline_vals, color=bar_colors,
                  edgecolor='black', alpha=0.8, width=0.5)
    for bar, val in zip(bars, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add best transfer results at 100% as reference
    if spl_bm_results:
        best_spl_100 = max(spl_bm_results[-1][f'{m}_r2_mean']
                           for m in ['target_only', 'residual', 'augmented', 'transrf'])
        ax.bar(['Best SPL\nTransfer (100%)'], [best_spl_100],
               color='#ff7f0e', edgecolor='black', alpha=0.8, width=0.5)
        ax.text(2, best_spl_100 + 0.02, f'{best_spl_100:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    if f0_full_results:
        best_f0_100 = max(f0_full_results[-1][f'{m}_r2_mean']
                          for m in ['target_only', 'residual', 'augmented', 'transrf'])
        ax.bar(['Best F0\nTransfer (100%)'], [best_f0_100],
               color='#9467bd', edgecolor='black', alpha=0.8, width=0.5)
        ax.text(3, best_f0_100 + 0.02, f'{best_f0_100:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('R2')
    ax.set_title('Panel 4: Male BCM Baseline & Best Transfer Performance')
    ax.grid(True, alpha=0.3, axis='y')
    y_lo = min(0, min(baseline_vals) - 0.1)
    ax.set_ylim([y_lo, 1.1])
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'figs', 'spl_transfer_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {fig_path}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nMale BCM Baseline on BM:")
    print(f"  F0  R2 = {baseline_f0_r2:.4f}")
    print(f"  SPL R2 = {baseline_spl_r2:.4f}")
    print(f"\nSPL Transfer (100% BM data):")
    if spl_bm_results:
        r = spl_bm_results[-1]
        print(f"  Target-only: {r['target_only_r2_mean']:.3f}")
        print(f"  TransRF:     {r['transrf_r2_mean']:.3f}")
    print(f"\nAverage SPL Transfer Gain:")
    print(f"  BM:         {np.mean(bm_gains):+.3f}")
    print(f"  Female BCM: {np.mean(fem_gains):+.3f}")
    if f0_consistent_results:
        r_con = f0_consistent_results[-1]
        r_full = f0_full_results[-1]
        print(f"\nF0 Transfer (100%) — TransRF R2:")
        print(f"  Full BM:       {r_full['transrf_r2_mean']:.3f}")
        print(f"  Consistent BM: {r_con['transrf_r2_mean']:.3f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
