"""
Transfer Learning from Male BCM -> Beam-Membrane Model (TransRF)

Uses the pre-trained male BCM Random Forest (large dataset, ~25k samples)
as a source model to boost performance on the Beam-Membrane target data
(~3300 samples). Tests all transfer methods at varying data fractions.

Methods:
  - Source Only:        Male BCM model applied directly
  - Target Only:        RF trained on Beam-Membrane data alone
  - Residual Correction: Learn residuals between male predictions and true values
  - Feature Augmentation: Augment inputs with male model predictions
  - Simple Ensemble:    Fixed 0.3/0.7 weighting of source/target
  - TransRF Ensemble:   Learned per-output weights over target, residual, augmented
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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


def learn_weights_kfold(X, Y, rf_male, scaler_X_male, model_params, k_folds=5):
    """
    Learn TransRF ensemble weights via K-fold cross-validation.
    Returns per-output weight dicts for [target-only, residual, augmented].
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_weights_f0 = []
    all_weights_spl = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        Y_tr, Y_va = Y.iloc[train_idx], Y.iloc[val_idx]
        Y_tr_arr = Y_tr.values
        Y_va_arr = Y_va.values

        fold_params = get_model_params(len(X_tr))

        # Scale features
        scaler_X = StandardScaler()
        X_tr_sc = scaler_X.fit_transform(X_tr)
        X_va_sc = scaler_X.transform(X_va)

        # --- Target-only ---
        rf_t = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **fold_params))
        rf_t.fit(X_tr_sc, Y_tr_arr)
        pred_t_va = rf_t.predict(X_va_sc)

        # --- Residual correction ---
        pred_male_tr = get_male_predictions(X_tr, scaler_X_male, rf_male)
        pred_male_va = get_male_predictions(X_va, scaler_X_male, rf_male)

        rf_r = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **fold_params))
        rf_r.fit(X_tr.values, Y_tr_arr - pred_male_tr)
        pred_r_va = pred_male_va + rf_r.predict(X_va.values)

        # --- Feature augmentation ---
        X_tr_aug = np.column_stack([X_tr.values, pred_male_tr])
        X_va_aug = np.column_stack([X_va.values, pred_male_va])

        rf_a = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **fold_params))
        rf_a.fit(X_tr_aug, Y_tr_arr)
        pred_a_va = rf_a.predict(X_va_aug)

        # --- Learn weights for this fold ---
        for i, wlist in enumerate([all_weights_f0, all_weights_spl]):
            stack = np.column_stack([
                pred_t_va[:, i], pred_r_va[:, i], pred_a_va[:, i]
            ])
            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack, Y_va_arr[:, i])
            w = reg.coef_
            w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
            wlist.append(w)

    return {
        'F0': np.mean(all_weights_f0, axis=0),
        'SPL': np.mean(all_weights_spl, axis=0),
    }


# ==================== SINGLE EXPERIMENT ====================
def run_experiment(df_full, frac, rf_male, scaler_X_male, random_state=42):
    """Run one transfer learning experiment at a given data fraction."""
    if frac < 1.0:
        df = df_full.sample(frac=frac, random_state=random_state)
    else:
        df = df_full.copy()

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    if len(df) < 20:
        return None

    # Split: train / val / test
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=random_state)

    n_train = len(X_train)
    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_val_arr = Y_val.values
    Y_test_arr = Y_test.values

    results = {
        'frac': frac, 'n_total': len(df),
        'n_train': n_train, 'n_val': len(X_val), 'n_test': len(X_test),
    }

    # === SOURCE ONLY (male BCM model) ===
    pred_male_test = get_male_predictions(X_test, scaler_X_male, rf_male)
    results['source_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_male_test[:, 0])
    results['source_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_male_test[:, 1])

    # === TARGET ONLY ===
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    results['target_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_target[:, 1])

    # === RESIDUAL CORRECTION ===
    pred_male_train = get_male_predictions(X_train, scaler_X_male, rf_male)
    rf_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_residuals.fit(X_train.values, Y_train_arr - pred_male_train)
    pred_residual = pred_male_test + rf_residuals.predict(X_test.values)

    results['residual_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual[:, 0])
    results['residual_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual[:, 1])

    # === FEATURE AUGMENTATION ===
    X_train_aug = np.column_stack([X_train.values, pred_male_train])
    X_test_aug = np.column_stack([X_test.values, pred_male_test])

    rf_augmented = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params))
    rf_augmented.fit(X_train_aug, Y_train_arr)
    pred_augmented = rf_augmented.predict(X_test_aug)

    results['augmented_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_augmented[:, 0])
    results['augmented_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_augmented[:, 1])

    # === SIMPLE ENSEMBLE (fixed 0.3 source / 0.7 target) ===
    pred_simple = 0.3 * pred_male_test + 0.7 * pred_target
    results['simple_ens_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_simple[:, 0])
    results['simple_ens_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_simple[:, 1])

    # === TRANSRF ENSEMBLE (learned weights on validation split) ===
    X_val_sc = scaler_X.transform(X_val)
    pred_male_val = get_male_predictions(X_val, scaler_X_male, rf_male)

    pred_t_val = rf_target.predict(X_val_sc)
    pred_r_val = pred_male_val + rf_residuals.predict(X_val.values)
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

    results['transrf_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_transrf[:, 0])
    results['transrf_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_transrf[:, 1])
    results['weights_f0'] = weights['F0']
    results['weights_spl'] = weights['SPL']

    return results


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("TRANSFER LEARNING: Male BCM -> Beam-Membrane (TransRF)")
    print("=" * 70)

    # Load source model
    print("\nLoading pre-trained male BCM model (source)...")
    rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
    scaler_X_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))
    print("Done.")

    # Load target data
    print("\nLoading Beam-Membrane data (target)...")
    df_full = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
    df_full.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    df_full = df_full[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    n_before = len(df_full)

    # Clean: physical bounds + z-score outliers
    from scipy import stats
    df_full = df_full[(df_full['F0'] >= 50) & (df_full['F0'] <= 500)]
    df_full = df_full[(df_full['SPL'] >= 40) & (df_full['SPL'] <= 140)]
    z_f0 = np.abs(stats.zscore(df_full['F0']))
    z_spl = np.abs(stats.zscore(df_full['SPL']))
    df_full = df_full[(z_f0 < 3) & (z_spl < 3)]
    print(f"Total samples: {n_before} -> {len(df_full)} after cleaning")

    # Run experiments
    FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    N_RUNS = 3

    all_results = []

    for frac in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of target data")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df_full, frac, rf_male, scaler_X_male, rs)
            if result is not None:
                frac_results.append(result)
                avg_tgt = (result['target_only_f0_r2'] + result['target_only_spl_r2']) / 2
                avg_trf = (result['transrf_f0_r2'] + result['transrf_spl_r2']) / 2
                print(f"  Run {run+1}: n_train={result['n_train']}, "
                      f"Target R2={avg_tgt:.3f}, TransRF R2={avg_trf:.3f}")

        if frac_results:
            avg = {'frac': frac,
                   'n_samples': int(np.mean([r['n_train'] for r in frac_results]))}
            for key in ['source_only', 'target_only', 'residual', 'augmented',
                        'simple_ens', 'transrf']:
                for t in ['f0', 'spl']:
                    vals = [r[f'{key}_{t}_r2'] for r in frac_results]
                    avg[f'{key}_{t}_r2_mean'] = np.mean(vals)
                    avg[f'{key}_{t}_r2_std'] = np.std(vals)
            avg['weights_f0'] = np.mean([r['weights_f0'] for r in frac_results], axis=0)
            avg['weights_spl'] = np.mean([r['weights_spl'] for r in frac_results], axis=0)
            all_results.append(avg)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Frac': f"{r['frac']*100:.0f}%",
            'N_Train': r['n_samples'],
            'Source': (r['source_only_f0_r2_mean'] + r['source_only_spl_r2_mean']) / 2,
            'Target': (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2,
            'Residual': (r['residual_f0_r2_mean'] + r['residual_spl_r2_mean']) / 2,
            'Augmented': (r['augmented_f0_r2_mean'] + r['augmented_spl_r2_mean']) / 2,
            'SimpleEns': (r['simple_ens_f0_r2_mean'] + r['simple_ens_spl_r2_mean']) / 2,
            'TransRF': (r['transrf_f0_r2_mean'] + r['transrf_spl_r2_mean']) / 2,
        })
    summary_df = pd.DataFrame(summary_data)
    print("\nAverage R2 (F0 + SPL) / 2:")
    print(summary_df.to_string(index=False, float_format='%.3f'))

    # Best method per fraction
    print("\n\nBEST METHOD PER DATA SIZE:")
    print("-" * 60)
    method_cols = ['Source', 'Target', 'Residual', 'Augmented', 'SimpleEns', 'TransRF']
    for _, row in summary_df.iterrows():
        best = max(method_cols, key=lambda m: row[m])
        print(f"  {row['Frac']:>5} ({row['N_Train']:>4} samples): "
              f"{best:<12} (R2={row[best]:.3f})")

    # Learned weights
    print("\n\nLEARNED TRANSRF WEIGHTS (Target, Residual, Augmented):")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['frac']*100:>5.0f}%: "
              f"F0=[{r['weights_f0'][0]:.2f}, {r['weights_f0'][1]:.2f}, {r['weights_f0'][2]:.2f}]  "
              f"SPL=[{r['weights_spl'][0]:.2f}, {r['weights_spl'][1]:.2f}, {r['weights_spl'][2]:.2f}]")

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating plots...")
    os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

    n_samples = [r['n_samples'] for r in all_results]
    methods_to_plot = [
        ('source_only', 'Source Only (Male BCM)', 'gray', '--'),
        ('target_only', 'Target Only (Beam-Membrane)', 'blue', '-'),
        ('residual', 'Residual Correction', 'purple', '-'),
        ('augmented', 'Feature Augmentation', 'green', '-'),
        ('simple_ens', 'Simple Ensemble (0.3/0.7)', 'orange', '--'),
        ('transrf', 'TransRF Ensemble', 'red', '-'),
    ]

    # --- Plot 1: R2 vs sample count (F0 and SPL side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Transfer Learning: Male BCM -> Beam-Membrane',
                 fontsize=14, fontweight='bold')

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        axes[0].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('R2 for F0')
    axes[0].set_title('F0 Prediction')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([min(0, min(r['source_only_f0_r2_mean'] for r in all_results) - 0.1), 1])

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        axes[1].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('R2 for SPL')
    axes[1].set_title('SPL Prediction')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([min(0, min(r['source_only_spl_r2_mean'] for r in all_results) - 0.1), 1])

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'figs', 'transfer_learning_f0_spl.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {fig_path}")

    # --- Plot 2: Combined average R2 ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Transfer Learning Data Efficiency: Average R2',
                  fontsize=14, fontweight='bold')
    for key, label, color, ls in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
                  for r in all_results]
        ax2.plot(n_samples, avg_r2, 'o-', label=label, color=color,
                 linestyle=ls, linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_ylabel('Average R2 (F0 + SPL) / 2', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    fig2_path = os.path.join(script_dir, 'figs', 'transfer_learning_combined.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {fig2_path}")

    # --- Plot 3: Learned weights evolution ---
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('TransRF Learned Weights vs Data Size', fontsize=14, fontweight='bold')
    fracs_pct = [r['frac'] * 100 for r in all_results]

    for ax, target, title in [(ax3, 'f0', 'F0'), (ax4, 'spl', 'SPL')]:
        w_t = [r[f'weights_{target}'][0] for r in all_results]
        w_r = [r[f'weights_{target}'][1] for r in all_results]
        w_a = [r[f'weights_{target}'][2] for r in all_results]
        ax.stackplot(fracs_pct, w_t, w_r, w_a,
                     labels=['Target-only', 'Residual', 'Augmented'],
                     colors=['blue', 'purple', 'green'], alpha=0.7)
        ax.set_xlabel('Target Data Fraction (%)')
        ax.set_ylabel('Weight')
        ax.set_title(f'Weights for {title}')
        ax.legend(loc='upper left')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(script_dir, 'figs', 'transfer_learning_weights.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {fig3_path}")

    plt.show()

    # ==================== RECOMMENDATIONS ====================
    print("\n\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    for r in all_results:
        target_r2 = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        transrf_r2 = (r['transrf_f0_r2_mean'] + r['transrf_spl_r2_mean']) / 2
        if target_r2 > transrf_r2 * 0.98:
            print(f"\nAt {r['frac']*100:.0f}% data ({r['n_samples']} samples):")
            print(f"  Target-only is competitive with TransRF")
            print(f"  Transfer learning benefit diminishes here")
            break

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
