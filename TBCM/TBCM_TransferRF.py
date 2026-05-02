"""
Transfer Learning: BCM -> TBCM (Triangular Body-Cover Model)

Uses BCM Random Forest (trained inline on full BCM dataset, ~54k samples)
as source model to boost performance on TBCM target data (~43k samples).
BCM and TBCM share the same lumped-element physics family, so transfer
should be much more effective than BCM->Beam-Membrane.

Shared features: a_CT, a_TA, PS -> F0, SPL

Methods:
  - Source Only:         BCM model applied directly (zero-shot)
  - Target Only:         RF trained on TBCM data alone
  - Residual Correction: Learn residuals between BCM predictions and TBCM truth
  - Feature Augmentation: Augment inputs with BCM model predictions
  - Simple Ensemble:     Fixed 0.3/0.7 weighting of source/target
  - TransRF Ensemble:    Learned per-output weights via LinearRegression(positive=True)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import json
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 5


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
def get_source_predictions(X, scaler, model):
    """Get predictions from the BCM source model."""
    return model.predict(scaler.transform(X))


# ==================== SINGLE EXPERIMENT ====================
def run_experiment(df_full, frac, rf_source, scaler_X_source, random_state=42):
    """Run one transfer learning experiment at a given data fraction."""
    if frac < 1.0:
        df = df_full.sample(frac=frac, random_state=random_state)
    else:
        df = df_full.copy()

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    if len(df) < 20:
        return None

    # 80/20 train/test split
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    # Further split train into train/val for weight learning
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

    # === SOURCE ONLY (BCM model, zero-shot) ===
    pred_source_test = get_source_predictions(X_test, scaler_X_source, rf_source)
    results['source_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_source_test[:, 0])
    results['source_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_source_test[:, 1])

    # === TARGET ONLY ===
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    results['target_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_target[:, 1])

    # === RESIDUAL CORRECTION ===
    pred_source_train = get_source_predictions(X_train, scaler_X_source, rf_source)
    rf_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_residuals.fit(X_train.values, Y_train_arr - pred_source_train)
    pred_residual = pred_source_test + rf_residuals.predict(X_test.values)

    results['residual_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual[:, 0])
    results['residual_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual[:, 1])

    # === FEATURE AUGMENTATION ===
    X_train_aug = np.column_stack([X_train.values, pred_source_train])
    X_test_aug = np.column_stack([X_test.values, pred_source_test])

    rf_augmented = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_augmented.fit(X_train_aug, Y_train_arr)
    pred_augmented = rf_augmented.predict(X_test_aug)

    results['augmented_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_augmented[:, 0])
    results['augmented_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_augmented[:, 1])

    # === SIMPLE ENSEMBLE (fixed 0.3 source / 0.7 target) ===
    pred_simple = 0.3 * pred_source_test + 0.7 * pred_target
    results['simple_ens_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_simple[:, 0])
    results['simple_ens_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_simple[:, 1])

    # === TRANSRF ENSEMBLE (learned weights on validation split) ===
    X_val_sc = scaler_X.transform(X_val)
    pred_source_val = get_source_predictions(X_val, scaler_X_source, rf_source)

    pred_t_val = rf_target.predict(X_val_sc)
    pred_r_val = pred_source_val + rf_residuals.predict(X_val.values)
    X_val_aug = np.column_stack([X_val.values, pred_source_val])
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
    results['weights_f0'] = weights['F0'].tolist()
    results['weights_spl'] = weights['SPL'].tolist()

    return results


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("TRANSFER LEARNING: BCM -> TBCM (TransRF)")
    print("=" * 70)

    # ---- Load source (BCM) data and train inline ----
    print("\nLoading BCM source data...")
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'), index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    print(f"  BCM samples: {len(df_bcm)}")
    print(f"  BCM columns: {list(df_bcm.columns)}")
    print(f"  BCM F0 range: [{df_bcm['F0'].min():.1f}, {df_bcm['F0'].max():.1f}]")
    print(f"  BCM SPL range: [{df_bcm['SPL'].min():.1f}, {df_bcm['SPL'].max():.1f}]")

    X_source = df_bcm[['a_CT', 'a_TA', 'PS']]
    Y_source = df_bcm[['F0', 'SPL']]

    scaler_X_source = StandardScaler()
    X_source_sc = scaler_X_source.fit_transform(X_source)

    print("\nTraining BCM source RF model (300 trees, full data)...")
    rf_source = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(X_source_sc, Y_source)
    pred_source_self = rf_source.predict(X_source_sc)
    r2_f0 = r2_score(Y_source['F0'], pred_source_self[:, 0])
    r2_spl = r2_score(Y_source['SPL'], pred_source_self[:, 1])
    print(f"  Source self-R2: F0={r2_f0:.4f}, SPL={r2_spl:.4f}")

    # ---- Load target (TBCM) data ----
    print("\nLoading TBCM target data...")
    df_tbcm = pd.read_csv(os.path.join(script_dir, 'dataset_TBCM.csv'), index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'})
    df_tbcm = df_tbcm.drop(columns=['PL'])  # Not shared with BCM
    print(f"  TBCM samples: {len(df_tbcm)}")
    print(f"  TBCM columns: {list(df_tbcm.columns)}")
    print(f"  TBCM F0 range: [{df_tbcm['F0'].min():.1f}, {df_tbcm['F0'].max():.1f}]")
    print(f"  TBCM SPL range: [{df_tbcm['SPL'].min():.1f}, {df_tbcm['SPL'].max():.1f}]")

    # ---- Sanity check: Spearman correlation ----
    print("\n" + "=" * 70)
    print("SANITY CHECK: Spearman rank correlation (BCM predictions vs TBCM truth)")
    print("=" * 70)
    # Sample TBCM for sanity check
    df_check = df_tbcm.sample(n=min(5000, len(df_tbcm)), random_state=42)
    X_check = df_check[['a_CT', 'a_TA', 'PS']]
    Y_check = df_check[['F0', 'SPL']].values
    pred_check = get_source_predictions(X_check, scaler_X_source, rf_source)

    for i, name in enumerate(['F0', 'SPL']):
        rho, pval = spearmanr(Y_check[:, i], pred_check[:, i])
        r2 = r2_score(Y_check[:, i], pred_check[:, i])
        print(f"  {name}: Spearman rho={rho:.4f} (p={pval:.2e}), R2={r2:.4f}")

    # ---- Run experiments ----
    print("\n" + "=" * 70)
    print("RUNNING TRANSFER EXPERIMENTS")
    print("=" * 70)

    all_results = []

    for frac in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of TBCM data")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df_tbcm, frac, rf_source, scaler_X_source, rs)
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
                    avg[f'{key}_{t}_r2_mean'] = float(np.mean(vals))
                    avg[f'{key}_{t}_r2_std'] = float(np.std(vals))
            avg['weights_f0'] = np.mean(
                [r['weights_f0'] for r in frac_results], axis=0).tolist()
            avg['weights_spl'] = np.mean(
                [r['weights_spl'] for r in frac_results], axis=0).tolist()
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

    # Transfer gain table
    print("\n\nTRANSFER GAIN (method R2 - target-only R2):")
    print("-" * 70)
    header = f"{'Frac':>5} {'N':>6} {'Residual':>10} {'Augmented':>10} {'SimpleEns':>10} {'TransRF':>10}"
    print(header)
    for r in all_results:
        tgt = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        res = (r['residual_f0_r2_mean'] + r['residual_spl_r2_mean']) / 2 - tgt
        aug = (r['augmented_f0_r2_mean'] + r['augmented_spl_r2_mean']) / 2 - tgt
        sim = (r['simple_ens_f0_r2_mean'] + r['simple_ens_spl_r2_mean']) / 2 - tgt
        trf = (r['transrf_f0_r2_mean'] + r['transrf_spl_r2_mean']) / 2 - tgt
        print(f"{r['frac']*100:>4.0f}% {r['n_samples']:>6} "
              f"{res:>+10.4f} {aug:>+10.4f} {sim:>+10.4f} {trf:>+10.4f}")

    # Best method per fraction
    print("\n\nBEST METHOD PER DATA SIZE:")
    print("-" * 60)
    method_cols = ['Source', 'Target', 'Residual', 'Augmented', 'SimpleEns', 'TransRF']
    for _, row in summary_df.iterrows():
        best = max(method_cols, key=lambda m: row[m])
        print(f"  {row['Frac']:>5} ({row['N_Train']:>6} samples): "
              f"{best:<12} (R2={row[best]:.3f})")

    # Learned weights
    print("\n\nLEARNED TRANSRF WEIGHTS (Target, Residual, Augmented):")
    print("-" * 60)
    for r in all_results:
        wf = r['weights_f0']
        ws = r['weights_spl']
        print(f"  {r['frac']*100:>5.0f}%: "
              f"F0=[{wf[0]:.2f}, {wf[1]:.2f}, {wf[2]:.2f}]  "
              f"SPL=[{ws[0]:.2f}, {ws[1]:.2f}, {ws[2]:.2f}]")

    # ==================== SAVE RESULTS ====================
    results_path = os.path.join(script_dir, 'results', 'rf_transfer_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ==================== VISUALIZATION ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')

    n_samples = [r['n_samples'] for r in all_results]
    methods_to_plot = [
        ('source_only', 'Source Only (BCM)', 'gray', '--'),
        ('target_only', 'Target Only (TBCM)', 'blue', '-'),
        ('residual', 'Residual Correction', 'purple', '-'),
        ('augmented', 'Feature Augmentation', 'green', '-'),
        ('simple_ens', 'Simple Ensemble (0.3/0.7)', 'orange', '--'),
        ('transrf', 'TransRF Ensemble', 'red', '-'),
    ]

    # --- Plot 1: R2 vs sample count (F0 and SPL side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Transfer Learning: BCM -> TBCM',
                 fontsize=14, fontweight='bold')

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        stds = [r[f'{key}_f0_r2_std'] for r in all_results]
        axes[0].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[0].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.1, color=color)
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('R2 for F0')
    axes[0].set_title('F0 Prediction')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([min(0, min(r['source_only_f0_r2_mean'] for r in all_results) - 0.1), 1.02])

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        stds = [r[f'{key}_spl_r2_std'] for r in all_results]
        axes[1].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[1].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.1, color=color)
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('R2 for SPL')
    axes[1].set_title('SPL Prediction')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([min(0, min(r['source_only_spl_r2_mean'] for r in all_results) - 0.1), 1.02])

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'tbcm_rf_transfer.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: Learned weights evolution ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('TransRF Learned Weights vs Data Size (BCM->TBCM)',
                  fontsize=14, fontweight='bold')
    fracs_pct = [r['frac'] * 100 for r in all_results]

    for ax, target, title in [(ax3, 'f0', 'F0'), (ax4, 'spl', 'SPL')]:
        w_key = f'weights_{target}'
        w_t = [r[w_key][0] for r in all_results]
        w_r = [r[w_key][1] for r in all_results]
        w_a = [r[w_key][2] for r in all_results]
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
    fig2_path = os.path.join(figs_dir, 'tbcm_rf_weights.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("RF TRANSFER EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
