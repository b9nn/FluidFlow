"""
Data Efficiency Experiment for TransRF Transfer Learning
Tests how model performance degrades with smaller target datasets
and identifies the most sample-efficient transfer strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import os
import warnings

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
FRACTIONS_TO_TEST = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 5  # Number of runs per fraction for stability
USE_KFOLD_WEIGHTS = True  # Use K-fold CV for ensemble weights
K_FOLDS = 5


# ==================== ADAPTIVE MODEL COMPLEXITY ====================
def get_model_params(n_samples):
    """
    Adjust model complexity based on available training samples.
    Smaller datasets need simpler models to avoid overfitting.
    """
    if n_samples < 100:
        return {
            'n_estimators': 20,
            'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20)
        }
    elif n_samples < 250:
        return {
            'n_estimators': 30,
            'max_depth': 5,
            'min_samples_leaf': 5,
            'min_samples_split': 5
        }
    elif n_samples < 500:
        return {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_leaf': 3,
            'min_samples_split': 3
        }
    elif n_samples < 1000:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 2,
            'min_samples_split': 2
        }
    else:
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2
        }


# ==================== HELPER FUNCTIONS ====================
def get_male_predictions(X, scaler, model):
    """Get predictions from male (source) model"""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def learn_weights_kfold(X, Y, rf_male, scaler_X_male, model_params, k_folds=5):
    """
    Learn ensemble weights using K-fold cross-validation.
    More robust than single validation split for small datasets.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_weights_f0 = []
    all_weights_spl = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        Y_train_fold = Y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        Y_val_fold = Y.iloc[val_idx]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_val_scaled = scaler_X.transform(X_val_fold)

        # Scale targets
        scaler_F0 = StandardScaler()
        scaler_SPL = StandardScaler()

        f0_train_scaled = scaler_F0.fit_transform(Y_train_fold[['F0']])
        spl_train_scaled = scaler_SPL.fit_transform(Y_train_fold[['SPL']])
        Y_train_scaled = np.hstack([f0_train_scaled, spl_train_scaled])

        Y_train_array = Y_train_fold.values
        Y_val_array = Y_val_fold.values

        # --- Model 0: Target-only ---
        rf_target = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **model_params)
        )
        rf_target.fit(X_train_scaled, Y_train_scaled)

        pred_target_scaled = rf_target.predict(X_val_scaled)
        pred_target_val = np.hstack([
            scaler_F0.inverse_transform(pred_target_scaled[:, 0].reshape(-1, 1)),
            scaler_SPL.inverse_transform(pred_target_scaled[:, 1].reshape(-1, 1))
        ])

        # --- Model 2: Residual Correction ---
        pred_male_train = get_male_predictions(X_train_fold, scaler_X_male, rf_male)
        residuals_train = Y_train_array - pred_male_train

        rf_residuals = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **model_params)
        )
        rf_residuals.fit(X_train_fold, residuals_train)

        pred_male_val = get_male_predictions(X_val_fold, scaler_X_male, rf_male)
        residuals_val = rf_residuals.predict(X_val_fold)
        pred_residual_val = pred_male_val + residuals_val

        # --- Model 3: Feature Augmentation ---
        X_train_aug = np.column_stack([X_train_fold.values, pred_male_train])
        X_val_aug = np.column_stack([X_val_fold.values, pred_male_val])

        rf_augmented = MultiOutputRegressor(
            RandomForestRegressor(random_state=42, **model_params)
        )
        rf_augmented.fit(X_train_aug, Y_train_array)
        pred_augmented_val = rf_augmented.predict(X_val_aug)

        # --- Learn weights for this fold ---
        for i, (weights_list, target_name) in enumerate([(all_weights_f0, 'F0'), (all_weights_spl, 'SPL')]):
            stack_val = np.column_stack([
                pred_target_val[:, i],
                pred_residual_val[:, i],
                pred_augmented_val[:, i]
            ])

            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack_val, Y_val_array[:, i])

            weights = reg.coef_
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.array([1/3, 1/3, 1/3])

            weights_list.append(weights)

    # Average weights across folds
    weights_f0 = np.mean(all_weights_f0, axis=0)
    weights_spl = np.mean(all_weights_spl, axis=0)

    return {'F0': weights_f0, 'SPL': weights_spl}


def run_single_experiment(df_full, frac, rf_male, scaler_X_male, random_state=42):
    """
    Run a single transfer learning experiment with given data fraction.
    Returns R² scores for all methods.
    """
    # Sample data
    if frac < 1.0:
        df = df_full.sample(frac=frac, random_state=random_state)
    else:
        df = df_full.copy()

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    n_total = len(df)

    # Need minimum samples for meaningful split
    if n_total < 20:
        return None

    # Split data
    test_size = 0.2
    val_size = 0.2

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=val_size, random_state=random_state
    )

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    # Get adaptive model parameters
    model_params = get_model_params(n_train)

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale targets
    scaler_F0 = StandardScaler()
    scaler_SPL = StandardScaler()

    f0_train_scaled = scaler_F0.fit_transform(Y_train[['F0']])
    spl_train_scaled = scaler_SPL.fit_transform(Y_train[['SPL']])
    Y_train_scaled = np.hstack([f0_train_scaled, spl_train_scaled])

    Y_train_array = Y_train.values
    Y_val_array = Y_val.values
    Y_test_array = Y_test.values

    results = {
        'frac': frac,
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'model_params': model_params
    }

    # ==================== MODEL 0: TARGET-ONLY ====================
    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_target.fit(X_train_scaled, Y_train_scaled)

    pred_target_scaled = rf_target.predict(X_test_scaled)
    pred_target = np.hstack([
        scaler_F0.inverse_transform(pred_target_scaled[:, 0].reshape(-1, 1)),
        scaler_SPL.inverse_transform(pred_target_scaled[:, 1].reshape(-1, 1))
    ])

    results['target_only_f0_r2'] = r2_score(Y_test_array[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_array[:, 1], pred_target[:, 1])

    # ==================== SOURCE-ONLY (MALE) ====================
    pred_male_test = get_male_predictions(X_test, scaler_X_male, rf_male)

    results['source_only_f0_r2'] = r2_score(Y_test_array[:, 0], pred_male_test[:, 0])
    results['source_only_spl_r2'] = r2_score(Y_test_array[:, 1], pred_male_test[:, 1])

    # ==================== MODEL 2: RESIDUAL CORRECTION ====================
    pred_male_train = get_male_predictions(X_train, scaler_X_male, rf_male)
    residuals_train = Y_train_array - pred_male_train

    rf_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_residuals.fit(X_train, residuals_train)

    residuals_test = rf_residuals.predict(X_test)
    pred_residual = pred_male_test + residuals_test

    results['residual_f0_r2'] = r2_score(Y_test_array[:, 0], pred_residual[:, 0])
    results['residual_spl_r2'] = r2_score(Y_test_array[:, 1], pred_residual[:, 1])

    # ==================== MODEL 3: FEATURE AUGMENTATION ====================
    X_train_aug = np.column_stack([X_train.values, pred_male_train])
    X_test_aug = np.column_stack([X_test.values, pred_male_test])

    rf_augmented = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_augmented.fit(X_train_aug, Y_train_array)
    pred_augmented = rf_augmented.predict(X_test_aug)

    results['augmented_f0_r2'] = r2_score(Y_test_array[:, 0], pred_augmented[:, 0])
    results['augmented_spl_r2'] = r2_score(Y_test_array[:, 1], pred_augmented[:, 1])

    # ==================== SIMPLE ENSEMBLE (FIXED WEIGHTS) ====================
    pred_simple = 0.3 * pred_male_test + 0.7 * pred_target

    results['simple_ensemble_f0_r2'] = r2_score(Y_test_array[:, 0], pred_simple[:, 0])
    results['simple_ensemble_spl_r2'] = r2_score(Y_test_array[:, 1], pred_simple[:, 1])

    # ==================== TRANSRF (LEARNED WEIGHTS) ====================
    if USE_KFOLD_WEIGHTS and n_train >= 50:
        # Use K-fold for weight learning
        weights = learn_weights_kfold(
            pd.concat([X_train, X_val]),
            pd.concat([Y_train, Y_val]),
            rf_male, scaler_X_male, model_params, k_folds=min(K_FOLDS, n_train // 10)
        )
    else:
        # Fall back to single validation split
        pred_male_val = get_male_predictions(X_val, scaler_X_male, rf_male)

        pred_target_val_scaled = rf_target.predict(X_val_scaled)
        pred_target_val = np.hstack([
            scaler_F0.inverse_transform(pred_target_val_scaled[:, 0].reshape(-1, 1)),
            scaler_SPL.inverse_transform(pred_target_val_scaled[:, 1].reshape(-1, 1))
        ])

        residuals_val = rf_residuals.predict(X_val)
        pred_residual_val = pred_male_val + residuals_val

        X_val_aug = np.column_stack([X_val.values, pred_male_val])
        pred_augmented_val = rf_augmented.predict(X_val_aug)

        weights = {}
        for i, target_name in enumerate(['F0', 'SPL']):
            stack_val = np.column_stack([
                pred_target_val[:, i],
                pred_residual_val[:, i],
                pred_augmented_val[:, i]
            ])

            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack_val, Y_val_array[:, i])

            w = reg.coef_
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = np.array([1/3, 1/3, 1/3])
            weights[target_name] = w

    # Apply TransRF weights
    pred_transrf = np.zeros_like(Y_test_array)
    for i, target_name in enumerate(['F0', 'SPL']):
        stack_test = np.column_stack([
            pred_target[:, i],
            pred_residual[:, i],
            pred_augmented[:, i]
        ])
        pred_transrf[:, i] = stack_test @ weights[target_name]

    results['transrf_f0_r2'] = r2_score(Y_test_array[:, 0], pred_transrf[:, 0])
    results['transrf_spl_r2'] = r2_score(Y_test_array[:, 1], pred_transrf[:, 1])
    results['weights_f0'] = weights['F0']
    results['weights_spl'] = weights['SPL']

    return results


# ==================== MAIN EXPERIMENT ====================
def main():
    print("="*70)
    print("DATA EFFICIENCY EXPERIMENT FOR TRANSRF TRANSFER LEARNING")
    print("="*70)

    # Load source model
    print("\nLoading pre-trained male (source) model...")
    rf_male = joblib.load('./models/RF_BCM.pkl')
    scaler_X_male = joblib.load('./models/x_scaler_BCM.pkl')
    print("Done.")

    # Load full female dataset
    print("\nLoading female (target) dataset...")
    # Get the script directory and construct path to CSV at project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    csv_path = os.path.join(project_root, 'FemaleBCM.csv')
    df_full = pd.read_csv(csv_path)
    df_full = df_full[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
    df_full = df_full[df_full['ACFL'] > 30]
    print(f"Total samples available: {len(df_full)}")

    # Run experiments
    all_results = []

    for frac in FRACTIONS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of target data")
        print(f"{'='*50}")

        frac_results = []

        for run in range(N_RUNS):
            random_state = 42 + run
            result = run_single_experiment(df_full, frac, rf_male, scaler_X_male, random_state)

            if result is not None:
                frac_results.append(result)
                print(f"  Run {run+1}: n_train={result['n_train']}, "
                      f"Target R²={(result['target_only_f0_r2']+result['target_only_spl_r2'])/2:.3f}, "
                      f"TransRF R²={(result['transrf_f0_r2']+result['transrf_spl_r2'])/2:.3f}")

        if frac_results:
            # Average across runs
            avg_result = {
                'frac': frac,
                'n_samples': int(np.mean([r['n_train'] for r in frac_results])),
            }

            for key in ['source_only', 'target_only', 'residual', 'augmented', 'simple_ensemble', 'transrf']:
                for target in ['f0', 'spl']:
                    values = [r[f'{key}_{target}_r2'] for r in frac_results]
                    avg_result[f'{key}_{target}_r2_mean'] = np.mean(values)
                    avg_result[f'{key}_{target}_r2_std'] = np.std(values)

            # Average weights
            avg_result['weights_f0'] = np.mean([r['weights_f0'] for r in frac_results], axis=0)
            avg_result['weights_spl'] = np.mean([r['weights_spl'] for r in frac_results], axis=0)

            all_results.append(avg_result)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Create results dataframe
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Fraction': f"{r['frac']*100:.0f}%",
            'N_Train': r['n_samples'],
            'Source_R2': (r['source_only_f0_r2_mean'] + r['source_only_spl_r2_mean']) / 2,
            'Target_R2': (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2,
            'Residual_R2': (r['residual_f0_r2_mean'] + r['residual_spl_r2_mean']) / 2,
            'Augmented_R2': (r['augmented_f0_r2_mean'] + r['augmented_spl_r2_mean']) / 2,
            'SimpleEns_R2': (r['simple_ensemble_f0_r2_mean'] + r['simple_ensemble_spl_r2_mean']) / 2,
            'TransRF_R2': (r['transrf_f0_r2_mean'] + r['transrf_spl_r2_mean']) / 2,
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nAverage R² (F0 + SPL) / 2:")
    print(summary_df.to_string(index=False))

    # Find best method per fraction
    print("\n\nBEST METHOD PER DATA SIZE:")
    print("-" * 50)
    methods = ['Source_R2', 'Target_R2', 'Residual_R2', 'Augmented_R2', 'SimpleEns_R2', 'TransRF_R2']
    for _, row in summary_df.iterrows():
        best_method = max(methods, key=lambda m: row[m])
        print(f"  {row['Fraction']:>5} ({row['N_Train']:>4} samples): {best_method.replace('_R2', ''):<12} (R²={row[best_method]:.3f})")

    # Print learned weights
    print("\n\nLEARNED TRANSRF WEIGHTS (Target-only, Residual, Augmented):")
    print("-" * 50)
    for r in all_results:
        print(f"  {r['frac']*100:>5.0f}%: F0=[{r['weights_f0'][0]:.2f}, {r['weights_f0'][1]:.2f}, {r['weights_f0'][2]:.2f}]  "
              f"SPL=[{r['weights_spl'][0]:.2f}, {r['weights_spl'][1]:.2f}, {r['weights_spl'][2]:.2f}]")

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transfer Learning Data Efficiency Analysis', fontsize=14, fontweight='bold')

    fracs = [r['frac'] * 100 for r in all_results]
    n_samples = [r['n_samples'] for r in all_results]

    # Plot 1: R² vs Data Fraction (Average)
    ax1 = axes[0, 0]
    methods_to_plot = [
        ('source_only', 'Source Only (Male)', 'gray', '--'),
        ('target_only', 'Target Only (Female)', 'blue', '-'),
        ('residual', 'Residual Correction', 'purple', '-'),
        ('augmented', 'Feature Augmentation', 'green', '-'),
        ('transrf', 'TransRF Ensemble', 'red', '-'),
    ]

    for key, label, color, linestyle in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 for r in all_results]
        ax1.plot(fracs, avg_r2, marker='o', label=label, color=color, linestyle=linestyle, linewidth=2)

    ax1.set_xlabel('Target Data Fraction (%)')
    ax1.set_ylabel('Average R² (F0 + SPL)')
    ax1.set_title('Model Performance vs Target Data Size')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: R² vs Actual Sample Count
    ax2 = axes[0, 1]
    for key, label, color, linestyle in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 for r in all_results]
        ax2.plot(n_samples, avg_r2, marker='o', label=label, color=color, linestyle=linestyle, linewidth=2)

    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('Average R² (F0 + SPL)')
    ax2.set_title('Model Performance vs Sample Count')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    # Use linear scale with actual sample numbers as ticks
    ax2.set_xticks(n_samples)
    ax2.set_xticklabels([str(n) for n in n_samples], rotation=45, ha='right')

    # Plot 3: Learned Weights Evolution (F0)
    ax3 = axes[1, 0]
    weights_target = [r['weights_f0'][0] for r in all_results]
    weights_residual = [r['weights_f0'][1] for r in all_results]
    weights_augmented = [r['weights_f0'][2] for r in all_results]

    ax3.stackplot(fracs, weights_target, weights_residual, weights_augmented,
                  labels=['Target-only', 'Residual', 'Augmented'],
                  colors=['blue', 'purple', 'green'], alpha=0.7)
    ax3.set_xlabel('Target Data Fraction (%)')
    ax3.set_ylabel('Weight')
    ax3.set_title('TransRF Learned Weights for F0')
    ax3.legend(loc='upper left')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learned Weights Evolution (SPL)
    ax4 = axes[1, 1]
    weights_target = [r['weights_spl'][0] for r in all_results]
    weights_residual = [r['weights_spl'][1] for r in all_results]
    weights_augmented = [r['weights_spl'][2] for r in all_results]

    ax4.stackplot(fracs, weights_target, weights_residual, weights_augmented,
                  labels=['Target-only', 'Residual', 'Augmented'],
                  colors=['blue', 'purple', 'green'], alpha=0.7)
    ax4.set_xlabel('Target Data Fraction (%)')
    ax4.set_ylabel('Weight')
    ax4.set_title('TransRF Learned Weights for SPL')
    ax4.legend(loc='upper left')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('./transfer-figs', exist_ok=True)
    plt.savefig('./transfer-figs/data_efficiency_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved to ./transfer-figs/data_efficiency_analysis.png")

    # ==================== DETAILED F0 vs SPL COMPARISON ====================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('F0 vs SPL Performance Comparison', fontsize=14, fontweight='bold')

    # F0
    ax = axes2[0]
    for key, label, color, linestyle in methods_to_plot:
        r2_values = [r[f'{key}_f0_r2_mean'] for r in all_results]
        ax.plot(fracs, r2_values, marker='o', label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Target Data Fraction (%)')
    ax.set_ylabel('R² for F0')
    ax.set_title('F0 Prediction Performance')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # SPL
    ax = axes2[1]
    for key, label, color, linestyle in methods_to_plot:
        r2_values = [r[f'{key}_spl_r2_mean'] for r in all_results]
        ax.plot(fracs, r2_values, marker='o', label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Target Data Fraction (%)')
    ax.set_ylabel('R² for SPL')
    ax.set_title('SPL Prediction Performance')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('./transfer-figs/f0_vs_spl_efficiency.png', dpi=150, bbox_inches='tight')
    print("Saved to ./transfer-figs/f0_vs_spl_efficiency.png")

    plt.show()

    # ==================== RECOMMENDATIONS ====================
    print("\n\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find crossover points
    for r in all_results:
        target_r2 = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        transrf_r2 = (r['transrf_f0_r2_mean'] + r['transrf_spl_r2_mean']) / 2
        residual_r2 = (r['residual_f0_r2_mean'] + r['residual_spl_r2_mean']) / 2

        if target_r2 > transrf_r2 * 0.98:  # Within 2%
            print(f"\nAt {r['frac']*100:.0f}% data ({r['n_samples']} samples):")
            print(f"  Target-only model is competitive with TransRF")
            print(f"  Consider using simpler target-only approach")
            break

    print("\n\nGENERAL GUIDELINES:")
    print("-" * 50)
    print("  < 50 samples:   Use source model only or simple ensemble")
    print("  50-200 samples: Residual correction is most sample-efficient")
    print("  200-500 samples: TransRF ensemble provides best balance")
    print("  > 500 samples:  Target-only model becomes competitive")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
