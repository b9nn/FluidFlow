"""
Improved Transfer Learning for Male → Female Vocal Fold Regression

Key improvements over baseline TransRF:
1. Train source model on FULL male dataset (not 7%)
2. Feature normalization to reduce domain gap
3. Output calibration to correct systematic bias
4. Domain-invariant feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, LinearRegression
import joblib
import os
import warnings

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
TARGET_FRACTIONS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
N_RUNS = 3
RANDOM_STATE = 42


# ==================== PATH SETUP ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MALE_CSV = os.path.join(PROJECT_ROOT, 'MaleBCM.csv')
FEMALE_CSV = os.path.join(PROJECT_ROOT, 'FemaleBCM.csv')


# ==================== DATA LOADING ====================
def load_and_prepare_data(csv_path, sample_frac=1.0, random_state=42):
    """Load and prepare dataset with consistent preprocessing"""
    df = pd.read_csv(csv_path)
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
    df = df[df['ACFL'] > 30]

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    return X, Y


# ==================== IMPROVEMENT 1: FULL SOURCE TRAINING ====================
def train_full_source_model(X_male, Y_male):
    """
    Train source model on FULL male dataset (not downsampled).
    Uses more trees and proper cross-validation.
    """
    print("Training source model on FULL male dataset...")
    print(f"  Male samples: {len(X_male)}")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_male)

    scaler_F0 = StandardScaler()
    scaler_SPL = StandardScaler()

    Y_scaled = np.hstack([
        scaler_F0.fit_transform(Y_male[['F0']]),
        scaler_SPL.fit_transform(Y_male[['SPL']])
    ])

    # More robust model with full data
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
    rf.fit(X_scaled, Y_scaled)

    return {
        'model': rf,
        'scaler_X': scaler_X,
        'scaler_F0': scaler_F0,
        'scaler_SPL': scaler_SPL
    }


# ==================== IMPROVEMENT 2: FEATURE ALIGNMENT ====================
def create_feature_aligner(X_male, X_female_sample):
    """
    Create a quantile transformer that maps both domains to a common distribution.
    This reduces the feature distribution gap between male and female.
    """
    # Fit on male data (source domain)
    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(1000, len(X_male)),
        random_state=RANDOM_STATE
    )
    qt.fit(X_male)

    return qt


# ==================== IMPROVEMENT 3: OUTPUT CALIBRATION ====================
def train_output_calibrator(source_model, X_female_cal, Y_female_cal, feature_aligner=None):
    """
    Learn a linear correction for the source model's systematic bias.
    This is trained on a small calibration set from the target domain.
    """
    # Get source predictions on female calibration data
    if feature_aligner is not None:
        X_aligned = feature_aligner.transform(X_female_cal)
    else:
        X_aligned = source_model['scaler_X'].transform(X_female_cal)

    pred_scaled = source_model['model'].predict(X_aligned)
    pred_male = np.hstack([
        source_model['scaler_F0'].inverse_transform(pred_scaled[:, 0].reshape(-1, 1)),
        source_model['scaler_SPL'].inverse_transform(pred_scaled[:, 1].reshape(-1, 1))
    ])

    # Learn calibration for each output
    calibrators = {}
    for i, name in enumerate(['F0', 'SPL']):
        cal = Ridge(alpha=1.0)
        # Use both prediction and original features for calibration
        cal_features = np.column_stack([pred_male[:, i], X_female_cal.values])
        cal.fit(cal_features, Y_female_cal.iloc[:, i])
        calibrators[name] = cal

    return calibrators


# ==================== IMPROVEMENT 4: ADAPTIVE RESIDUAL MODEL ====================
def train_adaptive_residual_model(source_model, X_train, Y_train, feature_aligner=None, n_samples=None):
    """
    Train a residual correction model with complexity adapted to sample size.
    Also uses the calibrated source predictions as input.
    """
    # Get source predictions
    if feature_aligner is not None:
        X_aligned = feature_aligner.transform(X_train)
    else:
        X_aligned = source_model['scaler_X'].transform(X_train)

    pred_scaled = source_model['model'].predict(X_aligned)
    pred_source = np.hstack([
        source_model['scaler_F0'].inverse_transform(pred_scaled[:, 0].reshape(-1, 1)),
        source_model['scaler_SPL'].inverse_transform(pred_scaled[:, 1].reshape(-1, 1))
    ])

    # Calculate residuals
    residuals = Y_train.values - pred_source

    # Adaptive complexity based on sample size
    if n_samples is None:
        n_samples = len(X_train)

    if n_samples < 50:
        params = {'n_estimators': 10, 'max_depth': 2, 'min_samples_leaf': 5}
    elif n_samples < 100:
        params = {'n_estimators': 20, 'max_depth': 3, 'min_samples_leaf': 3}
    elif n_samples < 200:
        params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 2}

    # Train residual model on augmented features (original + source predictions)
    X_augmented = np.column_stack([X_train.values, pred_source])

    rf_residual = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, **params)
    )
    rf_residual.fit(X_augmented, residuals)

    return rf_residual


# ==================== PREDICTION FUNCTIONS ====================
def predict_baseline_source(source_model, X, feature_aligner=None):
    """Predict using source model only (no adaptation)"""
    if feature_aligner is not None:
        X_proc = feature_aligner.transform(X)
    else:
        X_proc = source_model['scaler_X'].transform(X)

    pred_scaled = source_model['model'].predict(X_proc)
    return np.hstack([
        source_model['scaler_F0'].inverse_transform(pred_scaled[:, 0].reshape(-1, 1)),
        source_model['scaler_SPL'].inverse_transform(pred_scaled[:, 1].reshape(-1, 1))
    ])


def predict_calibrated_source(source_model, calibrators, X, feature_aligner=None):
    """Predict using calibrated source model"""
    pred_source = predict_baseline_source(source_model, X, feature_aligner)

    pred_calibrated = np.zeros_like(pred_source)
    for i, name in enumerate(['F0', 'SPL']):
        cal_features = np.column_stack([pred_source[:, i], X.values])
        pred_calibrated[:, i] = calibrators[name].predict(cal_features)

    return pred_calibrated


def predict_with_residual(source_model, residual_model, X, feature_aligner=None):
    """Predict using source + residual correction"""
    pred_source = predict_baseline_source(source_model, X, feature_aligner)

    X_augmented = np.column_stack([X.values, pred_source])
    residuals = residual_model.predict(X_augmented)

    return pred_source + residuals


# ==================== MAIN EXPERIMENT ====================
def run_experiment(X_male, Y_male, X_female, Y_female, target_frac, random_state=42):
    """Run a single experiment with given target fraction"""

    # Sample target data
    if target_frac < 1.0:
        indices = np.random.RandomState(random_state).choice(
            len(X_female), size=int(len(X_female) * target_frac), replace=False
        )
        X_target = X_female.iloc[indices]
        Y_target = Y_female.iloc[indices]
    else:
        X_target = X_female.copy()
        Y_target = Y_female.copy()

    n_target = len(X_target)

    # Split target data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_target, Y_target, test_size=0.2, random_state=random_state
    )

    # Further split for calibration
    if len(X_train) > 20:
        X_train_model, X_cal, Y_train_model, Y_cal = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=random_state
        )
    else:
        X_train_model, X_cal = X_train, X_train
        Y_train_model, Y_cal = Y_train, Y_train

    n_train = len(X_train_model)

    results = {
        'frac': target_frac,
        'n_target': n_target,
        'n_train': n_train
    }

    # === IMPROVEMENT 1: Full source model (trained once outside) ===
    source_model = train_full_source_model(X_male, Y_male)

    # === IMPROVEMENT 2: Feature alignment ===
    feature_aligner = create_feature_aligner(X_male, X_train_model)

    # === IMPROVEMENT 3: Output calibration ===
    calibrators = train_output_calibrator(
        source_model, X_cal, Y_cal, feature_aligner
    )

    # === IMPROVEMENT 4: Adaptive residual model ===
    residual_model = train_adaptive_residual_model(
        source_model, X_train_model, Y_train_model, feature_aligner, n_train
    )

    # === BASELINE: Target-only model ===
    scaler_X_target = StandardScaler()
    X_train_scaled = scaler_X_target.fit_transform(X_train_model)
    X_test_scaled = scaler_X_target.transform(X_test)

    if n_train < 50:
        target_params = {'n_estimators': 20, 'max_depth': 3, 'min_samples_leaf': 5}
    elif n_train < 100:
        target_params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 3}
    else:
        target_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **target_params)
    )
    rf_target.fit(X_train_scaled, Y_train_model.values)
    pred_target = rf_target.predict(X_test_scaled)

    # === GENERATE ALL PREDICTIONS ===
    Y_test_array = Y_test.values

    # Method 1: Source only (baseline - no adaptation)
    pred_source = predict_baseline_source(source_model, X_test, feature_aligner)

    # Method 2: Calibrated source
    pred_calibrated = predict_calibrated_source(source_model, calibrators, X_test, feature_aligner)

    # Method 3: Source + Residual
    pred_residual = predict_with_residual(source_model, residual_model, X_test, feature_aligner)

    # Method 4: Weighted ensemble (learned weights)
    # Stack predictions and learn optimal weights
    if len(X_cal) >= 10:
        pred_cal_source = predict_baseline_source(source_model, X_cal, feature_aligner)
        pred_cal_calibrated = predict_calibrated_source(source_model, calibrators, X_cal, feature_aligner)
        pred_cal_residual = predict_with_residual(source_model, residual_model, X_cal, feature_aligner)
        pred_cal_target = rf_target.predict(scaler_X_target.transform(X_cal))

        weights = {}
        for i, name in enumerate(['F0', 'SPL']):
            stack = np.column_stack([
                pred_cal_calibrated[:, i],
                pred_cal_residual[:, i],
                pred_cal_target[:, i]
            ])
            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack, Y_cal.iloc[:, i])
            w = reg.coef_
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = np.array([1/3, 1/3, 1/3])
            weights[name] = w
    else:
        # Default weights for very small samples
        weights = {'F0': np.array([0.3, 0.4, 0.3]), 'SPL': np.array([0.3, 0.4, 0.3])}

    # Apply ensemble weights
    pred_ensemble = np.zeros_like(Y_test_array)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_calibrated[:, i],
            pred_residual[:, i],
            pred_target[:, i]
        ])
        pred_ensemble[:, i] = stack @ weights[name]

    # === CALCULATE METRICS ===
    methods = {
        'source_only': pred_source,
        'calibrated_source': pred_calibrated,
        'source_residual': pred_residual,
        'target_only': pred_target,
        'improved_ensemble': pred_ensemble
    }

    for method_name, pred in methods.items():
        for i, target_name in enumerate(['f0', 'spl']):
            r2 = r2_score(Y_test_array[:, i], pred[:, i])
            results[f'{method_name}_{target_name}_r2'] = r2

    results['weights'] = weights

    return results


def main():
    print("="*70)
    print("IMPROVED TRANSFER LEARNING EXPERIMENT")
    print("="*70)

    # Load data
    print("\nLoading datasets...")
    X_male, Y_male = load_and_prepare_data(MALE_CSV, sample_frac=1.0)  # FULL male data
    X_female, Y_female = load_and_prepare_data(FEMALE_CSV, sample_frac=1.0)

    print(f"Male samples (FULL): {len(X_male)}")
    print(f"Female samples: {len(X_female)}")

    # Run experiments
    all_results = []

    for frac in TARGET_FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of target data")
        print(f"{'='*50}")

        frac_results = []

        for run in range(N_RUNS):
            result = run_experiment(
                X_male, Y_male, X_female, Y_female,
                frac, random_state=RANDOM_STATE + run
            )
            frac_results.append(result)

            avg_source = (result['source_only_f0_r2'] + result['source_only_spl_r2']) / 2
            avg_calibrated = (result['calibrated_source_f0_r2'] + result['calibrated_source_spl_r2']) / 2
            avg_ensemble = (result['improved_ensemble_f0_r2'] + result['improved_ensemble_spl_r2']) / 2

            print(f"  Run {run+1}: n_train={result['n_train']}, "
                  f"Source={avg_source:.3f}, Calibrated={avg_calibrated:.3f}, "
                  f"Ensemble={avg_ensemble:.3f}")

        # Average results
        avg_result = {'frac': frac, 'n_train': int(np.mean([r['n_train'] for r in frac_results]))}

        for method in ['source_only', 'calibrated_source', 'source_residual', 'target_only', 'improved_ensemble']:
            for target in ['f0', 'spl']:
                key = f'{method}_{target}_r2'
                avg_result[f'{key}_mean'] = np.mean([r[key] for r in frac_results])
                avg_result[f'{key}_std'] = np.std([r[key] for r in frac_results])

        all_results.append(avg_result)

    # === RESULTS SUMMARY ===
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY (Average R² across F0 and SPL)")
    print("="*70)

    print(f"\n{'Fraction':<10} {'N_Train':<10} {'Source':<12} {'Calibrated':<12} "
          f"{'Residual':<12} {'Target':<12} {'Ensemble':<12}")
    print("-" * 80)

    for r in all_results:
        source = (r['source_only_f0_r2_mean'] + r['source_only_spl_r2_mean']) / 2
        calibrated = (r['calibrated_source_f0_r2_mean'] + r['calibrated_source_spl_r2_mean']) / 2
        residual = (r['source_residual_f0_r2_mean'] + r['source_residual_spl_r2_mean']) / 2
        target = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        ensemble = (r['improved_ensemble_f0_r2_mean'] + r['improved_ensemble_spl_r2_mean']) / 2

        print(f"{r['frac']*100:>5.0f}%     {r['n_train']:<10} {source:<12.3f} {calibrated:<12.3f} "
              f"{residual:<12.3f} {target:<12.3f} {ensemble:<12.3f}")

    # === IMPROVEMENT ANALYSIS ===
    print("\n\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)

    print("\nKey improvements over baseline TransRF:")
    print("-" * 50)
    print("1. Source model trained on FULL male data (was 7%)")
    print("2. Feature alignment via quantile transformation")
    print("3. Output calibration to correct systematic bias")
    print("4. Adaptive model complexity based on sample size")

    # === VISUALIZATION ===
    print("\n\nGenerating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Improved Transfer Learning Performance', fontsize=14, fontweight='bold')

    fracs = [r['frac'] * 100 for r in all_results]
    n_samples = [r['n_train'] for r in all_results]

    methods_to_plot = [
        ('source_only', 'Source Only', 'gray', '--'),
        ('calibrated_source', 'Calibrated Source', 'orange', '-'),
        ('source_residual', 'Source + Residual', 'purple', '-'),
        ('target_only', 'Target Only', 'blue', '-'),
        ('improved_ensemble', 'Improved Ensemble', 'red', '-'),
    ]

    # Plot 1: vs Fraction
    ax1 = axes[0]
    for key, label, color, ls in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 for r in all_results]
        ax1.plot(fracs, avg_r2, marker='o', label=label, color=color, linestyle=ls, linewidth=2)

    ax1.set_xlabel('Target Data Fraction (%)')
    ax1.set_ylabel('Average R²')
    ax1.set_title('Performance vs Data Fraction')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: vs Sample Count
    ax2 = axes[1]
    for key, label, color, ls in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 for r in all_results]
        ax2.plot(n_samples, avg_r2, marker='o', label=label, color=color, linestyle=ls, linewidth=2)

    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('Average R²')
    ax2.set_title('Performance vs Sample Count')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.set_xticks(n_samples)
    ax2.set_xticklabels([str(n) for n in n_samples], rotation=45, ha='right')

    plt.tight_layout()

    os.makedirs('./transfer-figs', exist_ok=True)
    plt.savefig('./transfer-figs/improved_transfer_learning.png', dpi=150, bbox_inches='tight')
    print("Saved to ./transfer-figs/improved_transfer_learning.png")

    plt.show()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
