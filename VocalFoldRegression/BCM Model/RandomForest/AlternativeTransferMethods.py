"""
Alternative Transfer Learning Methods for Male → Female Vocal Fold Regression

Since traditional transfer learning doesn't help (domains too different),
this script implements alternative approaches:

1. Combined Training - Train one model on both datasets with gender indicator
2. Feature Normalization - Normalize by gender-specific ranges
3. Instance Weighting - Only use similar male samples
4. Feature Importance Transfer - Use male data to guide feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
TARGET_FRACTIONS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
N_RUNS = 3
RANDOM_STATE = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MALE_CSV = os.path.join(PROJECT_ROOT, 'MaleBCM.csv')
FEMALE_CSV = os.path.join(PROJECT_ROOT, 'FemaleBCM.csv')


# ==================== DATA LOADING ====================
def load_data(csv_path):
    """Load and prepare dataset"""
    df = pd.read_csv(csv_path)
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
    df = df[df['ACFL'] > 30]
    return df


# ==================== METHOD 0: BASELINE (Target-Only) ====================
def train_target_only(X_train, Y_train, X_test, n_train):
    """Baseline: Train only on female data"""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if n_train < 50:
        params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 3}
    elif n_train < 100:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1}

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    )
    rf.fit(X_train_scaled, Y_train.values)

    return rf.predict(X_test_scaled)


# ==================== METHOD 1: COMBINED TRAINING ====================
def train_combined(X_male, Y_male, X_train_female, Y_train_female, X_test_female, n_female):
    """
    Train ONE model on both male and female data with gender indicator.
    The model learns gender-conditional patterns.
    """

    # Add gender indicator
    X_male_with_gender = X_male.copy()
    X_male_with_gender['is_female'] = 0

    X_train_female_with_gender = X_train_female.copy()
    X_train_female_with_gender['is_female'] = 1

    X_test_female_with_gender = X_test_female.copy()
    X_test_female_with_gender['is_female'] = 1

    # Sample male data to balance (use more male data for small female samples)
    # Rule: use up to 10x the female data from male
    n_male_to_use = min(len(X_male), max(1000, n_female * 10))
    male_indices = np.random.RandomState(RANDOM_STATE).choice(
        len(X_male), size=n_male_to_use, replace=False
    )
    X_male_sample = X_male_with_gender.iloc[male_indices]
    Y_male_sample = Y_male.iloc[male_indices]

    # Combine datasets
    X_combined = pd.concat([X_male_sample, X_train_female_with_gender], ignore_index=True)
    Y_combined = pd.concat([Y_male_sample, Y_train_female], ignore_index=True)

    # Scale features
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_test_scaled = scaler.transform(X_test_female_with_gender)

    # Train model
    n_total = len(X_combined)
    if n_total < 200:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1}

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    )
    rf.fit(X_combined_scaled, Y_combined.values)

    return rf.predict(X_test_scaled)


# ==================== METHOD 2: FEATURE NORMALIZATION ====================
def train_normalized(X_male, Y_male, X_train_female, Y_train_female, X_test_female, n_female):
    """
    Normalize features by gender-specific ranges before combining.
    This makes the feature distributions more similar.
    """

    # Calculate gender-specific statistics
    male_stats = {
        'mean': X_male.mean(),
        'std': X_male.std()
    }
    female_stats = {
        'mean': X_train_female.mean(),
        'std': X_train_female.std()
    }

    # Normalize each gender to N(0,1) using their own stats
    X_male_norm = (X_male - male_stats['mean']) / male_stats['std']
    X_train_female_norm = (X_train_female - female_stats['mean']) / female_stats['std']
    X_test_female_norm = (X_test_female - female_stats['mean']) / female_stats['std']

    # Also normalize Y values
    Y_male_stats = {'mean': Y_male.mean(), 'std': Y_male.std()}
    Y_female_stats = {'mean': Y_train_female.mean(), 'std': Y_train_female.std()}

    Y_male_norm = (Y_male - Y_male_stats['mean']) / Y_male_stats['std']
    Y_train_female_norm = (Y_train_female - Y_female_stats['mean']) / Y_female_stats['std']

    # Add gender indicator
    X_male_norm = X_male_norm.copy()
    X_male_norm['is_female'] = 0
    X_train_female_norm = X_train_female_norm.copy()
    X_train_female_norm['is_female'] = 1
    X_test_female_norm = X_test_female_norm.copy()
    X_test_female_norm['is_female'] = 1

    # Sample male data
    n_male_to_use = min(len(X_male), max(1000, n_female * 10))
    male_indices = np.random.RandomState(RANDOM_STATE).choice(
        len(X_male), size=n_male_to_use, replace=False
    )
    X_male_sample = X_male_norm.iloc[male_indices]
    Y_male_sample = Y_male_norm.iloc[male_indices]

    # Combine
    X_combined = pd.concat([X_male_sample, X_train_female_norm], ignore_index=True)
    Y_combined = pd.concat([Y_male_sample, Y_train_female_norm], ignore_index=True)

    # Train model (no additional scaling needed - already normalized)
    n_total = len(X_combined)
    if n_total < 200:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1}

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    )
    rf.fit(X_combined.values, Y_combined.values)

    # Predict and denormalize
    pred_norm = rf.predict(X_test_female_norm.values)
    pred = pred_norm * Y_female_stats['std'].values + Y_female_stats['mean'].values

    return pred


# ==================== METHOD 3: INSTANCE WEIGHTING ====================
def train_instance_weighted(X_male, Y_male, X_train_female, Y_train_female, X_test_female, n_female):
    """
    Only use male samples that are similar to female samples.
    Weight male samples by their similarity to the female feature space.
    """

    # Scale features for distance calculation
    scaler = StandardScaler()
    X_female_scaled = scaler.fit_transform(X_train_female)
    X_male_scaled = scaler.transform(X_male)

    # Find how similar each male sample is to female data
    nn = NearestNeighbors(n_neighbors=min(10, len(X_train_female)))
    nn.fit(X_female_scaled)

    distances, _ = nn.kneighbors(X_male_scaled)
    avg_distances = distances.mean(axis=1)

    # Convert distances to weights (closer = higher weight)
    weights = 1 / (1 + avg_distances)
    weights = weights / weights.sum()  # Normalize

    # Select male samples with probability proportional to weight
    n_male_to_use = min(len(X_male), max(500, n_female * 5))

    # Sample with replacement based on weights
    male_indices = np.random.RandomState(RANDOM_STATE).choice(
        len(X_male), size=n_male_to_use, replace=True, p=weights
    )

    X_male_similar = X_male.iloc[male_indices].reset_index(drop=True)
    Y_male_similar = Y_male.iloc[male_indices].reset_index(drop=True)

    # Add gender indicator
    X_male_similar['is_female'] = 0
    X_train_female_copy = X_train_female.copy()
    X_train_female_copy['is_female'] = 1
    X_test_female_copy = X_test_female.copy()
    X_test_female_copy['is_female'] = 1

    # Combine
    X_combined = pd.concat([X_male_similar, X_train_female_copy], ignore_index=True)
    Y_combined = pd.concat([Y_male_similar, Y_train_female], ignore_index=True)

    # Scale and train
    scaler2 = StandardScaler()
    X_combined_scaled = scaler2.fit_transform(X_combined)
    X_test_scaled = scaler2.transform(X_test_female_copy)

    n_total = len(X_combined)
    if n_total < 200:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1}

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    )
    rf.fit(X_combined_scaled, Y_combined.values)

    return rf.predict(X_test_scaled)


# ==================== METHOD 4: FEATURE IMPORTANCE TRANSFER ====================
def train_with_importance_transfer(X_male, Y_male, X_train_female, Y_train_female, X_test_female, n_female):
    """
    Use male data to learn feature importances, then apply as feature weights
    when training on female data.
    """

    # Train model on male data to get feature importances
    scaler_male = StandardScaler()
    X_male_scaled = scaler_male.fit_transform(X_male)

    rf_male = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    )
    rf_male.fit(X_male_scaled, Y_male.values)

    # Get feature importances (average across outputs)
    importances = np.mean([est.feature_importances_ for est in rf_male.estimators_], axis=0)

    # Weight features by importance
    importance_weights = importances / importances.sum() * len(importances)

    # Apply importance weighting to female features
    scaler_female = StandardScaler()
    X_train_scaled = scaler_female.fit_transform(X_train_female)
    X_test_scaled = scaler_female.transform(X_test_female)

    X_train_weighted = X_train_scaled * importance_weights
    X_test_weighted = X_test_scaled * importance_weights

    # Train on weighted features
    if n_female < 50:
        params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 3}
    elif n_female < 100:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}
    else:
        params = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1}

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    )
    rf.fit(X_train_weighted, Y_train_female.values)

    return rf.predict(X_test_weighted)


# ==================== RUN EXPERIMENT ====================
def run_experiment(df_male, df_female, target_frac, random_state=42):
    """Run all methods for a given target fraction"""

    # Prepare male data
    X_male = df_male[['a_CT', 'a_TA', 'PS']]
    Y_male = df_male[['F0', 'SPL']]

    # Sample female data
    if target_frac < 1.0:
        df_female_sample = df_female.sample(frac=target_frac, random_state=random_state)
    else:
        df_female_sample = df_female.copy()

    X_female = df_female_sample[['a_CT', 'a_TA', 'PS']]
    Y_female = df_female_sample[['F0', 'SPL']]

    # Split female data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_female, Y_female, test_size=0.2, random_state=random_state
    )

    n_train = len(X_train)
    Y_test_array = Y_test.values

    results = {
        'frac': target_frac,
        'n_train': n_train
    }

    # Method 0: Baseline (Target-Only)
    pred_baseline = train_target_only(X_train, Y_train, X_test, n_train)
    results['baseline_f0_r2'] = r2_score(Y_test_array[:, 0], pred_baseline[:, 0])
    results['baseline_spl_r2'] = r2_score(Y_test_array[:, 1], pred_baseline[:, 1])

    # Method 1: Combined Training
    pred_combined = train_combined(X_male, Y_male, X_train, Y_train, X_test, n_train)
    results['combined_f0_r2'] = r2_score(Y_test_array[:, 0], pred_combined[:, 0])
    results['combined_spl_r2'] = r2_score(Y_test_array[:, 1], pred_combined[:, 1])

    # Method 2: Feature Normalization
    pred_normalized = train_normalized(X_male, Y_male, X_train, Y_train, X_test, n_train)
    results['normalized_f0_r2'] = r2_score(Y_test_array[:, 0], pred_normalized[:, 0])
    results['normalized_spl_r2'] = r2_score(Y_test_array[:, 1], pred_normalized[:, 1])

    # Method 3: Instance Weighting
    pred_weighted = train_instance_weighted(X_male, Y_male, X_train, Y_train, X_test, n_train)
    results['weighted_f0_r2'] = r2_score(Y_test_array[:, 0], pred_weighted[:, 0])
    results['weighted_spl_r2'] = r2_score(Y_test_array[:, 1], pred_weighted[:, 1])

    # Method 4: Feature Importance Transfer
    pred_importance = train_with_importance_transfer(X_male, Y_male, X_train, Y_train, X_test, n_train)
    results['importance_f0_r2'] = r2_score(Y_test_array[:, 0], pred_importance[:, 0])
    results['importance_spl_r2'] = r2_score(Y_test_array[:, 1], pred_importance[:, 1])

    return results


# ==================== MAIN ====================
def main():
    print("="*70)
    print("ALTERNATIVE TRANSFER LEARNING METHODS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df_male = load_data(MALE_CSV)
    df_female = load_data(FEMALE_CSV)
    print(f"Male samples: {len(df_male)}")
    print(f"Female samples: {len(df_female)}")

    # Run experiments
    all_results = []

    for frac in TARGET_FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of target data")
        print(f"{'='*50}")

        frac_results = []

        for run in range(N_RUNS):
            result = run_experiment(df_male, df_female, frac, random_state=RANDOM_STATE + run)
            frac_results.append(result)

            baseline = (result['baseline_f0_r2'] + result['baseline_spl_r2']) / 2
            combined = (result['combined_f0_r2'] + result['combined_spl_r2']) / 2
            normalized = (result['normalized_f0_r2'] + result['normalized_spl_r2']) / 2
            weighted = (result['weighted_f0_r2'] + result['weighted_spl_r2']) / 2
            importance = (result['importance_f0_r2'] + result['importance_spl_r2']) / 2

            print(f"  Run {run+1}: n={result['n_train']}")
            print(f"    Baseline:   {baseline:.3f}")
            print(f"    Combined:   {combined:.3f} ({combined-baseline:+.3f})")
            print(f"    Normalized: {normalized:.3f} ({normalized-baseline:+.3f})")
            print(f"    Weighted:   {weighted:.3f} ({weighted-baseline:+.3f})")
            print(f"    Importance: {importance:.3f} ({importance-baseline:+.3f})")

        # Average results
        avg_result = {
            'frac': frac,
            'n_train': int(np.mean([r['n_train'] for r in frac_results]))
        }

        for method in ['baseline', 'combined', 'normalized', 'weighted', 'importance']:
            for target in ['f0', 'spl']:
                key = f'{method}_{target}_r2'
                avg_result[f'{key}_mean'] = np.mean([r[key] for r in frac_results])
                avg_result[f'{key}_std'] = np.std([r[key] for r in frac_results])

        all_results.append(avg_result)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY (Average R²)")
    print("="*70)

    print(f"\n{'Frac':<8} {'N':<6} {'Baseline':<10} {'Combined':<10} {'Normal':<10} "
          f"{'Weighted':<10} {'Import':<10} {'Best':<12}")
    print("-" * 86)

    for r in all_results:
        baseline = (r['baseline_f0_r2_mean'] + r['baseline_spl_r2_mean']) / 2
        combined = (r['combined_f0_r2_mean'] + r['combined_spl_r2_mean']) / 2
        normalized = (r['normalized_f0_r2_mean'] + r['normalized_spl_r2_mean']) / 2
        weighted = (r['weighted_f0_r2_mean'] + r['weighted_spl_r2_mean']) / 2
        importance = (r['importance_f0_r2_mean'] + r['importance_spl_r2_mean']) / 2

        methods = {
            'Baseline': baseline,
            'Combined': combined,
            'Normalized': normalized,
            'Weighted': weighted,
            'Importance': importance
        }
        best_name = max(methods, key=methods.get)
        best_val = methods[best_name]
        improvement = best_val - baseline

        print(f"{r['frac']*100:>5.0f}%  {r['n_train']:<6} {baseline:<10.3f} {combined:<10.3f} "
              f"{normalized:<10.3f} {weighted:<10.3f} {importance:<10.3f} "
              f"{best_name} ({improvement:+.3f})")

    # ==================== ANALYSIS ====================
    print("\n\n" + "="*70)
    print("ANALYSIS: WHICH METHOD HELPS MOST AT EACH DATA SIZE?")
    print("="*70)

    for r in all_results:
        baseline = (r['baseline_f0_r2_mean'] + r['baseline_spl_r2_mean']) / 2

        improvements = {
            'Combined Training': (r['combined_f0_r2_mean'] + r['combined_spl_r2_mean']) / 2 - baseline,
            'Feature Normalization': (r['normalized_f0_r2_mean'] + r['normalized_spl_r2_mean']) / 2 - baseline,
            'Instance Weighting': (r['weighted_f0_r2_mean'] + r['weighted_spl_r2_mean']) / 2 - baseline,
            'Feature Importance': (r['importance_f0_r2_mean'] + r['importance_spl_r2_mean']) / 2 - baseline,
        }

        print(f"\n{r['frac']*100:.0f}% data ({r['n_train']} samples):")
        for method, imp in sorted(improvements.items(), key=lambda x: -x[1]):
            status = "✓ HELPS" if imp > 0.01 else ("≈ SAME" if imp > -0.01 else "✗ HURTS")
            print(f"  {method:<25} {imp:+.3f}  {status}")

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Alternative Transfer Learning Methods Comparison', fontsize=14, fontweight='bold')

    n_samples = [r['n_train'] for r in all_results]

    methods_to_plot = [
        ('baseline', 'Target-Only (Baseline)', 'blue', '-'),
        ('combined', 'Combined Training', 'green', '-'),
        ('normalized', 'Feature Normalization', 'orange', '-'),
        ('weighted', 'Instance Weighting', 'purple', '-'),
        ('importance', 'Importance Transfer', 'red', '--'),
    ]

    # Plot 1: R² vs Sample Count
    ax1 = axes[0]
    for key, label, color, ls in methods_to_plot:
        r2_vals = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 for r in all_results]
        ax1.plot(n_samples, r2_vals, marker='o', label=label, color=color,
                 linestyle=ls, linewidth=2, markersize=8)

    ax1.set_xlabel('Training Samples')
    ax1.set_ylabel('Average R²')
    ax1.set_title('Model Performance vs Sample Count')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.set_xticks(n_samples)
    ax1.set_xticklabels([str(n) for n in n_samples], rotation=45, ha='right')

    # Plot 2: Improvement over Baseline
    ax2 = axes[1]

    x = np.arange(len(n_samples))
    width = 0.2

    for i, (key, label, color, _) in enumerate(methods_to_plot[1:]):  # Skip baseline
        improvements = [
            ((r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2) -
            ((r['baseline_f0_r2_mean'] + r['baseline_spl_r2_mean']) / 2)
            for r in all_results
        ]
        ax2.bar(x + i*width, improvements, width, label=label, color=color, alpha=0.7)

    ax2.set_xlabel('Training Samples')
    ax2.set_ylabel('R² Improvement over Baseline')
    ax2.set_title('Does Each Method Help?')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([str(n) for n in n_samples])
    ax2.legend(loc='upper left', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    os.makedirs('./transfer-figs', exist_ok=True)
    plt.savefig('./transfer-figs/alternative_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved to ./transfer-figs/alternative_methods_comparison.png")

    plt.show()

    # ==================== RECOMMENDATIONS ====================
    print("\n\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find which method helps most on average
    avg_improvements = {}
    for method in ['combined', 'normalized', 'weighted', 'importance']:
        total_imp = 0
        for r in all_results:
            baseline = (r['baseline_f0_r2_mean'] + r['baseline_spl_r2_mean']) / 2
            method_r2 = (r[f'{method}_f0_r2_mean'] + r[f'{method}_spl_r2_mean']) / 2
            total_imp += method_r2 - baseline
        avg_improvements[method] = total_imp / len(all_results)

    best_method = max(avg_improvements, key=avg_improvements.get)
    best_imp = avg_improvements[best_method]

    method_names = {
        'combined': 'Combined Training (gender indicator)',
        'normalized': 'Feature Normalization',
        'weighted': 'Instance Weighting',
        'importance': 'Feature Importance Transfer'
    }

    if best_imp > 0.01:
        print(f"\n✓ BEST METHOD: {method_names[best_method]}")
        print(f"  Average improvement: {best_imp:+.3f} R²")
        print("\n  This method works because it:")
        if best_method == 'combined':
            print("  - Lets the model learn gender-specific patterns")
            print("  - Uses male data to learn general feature relationships")
            print("  - The gender indicator tells it which patterns to apply")
        elif best_method == 'normalized':
            print("  - Reduces domain gap by normalizing to common distribution")
            print("  - Allows knowledge transfer despite different scales")
        elif best_method == 'weighted':
            print("  - Only uses male samples similar to female data")
            print("  - Reduces negative transfer from dissimilar samples")
        elif best_method == 'importance':
            print("  - Uses male data to identify which features matter")
            print("  - Guides female model to focus on important features")
    else:
        print("\n⚠️  NO METHOD SIGNIFICANTLY IMPROVES over target-only baseline.")
        print("   The male and female domains may be fundamentally incompatible.")
        print("   Recommendation: Focus on collecting more female data instead.")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
