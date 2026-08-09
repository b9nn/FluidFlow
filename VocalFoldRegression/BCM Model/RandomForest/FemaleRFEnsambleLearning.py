"""
Enhanced Transfer Learning with TransRF Methodology
Integrates multiple transfer learning strategies from the notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import os

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==================== DATA LOADING ====================
print("Loading data...")

#csv to parquet conversion 
df = pd.read_csv('FemaleBCM.csv')
#sample amount for testing with less target data
df = df.sample(frac=0.07, random_state=42)
df.to_parquet("./Female_binary.parquet", compression="snappy")
df = pd.read_parquet("./Female_binary.parquet")

df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
df = df[df['ACFL'] > 30]

X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# Split into Train/Validation/Test
# Validation set is CRITICAL for learning optimal ensemble weights
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# ==================== LOAD PRE-TRAINED MODELS ====================
print("\nLoading pre-trained models...")
rf_male = joblib.load('./models/RF_BCM.pkl')
scaler_X_male = joblib.load('./models/x_scaler_BCM.pkl')

rf_female = joblib.load('./transfer-models/transfer_female_rf.pkl')
scaler_X_female = joblib.load('./transfer-models/x_scaler.pkl')
scaler_Y_F0_female = joblib.load('./transfer-models/f0_scaler.pkl')
scaler_Y_SPL_female = joblib.load('./transfer-models/spl_scaler.pkl')


# ==================== HELPER FUNCTIONS ====================
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return f"R²={r2:.3f}, MSE={mse:.2f}, MAE={mae:.2f}"


def get_male_predictions(X, scaler, model):
    """Get predictions from male (source) model"""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def get_female_predictions(X, scaler_X, model, scaler_F0, scaler_SPL):
    """Get predictions from female (target) model"""
    X_scaled = scaler_X.transform(X)
    Y_scaled = model.predict(X_scaled)
    return np.hstack([
        scaler_F0.inverse_transform(Y_scaled[:, 0].reshape(-1, 1)),
        scaler_SPL.inverse_transform(Y_scaled[:, 1].reshape(-1, 1))
    ])


# ==================== MODEL 0: TARGET-ONLY (BASELINE) ====================
# This is already done - rf_female is our Model 0
print("\n=== Model 0: Target-Only (Female Model) ===")
print("Already trained and loaded.")


# ==================== MODEL 2: SOURCE + RESIDUAL CORRECTION ====================
print("\n=== Model 2: Training Residual Correction ===")

# Step 1: Get source (male) predictions on female training data
Y_pred_male_train = get_male_predictions(X_train, scaler_X_male, rf_male)

# Step 2: Calculate residuals (what male model got wrong on female data)
Y_train_array = Y_train.values
residuals_train = Y_train_array - Y_pred_male_train

print(f"Residual stats - F0: mean={residuals_train[:, 0].mean():.2f}, std={residuals_train[:, 0].std():.2f}")
print(f"Residual stats - SPL: mean={residuals_train[:, 1].mean():.2f}, std={residuals_train[:, 1].std():.2f}")

# Step 3: Train a model to predict these residuals
# This learns the SYSTEMATIC domain shift from male to female
rf_residuals = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
)
rf_residuals.fit(X_train, residuals_train)

print("✓ Residual correction model trained")


# ==================== MODEL 3: FEATURE AUGMENTATION ====================
print("\n=== Model 3: Training Feature Augmentation Model ===")

# Augment features with source predictions
# This gives the model access to what the male model "thinks"
Y_pred_male_train_for_aug = get_male_predictions(X_train, scaler_X_male, rf_male)
X_train_augmented = np.column_stack([X_train.values, Y_pred_male_train_for_aug])

print(f"Original features: {X_train.shape[1]}, Augmented features: {X_train_augmented.shape[1]}")

# Train on augmented features
# Model learns WHEN to trust male predictions vs original features
rf_augmented = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
)
rf_augmented.fit(X_train_augmented, Y_train_array)

print("✓ Augmented feature model trained")


# ==================== GENERATE PREDICTIONS ON VALIDATION SET ====================
print("\n=== Generating Validation Predictions for Weight Optimization ===")

Y_val_array = Y_val.values

# Source (Male) predictions
pred_male_val = get_male_predictions(X_val, scaler_X_male, rf_male)

# Model 0: Target-only (Female)
pred_female_val = get_female_predictions(
    X_val, scaler_X_female, rf_female, 
    scaler_Y_F0_female, scaler_Y_SPL_female
)

# Model 2: Source + Residual
residuals_val = rf_residuals.predict(X_val)
pred_residual_val = pred_male_val + residuals_val

# Model 3: Augmented Features
X_val_augmented = np.column_stack([
    X_val.values, 
    get_male_predictions(X_val, scaler_X_male, rf_male)
])
pred_augmented_val = rf_augmented.predict(X_val_augmented)

print("✓ All validation predictions generated")


# ==================== LEARN OPTIMAL WEIGHTS PER OUTPUT ====================
print("\n=== Learning Optimal Ensemble Weights (TransRF) ===")

target_names = ['F0', 'SPL']
weights_per_output = {}

for i, target_name in enumerate(target_names):
    print(f"\n--- Optimizing weights for {target_name} ---")
    
    # Stack predictions for THIS specific output
    # Shape: [N_val, 3] -> Columns: [Female-only, Residual, Augmented]
    stack_val = np.column_stack([
        pred_female_val[:, i],
        pred_residual_val[:, i],
        pred_augmented_val[:, i]
    ])
    
    y_val_target = Y_val_array[:, i]
    
    # Solve for weights using non-negative linear regression
    # This ensures all models contribute positively
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(stack_val, y_val_target)
    
    # Normalize weights to sum to 1
    weights = reg.coef_ / np.sum(reg.coef_)
    weights_per_output[target_name] = weights
    
    print(f"  M0 (Female-only): {weights[0]:.3f}")
    print(f"  M2 (Residual):    {weights[1]:.3f}")
    print(f"  M3 (Augmented):   {weights[2]:.3f}")
    
    # Validate on validation set
    pred_ensemble_val = stack_val @ weights
    val_r2 = r2_score(y_val_target, pred_ensemble_val)
    print(f"  Validation R² with learned weights: {val_r2:.4f}")


# ==================== GENERATE TEST PREDICTIONS ====================
print("\n\n=== Generating Test Set Predictions ===")

Y_test_array = Y_test.values

# All model predictions on test set
pred_male_test = get_male_predictions(X_test, scaler_X_male, rf_male)
pred_female_test = get_female_predictions(
    X_test, scaler_X_female, rf_female,
    scaler_Y_F0_female, scaler_Y_SPL_female
)

residuals_test = rf_residuals.predict(X_test)
pred_residual_test = pred_male_test + residuals_test

X_test_augmented = np.column_stack([
    X_test.values,
    get_male_predictions(X_test, scaler_X_male, rf_male)
])
pred_augmented_test = rf_augmented.predict(X_test_augmented)

# Original simple ensemble (fixed 0.3/0.7 weights)
pred_simple_ensemble_test = 0.3 * pred_male_test + 0.7 * pred_female_test

# TransRF ensemble (learned weights, per-output)
pred_transrf_test = np.zeros_like(Y_test_array)
for i, target_name in enumerate(target_names):
    weights = weights_per_output[target_name]
    stack_test = np.column_stack([
        pred_female_test[:, i],
        pred_residual_test[:, i],
        pred_augmented_test[:, i]
    ])
    pred_transrf_test[:, i] = stack_test @ weights


# ==================== EVALUATION ====================
print("\n" + "="*70)
print("FINAL TEST SET PERFORMANCE COMPARISON")
print("="*70)

models_to_compare = [
    ("Male Only (Source)", pred_male_test),
    ("Female Only (Target)", pred_female_test),
    ("Residual Correction", pred_residual_test),
    ("Feature Augmentation", pred_augmented_test),
    ("Simple Ensemble (0.3/0.7)", pred_simple_ensemble_test),
    ("TransRF (Optimized)", pred_transrf_test)
]

results = []
for model_name, predictions in models_to_compare:
    r2_scores = r2_score(Y_test_array, predictions, multioutput='raw_values')
    mse_scores = mean_squared_error(Y_test_array, predictions, multioutput='raw_values')
    mae_scores = mean_absolute_error(Y_test_array, predictions, multioutput='raw_values')
    
    results.append({
        'Model': model_name,
        'F0_R2': r2_scores[0],
        'SPL_R2': r2_scores[1],
        'Avg_R2': np.mean(r2_scores),
        'F0_MSE': mse_scores[0],
        'SPL_MSE': mse_scores[1],
        'F0_MAE': mae_scores[0],
        'SPL_MAE': mae_scores[1]
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Highlight improvement
simple_r2 = results_df[results_df['Model'] == "Simple Ensemble (0.3/0.7)"]['Avg_R2'].values[0]
transrf_r2 = results_df[results_df['Model'] == "TransRF (Optimized)"]['Avg_R2'].values[0]
improvement = ((transrf_r2 - simple_r2) / simple_r2) * 100

print(f"\n{'='*70}")
print(f"TransRF IMPROVEMENT: {improvement:+.2f}% over simple ensemble")
print(f"{'='*70}")


# ==================== VISUALIZATION ====================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Transfer Learning Comparison: Simple vs TransRF', fontsize=16, fontweight='bold')

# Get common axis limits
all_f0 = np.concatenate([Y_test_array[:, 0], pred_male_test[:, 0], 
                         pred_female_test[:, 0], pred_transrf_test[:, 0]])
all_spl = np.concatenate([Y_test_array[:, 1], pred_male_test[:, 1],
                          pred_female_test[:, 1], pred_transrf_test[:, 1]])

# F0 Predictions (Row 1)
models_to_plot = [
    (pred_male_test, "Male Only (Source)", 'gray'),
    (pred_female_test, "Female Only (Target)", 'blue'),
    (pred_simple_ensemble_test, "Simple Ensemble (30%/70%)", 'orange'),
]

for idx, (pred, title, color) in enumerate(models_to_plot):
    ax = axes[0, idx]
    ax.scatter(Y_test_array[:, 0], pred[:, 0], alpha=0.5, color=color, s=20)
    ax.plot([all_f0.min(), all_f0.max()], [all_f0.min(), all_f0.max()], 
            'r--', linewidth=2, label='Perfect')
    ax.set_title(f'{title}\n{calculate_metrics(Y_test_array[:, 0], pred[:, 0])}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual F0 (Hz)')
    ax.set_ylabel('Predicted F0 (Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend()

# SPL Predictions (Row 2)
for idx, (pred, title, color) in enumerate(models_to_plot):
    ax = axes[1, idx]
    ax.scatter(Y_test_array[:, 1], pred[:, 1], alpha=0.5, color=color, s=20)
    ax.plot([all_spl.min(), all_spl.max()], [all_spl.min(), all_spl.max()],
            'r--', linewidth=2, label='Perfect')
    ax.set_title(f'{title}\n{calculate_metrics(Y_test_array[:, 1], pred[:, 1])}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual SPL (Pa)')
    ax.set_ylabel('Predicted SPL (Pa)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()

# Save figure
os.makedirs('./transfer-figs', exist_ok=True)
plt.savefig('./transfer-figs/TransRF_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved to ./transfer-figs/TransRF_comparison.png")


# Second figure: Focus on TransRF components
fig2, axes2 = plt.subplots(2, 4, figsize=(24, 12))
fig2.suptitle('TransRF Methodology Breakdown', fontsize=16, fontweight='bold')

transrf_models = [
    (pred_female_test, "M0: Female Only", 'blue'),
    (pred_residual_test, "M2: Residual Correction", 'purple'),
    (pred_augmented_test, "M3: Feature Augmentation", 'green'),
    (pred_transrf_test, "TransRF: Optimized Ensemble", 'red'),
]

# F0
for idx, (pred, title, color) in enumerate(transrf_models):
    ax = axes2[0, idx]
    ax.scatter(Y_test_array[:, 0], pred[:, 0], alpha=0.5, color=color, s=20)
    ax.plot([all_f0.min(), all_f0.max()], [all_f0.min(), all_f0.max()],
            'k--', linewidth=2, label='Perfect')
    metrics = calculate_metrics(Y_test_array[:, 0], pred[:, 0])
    ax.set_title(f'{title}\n{metrics}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual F0 (Hz)')
    ax.set_ylabel('Predicted F0 (Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend()

# SPL
for idx, (pred, title, color) in enumerate(transrf_models):
    ax = axes2[1, idx]
    ax.scatter(Y_test_array[:, 1], pred[:, 1], alpha=0.5, color=color, s=20)
    ax.plot([all_spl.min(), all_spl.max()], [all_spl.min(), all_spl.max()],
            'k--', linewidth=2, label='Perfect')
    metrics = calculate_metrics(Y_test_array[:, 1], pred[:, 1])
    ax.set_title(f'{title}\n{metrics}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual SPL (Pa)')
    ax.set_ylabel('Predicted SPL (Pa)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('./transfer-figs/TransRF_methodology.png', dpi=150, bbox_inches='tight')
print("✓ Saved to ./transfer-figs/TransRF_methodology.png")

plt.show()


# ==================== SAVE ENHANCED MODEL ====================
print("\nSaving enhanced TransRF model...")
os.makedirs('./transfer-models', exist_ok=True)

joblib.dump({
    'rf_male': rf_male,
    'rf_female': rf_female,
    'rf_residuals': rf_residuals,
    'rf_augmented': rf_augmented,
    'weights_per_output': weights_per_output,
    'scaler_X_male': scaler_X_male,
    'scaler_X_female': scaler_X_female,
    'scaler_Y_F0_female': scaler_Y_F0_female,
    'scaler_Y_SPL_female': scaler_Y_SPL_female,
    'target_names': target_names
}, './transfer-models/RF_TransRF_enhanced.pkl')

print("✓ Saved to ./transfer-models/RF_TransRF_enhanced.pkl")
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)