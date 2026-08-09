"""
WHAT WE TEST:
Sample sizes: 50, 100, 200, 500, 1000, and full dataset
For each size, we measure:
  - R² (how well the model fits, 0.0-1.0, higher is better)
  - MAE (mean absolute error in Hz/Pa, lower is better)

R² EXPLAINED:
R² = 1 - (prediction errors / total variance)
- R² = 1.00: Perfect predictions
- R² = 0.95: Excellent (model explains 95% of variance)
- R² = 0.90: Very good (model explains 90% of variance)
- R² = 0.70: Acceptable
- R² = 0.00: Model just predicts average (no learning)


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seeds for reproducibility (same results every time)
np.random.seed(42)
tf.random.set_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(SCRIPT_DIR, "data-requirement-experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
MALE_MODEL_PATH = os.path.join(MODELS_DIR, 'standard_model.keras')
MALE_X_SCALER_PATH = os.path.join(MODELS_DIR, 'x_scaler_BCM.pkl')
MALE_Y_SCALER_PATH = os.path.join(MODELS_DIR, 'y_scaler_BCM.pkl')

# Load the neural network trained on male data
base_model = load_model(MALE_MODEL_PATH, compile=False)

# Load the scalers (these normalize inputs/outputs to 0-1 range)
x_scaler_male = joblib.load(MALE_X_SCALER_PATH)
y_scaler_male = joblib.load(MALE_Y_SCALER_PATH)

BASE_DIR = os.path.dirname(SCRIPT_DIR)  # VocalFoldRegression/BCM Model
DATA_PATH = os.path.join(BASE_DIR, 'FemaleBCM.csv')

df_full = pd.read_csv(DATA_PATH, on_bad_lines="skip")
df_full = df_full.dropna()

print(f"Total female dataset size: {len(df_full)} rows\n")

# test with these dataset sizes (from small to full)
# Only include sizes that are smaller than the dataset
sample_sizes = [50, 100, 200, 500, 1000]
sample_sizes = [s for s in sample_sizes if s <= len(df_full)]  # filter out sizes larger than dataset
sample_sizes.append(len(df_full)) # add full dataset

early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=0,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=0
)

results = {
    'sample_size': [],
    'mae_f0': [],
    'mae_spl': [],
    'rmse_f0': [],
    'rmse_spl': [],
    'r2_f0': [],
    'r2_spl': [],
    'epochs_trained': []
}

for sample_size in sample_sizes:
    print(f"Training with {sample_size}")

    df_sample = df_full.sample(n=sample_size, random_state=42)
    X = df_sample[['a_CT', 'a_TA', 'PS']]
    Y = df_sample[['F0', 'SPL']]

    # first split: 80% train+val, 20% test
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    # second split: 90% train, 10% validation (training split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # scalers from the male model to keep the same normalization
    x_train_scaled = x_scaler_male.transform(x_train)
    x_val_scaled = x_scaler_male.transform(x_val)
    x_test_scaled = x_scaler_male.transform(x_test)

    y_train_scaled = y_scaler_male.transform(y_train)
    y_val_scaled = y_scaler_male.transform(y_val)
    y_test_scaled = y_scaler_male.transform(y_test)
    
    # create a copy with same architecture
    model_frozen = tf.keras.models.clone_model(base_model)
    model_frozen.set_weights(base_model.get_weights())  # Copy trained weights

    for layer in model_frozen.layers[:4]: # freeze first 4 layers
        layer.trainable = False

    # Compile
    model_frozen.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.00001), # low learning rate obv better for transfer learning
        metrics=['mae']
    )

    print(f"  Training...", end=" ")
    history = model_frozen.fit(
        x_train_scaled, y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=100,
        batch_size=min(128, len(x_train)),  # Use 128 or smaller if dataset is tiny
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    epochs_trained = len(history.history['loss'])

    y_pred_scaled = model_frozen.predict(x_test_scaled, verbose=0)
    y_pred = y_scaler_male.inverse_transform(y_pred_scaled)  # inverse the scaling back to Hz/Pa
    y_test_array = np.array(y_test)

    # metrics
    mae_f0 = mean_absolute_error(y_test_array[:, 0], y_pred[:, 0])
    mae_spl = mean_absolute_error(y_test_array[:, 1], y_pred[:, 1])
    rmse_f0 = np.sqrt(mean_squared_error(y_test_array[:, 0], y_pred[:, 0]))
    rmse_spl = np.sqrt(mean_squared_error(y_test_array[:, 1], y_pred[:, 1]))
    r2_f0 = r2_score(y_test_array[:, 0], y_pred[:, 0])
    r2_spl = r2_score(y_test_array[:, 1], y_pred[:, 1])

    # store results
    results['sample_size'].append(sample_size)
    results['mae_f0'].append(mae_f0)
    results['mae_spl'].append(mae_spl)
    results['rmse_f0'].append(rmse_f0)
    results['rmse_spl'].append(rmse_spl)
    results['r2_f0'].append(r2_f0)
    results['r2_spl'].append(r2_spl)
    results['epochs_trained'].append(epochs_trained)

    # Print results
    print(f"\n  RESULTS:")
    print(f"    F0  → R²: {r2_f0:.4f} | MAE: {mae_f0:6.2f} Hz | RMSE: {rmse_f0:6.2f} Hz")
    print(f"    SPL → R²: {r2_spl:.4f} | MAE: {mae_spl:.4f} Pa | RMSE: {rmse_spl:.4f} Pa")

# save results
results_df = pd.DataFrame(results)
results_path = os.path.join(EXPERIMENT_DIR, 'data_requirements_results.csv')
results_df.to_csv(results_path, index=False)


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Transfer Learning: Performance vs Training Data Size', fontsize=16, fontweight='bold')

# Plot 1: MAE vs Sample Size (F0)
ax = axes[0, 0]
ax.plot(results['sample_size'], results['mae_f0'], 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('Training Samples')
ax.set_ylabel('MAE (Hz)')
ax.set_title('F0 - Mean Absolute Error')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Plot 2: MAE vs Sample Size (SPL)
ax = axes[0, 1]
ax.plot(results['sample_size'], results['mae_spl'], 'o-', linewidth=2, markersize=8, color='darkorange')
ax.set_xlabel('Training Samples')
ax.set_ylabel('MAE (Pa)')
ax.set_title('SPL - Mean Absolute Error')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Plot 3: RMSE vs Sample Size (both F0 and SPL)
ax = axes[0, 2]
ax.plot(results['sample_size'], results['rmse_f0'], 'o-', linewidth=2, markersize=8,
        color='steelblue', label='F0')
ax.plot(results['sample_size'], results['rmse_spl'], 's-', linewidth=2, markersize=8,
        color='darkorange', label='SPL')
ax.set_xlabel('Training Samples')
ax.set_ylabel('RMSE')
ax.set_title('Root Mean Squared Error')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: R² vs Sample Size (F0)
ax = axes[1, 0]
ax.plot(results['sample_size'], results['r2_f0'], 'o-', linewidth=2, markersize=8, color='steelblue')
ax.axhline(y=0.95, color='red', linestyle='--', label='R²=0.95 (Excellent)')
ax.axhline(y=0.90, color='orange', linestyle='--', label='R²=0.90 (Very Good)')
ax.set_xlabel('Training Samples')
ax.set_ylabel('R² Score')
ax.set_title('F0 - R² Score (Coefficient of Determination)')
ax.set_xscale('log')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: R² vs Sample Size (SPL)
ax = axes[1, 1]
ax.plot(results['sample_size'], results['r2_spl'], 'o-', linewidth=2, markersize=8, color='darkorange')
ax.axhline(y=0.95, color='red', linestyle='--', label='R²=0.95 (Excellent)')
ax.axhline(y=0.90, color='orange', linestyle='--', label='R²=0.90 (Very Good)')
ax.set_xlabel('Training Samples')
ax.set_ylabel('R² Score')
ax.set_title('SPL - R² Score (Coefficient of Determination)')
ax.set_xscale('log')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Epochs Trained vs Sample Size
ax = axes[1, 2]
ax.plot(results['sample_size'], results['epochs_trained'], 'o-', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Training Samples')
ax.set_ylabel('Epochs Until Early Stop')
ax.set_title('Training Duration')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(EXPERIMENT_DIR, 'data_requirements_analysis.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
