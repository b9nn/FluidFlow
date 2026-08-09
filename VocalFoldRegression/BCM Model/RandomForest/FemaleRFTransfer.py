'''
excerpt: (lines 460-462)
For RF, a similar strategy can be applied by constructing an ensemble of regressors trained on each dataset (Gu et al., 2022),
with predictions combined through weighted estimates of the two datasets, or by refining
branch weights using subject-specific data once the baseline architecture has been defined
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import joblib
import os

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load female dataset
df = pd.read_parquet("./Female_binary.parquet")
# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
df = df[df['ACFL'] > 30] # new df that only has ACFL values > 30

# define axis
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# t/t/v
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=42)

rf_male = joblib.load('./models/RF_BCM.pkl')
scaler_X_male = joblib.load('./models/x_scaler_BCM.pkl')

rf_female = joblib.load('./transfer-models/transfer_female_rf.pkl')
scaler_X_female = joblib.load('./transfer-models/x_scaler.pkl')
scaler_Y_F0_female = joblib.load('./transfer-models/f0_scaler.pkl')
scaler_Y_SPL_female = joblib.load('./transfer-models/spl_scaler.pkl')

w_base = 0.3
w_new = 0.7

X_test_scaled_male = scaler_X_male.transform(X_test)
Y_pred_male_original = rf_male.predict(X_test_scaled_male)

X_test_scaled_female = scaler_X_female.transform(X_test)
Y_pred_female_scaled = rf_female.predict(X_test_scaled_female)
Y_pred_female_original = np.hstack([
    scaler_Y_F0_female.inverse_transform(Y_pred_female_scaled[:, 0].reshape(-1, 1)),
    scaler_Y_SPL_female.inverse_transform(Y_pred_female_scaled[:, 1].reshape(-1, 1))
])

Y_pred_ensemble_original = w_base * Y_pred_male_original + w_new * Y_pred_female_original # numpy linear combinations

Y_test_array = Y_test.values

r2_male_test = r2_score(Y_test_array, Y_pred_male_original, multioutput='raw_values')
r2_female_test = r2_score(Y_test_array, Y_pred_female_original, multioutput='raw_values')
r2_ensemble_test = r2_score(Y_test_array, Y_pred_ensemble_original, multioutput='raw_values')

mse_male_test = mean_squared_error(Y_test_array, Y_pred_male_original, multioutput='raw_values')
mse_female_test = mean_squared_error(Y_test_array, Y_pred_female_original, multioutput='raw_values')
mse_ensemble_test = mean_squared_error(Y_test_array, Y_pred_ensemble_original, multioutput='raw_values')

print("\n=== Test Set Performance ===")
print(f"Male Model - R² (F0, SPL): {r2_male_test}, MSE: {mse_male_test}")
print(f"Female Model - R² (F0, SPL): {r2_female_test}, MSE: {mse_female_test}")
print(f"Ensemble Model - R² (F0, SPL): {r2_ensemble_test}, MSE: {mse_ensemble_test}")
print(f"Average R² - Male: {np.mean(r2_male_test):.4f}, Female: {np.mean(r2_female_test):.4f}, Ensemble: {np.mean(r2_ensemble_test):.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Transfer Learning: Random Forest Ensemble Comparison', fontsize=16)

# F0 predictions
for idx, (y_pred, title) in enumerate([
    (Y_pred_male_original, 'Male Model (Baseline)'),
    (Y_pred_female_original, 'Female Model (New)'),
    (Y_pred_ensemble_original, 'Ensemble (Transfer Learning)')
]):
    ax = axes[0, idx]
    ax.scatter(Y_test_array[:, 0], y_pred[:, 0], alpha=0.5)
    ax.plot([Y_test_array[:, 0].min(), Y_test_array[:, 0].max()],
            [Y_test_array[:, 0].min(), Y_test_array[:, 0].max()], 'r--', linewidth=2)
    ax.set_title(f'{title}\nF0 Prediction')
    ax.set_xlabel('Actual F0 (Hz)')
    ax.set_ylabel('Predicted F0 (Hz)')
    ax.grid(True, alpha=0.3)

# SPL predictions
for idx, (y_pred, title) in enumerate([
    (Y_pred_male_original, 'Male Model (Baseline)'),
    (Y_pred_female_original, 'Female Model (New)'),
    (Y_pred_ensemble_original, 'Ensemble (Transfer Learning)')
]):
    ax = axes[1, idx]
    ax.scatter(Y_test_array[:, 1], y_pred[:, 1], alpha=0.5)
    ax.plot([Y_test_array[:, 1].min(), Y_test_array[:, 1].max()],
            [Y_test_array[:, 1].min(), Y_test_array[:, 1].max()], 'r--', linewidth=2)
    ax.set_title(f'{title}\nSPL Prediction')
    ax.set_xlabel('Actual SPL (Pa)')
    ax.set_ylabel('Predicted SPL (Pa)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('./transfer-figs', exist_ok=True)
plt.savefig('./transfer-figs/RF_transfer_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to ./transfer-figs/RF_transfer_comparison.png")
plt.show()

os.makedirs('./transfer-models', exist_ok=True)
transfer_path = './transfer-models/RF_BCM_transfer.pkl'
joblib.dump({
    'rf_male': rf_male,
    'rf_female': rf_female,
    'weights': (w_base, w_new),
    'scaler_X_male': scaler_X_male,
    'scaler_X_female': scaler_X_female,
    'scaler_Y_F0': scaler_Y_F0_female,
    'scaler_Y_SPL': scaler_Y_SPL_female
}, transfer_path)

