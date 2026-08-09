import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

NN_DIR = os.path.join(BASE_DIR, 'NeuralNetwork')
RF_DIR = os.path.join(BASE_DIR, 'RandomForest')
PR_DIR = os.path.join(BASE_DIR, 'PolynomialRegressor')

DATA_PATH = os.path.join(RF_DIR, 'Female_binary.parquet')

df = pd.read_parquet(DATA_PATH)
df = df.dropna()
df = df[df['ACFL'] > 30]

print(f"dataset size: {len(df)}")

# load pre-trained models ONCE (outside loop)
print("\nloading pre-trained models...")
nn_model = load_model(os.path.join(NN_DIR, 'transfer-models', 'transfer_frozen_4_layers.keras'), compile=False)
nn_x_scaler = joblib.load(os.path.join(NN_DIR, 'transfer-models', 'x_scaler_female.pkl'))
nn_y_scaler = joblib.load(os.path.join(NN_DIR, 'transfer-models', 'y_scaler_female.pkl'))

rf_transfer = joblib.load(os.path.join(RF_DIR, 'transfer-models', 'RF_BCM_transfer.pkl'))
rf_male = rf_transfer['rf_male']
rf_female = rf_transfer['rf_female']
rf_x_scaler_male = rf_transfer['scaler_X_male']
rf_x_scaler_female = rf_transfer['scaler_X_female']
rf_f0_scaler = rf_transfer['scaler_Y_F0']
rf_spl_scaler = rf_transfer['scaler_Y_SPL']
w_base, w_new = rf_transfer['weights']

sample_sizes = [25, 50, 100, 250, 500, 750, len(df)]
results_list = []

# extract full test set from original dataset
X_full = df[['a_CT', 'a_TA', 'PS']]
Y_full = df[['F0', 'SPL']]
_, x_test_full, _, y_test_full = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=42
)

print(f"full test set size: {len(x_test_full)} samples\n")

for sample_size in sample_sizes:
    # limit how much of the test set to use (simulating limited test data scenarios)
    # this shows how metrics stabilize as you get more test samples
    test_sample_size = min(sample_size, len(x_test_full))

    print(f"\nevaluating with {test_sample_size} test samples")

    # subsample test set consistently (larger sample_size = more test samples)
    if test_sample_size < len(x_test_full):
        x_test_subset = x_test_full.iloc[:test_sample_size]
        y_test_subset = y_test_full.iloc[:test_sample_size]
    else:
        x_test_subset = x_test_full
        y_test_subset = y_test_full

    y_test_array = np.array(y_test_subset)

    # =====================================================================
    # NEURAL NETWORK - using pre-trained model
    # =====================================================================
    x_test_scaled = nn_x_scaler.transform(x_test_subset)
    y_pred_scaled = nn_model.predict(x_test_scaled, verbose=0)
    y_pred_nn = nn_y_scaler.inverse_transform(y_pred_scaled)

    r2_f0_nn = r2_score(y_test_array[:, 0], y_pred_nn[:, 0])
    r2_spl_nn = r2_score(y_test_array[:, 1], y_pred_nn[:, 1])
    mae_f0_nn = mean_absolute_error(y_test_array[:, 0], y_pred_nn[:, 0])
    mae_spl_nn = mean_absolute_error(y_test_array[:, 1], y_pred_nn[:, 1])

    results_list.append({
        'sample_size': test_sample_size,
        'regressor': 'NN',
        'r2_f0': r2_f0_nn,
        'r2_spl': r2_spl_nn,
        'r2_avg': (r2_f0_nn + r2_spl_nn) / 2,
        'mae_f0': mae_f0_nn,
        'mae_spl': mae_spl_nn
    })

    # =====================================================================
    # RANDOM FOREST - using pre-trained ensemble
    # =====================================================================
    x_test_scaled_male = rf_x_scaler_male.transform(x_test_subset)
    y_pred_male_original = rf_male.predict(x_test_scaled_male)

    x_test_scaled_female = rf_x_scaler_female.transform(x_test_subset)
    y_pred_female_scaled = rf_female.predict(x_test_scaled_female)
    y_pred_female_original = np.hstack([
        rf_f0_scaler.inverse_transform(y_pred_female_scaled[:, 0].reshape(-1, 1)),
        rf_spl_scaler.inverse_transform(y_pred_female_scaled[:, 1].reshape(-1, 1))
    ])

    y_pred_rf = w_base * y_pred_male_original + w_new * y_pred_female_original

    r2_f0_rf = r2_score(y_test_array[:, 0], y_pred_rf[:, 0])
    r2_spl_rf = r2_score(y_test_array[:, 1], y_pred_rf[:, 1])
    mae_f0_rf = mean_absolute_error(y_test_array[:, 0], y_pred_rf[:, 0])
    mae_spl_rf = mean_absolute_error(y_test_array[:, 1], y_pred_rf[:, 1])

    results_list.append({
        'sample_size': test_sample_size,
        'regressor': 'RF',
        'r2_f0': r2_f0_rf,
        'r2_spl': r2_spl_rf,
        'r2_avg': (r2_f0_rf + r2_spl_rf) / 2,
        'mae_f0': mae_f0_rf,
        'mae_spl': mae_spl_rf
    })

    print(f"  nn: r²_f0={r2_f0_nn:.3f}, r²_spl={r2_spl_nn:.3f}, avg={((r2_f0_nn + r2_spl_nn) / 2):.3f}")
    print(f"  rf: r²_f0={r2_f0_rf:.3f}, r²_spl={r2_spl_rf:.3f}, avg={((r2_f0_rf + r2_spl_rf) / 2):.3f}")

print("\n" + "="*60)
print("transfer learning regressor comparison (nn vs rf)")
print("="*60)
results_df = pd.DataFrame(results_list)
print(results_df.to_string(index=False))
print("="*60)

os.makedirs(os.path.join(SCRIPT_DIR, 'figs'), exist_ok=True)
results_csv_path = os.path.join(SCRIPT_DIR, 'figs', 'transfer_comparison.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nresults saved to: {results_csv_path}")

# create visualization with 4 subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('transfer learning regressor comparison (nn vs rf)', fontsize=16, fontweight='bold')

# color scheme for different regressors
colors = {'NN': 'steelblue', 'RF': 'darkorange'}

# plot each regressor
for regressor in ['NN', 'RF']:
    subset = results_df[results_df['regressor'] == regressor]
    color = colors[regressor]

    # r² for f0
    axes[0, 0].plot(subset['sample_size'], subset['r2_f0'], marker='o',
                    label=regressor, color=color, linewidth=2)
    # r² for spl
    axes[0, 1].plot(subset['sample_size'], subset['r2_spl'], marker='o',
                    label=regressor, color=color, linewidth=2)
    # mae for f0
    axes[1, 0].plot(subset['sample_size'], subset['mae_f0'], marker='o',
                    label=regressor, color=color, linewidth=2)
    # mae for spl
    axes[1, 1].plot(subset['sample_size'], subset['mae_spl'], marker='o',
                    label=regressor, color=color, linewidth=2)

# configure top-left subplot: r² f0
axes[0, 0].set_title('r² f0 vs sample size', fontweight='bold')
axes[0, 0].set_xlabel('sample size')
axes[0, 0].set_ylabel('r² f0')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# configure top-right subplot: r² spl
axes[0, 1].set_title('r² spl vs sample size', fontweight='bold')
axes[0, 1].set_xlabel('sample size')
axes[0, 1].set_ylabel('r² spl')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# configure bottom-left subplot: mae f0
axes[1, 0].set_title('mae f0 vs sample size', fontweight='bold')
axes[1, 0].set_xlabel('sample size')
axes[1, 0].set_ylabel('mae f0 (hz)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# configure bottom-right subplot: mae spl
axes[1, 1].set_title('mae spl vs sample size', fontweight='bold')
axes[1, 1].set_xlabel('sample size')
axes[1, 1].set_ylabel('mae spl (pa)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# save figure
plt.tight_layout()
fig_path = os.path.join(SCRIPT_DIR, 'figs', 'transfer_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"visualization saved to: {fig_path}")
plt.show()

