'''
comprehensive transfer learning comparison across all three regressor types:
- neural network (nn)
- random forest (rf)
- polynomial regression (pr)

this script evaluates pre-trained transfer learning models on varying test set sizes
to show how evaluation metrics stabilize with more test samples
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from keras.models import load_model
import joblib

from sklearn.metrics import r2_score, mean_absolute_error

# paths resolved relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # VocalFoldRegression/BCM Model
NN_DIR = f'{BASE_DIR}/NeuralNetwork'
RF_DIR = f'{BASE_DIR}/RandomForest'
PR_DIR = f'{BASE_DIR}/PolynomialRegressor'
REGRESSOR_ANALYSIS_DIR = f'{BASE_DIR}/ResgressorAnalysis'

DATA_PATH = f'{RF_DIR}/Female_binary.parquet'

df = pd.read_parquet(DATA_PATH)
# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
# CRITICAL: Apply ACFL > 30 filter to match training data for NN, RF, and PR models
df = df[df['ACFL'] > 30]

print(f"dataset size: {len(df)}")

# load pre-trained models ONCE (outside loop)
print("\nloading pre-trained models...")

# neural network
nn_model = load_model(f'{NN_DIR}/transfer-models/transfer_frozen_4_layers.keras', compile=False)
nn_x_scaler = joblib.load(f'{NN_DIR}/transfer-models/x_scaler_female.pkl')
nn_f0_scaler = joblib.load(f'{NN_DIR}/transfer-models/f0_scaler_female.pkl')
nn_spl_scaler = joblib.load(f'{NN_DIR}/transfer-models/spl_scaler_female.pkl')

# random forest
rf_transfer = joblib.load(f'{RF_DIR}/transfer-models/RF_BCM_transfer.pkl')
rf_male = rf_transfer['rf_male']
rf_female = rf_transfer['rf_female']
rf_x_scaler_male = rf_transfer['scaler_X_male']
rf_x_scaler_female = rf_transfer['scaler_X_female']
rf_f0_scaler = rf_transfer['scaler_Y_F0']
rf_spl_scaler = rf_transfer['scaler_Y_SPL']
rf_w_base, rf_w_new = rf_transfer['weights']

# polynomial regression
pr_transfer = joblib.load(f'{PR_DIR}/transfer-models/PR_transfer_ensemble.pkl')
pr_male_model = pr_transfer['male_model']  # MultiOutputRegressor
pr_female_f0 = pr_transfer['female_f0_model']
pr_female_spl = pr_transfer['female_spl_model']
pr_w_male, pr_w_female = pr_transfer['weights']
pr_male_x_scaler = pr_transfer['male_x_scaler']
pr_male_f0_scaler = pr_transfer['male_f0_scaler']
pr_male_spl_scaler = pr_transfer['male_spl_scaler']
pr_female_x_scaler = pr_transfer['female_x_scaler']
pr_female_f0_scaler = pr_transfer['female_f0_scaler']
pr_female_spl_scaler = pr_transfer['female_spl_scaler']
pr_female_poly_f0 = pr_transfer['female_poly_f0']
pr_female_poly_spl = pr_transfer['female_poly_spl']
pr_male_poly = pr_transfer['male_poly']

# use full dataset for evaluation since models are already trained
X_full = df[['a_CT', 'a_TA', 'PS']]
Y_full = df[['F0', 'SPL']]

print(f"full dataset size: {len(df)} samples\n")

# define sample sizes up to full dataset
# extended 2026-05-14 to match the alternates N grid in Female_Showcase
# (N=[5,10,20,30,50,75,100,150,200,300,500]) for direct head-to-head.
# Don't interpolate.
sample_sizes = [5, 10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 300, 500, len(df)]
n_bootstrap = 10  # number of random shuffles to average over
results_list = []

for sample_size in sample_sizes:
    eval_sample_size = min(sample_size, len(df))
    print(f"\nevaluating with {eval_sample_size} samples (averaging over {n_bootstrap} shuffles)")

    # store metrics across bootstrap iterations
    bootstrap_metrics = {
        'nn_r2_f0': [], 'nn_r2_spl': [], 'nn_mae_f0': [], 'nn_mae_spl': [],
        'rf_r2_f0': [], 'rf_r2_spl': [], 'rf_mae_f0': [], 'rf_mae_spl': [],
        'pr_r2_f0': [], 'pr_r2_spl': [], 'pr_mae_f0': [], 'pr_mae_spl': []
    }

    # run multiple bootstrap iterations
    for bootstrap_iter in range(n_bootstrap):
        # use different random seed for each iteration
        np.random.seed(42 + bootstrap_iter)
        random_indices = np.random.choice(len(df), size=eval_sample_size, replace=False)

        x_eval_subset = X_full.iloc[random_indices]
        y_eval_subset = Y_full.iloc[random_indices]
        y_eval_array = np.array(y_eval_subset)

        # =====================================================================
        # NEURAL NETWORK
        # =====================================================================
        x_eval_scaled_nn = nn_x_scaler.transform(x_eval_subset)

        # predict using NN
        y_pred_nn_scaled = nn_model.predict(x_eval_scaled_nn, verbose=0)

        # inverse transform - handle separate scalers
        f0_pred_nn = nn_f0_scaler.inverse_transform(y_pred_nn_scaled[:, 0].reshape(-1, 1))
        spl_pred_nn = nn_spl_scaler.inverse_transform(y_pred_nn_scaled[:, 1].reshape(-1, 1))
        y_pred_nn = np.hstack([f0_pred_nn, spl_pred_nn])

        r2_f0_nn = r2_score(y_eval_array[:, 0], y_pred_nn[:, 0])
        r2_spl_nn = r2_score(y_eval_array[:, 1], y_pred_nn[:, 1])
        mae_f0_nn = mean_absolute_error(y_eval_array[:, 0], y_pred_nn[:, 0])
        mae_spl_nn = mean_absolute_error(y_eval_array[:, 1], y_pred_nn[:, 1])

        bootstrap_metrics['nn_r2_f0'].append(r2_f0_nn)
        bootstrap_metrics['nn_r2_spl'].append(r2_spl_nn)
        bootstrap_metrics['nn_mae_f0'].append(mae_f0_nn)
        bootstrap_metrics['nn_mae_spl'].append(mae_spl_nn)

        # =====================================================================
        # RANDOM FOREST
        # =====================================================================
        x_eval_scaled_male = rf_x_scaler_male.transform(x_eval_subset)
        y_pred_male_original = rf_male.predict(x_eval_scaled_male)

        x_eval_scaled_female = rf_x_scaler_female.transform(x_eval_subset)
        y_pred_female_scaled = rf_female.predict(x_eval_scaled_female)
        y_pred_female_original = np.hstack([
            rf_f0_scaler.inverse_transform(y_pred_female_scaled[:, 0].reshape(-1, 1)),
            rf_spl_scaler.inverse_transform(y_pred_female_scaled[:, 1].reshape(-1, 1))
        ])

        y_pred_rf = rf_w_base * y_pred_male_original + rf_w_new * y_pred_female_original

        r2_f0_rf = r2_score(y_eval_array[:, 0], y_pred_rf[:, 0])
        r2_spl_rf = r2_score(y_eval_array[:, 1], y_pred_rf[:, 1])
        mae_f0_rf = mean_absolute_error(y_eval_array[:, 0], y_pred_rf[:, 0])
        mae_spl_rf = mean_absolute_error(y_eval_array[:, 1], y_pred_rf[:, 1])

        bootstrap_metrics['rf_r2_f0'].append(r2_f0_rf)
        bootstrap_metrics['rf_r2_spl'].append(r2_spl_rf)
        bootstrap_metrics['rf_mae_f0'].append(mae_f0_rf)
        bootstrap_metrics['rf_mae_spl'].append(mae_spl_rf)

        # =====================================================================
        # POLYNOMIAL REGRESSION
        # =====================================================================
        # male model predictions
        x_eval_scaled_pr_male = pr_male_x_scaler.transform(x_eval_subset)
        x_eval_poly_male = pr_male_poly.transform(x_eval_scaled_pr_male)
        y_pred_pr_male_scaled = pr_male_model.predict(x_eval_poly_male)

        # inverse transform using male scalers
        f0_pred_pr_male = pr_male_f0_scaler.inverse_transform(y_pred_pr_male_scaled[:, 0].reshape(-1, 1))
        spl_pred_pr_male = pr_male_spl_scaler.inverse_transform(y_pred_pr_male_scaled[:, 1].reshape(-1, 1))
        y_pred_pr_male = np.hstack([f0_pred_pr_male, spl_pred_pr_male])

        # female model predictions
        x_eval_scaled_pr_female = pr_female_x_scaler.transform(x_eval_subset)
        x_eval_poly_f0_female = pr_female_poly_f0.transform(x_eval_scaled_pr_female)
        x_eval_poly_spl_female = pr_female_poly_spl.transform(x_eval_scaled_pr_female)

        f0_pred_pr_female_scaled = pr_female_f0.predict(x_eval_poly_f0_female)
        spl_pred_pr_female_scaled = pr_female_spl.predict(x_eval_poly_spl_female)
        f0_pred_pr_female = pr_female_f0_scaler.inverse_transform(f0_pred_pr_female_scaled.reshape(-1, 1))
        spl_pred_pr_female = pr_female_spl_scaler.inverse_transform(spl_pred_pr_female_scaled.reshape(-1, 1))
        y_pred_pr_female = np.hstack([f0_pred_pr_female, spl_pred_pr_female])

        # weighted ensemble
        y_pred_pr = pr_w_male * y_pred_pr_male + pr_w_female * y_pred_pr_female

        r2_f0_pr = r2_score(y_eval_array[:, 0], y_pred_pr[:, 0])
        r2_spl_pr = r2_score(y_eval_array[:, 1], y_pred_pr[:, 1])
        mae_f0_pr = mean_absolute_error(y_eval_array[:, 0], y_pred_pr[:, 0])
        mae_spl_pr = mean_absolute_error(y_eval_array[:, 1], y_pred_pr[:, 1])

        bootstrap_metrics['pr_r2_f0'].append(r2_f0_pr)
        bootstrap_metrics['pr_r2_spl'].append(r2_spl_pr)
        bootstrap_metrics['pr_mae_f0'].append(mae_f0_pr)
        bootstrap_metrics['pr_mae_spl'].append(mae_spl_pr)

    # average bootstrap results
    r2_f0_nn_avg = np.mean(bootstrap_metrics['nn_r2_f0'])
    r2_spl_nn_avg = np.mean(bootstrap_metrics['nn_r2_spl'])
    mae_f0_nn_avg = np.mean(bootstrap_metrics['nn_mae_f0'])
    mae_spl_nn_avg = np.mean(bootstrap_metrics['nn_mae_spl'])

    r2_f0_rf_avg = np.mean(bootstrap_metrics['rf_r2_f0'])
    r2_spl_rf_avg = np.mean(bootstrap_metrics['rf_r2_spl'])
    mae_f0_rf_avg = np.mean(bootstrap_metrics['rf_mae_f0'])
    mae_spl_rf_avg = np.mean(bootstrap_metrics['rf_mae_spl'])

    r2_f0_pr_avg = np.mean(bootstrap_metrics['pr_r2_f0'])
    r2_spl_pr_avg = np.mean(bootstrap_metrics['pr_r2_spl'])
    mae_f0_pr_avg = np.mean(bootstrap_metrics['pr_mae_f0'])
    mae_spl_pr_avg = np.mean(bootstrap_metrics['pr_mae_spl'])

    results_list.append({
        'sample_size': eval_sample_size,
        'regressor': 'NN',
        'r2_f0': r2_f0_nn_avg,
        'r2_spl': r2_spl_nn_avg,
        'r2_avg': (r2_f0_nn_avg + r2_spl_nn_avg) / 2,
        'mae_f0': mae_f0_nn_avg,
        'mae_spl': mae_spl_nn_avg
    })

    results_list.append({
        'sample_size': eval_sample_size,
        'regressor': 'RF',
        'r2_f0': r2_f0_rf_avg,
        'r2_spl': r2_spl_rf_avg,
        'r2_avg': (r2_f0_rf_avg + r2_spl_rf_avg) / 2,
        'mae_f0': mae_f0_rf_avg,
        'mae_spl': mae_spl_rf_avg
    })

    results_list.append({
        'sample_size': eval_sample_size,
        'regressor': 'PR',
        'r2_f0': r2_f0_pr_avg,
        'r2_spl': r2_spl_pr_avg,
        'r2_avg': (r2_f0_pr_avg + r2_spl_pr_avg) / 2,
        'mae_f0': mae_f0_pr_avg,
        'mae_spl': mae_spl_pr_avg
    })

    print(f"  nn: r²_f0={r2_f0_nn_avg:.3f}, r²_spl={r2_spl_nn_avg:.3f}, avg={((r2_f0_nn_avg + r2_spl_nn_avg) / 2):.3f}")
    print(f"  rf: r²_f0={r2_f0_rf_avg:.3f}, r²_spl={r2_spl_rf_avg:.3f}, avg={((r2_f0_rf_avg + r2_spl_rf_avg) / 2):.3f}")
    print(f"  pr: r²_f0={r2_f0_pr_avg:.3f}, r²_spl={r2_spl_pr_avg:.3f}, avg={((r2_f0_pr_avg + r2_spl_pr_avg) / 2):.3f}")

print("\n" + "="*60)
print("transfer learning regressor comparison (nn vs rf vs pr)")
print("="*60)
results_df = pd.DataFrame(results_list)
print(results_df.to_string(index=False))
print("="*60)

os.makedirs(f'{REGRESSOR_ANALYSIS_DIR}/figs', exist_ok=True)
results_csv_path = f'{REGRESSOR_ANALYSIS_DIR}/figs/all_regressors_transfer_comparison.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"\nresults saved to: {results_csv_path}")

# create visualization with 4 subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('transfer learning: all regressors comparison (nn vs rf vs pr)', fontsize=16, fontweight='bold')

# color scheme for different regressors
colors = {'NN': 'steelblue', 'RF': 'darkorange', 'PR': 'green'}

# plot each regressor
for regressor in ['NN', 'RF', 'PR']:
    subset = results_df[results_df['regressor'] == regressor]
    color = colors[regressor]

    # r² for f0
    axes[0, 0].plot(subset['sample_size'], subset['r2_f0'], marker='o',
                    label=regressor, color=color, linewidth=2, markersize=6)
    # r² for spl
    axes[0, 1].plot(subset['sample_size'], subset['r2_spl'], marker='o',
                    label=regressor, color=color, linewidth=2, markersize=6)
    # mae for f0
    axes[1, 0].plot(subset['sample_size'], subset['mae_f0'], marker='o',
                    label=regressor, color=color, linewidth=2, markersize=6)
    # mae for spl
    axes[1, 1].plot(subset['sample_size'], subset['mae_spl'], marker='o',
                    label=regressor, color=color, linewidth=2, markersize=6)

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
fig_path = f'{REGRESSOR_ANALYSIS_DIR}/figs/all_regressors_transfer_comparison.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"visualization saved to: {fig_path}")
plt.show()
