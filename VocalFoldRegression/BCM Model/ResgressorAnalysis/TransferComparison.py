import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

from sklearn.metrics import r2_score

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

sample_sizes = [25, 50, 100, 250, 500, 750, len(df)]

results_list = []

for sample_size in sample_sizes:
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()

    X_sample = df_sample[['a_CT', 'a_TA', 'PS']]
    Y_sample = df_sample[['F0', 'SPL']]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        X_sample, Y_sample, test_size=0.2, random_state=42
    )

    y_test_array = np.array(y_test)

    # neural network
    nn_model = load_model(os.path.join(NN_DIR, 'transfer-models', 'transfer_frozen_4_layers.keras'), compile=False)
    nn_x_scaler = joblib.load(os.path.join(NN_DIR, 'transfer-models', 'x_scaler_female.pkl'))
    nn_y_scaler = joblib.load(os.path.join(NN_DIR, 'transfer-models', 'y_scaler_female.pkl'))

    x_test_scaled = nn_x_scaler.transform(x_test)
    y_pred_scaled = nn_model.predict(x_test_scaled, verbose=0)
    y_pred_nn = nn_y_scaler.inverse_transform(y_pred_scaled)

    r2_f0_nn = r2_score(y_test_array[:, 0], y_pred_nn[:, 0])
    r2_spl_nn = r2_score(y_test_array[:, 1], y_pred_nn[:, 1])

    results_list.append({
        'sample_size': sample_size,
        'regressor': 'NN',
        'r2_f0': r2_f0_nn,
        'r2_spl': r2_spl_nn,
        'r2_avg': (r2_f0_nn + r2_spl_nn) / 2
    })

    # random forest (loaded as a dict of everything saved)
    rf_transfer = joblib.load(os.path.join(RF_DIR, 'transfer-models', 'RF_BCM_transfer.pkl'))
    rf_male = rf_transfer['rf_male']
    rf_female = rf_transfer['rf_female']
    rf_x_scaler = rf_transfer['scaler_X']
    rf_f0_scaler = rf_transfer['scaler_Y_F0']
    rf_spl_scaler = rf_transfer['scaler_Y_SPL']
    w_base, w_new = rf_transfer['weights']

    x_test_scaled = rf_x_scaler.transform(x_test)

    y_pred_male = rf_male.predict(x_test_scaled)
    y_pred_female = rf_female.predict(x_test_scaled)
    y_pred_ensemble_scaled = w_base * y_pred_male + w_new * y_pred_female # numpy momemt

    y_pred_rf = np.hstack([
        rf_f0_scaler.inverse_transform(y_pred_ensemble_scaled[:, 0].reshape(-1, 1)),
        rf_spl_scaler.inverse_transform(y_pred_ensemble_scaled[:, 1].reshape(-1, 1))
    ])

    r2_f0_rf = r2_score(y_test_array[:, 0], y_pred_rf[:, 0])
    r2_spl_rf = r2_score(y_test_array[:, 1], y_pred_rf[:, 1])

    results_list.append({
        'sample_size': sample_size,
        'regressor': 'RF',
        'r2_f0': r2_f0_rf,
        'r2_spl': r2_spl_rf,
        'r2_avg': (r2_f0_rf + r2_spl_rf) / 2
    })

    # pr (joblib saved as a dict in the transfer learning script)
    pr_transfer = joblib.load(os.path.join(PR_DIR, 'transfer-models', 'PR_transfer_ensemble.pkl'))
    pr_male_f0 = pr_transfer['male_f0_model']
    pr_male_spl = pr_transfer['male_spl_model']
    pr_female_f0 = pr_transfer['female_f0_model']
    pr_female_spl = pr_transfer['female_spl_model']
    pr_x_scaler = pr_transfer['female_x_scaler']
    pr_f0_scaler = pr_transfer['female_f0_scaler']
    pr_spl_scaler = pr_transfer['female_spl_scaler']
    pr_poly_f0_female = pr_transfer['female_poly_f0']
    pr_poly_spl_female = pr_transfer['female_poly_spl']
    pr_poly_f0_male = pr_transfer['male_poly_f0']
    pr_poly_spl_male = pr_transfer['male_poly_spl']
    w_male, w_female = pr_transfer['weights']

    x_test_scaled = pr_x_scaler.transform(x_test)
    x_test_poly_f0 = pr_poly_f0_female.transform(x_test_scaled)
    x_test_poly_spl = pr_poly_spl_female.transform(x_test_scaled)
    x_test_poly_f0_male = pr_poly_f0_male.transform(x_test_scaled)
    x_test_poly_spl_male = pr_poly_spl_male.transform(x_test_scaled)

    f0_pred_male_scaled = pr_male_f0.predict(x_test_poly_f0_male)
    spl_pred_male_scaled = pr_male_spl.predict(x_test_poly_spl_male)
    f0_pred_female_scaled = pr_female_f0.predict(x_test_poly_f0)
    spl_pred_female_scaled = pr_female_spl.predict(x_test_poly_spl)

    f0_pred_male = pr_f0_scaler.inverse_transform(f0_pred_male_scaled.reshape(-1, 1))
    spl_pred_male = pr_spl_scaler.inverse_transform(spl_pred_male_scaled.reshape(-1, 1))
    y_pred_male = np.hstack([f0_pred_male, spl_pred_male])

    f0_pred_female = pr_f0_scaler.inverse_transform(f0_pred_female_scaled.reshape(-1, 1))
    spl_pred_female = pr_spl_scaler.inverse_transform(spl_pred_female_scaled.reshape(-1, 1))
    y_pred_female = np.hstack([f0_pred_female, spl_pred_female])

    y_pred_pr = w_male * y_pred_male + w_female * y_pred_female

    r2_f0_pr = r2_score(y_test_array[:, 0], y_pred_pr[:, 0])
    r2_spl_pr = r2_score(y_test_array[:, 1], y_pred_pr[:, 1])

    results_list.append({
        'sample_size': sample_size,
        'regressor': 'PR',
        'r2_f0': r2_f0_pr,
        'r2_spl': r2_spl_pr,
        'r2_avg': (r2_f0_pr + r2_spl_pr) / 2
    })

    print(f"  pr: r²_f0={r2_f0_pr:.3f}, r²_spl={r2_spl_pr:.3f}, avg={((r2_f0_pr + r2_spl_pr) / 2):.3f}")

results_df = pd.DataFrame(results_list)
os.makedirs(os.path.join(SCRIPT_DIR, 'figs'), exist_ok=True)
results_csv_path = os.path.join(SCRIPT_DIR, 'figs', 'transfer_comparison.csv')
results_df.to_csv(results_csv_path, index=False)

