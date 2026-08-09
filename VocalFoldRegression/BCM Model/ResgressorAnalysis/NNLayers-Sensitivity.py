'''
this script was designed to test the senstivity of the various transfer learning models
when it came to dataset size. hopefully this will provide some insight into how we
can apply finite element data, which requires heavy compute, so knowing how different
sample sizes preform is important

'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import mean_absolute_error, r2_score
# create dirs to store transfer learning figs and models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_FIGS_DIR = os.path.join(SCRIPT_DIR, "transfer-figs")
# models are in the NeuralNetwork folder, not ResgressorAnalysis
NN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "NeuralNetwork")
TRANSFER_MODELS_DIR = os.path.join(NN_DIR, "transfer-models")

df = pd.read_parquet(os.path.join(NN_DIR, 'FemaleNN_binary.parquet'))
df = df.dropna()

df = df.dropna()
df = df[df['ACFL'] > 30]  # apply quality filter

print(f"dataset size: {len(df)}")

# define sample sizes to test (from small to full dataset)
sample_sizes = [25, 50, 100, 250, 500, 750, len(df)]
fro_config = [2, 4, 5, 6]  # frozen layer configurations to test

# load all pre-trained models
print("\nloading trained models...")
models = {}
for fro in fro_config:
    model_path = os.path.join(TRANSFER_MODELS_DIR, f'transfer_frozen_{fro}_layers.keras')
    models[fro] = load_model(model_path, compile=False) # add dict entry

# load the original scalers used during training (critical for consistent scaling)
x_scaler_orig = joblib.load(os.path.join(TRANSFER_MODELS_DIR, 'x_scaler_female.pkl'))
y_scaler_orig = joblib.load(os.path.join(TRANSFER_MODELS_DIR, 'y_scaler_female.pkl'))

# storage for results
results_list = []

# test each sample size
for sample_size in sample_sizes:
    print(f"\ntesting with {sample_size} samples ({sample_size/len(df)*100:.1f}% of data)")

    # sample data or use full dataset
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()

    # define features and targets
    X_sample = df_sample[['a_CT', 'a_TA', 'PS']]
    Y_sample = df_sample[['F0', 'SPL']]

    # split into train/val/test sets
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        X_sample, Y_sample, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # use the ORIGINAL scalers from training (not new ones!)
    # this ensures models receive data in the same format they were trained on
    x_test_scaled = x_scaler_orig.transform(x_test)
    y_test_array = np.array(y_test)  # keep unscaled for metric calculation

    # test each frozen layer configuration
    for fro in fro_config:
        # get pre-trained model (no retraining, just evaluate)
        model = models[fro]

        # make predictions on test set
        y_pred_scaled = model.predict(x_test_scaled, verbose=0)
        y_pred = y_scaler_orig.inverse_transform(y_pred_scaled)  # use original scaler

        # calculate performance metrics (y_test_array already defined above)
        r2_f0 = r2_score(y_test_array[:, 0], y_pred[:, 0])
        r2_spl = r2_score(y_test_array[:, 1], y_pred[:, 1])
        mae_f0 = mean_absolute_error(y_test_array[:, 0], y_pred[:, 0])
        mae_spl = mean_absolute_error(y_test_array[:, 1], y_pred[:, 1])

        # store results for this configuration
        results_list.append({
            'sample_size': sample_size,
            'frozen_layers': fro,
            'r2_f0': r2_f0,
            'r2_spl': r2_spl,
            'mae_f0': mae_f0,
            'mae_spl': mae_spl,
            'test_samples': len(x_test)
        })

        print(f"  {fro} frozen: r²_f0={r2_f0:.3f}, r²_spl={r2_spl:.3f}")

# convert results to dataframe and save as csv
results_df = pd.DataFrame(results_list)
results_csv_path = os.path.join(TRANSFER_FIGS_DIR, 'sample_size_sensitivity.csv')
results_df.to_csv(results_csv_path, index=False)

# create visualization with 4 subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('transfer learning sample size sensitivity', fontsize=16, fontweight='bold')

# color scheme for different frozen layer configs
colors = {'2': 'steelblue', '4': 'darkorange', '5': 'green', '6': 'red'}

# plot each frozen layer configuration
for fro in fro_config:
    subset = results_df[results_df['frozen_layers'] == fro]
    color = colors[str(fro)]

    # r² for f0
    axes[0, 0].plot(subset['sample_size'], subset['r2_f0'], marker='o',
                    label=f'{fro} frozen', color=color, linewidth=2)
    # r² for spl
    axes[0, 1].plot(subset['sample_size'], subset['r2_spl'], marker='o',
                    label=f'{fro} frozen', color=color, linewidth=2)
    # mae for f0
    axes[1, 0].plot(subset['sample_size'], subset['mae_f0'], marker='o',
                    label=f'{fro} frozen', color=color, linewidth=2)
    # mae for spl
    axes[1, 1].plot(subset['sample_size'], subset['mae_spl'], marker='o',
                    label=f'{fro} frozen', color=color, linewidth=2)

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
fig_path = os.path.join(TRANSFER_FIGS_DIR, 'sample_size_sensitivity.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
