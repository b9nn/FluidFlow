import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# multi output network to test which type of netwokr is better
def build_oheads(layers, idim):
    input = tf.keras.Input(shape=(idim,))
    x = Dense(layers[0], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input)
    x = Dropout(0.1)(x)

    for neurons in layers[1:]:
        x = Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.1)(x)

    f0 = Dense(1, name='f0', activation='linear')(x)
    spl = Dense(1, name='spl', activation='linear')(x)
    model = tf.keras.Model(inputs=input, outputs=[f0, spl]) # multi head
    optimized_opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimized_opt, loss={'f0': 'mse', 'spl': 'mse'}, metrics={'f0': 'mae', 'spl': 'mae'})
    return model

BCM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # VocalFoldRegression/BCM Model
df = pd.read_csv(os.path.join(BCM_DIR, 'MaleBCM.csv'))

# downsample for faster experimentation ~ 90000 rows
#df = df.sample(frac=0.25, random_state=42) use full dataset for now
df.to_parquet("./data_binary.parquet", compression="snappy") # convert to binary
df = pd.read_parquet("./data_binary.parquet")

# clean (not much needed)
df = df.dropna()

# define axis
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# Proper train/val/test split
# Step 1: Split into train+val (80%) and test (20%)
x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Split train+val into train (72% total) and val (8% total)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

print("scaling features...")
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_val_scaled = x_scaler.transform(x_val)
x_test_scaled = x_scaler.transform(x_test)

# adding y scaler
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

# for multi output model, need separate outputs
y_train_f0_scaled = y_train_scaled[:, 0:1]
y_train_spl_scaled = y_train_scaled[:, 1:2]
y_val_f0_scaled = y_val_scaled[:, 0:1]
y_val_spl_scaled = y_val_scaled[:, 1:2]
y_test_f0_scaled = y_test_scaled[:, 0:1]
y_test_spl_scaled = y_test_scaled[:, 1:2]

print(f"Data split: Train={len(x_train)} ({len(x_train)/len(X)*100:.1f}%), Val={len(x_val)} ({len(x_val)/len(X)*100:.1f}%), Test={len(x_test)} ({len(x_test)/len(X)*100:.1f}%)")


early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20, # epochs to wait wo improvment to be increased when looking at the full dataset
    restore_best_weights=True,
    verbose=1,
)

# learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # reduce LR by half
    patience=5,  # wait 5 epochs before reducing (make sure this is less than earlystoping paitence)
    min_lr=1e-7,
    verbose=1
)

layers = [512, 256, 128, 64]       # three hidden layers with progressively fewer neurons
idim = x_train.shape[1]       # number of input features
odim = y_train.shape[1]       # usually 1 for regression

multi_modelNN = build_oheads(layers, idim)

print("Training multi-output model...")
history_multi = multi_modelNN.fit(
    x_train_scaled,
    {'f0': y_train_f0_scaled, 'spl': y_train_spl_scaled},
    validation_data=(x_val_scaled, {'f0': y_val_f0_scaled, 'spl': y_val_spl_scaled}),  # Now using separate validation set!
    epochs=150,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

predictions_multi = multi_modelNN.predict(x_test_scaled, verbose=0)
y_pred_f0_scaled = predictions_multi[0]
y_pred_spl_scaled = predictions_multi[1]

# inverse scale 
y_pred_multi_f0 = y_scaler.inverse_transform(
    np.concatenate([y_pred_f0_scaled, np.zeros_like(y_pred_f0_scaled)], axis=1)
)[:, 0:1]
y_pred_multi_spl = y_scaler.inverse_transform(
    np.concatenate([np.zeros_like(y_pred_spl_scaled), y_pred_spl_scaled], axis=1)
)[:, 1:2]
y_pred_multi = np.concatenate([y_pred_multi_f0, y_pred_multi_spl], axis=1)

# Calculate performance metrics on TEST set
mae_f0_multi = mean_absolute_error(y_test.iloc[:, 0], y_pred_multi[:, 0])
mae_spl_multi = mean_absolute_error(y_test.iloc[:, 1], y_pred_multi[:, 1])
rmse_f0_multi = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred_multi[:, 0]))
rmse_spl_multi = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred_multi[:, 1]))
r2_f0_multi = r2_score(y_test.iloc[:, 0], y_pred_multi[:, 0])
r2_spl_multi = r2_score(y_test.iloc[:, 1], y_pred_multi[:, 1])

print("\n=== Multi-Output Model Performance Metrics (on Test Set) ===")
print(f"\nF0 (Fundamental Frequency):")
print(f"  MAE:  {mae_f0_multi:.4f} Hz")
print(f"  RMSE: {rmse_f0_multi:.4f} Hz")
print(f"  R²:   {r2_f0_multi:.4f}")
print(f"\nSPL (Sound Pressure Level):")
print(f"  MAE:  {mae_spl_multi:.4f} Pa")
print(f"  RMSE: {rmse_spl_multi:.4f} Pa")
print(f"  R²:   {r2_spl_multi:.4f}")

os.makedirs("./figs", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Multi-Output Model', fontsize=16, fontweight='bold')

# Multi-Output Model - F0
ax = axes[0]
ax.plot(np.array(y_test)[:, 0], y_pred_multi[:, 0], 'o', alpha=0.5, markersize=3, color='darkorange')
lin = [min(np.array(y_test)[:, 0]), max(np.array(y_test)[:, 0])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Multi-Output Model - $f_o$ (R² = {r2_f0_multi:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Hz)')
ax.set_xlabel('Real (Hz)')
ax.legend()
ax.grid(True, alpha=0.3)

# Multi-Output Model - SPL
ax = axes[1]
ax.plot(np.array(y_test)[:, 1], y_pred_multi[:, 1], 'o', alpha=0.5, markersize=3, color='darkorange')
lin = [min(np.array(y_test)[:, 1]), max(np.array(y_test)[:, 1])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Multi-Output Model - SPL (R² = {r2_spl_multi:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Pa)')
ax.set_xlabel('Real (Pa)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./figs/multioutput_comparison.png", dpi=300, bbox_inches='tight')
