import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Layer,Dropout,Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

# october 24th, 2025
# - not creating an insane model yet just laying the groundwork 

# to save script output
os.makedirs("./figs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# build nueral network
def build(layers, idim, odim = 1, act = 'relu', oact = 'linear', opt = 'adam', loss = 'mse'):
    '''
    parameters:

    layers : list[int], each index = layer, and associated int is neurons
    idim (input dimension): int, num features the network takes
    odim (output dimension): int, num outputs (1 for regression)
    act (activation): str, activation function for neurons, relu captures non-linear, saturated relationships best (good for muscle activation)
    oact (output activation): str, linear by default, best for regression
    opt (optimizer): str, used to update weights during training, adam is best by default
    loss : str, function to measure prediction error, mean squared error (mse) is standard
    '''

    model = Sequential() # simple linear stack

    # define input layer

    model.add(Dense(layers[0], input_dim=idim, activation=act, kernel_regularizer=tf.keras.regularizers.l2(0.001))) 
    model.add(Dropout(0.1)) # add random dropout to prevent overfitting (first layer)
    for lay in layers[1:]:
        model.add(Dense(lay, activation=act, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(Dropout(0.1)) # add random dropout to prevent overfitting

    # output layer
    model.add(Dense(odim, activation=oact))

    optimized_opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # compile model
    model.compile(loss=loss, optimizer=optimized_opt, metrics=['mae'])

    return model 

# custom class to plug into callbacks to print metrics every epoch
class PrintEpochProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs
        print(
            f"Epoch {epoch} | "
            f"Loss: {logs.get('loss'):.5f} | "
            f"Val Loss: {logs.get('val_loss'):.5f} | "
            f"MAE: {logs.get('mae'):.5f} | "
            f"Val MAE: {logs.get('val_mae'):.5f}"
        )

# full dataset
df = pd.read_csv('../MaleBCM.csv')

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

# create and fit scaler on training data only
print("scaling features...")
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_val_scaled = x_scaler.transform(x_val)
x_test_scaled = x_scaler.transform(x_test)

# separate scalers for f0 and spl
f0_scaler = StandardScaler()
spl_scaler = StandardScaler()

# fit on training data
f0_train_scaled = f0_scaler.fit_transform(y_train[['F0']])
spl_train_scaled = spl_scaler.fit_transform(y_train[['SPL']])
y_train_scaled = np.hstack([f0_train_scaled, spl_train_scaled])

# transform val data
f0_val_scaled = f0_scaler.transform(y_val[['F0']])
spl_val_scaled = spl_scaler.transform(y_val[['SPL']])
y_val_scaled = np.hstack([f0_val_scaled, spl_val_scaled])

# transform test data
f0_test_scaled = f0_scaler.transform(y_test[['F0']])
spl_test_scaled = spl_scaler.transform(y_test[['SPL']])
y_test_scaled = np.hstack([f0_test_scaled, spl_test_scaled])

print(f"Data split: Train={len(x_train)} ({len(x_train)/len(X)*100:.1f}%), Val={len(x_val)} ({len(x_val)/len(X)*100:.1f}%), Test={len(x_test)} ({len(x_test)/len(X)*100:.1f}%)")

# === training ===

# early stopping during training
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

modelNN = build(layers, idim, odim, act='relu', oact='linear', opt='adam', loss='mse')

print("training...")
history = modelNN.fit(
    x_train_scaled, y_train_scaled,
    validation_data=(x_val_scaled, y_val_scaled),  # Now using separate validation set!
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr, PrintEpochProgress()],
    verbose=1
)

y_pred_scaled = modelNN.predict(x_test_scaled, verbose=0)
# inverse transform using separate scalers
f0_pred = f0_scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1))
spl_pred = spl_scaler.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1))
y_pred = np.hstack([f0_pred, spl_pred])  # combine back to original scale

# Calculate performance metrics on TEST set
mae_f0 = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
mae_spl = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
rmse_f0 = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]))
rmse_spl = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]))
r2_f0 = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_spl = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

print("\n=== Model Performance Metrics (on Test Set) ===")
print(f"\nF0 (Fundamental Frequency):")
print(f"  MAE:  {mae_f0:.4f} Hz")
print(f"  RMSE: {rmse_f0:.4f} Hz")
print(f"  R²:   {r2_f0:.4f}")
print(f"\nSPL (Sound Pressure Level):")
print(f"  MAE:  {mae_spl:.4f} Pa")
print(f"  RMSE: {rmse_spl:.4f} Pa")
print(f"  R²:   {r2_spl:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Model Comparison: Standard vs Multi-Output', fontsize=16, fontweight='bold')

# Standard Model - F0
ax = axes[0, 0]
ax.plot(np.array(y_test)[:, 0], y_pred[:, 0], 'o', alpha=0.5, markersize=3, color='steelblue')
lin = [min(np.array(y_test)[:, 0]), max(np.array(y_test)[:, 0])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Standard Model - $f_o$ (R² = {r2_f0:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Hz)')
ax.set_xlabel('Real (Hz)')
ax.legend()
ax.grid(True, alpha=0.3)

# Standard Model - SPL
ax = axes[0, 1]
ax.plot(np.array(y_test)[:, 1], y_pred[:, 1], 'o', alpha=0.5, markersize=3, color='steelblue')
lin = [min(np.array(y_test)[:, 1]), max(np.array(y_test)[:, 1])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Standard Model - SPL (R² = {r2_spl:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Pa)')
ax.set_xlabel('Real (Pa)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./figs/model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

'''
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Model Comparison: Standard vs Multi-Output', fontsize=16, fontweight='bold')

# Standard Model - F0
ax = axes[0, 0]
ax.plot(np.array(y_test)[:, 0], y_pred[:, 0], 'o', alpha=0.5, markersize=3, color='steelblue')
lin = [min(np.array(y_test)[:, 0]), max(np.array(y_test)[:, 0])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Standard Model - $f_o$ (R² = {r2_f0:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Hz)')
ax.set_xlabel('Real (Hz)')
ax.legend()
ax.grid(True, alpha=0.3)

# Standard Model - SPL
ax = axes[0, 1]
ax.plot(np.array(y_test)[:, 1], y_pred[:, 1], 'o', alpha=0.5, markersize=3, color='steelblue')
lin = [min(np.array(y_test)[:, 1]), max(np.array(y_test)[:, 1])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Standard Model - SPL (R² = {r2_spl:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Pa)')
ax.set_xlabel('Real (Pa)')
ax.legend()
ax.grid(True, alpha=0.3)

# Multi-Output Model - F0
ax = axes[1, 0]
ax.plot(np.array(y_test)[:, 0], y_pred_multi[:, 0], 'o', alpha=0.5, markersize=3, color='darkorange')
lin = [min(np.array(y_test)[:, 0]), max(np.array(y_test)[:, 0])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Multi-Output Model - $f_o$ (R² = {r2_f0_multi:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Hz)')
ax.set_xlabel('Real (Hz)')
ax.legend()
ax.grid(True, alpha=0.3)

# Multi-Output Model - SPL
ax = axes[1, 1]
ax.plot(np.array(y_test)[:, 1], y_pred_multi[:, 1], 'o', alpha=0.5, markersize=3, color='darkorange')
lin = [min(np.array(y_test)[:, 1]), max(np.array(y_test)[:, 1])]
ax.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Multi-Output Model - SPL (R² = {r2_spl_multi:.4f})', fontweight='bold')
ax.set_ylabel('Predicted (Pa)')
ax.set_xlabel('Real (Pa)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./figs/model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# save model + scalers
joblib.dump(x_scaler, './models/x_scaler_BCM.pkl')
joblib.dump(f0_scaler, './models/f0_scaler_BCM.pkl')
joblib.dump(spl_scaler, './models/spl_scaler_BCM.pkl')
modelNN.save('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/NeuralNetwork/models')
'''

joblib.dump(x_scaler, './models/x_scaler_BCM.pkl')
joblib.dump(f0_scaler, './models/f0_scaler_BCM.pkl')
joblib.dump(spl_scaler, './models/spl_scaler_BCM.pkl')  
modelNN.save('./models/standard_model.keras')
