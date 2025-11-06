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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.io import loadmat

np.random.seed(42)
tf.random.set_seed(42)

# create dirs to store transfer learning figs and models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_FIGS_DIR = os.path.join(SCRIPT_DIR, "transfer-figs")
TRANSFER_MODELS_DIR = os.path.join(SCRIPT_DIR, "transfer-models")

os.makedirs(TRANSFER_FIGS_DIR, exist_ok=True)
os.makedirs(TRANSFER_MODELS_DIR, exist_ok=True)

# thresholds to enable an early stop
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20, # epochs to wait wo improvment to be increased when looking at the full dataset
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # multiply LR by 0.5 when plateau detected
    patience=10,          # wait 10 epochs before reducing
    min_lr=1e-6,          # don't go below this learning rate
    verbose=1
)

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

# load pretrained
print("loading pre-trained model + scalers...")
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
MALE_MODEL_PATH = os.path.join(MODELS_DIR, 'standard_model.keras')
MALE_X_SCALER_PATH = os.path.join(MODELS_DIR, 'x_scaler_BCM.pkl')
MALE_Y_SCALER_PATH = os.path.join(MODELS_DIR, 'y_scaler_BCM.pkl')

base_model = load_model(MALE_MODEL_PATH, compile=False)
x_scaler_male = joblib.load(MALE_X_SCALER_PATH)
y_scaler_male = joblib.load(MALE_Y_SCALER_PATH)

base_model.summary() # model architecture

# load dataset
BASE_DIR = r'C:\Users\bglad\OneDrive\Desktop\Job\Fluid Flow\ml\BCM Model'
DATA_PATH = os.path.join(BASE_DIR, 'FemaleBCM.csv')
PARQUET_PATH = os.path.join(SCRIPT_DIR, 'FemaleNN_binary.parquet')

df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
df.to_parquet(PARQUET_PATH, compression="snappy") # convert to binary
df = pd.read_parquet(PARQUET_PATH)

df = df.dropna()

print(df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].describe())

# define axis
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# IMPROVEMENT 4: Proper train/val/test split
# Step 1: Split into train+val (80%) and test (20%)
x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Split train+val into train (72% total) and val (8% total)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# applying same male scalers to all three sets
print("scaling...")
x_train_scaled = x_scaler_male.transform(x_train)
x_val_scaled = x_scaler_male.transform(x_val)
x_test_scaled = x_scaler_male.transform(x_test)

y_train_scaled = y_scaler_male.transform(y_train)
y_val_scaled = y_scaler_male.transform(y_val)
y_test_scaled = y_scaler_male.transform(y_test)

print(f"Data split: Train={len(x_train)} ({len(x_train)/len(X)*100:.1f}%), Val={len(x_val)} ({len(x_val)/len(X)*100:.1f}%), Test={len(x_test)} ({len(x_test)/len(X)*100:.1f}%)")

# Method: Frozen Early Layers (Feature Extraction)
# - Freezes first 4 layers (2 Dense + 2 Dropout) trained on male data
# - Fine-tunes remaining layers on female dataset
# - Preserves general muscle activation patterns, adapts to sex-specific vocal outputs
#
# Future strategies: progressive unfreezing, full fine-tuning with low LR,
# domain adaptation, multi-task learning with sex indicator

model_frozen = tf.keras.models.clone_model(base_model)
model_frozen.set_weights(base_model.get_weights())

for layer in model_frozen.layers[:6]: # freeze first 4 layers
    layer.trainable = False

# compile frozen model
model_frozen.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.0001), # lower learning rate is obviously etter for transfer learning
    metrics=['mae']
)

history_frozen = model_frozen.fit(
    x_train_scaled, y_train_scaled,
    validation_data=(x_val_scaled, y_val_scaled),  # Now using separate validation set!
    epochs=100,  # more epochs since we have reduce_lr and early stopping
    batch_size=128,
    callbacks=[early_stopping, reduce_lr, PrintEpochProgress()],
    verbose=0
)

# predict with transfer model
y_pred_scaled = model_frozen.predict(x_test_scaled, verbose=0)
y_pred = y_scaler_male.inverse_transform(y_pred_scaled)

y_test_array = np.array(y_test)
print("\n=== Model Performance Metrics (on Test Set) ===")
print(f"\nF0 (Fundamental Frequency):")
print(f"  MAE:  {mean_absolute_error(y_test_array[:, 0], y_pred[:, 0]):.4f} Hz")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_array[:, 0], y_pred[:, 0])):.4f} Hz")
print(f"  R²:   {r2_score(y_test_array[:, 0], y_pred[:, 0]):.4f}")
print(f"\nSPL (Sound Pressure Level):")
print(f"  MAE:  {mean_absolute_error(y_test_array[:, 1], y_pred[:, 1]):.4f} Pa")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_array[:, 1], y_pred[:, 1])):.4f} Pa")
print(f"  R²:   {r2_score(y_test_array[:, 1], y_pred[:, 1]):.4f}")

# save transfer model
FEMALE_MODEL_PATH = os.path.join(TRANSFER_MODELS_DIR, 'NN_BCM_female_transfer.h5')
FEMALE_X_SCALER_PATH = os.path.join(TRANSFER_MODELS_DIR, 'x_scaler_female.pkl')
FEMALE_Y_SCALER_PATH = os.path.join(TRANSFER_MODELS_DIR, 'y_scaler_female.pkl')

model_frozen.save(FEMALE_MODEL_PATH)
joblib.dump(x_scaler_male, FEMALE_X_SCALER_PATH)
joblib.dump(y_scaler_male, FEMALE_Y_SCALER_PATH)
print(f"\nModel saved to: {FEMALE_MODEL_PATH}")
print(f"Scalers saved to: {TRANSFER_MODELS_DIR}")

# simple visual as per paper
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction using NN Regressor')
ax1.plot(np.array(y_test)[:,0], y_pred[:,0], 'o')
lin = [min(np.array(y_test)[:,0]), max(np.array(y_test)[:,0])]
ax1.plot(lin,lin)
ax1.set_title('$f_o$')
ax1.set_ylabel('Predicted (Hz)')
ax1.set_xlabel('Real (Hz)')
ax2.plot(np.array(y_test)[:,1], y_pred[:,1], 'o')
lin = [min(np.array(y_test)[:,1]), max(np.array(y_test)[:,1])]
ax2.plot(lin,lin)
ax2.set_ylabel('Predicted (Pa)')
ax2.set_xlabel('Real (Pa)')
ax2.set_title('SPL')
PREDICTIONS_FIG_PATH = os.path.join(TRANSFER_FIGS_DIR, 'predictions.png')
plt.savefig(PREDICTIONS_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\nPredictions plot saved to: {PREDICTIONS_FIG_PATH}")
plt.show()

fig_history, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig_history.suptitle('Training History - Transfer Learning')

# Plot loss curves
ax3.plot(history_frozen.history['loss'], label='Training Loss')
ax3.plot(history_frozen.history['val_loss'], label='Validation Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (MSE)')
ax3.set_title('Loss over Epochs')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot MAE curves
ax4.plot(history_frozen.history['mae'], label='Training MAE')
ax4.plot(history_frozen.history['val_mae'], label='Validation MAE')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('MAE')
ax4.set_title('Mean Absolute Error over Epochs')
ax4.legend()
ax4.grid(True, alpha=0.3)

TRAINING_CURVES_FIG_PATH = os.path.join(TRANSFER_FIGS_DIR, 'training_curves.png')
plt.savefig(TRAINING_CURVES_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"Training curves plot saved to: {TRAINING_CURVES_FIG_PATH}")
plt.show()
