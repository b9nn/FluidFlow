import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import keras.backend as K
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

# custom function to track r
def r2_metric(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# custom class to plug into callbacks to print metrics every epoch
class PrintEpochProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs
        print(
            f"Epoch {epoch} | "
            f"Loss: {logs.get('loss'):.5f} | "
        )

# load pretrained
print("loading pre-trained model + scalers...")
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
MALE_MODEL_PATH = os.path.join(MODELS_DIR, 'standard_model.keras')

base_model = load_model(MALE_MODEL_PATH, compile=False)
x_scalar_female = StandardScaler()
y_scalar_female = StandardScaler()

# load dataset
df = pd.read_parquet(os.path.join(SCRIPT_DIR, 'FemaleNN_binary.parquet'))
# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()

# Apply ACFL > 30 filter to match baseline female model (critical for data quality)
print(f"Dataset size before ACFL filter: {len(df)}")
df = df[df['ACFL'] > 30]
print(f"Dataset size after ACFL > 30 filter: {len(df)} ({len(df)/1331*100:.1f}% retained)")

print(df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].describe())

# define axis
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# split train+val into train (72% total) and val (8% total)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# scaling with separate scalers for f0 and spl
print("scaling...")
x_scalar_female = StandardScaler()
x_train_scaled = x_scalar_female.fit_transform(x_train)
x_val_scaled = x_scalar_female.transform(x_val)
x_test_scaled = x_scalar_female.transform(x_test)

# separate scalers for f0 and spl
f0_scaler_female = StandardScaler()
spl_scaler_female = StandardScaler()

# fit on training data
f0_train_scaled = f0_scaler_female.fit_transform(y_train[['F0']])
spl_train_scaled = spl_scaler_female.fit_transform(y_train[['SPL']])
y_train_scaled = np.hstack([f0_train_scaled, spl_train_scaled])

# transform val data
f0_val_scaled = f0_scaler_female.transform(y_val[['F0']])
spl_val_scaled = spl_scaler_female.transform(y_val[['SPL']])
y_val_scaled = np.hstack([f0_val_scaled, spl_val_scaled])

# transform test data
f0_test_scaled = f0_scaler_female.transform(y_test[['F0']])
spl_test_scaled = spl_scaler_female.transform(y_test[['SPL']])
y_test_scaled = np.hstack([f0_test_scaled, spl_test_scaled])

# Transfer Learning Strategy: Partial Freezing

learning_rate = 0.005
frozen_layers = 4 # how many layers you want to be frozen (staring with the input)
fro_config = [2, 4, 5, 6]  # testing various layers
ret = []
saved_models = []

for fro in fro_config: # testing various amounts of freezing
    # clone base model and freeze first x layers
    transfer_model = tf.keras.models.clone_model(base_model)
    transfer_model.set_weights(base_model.get_weights())

    for layer in transfer_model.layers[:fro]:
        layer.trainable = False

    transfer_model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate),  # higher LR works better with partial freezing
        metrics=['mae']
    )

    # Train model
    history = transfer_model.fit(
        x_train_scaled, y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=50,
        batch_size=128,
        # callback scausing issues, debug later
       # callbacks=[early_stopping, reduce_lr, PrintEpochProgress()] 
    )

    # predictions
    y_pred_scaled = transfer_model.predict(x_test_scaled, verbose=0)
    # inverse transform using separate scalers
    f0_pred = f0_scaler_female.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1))
    spl_pred = spl_scaler_female.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1))
    y_pred = np.hstack([f0_pred, spl_pred])  # scale back to original metric

    # calculate metrics
    y_test_array = np.array(y_test)

    # f0 is first col of y, spl is second. syntax tax is (np_array[amount of rows, col]) ":" grabs all rows
    r2_f0 = r2_score(y_test_array[:, 0], y_pred[:, 0])
    r2_spl = r2_score(y_test_array[:, 1], y_pred[:, 1])

    ret.append({
        'frozen_layers': fro,
        'r2f0': r2_f0,
        'r2spl': r2_spl
    })

    # saving model for data senstivity analysis
    path = os.path.join(TRANSFER_MODELS_DIR, f'transfer_frozen_{fro}_layers.keras')
    transfer_model.save(path)

# save scalers for sensitivity analysis (needed for consistent scaling)
joblib.dump(x_scalar_female, os.path.join(TRANSFER_MODELS_DIR, 'x_scaler_female.pkl'))
joblib.dump(f0_scaler_female, os.path.join(TRANSFER_MODELS_DIR, 'f0_scaler_female.pkl'))
joblib.dump(spl_scaler_female, os.path.join(TRANSFER_MODELS_DIR, 'spl_scaler_female.pkl'))
print(f"\n✓ scalers saved to: {TRANSFER_MODELS_DIR}")

for r in ret:
    print(f"layer = {r['frozen_layers']}, r2f0 = {r['r2f0']}, r2spl = {r['r2spl']}")

best_f0 = max(ret, key=lambda x: x['r2f0'])
print(f'best f0 = {best_f0['frozen_layers']} layers, achived r2 = {best_f0['r2f0']}')

best_spl = max(ret, key=lambda x : x['r2spl'])
print(f'best spl = {best_f0['frozen_layers']} layers, achived r2 = {best_spl['r2spl']}')

best_overall = max(ret, key=lambda y: (y['r2f0'] + y['r2f0']) / 2)
print(f'best overall = {best_f0['frozen_layers']} layers, achived avg r2 = {(best_overall['r2f0'] + best_overall['r2spl']) / 2}')

'''
# Save transfer model
FEMALE_MODEL_PATH = os.path.join(TRANSFER_MODELS_DIR, 'NN_BCM_female_transfer.keras')
FEMALE_X_SCALER_PATH = os.path.join(TRANSFER_MODELS_DIR, 'x_scaler_female.pkl')
FEMALE_Y_SCALER_PATH = os.path.join(TRANSFER_MODELS_DIR, 'y_scaler_female.pkl')

transfer_model.save(FEMALE_MODEL_PATH)
joblib.dump(x_scalar_female, FEMALE_X_SCALER_PATH)
joblib.dump(y_scalar_female, FEMALE_Y_SCALER_PATH)
print(f"\nModel saved to: {FEMALE_MODEL_PATH}")
print(f"Scalers saved to: {TRANSFER_MODELS_DIR}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Transfer Learning Model - Prediction Performance', fontsize=14, fontweight='bold')

# F0 predictions
ax1.plot(np.array(y_test)[:,0], y_pred[:,0], 'o', alpha=0.5, markersize=4, color='steelblue')
lin = [min(np.array(y_test)[:,0]), max(np.array(y_test)[:,0])]
ax1.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax1.set_title(f'$f_o$ (R² = {r2_f0:.4f})', fontweight='bold')
ax1.set_ylabel('Predicted (Hz)')
ax1.set_xlabel('Real (Hz)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# SPL predictions
ax2.plot(np.array(y_test)[:,1], y_pred[:,1], 'o', alpha=0.5, markersize=4, color='darkorange')
lin = [min(np.array(y_test)[:,1]), max(np.array(y_test)[:,1])]
ax2.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax2.set_title(f'SPL (R² = {r2_spl:.4f})', fontweight='bold')
ax2.set_ylabel('Predicted (Pa)')
ax2.set_xlabel('Real (Pa)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
PREDICTIONS_FIG_PATH = os.path.join(TRANSFER_FIGS_DIR, 'predictions.png')
plt.savefig(PREDICTIONS_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\nPredictions plot saved to: {PREDICTIONS_FIG_PATH}")
plt.show()

fig_history, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig_history.suptitle('Training History - Transfer Learning Model', fontsize=14, fontweight='bold')

# Plot loss curves
ax3.plot(history.history['loss'], label='Training Loss')
ax3.plot(history.history['val_loss'], label='Validation Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (MSE)')
ax3.set_title('Loss over Epochs')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot MAE curves
ax4.plot(history.history['mae'], label='Training MAE')
ax4.plot(history.history['val_mae'], label='Validation MAE')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('MAE')
ax4.set_title('Mean Absolute Error over Epochs')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
TRAINING_CURVES_FIG_PATH = os.path.join(TRANSFER_FIGS_DIR, 'training_curves.png')
plt.savefig(TRAINING_CURVES_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"Training curves plot saved to: {TRAINING_CURVES_FIG_PATH}")
plt.show()
'''
