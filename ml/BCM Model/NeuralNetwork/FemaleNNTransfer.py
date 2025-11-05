import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.io import loadmat

# create dirs to store figs and models
os.makedirs("./figs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# thresholds to enable an early stop
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20, # epochs to wait wo improvment to be increased when looking at the full dataset
    restore_best_weights=True,
    verbose=1,
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
base_model = load_model('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/NeuralNetwork/models/NN_BCM.h5', compile=False)
x_scaler_male = joblib.load('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/NeuralNetwork/models/x_scaler_BCM.pkl')
y_scaler_male = joblib.load('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/NeuralNetwork/models/y_scaler_BCM.pkl')

base_model.summary() # model architecture

# load dataset
df = pd.read_csv('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/FemaleBCM.csv')
df.to_parquet("./FemaleNN_binary.parquet", compression="snappy") # convert to binary
df = pd.read_parquet("./FemaleNN_binary.parquet")

df = df.dropna()

# define axis
X = df[['a_CT', 'a_TA', 'PS']]  
Y = df[['F0', 'SPL']]

# split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 20% to testing, 80% to training

# applying same male scalers
print("scaling...")
x_train_scaled = x_scaler_male.transform(x_train)
x_test_scaled = x_scaler_male.transform(x_test)
y_train_scaled = y_scaler_male.transform(y_train)
y_test_scaled = y_scaler_male.transform(y_test)

# Method: Frozen Early Layers (Feature Extraction)
# - Freezes first 4 layers (2 Dense + 2 Dropout) trained on male data
# - Fine-tunes remaining layers on female dataset
# - Preserves general muscle activation patterns, adapts to sex-specific vocal outputs
# 
# Future strategies: progressive unfreezing, full fine-tuning with low LR, 
# domain adaptation, multi-task learning with sex indicator

model_frozen = tf.keras.models.clone_model(base_model)
model_frozen.set_weights(base_model.get_weights())

for layer in model_frozen.layers[:4]: # freeze first 4 layers
    layer.trainable = False

# compile frozen model
model_frozen.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mae']
)

history_frozen = model_frozen.fit(
    x_train_scaled, y_train_scaled,
    validation_data=(x_test_scaled, y_test_scaled),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping, PrintEpochProgress()],
    verbose=0
)

# predict with transfer model
y_pred_scaled = model_frozen.predict(x_test_scaled, verbose=0)
y_pred = y_scaler_male.inverse_transform(y_pred_scaled)

# save transfer model
model_frozen.save('./models/NN_BCM_female_transfer.h5')
joblib.dump(x_scaler_male, './models/x_scaler_female.pkl')
joblib.dump(y_scaler_male, './models/y_scaler_female.pkl')

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
plt.savefig("./figs/predictions.png", dpi=300, bbox_inches='tight') # save fig
plt.show()
