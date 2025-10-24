import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Layer,Dropout,Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score


# october 24th, 2025
# - not creating an insane model yet just laying the groundwork 

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

    model.add(Dense(layers[0], input_dim=idim, activation=act)) 
    model.add(Dropout(0.2)) # add random dropout to prevent overfitting (first layer)
    for lay in layers[1:]:
        model.add(Dense(lay, activation=act))
        model.add(Dropout(0.2)) # add random dropout to prevent overfitting

    # output layer
    model.add(Dense(odim, activation=oact))

    # compile model
    model.compile(loss=loss, optimizer=opt, metrics=['mae'])

    return model 

# full dataset
df = pd.read_csv('../DataTBCM_GVV_Feature_new(in).csv')

# downsample for faster experimentation ~ 90000 rows
df = df.sample(frac=0.25, random_state=42)
df.to_parquet("./data_binary.parquet", compression="snappy") # convert to binary
df = pd.read_parquet("./data_binary.parquet")

# clean (not much needed)
df = df.dropna()

# define axis
X = df[['a_CT', 'a_TA', 'PS']]  
Y = df[['F0', 'SPL']]

# split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 20% to testing, 80% to training

# Create and fit scaler
print("scaling features...")
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

# adding y scaler
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# === training ===

# early stopping during training
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10, # epochs to wait wo improvment to be increased when looking at the full dataset
    restore_best_weights=True,
    verbose=1
)

layers = [128, 128, 128]       # three hidden layers with progressively fewer neurons
idim = x_train.shape[1]       # number of input features
odim = y_train.shape[1]       # usually 1 for regression

modelNN = build(layers, idim, odim, act='relu', oact='linear', opt='adam', loss='mse')

history = modelNN.fit(
    x_train_scaled, y_train_scaled,
    validation_data=(x_test_scaled, y_test_scaled), # takes care of split
    epochs=75, # go up to 100 eventually
    batch_size=64, # for 90000 samples --> to be moved up
    callbacks=[early_stopping],
    verbose=1
)


# predict
y_pred_scaled = modelNN.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)  # convert back to original scale

results = {'F0': [], 'SPL': []}
for i in range(len(list(results.keys()))):
  results[list(results.keys())[i]] = [mean_absolute_error(y_true=np.array(y_test)[:,(i)],y_pred=y_pred[:,(i)]), mean_squared_error(y_true=np.array(y_test)[:,i],y_pred=y_pred[:,i]), mean_squared_error(y_true=np.array(y_test)[:,i],y_pred=y_pred[:,i],squared=False), r2_score(np.array(y_test)[:,i], y_pred[:,i])]
pd.DataFrame.from_dict(results, orient='index', columns=['MAE','MSE', 'RMSE', 'R2'])

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
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model MAE')
plt.tight_layout()
plt.show()

# save model + scalers
joblib.dump(x_scaler, 'models/x_scaler_BCM.pkl')
joblib.dump(y_scaler, 'models/y_scaler_BCM.pkl')  
modelNN.save('models/NN_BCM.h5')
