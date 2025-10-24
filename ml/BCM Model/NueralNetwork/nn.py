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

    model.add(Dense(layers[0], input_dim=idim, activation='act')) 
    for lay in layers[1:]:
        model.add(Dense(lay, activation=act))

    # output layer
    model.add(Dense(odim, activation=oact))

    # compile model
    model.compile(loss=loss, optimizer=opt, metrics=['mae'])

    return model 

# full dataset
df = pd.read_csv('../DataTBCM_GVV_Feature_new(in).csv')

# downsample for faster experimentation ~ 25000 rows
df = df.sample(frac=0.07, random_state=42)
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

# === training ===

# early stopping during training
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

layers = [128, 64, 32]       # three hidden layers with progressively fewer neurons
idim = x_train.shape[1]       # number of input features
odim = y_train.shape[1]       # usually 1 for regression

modelNN = build(layers, idim, odim, act='relu', oact='linear', opt='adam', loss='mse')

history = modelNN.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)
