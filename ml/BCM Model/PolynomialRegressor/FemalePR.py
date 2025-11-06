'''
excerpt: (lines 463 - 466)
For PR, closely related to linear models, two-stage transfer learningapproaches have also been developed: the first stage involves training on the base dataset,
followed by transfer through methods such as Lasso-type regularization (Li et al., 2022) or
fine-tuning with gradient descent iterations to minimize error on the subject-specific dataset
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.multioutput import MultiOutputRegressor

def create_new_dataset(data_dict,N):
    df_aux = {'a_CT': [], 'a_TA': [], 'Ps': [], 'F0': [], 'SPL': []}
    for i in range(0,401, N): #index Ps
        for j in range(0,1001, N): #index a_TA
            for k in range(0,1001, N): #index a_CT
                df_aux['a_CT'].append(data_dict['lookuptable']['aCTrange'][k])
                df_aux['a_TA'].append(data_dict['lookuptable']['aTArange'][j])
                df_aux['Ps'].append(data_dict['lookuptable']['Psrange'][i])
                df_aux['F0'].append(data_dict['lookuptable']['F0'][i,j,k])
                df_aux['SPL'].append(data_dict['lookuptable']['SPL'][i,j,k])
    df = pd.DataFrame(df_aux)
    return df

data_file = 'C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/mujer_v2.mat'
data_dict = loadmat(data_file)
df = create_new_dataset(data_dict,10)

x_scaler = joblib.load("C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/PolynomialRegressor/models/x_scaler_BCM.pkl")
y_scaler = joblib.load("C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/PolynomialRegressor/models/y_scaler_BCM.pkl")
base_model = joblib.load("C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/PolynomialRegressor/models/firstPR")


# Polynomial transformer (must match degree from original training)
degree = 12
poly = PolynomialFeatures(degree=degree, include_bias=False)

subject_df = pd.read_csv("C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/SubjectBCM.csv")
subject_df = subject_df.dropna()

X_subject = subject_df[['a_CT', 'a_TA', 'PS']]
Y_subject = subject_df[['F0', 'SPL']]

X_subject_scaled = x_scaler.transform(X_subject)
Y_subject_scaled = y_scaler.transform(Y_subject)

# polynomial features
X_subject_poly = poly.fit_transform(X_subject_scaled)  # we can use fit_transform since poly is stateless

# use Lasso with small alpha to slightly adjust base model coefficients
lasso_model = MultiOutputRegressor(Lasso(alpha=0.001, max_iter=10000))

# Optional: warm-start using base model coefficients (if desired)
# Here we simply fit Lasso on the subject-specific data
lasso_model.fit(X_subject_poly, Y_subject_scaled)

y_pred_scaled = lasso_model.predict(X_subject_poly)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

joblib.dump(lasso_model, "C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/PolynomialRegressor/models/subjectPR_model.pkl")
print("Transfer learning completed and model saved!")
