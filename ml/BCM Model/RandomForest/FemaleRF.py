'''
excerpt: (lines 460-462)
For RF, a similar strategy can be applied by constructing an ensemble of regressors trained on each dataset (Gu et al., 2022),
with predictions combined through weighted estimates of the two datasets, or by refining
branch weights using subject-specific data once the baseline architecture has been defined
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import joblib
import os
import h5py
import mat73


base = 'C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/RandomForest/models/RF_BCM.pkl'
scaler = 'C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/RandomForest/models/x_scaler_BCM.pkl'

# load models
male_rf = joblib.load(base)
x_scaler = joblib.load(scaler)

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

df= df.dropna()

variables = ['a_CT', 'a_TA', 'Ps']
covar = ['F0', 'SPL']
X = df[variables]
Y = df[covar]

female_scaled = x_scaler.transform(X)

female_rf = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
))

print("training...")
female_rf.fit(female_scaled, Y)

# transfer learning 
w_base = 0.8 # from baseline
w_new = 0.2 # from female dataset

# combine predictions
y_pred_male = male_rf.predict(female_scaled)
y_pred_female = female_rf.predict(female_scaled)
y_pred_ensemble = w_base * y_pred_male + w_new * y_pred_female

r2 = r2_score(female_rf, y_pred_ensemble, multioutput='uniform_average')
mse = mean_squared_error(female_rf, y_pred_ensemble, multioutput='uniform_average')
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Transfer-Learned Random Forest Ensemble')
ax1.scatter(Y['F0'], y_pred_ensemble[:, 0], alpha=0.5)
ax1.plot([Y['F0'].min(), Y['F0'].max()], [Y['F0'].min(), Y['F0'].max()], 'r--')
ax1.set_title('F0 Prediction')
ax1.set_xlabel('Actual (Hz)')
ax1.set_ylabel('Predicted (Hz)')
ax2.scatter(Y['SPL'], y_pred_ensemble[:, 1], alpha=0.5)
ax2.plot([Y['SPL'].min(), Y['SPL'].max()], [Y['SPL'].min(), Y['SPL'].max()], 'r--')
ax2.set_title('SPL Prediction')
ax2.set_xlabel('Actual (dB)')
ax2.set_ylabel('Predicted (dB)')

plt.tight_layout()
plt.savefig('Figs/RF_transfer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

transfer_path = 'C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/RandomForest/models/RF_BCM_transfer.pkl'
joblib.dump({'rf_base': male_rf, 'rf_new': female_rf, 'weights': (w_base, w_new)}, transfer_path)
