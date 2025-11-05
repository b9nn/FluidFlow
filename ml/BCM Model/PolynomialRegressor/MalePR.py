import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

os.makedirs("./figs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# full dataset
df = pd.read_csv('C:/Users/bglad/OneDrive/Desktop/Job/Fluid Flow/ml/BCM Model/MaleBCM.csv')

# downsample for faster experimentation ~ 90000 rows
#df = df.sample(frac=0.25, random_state=42) use full dataset for now
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

degree = 12

poly = PolynomialFeatures(degree=degree, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)  # Add this line
modelPR = MultiOutputRegressor(estimator=LinearRegression())
modelPR.fit(poly.fit_transform(x_train_scaled), y_train_scaled)

y_pred_scaled = modelPR.predict(x_test_poly)
y_pred = y_scaler.inverse_transform(y_pred_scaled)  # convert back to original scale

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction using PR Regressor')
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

# Residual Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.suptitle('Residual Analysis')

residuals_f0 = np.array(y_test)[:,0] - y_pred[:,0]
axes[0].scatter(y_pred[:,0], residuals_f0, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted F0 (Hz)')
axes[0].set_ylabel('Residuals (Hz)')
axes[0].set_title('F0 Residual Plot')
axes[0].grid(alpha=0.3)

residuals_spl = np.array(y_test)[:,1] - y_pred[:,1]
axes[1].scatter(y_pred[:,1], residuals_spl, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted SPL (Pa)')
axes[1].set_ylabel('Residuals (Pa)')
axes[1].set_title('SPL Residual Plot')
axes[1].grid(alpha=0.3)

plt.savefig("./figs/residual_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

joblib.dump(x_scaler, './models/x_scaler_BCM.pkl')
joblib.dump(y_scaler, './models/y_scaler_BCM.pkl')  
joblib.dump(modelPR, './models/firstPR')
