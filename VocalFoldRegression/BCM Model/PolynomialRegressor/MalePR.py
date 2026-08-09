import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# resolve ./ paths below against this script's folder, not the caller's cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.makedirs("./figs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

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

# split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 20% to testing, 80% to training

# create and fit scaler
print("scaling features...")
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

# separate scalers for f0 and spl
f0_scaler = StandardScaler()
spl_scaler = StandardScaler()

# fit on training data
f0_train_scaled = f0_scaler.fit_transform(y_train[['F0']])
spl_train_scaled = spl_scaler.fit_transform(y_train[['SPL']])

# combine back together for training
y_train_scaled = np.hstack([f0_train_scaled, spl_train_scaled])

# transform test data
f0_test_scaled = f0_scaler.transform(y_test[['F0']])
spl_test_scaled = spl_scaler.transform(y_test[['SPL']])
y_test_scaled = np.hstack([f0_test_scaled, spl_test_scaled])

degree = 12

poly = PolynomialFeatures(degree=degree, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)
modelPR = MultiOutputRegressor(estimator=LinearRegression())
modelPR.fit(x_train_poly, y_train_scaled)

y_pred_scaled = modelPR.predict(x_test_poly)
# inverse transform using separate scalers
f0_pred = f0_scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1))
spl_pred = spl_scaler.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1))
y_pred = np.hstack([f0_pred, spl_pred])  # combine back to original scale

# Calculate R² scores
r2_f0 = r2_score(np.array(y_test)[:, 0], y_pred[:, 0])
r2_spl = r2_score(np.array(y_test)[:, 1], y_pred[:, 1])
mae_f0 = mean_absolute_error(np.array(y_test)[:, 0], y_pred[:, 0])
mae_spl = mean_absolute_error(np.array(y_test)[:, 1], y_pred[:, 1])

print(f"\n=== Model Performance Metrics ===")
print(f"F0:  R²={r2_f0:.4f}, MAE={mae_f0:.4f} Hz")
print(f"SPL: R²={r2_spl:.4f}, MAE={mae_spl:.4f} Pa")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction using PR Regressor', fontsize=14, fontweight='bold')

# F0 plot
ax1.plot(np.array(y_test)[:,0], y_pred[:,0], 'o', alpha=0.5, markersize=4, color='steelblue')
lin = [min(np.array(y_test)[:,0]), max(np.array(y_test)[:,0])]
ax1.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax1.set_title(f'$f_o$ (R² = {r2_f0:.4f})', fontweight='bold')
ax1.set_ylabel('Predicted (Hz)')
ax1.set_xlabel('Real (Hz)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# SPL plot
ax2.plot(np.array(y_test)[:,1], y_pred[:,1], 'o', alpha=0.5, markersize=4, color='darkorange')
lin = [min(np.array(y_test)[:,1]), max(np.array(y_test)[:,1])]
ax2.plot(lin, lin, 'r-', linewidth=2, label='Perfect Prediction')
ax2.set_ylabel('Predicted (Pa)')
ax2.set_xlabel('Real (Pa)')
ax2.set_title(f'SPL (R² = {r2_spl:.4f})', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
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
joblib.dump(f0_scaler, './models/f0_scaler_BCM.pkl')
joblib.dump(spl_scaler, './models/spl_scaler_BCM.pkl')  
joblib.dump(modelPR, './models/firstPR')
