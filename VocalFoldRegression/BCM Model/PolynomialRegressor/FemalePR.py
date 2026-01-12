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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# create directories for saving outputs
os.makedirs("./FemaleFigs", exist_ok=True)
os.makedirs("./FemaleModels", exist_ok=True)

df = pd.read_csv('../FemaleBCM.csv')

# downsample for faster experimentation ~ 90000 rows
#df = df.sample(frac=0.25, random_state=42) use full dataset for now
df.to_parquet("./FemaleNN_binary.parquet", compression="snappy") # convert to binary
df = pd.read_parquet("./FemaleNN_binary.parquet")
# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
# NOTE: removed ACFL > 30 filter to retain more data for better SPL prediction
# the female dataset is already small, filtering reduces it further

X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# distribute the data
x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# fit and transform on the scalars
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_val_scaled = x_scaler.transform(x_val)
x_test_scaled = x_scaler.transform(x_test)

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

# fit-transform features to polynomials
# F0 needs higher degree for better fit
degree_f0 = 5
poly_f0 = PolynomialFeatures(degree=degree_f0, include_bias=False)
x_train_poly_f0 = poly_f0.fit_transform(x_train_scaled)
x_val_poly_f0 = poly_f0.transform(x_val_scaled)
x_test_poly_f0 = poly_f0.transform(x_test_scaled)

# SPL: use moderate degree with Ridge regularization
# the female SPL data has weak signal and small sample size
degree_spl = 5
poly_spl = PolynomialFeatures(degree=degree_spl, include_bias=False)
x_train_poly_spl = poly_spl.fit_transform(x_train_scaled)
x_val_poly_spl = poly_spl.transform(x_val_scaled)
x_test_poly_spl = poly_spl.transform(x_test_scaled)

print(f"F0 polynomial features: {x_train_poly_f0.shape[1]} (degree {degree_f0})")
print(f"SPL polynomial features: {x_train_poly_spl.shape[1]} (degree {degree_spl})")
print(f"Training samples: {x_train_poly_f0.shape[0]}")

# F0: Use LinearRegression
# SPL: Use Ridge with tuned regularization to prevent overfitting
f0_model = LinearRegression()
spl_model = Ridge(alpha=5.0)

# train separate models with their respective polynomial features
f0_model.fit(x_train_poly_f0, f0_train_scaled)
spl_model.fit(x_train_poly_spl, spl_train_scaled)

# predict separately using respective polynomial features
f0_pred_scaled = f0_model.predict(x_test_poly_f0)
spl_pred_scaled = spl_model.predict(x_test_poly_spl)

# inverse transform
f0_pred = f0_scaler.inverse_transform(f0_pred_scaled.reshape(-1, 1))
spl_pred = spl_scaler.inverse_transform(spl_pred_scaled.reshape(-1, 1))
y_pred = np.hstack([f0_pred, spl_pred])  # combine back to original scale

# print metrics (recale .iloc[rows, column])
r2_f0 = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_spl = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
mae_f0 = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
mae_spl = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
print(f"\nF0: R²={r2_f0:.4f}, MAE={mae_f0:.4f} Hz")
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
ax2.set_title(f'SPL (R² = {r2_spl:.4f})', fontweight='bold')
ax2.set_ylabel('Predicted (Pa)')
ax2.set_xlabel('Real (Pa)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./FemaleFigs/predictions.png", dpi=300, bbox_inches='tight') # save fig
plt.show()

# save both models separately
joblib.dump(f0_model, './FemaleModels/PR_f0_female.pkl')
joblib.dump(spl_model, './FemaleModels/PR_spl_female.pkl')
joblib.dump(x_scaler, './FemaleModels/x_scaler_female_PR.pkl')
joblib.dump(f0_scaler, './FemaleModels/f0_scaler_female_PR.pkl')
joblib.dump(spl_scaler, './FemaleModels/spl_scaler_female_PR.pkl')
joblib.dump(poly_f0, './FemaleModels/poly_f0_features_PR.pkl')
joblib.dump(poly_spl, './FemaleModels/poly_spl_features_PR.pkl')
