import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# paths resolved relative to this script, so the script runs from any cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(SCRIPT_DIR, "Female_binary.parquet"))

# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
df = df[df['ACFL'] > 30] # new df that only has ACFL values > 30

# define axis
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_val_scaled = x_scaler.transform(x_val)
x_test_scaled = x_scaler.transform(x_test)

# separate scalers for f0 and spl
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

rf = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
))

print("training...")
rf.fit(x_train_scaled, y_train_scaled)

y_pred_scaled = rf.predict(X=x_test_scaled)

# inverse transform using separate scalers
f0_pred = f0_scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1))
spl_pred = spl_scaler.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1))
y_pred = np.hstack([f0_pred, spl_pred])  # combine back to original scale

# iloc is a pd method, so y_pred is a nparray can just gran wihtout iloc
r2f0 = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2spl = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

print(f"F0:  R²={r2f0:.4f}")
print(f"SPL: R²={r2spl:.4f}")

TRANSFER_MODELS_DIR = os.path.join(SCRIPT_DIR, "transfer-models")
os.makedirs(TRANSFER_MODELS_DIR, exist_ok=True)
joblib.dump(rf, os.path.join(TRANSFER_MODELS_DIR, "transfer_female_rf.pkl"))
joblib.dump(x_scaler, os.path.join(TRANSFER_MODELS_DIR, "x_scaler.pkl"))
joblib.dump(f0_scaler, os.path.join(TRANSFER_MODELS_DIR, "f0_scaler.pkl"))
joblib.dump(spl_scaler, os.path.join(TRANSFER_MODELS_DIR, "spl_scaler.pkl"))

