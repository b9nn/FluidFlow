import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# paths resolved relative to this script, so the script runs from any cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BCM_DIR = os.path.dirname(SCRIPT_DIR)  # VocalFoldRegression/BCM Model
FIGS_DIR = os.path.join(SCRIPT_DIR, 'figs')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# full dataset
df = pd.read_csv(os.path.join(BCM_DIR, 'MaleBCM.csv'))

# downsample for faster experimentation ~ 25000 rows
df = df.sample(frac=1, random_state=42)
cache_path = os.path.join(SCRIPT_DIR, 'data_binary.parquet')
df.to_parquet(cache_path, compression="snappy") # convert to binary
df = pd.read_parquet(cache_path)

print(df.head())
print(df.describe())

# cleaning
df = df.dropna() # handle NaN

# another option
'''
for column in df.columns:
    df[column].fillna(df[column].mode()[0])
'''

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

# grid search - for hyperparameters, to be adjusted 
parameter_grid = {'estimator__n_estimators':[300], 'estimator__max_depth': [None], 'estimator__min_samples_split':[2]} # for best preformence
# parameter_grid = {'estimator__n_estimators':[50, 100], 'estimator__max_depth': [None, 10], 'estimator__min_samples_split':[2, 10]} # for default

# model
rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))

# 3cv for now, trying to keep first model light, up to 5 eventually
gs = GridSearchCV(rf, param_grid=parameter_grid, cv=3, scoring='r2', verbose=2)
gs.fit(x_train_scaled, y_train)
best_rf = gs.best_estimator_

# analysis
print("Best parameters:", gs.best_params_)
print("Best CV R^2:", gs.best_score_) # cross validated on 3 folds (cv)
print("Best model:", best_rf)

# predict
y_pred = best_rf.predict(x_test_scaled)

# post-mortem
print("="*50)
print("Overall R^2:", r2_score(y_test, y_pred, multioutput="uniform_average"))
print("Overall MSE:", mean_squared_error(y_test, y_pred, multioutput="uniform_average"))

# save the trained model and scaler
joblib.dump(best_rf, os.path.join(MODELS_DIR, 'RF_BCM.pkl'))
joblib.dump(x_scaler, os.path.join(MODELS_DIR, 'x_scaler_BCM.pkl'))
print("saved!")

# predicted vs actual
for i, target in enumerate(['F0', 'SPL']):
    plt.figure(figsize=(5,5))
    plt.scatter(y_test[target], y_pred[:, i], alpha=0.5)
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()],
             'r--')  # ideal line
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual: {target}")
    plt.savefig(os.path.join(FIGS_DIR, f'pred_vs_actual_{target}.png'), dpi=150, bbox_inches='tight')
    plt.show()


'''
~~~ residuals ~~~

the differences between actual and predicted (how far off the model was from actual)

element wise subtraction -->

actual - predicted:
< 0 == predicted too high
= 0 == perfect
> 0 == predicted too low
'''
residuals = y_test.values - y_pred
print("\nSome residuals:", residuals[:10])

for i, target in enumerate(['F0','SPL']):
    plt.figure(figsize=(5,5))
    plt.scatter(y_pred[:, i], residuals[:, i], alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Plot: {target}")
    plt.savefig(os.path.join(FIGS_DIR, f'residuals_{target}.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ~~~ Predicted vs Actual ~~~

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction using RF Regressor')
ax1.plot(np.array(y_test)[:,0], y_pred[:,0], 'o')
lin = [min(np.array(y_test)[:,0]), max(np.array(y_test)[:,0])]
ax1.plot(lin,lin)
ax1.set_title('$f_0$')
ax1.set_ylabel('Predicted (Hz)')
ax1.set_xlabel('Real (Hz)')
ax2.plot(np.array(y_test)[:,1], y_pred[:,1], 'o')
lin = [min(np.array(y_test)[:,1]), max(np.array(y_test)[:,1])]
ax2.plot(lin,lin)
ax2.set_ylabel('Predicted (dB)')
ax2.set_xlabel('Real (dB)')
ax2.set_title('SPL')
plt.savefig(os.path.join(FIGS_DIR, 'RF_prediction_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

