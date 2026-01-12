'''
excerpt: (lines 463-466)
For PR, closely related to linear models, two-stage transfer learning approaches have also been developed:
the first stage involves training on the base dataset, followed by transfer through methods such as
Lasso-type regularization (Li et al., 2022) or fine-tuning with gradient descent iterations to minimize
error on the subject-specific dataset
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error

# create directories for transfer learning outputs
os.makedirs("./transfer-figs", exist_ok=True)
os.makedirs("./transfer-models", exist_ok=True)

# load baseline male polynomial regression models
print("loading baseline male polynomial regression models...")
male_model = joblib.load('./models/firstPR')  # MultiOutputRegressor directly
male_x_scaler = joblib.load('./models/x_scaler_BCM.pkl')
male_f0_scaler = joblib.load('./models/f0_scaler_BCM.pkl')
male_spl_scaler = joblib.load('./models/spl_scaler_BCM.pkl')

# male model uses degree 12 polynomial (from MalePR.py)
from sklearn.preprocessing import PolynomialFeatures
male_poly = PolynomialFeatures(degree=12, include_bias=False)

# load pre-trained female polynomial regression models
print("loading pre-trained female polynomial regression models...")
female_f0_model = joblib.load('./FemaleModels/PR_f0_female.pkl')
female_spl_model = joblib.load('./FemaleModels/PR_spl_female.pkl')
female_x_scaler = joblib.load('./FemaleModels/x_scaler_female_PR.pkl')
female_f0_scaler = joblib.load('./FemaleModels/f0_scaler_female_PR.pkl')
female_spl_scaler = joblib.load('./FemaleModels/spl_scaler_female_PR.pkl')
female_poly_f0 = joblib.load('./FemaleModels/poly_f0_features_PR.pkl')
female_poly_spl = joblib.load('./FemaleModels/poly_spl_features_PR.pkl')

# load female dataset
df = pd.read_parquet("./FemaleNN_binary.parquet")
# only drop NaN in columns we actually use
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
df = df[df['ACFL'] > 30]

# define features and targets
X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# split data: use same split as FemalePR.py for consistency
x_train_full, x_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# scale test data for female model
x_test_scaled_female = female_x_scaler.transform(x_test)
x_test_poly_f0_female = female_poly_f0.transform(x_test_scaled_female)
x_test_poly_spl_female = female_poly_spl.transform(x_test_scaled_female)

# scale test data for male model
x_test_scaled_male = male_x_scaler.transform(x_test)
x_test_poly_male = male_poly.fit_transform(x_test_scaled_male)  # male uses single poly transformer

print("\nevaluating models on test set...")

# 1. male baseline model predictions (use male scalers throughout)
# male model outputs both F0 and SPL together (MultiOutputRegressor)
y_pred_male_scaled = male_model.predict(x_test_poly_male)

# inverse transform male predictions using male scalers
f0_pred_male = male_f0_scaler.inverse_transform(y_pred_male_scaled[:, 0].reshape(-1, 1))
spl_pred_male = male_spl_scaler.inverse_transform(y_pred_male_scaled[:, 1].reshape(-1, 1))
y_pred_male = np.hstack([f0_pred_male, spl_pred_male])

# 2. female model predictions (use female scalers throughout)
f0_pred_female_scaled = female_f0_model.predict(x_test_poly_f0_female)
spl_pred_female_scaled = female_spl_model.predict(x_test_poly_spl_female)

# inverse transform
f0_pred_female = female_f0_scaler.inverse_transform(f0_pred_female_scaled.reshape(-1, 1))
spl_pred_female = female_spl_scaler.inverse_transform(spl_pred_female_scaled.reshape(-1, 1))
y_pred_female = np.hstack([f0_pred_female, spl_pred_female])

# transfer learning: weighted ensemble strategy
# weight for baseline male model - very low since it performs poorly on female domain
w_male = 0.05
# weight for female model - very high since it's trained on target domain
w_female = 0.95

print(f"\nusing ensemble weights: male={w_male}, female={w_female}")

# combine predictions using weighted average
y_pred_ensemble = w_male * y_pred_male + w_female * y_pred_female

# calculate metrics for all three approaches
# male baseline
r2_f0_male = r2_score(y_test.iloc[:, 0], y_pred_male[:, 0])
r2_spl_male = r2_score(y_test.iloc[:, 1], y_pred_male[:, 1])
mae_f0_male = mean_absolute_error(y_test.iloc[:, 0], y_pred_male[:, 0])
mae_spl_male = mean_absolute_error(y_test.iloc[:, 1], y_pred_male[:, 1])

# female model
r2_f0_female = r2_score(y_test.iloc[:, 0], y_pred_female[:, 0])
r2_spl_female = r2_score(y_test.iloc[:, 1], y_pred_female[:, 1])
mae_f0_female = mean_absolute_error(y_test.iloc[:, 0], y_pred_female[:, 0])
mae_spl_female = mean_absolute_error(y_test.iloc[:, 1], y_pred_female[:, 1])

# ensemble
r2_f0_ensemble = r2_score(y_test.iloc[:, 0], y_pred_ensemble[:, 0])
r2_spl_ensemble = r2_score(y_test.iloc[:, 1], y_pred_ensemble[:, 1])
mae_f0_ensemble = mean_absolute_error(y_test.iloc[:, 0], y_pred_ensemble[:, 0])
mae_spl_ensemble = mean_absolute_error(y_test.iloc[:, 1], y_pred_ensemble[:, 1])

print("\n=== test set performance ===")
print(f"male baseline - f0: r²={r2_f0_male:.4f}, mae={mae_f0_male:.4f} hz | spl: r²={r2_spl_male:.4f}, mae={mae_spl_male:.4f} pa")
print(f"female model  - f0: r²={r2_f0_female:.4f}, mae={mae_f0_female:.4f} hz | spl: r²={r2_spl_female:.4f}, mae={mae_spl_female:.4f} pa")
print(f"ensemble      - f0: r²={r2_f0_ensemble:.4f}, mae={mae_f0_ensemble:.4f} hz | spl: r²={r2_spl_ensemble:.4f}, mae={mae_spl_ensemble:.4f} pa")
print(f"\naverage r² - male: {(r2_f0_male + r2_spl_male)/2:.4f}, female: {(r2_f0_female + r2_spl_female)/2:.4f}, ensemble: {(r2_f0_ensemble + r2_spl_ensemble)/2:.4f}")

# get y_test as array for plotting
y_test_array = y_test.values

# visualization: compare all three approaches
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('transfer learning: polynomial regression ensemble comparison', fontsize=16)

# f0 predictions
for idx, (y_pred, title, r2) in enumerate([
    (y_pred_male, 'male model (baseline)', r2_f0_male),
    (y_pred_female, 'female model (new)', r2_f0_female),
    (y_pred_ensemble, 'ensemble (transfer learning)', r2_f0_ensemble)
]):
    ax = axes[0, idx]
    ax.scatter(y_test_array[:, 0], y_pred[:, 0], alpha=0.5, s=10)
    ax.plot([y_test_array[:, 0].min(), y_test_array[:, 0].max()],
            [y_test_array[:, 0].min(), y_test_array[:, 0].max()], 'r--', linewidth=2)
    ax.set_title(f'{title}\nf0 prediction (r²={r2:.4f})')
    ax.set_xlabel('actual f0 (hz)')
    ax.set_ylabel('predicted f0 (hz)')
    ax.grid(True, alpha=0.3)

# spl predictions
for idx, (y_pred, title, r2) in enumerate([
    (y_pred_male, 'male model (baseline)', r2_spl_male),
    (y_pred_female, 'female model (new)', r2_spl_female),
    (y_pred_ensemble, 'ensemble (transfer learning)', r2_spl_ensemble)
]):
    ax = axes[1, idx]
    ax.scatter(y_test_array[:, 1], y_pred[:, 1], alpha=0.5, s=10)
    ax.plot([y_test_array[:, 1].min(), y_test_array[:, 1].max()],
            [y_test_array[:, 1].min(), y_test_array[:, 1].max()], 'r--', linewidth=2)
    ax.set_title(f'{title}\nspl prediction (r²={r2:.4f})')
    ax.set_xlabel('actual spl (pa)')
    ax.set_ylabel('predicted spl (pa)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./transfer-figs/PR_transfer_comparison.png', dpi=300, bbox_inches='tight')
print("\nvisualization saved to ./transfer-figs/PR_transfer_comparison.png")
plt.show()

# save transfer learning model
print("\nsaving transfer learning models...")
transfer_config = {
    'male_model': male_model,  # MultiOutputRegressor for both F0 and SPL
    'female_f0_model': female_f0_model,
    'female_spl_model': female_spl_model,
    'weights': (w_male, w_female),
    'male_x_scaler': male_x_scaler,
    'male_f0_scaler': male_f0_scaler,
    'male_spl_scaler': male_spl_scaler,
    'female_x_scaler': female_x_scaler,
    'female_f0_scaler': female_f0_scaler,
    'female_spl_scaler': female_spl_scaler,
    'female_poly_f0': female_poly_f0,
    'female_poly_spl': female_poly_spl,
    'male_poly': male_poly  # single poly transformer for male model
}
joblib.dump(transfer_config, './transfer-models/PR_transfer_ensemble.pkl')
print(f"transfer learning model saved to ./transfer-models/PR_transfer_ensemble.pkl")
