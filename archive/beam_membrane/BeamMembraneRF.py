"""
Random Forest trained on Beam-Membrane model data at differing data fractions.
Tests how RF performance scales with training set size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== LOAD DATA ====================
print("Loading Beam-Membrane data...")
# Fixed-width columns: header has spaces inside names, so read raw and assign
df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
# read_fwf parses the 8 fixed-width fields correctly
df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
print(f"Samples after dropping NaN: {len(df)}")

# ==================== DATA CLEANING ====================
# Physical bounds — remove non-physical simulation results
n_before = len(df)
df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]   # Hz
df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]  # dB
print(f"Removed {n_before - len(df)} physically unreasonable rows")

# Z-score filter (|z| > 3) — remove extreme statistical outliers
from scipy import stats
z_f0 = np.abs(stats.zscore(df['F0']))
z_spl = np.abs(stats.zscore(df['SPL']))
n_before = len(df)
df = df[(z_f0 < 3) & (z_spl < 3)]
print(f"Removed {n_before - len(df)} z-score outliers (|z| > 3)")

print(f"Clean samples: {len(df)}")
print(df.describe())

X = df[['a_CT', 'a_TA', 'PS']]
Y = df[['F0', 'SPL']]

# hold out a fixed 20% test set across all fractions
X_pool, X_test, Y_pool, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Test set: {len(X_test)} samples (fixed across all fractions)")
print(f"Training pool: {len(X_pool)} samples")

# ==================== EXPERIMENT ====================
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
results = []

for frac in FRACTIONS:
    if frac < 1.0:
        X_train = X_pool.sample(frac=frac, random_state=42)
        Y_train = Y_pool.loc[X_train.index]
    else:
        X_train = X_pool
        Y_train = Y_pool

    n_train = len(X_train)
    print(f"\n{'='*50}")
    print(f"Training with {frac*100:.0f}% of data ({n_train} samples)")
    print(f"{'='*50}")

    # scale
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # train
    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    ))
    rf.fit(X_train_scaled, Y_train)

    # predict
    y_pred = rf.predict(X_test_scaled)

    r2_f0 = r2_score(Y_test.iloc[:, 0], y_pred[:, 0])
    r2_spl = r2_score(Y_test.iloc[:, 1], y_pred[:, 1])
    mae_f0 = mean_absolute_error(Y_test.iloc[:, 0], y_pred[:, 0])
    mae_spl = mean_absolute_error(Y_test.iloc[:, 1], y_pred[:, 1])

    print(f"  F0:  R2={r2_f0:.4f}  MAE={mae_f0:.2f} Hz")
    print(f"  SPL: R2={r2_spl:.4f}  MAE={mae_spl:.2f} dB")

    results.append({
        'frac': frac,
        'n_train': n_train,
        'r2_f0': r2_f0,
        'r2_spl': r2_spl,
        'mae_f0': mae_f0,
        'mae_spl': mae_spl,
    })

    # save the full-data model
    if frac == 1.0:
        os.makedirs(os.path.join(script_dir, 'models'), exist_ok=True)
        joblib.dump(rf, os.path.join(script_dir, 'models', 'RF_BeamMembrane.pkl'))
        joblib.dump(x_scaler, os.path.join(script_dir, 'models', 'x_scaler_BeamMembrane.pkl'))
        print("  Saved full model to models/RF_BeamMembrane.pkl")

# ==================== SUMMARY ====================
print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format='%.4f'))

# ==================== PLOTS ====================
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Beam-Membrane RF: Performance vs Training Data Size', fontsize=14, fontweight='bold')

n_trains = [r['n_train'] for r in results]

# R2
ax = axes[0]
ax.plot(n_trains, [r['r2_f0'] for r in results], 'o-', label='F0', linewidth=2)
ax.plot(n_trains, [r['r2_spl'] for r in results], 's-', label='SPL', linewidth=2)
ax.set_xlabel('Number of Training Samples')
ax.set_ylabel('R2')
ax.set_title('R2 vs Training Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# MAE
ax = axes[1]
ax.plot(n_trains, [r['mae_f0'] for r in results], 'o-', label='F0 (Hz)', linewidth=2)
ax.plot(n_trains, [r['mae_spl'] for r in results], 's-', label='SPL (dB)', linewidth=2)
ax.set_xlabel('Number of Training Samples')
ax.set_ylabel('MAE')
ax.set_title('MAE vs Training Size')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figs', 'data_fraction_experiment.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved plot to figs/data_fraction_experiment.png")

# pred vs actual for full model
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Beam-Membrane RF (full data): Predicted vs Actual')

# reload full-model predictions
rf_full = joblib.load(os.path.join(script_dir, 'models', 'RF_BeamMembrane.pkl'))
scaler_full = joblib.load(os.path.join(script_dir, 'models', 'x_scaler_BeamMembrane.pkl'))
y_pred_full = rf_full.predict(scaler_full.transform(X_test))

ax1.scatter(Y_test.iloc[:, 0], y_pred_full[:, 0], alpha=0.5, s=10)
lim = [Y_test.iloc[:, 0].min(), Y_test.iloc[:, 0].max()]
ax1.plot(lim, lim, 'r--')
ax1.set_xlabel('Actual (Hz)')
ax1.set_ylabel('Predicted (Hz)')
ax1.set_title('F0')

ax2.scatter(Y_test.iloc[:, 1], y_pred_full[:, 1], alpha=0.5, s=10)
lim = [Y_test.iloc[:, 1].min(), Y_test.iloc[:, 1].max()]
ax2.plot(lim, lim, 'r--')
ax2.set_xlabel('Actual (dB)')
ax2.set_ylabel('Predicted (dB)')
ax2.set_title('SPL')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figs', 'pred_vs_actual.png'), dpi=150, bbox_inches='tight')
print(f"Saved plot to figs/pred_vs_actual.png")

plt.show()
