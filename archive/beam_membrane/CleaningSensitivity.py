"""
Outlier Cleaning Sensitivity Analysis for Beam-Membrane Data.

Tests 5 cleaning levels (from dropna-only to full z-score pipeline) and
measures the impact on:
  1. Direct RF performance (R² for F0 and SPL)
  2. Domain gap (Pearson r between male BCM predictions and BM actuals)
  3. Male model R² (raw male BCM on BM test set)
  4. Sample count
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# ==================== LOAD RAW DATA ====================
print("Loading Beam-Membrane data...")
df_raw = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
df_raw.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
df_raw = df_raw[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
print(f"Samples after dropna: {len(df_raw)}")

# ==================== LOAD MALE MODEL ====================
print("Loading male BCM model...")
rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
scaler_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))


# ==================== CLEANING LEVELS ====================
def apply_cleaning(df, level):
    """Apply cleaning at the specified level. Returns cleaned DataFrame."""
    d = df.copy()

    if level == 0:
        # dropna only (already done)
        pass

    elif level == 1:
        # Loose physical bounds
        d = d[(d['F0'] >= 20) & (d['F0'] <= 600)]
        d = d[(d['SPL'] >= 0) & (d['SPL'] <= 150)]

    elif level == 2:
        # Tight physical bounds (current)
        d = d[(d['F0'] >= 50) & (d['F0'] <= 500)]
        d = d[(d['SPL'] >= 40) & (d['SPL'] <= 140)]

    elif level == 3:
        # Tight bounds + relaxed z-score (|z| < 4)
        d = d[(d['F0'] >= 50) & (d['F0'] <= 500)]
        d = d[(d['SPL'] >= 40) & (d['SPL'] <= 140)]
        z_f0 = np.abs(stats.zscore(d['F0']))
        z_spl = np.abs(stats.zscore(d['SPL']))
        d = d[(z_f0 < 4) & (z_spl < 4)]

    elif level == 4:
        # Full current pipeline: tight bounds + z < 3
        d = d[(d['F0'] >= 50) & (d['F0'] <= 500)]
        d = d[(d['SPL'] >= 40) & (d['SPL'] <= 140)]
        z_f0 = np.abs(stats.zscore(d['F0']))
        z_spl = np.abs(stats.zscore(d['SPL']))
        d = d[(z_f0 < 3) & (z_spl < 3)]

    return d


LEVELS = {
    0: 'dropna only',
    1: 'Loose bounds (F0:20-600, SPL:0-150)',
    2: 'Tight bounds (F0:50-500, SPL:40-140)',
    3: 'Tight bounds + z<4',
    4: 'Tight bounds + z<3 (current)',
}

# ==================== RUN EXPERIMENT ====================
results = []

for level, label in LEVELS.items():
    print(f"\n{'='*60}")
    print(f"Level {level}: {label}")
    print(f"{'='*60}")

    df = apply_cleaning(df_raw, level)
    n_samples = len(df)
    print(f"  Samples: {n_samples}")

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    # --- Train/test split ---
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --- 1. Direct RF performance ---
    x_scaler = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)
    X_test_sc = x_scaler.transform(X_test)

    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42,
    ))
    rf.fit(X_train_sc, Y_train)
    y_pred = rf.predict(X_test_sc)

    r2_f0 = r2_score(Y_test.iloc[:, 0], y_pred[:, 0])
    r2_spl = r2_score(Y_test.iloc[:, 1], y_pred[:, 1])
    print(f"  Direct RF — F0 R²={r2_f0:.4f}, SPL R²={r2_spl:.4f}")

    # --- 2 & 3. Male model on BM test set ---
    X_test_male_sc = scaler_male.transform(X_test)
    male_pred = rf_male.predict(X_test_male_sc)

    # Pearson r (domain gap correlation)
    corr_f0 = np.corrcoef(male_pred[:, 0], Y_test.iloc[:, 0].values)[0, 1]
    corr_spl = np.corrcoef(male_pred[:, 1], Y_test.iloc[:, 1].values)[0, 1]
    print(f"  Domain gap — F0 r={corr_f0:.4f}, SPL r={corr_spl:.4f}")

    # Male model R² on BM test set
    male_r2_f0 = r2_score(Y_test.iloc[:, 0], male_pred[:, 0])
    male_r2_spl = r2_score(Y_test.iloc[:, 1], male_pred[:, 1])
    print(f"  Male R² on BM — F0={male_r2_f0:.4f}, SPL={male_r2_spl:.4f}")

    results.append({
        'level': level,
        'label': label,
        'n_samples': n_samples,
        'rf_r2_f0': r2_f0,
        'rf_r2_spl': r2_spl,
        'corr_f0': corr_f0,
        'corr_spl': corr_spl,
        'male_r2_f0': male_r2_f0,
        'male_r2_spl': male_r2_spl,
    })


# ==================== SUMMARY TABLE ====================
print("\n\n" + "=" * 90)
print("CLEANING SENSITIVITY SUMMARY")
print("=" * 90)

header = (f"{'Lvl':<4} {'Cleaning':<38} {'N':>5}  "
          f"{'RF R² F0':>9} {'RF R² SPL':>10}  "
          f"{'r F0':>6} {'r SPL':>7}  "
          f"{'Male R² F0':>11} {'Male R² SPL':>12}")
print(header)
print("-" * len(header))

for r in results:
    print(f"{r['level']:<4} {r['label']:<38} {r['n_samples']:>5}  "
          f"{r['rf_r2_f0']:>9.4f} {r['rf_r2_spl']:>10.4f}  "
          f"{r['corr_f0']:>6.3f} {r['corr_spl']:>7.3f}  "
          f"{r['male_r2_f0']:>11.4f} {r['male_r2_spl']:>12.4f}")


# ==================== PLOT ====================
print("\nGenerating plot...")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Outlier Cleaning Sensitivity Analysis',
             fontsize=14, fontweight='bold')

levels = [r['level'] for r in results]
x_labels = [f"L{r['level']}\n({r['n_samples']})" for r in results]
x_pos = np.arange(len(levels))

# --- Panel 1: Direct RF R² ---
ax = axes[0, 0]
ax.bar(x_pos - 0.15, [r['rf_r2_f0'] for r in results], 0.3,
       label='F0', color='steelblue')
ax.bar(x_pos + 0.15, [r['rf_r2_spl'] for r in results], 0.3,
       label='SPL', color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel('R²')
ax.set_title('Direct RF Performance (BM train → BM test)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# --- Panel 2: Domain gap correlation ---
ax = axes[0, 1]
ax.bar(x_pos - 0.15, [r['corr_f0'] for r in results], 0.3,
       label='F0', color='steelblue')
ax.bar(x_pos + 0.15, [r['corr_spl'] for r in results], 0.3,
       label='SPL', color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel('Pearson r')
ax.set_title('Domain Gap (Male BCM pred vs BM actual)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 3: Male model R² ---
ax = axes[1, 0]
ax.bar(x_pos - 0.15, [r['male_r2_f0'] for r in results], 0.3,
       label='F0', color='steelblue')
ax.bar(x_pos + 0.15, [r['male_r2_spl'] for r in results], 0.3,
       label='SPL', color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel('R²')
ax.set_title('Male BCM R² on BM Test Set')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 4: Sample count ---
ax = axes[1, 1]
counts = [r['n_samples'] for r in results]
bars = ax.bar(x_pos, counts, 0.5, color='gray', edgecolor='black')
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            str(c), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel('Samples')
ax.set_title('Dataset Size at Each Cleaning Level')
ax.grid(True, alpha=0.3, axis='y')

# Add a legend below with cleaning descriptions
legend_text = "\n".join(f"L{lvl}: {lbl}" for lvl, lbl in LEVELS.items())
fig.text(0.5, -0.02, legend_text, ha='center', fontsize=8,
         family='monospace', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
path = os.path.join(script_dir, 'figs', 'cleaning_sensitivity.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {path}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
