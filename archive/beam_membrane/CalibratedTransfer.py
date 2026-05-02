"""
Calibrated Transfer: How far can we push the male BCM model on beam-membrane
data using ONLY mathematical corrections?

No new models trained from scratch on target data. Every method starts from
the male BCM RF predictions and applies a correction learned from a small
number of calibration samples.

The domain gap:
  - F0: r = -0.42 (negative correlation), +110 Hz mean overshoot
  - SPL: r = 0.30 (weak positive), -16 dB mean undershoot
  - Raw male model R^2 ~ -5

Methods (increasing complexity):
  0. Raw male model        — 0 parameters, 0 calibration samples needed
  1. Constant offset       — 2 params (mean residual per output)
  2. Linear calibration    — 4 params (slope + intercept per output)
  3. Poly residual (deg 2) — 20 params (input-dependent correction)
  4. Poly residual (deg 3) — 40 params (higher-order correction)
  5. Poly full (deg 2)     — 42 params (inputs + male preds as features)

Tested with [25%, 50%, 75%, 100%] of the training pool as calibration
samples from beam-membrane to learn each correction. Fixed 20% test set.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Calibration fractions of the training pool to test (25% and above)
CAL_FRACTIONS = [0.25, 0.50, 0.75, 1.0]
N_RUNS = 5
RANDOM_STATE = 42


# ==================== LOAD ====================
def load_data():
    rf_male = joblib.load(os.path.join(project_root, 'models', 'RF_BCM.pkl'))
    scaler_male = joblib.load(os.path.join(project_root, 'models', 'x_scaler_BCM.pkl'))

    df = pd.read_fwf(os.path.join(script_dir, 'Beam_Membrane_Data.csv'))
    df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
    df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
    z_f0 = np.abs(stats.zscore(df['F0']))
    z_spl = np.abs(stats.zscore(df['SPL']))
    df = df[(z_f0 < 3) & (z_spl < 3)]

    return rf_male, scaler_male, df


def male_predict(X, rf_male, scaler_male):
    return rf_male.predict(scaler_male.transform(X))


# ==================== CORRECTION METHODS ====================
# All methods take: calibration inputs, calibration actuals,
# male predictions on calibration set, male predictions on test set,
# and optionally raw test inputs.

def correct_none(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 0: No correction. Raw male model."""
    return mp_test


def correct_offset(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 1: y = male_pred + constant_offset (per output)."""
    offsets = np.mean(Y_cal - mp_cal, axis=0)
    return mp_test + offsets


def correct_linear(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 2: y = a * male_pred + b (per output)."""
    pred = np.zeros_like(mp_test)
    for i in range(2):
        reg = Ridge(alpha=0.1)
        reg.fit(mp_cal[:, i].reshape(-1, 1), Y_cal[:, i])
        pred[:, i] = reg.predict(mp_test[:, i].reshape(-1, 1))
    return pred


def correct_poly_residual_d2(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 3: y = male_pred + degree-2 polynomial(inputs)."""
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_cal_p = poly.fit_transform(X_cal)
    X_test_p = poly.transform(X_test)

    pred = np.zeros_like(mp_test)
    for i in range(2):
        residuals = Y_cal[:, i] - mp_cal[:, i]
        reg = Ridge(alpha=1.0)
        reg.fit(X_cal_p, residuals)
        pred[:, i] = mp_test[:, i] + reg.predict(X_test_p)
    return pred


def correct_poly_residual_d3(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 4: y = male_pred + degree-3 polynomial(inputs)."""
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_cal_p = poly.fit_transform(X_cal)
    X_test_p = poly.transform(X_test)

    pred = np.zeros_like(mp_test)
    for i in range(2):
        residuals = Y_cal[:, i] - mp_cal[:, i]
        reg = Ridge(alpha=10.0)  # stronger regularisation for more params
        reg.fit(X_cal_p, residuals)
        pred[:, i] = mp_test[:, i] + reg.predict(X_test_p)
    return pred


def correct_poly_full_d2(X_cal, Y_cal, mp_cal, mp_test, X_test):
    """Method 5: y = degree-2 polynomial(inputs, male_pred_F0, male_pred_SPL)."""
    X_cal_aug = np.column_stack([X_cal, mp_cal])
    X_test_aug = np.column_stack([X_test, mp_test])

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_cal_p = poly.fit_transform(X_cal_aug)
    X_test_p = poly.transform(X_test_aug)

    pred = np.zeros_like(mp_test)
    for i in range(2):
        reg = Ridge(alpha=1.0)
        reg.fit(X_cal_p, Y_cal[:, i])
        pred[:, i] = reg.predict(X_test_p)
    return pred


# ==================== MAIN ====================
def main():
    t0 = time.time()

    print("=" * 70)
    print("CALIBRATED TRANSFER: Correcting the Male BCM Model")
    print("How far can mathematical corrections push the -5 R^2 baseline?")
    print("=" * 70)

    rf_male, scaler_male, df = load_data()
    print(f"\nBeam-Membrane clean samples: {len(df)}")

    # Fixed test set (never used for calibration)
    X_all = df[['a_CT', 'a_TA', 'PS']].values
    Y_all = df[['F0', 'SPL']].values
    X_pool, X_test, Y_pool, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=RANDOM_STATE
    )

    mp_test = male_predict(X_test, rf_male, scaler_male)

    # Baseline: raw male model on test set
    raw_f0 = r2_score(Y_test[:, 0], mp_test[:, 0])
    raw_spl = r2_score(Y_test[:, 1], mp_test[:, 1])
    raw_avg = (raw_f0 + raw_spl) / 2
    print(f"\nRaw male model on test set ({len(X_test)} samples):")
    print(f"  F0  R^2 = {raw_f0:.3f}")
    print(f"  SPL R^2 = {raw_spl:.3f}")
    print(f"  Avg R^2 = {raw_avg:.3f}")

    methods = [
        ('None (raw)',       correct_none),
        ('Offset',           correct_offset),
        ('Linear',           correct_linear),
        ('Poly-2 residual',  correct_poly_residual_d2),
        ('Poly-3 residual',  correct_poly_residual_d3),
        ('Poly-2 full',      correct_poly_full_d2),
    ]

    # Results storage
    all_results = []

    for frac in CAL_FRACTIONS:
        n_cal = int(len(X_pool) * frac)
        print(f"\n{'=' * 50}")
        print(f"Calibration: {frac*100:.0f}% of pool ({n_cal} samples)")
        print(f"{'=' * 50}")

        run_results = {m_name: {'f0': [], 'spl': []} for m_name, _ in methods}

        for run in range(N_RUNS):
            rs = RANDOM_STATE + run

            # Sample calibration set from pool
            if n_cal >= len(X_pool):
                X_cal, Y_cal = X_pool, Y_pool
            else:
                idx = np.random.RandomState(rs).choice(
                    len(X_pool), size=n_cal, replace=False)
                X_cal, Y_cal = X_pool[idx], Y_pool[idx]

            mp_cal = male_predict(X_cal, rf_male, scaler_male)

            for m_name, m_func in methods:
                try:
                    pred = m_func(X_cal, Y_cal, mp_cal, mp_test, X_test)
                    r2_f0 = r2_score(Y_test[:, 0], pred[:, 0])
                    r2_spl = r2_score(Y_test[:, 1], pred[:, 1])
                except Exception:
                    r2_f0, r2_spl = float('nan'), float('nan')

                run_results[m_name]['f0'].append(r2_f0)
                run_results[m_name]['spl'].append(r2_spl)

        # Average across runs
        avg = {'n_cal': n_cal, 'frac': frac}
        for m_name, _ in methods:
            avg[f'{m_name}_f0'] = np.nanmean(run_results[m_name]['f0'])
            avg[f'{m_name}_spl'] = np.nanmean(run_results[m_name]['spl'])
        all_results.append(avg)

        # Print this fraction
        for m_name, _ in methods:
            f0 = avg[f'{m_name}_f0']
            spl = avg[f'{m_name}_spl']
            combined = (f0 + spl) / 2
            print(f"  {m_name:<18} F0={f0:>7.3f}  SPL={spl:>7.3f}  "
                  f"Avg={combined:>7.3f}")

    # ==================== SUMMARY ====================
    m_names = [m[0] for m in methods]

    print("\n\n" + "=" * 70)
    print("SUMMARY — Combined R^2 (F0 + SPL) / 2")
    print("=" * 70)

    header = f"{'Frac':>6} {'N_cal':>6}"
    for m in m_names:
        header += f"  {m:>16}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['frac']*100:>5.0f}% {r['n_cal']:>6}"
        for m in m_names:
            combined = (r[f'{m}_f0'] + r[f'{m}_spl']) / 2
            line += f"  {combined:>16.3f}"
        print(line)

    print(f"\n\nBaseline (0 calibration samples): R^2 = {raw_avg:.3f}")

    # F0 breakdown
    print(f"\n\nF0 R^2:")
    header = f"{'Frac':>6} {'N_cal':>6}"
    for m in m_names:
        header += f"  {m:>16}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        line = f"{r['frac']*100:>5.0f}% {r['n_cal']:>6}"
        for m in m_names:
            line += f"  {r[f'{m}_f0']:>16.3f}"
        print(line)

    # SPL breakdown
    print(f"\n\nSPL R^2:")
    header = f"{'Frac':>6} {'N_cal':>6}"
    for m in m_names:
        header += f"  {m:>16}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        line = f"{r['frac']*100:>5.0f}% {r['n_cal']:>6}"
        for m in m_names:
            line += f"  {r[f'{m}_spl']:>16.3f}"
        print(line)

    # ==================== PLOTS ====================
    print("\n\nGenerating plots...")
    os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

    fracs = [r['frac'] for r in all_results]
    frac_labels = [f"{f*100:.0f}%" for f in fracs]

    plot_styles = {
        'None (raw)':       ('gray',   '--', 'x'),
        'Offset':           ('brown',  '-',  'v'),
        'Linear':           ('orange', '-',  '^'),
        'Poly-2 residual':  ('green',  '-',  's'),
        'Poly-3 residual':  ('blue',   '-',  'D'),
        'Poly-2 full':      ('red',    '-',  '*'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Correcting the Male BCM Model for Beam-Membrane Prediction',
                 fontsize=14, fontweight='bold')

    for ax_idx, (title, target) in enumerate([
        ('Combined (F0+SPL)/2', None),
        ('F0', 'f0'),
        ('SPL', 'spl'),
    ]):
        ax = axes[ax_idx]

        # Raw baseline as horizontal line
        if target is None:
            baseline = raw_avg
        elif target == 'f0':
            baseline = raw_f0
        else:
            baseline = raw_spl
        ax.axhline(y=baseline, color='gray', linestyle=':', linewidth=1,
                   label=f'Raw male model ({baseline:.2f})')

        for m_name in m_names:
            if m_name == 'None (raw)':
                continue
            c, ls, mk = plot_styles[m_name]
            if target is None:
                vals = [(r[f'{m_name}_f0'] + r[f'{m_name}_spl']) / 2
                        for r in all_results]
            else:
                vals = [r[f'{m_name}_{target}'] for r in all_results]
            ax.plot(fracs, vals, marker=mk, label=m_name, color=c,
                    linestyle=ls, linewidth=2, markersize=8)

        ax.set_xlabel('Calibration Fraction of Training Pool')
        ax.set_ylabel('R^2')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fracs)
        ax.set_xticklabels(frac_labels)

    plt.tight_layout()
    path = os.path.join(script_dir, 'figs', 'calibrated_transfer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved to {path}")

    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f} seconds")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
