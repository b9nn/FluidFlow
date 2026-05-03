"""
Phase 1: Gaussian Process with Physics-Informed Features and Male-Model Prior
Target: Beam-Membrane model data (source: Male BCM)

Tests whether GP regression with physics-derived features and Bayesian transfer
(using the male BCM model as prior mean) can produce better predictions at
small data sizes compared to Random Forest.

Methods compared:
  1. RF Target-Only          — RF trained only on target data (existing baseline)
  2. RF + Physics Features   — RF with physics-derived extra features
  3. GP Raw                  — GP on (a_CT, a_TA, PS) only
  4. GP + Physics            — GP on raw + physics-derived features
  5. GP + Physics + Prior    — GP learns residual from male model (Bayesian transfer)

Physics features (from MATLAB equations, zero simulation cost):
  eps0       = 0.2*(3*a_CT - a_TA)                — vocal fold strain
  sigma_TA   = sign(e)*1000*(exp(8|e|)-1) + a_TA*105kPa — muscle stress
  sigma_LIG  = sign(e)*2000*(exp(10|e|)-1)        — ligament stress
  F_beam     = A_LIG*sigma_LIG + A_TA*sigma_TA    — axial tension
  freq_scale = (1/2L)*sqrt(F_beam/mu)             — analytical F0 estimate
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
FRACTIONS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
N_RUNS = 3
RANDOM_STATE = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))


# ==================== GEOMETRY CONSTANTS ====================
# From MATLAB: Stiffness_Damping_Membrane_Beam.m, BodyCoverModel.m

MALE_GEOMETRY = {
    'L0': 1.5e-2,        # vocal fold length (m)
    'b': 5.0e-3,         # vocal fold depth (m)
    'A_MUC': 5.0e-6,     # mucosa cross-section (m^2)
    'A_LIG': 6.1e-6,     # ligament cross-section (m^2)
    'A_TA': 40.9e-6,     # thyroarytenoid cross-section (m^2)
    'rho_MUC': 1000,     # mucosa density (kg/m^3)
    'rho_LIG': 1030,     # ligament density (kg/m^3)
    'rho_TA': 1050,      # thyroarytenoid density (kg/m^3)
}

FEMALE_GEOMETRY = {
    'L0': 1.0e-2,
    'b': 2.0e-3,
    'A_MUC': 3.0e-6,
    'A_LIG': 3.0e-6,
    'A_TA': 6.0e-6,
    'rho_MUC': 1000,
    'rho_LIG': 1030,
    'rho_TA': 1050,
}

# Tissue constitutive parameters (from MUC.m, LIG.m, TA.m)
TISSUE = {
    'MUC': {'A': 1500, 'B': 7},
    'LIG': {'A': 2000, 'B': 10},
    'TA':  {'A': 1000, 'B': 8, 'sigma_max': 105000},
}


# ==================== PHYSICS FEATURES ====================
def compute_physics_features(a_CT, a_TA, PS, geometry):
    """
    Compute physics-derived features from the MATLAB vocal fold equations.
    Pure algebra — microseconds to compute, zero simulation cost.

    Parameters
    ----------
    a_CT, a_TA : array-like, muscle activations (0-1)
    PS         : array-like, subglottal pressure (Pa)
    geometry   : dict with vocal fold geometry constants

    Returns
    -------
    np.ndarray of shape (n_samples, 6) with columns:
        eps0, sigma_TA, sigma_LIG, F_beam, freq_scale, PS_kPa
    """
    a_CT = np.asarray(a_CT, dtype=float)
    a_TA = np.asarray(a_TA, dtype=float)
    PS = np.asarray(PS, dtype=float)

    # Linearised vocal fold strain (CalcBodyCoverParameters.m)
    eps0 = 0.2 * (3.0 * a_CT - a_TA)

    # Tissue stresses — exponential constitutive laws
    abs_eps = np.abs(eps0)
    sign_eps = np.sign(eps0)

    sigma_LIG = sign_eps * TISSUE['LIG']['A'] * (
        np.exp(TISSUE['LIG']['B'] * abs_eps) - 1
    )

    sigma_TA = (
        sign_eps * TISSUE['TA']['A'] * (
            np.exp(TISSUE['TA']['B'] * abs_eps) - 1
        ) + a_TA * TISSUE['TA']['sigma_max']
    )

    # Axial beam forces (Membrane_Beam_Parameters.m)
    F_beam = geometry['A_LIG'] * sigma_LIG + geometry['A_TA'] * sigma_TA

    # Linear mass density (kg/m)
    linear_density = (
        geometry['rho_MUC'] * geometry['A_MUC'] +
        geometry['rho_LIG'] * geometry['A_LIG'] +
        geometry['rho_TA']  * geometry['A_TA']
    )

    # Analytical frequency estimate: f = (1/2L) * sqrt(F_beam / mu)
    F_beam_safe = np.maximum(F_beam, 1e-6)
    freq_scale = (1.0 / (2.0 * geometry['L0'])) * np.sqrt(F_beam_safe / linear_density)

    # Stack and scale to ~O(1) for numerical stability
    features = np.column_stack([
        eps0,
        sigma_TA / 1e5,
        sigma_LIG / 1e4,
        F_beam * 1e3,
        freq_scale / 100.0,
        PS / 1000.0,
    ])

    return features


# ==================== MODEL HELPERS ====================
def get_rf_params(n_samples):
    """Adaptive RF complexity based on training set size."""
    if n_samples < 30:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(3, n_samples // 5)}
    elif n_samples < 100:
        return {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 3}
    elif n_samples < 300:
        return {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 2}
    else:
        return {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1}


def train_rf(X_train, Y_train, X_test, random_state=42):
    """Train a multi-output RF and return predictions on test set."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    params = get_rf_params(len(X_train))
    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **params)
    )
    rf.fit(X_tr, Y_train)
    return rf.predict(X_te)


def build_gp_kernel(n_features):
    """Matern 2.5 kernel with ARD (separate length scale per feature)."""
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            nu=2.5,
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-2, 1e3),
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )


MAX_GP_SAMPLES = 800  # GP is O(n^3); cap training size for speed


def train_gp(X_train, y_train, X_test):
    """
    Train a single-output GP. Returns predictions on test set.
    Subsamples training data if > MAX_GP_SAMPLES for speed.
    Falls back to mean prediction on numerical failure.
    """
    # Subsample if training set is too large for GP
    if len(X_train) > MAX_GP_SAMPLES:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), size=MAX_GP_SAMPLES, replace=False
        )
        X_train = X_train[idx]
        y_train = y_train[idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    kernel = build_gp_kernel(X_tr.shape[1])

    try:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=1e-6,
        )
        gp.fit(X_tr, y_train)
        pred = gp.predict(X_te)
        return pred
    except Exception as e:
        print(f"    [GP numerical failure: {e} — using mean fallback]")
        return np.full(len(X_te), y_train.mean())


# ==================== SINGLE EXPERIMENT ====================
def run_experiment(df_target, rf_male, scaler_male, frac, geometry, random_state=42):
    """
    Run all 5 methods for one data fraction and one random seed.
    Returns dict of R^2 scores for F0 and SPL, or None if too few samples.
    """
    # Sample target data
    if frac < 1.0:
        df = df_target.sample(frac=frac, random_state=random_state)
    else:
        df = df_target.copy()

    X_raw = df[['a_CT', 'a_TA', 'PS']].values
    Y = df[['F0', 'SPL']].values
    n_total = len(df)

    if n_total < 10:
        return None

    # 80/20 train/test split
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=0.2, random_state=random_state
    )
    n_train = len(X_train_raw)

    # Physics features for target geometry
    phys_train = compute_physics_features(
        X_train_raw[:, 0], X_train_raw[:, 1], X_train_raw[:, 2], geometry
    )
    phys_test = compute_physics_features(
        X_test_raw[:, 0], X_test_raw[:, 1], X_test_raw[:, 2], geometry
    )

    # Augmented features: raw inputs + physics
    X_train_aug = np.column_stack([X_train_raw, phys_train])
    X_test_aug = np.column_stack([X_test_raw, phys_test])

    # Male model predictions (for Bayesian prior)
    male_pred_train = rf_male.predict(scaler_male.transform(X_train_raw))
    male_pred_test = rf_male.predict(scaler_male.transform(X_test_raw))

    results = {'frac': frac, 'n_train': n_train}

    # --- Method 1: RF Target-Only (baseline) ---
    pred_rf = train_rf(X_train_raw, Y_train, X_test_raw, random_state)
    results['rf_f0'] = r2_score(Y_test[:, 0], pred_rf[:, 0])
    results['rf_spl'] = r2_score(Y_test[:, 1], pred_rf[:, 1])

    # --- Method 2: RF + Physics Features ---
    pred_rf_phys = train_rf(X_train_aug, Y_train, X_test_aug, random_state)
    results['rf_phys_f0'] = r2_score(Y_test[:, 0], pred_rf_phys[:, 0])
    results['rf_phys_spl'] = r2_score(Y_test[:, 1], pred_rf_phys[:, 1])

    # --- Method 3: GP on raw features ---
    for i, name in enumerate(['f0', 'spl']):
        pred = train_gp(X_train_raw, Y_train[:, i], X_test_raw)
        results[f'gp_raw_{name}'] = r2_score(Y_test[:, i], pred)

    # --- Method 4: GP + Physics features ---
    for i, name in enumerate(['f0', 'spl']):
        pred = train_gp(X_train_aug, Y_train[:, i], X_test_aug)
        results[f'gp_phys_{name}'] = r2_score(Y_test[:, i], pred)

    # --- Method 5: GP + Physics + Male Prior (Bayesian transfer) ---
    # Train GP on residuals: y_female - y_male_pred
    # At test time: y_pred = y_male_pred + gp_residual.predict(X_test)
    for i, name in enumerate(['f0', 'spl']):
        residuals = Y_train[:, i] - male_pred_train[:, i]
        pred_residual = train_gp(X_train_aug, residuals, X_test_aug)
        pred_total = male_pred_test[:, i] + pred_residual
        results[f'gp_prior_{name}'] = r2_score(Y_test[:, i], pred_total)

    return results


# ==================== MAIN ====================
def main():
    t0 = time.time()

    print("=" * 70)
    print("PHASE 1: GP WITH PHYSICS FEATURES & MALE-MODEL PRIOR")
    print("Target: Beam-Membrane model  |  Source: Male BCM")
    print("=" * 70)

    # --- Load male source model ---
    print("\nLoading pre-trained male BCM model (source)...")
    rf_male = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'RF_BCM.pkl'))
    scaler_male = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'x_scaler_BCM.pkl'))
    print("Done.")

    # --- Load Beam-Membrane target data ---
    print("\nLoading Beam-Membrane data (target)...")
    bm_path = os.path.join(PROJECT_ROOT, 'Beam_Membrane', 'Beam_Membrane_Data.csv')
    df = pd.read_fwf(bm_path)
    df.columns = ['a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'PS', 'F0', 'SPL']
    df = df[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    n_before = len(df)
    print(f"Total samples after dropping NaN: {n_before}")

    # Data cleaning — same pipeline as BeamMembraneRF.py
    df = df[(df['F0'] >= 50) & (df['F0'] <= 500)]
    df = df[(df['SPL'] >= 40) & (df['SPL'] <= 140)]
    z_f0 = np.abs(stats.zscore(df['F0']))
    z_spl = np.abs(stats.zscore(df['SPL']))
    df = df[(z_f0 < 3) & (z_spl < 3)]
    print(f"Clean samples: {len(df)} (removed {n_before - len(df)})")

    # Beam-Membrane uses male geometry (it's a more sophisticated male model)
    target_geometry = MALE_GEOMETRY

    # --- Sanity check: physics features ---
    print("\nPhysics feature sanity check (first 3 rows, male geometry):")
    sample = df.head(3)
    phys = compute_physics_features(
        sample['a_CT'].values, sample['a_TA'].values,
        sample['PS'].values, target_geometry
    )
    col_names = ['eps0', 'sigma_TA(1e5)', 'sigma_LIG(1e4)',
                 'F_beam(1e-3)', 'freq_scale/100', 'PS_kPa']
    phys_df = pd.DataFrame(phys, columns=col_names)
    print(phys_df.to_string(index=False, float_format='%.4f'))

    # --- Sanity check: male model on beam-membrane data ---
    X_sample = sample[['a_CT', 'a_TA', 'PS']].values
    male_pred = rf_male.predict(scaler_male.transform(X_sample))
    print(f"\nMale BCM predictions vs Beam-Membrane actual (first 3 rows):")
    print(f"  F0 actual:    {sample['F0'].values.round(1)}")
    print(f"  F0 male pred: {male_pred[:, 0].round(1)}")
    print(f"  SPL actual:   {sample['SPL'].values.round(1)}")
    print(f"  SPL male pred:{male_pred[:, 1].round(1)}")

    # --- Run experiments ---
    all_results = []
    methods = ['rf', 'rf_phys', 'gp_raw', 'gp_phys', 'gp_prior']

    for frac in FRACTIONS:
        print(f"\n{'=' * 50}")
        print(f"Testing with {frac*100:.0f}% of target data")
        print(f"{'=' * 50}")

        frac_results = []

        for run in range(N_RUNS):
            rs = RANDOM_STATE + run
            result = run_experiment(df, rf_male, scaler_male, frac,
                                    target_geometry, rs)

            if result is not None:
                frac_results.append(result)
                rf_avg = (result['rf_f0'] + result['rf_spl']) / 2
                gp_prior_avg = (result['gp_prior_f0'] + result['gp_prior_spl']) / 2
                print(f"  Run {run+1}: n_train={result['n_train']}, "
                      f"RF={rf_avg:.3f}, GP+Prior={gp_prior_avg:.3f}")

        if frac_results:
            avg = {
                'frac': frac,
                'n_train': int(np.mean([r['n_train'] for r in frac_results])),
            }
            for m in methods:
                for t in ['f0', 'spl']:
                    vals = [r[f'{m}_{t}'] for r in frac_results]
                    avg[f'{m}_{t}_mean'] = np.mean(vals)
                    avg[f'{m}_{t}_std'] = np.std(vals)
            all_results.append(avg)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY — Average R^2 (F0 + SPL) / 2")
    print("=" * 70)

    header = (f"{'Frac':<7} {'N':>5}  {'RF':>7} {'RF+Phy':>7} "
              f"{'GP Raw':>7} {'GP+Phy':>7} {'GP+Pri':>7}  {'Best':<10}")
    print(f"\n{header}")
    print("-" * 72)

    for r in all_results:
        scores = {}
        for m, label in [('rf', 'RF'), ('rf_phys', 'RF+Phy'),
                         ('gp_raw', 'GP Raw'), ('gp_phys', 'GP+Phy'),
                         ('gp_prior', 'GP+Pri')]:
            scores[label] = (r[f'{m}_f0_mean'] + r[f'{m}_spl_mean']) / 2

        best = max(scores, key=scores.get)

        print(f"{r['frac']*100:>4.0f}%  {r['n_train']:>5}  "
              f"{scores['RF']:>7.3f} {scores['RF+Phy']:>7.3f} "
              f"{scores['GP Raw']:>7.3f} {scores['GP+Phy']:>7.3f} "
              f"{scores['GP+Pri']:>7.3f}  {best}")

    # --- F0 breakdown ---
    print(f"\n\nF0 R^2 BREAKDOWN:")
    print(f"{'Frac':<7} {'N':>5}  {'RF':>7} {'RF+Phy':>7} "
          f"{'GP Raw':>7} {'GP+Phy':>7} {'GP+Pri':>7}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['frac']*100:>4.0f}%  {r['n_train']:>5}  "
              f"{r['rf_f0_mean']:>7.3f} {r['rf_phys_f0_mean']:>7.3f} "
              f"{r['gp_raw_f0_mean']:>7.3f} {r['gp_phys_f0_mean']:>7.3f} "
              f"{r['gp_prior_f0_mean']:>7.3f}")

    # --- SPL breakdown ---
    print(f"\n\nSPL R^2 BREAKDOWN:")
    print(f"{'Frac':<7} {'N':>5}  {'RF':>7} {'RF+Phy':>7} "
          f"{'GP Raw':>7} {'GP+Phy':>7} {'GP+Pri':>7}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['frac']*100:>4.0f}%  {r['n_train']:>5}  "
              f"{r['rf_spl_mean']:>7.3f} {r['rf_phys_spl_mean']:>7.3f} "
              f"{r['gp_raw_spl_mean']:>7.3f} {r['gp_phys_spl_mean']:>7.3f} "
              f"{r['gp_prior_spl_mean']:>7.3f}")

    # ==================== PLOTS ====================
    print("\n\nGenerating plots...")
    fig_dir = os.path.join(PROJECT_ROOT, 'Beam_Membrane', 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    n_trains = [r['n_train'] for r in all_results]

    methods_plot = [
        ('rf',       'RF Target-Only',         'blue',   '--', 'o'),
        ('rf_phys',  'RF + Physics Features',  'cyan',   '--', 's'),
        ('gp_raw',   'GP (raw features)',       'green',  '-',  '^'),
        ('gp_phys',  'GP + Physics',            'orange', '-',  'D'),
        ('gp_prior', 'GP + Physics + Prior',    'red',    '-',  '*'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GP vs RF: Data Efficiency for Beam-Membrane Prediction',
                 fontsize=14, fontweight='bold')

    # --- Plot 1: Combined R^2 ---
    ax = axes[0, 0]
    for key, label, color, ls, marker in methods_plot:
        r2 = [(r[f'{key}_f0_mean'] + r[f'{key}_spl_mean']) / 2
              for r in all_results]
        ax.plot(n_trains, r2, marker=marker, label=label, color=color,
                linestyle=ls, linewidth=2, markersize=8)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Average R^2 (F0 + SPL)')
    ax.set_title('Combined Performance')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.0])

    # --- Plot 2: F0 R^2 ---
    ax = axes[0, 1]
    for key, label, color, ls, marker in methods_plot:
        r2 = [r[f'{key}_f0_mean'] for r in all_results]
        ax.plot(n_trains, r2, marker=marker, label=label, color=color,
                linestyle=ls, linewidth=2, markersize=8)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R^2 for F0')
    ax.set_title('F0 Prediction')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.0])

    # --- Plot 3: SPL R^2 ---
    ax = axes[1, 0]
    for key, label, color, ls, marker in methods_plot:
        r2 = [r[f'{key}_spl_mean'] for r in all_results]
        ax.plot(n_trains, r2, marker=marker, label=label, color=color,
                linestyle=ls, linewidth=2, markersize=8)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R^2 for SPL')
    ax.set_title('SPL Prediction')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.0])

    # --- Plot 4: Improvement over RF baseline ---
    ax = axes[1, 1]
    for key, label, color, ls, marker in methods_plot[1:]:
        improvement = [
            ((r[f'{key}_f0_mean'] + r[f'{key}_spl_mean']) / 2) -
            ((r['rf_f0_mean'] + r['rf_spl_mean']) / 2)
            for r in all_results
        ]
        ax.plot(n_trains, improvement, marker=marker, label=label, color=color,
                linestyle=ls, linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R^2 Improvement over RF')
    ax.set_title('Improvement vs RF Target-Only')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(fig_dir, 'gp_vs_rf_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {plot_path}")

    # ==================== ANALYSIS ====================
    print("\n\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nGP + Physics + Prior vs RF Target-Only at each data size:")
    print("-" * 60)

    for r in all_results:
        rf_r2 = (r['rf_f0_mean'] + r['rf_spl_mean']) / 2
        gp_prior_r2 = (r['gp_prior_f0_mean'] + r['gp_prior_spl_mean']) / 2
        gp_phys_r2 = (r['gp_phys_f0_mean'] + r['gp_phys_spl_mean']) / 2
        improvement = gp_prior_r2 - rf_r2

        if improvement > 0.02:
            status = "BETTER"
        elif improvement > -0.02:
            status = "SIMILAR"
        else:
            status = "WORSE"

        print(f"  {r['frac']*100:>4.0f}% ({r['n_train']:>3} samples): "
              f"RF={rf_r2:.3f}  GP+Phy={gp_phys_r2:.3f}  "
              f"GP+Prior={gp_prior_r2:.3f}  [{status}, {improvement:+.3f}]")

    # Find where GP+Prior matches RF at higher data level
    print("\nDATA SAVINGS ANALYSIS:")
    print("-" * 60)
    for i in range(len(all_results)):
        gp_r2 = (all_results[i]['gp_prior_f0_mean'] +
                  all_results[i]['gp_prior_spl_mean']) / 2
        # Find smallest RF that matches this R^2
        for j in range(len(all_results)):
            rf_r2 = (all_results[j]['rf_f0_mean'] +
                      all_results[j]['rf_spl_mean']) / 2
            if rf_r2 >= gp_r2 - 0.01:
                if all_results[j]['n_train'] > all_results[i]['n_train']:
                    savings = 1 - all_results[i]['n_train'] / all_results[j]['n_train']
                    print(f"  GP+Prior with {all_results[i]['n_train']} samples "
                          f"(R^2={gp_r2:.3f}) matches RF with "
                          f"{all_results[j]['n_train']} samples "
                          f"-> {savings*100:.0f}% fewer simulations needed")
                break

    elapsed = time.time() - t0
    print(f"\n\nTotal runtime: {elapsed:.1f} seconds")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
