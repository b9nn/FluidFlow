"""
Physics-Informed Transfer Learning for Male->Female Vocal Fold RF

Ports cheap-to-compute physics equations from the MATLAB vocal fold model
(tissue stress-strain laws, beam mechanics) to Python and uses them as
features to bridge the male-female domain gap -- without running any
MATLAB simulations.

Physics equations extracted from:
  - MUC.m, LIG.m, TA.m            (tissue stress-strain)
  - Membrane_Beam_Parameters.m     (axial tension, bending stiffness)
  - MuscleControlModel.m           (strain from muscle activations)
  - Stiffness_Damping_Membrane_Beam.m  (geometry constants)
  - BodyCoverModel.m               (male/female geometry)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ==================== GEOMETRY PARAMETERS ====================
# From Stiffness_Damping_Membrane_Beam.m and BodyCoverModel.m

MALE_GEOM = {
    'L0': 1.5e-2,       # [m] vocal fold length at rest
    'b': 5e-3,          # [m] thickness (medial-lateral)
    'A_MUC': 5.0e-6,    # [m^2] mucosa cross-sectional area
    'A_LIG': 6.1e-6,    # [m^2] ligament cross-sectional area
    'A_TA': 40.9e-6,    # [m^2] TA muscle cross-sectional area
}

FEMALE_GEOM = {
    'L0': 1.0e-2,       # [m] vocal fold length at rest
    'b': 2.0e-3,        # [m] thickness (medial-lateral)
    'A_MUC': 3.0e-6,    # [m^2] mucosa cross-section (depth 1.5mm x thickness 2mm)
    'A_LIG': 3.0e-6,    # [m^2] ligament cross-section (depth 1.5mm x thickness 2mm)
    'A_TA': 6.0e-6,     # [m^2] TA muscle cross-section (depth 3mm x thickness 2mm)
}

# Material densities (same for both sexes)
RHO_MUC = 1000.0    # [kg/m^3]
RHO_TA = 1050.0     # [kg/m^3]
RHO_LIG = 1030.0    # [kg/m^3]

# Strain model constants (from MuscleControlModel.m)
G = 0.2   # Gain of elongation
R = 3.0   # Torque ratio


# ==================== TISSUE STRESS-STRAIN LAWS ====================
# From MUC.m, LIG.m, TA.m

def stress_MUC(eps):
    """Mucosa: sigma = sign(eps)*A*(exp(B*|eps|)-1)"""
    A, B = 1.5e3, 7.0
    sigma = np.sign(eps) * A * (np.exp(B * np.abs(eps)) - 1)
    E = A * B * np.exp(B * np.abs(eps))
    return sigma, E


def stress_LIG(eps):
    """Ligament: sigma = sign(eps)*A*(exp(B*|eps|)-1)"""
    A, B = 2.0e3, 10.0
    sigma = np.sign(eps) * A * (np.exp(B * np.abs(eps)) - 1)
    E = A * B * np.exp(B * np.abs(eps))
    return sigma, E


def stress_TA(eps, a_TA):
    """TA muscle: passive + active stress = sign(eps)*A*(exp(B*|eps|)-1) + a_TA*Sigma_m"""
    A, B = 1.0e3, 8.0
    Sigma_m = 105.0e3   # [Pa] maximum active stress
    sigma = np.sign(eps) * A * (np.exp(B * np.abs(eps)) - 1) + a_TA * Sigma_m
    E = A * B * np.exp(B * np.abs(eps))
    return sigma, E


# ==================== PHYSICS FEATURE COMPUTATION ====================

def compute_eps0(a_CT, a_TA):
    """
    Linearised strain from muscle activations.
    From MuscleControlModel: eps0 ~ G*(R*a_CT - a_TA)
    G=0.2 (elongation gain), R=3.0 (torque ratio)
    """
    return G * (R * a_CT - a_TA)


def compute_beam_parameters(eps0, a_TA, geom):
    """
    Port of Membrane_Beam_Parameters.m
    Computes axial tension F_beam and bending stiffness Mu_c_beam
    for a given geometry.
    """
    b = geom['b']
    A_MUC = geom['A_MUC']
    A_LIG = geom['A_LIG']
    A_TA = geom['A_TA']

    # Layer depths
    d_MUC = A_MUC / b
    d_LIG = A_LIG / b
    d_TA = A_TA / b

    # Area moments of inertia
    I_LIG = (b * d_LIG**3) / 12.0
    I_TA = (b * d_TA**3) / 12.0

    # Centerline positions from base of TA
    rc = d_TA + 0.5 * d_LIG
    r_LIG = d_TA + 0.5 * d_LIG
    r_TA = 0.5 * d_TA

    # Stresses and tangent moduli
    sigma_MUC, _ = stress_MUC(eps0)
    sigma_LIG, E_LIG = stress_LIG(eps0)
    sigma_TA, E_TA = stress_TA(eps0, a_TA)

    # Axial forces
    F_LIG = A_LIG * sigma_LIG
    F_TA = A_TA * sigma_TA
    F_beam = F_LIG + F_TA

    # Bending stiffness (from Membrane_Beam_Parameters.m)
    alpha_LIG_denom = 2.0 * ((E_LIG * A_LIG) / (E_TA * A_TA + 1e-30)) + 1.0
    alpha_LIG = -(d_TA + d_LIG) / alpha_LIG_denom

    alpha_TA_denom = 2.0 * ((E_TA * A_TA) / (E_LIG * A_LIG + 1e-30)) + 1.0
    alpha_TA = (d_TA + d_LIG) / alpha_TA_denom

    Theta = E_LIG * I_LIG + E_TA * I_TA
    Theta += (rc - r_LIG) * A_LIG * E_LIG * alpha_LIG
    Theta += (rc - r_TA) * A_TA * E_TA * alpha_TA
    Mu_c_beam = Theta * (1.0 + eps0)

    return F_beam, Mu_c_beam, sigma_MUC


def compute_freq_scale(F_beam, geom):
    """
    Analytical frequency estimate from beam theory:
    f ~ (1/L) * sqrt(F_beam / rho_b)
    where rho_b = rho_TA * A_TA + rho_LIG * A_LIG (linear mass density)
    """
    L0 = geom['L0']
    rho_b = RHO_TA * geom['A_TA'] + RHO_LIG * geom['A_LIG']
    # Avoid sqrt of negative
    if np.isscalar(F_beam):
        if F_beam <= 0 or rho_b <= 0:
            return 0.0
        return (1.0 / L0) * np.sqrt(F_beam / rho_b)
    else:
        result = np.zeros_like(F_beam)
        valid = (F_beam > 0) & (rho_b > 0)
        result[valid] = (1.0 / L0) * np.sqrt(F_beam[valid] / rho_b)
        return result


def compute_physics_features(a_CT_arr, a_TA_arr, PS_arr, geom):
    """
    Compute all physics features for a given geometry.
    Returns a dict of feature arrays.
    """
    n = len(a_CT_arr)
    eps0 = np.array([compute_eps0(a_CT_arr.iloc[i], a_TA_arr.iloc[i])
                     for i in range(n)])

    F_beam = np.zeros(n)
    Mu_c = np.zeros(n)
    sigma_muc = np.zeros(n)

    for i in range(n):
        fb, mc, sm = compute_beam_parameters(eps0[i], a_TA_arr.iloc[i], geom)
        F_beam[i] = fb
        Mu_c[i] = mc
        sigma_muc[i] = sm

    freq_scale = compute_freq_scale(F_beam, geom)

    return {
        'eps0': eps0,
        'F_beam': F_beam,
        'Mu_c': Mu_c,
        'sigma_MUC': sigma_muc,
        'freq_scale': freq_scale,
    }


def build_physics_feature_matrix(X, geom):
    """Build physics features for a dataframe with columns [a_CT, a_TA, PS]."""
    feats = compute_physics_features(X['a_CT'], X['a_TA'], X['PS'], geom)
    return np.column_stack([
        feats['eps0'],
        feats['F_beam'],
        feats['Mu_c'],
        feats['sigma_MUC'],
        feats['freq_scale'],
    ])


PHYSICS_FEATURE_NAMES = ['eps0', 'F_beam', 'Mu_c', 'sigma_MUC', 'freq_scale']


# ==================== ADAPTIVE MODEL COMPLEXITY ====================

def get_model_params(n_samples):
    """Same adaptive complexity as DataEfficiencyExperiment.py"""
    if n_samples < 100:
        return {
            'n_estimators': 20,
            'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20)
        }
    elif n_samples < 250:
        return {
            'n_estimators': 30,
            'max_depth': 5,
            'min_samples_leaf': 5,
            'min_samples_split': 5
        }
    elif n_samples < 500:
        return {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_leaf': 3,
            'min_samples_split': 3
        }
    elif n_samples < 1000:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 2,
            'min_samples_split': 2
        }
    else:
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2
        }


# ==================== EXPERIMENT ====================

def run_experiment(df_full, n_female, rf_male, scaler_X_male, random_state=42):
    """
    Run a single physics-informed transfer experiment with n_female training samples.
    Returns R^2 scores for all methods.
    """
    # Sample data
    if n_female < len(df_full):
        df = df_full.sample(n=n_female, random_state=random_state)
    else:
        df = df_full.copy()

    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    if len(df) < 20:
        return None

    # Split: 80% train, 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state
    )

    n_train = len(X_train)
    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_test_arr = Y_test.values

    results = {'n_female': n_female, 'n_train': n_train, 'n_test': len(X_test)}

    # --- Male model predictions ---
    X_train_male_scaled = scaler_X_male.transform(X_train)
    X_test_male_scaled = scaler_X_male.transform(X_test)
    pred_male_train = rf_male.predict(X_train_male_scaled)
    pred_male_test = rf_male.predict(X_test_male_scaled)

    results['source_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_male_test[:, 0])
    results['source_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_male_test[:, 1])

    # --- Compute physics features ---
    phys_male_train = build_physics_feature_matrix(X_train, MALE_GEOM)
    phys_male_test = build_physics_feature_matrix(X_test, MALE_GEOM)
    phys_female_train = build_physics_feature_matrix(X_train, FEMALE_GEOM)
    phys_female_test = build_physics_feature_matrix(X_test, FEMALE_GEOM)

    freq_scale_male_train = phys_male_train[:, 4]
    freq_scale_male_test = phys_male_test[:, 4]
    freq_scale_female_train = phys_female_train[:, 4]
    freq_scale_female_test = phys_female_test[:, 4]

    # --- BASELINE: Target-only RF (same as DataEfficiencyExperiment.py) ---
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_target.fit(X_train_scaled, Y_train_arr)
    pred_target = rf_target.predict(X_test_scaled)

    results['target_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_target[:, 1])

    # --- BASELINE: Residual correction (from DataEfficiencyExperiment.py) ---
    residuals_train = Y_train_arr - pred_male_train
    rf_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_residuals.fit(X_train.values, residuals_train)
    pred_residual = pred_male_test + rf_residuals.predict(X_test.values)

    results['residual_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual[:, 0])
    results['residual_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual[:, 1])

    # --- BASELINE: Feature augmentation with male predictions ---
    X_train_aug_base = np.column_stack([X_train.values, pred_male_train])
    X_test_aug_base = np.column_stack([X_test.values, pred_male_test])

    rf_aug_base = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_aug_base.fit(X_train_aug_base, Y_train_arr)
    pred_aug_base = rf_aug_base.predict(X_test_aug_base)

    results['augmented_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_aug_base[:, 0])
    results['augmented_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_aug_base[:, 1])

    # ==================== PHYSICS-INFORMED METHODS ====================

    # --- METHOD 1: Physics-scaled correction ---
    # F0_female ~ F0_male_pred * (freq_scale_female / freq_scale_male)
    # Then Ridge regression to calibrate with female samples
    freq_ratio_train = np.where(
        freq_scale_male_train > 0,
        freq_scale_female_train / freq_scale_male_train,
        1.0
    )
    freq_ratio_test = np.where(
        freq_scale_male_test > 0,
        freq_scale_female_test / freq_scale_male_test,
        1.0
    )

    # Raw physics-scaled prediction
    f0_scaled_train = pred_male_train[:, 0] * freq_ratio_train
    f0_scaled_test = pred_male_test[:, 0] * freq_ratio_test

    # Calibrate with Ridge regression: learn (scale, offset) from female data
    ridge_features_train = np.column_stack([f0_scaled_train, freq_ratio_train, pred_male_train[:, 1]])
    ridge_features_test = np.column_stack([f0_scaled_test, freq_ratio_test, pred_male_test[:, 1]])

    ridge = Ridge(alpha=1.0)
    ridge.fit(ridge_features_train, Y_train_arr)
    pred_physics_scaled = ridge.predict(ridge_features_test)

    results['phys_scaled_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_physics_scaled[:, 0])
    results['phys_scaled_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_physics_scaled[:, 1])

    # --- METHOD 2: Physics-augmented RF ---
    # Train female RF on (a_CT, a_TA, PS, F0_male_pred, SPL_male_pred,
    #                      freq_scale_female, F_beam_female, eps0, ...)
    X_train_phys_aug = np.column_stack([
        X_train.values,
        pred_male_train,
        phys_female_train,
    ])
    X_test_phys_aug = np.column_stack([
        X_test.values,
        pred_male_test,
        phys_female_test,
    ])

    rf_phys_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **model_params)
    )
    rf_phys_aug.fit(X_train_phys_aug, Y_train_arr)
    pred_phys_aug = rf_phys_aug.predict(X_test_phys_aug)

    results['phys_augmented_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_phys_aug[:, 0])
    results['phys_augmented_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_phys_aug[:, 1])

    # --- METHOD 3: Physics-normalised transfer ---
    # Divide F0 by freq_scale to get a normalised residual that should be
    # more similar across sexes
    safe_fs_female_train = np.where(freq_scale_female_train > 0, freq_scale_female_train, 1.0)
    safe_fs_female_test = np.where(freq_scale_female_test > 0, freq_scale_female_test, 1.0)
    safe_fs_male_train = np.where(freq_scale_male_train > 0, freq_scale_male_train, 1.0)
    safe_fs_male_test = np.where(freq_scale_male_test > 0, freq_scale_male_test, 1.0)

    # Normalise male predictions by male freq_scale
    f0_male_norm_train = pred_male_train[:, 0] / safe_fs_male_train
    f0_male_norm_test = pred_male_test[:, 0] / safe_fs_male_test

    # Normalise female targets by female freq_scale
    f0_female_norm_train = Y_train_arr[:, 0] / safe_fs_female_train

    # Train RF to predict normalised residual = (F0_female/freq_female) - (F0_male/freq_male)
    norm_residual_train = f0_female_norm_train - f0_male_norm_train

    # Features: activations + male normalised pred + physics features
    X_train_norm = np.column_stack([
        X_train.values,
        f0_male_norm_train.reshape(-1, 1),
        pred_male_train[:, 1].reshape(-1, 1),
        phys_female_train,
    ])
    X_test_norm = np.column_stack([
        X_test.values,
        f0_male_norm_test.reshape(-1, 1),
        pred_male_test[:, 1].reshape(-1, 1),
        phys_female_test,
    ])

    # Train RF for normalised F0 residual
    rf_norm_f0 = RandomForestRegressor(random_state=random_state, **model_params)
    rf_norm_f0.fit(X_train_norm, norm_residual_train)
    norm_residual_test = rf_norm_f0.predict(X_test_norm)

    # Reconstruct: F0_female = (male_norm + residual) * freq_scale_female
    f0_norm_pred = (f0_male_norm_test + norm_residual_test) * safe_fs_female_test

    # For SPL, use a separate RF with physics features (SPL doesn't scale as cleanly)
    rf_norm_spl = RandomForestRegressor(random_state=random_state, **model_params)
    rf_norm_spl.fit(X_train_norm, Y_train_arr[:, 1])
    spl_norm_pred = rf_norm_spl.predict(X_test_norm)

    results['phys_normalised_f0_r2'] = r2_score(Y_test_arr[:, 0], f0_norm_pred)
    results['phys_normalised_spl_r2'] = r2_score(Y_test_arr[:, 1], spl_norm_pred)

    return results


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("PHYSICS-INFORMED TRANSFER LEARNING EXPERIMENT")
    print("=" * 70)

    # --- Validate physics features ---
    print("\n--- Physics Feature Sanity Check ---")
    test_a_CT = np.array([0.0, 0.5, 1.0])
    test_a_TA = np.array([0.0, 0.5, 1.0])
    for a_ct, a_ta in zip(test_a_CT, test_a_TA):
        eps = compute_eps0(a_ct, a_ta)
        fb_m, mc_m, sm_m = compute_beam_parameters(eps, a_ta, MALE_GEOM)
        fb_f, mc_f, sm_f = compute_beam_parameters(eps, a_ta, FEMALE_GEOM)
        fs_m = compute_freq_scale(fb_m, MALE_GEOM)
        fs_f = compute_freq_scale(fb_f, FEMALE_GEOM)
        print(f"  a_CT={a_ct:.1f}, a_TA={a_ta:.1f}: eps0={eps:.3f}, "
              f"F_beam_male={fb_m:.2f}, F_beam_female={fb_f:.2f}, "
              f"freq_male={fs_m:.1f} Hz, freq_female={fs_f:.1f} Hz")

    # --- Load models and data ---
    print("\nLoading pre-trained male model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rf_male = joblib.load(os.path.join(script_dir, 'models', 'RF_BCM.pkl'))
    scaler_X_male = joblib.load(os.path.join(script_dir, 'models', 'x_scaler_BCM.pkl'))
    print("Done.")

    print("\nLoading female dataset...")
    df_full = pd.read_parquet(os.path.join(script_dir, 'Female_binary.parquet'))
    df_full = df_full[['a_CT', 'a_TA', 'PS', 'F0', 'SPL', 'ACFL']].dropna()
    df_full = df_full[df_full['ACFL'] > 30]
    print(f"Total female samples: {len(df_full)}")

    # --- Run experiments ---
    SAMPLE_SIZES = [10, 25, 50, 100, 250, 500, 1000]
    N_RUNS = 5

    all_results = []

    for n_female in SAMPLE_SIZES:
        if n_female > len(df_full):
            print(f"\nSkipping n={n_female} (only {len(df_full)} samples available)")
            continue

        print(f"\n{'='*50}")
        print(f"Testing with {n_female} female training samples")
        print(f"{'='*50}")

        run_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df_full, n_female, rf_male, scaler_X_male, rs)
            if result is not None:
                run_results.append(result)
                avg_target = (result['target_only_f0_r2'] + result['target_only_spl_r2']) / 2
                avg_phys_aug = (result['phys_augmented_f0_r2'] + result['phys_augmented_spl_r2']) / 2
                print(f"  Run {run+1}: Target R2={avg_target:.3f}, "
                      f"PhysAug R2={avg_phys_aug:.3f}")

        if run_results:
            avg = {'n_female': n_female, 'n_train': int(np.mean([r['n_train'] for r in run_results]))}
            methods = ['source_only', 'target_only', 'residual', 'augmented',
                       'phys_scaled', 'phys_augmented', 'phys_normalised']
            for method in methods:
                for target in ['f0', 'spl']:
                    key = f'{method}_{target}_r2'
                    vals = [r[key] for r in run_results]
                    avg[f'{key}_mean'] = np.mean(vals)
                    avg[f'{key}_std'] = np.std(vals)
            all_results.append(avg)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary_data = []
    for r in all_results:
        row = {
            'N_Female': r['n_female'],
            'N_Train': r['n_train'],
        }
        methods_labels = [
            ('source_only', 'Source(Male)'),
            ('target_only', 'Target(Fem)'),
            ('residual', 'Residual'),
            ('augmented', 'FeatAug'),
            ('phys_scaled', 'PhysScale'),
            ('phys_augmented', 'PhysAug'),
            ('phys_normalised', 'PhysNorm'),
        ]
        for key, label in methods_labels:
            row[label] = (r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("\nAverage R2 (F0 + SPL) / 2:")
    print(summary_df.to_string(index=False, float_format='%.3f'))

    # Best method per sample size
    print("\n\nBEST METHOD PER SAMPLE SIZE:")
    print("-" * 60)
    method_cols = [l for _, l in methods_labels]
    for _, row in summary_df.iterrows():
        best = max(method_cols, key=lambda m: row[m])
        print(f"  N={int(row['N_Female']):>5}: {best:<12} (R2={row[best]:.3f})")

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating plots...")
    os.makedirs(os.path.join(script_dir, 'transfer-figs'), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Physics-Informed Transfer Learning: R2 vs Female Sample Count',
                 fontsize=14, fontweight='bold')

    n_samples = [r['n_female'] for r in all_results]

    methods_to_plot = [
        ('source_only', 'Source Only (Male)', 'gray', '--', 'x'),
        ('target_only', 'Target Only (Female)', 'blue', '-', 'o'),
        ('residual', 'Residual Correction', 'purple', '-', 's'),
        ('augmented', 'Feature Augmentation', 'green', '-', 'D'),
        ('phys_scaled', 'Physics-Scaled', 'orange', '-', '^'),
        ('phys_augmented', 'Physics-Augmented RF', 'red', '-', 'v'),
        ('phys_normalised', 'Physics-Normalised', 'darkred', '-', 'P'),
    ]

    # F0 plot
    ax = axes[0]
    for key, label, color, ls, marker in methods_to_plot:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        stds = [r[f'{key}_f0_r2_std'] for r in all_results]
        ax.errorbar(n_samples, vals, yerr=stds, marker=marker, label=label,
                    color=color, linestyle=ls, linewidth=2, capsize=3, markersize=6)
    ax.set_xlabel('Number of Female Training Samples')
    ax.set_ylabel('R2 for F0')
    ax.set_title('F0 Prediction')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])
    ax.set_xscale('log')

    # SPL plot
    ax = axes[1]
    for key, label, color, ls, marker in methods_to_plot:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        stds = [r[f'{key}_spl_r2_std'] for r in all_results]
        ax.errorbar(n_samples, vals, yerr=stds, marker=marker, label=label,
                    color=color, linestyle=ls, linewidth=2, capsize=3, markersize=6)
    ax.set_xlabel('Number of Female Training Samples')
    ax.set_ylabel('R2 for SPL')
    ax.set_title('SPL Prediction')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])
    ax.set_xscale('log')

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'transfer-figs', 'physics_informed_transfer.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {fig_path}")

    # Combined average plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Physics-Informed Transfer: Average R2 vs Sample Count',
                  fontsize=14, fontweight='bold')

    for key, label, color, ls, marker in methods_to_plot:
        avg_r2 = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
                  for r in all_results]
        ax2.plot(n_samples, avg_r2, marker=marker, label=label,
                 color=color, linestyle=ls, linewidth=2, markersize=8)

    ax2.set_xlabel('Number of Female Training Samples', fontsize=12)
    ax2.set_ylabel('Average R2 (F0 + SPL) / 2', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.5, 1.0])
    ax2.set_xscale('log')

    plt.tight_layout()
    fig2_path = os.path.join(script_dir, 'transfer-figs', 'physics_informed_combined.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {fig2_path}")

    plt.show()

    # ==================== KEY FINDINGS ====================
    print("\n\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find where physics-informed beats target-only
    for r in all_results:
        target_avg = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        phys_aug_avg = (r['phys_augmented_f0_r2_mean'] + r['phys_augmented_spl_r2_mean']) / 2
        phys_norm_avg = (r['phys_normalised_f0_r2_mean'] + r['phys_normalised_spl_r2_mean']) / 2
        best_phys = max(phys_aug_avg, phys_norm_avg)
        improvement = best_phys - target_avg

        if improvement > 0.01:
            print(f"  N={r['n_female']:>5}: Physics-informed improves by {improvement:.3f} R2 "
                  f"(target={target_avg:.3f}, best_physics={best_phys:.3f})")
        else:
            print(f"  N={r['n_female']:>5}: Target-only competitive "
                  f"(target={target_avg:.3f}, best_physics={best_phys:.3f})")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
