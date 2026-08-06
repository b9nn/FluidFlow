"""
Transfer Learning: TBCM -> Beam-Membrane (BM) Model + Chain Methods

Tests whether TBCM is a better source than BCM for BM transfer, and
whether using both sources together (chain) beats either alone.

Why TBCM might be better:
  - TBCM Ps [448-2163] covers BM [600-1000] with less extrapolation than BCM [10-2010]
  - TBCM SPL mean (~95) closer to BM (~107) than BCM (~69)
  - TBCM F0 [80-384] overlaps well with BM [116-426]

Methods:
  1. source_only:   TBCM model zero-shot on BM (shared features only)
  2. target_only:   RF on BM data alone (4 features incl a_LCA)
  3. residual:      Learn TBCM->BM residuals
  4. augmented:     BM inputs + TBCM predictions as features
  5. simple_ens:    0.3*TBCM + 0.7*target fixed weighting
  6. transrf:       Learned 3-component weights (target, residual, augmented)
  7. chain_aug:     BM inputs + TBCM preds + BCM preds (8 features)
  8. chain_transrf: Learned 4-component weights (target, TBCM-residual, BCM-residual, chain-aug)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import json
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 5

# Shared features between source models (BCM/TBCM) and BM
SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
# Full target features (BM has extra a_LCA input)
TARGET_FEATURES = ['a_CT', 'a_TA', 'PS', 'a_LCA']
TARGETS = ['F0', 'SPL']


# ==================== ADAPTIVE MODEL COMPLEXITY ====================
def get_model_params(n_samples):
    """Adjust RF complexity based on available training samples."""
    if n_samples < 100:
        return {
            'n_estimators': 20, 'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20),
        }
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    elif n_samples < 500:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}
    elif n_samples < 1000:
        return {'n_estimators': 100, 'max_depth': 10,
                'min_samples_leaf': 2, 'min_samples_split': 2}
    else:
        return {'n_estimators': 200, 'max_depth': None,
                'min_samples_leaf': 1, 'min_samples_split': 2}


# ==================== HELPERS ====================
def get_source_predictions(X, scaler, model):
    """Get predictions from a source model (shared features only)."""
    return model.predict(scaler.transform(X[SHARED_FEATURES]))


# ==================== SINGLE EXPERIMENT ====================
def run_experiment(df_full, frac, rf_tbcm, scaler_X_tbcm,
                   rf_bcm, scaler_X_bcm, random_state=42,
                   target_features=None):
    """Run one transfer learning experiment at a given data fraction.

    Implements all 8 methods: 6 standard (TBCM as source) + 2 chain (TBCM+BCM).
    """
    if target_features is None:
        target_features = TARGET_FEATURES

    if frac < 1.0:
        df = df_full.sample(frac=frac, random_state=random_state)
    else:
        df = df_full.copy()

    X = df[target_features]
    Y = df[TARGETS]

    if len(df) < 20:
        return None

    # 80/20 train/test split
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    # Further split train into train/val for weight learning
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=random_state)

    n_train = len(X_train)
    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_val_arr = Y_val.values
    Y_test_arr = Y_test.values

    results = {
        'frac': frac, 'n_total': len(df),
        'n_train': n_train, 'n_val': len(X_val), 'n_test': len(X_test),
    }

    # === 1. SOURCE ONLY (TBCM model, zero-shot) ===
    pred_tbcm_test = get_source_predictions(X_test, scaler_X_tbcm, rf_tbcm)
    results['source_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_tbcm_test[:, 0])
    results['source_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_tbcm_test[:, 1])

    # === 2. TARGET ONLY ===
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    results['target_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_target[:, 1])

    # === 3. RESIDUAL CORRECTION (TBCM) ===
    pred_tbcm_train = get_source_predictions(X_train, scaler_X_tbcm, rf_tbcm)
    rf_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_residuals.fit(X_train.values, Y_train_arr - pred_tbcm_train)
    pred_residual = pred_tbcm_test + rf_residuals.predict(X_test.values)

    results['residual_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual[:, 0])
    results['residual_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual[:, 1])

    # === 4. FEATURE AUGMENTATION (TBCM preds as extra features) ===
    X_train_aug = np.column_stack([X_train.values, pred_tbcm_train])
    X_test_aug = np.column_stack([X_test.values, pred_tbcm_test])

    rf_augmented = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_augmented.fit(X_train_aug, Y_train_arr)
    pred_augmented = rf_augmented.predict(X_test_aug)

    results['augmented_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_augmented[:, 0])
    results['augmented_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_augmented[:, 1])

    # === 5. SIMPLE ENSEMBLE (fixed 0.3 source / 0.7 target) ===
    pred_simple = 0.3 * pred_tbcm_test + 0.7 * pred_target
    results['simple_ens_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_simple[:, 0])
    results['simple_ens_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_simple[:, 1])

    # === 6. TRANSRF ENSEMBLE (learned 3-component weights) ===
    X_val_sc = scaler_X.transform(X_val)
    pred_tbcm_val = get_source_predictions(X_val, scaler_X_tbcm, rf_tbcm)

    pred_t_val = rf_target.predict(X_val_sc)
    pred_r_val = pred_tbcm_val + rf_residuals.predict(X_val.values)
    X_val_aug = np.column_stack([X_val.values, pred_tbcm_val])
    pred_a_val = rf_augmented.predict(X_val_aug)

    weights_3 = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i], pred_r_val[:, i], pred_a_val[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights_3[name] = w

    pred_transrf = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i], pred_residual[:, i], pred_augmented[:, i]])
        pred_transrf[:, i] = stack @ weights_3[name]

    results['transrf_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_transrf[:, 0])
    results['transrf_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_transrf[:, 1])
    results['weights_f0'] = weights_3['F0'].tolist()
    results['weights_spl'] = weights_3['SPL'].tolist()

    # === 7. CHAIN AUGMENTATION (BM inputs + TBCM preds + BCM preds) ===
    pred_bcm_train = get_source_predictions(X_train, scaler_X_bcm, rf_bcm)
    pred_bcm_test = get_source_predictions(X_test, scaler_X_bcm, rf_bcm)

    X_train_chain = np.column_stack([X_train.values, pred_tbcm_train, pred_bcm_train])
    X_test_chain = np.column_stack([X_test.values, pred_tbcm_test, pred_bcm_test])

    rf_chain_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_chain_aug.fit(X_train_chain, Y_train_arr)
    pred_chain_aug = rf_chain_aug.predict(X_test_chain)

    results['chain_aug_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_chain_aug[:, 0])
    results['chain_aug_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_chain_aug[:, 1])

    # === 8. CHAIN TRANSRF (4-component learned weights) ===
    # Train BCM residual RF
    rf_bcm_residuals = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_bcm_residuals.fit(X_train.values, Y_train_arr - pred_bcm_train)
    pred_bcm_residual_test = pred_bcm_test + rf_bcm_residuals.predict(X_test.values)

    # Validation predictions for all 4 sub-models
    pred_bcm_val = get_source_predictions(X_val, scaler_X_bcm, rf_bcm)
    pred_bcm_r_val = pred_bcm_val + rf_bcm_residuals.predict(X_val.values)
    X_val_chain = np.column_stack([X_val.values, pred_tbcm_val, pred_bcm_val])
    pred_chain_a_val = rf_chain_aug.predict(X_val_chain)

    weights_4 = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i],       # target-only
            pred_r_val[:, i],       # TBCM-residual
            pred_bcm_r_val[:, i],   # BCM-residual
            pred_chain_a_val[:, i], # chain-augmented
        ])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([0.25, 0.25, 0.25, 0.25])
        weights_4[name] = w

    pred_chain_transrf = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i],
            pred_residual[:, i],
            pred_bcm_residual_test[:, i],
            pred_chain_aug[:, i],
        ])
        pred_chain_transrf[:, i] = stack @ weights_4[name]

    results['chain_transrf_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_chain_transrf[:, 0])
    results['chain_transrf_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_chain_transrf[:, 1])
    results['chain_weights_f0'] = weights_4['F0'].tolist()
    results['chain_weights_spl'] = weights_4['SPL'].tolist()

    return results


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("TRANSFER LEARNING: TBCM -> BEAM-MEMBRANE (+ Chain Methods)")
    print("=" * 70)

    # ---- Load TBCM source data and train inline ----
    print("\nLoading TBCM source data...")
    tbcm_path = os.path.join(script_dir, '..', 'TBCM', 'dataset_TBCM.csv')
    df_tbcm = pd.read_csv(tbcm_path, index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'})
    df_tbcm = df_tbcm.drop(columns=['PL'])  # Not shared with BM
    print(f"  TBCM samples: {len(df_tbcm)}")
    print(f"  TBCM columns: {list(df_tbcm.columns)}")
    print(f"  TBCM F0 range: [{df_tbcm['F0'].min():.1f}, {df_tbcm['F0'].max():.1f}]")
    print(f"  TBCM SPL range: [{df_tbcm['SPL'].min():.1f}, {df_tbcm['SPL'].max():.1f}]")
    print(f"  TBCM PS range: [{df_tbcm['PS'].min():.1f}, {df_tbcm['PS'].max():.1f}]")

    X_tbcm = df_tbcm[SHARED_FEATURES]
    Y_tbcm = df_tbcm[TARGETS]

    scaler_X_tbcm = StandardScaler()
    X_tbcm_sc = scaler_X_tbcm.fit_transform(X_tbcm)

    print("\nTraining TBCM source RF model (300 trees, full data)...")
    rf_tbcm = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_tbcm.fit(X_tbcm_sc, Y_tbcm)
    pred_tbcm_self = rf_tbcm.predict(X_tbcm_sc)
    r2_f0 = r2_score(Y_tbcm['F0'], pred_tbcm_self[:, 0])
    r2_spl = r2_score(Y_tbcm['SPL'], pred_tbcm_self[:, 1])
    print(f"  TBCM self-R2: F0={r2_f0:.4f}, SPL={r2_spl:.4f}")

    # ---- Load BCM source data and train inline (for chain methods) ----
    print("\nLoading BCM source data...")
    bcm_path = os.path.join(script_dir, 'dataset_BCM.csv')
    df_bcm = pd.read_csv(bcm_path, index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    print(f"  BCM samples: {len(df_bcm)}")
    print(f"  BCM F0 range: [{df_bcm['F0'].min():.1f}, {df_bcm['F0'].max():.1f}]")
    print(f"  BCM SPL range: [{df_bcm['SPL'].min():.1f}, {df_bcm['SPL'].max():.1f}]")
    print(f"  BCM PS range: [{df_bcm['PS'].min():.1f}, {df_bcm['PS'].max():.1f}]")

    X_bcm = df_bcm[SHARED_FEATURES]
    Y_bcm = df_bcm[TARGETS]

    scaler_X_bcm = StandardScaler()
    X_bcm_sc = scaler_X_bcm.fit_transform(X_bcm)

    print("\nTraining BCM source RF model (300 trees, full data)...")
    rf_bcm = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_bcm.fit(X_bcm_sc, Y_bcm)
    pred_bcm_self = rf_bcm.predict(X_bcm_sc)
    r2_f0 = r2_score(Y_bcm['F0'], pred_bcm_self[:, 0])
    r2_spl = r2_score(Y_bcm['SPL'], pred_bcm_self[:, 1])
    print(f"  BCM self-R2: F0={r2_f0:.4f}, SPL={r2_spl:.4f}")

    # ---- Load target (BM) data ----
    print("\nLoading BM target data...")
    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    if not os.path.exists(bm_path):
        print(f"\n  ERROR: BM dataset not found!")
        print(f"  Expected: {os.path.join(script_dir, 'dataset_BM.csv')}")
        print(f"  Run Generate_BM_Dataset.m in MATLAB first.")
        return

    df_bm = pd.read_csv(bm_path)
    if 'Ps' in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})

    # Drop rows with NaN in F0 or SPL
    n_before = len(df_bm)
    df_bm = df_bm.dropna(subset=['F0', 'SPL'])
    n_dropped = n_before - len(df_bm)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with NaN F0/SPL")

    # Keep a_LCA as extra target-side input
    df_bm_full = df_bm[TARGET_FEATURES + TARGETS].copy()

    print(f"  BM samples: {len(df_bm_full)}")
    print(f"  BM columns: {list(df_bm_full.columns)}")
    print(f"  BM a_LCA range: [{df_bm_full['a_LCA'].min():.3f}, {df_bm_full['a_LCA'].max():.3f}]")
    print(f"  BM F0 range: [{df_bm_full['F0'].min():.1f}, {df_bm_full['F0'].max():.1f}]")
    print(f"  BM SPL range: [{df_bm_full['SPL'].min():.1f}, {df_bm_full['SPL'].max():.1f}]")
    print(f"  BM PS range: [{df_bm_full['PS'].min():.1f}, {df_bm_full['PS'].max():.1f}]")

    # ---- Domain gap summary ----
    print("\n" + "-" * 50)
    print("DOMAIN GAP SUMMARY (TBCM vs BCM vs BM)")
    print("-" * 50)
    print(f"  {'':>6} {'TBCM':<25} {'BCM':<25} {'BM':<25}")
    for col in SHARED_FEATURES + TARGETS:
        tbcm_r = f"[{df_tbcm[col].min():.1f}, {df_tbcm[col].max():.1f}]"
        bcm_r = f"[{df_bcm[col].min():.1f}, {df_bcm[col].max():.1f}]"
        bm_r = f"[{df_bm_full[col].min():.1f}, {df_bm_full[col].max():.1f}]"
        print(f"  {col:>6}: {tbcm_r:<25} {bcm_r:<25} {bm_r:<25}")

    # ---- Spearman correlation: both sources vs BM ----
    print("\n" + "=" * 70)
    print("SPEARMAN RANK CORRELATION (source predictions vs BM truth)")
    print("=" * 70)
    df_check = df_bm_full.sample(n=min(5000, len(df_bm_full)), random_state=42)
    X_check = df_check[SHARED_FEATURES]
    Y_check = df_check[TARGETS].values

    pred_tbcm_check = rf_tbcm.predict(scaler_X_tbcm.transform(X_check))
    pred_bcm_check = rf_bcm.predict(scaler_X_bcm.transform(X_check))

    print(f"\n  {'Source':<8} {'F0 rho':<12} {'F0 R2':<12} {'SPL rho':<12} {'SPL R2':<12}")
    print(f"  {'-'*52}")
    for src_name, pred in [('TBCM', pred_tbcm_check), ('BCM', pred_bcm_check)]:
        rho_f0, _ = spearmanr(Y_check[:, 0], pred[:, 0])
        rho_spl, _ = spearmanr(Y_check[:, 1], pred[:, 1])
        r2_f0_val = r2_score(Y_check[:, 0], pred[:, 0])
        r2_spl_val = r2_score(Y_check[:, 1], pred[:, 1])
        print(f"  {src_name:<8} {rho_f0:<12.4f} {r2_f0_val:<12.4f} "
              f"{rho_spl:<12.4f} {r2_spl_val:<12.4f}")

    print("\n  (Higher Spearman = better rank agreement → better transfer potential)")

    # ---- Run experiments ----
    print("\n" + "=" * 70)
    print("RUNNING TRANSFER EXPERIMENTS")
    print("  Primary source: TBCM | Secondary source: BCM (chain only)")
    print("  Target: BM (4 features: a_CT, a_TA, PS, a_LCA)")
    print("=" * 70)

    all_results = []

    for frac in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of BM data")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df_bm_full, frac, rf_tbcm, scaler_X_tbcm,
                                    rf_bcm, scaler_X_bcm, rs,
                                    target_features=TARGET_FEATURES)
            if result is not None:
                frac_results.append(result)
                avg_tgt = (result['target_only_f0_r2'] +
                           result['target_only_spl_r2']) / 2
                avg_trf = (result['transrf_f0_r2'] +
                           result['transrf_spl_r2']) / 2
                avg_chain = (result['chain_transrf_f0_r2'] +
                             result['chain_transrf_spl_r2']) / 2
                print(f"  Run {run+1}: n_train={result['n_train']}, "
                      f"Target={avg_tgt:.3f}, TransRF={avg_trf:.3f}, "
                      f"ChainTRF={avg_chain:.3f}")

        if frac_results:
            avg = {'frac': frac,
                   'n_samples': int(np.mean([r['n_train'] for r in frac_results]))}
            for key in ['source_only', 'target_only', 'residual', 'augmented',
                        'simple_ens', 'transrf', 'chain_aug', 'chain_transrf']:
                for t in ['f0', 'spl']:
                    vals = [r[f'{key}_{t}_r2'] for r in frac_results]
                    avg[f'{key}_{t}_r2_mean'] = float(np.mean(vals))
                    avg[f'{key}_{t}_r2_std'] = float(np.std(vals))
            # 3-component weights (TransRF)
            avg['weights_f0'] = np.mean(
                [r['weights_f0'] for r in frac_results], axis=0).tolist()
            avg['weights_spl'] = np.mean(
                [r['weights_spl'] for r in frac_results], axis=0).tolist()
            # 4-component weights (Chain TransRF)
            avg['chain_weights_f0'] = np.mean(
                [r['chain_weights_f0'] for r in frac_results], axis=0).tolist()
            avg['chain_weights_spl'] = np.mean(
                [r['chain_weights_spl'] for r in frac_results], axis=0).tolist()
            all_results.append(avg)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY (all 8 methods)")
    print("=" * 70)

    methods_all = ['source_only', 'target_only', 'residual', 'augmented',
                   'simple_ens', 'transrf', 'chain_aug', 'chain_transrf']
    method_labels = ['Source(TBCM)', 'Target', 'Residual', 'Augmented',
                     'SimpleEns', 'TransRF', 'ChainAug', 'ChainTRF']

    summary_data = []
    for r in all_results:
        row = {'Frac': f"{r['frac']*100:.0f}%", 'N': r['n_samples']}
        for key, label in zip(methods_all, method_labels):
            row[label] = (r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)
    print("\nAverage R2 (F0 + SPL) / 2:")
    print(summary_df.to_string(index=False, float_format='%.3f'))

    # Transfer gain table
    print("\n\nTRANSFER GAIN (method R2 - target-only R2):")
    print("-" * 80)
    header = (f"{'Frac':>5} {'N':>5} {'Resid':>8} {'Aug':>8} {'SimpE':>8} "
              f"{'TrRF':>8} {'ChAug':>8} {'ChTRF':>8}")
    print(header)
    for r in all_results:
        tgt = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        gains = []
        for key in ['residual', 'augmented', 'simple_ens', 'transrf',
                    'chain_aug', 'chain_transrf']:
            g = (r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2 - tgt
            gains.append(g)
        print(f"{r['frac']*100:>4.0f}% {r['n_samples']:>5} " +
              " ".join(f"{g:>+8.4f}" for g in gains))

    # F0 and SPL breakdown
    for target_name, target_key in [('F0', 'f0'), ('SPL', 'spl')]:
        print(f"\n{target_name} R2 BREAKDOWN:")
        print("-" * 90)
        header = (f"{'Frac':>5} {'Src':>7} {'Tgt':>7} {'Res':>7} {'Aug':>7} "
                  f"{'SmpE':>7} {'TrRF':>7} {'ChAug':>7} {'ChTRF':>7}")
        print(header)
        for r in all_results:
            vals = [r[f'{m}_{target_key}_r2_mean'] for m in methods_all]
            print(f"{r['frac']*100:>4.0f}% " +
                  " ".join(f"{v:>7.3f}" for v in vals))

    # Learned weights
    print("\n\nLEARNED TRANSRF WEIGHTS (Target, Residual, Augmented):")
    print("-" * 60)
    for r in all_results:
        wf = r['weights_f0']
        ws = r['weights_spl']
        print(f"  {r['frac']*100:>5.0f}%: "
              f"F0=[{wf[0]:.2f}, {wf[1]:.2f}, {wf[2]:.2f}]  "
              f"SPL=[{ws[0]:.2f}, {ws[1]:.2f}, {ws[2]:.2f}]")

    print("\n\nLEARNED CHAIN TRANSRF WEIGHTS (Target, TBCM-Resid, BCM-Resid, ChainAug):")
    print("-" * 70)
    for r in all_results:
        wf = r['chain_weights_f0']
        ws = r['chain_weights_spl']
        print(f"  {r['frac']*100:>5.0f}%: "
              f"F0=[{wf[0]:.2f}, {wf[1]:.2f}, {wf[2]:.2f}, {wf[3]:.2f}]  "
              f"SPL=[{ws[0]:.2f}, {ws[1]:.2f}, {ws[2]:.2f}, {ws[3]:.2f}]")

    # ==================== HEAD-TO-HEAD: BCM->BM vs TBCM->BM ====================
    prev_path = os.path.join(script_dir, 'results', 'rf_transfer_results_aLCA.json')
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev_results = json.load(f)
        print("\n\n" + "=" * 70)
        print("HEAD-TO-HEAD: BCM->BM TransRF vs TBCM->BM TransRF vs Chain TransRF")
        print("=" * 70)
        print(f"\n{'Frac':>5} {'N':>5}  {'BCM-TrRF':>10} {'TBCM-TrRF':>10} "
              f"{'Chain-TrRF':>10} {'Best':>12}")
        print("-" * 65)
        for r_new in all_results:
            frac = r_new['frac']
            tbcm_avg = (r_new['transrf_f0_r2_mean'] +
                        r_new['transrf_spl_r2_mean']) / 2
            chain_avg = (r_new['chain_transrf_f0_r2_mean'] +
                         r_new['chain_transrf_spl_r2_mean']) / 2
            for r_old in prev_results:
                if abs(r_old['frac'] - frac) < 0.001:
                    bcm_avg = (r_old['transrf_f0_r2_mean'] +
                               r_old['transrf_spl_r2_mean']) / 2
                    best = max(bcm_avg, tbcm_avg, chain_avg)
                    if best == chain_avg:
                        winner = "Chain"
                    elif best == tbcm_avg:
                        winner = "TBCM"
                    else:
                        winner = "BCM"
                    print(f"{frac*100:>4.0f}% {r_new['n_samples']:>5}  "
                          f"{bcm_avg:>10.4f} {tbcm_avg:>10.4f} "
                          f"{chain_avg:>10.4f} {winner:>12}")
                    break

    # ==================== SAVE RESULTS ====================
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'tbcm_to_bm_transfer_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ==================== VISUALIZATION ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    n_samples = [r['n_samples'] for r in all_results]
    methods_to_plot = [
        ('source_only', 'Source Only (TBCM)', 'gray', '--'),
        ('target_only', 'Target Only (BM)', 'blue', '-'),
        ('residual', 'Residual (TBCM)', 'purple', '-'),
        ('augmented', 'Augmented (TBCM)', 'green', '-'),
        ('simple_ens', 'Simple Ensemble', 'orange', '--'),
        ('transrf', 'TransRF (TBCM)', 'red', '-'),
        ('chain_aug', 'Chain Augmented', 'cyan', '-'),
        ('chain_transrf', 'Chain TransRF', 'darkred', '-'),
    ]

    # --- Plot 1: R2 vs sample count (F0 and SPL), all 8 methods ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Transfer Learning: TBCM -> BM (+ Chain Methods)',
                 fontsize=14, fontweight='bold')

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        stds = [r[f'{key}_f0_r2_std'] for r in all_results]
        axes[0].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[0].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.1, color=color)
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('R² for F0')
    axes[0].set_title('F0 Prediction')
    axes[0].legend(loc='lower right', fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for key, label, color, ls in methods_to_plot:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        stds = [r[f'{key}_spl_r2_std'] for r in all_results]
        axes[1].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[1].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.1, color=color)
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('R² for SPL')
    axes[1].set_title('SPL Prediction')
    axes[1].legend(loc='lower right', fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'tbcm_to_bm_transfer.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: Head-to-head comparison (BCM-TransRF vs TBCM-TransRF vs Chain) ---
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev_results = json.load(f)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('Source Comparison: BCM vs TBCM vs Chain -> BM',
                      fontsize=14, fontweight='bold')

        for ax, target_key, title in [(axes2[0], 'f0', 'F0'),
                                       (axes2[1], 'spl', 'SPL')]:
            # Target-only baseline
            tgt_vals = [r[f'target_only_{target_key}_r2_mean'] for r in all_results]
            ax.plot(n_samples, tgt_vals, 'o--', label='Target Only', color='blue',
                    linewidth=2, markersize=6)

            # BCM TransRF (from previous results)
            bcm_n = [r['n_samples'] for r in prev_results]
            bcm_vals = [r[f'transrf_{target_key}_r2_mean'] for r in prev_results]
            ax.plot(bcm_n, bcm_vals, 's-', label='BCM TransRF', color='orange',
                    linewidth=2, markersize=6)

            # TBCM TransRF
            tbcm_vals = [r[f'transrf_{target_key}_r2_mean'] for r in all_results]
            ax.plot(n_samples, tbcm_vals, '^-', label='TBCM TransRF', color='red',
                    linewidth=2, markersize=6)

            # Chain TransRF
            chain_vals = [r[f'chain_transrf_{target_key}_r2_mean'] for r in all_results]
            ax.plot(n_samples, chain_vals, 'D-', label='Chain TransRF', color='darkred',
                    linewidth=2, markersize=7)

            ax.set_xlabel('Number of Training Samples')
            ax.set_ylabel(f'R² for {title}')
            ax.set_title(f'{title} Prediction')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_path = os.path.join(figs_dir, 'tbcm_vs_bcm_source_comparison.png')
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig2_path}")
        plt.close()

    # --- Plot 3: Chain TransRF 4-component weight evolution ---
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Chain TransRF: 4-Component Weight Evolution',
                  fontsize=14, fontweight='bold')
    fracs_pct = [r['frac'] * 100 for r in all_results]

    comp_labels = ['Target-only', 'TBCM-Residual', 'BCM-Residual', 'Chain-Augmented']
    comp_colors = ['blue', 'red', 'orange', 'cyan']

    for ax, target, title in [(ax3, 'f0', 'F0'), (ax4, 'spl', 'SPL')]:
        w_key = f'chain_weights_{target}'
        weights_arr = np.array([r[w_key] for r in all_results])
        ax.stackplot(fracs_pct,
                     weights_arr[:, 0], weights_arr[:, 1],
                     weights_arr[:, 2], weights_arr[:, 3],
                     labels=comp_labels, colors=comp_colors, alpha=0.7)
        ax.set_xlabel('Target Data Fraction (%)')
        ax.set_ylabel('Weight')
        ax.set_title(f'Chain Weights for {title}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(figs_dir, 'tbcm_bm_chain_weights.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig3_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("TBCM->BM TRANSFER EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
