"""
Small-Data Regime: Chain Transfer vs Target-Only for BM

Focused comparison at very small sample counts (25–500 training samples)
to see where TBCM+BCM chain transfer helps most vs learning from scratch.
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
import json
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute training sample counts to test
N_TARGETS = [25, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 5

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGET_FEATURES = ['a_CT', 'a_TA', 'PS', 'a_LCA']
TARGETS = ['F0', 'SPL']


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


def get_source_predictions(X, scaler, model):
    """Get predictions from a source model (shared features only)."""
    return model.predict(scaler.transform(X[SHARED_FEATURES]))


def run_experiment(df_full, n_target, rf_tbcm, scaler_X_tbcm,
                   rf_bcm, scaler_X_bcm, random_state=42):
    """Run one experiment at a given absolute sample count.

    Draws n_target samples from df_full, then splits 80/20 train/test
    with a further 80/20 train/val for weight learning.
    """
    if n_target >= len(df_full):
        df = df_full.copy()
    else:
        df = df_full.sample(n=n_target, random_state=random_state)

    X = df[TARGET_FEATURES]
    Y = df[TARGETS]

    if len(df) < 20:
        return None

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=random_state)

    n_train = len(X_train)
    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_val_arr = Y_val.values
    Y_test_arr = Y_test.values

    results = {'n_target': n_target, 'n_train': n_train,
               'n_val': len(X_val), 'n_test': len(X_test)}

    # --- Target Only ---
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    results['target_only_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_target[:, 0])
    results['target_only_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_target[:, 1])

    # --- TBCM predictions (train/val/test) ---
    pred_tbcm_train = get_source_predictions(X_train, scaler_X_tbcm, rf_tbcm)
    pred_tbcm_test = get_source_predictions(X_test, scaler_X_tbcm, rf_tbcm)
    pred_tbcm_val = get_source_predictions(X_val, scaler_X_tbcm, rf_tbcm)

    # --- BCM predictions (train/val/test) ---
    pred_bcm_train = get_source_predictions(X_train, scaler_X_bcm, rf_bcm)
    pred_bcm_test = get_source_predictions(X_test, scaler_X_bcm, rf_bcm)
    pred_bcm_val = get_source_predictions(X_val, scaler_X_bcm, rf_bcm)

    # --- Residual Correction (TBCM) ---
    rf_residuals_tbcm = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_residuals_tbcm.fit(X_train.values, Y_train_arr - pred_tbcm_train)
    pred_residual_tbcm_test = pred_tbcm_test + rf_residuals_tbcm.predict(X_test.values)

    results['residual_tbcm_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual_tbcm_test[:, 0])
    results['residual_tbcm_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual_tbcm_test[:, 1])

    # --- Residual Correction (BCM) ---
    rf_residuals_bcm = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_residuals_bcm.fit(X_train.values, Y_train_arr - pred_bcm_train)
    pred_residual_bcm_test = pred_bcm_test + rf_residuals_bcm.predict(X_test.values)

    results['residual_bcm_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_residual_bcm_test[:, 0])
    results['residual_bcm_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_residual_bcm_test[:, 1])

    # --- Feature Augmentation (TBCM) ---
    X_train_aug_tbcm = np.column_stack([X_train.values, pred_tbcm_train])
    X_test_aug_tbcm = np.column_stack([X_test.values, pred_tbcm_test])

    rf_aug_tbcm = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_aug_tbcm.fit(X_train_aug_tbcm, Y_train_arr)
    pred_aug_tbcm = rf_aug_tbcm.predict(X_test_aug_tbcm)

    results['augmented_tbcm_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_aug_tbcm[:, 0])
    results['augmented_tbcm_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_aug_tbcm[:, 1])

    # --- TransRF (TBCM, 3-component) ---
    X_val_sc = scaler_X.transform(X_val)
    pred_t_val = rf_target.predict(X_val_sc)
    pred_r_tbcm_val = pred_tbcm_val + rf_residuals_tbcm.predict(X_val.values)
    X_val_aug_tbcm = np.column_stack([X_val.values, pred_tbcm_val])
    pred_a_tbcm_val = rf_aug_tbcm.predict(X_val_aug_tbcm)

    weights_tbcm = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i], pred_r_tbcm_val[:, i], pred_a_tbcm_val[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights_tbcm[name] = w

    pred_transrf_tbcm = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i], pred_residual_tbcm_test[:, i], pred_aug_tbcm[:, i]])
        pred_transrf_tbcm[:, i] = stack @ weights_tbcm[name]

    results['transrf_tbcm_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_transrf_tbcm[:, 0])
    results['transrf_tbcm_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_transrf_tbcm[:, 1])

    # --- TransRF (BCM, 3-component) ---
    pred_r_bcm_val = pred_bcm_val + rf_residuals_bcm.predict(X_val.values)
    X_val_aug_bcm = np.column_stack([X_val.values, pred_bcm_val])
    rf_aug_bcm = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_aug_bcm.fit(np.column_stack([X_train.values, pred_bcm_train]), Y_train_arr)
    pred_a_bcm_val = rf_aug_bcm.predict(X_val_aug_bcm)
    pred_aug_bcm_test = rf_aug_bcm.predict(np.column_stack([X_test.values, pred_bcm_test]))

    weights_bcm = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i], pred_r_bcm_val[:, i], pred_a_bcm_val[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights_bcm[name] = w

    pred_transrf_bcm = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i], pred_residual_bcm_test[:, i], pred_aug_bcm_test[:, i]])
        pred_transrf_bcm[:, i] = stack @ weights_bcm[name]

    results['transrf_bcm_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_transrf_bcm[:, 0])
    results['transrf_bcm_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_transrf_bcm[:, 1])

    # --- Chain Augmentation (TBCM + BCM preds) ---
    X_train_chain = np.column_stack([X_train.values, pred_tbcm_train, pred_bcm_train])
    X_test_chain = np.column_stack([X_test.values, pred_tbcm_test, pred_bcm_test])

    rf_chain_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_chain_aug.fit(X_train_chain, Y_train_arr)
    pred_chain_aug = rf_chain_aug.predict(X_test_chain)

    results['chain_aug_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_chain_aug[:, 0])
    results['chain_aug_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_chain_aug[:, 1])

    # --- Chain TransRF (4-component) ---
    X_val_chain = np.column_stack([X_val.values, pred_tbcm_val, pred_bcm_val])
    pred_chain_a_val = rf_chain_aug.predict(X_val_chain)

    weights_chain = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_val[:, i],
            pred_r_tbcm_val[:, i],
            pred_r_bcm_val[:, i],
            pred_chain_a_val[:, i],
        ])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_val_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([0.25, 0.25, 0.25, 0.25])
        weights_chain[name] = w

    pred_chain_transrf = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i],
            pred_residual_tbcm_test[:, i],
            pred_residual_bcm_test[:, i],
            pred_chain_aug[:, i],
        ])
        pred_chain_transrf[:, i] = stack @ weights_chain[name]

    results['chain_transrf_f0_r2'] = r2_score(Y_test_arr[:, 0], pred_chain_transrf[:, 0])
    results['chain_transrf_spl_r2'] = r2_score(Y_test_arr[:, 1], pred_chain_transrf[:, 1])

    return results


def main():
    print("=" * 70)
    print("SMALL-DATA REGIME: Chain Transfer vs Target-Only for BM")
    print(f"Sample counts: {N_TARGETS}")
    print("=" * 70)

    # ---- Load and train TBCM source ----
    print("\nLoading TBCM source data...")
    df_tbcm = pd.read_csv(os.path.join(script_dir, '..', 'TBCM', 'dataset_TBCM.csv'),
                           index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'}).drop(columns=['PL'])
    X_tbcm = df_tbcm[SHARED_FEATURES]
    Y_tbcm = df_tbcm[TARGETS]
    scaler_X_tbcm = StandardScaler()
    X_tbcm_sc = scaler_X_tbcm.fit_transform(X_tbcm)
    print("  Training TBCM source RF (300 trees)...")
    rf_tbcm = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_tbcm.fit(X_tbcm_sc, Y_tbcm)
    print(f"  TBCM samples: {len(df_tbcm)}")

    # ---- Load and train BCM source ----
    print("\nLoading BCM source data...")
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'), index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    X_bcm = df_bcm[SHARED_FEATURES]
    Y_bcm = df_bcm[TARGETS]
    scaler_X_bcm = StandardScaler()
    X_bcm_sc = scaler_X_bcm.fit_transform(X_bcm)
    print("  Training BCM source RF (300 trees)...")
    rf_bcm = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_bcm.fit(X_bcm_sc, Y_bcm)
    print(f"  BCM samples: {len(df_bcm)}")

    # ---- Load BM target ----
    print("\nLoading BM target data...")
    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    df_bm = pd.read_csv(bm_path)
    if 'Ps' in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})
    df_bm = df_bm.dropna(subset=['F0', 'SPL'])
    df_bm_full = df_bm[TARGET_FEATURES + TARGETS].copy()
    print(f"  BM samples: {len(df_bm_full)}")

    # ---- Run experiments ----
    all_results = []

    for n_target in N_TARGETS:
        print(f"\n{'='*50}")
        print(f"N = {n_target} BM samples")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df_bm_full, n_target, rf_tbcm, scaler_X_tbcm,
                                    rf_bcm, scaler_X_bcm, rs)
            if result is not None:
                frac_results.append(result)
                avg_tgt = (result['target_only_f0_r2'] +
                           result['target_only_spl_r2']) / 2
                avg_chain = (result['chain_transrf_f0_r2'] +
                             result['chain_transrf_spl_r2']) / 2
                print(f"  Run {run+1}: n_train={result['n_train']}, "
                      f"Target={avg_tgt:.3f}, ChainTRF={avg_chain:.3f}")

        if frac_results:
            avg = {'n_target': n_target,
                   'n_train': int(np.mean([r['n_train'] for r in frac_results]))}
            for key in ['target_only', 'residual_tbcm', 'residual_bcm',
                        'augmented_tbcm', 'transrf_tbcm', 'transrf_bcm',
                        'chain_aug', 'chain_transrf']:
                for t in ['f0', 'spl']:
                    vals = [r[f'{key}_{t}_r2'] for r in frac_results]
                    avg[f'{key}_{t}_r2_mean'] = float(np.mean(vals))
                    avg[f'{key}_{t}_r2_std'] = float(np.std(vals))
            all_results.append(avg)

    # ==================== RESULTS TABLE ====================
    print("\n\n" + "=" * 70)
    print("SMALL-DATA RESULTS: Average R2 (F0+SPL)/2")
    print("=" * 70)

    methods = ['target_only', 'transrf_bcm', 'transrf_tbcm', 'chain_aug', 'chain_transrf']
    labels = ['Target Only', 'BCM TransRF', 'TBCM TransRF', 'Chain Aug', 'Chain TransRF']

    header = f"{'N':>5} {'Train':>5}"
    for lbl in labels:
        header += f" {lbl:>12}"
    header += "  Best"
    print(header)
    print("-" * (len(header) + 5))

    for r in all_results:
        line = f"{r['n_target']:>5} {r['n_train']:>5}"
        avgs = []
        for key in methods:
            avg = (r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
            avgs.append(avg)
            line += f" {avg:>12.3f}"
        best_idx = np.argmax(avgs)
        line += f"  {labels[best_idx]}"
        print(line)

    # Transfer gain over target
    print("\n\nTRANSFER GAIN over Target-Only:")
    print("-" * 65)
    header = f"{'N':>5}"
    for lbl in labels[1:]:
        header += f" {lbl:>12}"
    print(header)
    for r in all_results:
        tgt = (r['target_only_f0_r2_mean'] + r['target_only_spl_r2_mean']) / 2
        line = f"{r['n_target']:>5}"
        for key in methods[1:]:
            avg = (r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
            line += f" {avg - tgt:>+12.4f}"
        print(line)

    # ==================== SAVE RESULTS ====================
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'small_data_chain_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ==================== PLOT ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    n_train_vals = [r['n_train'] for r in all_results]

    plot_methods = [
        ('target_only', 'Target Only (BM)', 'blue', 'o', '-'),
        ('transrf_bcm', 'BCM TransRF', 'orange', 's', '-'),
        ('transrf_tbcm', 'TBCM TransRF', 'red', '^', '-'),
        ('chain_aug', 'Chain Augmented', 'cyan', 'D', '--'),
        ('chain_transrf', 'Chain TransRF', 'darkred', 'p', '-'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Small-Data Regime: Transfer vs Target-Only for BM',
                 fontsize=14, fontweight='bold')

    # Panel 1: Combined R2
    ax = axes[0]
    for key, label, color, marker, ls in plot_methods:
        vals = [(r[f'{key}_f0_r2_mean'] + r[f'{key}_spl_r2_mean']) / 2
                for r in all_results]
        stds = [np.sqrt(r[f'{key}_f0_r2_std']**2 + r[f'{key}_spl_r2_std']**2) / 2
                for r in all_results]
        ax.plot(n_train_vals, vals, marker=marker, linestyle=ls, label=label,
                color=color, linewidth=2, markersize=7)
        ax.fill_between(n_train_vals,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.1, color=color)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R² (avg F0 + SPL)')
    ax.set_title('Combined R²')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='N=100')

    # Panel 2: F0
    ax = axes[1]
    for key, label, color, marker, ls in plot_methods:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        stds = [r[f'{key}_f0_r2_std'] for r in all_results]
        ax.plot(n_train_vals, vals, marker=marker, linestyle=ls, label=label,
                color=color, linewidth=2, markersize=7)
        ax.fill_between(n_train_vals,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.1, color=color)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R² for F0')
    ax.set_title('F0 Prediction')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Panel 3: SPL
    ax = axes[2]
    for key, label, color, marker, ls in plot_methods:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        stds = [r[f'{key}_spl_r2_std'] for r in all_results]
        ax.plot(n_train_vals, vals, marker=marker, linestyle=ls, label=label,
                color=color, linewidth=2, markersize=7)
        ax.fill_between(n_train_vals,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.1, color=color)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R² for SPL')
    ax.set_title('SPL Prediction')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'small_data_chain_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Transfer gain bar chart at N=100 ---
    r100 = None
    for r in all_results:
        if r['n_target'] == 100:
            r100 = r
            break

    if r100:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        tgt_f0 = r100['target_only_f0_r2_mean']
        tgt_spl = r100['target_only_spl_r2_mean']

        bar_methods = ['transrf_bcm', 'transrf_tbcm', 'chain_aug', 'chain_transrf']
        bar_labels = ['BCM\nTransRF', 'TBCM\nTransRF', 'Chain\nAugmented', 'Chain\nTransRF']
        bar_colors = ['orange', 'red', 'cyan', 'darkred']

        x = np.arange(len(bar_methods))
        width = 0.35

        f0_gains = [r100[f'{m}_f0_r2_mean'] - tgt_f0 for m in bar_methods]
        spl_gains = [r100[f'{m}_spl_r2_mean'] - tgt_spl for m in bar_methods]

        bars1 = ax2.bar(x - width/2, f0_gains, width, label='F0 gain',
                        color=[c for c in bar_colors], alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, spl_gains, width, label='SPL gain',
                        color=[c for c in bar_colors], alpha=0.4, edgecolor='black',
                        hatch='//')

        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.set_xlabel('Transfer Method')
        ax2.set_ylabel('R² Gain over Target-Only')
        ax2.set_title(f'Transfer Gain at N=100 BM Samples '
                      f'(Target-Only: F0={tgt_f0:.3f}, SPL={tgt_spl:.3f})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bar_labels)
        ax2.legend(['F0', 'SPL'], loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig2_path = os.path.join(figs_dir, 'small_data_gain_at_100.png')
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig2_path}")
        plt.close()

    print("\n" + "=" * 70)
    print("SMALL-DATA ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
