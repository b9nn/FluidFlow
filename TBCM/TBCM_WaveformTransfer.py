"""
Waveform Feature Experiment: Do waveform features improve TBCM prediction?

Four approaches compared (all RF, TBCM-only, using the 5,435 enriched samples):
  A) Baseline:      inputs=(a_CT, a_TA, PS) -> targets=(F0, SPL)
  B) Extra Inputs:  inputs=(a_CT, a_TA, PS, waveform_feats) -> targets=(F0, SPL)
  C) Extra Targets: inputs=(a_CT, a_TA, PS) -> targets=(F0, SPL, waveform_feats)
  D) Both:          inputs=(a_CT, a_TA, PS, waveform_feats) -> targets=(F0, SPL, waveform_feats)

For C and D, we predict waveform features alongside F0/SPL (multi-task learning)
but only evaluate on F0/SPL. The idea is multi-task forces a richer representation.

Also tests BCM->TBCM transfer with waveform features:
  E) TransRF + Extra Targets: BCM source predicts (F0, SPL), TBCM target predicts
     (F0, SPL, waveform_feats), TransRF ensemble on F0/SPL.
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
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 5


def get_model_params(n_samples):
    if n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(10, n_samples // 10),
                'min_samples_split': max(5, n_samples // 20)}
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


def run_experiment(df, frac, base_input_cols, wf_input_cols, wf_target_cols,
                   rf_source=None, scaler_X_source=None, random_state=42):
    """Run all approaches at a given data fraction."""
    if frac < 1.0:
        df_sub = df.sample(frac=frac, random_state=random_state)
    else:
        df_sub = df.copy()

    if len(df_sub) < 20:
        return None

    X_base = df_sub[base_input_cols].values
    X_wf = df_sub[base_input_cols + wf_input_cols].values
    Y_base = df_sub[['F0', 'SPL']].values
    Y_multi = df_sub[['F0', 'SPL'] + wf_target_cols].values

    test_size = 0.2
    rs = random_state

    # Split
    (X_base_tr, X_base_te, X_wf_tr, X_wf_te,
     Y_base_tr, Y_base_te, Y_multi_tr, Y_multi_te) = train_test_split(
        X_base, X_wf, Y_base, Y_multi, test_size=test_size, random_state=rs)

    n_train = len(X_base_tr)
    params = get_model_params(n_train)

    results = {'frac': frac, 'n_train': n_train, 'n_test': len(X_base_te)}

    # --- A) Baseline: base inputs -> F0, SPL ---
    sc_a = StandardScaler()
    X_a_tr = sc_a.fit_transform(X_base_tr)
    X_a_te = sc_a.transform(X_base_te)
    rf_a = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_a.fit(X_a_tr, Y_base_tr)
    pred_a = rf_a.predict(X_a_te)
    results['baseline_f0'] = r2_score(Y_base_te[:, 0], pred_a[:, 0])
    results['baseline_spl'] = r2_score(Y_base_te[:, 1], pred_a[:, 1])

    # --- B) Extra Inputs: base + waveform inputs -> F0, SPL ---
    sc_b = StandardScaler()
    X_b_tr = sc_b.fit_transform(X_wf_tr)
    X_b_te = sc_b.transform(X_wf_te)
    rf_b = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_b.fit(X_b_tr, Y_base_tr)
    pred_b = rf_b.predict(X_b_te)
    results['extra_inputs_f0'] = r2_score(Y_base_te[:, 0], pred_b[:, 0])
    results['extra_inputs_spl'] = r2_score(Y_base_te[:, 1], pred_b[:, 1])

    # --- C) Extra Targets: base inputs -> F0, SPL, waveform targets ---
    rf_c = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_c.fit(X_a_tr, Y_multi_tr)
    pred_c = rf_c.predict(X_a_te)
    results['extra_targets_f0'] = r2_score(Y_base_te[:, 0], pred_c[:, 0])
    results['extra_targets_spl'] = r2_score(Y_base_te[:, 1], pred_c[:, 1])

    # --- D) Both: base + waveform inputs -> F0, SPL, waveform targets ---
    rf_d = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_d.fit(X_b_tr, Y_multi_tr)
    pred_d = rf_d.predict(X_b_te)
    results['both_f0'] = r2_score(Y_base_te[:, 0], pred_d[:, 0])
    results['both_spl'] = r2_score(Y_base_te[:, 1], pred_d[:, 1])

    # --- E) TransRF + Extra Targets (if source model provided) ---
    if rf_source is not None and scaler_X_source is not None:
        pred_source = rf_source.predict(scaler_X_source.transform(X_base_te))
        pred_source_tr = rf_source.predict(scaler_X_source.transform(X_base_tr))

        # Target-only (same as baseline A)
        pred_target = pred_a

        # Residual: learn residual with multi-target
        rf_res = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
        # Residuals for F0/SPL, plus predict waveform features from scratch
        residuals_tr = np.column_stack([
            Y_base_tr - pred_source_tr,
            Y_multi_tr[:, 2:]  # waveform targets
        ])
        rf_res.fit(X_base_tr, residuals_tr)
        res_pred = rf_res.predict(X_base_te)
        pred_residual = pred_source + res_pred[:, :2]  # only F0, SPL

        # Feature augmentation with multi-target
        X_aug_tr = np.column_stack([X_base_tr, pred_source_tr])
        X_aug_te = np.column_stack([X_base_te, pred_source])
        rf_aug = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
        rf_aug.fit(X_aug_tr, Y_multi_tr)
        pred_augmented = rf_aug.predict(X_aug_te)[:, :2]  # only F0, SPL

        # Split train into train/val for weight learning
        X_tr2, X_val, Y_tr2, Y_val = train_test_split(
            X_base_tr, Y_base_tr, test_size=0.2, random_state=rs)
        params2 = get_model_params(len(X_tr2))

        sc_v = StandardScaler()
        X_tr2_sc = sc_v.fit_transform(X_tr2)
        X_val_sc = sc_v.transform(X_val)

        rf_t2 = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params2))
        rf_t2.fit(X_tr2_sc, Y_tr2)
        pred_t_val = rf_t2.predict(X_val_sc)

        pred_src_tr2 = rf_source.predict(scaler_X_source.transform(X_tr2))
        pred_src_val = rf_source.predict(scaler_X_source.transform(X_val))

        rf_r2 = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params2))
        rf_r2.fit(X_tr2, Y_tr2 - pred_src_tr2)
        pred_r_val = pred_src_val + rf_r2.predict(X_val)

        X_aug_tr2 = np.column_stack([X_tr2, pred_src_tr2])
        X_aug_val = np.column_stack([X_val, pred_src_val])
        rf_a2 = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params2))
        rf_a2.fit(X_aug_tr2, Y_tr2)
        pred_a_val = rf_a2.predict(X_aug_val)

        # Learn weights
        weights = {}
        for i, name in enumerate(['F0', 'SPL']):
            stack = np.column_stack([
                pred_t_val[:, i], pred_r_val[:, i], pred_a_val[:, i]])
            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack, Y_val[:, i])
            w = reg.coef_
            w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
            weights[name] = w

        pred_transrf = np.zeros_like(Y_base_te)
        for i, name in enumerate(['F0', 'SPL']):
            stack = np.column_stack([
                pred_target[:, i], pred_residual[:, i], pred_augmented[:, i]])
            pred_transrf[:, i] = stack @ weights[name]

        results['transrf_mt_f0'] = r2_score(Y_base_te[:, 0], pred_transrf[:, 0])
        results['transrf_mt_spl'] = r2_score(Y_base_te[:, 1], pred_transrf[:, 1])

        # Also run plain TransRF (no multi-target) for comparison
        rf_res_plain = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
        rf_res_plain.fit(X_base_tr, Y_base_tr - pred_source_tr)
        pred_res_plain = pred_source + rf_res_plain.predict(X_base_te)

        rf_aug_plain = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
        rf_aug_plain.fit(X_aug_tr, Y_base_tr)
        pred_aug_plain = rf_aug_plain.predict(X_aug_te)

        # Reuse val split for plain weights
        rf_r2p = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params2))
        rf_r2p.fit(X_tr2, Y_tr2 - pred_src_tr2)
        pred_rp_val = pred_src_val + rf_r2p.predict(X_val)

        rf_a2p = MultiOutputRegressor(
            RandomForestRegressor(random_state=rs, n_jobs=-1, **params2))
        rf_a2p.fit(X_aug_tr2, Y_tr2)
        pred_ap_val = rf_a2p.predict(X_aug_val)

        weights_plain = {}
        for i, name in enumerate(['F0', 'SPL']):
            stack = np.column_stack([
                pred_t_val[:, i], pred_rp_val[:, i], pred_ap_val[:, i]])
            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(stack, Y_val[:, i])
            w = reg.coef_
            w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
            weights_plain[name] = w

        pred_transrf_plain = np.zeros_like(Y_base_te)
        for i, name in enumerate(['F0', 'SPL']):
            stack = np.column_stack([
                pred_target[:, i], pred_res_plain[:, i], pred_aug_plain[:, i]])
            pred_transrf_plain[:, i] = stack @ weights_plain[name]

        results['transrf_plain_f0'] = r2_score(Y_base_te[:, 0], pred_transrf_plain[:, 0])
        results['transrf_plain_spl'] = r2_score(Y_base_te[:, 1], pred_transrf_plain[:, 1])

    return results


def main():
    print("=" * 70)
    print("WAVEFORM FEATURE EXPERIMENT: TBCM")
    print("=" * 70)

    # Load enriched dataset
    df = pd.read_csv(os.path.join(script_dir, 'dataset_TBCM_enriched.csv'))
    print(f"Enriched dataset: {len(df)} samples, {len(df.columns)} columns")

    # Drop rows with NaNs in waveform features
    n_before = len(df)
    df = df.dropna()
    print(f"After dropping NaNs: {len(df)} samples ({n_before - len(df)} dropped)")

    # Define column groups
    base_input_cols = ['a_CT', 'a_TA', 'Ps']

    # Select waveform features for inputs (those that correlate with F0 or SPL)
    all_wf_cols = [c for c in df.columns if c not in
                   ['a_CT', 'a_TA', 'PL', 'Ps', 'F0', 'SPL']]
    print(f"Available waveform features: {len(all_wf_cols)}")

    # Use all waveform features as potential inputs
    wf_input_cols = all_wf_cols

    # For extra targets, select the "independent" ones (low corr with F0/SPL)
    # These add new information the model wouldn't get from F0/SPL alone
    independent_features = []
    for c in all_wf_cols:
        corr_f0 = abs(np.corrcoef(df[c], df['F0'])[0, 1])
        corr_spl = abs(np.corrcoef(df[c], df['SPL'])[0, 1])
        if corr_f0 < 0.5 and corr_spl < 0.5:
            independent_features.append(c)

    wf_target_cols = independent_features
    print(f"Independent waveform features (extra targets): {len(wf_target_cols)}")
    for c in wf_target_cols:
        print(f"  {c}")

    # Train BCM source model for transfer comparison
    print("\nTraining BCM source model...")
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'), index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    # Note: enriched dataset has 'Ps' not 'PS', and base_input_cols uses 'Ps'
    # BCM needs to match
    scaler_X_source = StandardScaler()
    X_src = scaler_X_source.fit_transform(df_bcm[['a_CT', 'a_TA', 'PS']])
    rf_source = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(X_src, df_bcm[['F0', 'SPL']])
    print("  Done.")

    # Rename Ps in enriched df to match
    # Actually base_input_cols already uses 'Ps' which matches the enriched CSV
    # But the source scaler was fit on BCM 'PS'. Need consistent naming.
    # The source model expects [a_CT, a_TA, PS] scaled. Let's just make sure
    # we pass the right data.

    # Run experiments
    all_results = []

    for frac in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of enriched data ({int(len(df)*frac)} samples)")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            result = run_experiment(df, frac, base_input_cols, wf_input_cols,
                                    wf_target_cols, rf_source, scaler_X_source, rs)
            if result:
                frac_results.append(result)

        if frac_results:
            avg = {'frac': frac,
                   'n_train': int(np.mean([r['n_train'] for r in frac_results]))}
            for key in ['baseline', 'extra_inputs', 'extra_targets', 'both',
                        'transrf_plain', 'transrf_mt']:
                for t in ['f0', 'spl']:
                    k = f'{key}_{t}'
                    vals = [r[k] for r in frac_results if k in r]
                    if vals:
                        avg[f'{k}_mean'] = float(np.mean(vals))
                        avg[f'{k}_std'] = float(np.std(vals))
            all_results.append(avg)

            # Print summary for this fraction
            bl = (avg.get('baseline_f0_mean', 0) + avg.get('baseline_spl_mean', 0)) / 2
            ei = (avg.get('extra_inputs_f0_mean', 0) + avg.get('extra_inputs_spl_mean', 0)) / 2
            et = (avg.get('extra_targets_f0_mean', 0) + avg.get('extra_targets_spl_mean', 0)) / 2
            bo = (avg.get('both_f0_mean', 0) + avg.get('both_spl_mean', 0)) / 2
            tp = (avg.get('transrf_plain_f0_mean', 0) + avg.get('transrf_plain_spl_mean', 0)) / 2
            tm = (avg.get('transrf_mt_f0_mean', 0) + avg.get('transrf_mt_spl_mean', 0)) / 2
            print(f"  Baseline={bl:.3f}, +Inputs={ei:.3f}, +Targets={et:.3f}, "
                  f"Both={bo:.3f}")
            print(f"  TransRF={tp:.3f}, TransRF+MultiTask={tm:.3f}")

    # ==================== RESULTS ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Frac':>5} {'N':>5} {'Baseline':>9} {'+Inputs':>9} {'+Targets':>9} "
          f"{'Both':>9} {'TransRF':>9} {'TRF+MT':>9}")
    print("-" * 65)
    for r in all_results:
        bl = (r.get('baseline_f0_mean', 0) + r.get('baseline_spl_mean', 0)) / 2
        ei = (r.get('extra_inputs_f0_mean', 0) + r.get('extra_inputs_spl_mean', 0)) / 2
        et = (r.get('extra_targets_f0_mean', 0) + r.get('extra_targets_spl_mean', 0)) / 2
        bo = (r.get('both_f0_mean', 0) + r.get('both_spl_mean', 0)) / 2
        tp = (r.get('transrf_plain_f0_mean', 0) + r.get('transrf_plain_spl_mean', 0)) / 2
        tm = (r.get('transrf_mt_f0_mean', 0) + r.get('transrf_mt_spl_mean', 0)) / 2
        print(f"{r['frac']*100:>4.0f}% {r['n_train']:>5} {bl:>9.3f} {ei:>9.3f} "
              f"{et:>9.3f} {bo:>9.3f} {tp:>9.3f} {tm:>9.3f}")

    # Gain over baseline
    print(f"\nGAIN OVER BASELINE:")
    print(f"{'Frac':>5} {'+Inputs':>9} {'+Targets':>9} {'Both':>9} {'TransRF':>9} {'TRF+MT':>9}")
    print("-" * 55)
    for r in all_results:
        bl = (r.get('baseline_f0_mean', 0) + r.get('baseline_spl_mean', 0)) / 2
        ei = (r.get('extra_inputs_f0_mean', 0) + r.get('extra_inputs_spl_mean', 0)) / 2 - bl
        et = (r.get('extra_targets_f0_mean', 0) + r.get('extra_targets_spl_mean', 0)) / 2 - bl
        bo = (r.get('both_f0_mean', 0) + r.get('both_spl_mean', 0)) / 2 - bl
        tp = (r.get('transrf_plain_f0_mean', 0) + r.get('transrf_plain_spl_mean', 0)) / 2 - bl
        tm = (r.get('transrf_mt_f0_mean', 0) + r.get('transrf_mt_spl_mean', 0)) / 2 - bl
        print(f"{r['frac']*100:>4.0f}% {ei:>+9.4f} {et:>+9.4f} {bo:>+9.4f} {tp:>+9.4f} {tm:>+9.4f}")

    # F0 vs SPL breakdown
    print(f"\nF0 R2 BREAKDOWN:")
    print(f"{'Frac':>5} {'Baseline':>9} {'+Inputs':>9} {'+Targets':>9} {'Both':>9}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['frac']*100:>4.0f}% {r.get('baseline_f0_mean',0):>9.3f} "
              f"{r.get('extra_inputs_f0_mean',0):>9.3f} "
              f"{r.get('extra_targets_f0_mean',0):>9.3f} "
              f"{r.get('both_f0_mean',0):>9.3f}")

    print(f"\nSPL R2 BREAKDOWN:")
    print(f"{'Frac':>5} {'Baseline':>9} {'+Inputs':>9} {'+Targets':>9} {'Both':>9}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['frac']*100:>4.0f}% {r.get('baseline_spl_mean',0):>9.3f} "
              f"{r.get('extra_inputs_spl_mean',0):>9.3f} "
              f"{r.get('extra_targets_spl_mean',0):>9.3f} "
              f"{r.get('both_spl_mean',0):>9.3f}")

    # ==================== PLOT ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')
    ns = [r['n_train'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Waveform Features: Do They Improve TBCM Prediction?',
                 fontsize=14, fontweight='bold')

    methods = [
        ('baseline', 'Baseline (3 inputs)', 'blue', '-', 'o'),
        ('extra_inputs', '+ Waveform Inputs', 'green', '-', 's'),
        ('extra_targets', '+ Waveform Targets', 'purple', '-', '^'),
        ('both', '+ Both', 'red', '-', 'D'),
    ]

    # F0
    ax = axes[0]
    for key, label, color, ls, marker in methods:
        vals = [r.get(f'{key}_f0_mean', 0) for r in all_results]
        stds = [r.get(f'{key}_f0_std', 0) for r in all_results]
        ax.plot(ns, vals, marker=marker, linestyle=ls, color=color,
                linewidth=2, markersize=6, label=label)
        ax.fill_between(ns, [v-s for v,s in zip(vals,stds)],
                        [v+s for v,s in zip(vals,stds)], alpha=0.1, color=color)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R2')
    ax.set_title('F0 Prediction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # SPL
    ax = axes[1]
    for key, label, color, ls, marker in methods:
        vals = [r.get(f'{key}_spl_mean', 0) for r in all_results]
        stds = [r.get(f'{key}_spl_std', 0) for r in all_results]
        ax.plot(ns, vals, marker=marker, linestyle=ls, color=color,
                linewidth=2, markersize=6, label=label)
        ax.fill_between(ns, [v-s for v,s in zip(vals,stds)],
                        [v+s for v,s in zip(vals,stds)], alpha=0.1, color=color)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('R2')
    ax.set_title('SPL Prediction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # TransRF comparison
    ax = axes[2]
    transfer_methods = [
        ('baseline', 'Target Only', 'blue', '-', 'o'),
        ('transrf_plain', 'TransRF', 'red', '-', 'D'),
        ('transrf_mt', 'TransRF + Multi-Task', 'darkred', '--', 's'),
    ]
    for key, label, color, ls, marker in transfer_methods:
        vals_f0 = [r.get(f'{key}_f0_mean', 0) for r in all_results]
        vals_spl = [r.get(f'{key}_spl_mean', 0) for r in all_results]
        vals = [(f+s)/2 for f,s in zip(vals_f0, vals_spl)]
        ax.plot(ns, vals, marker=marker, linestyle=ls, color=color,
                linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Average R2')
    ax.set_title('Transfer Learning Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'tbcm_waveform_experiment.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("WAVEFORM EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
