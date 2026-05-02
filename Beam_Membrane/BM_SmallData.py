"""
BCM -> Beam-Membrane: Small Data Experiment

Tests transfer learning methods at truly small target data sizes
(10, 20, 30, 50, 75, 100, 200, 500 BM samples).

This is the real use case: you have 54k cheap BCM simulations and
want to know how few expensive BM simulations you need.

Methods compared:
  - Target Only (RF on BM data alone)
  - TransRF Ensemble (learned weights over sub-models)
  - Vanilla AE (train on BCM, fine-tune on BM)
  - Feature Augmentation (BCM predictions as extra inputs)
  - Residual Correction (learn BCM→BM residuals)
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))

N_TARGETS = [10, 20, 30, 50, 75, 100, 200, 500]
N_RUNS = 10  # More runs for stability at small sizes
DEVICE = torch.device('cpu')

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']


# ==================== ADAPTIVE RF COMPLEXITY ====================
def get_model_params(n_samples):
    if n_samples < 20:
        return {'n_estimators': 10, 'max_depth': 2,
                'min_samples_leaf': max(3, n_samples // 5),
                'min_samples_split': max(2, n_samples // 5)}
    elif n_samples < 50:
        return {'n_estimators': 15, 'max_depth': 3,
                'min_samples_leaf': max(5, n_samples // 8),
                'min_samples_split': max(3, n_samples // 10)}
    elif n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(5, n_samples // 10),
                'min_samples_split': max(3, n_samples // 15)}
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    elif n_samples < 500:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}
    else:
        return {'n_estimators': 100, 'max_depth': 10,
                'min_samples_leaf': 2, 'min_samples_split': 2}


# ==================== VANILLA AE ====================
class Encoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, z):
        return self.net(z)

class Predictor(nn.Module):
    def __init__(self, latent_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, output_dim),
        )
    def forward(self, z):
        return self.net(z)


def train_vanilla_ae(X_source, Y_source, X_tgt_train, Y_tgt_train,
                     X_tgt_test, epochs=200, patience=20):
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(predictor.parameters()), lr=1e-3)
    src_dataset = TensorDataset(
        torch.FloatTensor(X_source).to(DEVICE),
        torch.FloatTensor(Y_source).to(DEVICE))
    src_loader = DataLoader(src_dataset, batch_size=256, shuffle=True)
    mse = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        encoder.train(); decoder.train(); predictor.train()
        epoch_loss = 0
        for xb, yb in src_loader:
            z = encoder(xb)
            loss = mse(decoder(z), xb) + mse(predictor(z), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(src_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {
                'enc': {k: v.clone() for k, v in encoder.state_dict().items()},
                'dec': {k: v.clone() for k, v in decoder.state_dict().items()},
                'pred': {k: v.clone() for k, v in predictor.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    encoder.load_state_dict(best_state['enc'])
    predictor.load_state_dict(best_state['pred'])

    # Fine-tune predictor on target
    for p in encoder.parameters():
        p.requires_grad = False
    ft_opt = optim.Adam(predictor.parameters(), lr=5e-4)
    tgt_dataset = TensorDataset(
        torch.FloatTensor(X_tgt_train).to(DEVICE),
        torch.FloatTensor(Y_tgt_train).to(DEVICE))
    tgt_loader = DataLoader(tgt_dataset, batch_size=min(64, len(X_tgt_train)),
                            shuffle=True)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        predictor.train()
        epoch_loss = 0
        for xb, yb in tgt_loader:
            z = encoder(xb)
            loss = mse(predictor(z), yb)
            ft_opt.zero_grad(); loss.backward(); ft_opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(tgt_loader), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_pred = {k: v.clone() for k, v in predictor.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    predictor.load_state_dict(best_pred)
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        preds = predictor(encoder(
            torch.FloatTensor(X_tgt_test).to(DEVICE))).cpu().numpy()
    return preds


# ==================== EXPERIMENT ====================
def run_small_data_experiment(n_target, df_bm, rf_source, scaler_X_source,
                               scaler_X_ae, scaler_Y_ae, X_src_ae, Y_src_ae,
                               random_state=42):
    """Run all methods at a specific number of target samples."""
    rng = np.random.RandomState(random_state)

    # Sample n_target points from BM
    if n_target >= len(df_bm):
        df = df_bm.copy()
    else:
        df = df_bm.sample(n=n_target, random_state=random_state)

    X = df[SHARED_FEATURES]
    Y = df[TARGETS]

    # Split: 80% train, 20% test (but use full BM as test for stability)
    # At very small sizes, use held-out test set from full data
    test_pool = df_bm.drop(df.index)
    if len(test_pool) < 100:
        test_pool = df_bm
    X_test_pool = test_pool.sample(n=min(1000, len(test_pool)),
                                    random_state=random_state)
    Y_test = X_test_pool[TARGETS].values
    X_test = X_test_pool[SHARED_FEATURES]

    # Use all n_target samples for training (test from separate pool)
    X_train = X
    Y_train = Y.values
    n_train = len(X_train)

    # Split training into train/val for TransRF weight learning
    if n_train > 10:
        val_size = max(3, int(n_train * 0.2))
        idx = rng.permutation(n_train)
        train_idx = idx[val_size:]
        val_idx = idx[:val_size]
    else:
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train)  # Reuse train as val when tiny

    X_tr = X_train.iloc[train_idx]
    Y_tr = Y_train[train_idx]
    X_vl = X_train.iloc[val_idx]
    Y_vl = Y_train[val_idx]

    model_params = get_model_params(len(X_tr))

    results = {'n_target': n_target, 'n_train': len(X_tr)}

    # Source predictions
    pred_src_test = rf_source.predict(scaler_X_source.transform(X_test))
    pred_src_train = rf_source.predict(scaler_X_source.transform(X_tr))
    pred_src_val = rf_source.predict(scaler_X_source.transform(X_vl))

    # === SOURCE ONLY ===
    results['source_f0'] = r2_score(Y_test[:, 0], pred_src_test[:, 0])
    results['source_spl'] = r2_score(Y_test[:, 1], pred_src_test[:, 1])

    # === TARGET ONLY ===
    scaler_X = StandardScaler()
    X_tr_sc = scaler_X.fit_transform(X_tr)
    X_test_sc = scaler_X.transform(X_test)
    X_vl_sc = scaler_X.transform(X_vl)

    rf_tgt = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1,
                              **model_params))
    rf_tgt.fit(X_tr_sc, Y_tr)
    pred_tgt_test = rf_tgt.predict(X_test_sc)
    results['target_f0'] = r2_score(Y_test[:, 0], pred_tgt_test[:, 0])
    results['target_spl'] = r2_score(Y_test[:, 1], pred_tgt_test[:, 1])

    # === RESIDUAL CORRECTION ===
    rf_res = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1,
                              **model_params))
    rf_res.fit(X_tr.values, Y_tr - pred_src_train)
    pred_res_test = pred_src_test + rf_res.predict(X_test.values)
    results['residual_f0'] = r2_score(Y_test[:, 0], pred_res_test[:, 0])
    results['residual_spl'] = r2_score(Y_test[:, 1], pred_res_test[:, 1])

    # === FEATURE AUGMENTATION ===
    X_tr_aug = np.column_stack([X_tr.values, pred_src_train])
    X_test_aug = np.column_stack([X_test.values, pred_src_test])
    rf_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1,
                              **model_params))
    rf_aug.fit(X_tr_aug, Y_tr)
    pred_aug_test = rf_aug.predict(X_test_aug)
    results['augmented_f0'] = r2_score(Y_test[:, 0], pred_aug_test[:, 0])
    results['augmented_spl'] = r2_score(Y_test[:, 1], pred_aug_test[:, 1])

    # === TRANSRF ENSEMBLE ===
    pred_tgt_val = rf_tgt.predict(X_vl_sc)
    pred_res_val = pred_src_val + rf_res.predict(X_vl.values)
    X_vl_aug = np.column_stack([X_vl.values, pred_src_val])
    pred_aug_val = rf_aug.predict(X_vl_aug)

    weights = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_tgt_val[:, i], pred_res_val[:, i], pred_aug_val[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_vl[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights[name] = w

    pred_transrf = np.zeros_like(Y_test)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_tgt_test[:, i], pred_res_test[:, i], pred_aug_test[:, i]])
        pred_transrf[:, i] = stack @ weights[name]
    results['transrf_f0'] = r2_score(Y_test[:, 0], pred_transrf[:, 0])
    results['transrf_spl'] = r2_score(Y_test[:, 1], pred_transrf[:, 1])

    # === VANILLA AE ===
    X_tgt_train_sc = scaler_X_ae.transform(X_train.values)
    X_tgt_test_sc = scaler_X_ae.transform(X_test.values)
    Y_tgt_train_sc = scaler_Y_ae.transform(Y_train)

    preds_ae = train_vanilla_ae(
        X_src_ae, Y_src_ae, X_tgt_train_sc, Y_tgt_train_sc, X_tgt_test_sc)
    preds_ae_inv = scaler_Y_ae.inverse_transform(preds_ae)
    results['vanilla_f0'] = r2_score(Y_test[:, 0], preds_ae_inv[:, 0])
    results['vanilla_spl'] = r2_score(Y_test[:, 1], preds_ae_inv[:, 1])

    return results


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("BCM -> BEAM-MEMBRANE: SMALL DATA EXPERIMENT")
    print("=" * 70)

    # Load data
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'),
                          index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})

    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    df_bm = pd.read_csv(bm_path)
    if 'Ps' in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})
    df_bm = df_bm.dropna(subset=['F0', 'SPL'])
    df_bm = df_bm[SHARED_FEATURES + TARGETS].copy()

    print(f"  BCM: {len(df_bcm)} samples (source)")
    print(f"  BM:  {len(df_bm)} samples (target)")

    # Train BCM source model
    print("\nTraining BCM source RF...")
    scaler_X_source = StandardScaler()
    X_src_sc = scaler_X_source.fit_transform(df_bcm[SHARED_FEATURES])
    rf_source = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(X_src_sc, df_bcm[TARGETS])
    print("  Done.")

    # Prepare AE scalers and source data
    scaler_X_ae = StandardScaler()
    scaler_Y_ae = StandardScaler()
    scaler_X_ae.fit(df_bcm[SHARED_FEATURES].values)
    scaler_Y_ae.fit(df_bcm[TARGETS].values)
    # Subsample source for AE efficiency
    rng = np.random.RandomState(42)
    src_idx = rng.choice(len(df_bcm), min(10000, len(df_bcm)), replace=False)
    X_src_ae = scaler_X_ae.transform(df_bcm[SHARED_FEATURES].values[src_idx])
    Y_src_ae = scaler_Y_ae.transform(df_bcm[TARGETS].values[src_idx])

    # Run experiments
    all_results = {}
    for n_tgt in N_TARGETS:
        all_results[n_tgt] = []

    for n_tgt in N_TARGETS:
        print(f"\n{'='*50}")
        print(f"Testing with {n_tgt} BM samples")
        print(f"{'='*50}")

        for run in range(N_RUNS):
            rs = 42 + run
            torch.manual_seed(rs)
            np.random.seed(rs)

            result = run_small_data_experiment(
                n_tgt, df_bm, rf_source, scaler_X_source,
                scaler_X_ae, scaler_Y_ae, X_src_ae, Y_src_ae, rs)
            all_results[n_tgt].append(result)

            tgt_avg = (result['target_f0'] + result['target_spl']) / 2
            trf_avg = (result['transrf_f0'] + result['transrf_spl']) / 2
            ae_avg = (result['vanilla_f0'] + result['vanilla_spl']) / 2
            print(f"  Run {run+1:>2}: Target={tgt_avg:.3f}, "
                  f"TransRF={trf_avg:.3f}, VanillaAE={ae_avg:.3f}")

    # ==================== RESULTS ====================
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    methods = ['source', 'target', 'residual', 'augmented', 'transrf', 'vanilla']
    method_labels = ['Source Only', 'Target Only', 'Residual',
                     'Feature Aug', 'TransRF', 'Vanilla AE']

    # Compute averages
    summary = {}
    for n_tgt in N_TARGETS:
        summary[n_tgt] = {}
        for m in methods:
            f0_vals = [r[f'{m}_f0'] for r in all_results[n_tgt]]
            spl_vals = [r[f'{m}_spl'] for r in all_results[n_tgt]]
            avg_vals = [(f + s) / 2 for f, s in zip(f0_vals, spl_vals)]
            summary[n_tgt][m] = {
                'f0_mean': np.mean(f0_vals), 'f0_std': np.std(f0_vals),
                'spl_mean': np.mean(spl_vals), 'spl_std': np.std(spl_vals),
                'avg_mean': np.mean(avg_vals), 'avg_std': np.std(avg_vals),
            }

    # Print table
    header = f"{'N':>5}"
    for label in method_labels:
        header += f" {label:>12}"
    print(f"\nAverage R2 (F0+SPL)/2:")
    print(header)
    print("-" * len(header))
    for n_tgt in N_TARGETS:
        line = f"{n_tgt:>5}"
        for m in methods:
            val = summary[n_tgt][m]['avg_mean']
            line += f" {val:>12.3f}"
        print(line)

    # Transfer gain
    print(f"\nTransfer Gain over Target-Only:")
    header2 = f"{'N':>5}"
    for label in method_labels[2:]:
        header2 += f" {label:>12}"
    print(header2)
    print("-" * len(header2))
    for n_tgt in N_TARGETS:
        baseline = summary[n_tgt]['target']['avg_mean']
        line = f"{n_tgt:>5}"
        for m in methods[2:]:
            gain = summary[n_tgt][m]['avg_mean'] - baseline
            line += f" {gain:>+12.4f}"
        print(line)

    # Best method per size
    print(f"\nBest Method per Size:")
    print("-" * 50)
    for n_tgt in N_TARGETS:
        best_m = max(methods[1:], key=lambda m: summary[n_tgt][m]['avg_mean'])
        best_label = method_labels[methods.index(best_m)]
        val = summary[n_tgt][best_m]['avg_mean']
        print(f"  {n_tgt:>5} samples: {best_label:<15} (avg R2={val:.3f})")

    # ==================== PLOTS ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')

    plot_methods = [
        ('target', 'Target Only (no transfer)', 'blue', '-', 'o'),
        ('transrf', 'TransRF Ensemble', 'red', '-', 's'),
        ('vanilla', 'Vanilla AE', 'steelblue', '--', 'D'),
        ('augmented', 'Feature Augmentation', 'green', '-', '^'),
        ('residual', 'Residual Correction', 'purple', '-', 'v'),
    ]

    # --- Plot 1: All methods, avg R2 ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('BCM -> Beam-Membrane: Small Data Transfer Learning',
                 fontsize=14, fontweight='bold')

    for target_name, ax_idx, title in [('avg', 0, 'Average R2 (F0+SPL)/2'),
                                        ('f0', 1, 'F0 R2'),
                                        ('spl', 2, 'SPL R2')]:
        ax = axes[ax_idx]
        for key, label, color, ls, marker in plot_methods:
            means = [summary[n][key][f'{target_name}_mean'] for n in N_TARGETS]
            stds = [summary[n][key][f'{target_name}_std'] for n in N_TARGETS]
            ax.plot(N_TARGETS, means, marker=marker, label=label,
                    color=color, linestyle=ls, linewidth=2, markersize=7)
            ax.fill_between(N_TARGETS,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.1, color=color)
        ax.set_xlabel('Number of BM Target Samples')
        ax.set_ylabel('R2')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(N_TARGETS)
        ax.set_xticklabels([str(n) for n in N_TARGETS])

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'bm_small_data_all.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: Head-to-head TransRF vs Vanilla AE vs Target Only ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('BCM -> BM: TransRF vs Vanilla AE (Small Data)',
                  fontsize=14, fontweight='bold')

    key_methods = [
        ('target', 'Target Only', 'blue', 'o'),
        ('transrf', 'TransRF', 'red', 's'),
        ('vanilla', 'Vanilla AE', 'steelblue', 'D'),
    ]

    for key, label, color, marker in key_methods:
        means = [summary[n][key]['avg_mean'] for n in N_TARGETS]
        stds = [summary[n][key]['avg_std'] for n in N_TARGETS]
        ax1.plot(N_TARGETS, means, f'{marker}-', label=label, color=color,
                 linewidth=2.5, markersize=9)
        ax1.fill_between(N_TARGETS,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=color)

    ax1.set_xlabel('Number of BM Target Samples')
    ax1.set_ylabel('Average R2')
    ax1.set_title('Head-to-Head Comparison')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(N_TARGETS)
    ax1.set_xticklabels([str(n) for n in N_TARGETS])

    # Transfer gain plot
    for key, label, color, marker in key_methods[1:]:
        gains = [summary[n][key]['avg_mean'] - summary[n]['target']['avg_mean']
                 for n in N_TARGETS]
        ax2.plot(N_TARGETS, gains, f'{marker}-', label=f'{label} gain',
                 color=color, linewidth=2.5, markersize=9)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Number of BM Target Samples')
    ax2.set_ylabel('R2 Gain over Target-Only')
    ax2.set_title('Transfer Gain')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(N_TARGETS)
    ax2.set_xticklabels([str(n) for n in N_TARGETS])

    # Annotate crossover if it exists
    transrf_gains = [summary[n]['transrf']['avg_mean'] -
                     summary[n]['target']['avg_mean'] for n in N_TARGETS]
    ae_gains = [summary[n]['vanilla']['avg_mean'] -
                summary[n]['target']['avg_mean'] for n in N_TARGETS]

    plt.tight_layout()
    fig2_path = os.path.join(figs_dir, 'bm_small_data_comparison.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("SMALL DATA EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
