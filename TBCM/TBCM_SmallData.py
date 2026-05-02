"""
Quick experiment: BCM -> TBCM transfer with very small target data (<100 samples).

Tests TransRF and Vanilla AE (the two best methods) at tiny sample counts
to see where transfer learning really shines.
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute sample counts to test
N_TARGETS = [10, 20, 30, 50, 75, 100, 200, 500]
N_RUNS = 5
DEVICE = torch.device('cpu')


# ==================== RF HELPERS ====================
def get_model_params(n_samples):
    if n_samples < 100:
        return {
            'n_estimators': 20, 'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20),
        }
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    else:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}


def run_rf_experiment(df_tbcm, n_target, rf_source, scaler_X_source, random_state=42):
    """Run RF transfer methods at a given absolute sample count."""
    df = df_tbcm.sample(n=n_target, random_state=random_state)
    X = df[['a_CT', 'a_TA', 'PS']]
    Y = df[['F0', 'SPL']]

    # With tiny data, use leave-more-for-training split
    test_size = max(0.2, min(0.4, 10 / len(df)))  # at least 10 test samples if possible
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)

    n_train = len(X_train)
    if n_train < 5:
        return None

    model_params = get_model_params(n_train)
    Y_train_arr = Y_train.values
    Y_test_arr = Y_test.values

    # Source-only
    pred_source = rf_source.predict(scaler_X_source.transform(X_test))

    # Target-only
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_target.fit(X_train_sc, Y_train_arr)
    pred_target = rf_target.predict(X_test_sc)

    # Residual correction
    pred_src_train = rf_source.predict(scaler_X_source.transform(X_train))
    rf_res = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_res.fit(X_train.values, Y_train_arr - pred_src_train)
    pred_residual = pred_source + rf_res.predict(X_test.values)

    # Feature augmentation
    X_train_aug = np.column_stack([X_train.values, pred_src_train])
    X_test_aug = np.column_stack([X_test.values, pred_source])
    rf_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params))
    rf_aug.fit(X_train_aug, Y_train_arr)
    pred_augmented = rf_aug.predict(X_test_aug)

    # TransRF (use all training data for weight learning since no val split at this size)
    pred_t_tr = rf_target.predict(X_train_sc)
    pred_r_tr = pred_src_train + rf_res.predict(X_train.values)
    pred_a_tr = rf_aug.predict(X_train_aug)

    weights = {}
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_t_tr[:, i], pred_r_tr[:, i], pred_a_tr[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_train_arr[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        weights[name] = w

    pred_transrf = np.zeros_like(Y_test_arr)
    for i, name in enumerate(['F0', 'SPL']):
        stack = np.column_stack([
            pred_target[:, i], pred_residual[:, i], pred_augmented[:, i]])
        pred_transrf[:, i] = stack @ weights[name]

    return {
        'n_target': n_target, 'n_train': n_train, 'n_test': len(X_test),
        'source_f0': r2_score(Y_test_arr[:, 0], pred_source[:, 0]),
        'source_spl': r2_score(Y_test_arr[:, 1], pred_source[:, 1]),
        'target_f0': r2_score(Y_test_arr[:, 0], pred_target[:, 0]),
        'target_spl': r2_score(Y_test_arr[:, 1], pred_target[:, 1]),
        'residual_f0': r2_score(Y_test_arr[:, 0], pred_residual[:, 0]),
        'residual_spl': r2_score(Y_test_arr[:, 1], pred_residual[:, 1]),
        'augmented_f0': r2_score(Y_test_arr[:, 0], pred_augmented[:, 0]),
        'augmented_spl': r2_score(Y_test_arr[:, 1], pred_augmented[:, 1]),
        'transrf_f0': r2_score(Y_test_arr[:, 0], pred_transrf[:, 0]),
        'transrf_spl': r2_score(Y_test_arr[:, 1], pred_transrf[:, 1]),
        'weights_f0': weights['F0'].tolist(),
        'weights_spl': weights['SPL'].tolist(),
    }


# ==================== AE HELPERS ====================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, z):
        return self.net(z)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
    def forward(self, z):
        return self.net(z)


def run_vanilla_ae(X_src, Y_src, X_tgt_train, Y_tgt_train, X_tgt_test, Y_tgt_test):
    """Vanilla AE: train on source, fine-tune predictor on target."""
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)
    mse = nn.MSELoss()

    # Phase 1: Train on source
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                     list(predictor.parameters()), lr=1e-3)
    src_ds = TensorDataset(torch.FloatTensor(X_src), torch.FloatTensor(Y_src))
    src_loader = DataLoader(src_ds, batch_size=256, shuffle=True)

    best_loss = float('inf')
    patience = 0
    for epoch in range(200):
        encoder.train(); decoder.train(); predictor.train()
        eloss = 0
        for xb, yb in src_loader:
            z = encoder(xb)
            loss = mse(decoder(z), xb) + mse(predictor(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item()
        eloss /= len(src_loader)
        if eloss < best_loss:
            best_loss = eloss; patience = 0
            best = {k: {a: b.clone() for a, b in v.state_dict().items()}
                    for k, v in [('e', encoder), ('d', decoder), ('p', predictor)]}
        else:
            patience += 1
            if patience >= 20: break

    encoder.load_state_dict(best['e'])
    predictor.load_state_dict(best['p'])

    # Phase 2: Fine-tune predictor on target
    for p in encoder.parameters(): p.requires_grad = False
    ft_opt = optim.Adam(predictor.parameters(), lr=5e-4)
    tgt_ds = TensorDataset(torch.FloatTensor(X_tgt_train), torch.FloatTensor(Y_tgt_train))
    tgt_loader = DataLoader(tgt_ds, batch_size=min(64, len(X_tgt_train)), shuffle=True)

    best_loss = float('inf'); patience = 0
    for epoch in range(300):  # more epochs for tiny data
        predictor.train()
        eloss = 0
        for xb, yb in tgt_loader:
            z = encoder(xb)
            loss = mse(predictor(z), yb)
            ft_opt.zero_grad(); loss.backward(); ft_opt.step()
            eloss += loss.item()
        eloss /= max(len(tgt_loader), 1)
        if eloss < best_loss:
            best_loss = eloss; patience = 0
            best_p = {a: b.clone() for a, b in predictor.state_dict().items()}
        else:
            patience += 1
            if patience >= 30: break

    predictor.load_state_dict(best_p)

    encoder.eval(); predictor.eval()
    with torch.no_grad():
        preds = predictor(encoder(torch.FloatTensor(X_tgt_test))).numpy()
    return preds


def run_ae_experiment(df_bcm, df_tbcm, n_target, scaler_X, scaler_Y, random_state=42):
    """Run Vanilla AE at a given absolute sample count."""
    df = df_tbcm.sample(n=n_target, random_state=random_state)
    X_tgt = df[['a_CT', 'a_TA', 'PS']].values
    Y_tgt = df[['F0', 'SPL']].values

    test_size = max(0.2, min(0.4, 10 / len(df)))
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_tgt, Y_tgt, test_size=test_size, random_state=random_state)

    if len(X_train) < 5:
        return None

    X_train_sc = scaler_X.transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    Y_train_sc = scaler_Y.transform(Y_train)

    # Subsample source
    rng = np.random.RandomState(random_state)
    X_src = scaler_X.transform(df_bcm[['a_CT', 'a_TA', 'PS']].values)
    Y_src = scaler_Y.transform(df_bcm[['F0', 'SPL']].values)
    idx = rng.choice(len(X_src), min(10000, len(X_src)), replace=False)

    preds_sc = run_vanilla_ae(X_src[idx], Y_src[idx], X_train_sc, Y_train_sc, X_test_sc, None)
    preds = scaler_Y.inverse_transform(preds_sc)

    return {
        'n_target': n_target,
        'vanilla_f0': float(r2_score(Y_test[:, 0], preds[:, 0])),
        'vanilla_spl': float(r2_score(Y_test[:, 1], preds[:, 1])),
    }


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("SMALL-DATA EXPERIMENT: BCM -> TBCM")
    print(f"Target sizes: {N_TARGETS}")
    print("=" * 70)

    # Load and prep data
    df_bcm = pd.read_csv(os.path.join(script_dir, 'dataset_BCM.csv'), index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    df_tbcm = pd.read_csv(os.path.join(script_dir, 'dataset_TBCM.csv'), index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'}).drop(columns=['PL'])

    # Train source RF
    print("\nTraining BCM source model...")
    scaler_X_source = StandardScaler()
    X_src_sc = scaler_X_source.fit_transform(df_bcm[['a_CT', 'a_TA', 'PS']])
    rf_source = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(X_src_sc, df_bcm[['F0', 'SPL']])
    print("  Done.")

    # AE scalers
    scaler_X_ae = StandardScaler()
    scaler_Y_ae = StandardScaler()
    scaler_X_ae.fit(df_bcm[['a_CT', 'a_TA', 'PS']].values)
    scaler_Y_ae.fit(df_bcm[['F0', 'SPL']].values)

    # Run experiments
    rf_all = []
    ae_all = []

    for n in N_TARGETS:
        print(f"\n{'='*50}")
        print(f"N = {n} target samples")
        print(f"{'='*50}")

        rf_runs = []
        ae_runs = []
        for run in range(N_RUNS):
            rs = 42 + run

            # RF
            r = run_rf_experiment(df_tbcm, n, rf_source, scaler_X_source, rs)
            if r:
                rf_runs.append(r)

            # AE
            torch.manual_seed(rs)
            a = run_ae_experiment(df_bcm, df_tbcm, n, scaler_X_ae, scaler_Y_ae, rs)
            if a:
                ae_runs.append(a)

        if rf_runs:
            avg_rf = {'n': n, 'n_train': int(np.mean([r['n_train'] for r in rf_runs]))}
            for key in ['source', 'target', 'residual', 'augmented', 'transrf']:
                for t in ['f0', 'spl']:
                    vals = [r[f'{key}_{t}'] for r in rf_runs]
                    avg_rf[f'{key}_{t}_mean'] = np.mean(vals)
                    avg_rf[f'{key}_{t}_std'] = np.std(vals)
            avg_rf['weights_f0'] = np.mean([r['weights_f0'] for r in rf_runs], axis=0).tolist()
            avg_rf['weights_spl'] = np.mean([r['weights_spl'] for r in rf_runs], axis=0).tolist()
            rf_all.append(avg_rf)

            tgt = (avg_rf['target_f0_mean'] + avg_rf['target_spl_mean']) / 2
            trf = (avg_rf['transrf_f0_mean'] + avg_rf['transrf_spl_mean']) / 2
            aug = (avg_rf['augmented_f0_mean'] + avg_rf['augmented_spl_mean']) / 2
            res = (avg_rf['residual_f0_mean'] + avg_rf['residual_spl_mean']) / 2
            print(f"  RF  - Target={tgt:.3f}, Residual={res:.3f}, "
                  f"Augmented={aug:.3f}, TransRF={trf:.3f}")

        if ae_runs:
            avg_ae = {'n': n}
            for t in ['f0', 'spl']:
                vals = [r[f'vanilla_{t}'] for r in ae_runs]
                avg_ae[f'vanilla_{t}_mean'] = np.mean(vals)
                avg_ae[f'vanilla_{t}_std'] = np.std(vals)
            ae_all.append(avg_ae)
            van = (avg_ae['vanilla_f0_mean'] + avg_ae['vanilla_spl_mean']) / 2
            print(f"  AE  - Vanilla={van:.3f}")

    # ==================== SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("SMALL-DATA RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'N':>5} {'Train':>5} {'Source':>8} {'Target':>8} {'Residual':>8} "
          f"{'Augment':>8} {'TransRF':>8} {'VanilAE':>8} {'Best':>10} {'Gain':>8}")
    print("-" * 90)

    for i, rf in enumerate(rf_all):
        src = (rf['source_f0_mean'] + rf['source_spl_mean']) / 2
        tgt = (rf['target_f0_mean'] + rf['target_spl_mean']) / 2
        res = (rf['residual_f0_mean'] + rf['residual_spl_mean']) / 2
        aug = (rf['augmented_f0_mean'] + rf['augmented_spl_mean']) / 2
        trf = (rf['transrf_f0_mean'] + rf['transrf_spl_mean']) / 2

        van = None
        if i < len(ae_all):
            van = (ae_all[i]['vanilla_f0_mean'] + ae_all[i]['vanilla_spl_mean']) / 2

        methods = {'Source': src, 'Target': tgt, 'Residual': res,
                   'Augment': aug, 'TransRF': trf}
        if van is not None:
            methods['VanilAE'] = van

        best_name = max(methods, key=methods.get)
        best_val = methods[best_name]
        gain = best_val - tgt  # gain over target-only

        van_str = f"{van:>8.3f}" if van is not None else f"{'N/A':>8}"
        print(f"{rf['n']:>5} {rf['n_train']:>5} {src:>8.3f} {tgt:>8.3f} {res:>8.3f} "
              f"{aug:>8.3f} {trf:>8.3f} {van_str} {best_name:>10} {gain:>+8.3f}")

    # ==================== PLOT ====================
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Small-Data Transfer: BCM -> TBCM', fontsize=14, fontweight='bold')

    ns = [r['n'] for r in rf_all]

    plot_methods = [
        ('target', 'Target Only', 'blue', '-'),
        ('residual', 'Residual Correction', 'purple', '-'),
        ('augmented', 'Feature Augmentation', 'green', '-'),
        ('transrf', 'TransRF', 'red', '-'),
    ]

    for key, label, color, ls in plot_methods:
        for ax_idx, t in enumerate(['f0', 'spl']):
            vals = [r[f'{key}_{t}_mean'] for r in rf_all]
            stds = [r[f'{key}_{t}_std'] for r in rf_all]
            axes[ax_idx].plot(ns, vals, 'o-', label=label, color=color,
                              linestyle=ls, linewidth=2, markersize=6)
            axes[ax_idx].fill_between(ns,
                                      [v-s for v, s in zip(vals, stds)],
                                      [v+s for v, s in zip(vals, stds)],
                                      alpha=0.1, color=color)

    # Add Vanilla AE
    if ae_all:
        ae_ns = [r['n'] for r in ae_all]
        for ax_idx, t in enumerate(['f0', 'spl']):
            vals = [r[f'vanilla_{t}_mean'] for r in ae_all]
            stds = [r[f'vanilla_{t}_std'] for r in ae_all]
            axes[ax_idx].plot(ae_ns, vals, 's--', label='Vanilla AE', color='steelblue',
                              linewidth=2, markersize=6)
            axes[ax_idx].fill_between(ae_ns,
                                      [v-s for v, s in zip(vals, stds)],
                                      [v+s for v, s in zip(vals, stds)],
                                      alpha=0.1, color='steelblue')

    for ax_idx, title in enumerate(['F0', 'SPL']):
        axes[ax_idx].set_xlabel('Number of Target Samples')
        axes[ax_idx].set_ylabel(f'R2 for {title}')
        axes[ax_idx].set_title(f'{title} Prediction')
        axes[ax_idx].legend(loc='lower right', fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xscale('log')
        axes[ax_idx].set_xticks(ns)
        axes[ax_idx].set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'figs', 'tbcm_small_data.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("SMALL-DATA EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
