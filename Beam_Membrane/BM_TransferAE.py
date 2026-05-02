"""
Autoencoder-Based Domain Adaptation: BCM -> Beam-Membrane (BM)

Three approaches compared:
  A) Vanilla Autoencoder + Fine-Tuning
  B) MMD-Regularized Autoencoder (domain-invariant latent space)
  C) Domain-Adversarial Autoencoder (DAAE with gradient reversal)

Architecture:
  Encoder:       3 -> 64 -> 32 -> 16 (latent z)
  Decoder:       16 -> 32 -> 64 -> 3
  Predictor:     16 -> 32 -> 16 -> 2
  Discriminator: 16 -> 16 -> 8 -> 1

Shared features: a_CT, a_TA, Ps -> F0, SPL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import json
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 3
DEVICE = torch.device('cpu')

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']

np.random.seed(42)
torch.manual_seed(42)


# ==================== NETWORK MODULES ====================
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


class Discriminator(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# ==================== LOSS HELPERS ====================
def compute_mmd(x, y, bandwidth=1.0):
    """Maximum Mean Discrepancy with Gaussian kernel."""
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    k_xx = torch.exp(-xx / (2 * bandwidth ** 2)).mean()
    k_yy = torch.exp(-yy / (2 * bandwidth ** 2)).mean()
    k_xy = torch.exp(-xy / (2 * bandwidth ** 2)).mean()

    return k_xx + k_yy - 2 * k_xy


# ==================== APPROACH A: VANILLA AE + FINE-TUNE ====================
def train_vanilla_ae(X_source, Y_source, X_target_train, Y_target_train,
                     X_target_test, Y_target_test, epochs=200, patience=20):
    """Train AE on source, fine-tune predictor on target."""
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)

    # Phase 1: Train on source (reconstruction + prediction)
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
            x_recon = decoder(z)
            y_pred = predictor(z)
            loss = mse(x_recon, xb) + mse(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(src_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {
                'encoder': {k: v.clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.clone() for k, v in decoder.state_dict().items()},
                'predictor': {k: v.clone() for k, v in predictor.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    encoder.load_state_dict(best_state['encoder'])
    decoder.load_state_dict(best_state['decoder'])
    predictor.load_state_dict(best_state['predictor'])

    # Phase 2: Fine-tune predictor on target (freeze encoder)
    for p in encoder.parameters():
        p.requires_grad = False

    ft_optimizer = optim.Adam(predictor.parameters(), lr=5e-4)
    tgt_dataset = TensorDataset(
        torch.FloatTensor(X_target_train).to(DEVICE),
        torch.FloatTensor(Y_target_train).to(DEVICE))
    tgt_loader = DataLoader(tgt_dataset, batch_size=256, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        predictor.train()
        epoch_loss = 0
        for xb, yb in tgt_loader:
            z = encoder(xb)
            y_pred = predictor(z)
            loss = mse(y_pred, yb)
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(tgt_loader), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_pred_state = {k: v.clone()
                               for k, v in predictor.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    predictor.load_state_dict(best_pred_state)

    # Evaluate
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_target_test).to(DEVICE)
        z_test = encoder(X_test_t)
        preds = predictor(z_test).cpu().numpy()
        latent = z_test.cpu().numpy()

    return preds, latent, encoder, predictor


# ==================== APPROACH B: MMD-REGULARIZED AE ====================
def train_mmd_ae(X_source, Y_source, X_target_train, Y_target_train,
                 X_target_test, Y_target_test, epochs=200, patience=20,
                 beta=1.0):
    """Train AE with MMD penalty to align source/target latent distributions."""
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(predictor.parameters()), lr=1e-3)

    mse = nn.MSELoss()

    X_src_t = torch.FloatTensor(X_source).to(DEVICE)
    Y_src_t = torch.FloatTensor(Y_source).to(DEVICE)
    X_tgt_t = torch.FloatTensor(X_target_train).to(DEVICE)
    Y_tgt_t = torch.FloatTensor(Y_target_train).to(DEVICE)

    best_loss = float('inf')
    patience_counter = 0
    n_src = len(X_source)
    n_tgt = len(X_target_train)
    batch_size = 256

    for epoch in range(epochs):
        encoder.train(); decoder.train(); predictor.train()

        src_perm = torch.randperm(n_src)
        tgt_perm = torch.randperm(n_tgt)

        epoch_loss = 0
        n_batches = max(n_src, n_tgt) // batch_size + 1

        for batch_idx in range(n_batches):
            s_start = (batch_idx * batch_size) % n_src
            s_idx = src_perm[s_start:s_start + batch_size]
            if len(s_idx) < 2:
                continue

            t_start = (batch_idx * batch_size) % n_tgt
            t_idx = tgt_perm[t_start:t_start + batch_size]
            if len(t_idx) < 2:
                continue

            xb_s = X_src_t[s_idx]
            yb_s = Y_src_t[s_idx]
            xb_t = X_tgt_t[t_idx]
            yb_t = Y_tgt_t[t_idx]

            z_s = encoder(xb_s)
            z_t = encoder(xb_t)

            recon_s = decoder(z_s)
            recon_t = decoder(z_t)
            loss_recon = mse(recon_s, xb_s) + mse(recon_t, xb_t)

            pred_s = predictor(z_s)
            pred_t = predictor(z_t)
            loss_pred = mse(pred_s, yb_s) + mse(pred_t, yb_t)

            loss_mmd = compute_mmd(z_s, z_t)

            loss = loss_recon + loss_pred + beta * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {
                'encoder': {k: v.clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.clone() for k, v in decoder.state_dict().items()},
                'predictor': {k: v.clone() for k, v in predictor.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    encoder.load_state_dict(best_state['encoder'])
    decoder.load_state_dict(best_state['decoder'])
    predictor.load_state_dict(best_state['predictor'])

    encoder.eval(); predictor.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_target_test).to(DEVICE)
        z_test = encoder(X_test_t)
        preds = predictor(z_test).cpu().numpy()
        latent = z_test.cpu().numpy()

    return preds, latent, encoder, predictor


# ==================== APPROACH C: DOMAIN-ADVERSARIAL AE (DAAE) ====================
def train_daae(X_source, Y_source, X_target_train, Y_target_train,
               X_target_test, Y_target_test, epochs=200, patience=20,
               lambda_max=1.0, warmup_epochs=50):
    """Domain-Adversarial Autoencoder with alternating optimization."""
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    opt_main = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(predictor.parameters()), lr=1e-3)
    opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    X_src_t = torch.FloatTensor(X_source).to(DEVICE)
    Y_src_t = torch.FloatTensor(Y_source).to(DEVICE)
    X_tgt_t = torch.FloatTensor(X_target_train).to(DEVICE)
    Y_tgt_t = torch.FloatTensor(Y_target_train).to(DEVICE)

    n_src = len(X_source)
    n_tgt = len(X_target_train)
    batch_size = 256

    best_pred_loss = float('inf')
    patience_counter = 0
    loss_history = {'recon': [], 'pred': [], 'disc': [], 'adv': []}

    for epoch in range(epochs):
        lam = lambda_max * min(1.0, epoch / warmup_epochs)

        encoder.train(); decoder.train()
        predictor.train(); discriminator.train()

        src_perm = torch.randperm(n_src)
        tgt_perm = torch.randperm(n_tgt)

        epoch_recon = 0; epoch_pred = 0; epoch_disc = 0; epoch_adv = 0
        n_batches = max(n_src, n_tgt) // batch_size + 1

        for batch_idx in range(n_batches):
            s_start = (batch_idx * batch_size) % n_src
            s_idx = src_perm[s_start:s_start + batch_size]
            if len(s_idx) < 2:
                continue

            t_start = (batch_idx * batch_size) % n_tgt
            t_idx = tgt_perm[t_start:t_start + batch_size]
            if len(t_idx) < 2:
                continue

            xb_s, yb_s = X_src_t[s_idx], Y_src_t[s_idx]
            xb_t, yb_t = X_tgt_t[t_idx], Y_tgt_t[t_idx]

            # Step 1: Update discriminator
            with torch.no_grad():
                z_s = encoder(xb_s)
                z_t = encoder(xb_t)

            labels_src = torch.full((len(z_s), 1), 0.9, device=DEVICE)
            labels_tgt = torch.full((len(z_t), 1), 0.1, device=DEVICE)

            disc_pred_s = discriminator(z_s.detach())
            disc_pred_t = discriminator(z_t.detach())
            loss_disc = bce(disc_pred_s, labels_src) + bce(disc_pred_t, labels_tgt)

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Step 2: Update encoder + decoder + predictor
            z_s = encoder(xb_s)
            z_t = encoder(xb_t)

            recon_s = decoder(z_s)
            recon_t = decoder(z_t)
            loss_recon = mse(recon_s, xb_s) + mse(recon_t, xb_t)

            pred_s = predictor(z_s)
            pred_t = predictor(z_t)
            loss_pred = mse(pred_s, yb_s) + mse(pred_t, yb_t)

            # Adversarial: fool discriminator (reverse labels)
            disc_pred_s2 = discriminator(z_s)
            disc_pred_t2 = discriminator(z_t)
            adv_labels_s = torch.full((len(z_s), 1), 0.1, device=DEVICE)
            adv_labels_t = torch.full((len(z_t), 1), 0.9, device=DEVICE)
            loss_adv = bce(disc_pred_s2, adv_labels_s) + \
                       bce(disc_pred_t2, adv_labels_t)

            loss_main = loss_recon + loss_pred + lam * loss_adv

            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

            epoch_recon += loss_recon.item()
            epoch_pred += loss_pred.item()
            epoch_disc += loss_disc.item()
            epoch_adv += loss_adv.item()

        epoch_recon /= n_batches
        epoch_pred /= n_batches
        epoch_disc /= n_batches
        epoch_adv /= n_batches

        loss_history['recon'].append(epoch_recon)
        loss_history['pred'].append(epoch_pred)
        loss_history['disc'].append(epoch_disc)
        loss_history['adv'].append(epoch_adv)

        if epoch_pred < best_pred_loss:
            best_pred_loss = epoch_pred
            patience_counter = 0
            best_state = {
                'encoder': {k: v.clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.clone() for k, v in decoder.state_dict().items()},
                'predictor': {k: v.clone() for k, v in predictor.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    encoder.load_state_dict(best_state['encoder'])
    decoder.load_state_dict(best_state['decoder'])
    predictor.load_state_dict(best_state['predictor'])

    encoder.eval(); predictor.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_target_test).to(DEVICE)
        z_test = encoder(X_test_t)
        preds = predictor(z_test).cpu().numpy()
        latent = z_test.cpu().numpy()

    return preds, latent, encoder, predictor, loss_history


# ==================== EXPERIMENT RUNNER ====================
def run_ae_experiment(df_bcm, df_bm, frac, scaler_X, scaler_Y,
                      random_state=42):
    """Run all 3 AE approaches at a given BM data fraction."""
    if frac < 1.0:
        df_target = df_bm.sample(frac=frac, random_state=random_state)
    else:
        df_target = df_bm.copy()

    X_target = df_target[SHARED_FEATURES].values
    Y_target = df_target[TARGETS].values

    if len(df_target) < 20:
        return None

    # Split target: 80/20 train/test
    X_tgt_train, X_tgt_test, Y_tgt_train, Y_tgt_test = train_test_split(
        X_target, Y_target, test_size=0.2, random_state=random_state)

    # Scale using source-fit scalers
    X_tgt_train_sc = scaler_X.transform(X_tgt_train)
    X_tgt_test_sc = scaler_X.transform(X_tgt_test)
    Y_tgt_train_sc = scaler_Y.transform(Y_tgt_train)
    Y_tgt_test_sc = scaler_Y.transform(Y_tgt_test)

    # Source data
    X_src_sc = scaler_X.transform(df_bcm[SHARED_FEATURES].values)
    Y_src_sc = scaler_Y.transform(df_bcm[TARGETS].values)

    # Subsample source for efficiency
    max_src = min(len(X_src_sc), 10000)
    rng = np.random.RandomState(random_state)
    src_idx = rng.choice(len(X_src_sc), max_src, replace=False)
    X_src_sub = X_src_sc[src_idx]
    Y_src_sub = Y_src_sc[src_idx]

    results = {
        'frac': frac,
        'n_train': len(X_tgt_train),
        'n_test': len(X_tgt_test),
    }

    # --- Approach A: Vanilla AE ---
    preds_a, latent_a, _, _ = train_vanilla_ae(
        X_src_sub, Y_src_sub, X_tgt_train_sc, Y_tgt_train_sc,
        X_tgt_test_sc, Y_tgt_test_sc)

    preds_a_inv = scaler_Y.inverse_transform(preds_a)
    results['vanilla_f0_r2'] = float(r2_score(Y_tgt_test[:, 0], preds_a_inv[:, 0]))
    results['vanilla_spl_r2'] = float(r2_score(Y_tgt_test[:, 1], preds_a_inv[:, 1]))

    # --- Approach B: MMD AE ---
    preds_b, latent_b, _, _ = train_mmd_ae(
        X_src_sub, Y_src_sub, X_tgt_train_sc, Y_tgt_train_sc,
        X_tgt_test_sc, Y_tgt_test_sc)

    preds_b_inv = scaler_Y.inverse_transform(preds_b)
    results['mmd_f0_r2'] = float(r2_score(Y_tgt_test[:, 0], preds_b_inv[:, 0]))
    results['mmd_spl_r2'] = float(r2_score(Y_tgt_test[:, 1], preds_b_inv[:, 1]))

    # --- Approach C: DAAE ---
    preds_c, latent_c, enc_c, _, loss_hist = train_daae(
        X_src_sub, Y_src_sub, X_tgt_train_sc, Y_tgt_train_sc,
        X_tgt_test_sc, Y_tgt_test_sc)

    preds_c_inv = scaler_Y.inverse_transform(preds_c)
    results['daae_f0_r2'] = float(r2_score(Y_tgt_test[:, 0], preds_c_inv[:, 0]))
    results['daae_spl_r2'] = float(r2_score(Y_tgt_test[:, 1], preds_c_inv[:, 1]))

    results['loss_history'] = loss_hist

    # Get source/target latent for t-SNE
    enc_c.eval()
    with torch.no_grad():
        src_sample_idx = rng.choice(len(X_src_sub),
                                    min(500, len(X_src_sub)), replace=False)
        tgt_sample_idx = rng.choice(len(X_tgt_test_sc),
                                    min(500, len(X_tgt_test_sc)), replace=False)
        z_src = enc_c(torch.FloatTensor(
            X_src_sub[src_sample_idx]).to(DEVICE)).cpu().numpy()
        z_tgt = enc_c(torch.FloatTensor(
            X_tgt_test_sc[tgt_sample_idx]).to(DEVICE)).cpu().numpy()
    results['tsne_data'] = {'z_src': z_src, 'z_tgt': z_tgt}

    return results


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("AUTOENCODER DOMAIN ADAPTATION: BCM -> BEAM-MEMBRANE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    bcm_path = os.path.join(script_dir, 'dataset_BCM.csv')
    df_bcm = pd.read_csv(bcm_path, index_col=0)
    df_bcm = df_bcm.rename(columns={'Ps': 'PS'})

    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    if not os.path.exists(bm_path):
        print(f"\n  ERROR: BM dataset not found!")
        print(f"  Expected: {os.path.join(script_dir, 'dataset_BM.csv')}")
        print(f"  Run Generate_BM_Dataset.m in MATLAB first, then copy")
        print(f"  the output CSV to the Beam_Membrane/ directory.")
        return

    df_bm = pd.read_csv(bm_path)
    if 'Ps' in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})

    # Drop NaN rows
    n_before = len(df_bm)
    df_bm = df_bm.dropna(subset=['F0', 'SPL'])
    if len(df_bm) < n_before:
        print(f"  Dropped {n_before - len(df_bm)} NaN rows from BM data")

    # Keep only shared features + targets
    df_bm = df_bm[SHARED_FEATURES + TARGETS].copy()

    print(f"  BCM: {len(df_bcm)} samples")
    print(f"  BM:  {len(df_bm)} samples")
    print(f"  Device: {DEVICE}")

    # Fit scalers on source (BCM) data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.fit(df_bcm[SHARED_FEATURES].values)
    scaler_Y.fit(df_bcm[TARGETS].values)

    # Run experiments
    all_results = []
    last_tsne = None
    last_loss_hist = None

    for frac in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with {frac*100:.0f}% of BM data")
        print(f"{'='*50}")

        frac_results = []
        for run in range(N_RUNS):
            rs = 42 + run
            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)

            result = run_ae_experiment(
                df_bcm, df_bm, frac, scaler_X, scaler_Y, rs)

            if result is not None:
                frac_results.append(result)
                print(f"Vanilla={result['vanilla_f0_r2']:.3f}/"
                      f"{result['vanilla_spl_r2']:.3f}, "
                      f"MMD={result['mmd_f0_r2']:.3f}/"
                      f"{result['mmd_spl_r2']:.3f}, "
                      f"DAAE={result['daae_f0_r2']:.3f}/"
                      f"{result['daae_spl_r2']:.3f}")

                if run == 0:
                    last_tsne = result.get('tsne_data')
                    last_loss_hist = result.get('loss_history')
            else:
                print("SKIPPED (too few samples)")

        if frac_results:
            avg = {'frac': frac,
                   'n_samples': int(np.mean([r['n_train']
                                             for r in frac_results]))}
            for method in ['vanilla', 'mmd', 'daae']:
                for target in ['f0', 'spl']:
                    vals = [r[f'{method}_{target}_r2'] for r in frac_results]
                    avg[f'{method}_{target}_r2_mean'] = float(np.mean(vals))
                    avg[f'{method}_{target}_r2_std'] = float(np.std(vals))
            all_results.append(avg)

    # ==================== RESULTS SUMMARY ====================
    print("\n\n" + "=" * 70)
    print("AUTOENCODER RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Frac':>5} {'N':>6} {'Vanilla F0':>10} {'Vanilla SPL':>11} "
          f"{'MMD F0':>8} {'MMD SPL':>9} {'DAAE F0':>9} {'DAAE SPL':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['frac']*100:>4.0f}% {r['n_samples']:>6} "
              f"{r['vanilla_f0_r2_mean']:>10.3f} "
              f"{r['vanilla_spl_r2_mean']:>11.3f} "
              f"{r['mmd_f0_r2_mean']:>8.3f} {r['mmd_spl_r2_mean']:>9.3f} "
              f"{r['daae_f0_r2_mean']:>9.3f} "
              f"{r['daae_spl_r2_mean']:>10.3f}")

    # Best AE method per fraction
    print("\nBEST AE METHOD PER DATA SIZE:")
    print("-" * 50)
    for r in all_results:
        methods = {
            'Vanilla': (r['vanilla_f0_r2_mean'] +
                        r['vanilla_spl_r2_mean']) / 2,
            'MMD': (r['mmd_f0_r2_mean'] + r['mmd_spl_r2_mean']) / 2,
            'DAAE': (r['daae_f0_r2_mean'] + r['daae_spl_r2_mean']) / 2,
        }
        best = max(methods, key=methods.get)
        print(f"  {r['frac']*100:>4.0f}% ({r['n_samples']:>6} samples): "
              f"{best:<8} (avg R2={methods[best]:.3f})")

    # ==================== SAVE RESULTS ====================
    results_path = os.path.join(script_dir, 'results',
                                'autoencoder_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ==================== VISUALIZATION ====================
    print("\nGenerating plots...")
    figs_dir = os.path.join(script_dir, 'figs')

    n_samples = [r['n_samples'] for r in all_results]

    # --- Plot 1: R2 vs sample count ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Autoencoder Domain Adaptation: BCM -> Beam-Membrane',
                 fontsize=14, fontweight='bold')

    ae_methods = [
        ('vanilla', 'Vanilla AE + Fine-Tune', 'steelblue', '-'),
        ('mmd', 'MMD-Regularized AE', 'darkorange', '-'),
        ('daae', 'Domain-Adversarial AE', 'darkred', '-'),
    ]

    for key, label, color, ls in ae_methods:
        vals = [r[f'{key}_f0_r2_mean'] for r in all_results]
        stds = [r[f'{key}_f0_r2_std'] for r in all_results]
        axes[0].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[0].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.15, color=color)
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('R2 for F0')
    axes[0].set_title('F0 Prediction')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for key, label, color, ls in ae_methods:
        vals = [r[f'{key}_spl_r2_mean'] for r in all_results]
        stds = [r[f'{key}_spl_r2_std'] for r in all_results]
        axes[1].plot(n_samples, vals, 'o-', label=label, color=color,
                     linestyle=ls, linewidth=2, markersize=6)
        axes[1].fill_between(n_samples,
                             [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             alpha=0.15, color=color)
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('R2 for SPL')
    axes[1].set_title('SPL Prediction')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'bm_ae_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: t-SNE of latent space ---
    if last_tsne is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle('DAAE Latent Space: BCM vs BM (t-SNE)',
                      fontsize=14, fontweight='bold')

        z_src = last_tsne['z_src']
        z_tgt = last_tsne['z_tgt']
        z_combined = np.vstack([z_src, z_tgt])

        if len(z_combined) > 0 and z_combined.shape[1] > 1:
            perplexity = min(30, len(z_combined) - 1)
            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=perplexity)
            z_2d = tsne.fit_transform(z_combined)

            n_src_pts = len(z_src)
            axes2[0].scatter(z_2d[:n_src_pts, 0], z_2d[:n_src_pts, 1],
                            alpha=0.4, s=10, c='blue', label='BCM (source)')
            axes2[0].scatter(z_2d[n_src_pts:, 0], z_2d[n_src_pts:, 1],
                            alpha=0.4, s=10, c='red', label='BM (target)')
            axes2[0].set_title('DAAE Latent Space')
            axes2[0].legend()
            axes2[0].set_xlabel('t-SNE 1')
            axes2[0].set_ylabel('t-SNE 2')

        if len(z_combined) > 0:
            axes2[1].hist(z_src.flatten(), bins=50, alpha=0.5,
                         density=True, color='blue', label='BCM')
            axes2[1].hist(z_tgt.flatten(), bins=50, alpha=0.5,
                         density=True, color='red', label='BM')
            axes2[1].set_title('Latent Activation Distribution')
            axes2[1].set_xlabel('Activation Value')
            axes2[1].set_ylabel('Density')
            axes2[1].legend()

        plt.tight_layout()
        fig2_path = os.path.join(figs_dir, 'bm_ae_latent.png')
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig2_path}")
        plt.close()

    # --- Plot 3: DAAE training curves ---
    if last_loss_hist is not None:
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
        fig3.suptitle('DAAE Training Curves (BCM -> BM)',
                      fontsize=14, fontweight='bold')

        epochs_x = range(1, len(last_loss_hist['recon']) + 1)

        axes3[0, 0].plot(epochs_x, last_loss_hist['recon'], color='blue')
        axes3[0, 0].set_title('Reconstruction Loss')
        axes3[0, 0].set_xlabel('Epoch')
        axes3[0, 0].grid(True, alpha=0.3)

        axes3[0, 1].plot(epochs_x, last_loss_hist['pred'], color='green')
        axes3[0, 1].set_title('Prediction Loss')
        axes3[0, 1].set_xlabel('Epoch')
        axes3[0, 1].grid(True, alpha=0.3)

        axes3[1, 0].plot(epochs_x, last_loss_hist['disc'], color='red')
        axes3[1, 0].set_title('Discriminator Loss')
        axes3[1, 0].set_xlabel('Epoch')
        axes3[1, 0].grid(True, alpha=0.3)

        axes3[1, 1].plot(epochs_x, last_loss_hist['adv'], color='purple')
        axes3[1, 1].set_title('Adversarial Loss (Encoder)')
        axes3[1, 1].set_xlabel('Epoch')
        axes3[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig3_path = os.path.join(figs_dir, 'bm_ae_training.png')
        plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig3_path}")
        plt.close()

    print("\n" + "=" * 70)
    print("AUTOENCODER EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
