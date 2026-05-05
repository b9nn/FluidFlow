"""
BCM -> Beam-Membrane: Non-Transfer Alternate Methods (PHASES 1-3)

Tracks team/TODO.md #1 — exploring whether non-transfer methods can match
or beat transfer at very small N (5-100 BM samples).

Methods:
  - GP:      Gaussian Process with Matern(2.5) kernel  [PHASE 1]
  - PINN:    Physics-informed MLP                       [PHASE 2]
  - TabPFN:  Pretrained tabular foundation model        [PHASE 3, this file]

Mirrors BM_SmallData.py harness:
  - Same sample sizes: N in [5, 10, 20, 30, 50, 75, 100]
  - Same fixed test pool (1000 BM samples from outside the train sub-sample)
  - Same convention: drop a_LCA, schema [a_CT, a_TA, PS] -> [F0, SPL]
  - Per-domain scalers, fit on each sub-sample
  - 10 bootstrap runs per (method, n_samples) with seeds 42 + run_idx

Results land in results/alternates_results.json. BM_Summary.py will be
extended in PHASE 4 to include alternate methods in the comparison figure.
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# TabPFN is optional — falls back gracefully if not installed.
# We prefer tabpfn-client (cloud-API; no model-weight download / license dance)
# and fall back to the local tabpfn package if the user installed that instead.
_TABPFN_AVAILABLE = False
_TABPFN_BACKEND = None  # 'tabpfn-client' or 'tabpfn'
try:
    from tabpfn_client import TabPFNRegressor as _TabPFNRegressor
    from tabpfn_client import set_access_token as _set_tabpfn_token
    _TABPFN_AVAILABLE = True
    _TABPFN_BACKEND = 'tabpfn-client'
except ImportError:
    try:
        from tabpfn import TabPFNRegressor as _TabPFNRegressor
        _set_tabpfn_token = None
        _TABPFN_AVAILABLE = True
        _TABPFN_BACKEND = 'tabpfn'
    except ImportError:
        pass


script_dir = os.path.dirname(os.path.abspath(__file__))

N_TARGETS = [5, 10, 20, 30, 50, 75, 100]
N_RUNS = 10

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
TEST_POOL_SIZE = 1000

# PINN config
DEVICE = torch.device('cpu')
PINN_HIDDEN = 32
PINN_LR = 1e-3
PINN_MAX_EPOCHS = 2000
PINN_PATIENCE = 200
PINN_LAMBDA_MONO = 0.1
PINN_N_MONO_PAIRS = 256
PINN_NUDGE = 0.1  # in standardized input space

# Monotonicity priors: list of (input_idx, output_idx) for d(output)/d(input) >= 0
# Index order: features [a_CT, a_TA, PS], targets [F0, SPL]
# Confirmed in 2026-05-04 brainstorm:
#   d F0  / d a_CT >= 0  (longer fold -> higher pitch)
#   d SPL / d PS   >= 0  (more pressure -> louder)
#   d F0  / d PS   >= 0  (chest-voice physiology — modest but real)
PINN_PRIORS = [(0, 0), (2, 1), (2, 0)]

# TabPFN config
TABPFN_MAX_TRAIN = 1000  # TabPFN's effective training cap; matches our regime


# ==================== METHODS ====================
def fit_predict_gp(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output):
    """Train one GP per output (F0, SPL) on standardized data; return
    predictions in original scale.

    Independent regressors per target. Matern(nu=2.5) kernel scaled by a
    constant, plus a WhiteKernel for noise. Hyperparameters optimized via
    sklearn's marginal-likelihood routine.
    """
    preds = np.zeros((X_test_sc.shape[0], Y_train_sc.shape[1]))
    for i in range(Y_train_sc.shape[1]):
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,  # we standardize Y ourselves
            n_restarts_optimizer=3,
            random_state=42,
        )
        gp.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = gp.predict(X_test_sc)
        preds[:, i] = scaler_Y_per_output[i].inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()
    return preds


class PinnMLP(nn.Module):
    """Small MLP with joint [F0, SPL] head, used for the physics-informed
    method. Inputs and outputs both standardized.
    """
    def __init__(self, in_dim=3, hidden=PINN_HIDDEN, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def monotonicity_penalty(model, n_pairs, in_dim, nudge, priors, device):
    """Sample n_pairs random anchor points in standardized input space,
    nudge the prior's input dimension by `nudge`, evaluate model at both,
    and penalize predictions that violate the d(output)/d(input) >= 0
    constraint.

    Penalty per prior is mean ReLU(y_anchor[out_idx] - y_nudged[out_idx]):
    positive when the model goes the wrong way.
    """
    # Anchors uniform in [-2, 2] (most of N(0, 1) mass; mirrors standardized features)
    anchor = torch.empty(n_pairs, in_dim, device=device).uniform_(-2.0, 2.0)
    total = torch.zeros((), device=device)
    for in_idx, out_idx in priors:
        nudged = anchor.clone()
        nudged[:, in_idx] = nudged[:, in_idx] + nudge
        y_anchor = model(anchor)
        y_nudged = model(nudged)
        # We want y_nudged >= y_anchor for this output. Penalize the gap when violated.
        violation = torch.relu(y_anchor[:, out_idx] - y_nudged[:, out_idx])
        total = total + violation.mean()
    return total


def fit_predict_pinn(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output,
                     random_state=42):
    """Train the physics-informed MLP and return predictions in original scale.

    Uses early stopping on a 10% held-out slice (or all train data if N is
    too small to slice). Loss = MSE + lambda * monotonicity_penalty.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_train = X_train_sc.shape[0]
    in_dim = X_train_sc.shape[1]
    out_dim = Y_train_sc.shape[1]

    # Held-out validation slice for early stopping
    if n_train >= 20:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n_train)
        val_size = max(2, n_train // 10)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        X_tr = X_train_sc[train_idx]
        Y_tr = Y_train_sc[train_idx]
        X_vl = X_train_sc[val_idx]
        Y_vl = Y_train_sc[val_idx]
    else:
        # Too small to slice — train on all, use train as val (no real early stopping)
        X_tr, Y_tr = X_train_sc, Y_train_sc
        X_vl, Y_vl = X_train_sc, Y_train_sc

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32, device=DEVICE)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=DEVICE)
    Y_vl_t = torch.tensor(Y_vl, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

    model = PinnMLP(in_dim=in_dim, out_dim=out_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=PINN_LR)
    mse = nn.MSELoss()

    best_val = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience = 0

    for epoch in range(PINN_MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss_mse = mse(pred, Y_tr_t)
        loss_mono = monotonicity_penalty(
            model, PINN_N_MONO_PAIRS, in_dim, PINN_NUDGE, PINN_PRIORS, DEVICE
        )
        loss = loss_mse + PINN_LAMBDA_MONO * loss_mono
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(X_vl_t), Y_vl_t).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PINN_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_sc = model(X_test_t).cpu().numpy()

    # Inverse-standardize per output
    preds = np.zeros_like(pred_sc)
    for i in range(out_dim):
        preds[:, i] = scaler_Y_per_output[i].inverse_transform(
            pred_sc[:, i].reshape(-1, 1)
        ).ravel()
    return preds


def _ensure_tabpfn_auth():
    """If using tabpfn-client and TABPFN_TOKEN is in the environment, set it.
    Otherwise rely on a previously cached interactive login. No-op for the
    local 'tabpfn' backend (which uses a different license/token system).
    """
    if _TABPFN_BACKEND == 'tabpfn-client':
        token = os.environ.get('TABPFN_TOKEN')
        if token and _set_tabpfn_token is not None:
            _set_tabpfn_token(token)


def fit_predict_tabpfn(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output,
                        random_state=42):
    """Predict with TabPFN — pretrained tabular foundation model.

    Single-output -> two regressors (one per target). Truncates train to
    TABPFN_MAX_TRAIN samples.

    AUTH (one-time):
      Two install options exist and both work here:

      A) Cloud client (recommended — no local model-weight download):
         pip install tabpfn-client
         python -c "from tabpfn_client import init; init()"   # interactive login
         OR set $env:TABPFN_TOKEN = "<token>" in your shell

      B) Local install:
         pip install tabpfn
         (TabPFN >= 7.x requires license acceptance at https://ux.priorlabs.ai
          and TABPFN_TOKEN; the wrapping `tabpfn` package handles its own auth)
    """
    if not _TABPFN_AVAILABLE:
        raise RuntimeError(
            "Neither tabpfn-client nor tabpfn installed. "
            "Recommended: pip install tabpfn-client"
        )

    _ensure_tabpfn_auth()

    # Truncate if above cap
    if X_train_sc.shape[0] > TABPFN_MAX_TRAIN:
        X_train_sc = X_train_sc[:TABPFN_MAX_TRAIN]
        Y_train_sc = Y_train_sc[:TABPFN_MAX_TRAIN]

    out_dim = Y_train_sc.shape[1]
    preds = np.zeros((X_test_sc.shape[0], out_dim))
    for i in range(out_dim):
        # tabpfn-client and local tabpfn share the TabPFNRegressor API but
        # differ slightly in kwargs — keep it minimal for portability
        reg = _TabPFNRegressor(random_state=random_state)
        reg.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = reg.predict(X_test_sc)
        preds[:, i] = scaler_Y_per_output[i].inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()
    return preds


# ==================== EXPERIMENT HARNESS ====================
def run_single(method_name, n_target, df_bm, random_state):
    """Run one method at one sample count with one seed. Returns
    {'r2_f0': float, 'r2_spl': float, 'n_train': int}.
    """
    # 1. Sample n_target rows for training
    if n_target >= len(df_bm):
        df_train = df_bm.copy()
    else:
        df_train = df_bm.sample(n=n_target, random_state=random_state)

    # 2. Test pool = remaining rows (or full data if not enough remaining)
    test_pool = df_bm.drop(df_train.index)
    if len(test_pool) < 100:
        test_pool = df_bm
    df_test = test_pool.sample(
        n=min(TEST_POOL_SIZE, len(test_pool)), random_state=random_state
    )

    X_train = df_train[SHARED_FEATURES].values
    Y_train = df_train[TARGETS].values
    X_test = df_test[SHARED_FEATURES].values
    Y_test = df_test[TARGETS].values

    # 3. Per-domain (per-sub-sample) scalers
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    scalers_Y = [StandardScaler(), StandardScaler()]
    Y_train_sc = np.zeros_like(Y_train, dtype=float)
    for i in range(Y_train.shape[1]):
        Y_train_sc[:, i] = scalers_Y[i].fit_transform(
            Y_train[:, i].reshape(-1, 1)
        ).ravel()

    # 4. Fit + predict per method
    if method_name == 'GP':
        preds = fit_predict_gp(X_train_sc, Y_train_sc, X_test_sc, scalers_Y)
    elif method_name == 'PINN':
        preds = fit_predict_pinn(X_train_sc, Y_train_sc, X_test_sc,
                                  scalers_Y, random_state=random_state)
    elif method_name == 'TabPFN':
        preds = fit_predict_tabpfn(X_train_sc, Y_train_sc, X_test_sc,
                                    scalers_Y, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # 5. Score in original units
    return {
        'r2_f0': float(r2_score(Y_test[:, 0], preds[:, 0])),
        'r2_spl': float(r2_score(Y_test[:, 1], preds[:, 1])),
        'n_train': int(len(X_train)),
    }


def run_method(method_name, df_bm):
    """Run one method across all N_TARGETS x N_RUNS. Return nested dict."""
    out = {}
    for n in N_TARGETS:
        runs = [run_single(method_name, n, df_bm, random_state=42 + r)
                for r in range(N_RUNS)]
        r2_f0 = [r['r2_f0'] for r in runs]
        r2_spl = [r['r2_spl'] for r in runs]
        out[str(n)] = {'r2_f0': r2_f0, 'r2_spl': r2_spl,
                       'n_train': runs[0]['n_train']}
        print(f"  N={n:>3}  F0 R2 = {np.mean(r2_f0):+.3f} +/- {np.std(r2_f0):.3f}  "
              f"SPL R2 = {np.mean(r2_spl):+.3f} +/- {np.std(r2_spl):.3f}")
    return out


# ==================== MAIN ====================
def load_bm():
    """Load BM dataset matching BM_TransferRF/BM_SmallData conventions."""
    bm_path = os.path.join(script_dir, 'dataset_BM.csv')
    if not os.path.exists(bm_path):
        bm_path = os.path.join(script_dir, 'dataset_BM_clean.csv')
    if not os.path.exists(bm_path):
        raise FileNotFoundError(
            f"BM dataset not found at {os.path.join(script_dir, 'dataset_BM.csv')} "
            f"or dataset_BM_clean.csv. Run Generate_BM_Dataset.m first."
        )

    df = pd.read_csv(bm_path)
    if 'Ps' in df.columns:
        df = df.rename(columns={'Ps': 'PS'})

    # Drop rows with NaN F0/SPL (physically invalid configs)
    df = df.dropna(subset=TARGETS).reset_index(drop=True)

    needed = SHARED_FEATURES + TARGETS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"BM CSV missing columns: {missing}. Has: {list(df.columns)}")

    return df[needed]


def merge_into_existing_results(new_method_results, method_name):
    """Load existing alternates_results.json (if any), merge in this method's
    results, write back. Allows adding methods incrementally across phases.
    """
    out_path = os.path.join(script_dir, 'results', 'alternates_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[method_name] = new_method_results
    existing['_meta'] = {
        'n_targets': N_TARGETS,
        'n_runs': N_RUNS,
        'shared_features': SHARED_FEATURES,
        'targets': TARGETS,
        'test_pool_size': TEST_POOL_SIZE,
    }

    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    return out_path


def main():
    print("=" * 70)
    print("BM ALTERNATES — PHASE 1 (GP only)")
    print("=" * 70)

    print("\nLoading BM data...")
    df_bm = load_bm()
    print(f"  Total rows (after NaN drop): {len(df_bm)}")
    print(f"  F0 range: [{df_bm['F0'].min():.1f}, {df_bm['F0'].max():.1f}]")
    print(f"  SPL range: [{df_bm['SPL'].min():.1f}, {df_bm['SPL'].max():.1f}]")
    print(f"  PS range:  [{df_bm['PS'].min():.1f}, {df_bm['PS'].max():.1f}]")

    print("\n" + "-" * 70)
    print("Method: GP (Matern 2.5 + WhiteKernel)")
    print("-" * 70)
    gp_results = run_method('GP', df_bm)
    out_path = merge_into_existing_results(gp_results, 'GP')
    print(f"  GP results written to: {out_path}")

    print("\n" + "-" * 70)
    print(f"Method: PINN (MLP {PINN_HIDDEN}/{PINN_HIDDEN}, lambda={PINN_LAMBDA_MONO}, "
          f"{len(PINN_PRIORS)} monotonicity priors)")
    print("-" * 70)
    pinn_results = run_method('PINN', df_bm)
    merge_into_existing_results(pinn_results, 'PINN')

    if _TABPFN_AVAILABLE:
        print("\n" + "-" * 70)
        print(f"Method: TabPFN (backend = {_TABPFN_BACKEND})")
        print("-" * 70)
        try:
            tabpfn_results = run_method('TabPFN', df_bm)
            merge_into_existing_results(tabpfn_results, 'TabPFN')
        except Exception as e:
            # Auth errors, network errors, etc. — skip and continue
            print(f"  SKIPPED: TabPFN run failed: {type(e).__name__}: {e}")
            print(f"  See fit_predict_tabpfn docstring for auth setup.")
    else:
        print("\nSKIPPED: TabPFN — install with `pip install tabpfn-client` "
              "(cloud) or `pip install tabpfn` (local) to enable")

    print(f"\nResults written to: {out_path}")
    print("\nAll method runs complete. Next phase: BM_Summary integration (#4).")


if __name__ == '__main__':
    main()
