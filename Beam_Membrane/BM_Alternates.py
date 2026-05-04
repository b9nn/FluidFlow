"""
BCM -> Beam-Membrane: Non-Transfer Alternate Methods (PHASE 1: GP only)

Tracks team/TODO.md #1 — exploring whether non-transfer methods can match
or beat transfer at very small N (5-100 BM samples).

Methods (added incrementally across phases):
  - GP:      Gaussian Process with Matern(2.5) kernel  [PHASE 1, this file]
  - PINN:    Physics-informed MLP                       [PHASE 2, pending]
  - TabPFN:  Pretrained tabular foundation model        [PHASE 3, pending]

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


script_dir = os.path.dirname(os.path.abspath(__file__))

N_TARGETS = [5, 10, 20, 30, 50, 75, 100]
N_RUNS = 10

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
TEST_POOL_SIZE = 1000


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
    print(f"\nResults written to: {out_path}")
    print("\nPhase 1 done. Next phases: PINN (#2), TabPFN (#3), "
          "BM_Summary integration (#4).")


if __name__ == '__main__':
    main()
