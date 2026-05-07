"""
BCM -> Beam-Membrane: Gaussian Process baseline (non-transfer)

Tracks team/TODO.md #1. One of two non-transfer alternates evaluated
against Callum's transfer methods on the same BM small-N regime.

Method:
  GP with kernel = ConstantKernel * Matern(nu=2.5) + WhiteKernel.
  Independent regressor per output (F0 and SPL trained separately).
  Hyperparameters tuned via marginal-likelihood maximization
  (sklearn default), 3 random restarts.

Harness mirrors BM_SmallData.py:
  - Sample sizes: N in [5, 10, 20, 30, 50, 75, 100]
  - Fixed test pool: 1000 BM samples drawn from outside the train sub-sample
  - Schema: drop a_LCA, [a_CT, a_TA, PS] -> [F0, SPL]
  - Per-sub-sample scalers (never reuse across runs)
  - 10 bootstrap runs per N, seeds 42 + run_idx

Output: writes/updates the 'GP' top-level key in
results/alternates_results.json. Run this script in any order with
BM_TabPFN.py — they merge into the same JSON without clobbering.
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


# ==================== METHOD ====================
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
def run_single(n_target, df_bm, random_state):
    """Run GP at one sample count with one seed. Returns
    {'r2_f0': float, 'r2_spl': float, 'n_train': int}.
    """
    if n_target >= len(df_bm):
        df_train = df_bm.copy()
    else:
        df_train = df_bm.sample(n=n_target, random_state=random_state)

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

    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    scalers_Y = [StandardScaler(), StandardScaler()]
    Y_train_sc = np.zeros_like(Y_train, dtype=float)
    for i in range(Y_train.shape[1]):
        Y_train_sc[:, i] = scalers_Y[i].fit_transform(
            Y_train[:, i].reshape(-1, 1)
        ).ravel()

    preds = fit_predict_gp(X_train_sc, Y_train_sc, X_test_sc, scalers_Y)

    return {
        'r2_f0': float(r2_score(Y_test[:, 0], preds[:, 0])),
        'r2_spl': float(r2_score(Y_test[:, 1], preds[:, 1])),
        'n_train': int(len(X_train)),
    }


def run_method(df_bm):
    """Run GP across all N_TARGETS x N_RUNS. Return nested dict."""
    out = {}
    for n in N_TARGETS:
        runs = [run_single(n, df_bm, random_state=42 + r) for r in range(N_RUNS)]
        r2_f0 = [r['r2_f0'] for r in runs]
        r2_spl = [r['r2_spl'] for r in runs]
        out[str(n)] = {'r2_f0': r2_f0, 'r2_spl': r2_spl,
                       'n_train': runs[0]['n_train']}
        print(f"  N={n:>3}  F0 R2 = {np.mean(r2_f0):+.3f} +/- {np.std(r2_f0):.3f}  "
              f"SPL R2 = {np.mean(r2_spl):+.3f} +/- {np.std(r2_spl):.3f}")
    return out


# ==================== I/O ====================
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
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    needed = SHARED_FEATURES + TARGETS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"BM CSV missing columns: {missing}. Has: {list(df.columns)}")
    return df[needed]


def merge_into_alternates(method_results, method_name='GP'):
    """Load existing alternates_results.json (if any), merge in this method's
    results, write back. Allows GP and TabPFN to share the JSON without
    clobbering each other.
    """
    out_path = os.path.join(script_dir, 'results', 'alternates_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    existing[method_name] = method_results
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


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("BM ALTERNATES — GP (Matern 2.5 + WhiteKernel)")
    print("=" * 70)

    print("\nLoading BM data...")
    df_bm = load_bm()
    print(f"  Total rows (after NaN drop): {len(df_bm)}")
    print(f"  F0 range: [{df_bm['F0'].min():.1f}, {df_bm['F0'].max():.1f}]")
    print(f"  SPL range: [{df_bm['SPL'].min():.1f}, {df_bm['SPL'].max():.1f}]")
    print(f"  PS range:  [{df_bm['PS'].min():.1f}, {df_bm['PS'].max():.1f}]")

    print("\nRunning GP across N x bootstrap runs...")
    results = run_method(df_bm)

    out_path = merge_into_alternates(results, 'GP')
    print(f"\nResults written to: {out_path}")
    print("Run BM_TabPFN.py separately to add TabPFN results, then "
          "BM_Summary.py to regenerate the comparison figure.")


if __name__ == '__main__':
    main()
