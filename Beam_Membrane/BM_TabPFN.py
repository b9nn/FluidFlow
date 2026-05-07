"""
BCM -> Beam-Membrane: TabPFN baseline (non-transfer)

Tracks team/TODO.md #1. One of two non-transfer alternates evaluated
against Callum's transfer methods on the same BM small-N regime.

Method:
  TabPFN — pretrained tabular foundation model. Single forward pass
  over (X_train, y_train, X_test) per target; no per-task gradient
  updates. Single-output -> two regressors (one per target). Train
  truncated at TABPFN_MAX_TRAIN samples (matches our N <= 100 regime).

  Backend selection at import time:
    1. tabpfn-client (cloud-API, recommended — no model-weight download)
    2. tabpfn (local install) as fallback

Auth (one-time):
  A) Cloud client (recommended):
       pip install tabpfn-client
       python -c "from tabpfn_client import init; init()"   # browser login
       OR set $env:TABPFN_TOKEN = "<token>"
  B) Local install:
       pip install tabpfn
       (TabPFN >= 7.x requires license + TABPFN_TOKEN; the package
        handles its own auth)

Harness mirrors BM_SmallData.py:
  - Sample sizes: N in [5, 10, 20, 30, 50, 75, 100]
  - Fixed test pool: 1000 BM samples drawn from outside the train sub-sample
  - Schema: drop a_LCA, [a_CT, a_TA, PS] -> [F0, SPL]
  - Per-sub-sample scalers
  - 10 bootstrap runs per N, seeds 42 + run_idx

Output: writes/updates the 'TabPFN' top-level key in
results/alternates_results.json. Run this script in any order with
BM_GP.py — they merge into the same JSON without clobbering.
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Backend selection: prefer tabpfn-client, fall back to local tabpfn
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

TABPFN_MAX_TRAIN = 1000


# ==================== METHOD ====================
def _ensure_tabpfn_auth():
    """If using tabpfn-client and TABPFN_TOKEN is in the environment, set it.
    Otherwise rely on a previously cached interactive login. No-op for the
    local 'tabpfn' backend (which uses its own license/token system).
    """
    if _TABPFN_BACKEND == 'tabpfn-client':
        token = os.environ.get('TABPFN_TOKEN')
        if token and _set_tabpfn_token is not None:
            _set_tabpfn_token(token)


def fit_predict_tabpfn(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output,
                       random_state=42):
    """Predict with TabPFN. Single-output -> two regressors (one per
    target). Truncates train to TABPFN_MAX_TRAIN samples.
    """
    if not _TABPFN_AVAILABLE:
        raise RuntimeError(
            "Neither tabpfn-client nor tabpfn installed. "
            "Recommended: pip install tabpfn-client"
        )
    _ensure_tabpfn_auth()

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
def run_single(n_target, df_bm, random_state):
    """Run TabPFN at one sample count with one seed. Returns
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

    preds = fit_predict_tabpfn(X_train_sc, Y_train_sc, X_test_sc, scalers_Y,
                                random_state=random_state)

    return {
        'r2_f0': float(r2_score(Y_test[:, 0], preds[:, 0])),
        'r2_spl': float(r2_score(Y_test[:, 1], preds[:, 1])),
        'n_train': int(len(X_train)),
    }


def run_method(df_bm):
    """Run TabPFN across all N_TARGETS x N_RUNS. Return nested dict."""
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


def merge_into_alternates(method_results, method_name='TabPFN'):
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
    print(f"BM ALTERNATES — TabPFN (backend = {_TABPFN_BACKEND})")
    print("=" * 70)

    if not _TABPFN_AVAILABLE:
        print("\nERROR: neither tabpfn-client nor tabpfn is installed.")
        print("Recommended: pip install tabpfn-client && python -c "
              "\"from tabpfn_client import init; init()\"")
        return

    print("\nLoading BM data...")
    df_bm = load_bm()
    print(f"  Total rows (after NaN drop): {len(df_bm)}")
    print(f"  F0 range: [{df_bm['F0'].min():.1f}, {df_bm['F0'].max():.1f}]")
    print(f"  SPL range: [{df_bm['SPL'].min():.1f}, {df_bm['SPL'].max():.1f}]")
    print(f"  PS range:  [{df_bm['PS'].min():.1f}, {df_bm['PS'].max():.1f}]")

    print("\nRunning TabPFN across N x bootstrap runs...")
    try:
        results = run_method(df_bm)
    except Exception as e:
        print(f"\nERROR: TabPFN run failed: {type(e).__name__}: {e}")
        print("See module docstring for auth setup.")
        return

    out_path = merge_into_alternates(results, 'TabPFN')
    print(f"\nResults written to: {out_path}")
    print("Run BM_GP.py separately to add GP results, then BM_Summary.py "
          "to regenerate the comparison figure.")


if __name__ == '__main__':
    main()
