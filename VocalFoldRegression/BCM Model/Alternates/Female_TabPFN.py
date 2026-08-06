"""
Male BCM -> Female BCM: TabPFN baseline (non-transfer)

Mirrors Beam_Membrane/BM_TabPFN.py — same backend selection (tabpfn-client
preferred, local tabpfn fallback), same harness shape, same N grid.
Companion to Female_GP.py.

Data convention matches Female_GP.py: ACFL > 30 quality filter, drop NaN
in shared cols, keep [a_CT, a_TA, PS] -> [F0, SPL]. Test pool capped at
500 (female dataset is ~1200 rows post-filter).

Output: writes/updates the 'TabPFN' top-level key in
results/alternates_results.json under this script's directory. Run in any
order with Female_GP.py — they merge into the same JSON.
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
_TABPFN_BACKEND = None
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

N_TARGETS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 10

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
TEST_POOL_SIZE = 500
ACFL_THRESHOLD = 30

TABPFN_MAX_TRAIN = 1000


# ==================== METHOD ====================
def _ensure_tabpfn_auth():
    if _TABPFN_BACKEND == 'tabpfn-client':
        token = os.environ.get('TABPFN_TOKEN')
        if token and _set_tabpfn_token is not None:
            _set_tabpfn_token(token)


def fit_predict_tabpfn(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output,
                       random_state=42):
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
        reg = _TabPFNRegressor(random_state=random_state)
        reg.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = reg.predict(X_test_sc)
        preds[:, i] = scaler_Y_per_output[i].inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()
    return preds


# ==================== HARNESS ====================
def run_single(n_target, df, random_state):
    if n_target >= len(df):
        df_train = df.copy()
    else:
        df_train = df.sample(n=n_target, random_state=random_state)
    test_pool = df.drop(df_train.index)
    if len(test_pool) < 50:
        test_pool = df
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


def _existing_complete_ns(method_name='TabPFN'):
    out_path = os.path.join(script_dir, 'results', 'alternates_results.json')
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        data = json.load(f)
    block = data.get(method_name, {})
    done = set()
    for n_key, rec in (block.items() if isinstance(block, dict) else []):
        if not n_key.isdigit():
            continue
        if len(rec.get('r2_f0', [])) >= N_RUNS and len(rec.get('r2_spl', [])) >= N_RUNS:
            done.add(int(n_key))
    return done


def run_method(df, skip_existing=True):
    done = _existing_complete_ns('TabPFN') if skip_existing else set()
    out = {}
    for n in N_TARGETS:
        if n in done:
            print(f"  N={n:>4}  skip (already in JSON)")
            continue
        runs = [run_single(n, df, random_state=42 + r) for r in range(N_RUNS)]
        r2_f0 = [r['r2_f0'] for r in runs]
        r2_spl = [r['r2_spl'] for r in runs]
        out[str(n)] = {'r2_f0': r2_f0, 'r2_spl': r2_spl,
                       'n_train': runs[0]['n_train']}
        print(f"  N={n:>4}  F0 R2 = {np.mean(r2_f0):+.3f} +/- {np.std(r2_f0):.3f}  "
              f"SPL R2 = {np.mean(r2_spl):+.3f} +/- {np.std(r2_spl):.3f}")
    return out


# ==================== I/O ====================
def load_female():
    """Load female BCM dataset. Same loader as Female_GP.py."""
    p = os.path.abspath(os.path.join(script_dir, '..', 'FemaleBCM.csv'))
    if not os.path.exists(p):
        raise FileNotFoundError(f"FemaleBCM.csv not found at {p}")
    df = pd.read_csv(p)
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    if 'ACFL' in df.columns:
        df = df[df['ACFL'] > ACFL_THRESHOLD]
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    needed = SHARED_FEATURES + TARGETS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"FemaleBCM CSV missing columns: {missing}. "
                         f"Has: {list(df.columns)}")
    return df[needed]


def merge_into_alternates(method_results, method_name='TabPFN'):
    out_path = os.path.join(script_dir, 'results', 'alternates_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    method_block = existing.get(method_name, {}) if isinstance(
        existing.get(method_name), dict) else {}
    for n_key, rec in method_results.items():
        method_block[n_key] = rec
    existing[method_name] = method_block
    all_n = sorted({int(k) for k in method_block.keys() if k.isdigit()})
    meta = existing.get('_meta', {}) or {}
    meta['n_targets'] = sorted(set(meta.get('n_targets', [])) | set(all_n))
    meta['n_runs'] = N_RUNS
    meta['shared_features'] = SHARED_FEATURES
    meta['targets'] = TARGETS
    meta['test_pool_size'] = TEST_POOL_SIZE
    meta['domain'] = 'FemaleBCM'
    meta['quality_filter'] = f'ACFL > {ACFL_THRESHOLD}'
    existing['_meta'] = meta
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    return out_path


def main():
    print("=" * 70)
    print(f"FemaleBCM ALTERNATES -- TabPFN (backend: {_TABPFN_BACKEND})")
    print("=" * 70)
    df = load_female()
    print(f"  rows after ACFL>{ACFL_THRESHOLD} filter: {len(df)}")
    print(f"  F0 range: [{df['F0'].min():.1f}, {df['F0'].max():.1f}]")
    print(f"  SPL range: [{df['SPL'].min():.1f}, {df['SPL'].max():.1f}]")
    print(f"  PS range:  [{df['PS'].min():.1f}, {df['PS'].max():.1f}]")
    results = run_method(df)
    out_path = merge_into_alternates(results, 'TabPFN')
    print(f"\nResults written to: {out_path}")


if __name__ == '__main__':
    main()
