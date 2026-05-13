"""
BCM -> TBCM: Gaussian Process baseline (non-transfer)

Mirrors Beam_Membrane/BM_GP.py — same harness, same kernel, same N grid.
Only the data loader and output path change. Companion to TBCM_TabPFN.py.

Produces the alternates side of the BCM->TBCM head-to-head; the transfer
comparator at small N comes from the new rf_transfer_small_n.json dump
produced by Task 5.5.

Output: writes/updates the 'GP' top-level key in
TBCM/results/alternates_results.json.
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

N_TARGETS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 10

SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
TEST_POOL_SIZE = 1000


def fit_predict_gp(X_train_sc, Y_train_sc, X_test_sc, scaler_Y_per_output):
    preds = np.zeros((X_test_sc.shape[0], Y_train_sc.shape[1]))
    for i in range(Y_train_sc.shape[1]):
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False,
            n_restarts_optimizer=3, random_state=42,
        )
        gp.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = gp.predict(X_test_sc)
        preds[:, i] = scaler_Y_per_output[i].inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()
    return preds


def run_single(n_target, df, random_state):
    if n_target >= len(df):
        df_train = df.copy()
    else:
        df_train = df.sample(n=n_target, random_state=random_state)
    test_pool = df.drop(df_train.index)
    if len(test_pool) < 100:
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

    preds = fit_predict_gp(X_train_sc, Y_train_sc, X_test_sc, scalers_Y)
    return {
        'r2_f0': float(r2_score(Y_test[:, 0], preds[:, 0])),
        'r2_spl': float(r2_score(Y_test[:, 1], preds[:, 1])),
        'n_train': int(len(X_train)),
    }


def _existing_complete_ns(method_name='GP'):
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
    done = _existing_complete_ns('GP') if skip_existing else set()
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


def load_tbcm():
    """Load TBCM dataset matching TBCM_TransferRF/TBCM_SmallData conventions."""
    p = os.path.join(script_dir, 'dataset_TBCM.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"TBCM dataset not found at {p}. "
            f"TBCM_TransferRF.py:212 expects index_col=0."
        )
    df = pd.read_csv(p, index_col=0)
    if 'Ps' in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    needed = SHARED_FEATURES + TARGETS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"TBCM CSV missing columns: {missing}. Has: {list(df.columns)}")
    return df[needed]


def merge_into_alternates(method_results, method_name='GP'):
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
    meta['domain'] = 'TBCM'
    existing['_meta'] = meta
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    return out_path


def main():
    print("=" * 70)
    print("TBCM ALTERNATES -- GP (Matern 2.5 + WhiteKernel)")
    print("=" * 70)
    df = load_tbcm()
    print(f"  rows: {len(df)}")
    print(f"  F0 range: [{df['F0'].min():.1f}, {df['F0'].max():.1f}]")
    print(f"  SPL range: [{df['SPL'].min():.1f}, {df['SPL'].max():.1f}]")
    print(f"  PS range:  [{df['PS'].min():.1f}, {df['PS'].max():.1f}]")
    results = run_method(df)
    out_path = merge_into_alternates(results, 'GP')
    print(f"\nResults written to: {out_path}")


if __name__ == '__main__':
    main()
