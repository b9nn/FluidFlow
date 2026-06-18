"""
Fair head-to-head: BCM->TBCM transfer (TransRF) vs TabPFN at matched N.

WHY THIS EXISTS
---------------
The original small-data comparison was biased against TransRF by an
evaluation-protocol mismatch, not a modeling difference:

  TBCM_SmallData.py (TransRF):  test_size = max(0.2, min(0.4, 10/len(df)))
      -> at nominal N=10 it TRAINS on only 6 points and TESTS on 4.
      -> R^2 measured on 4-20 points is high-variance and downward-biased.

  TBCM_TabPFN.py (TabPFN):      df_train = sample(N); df_test = up to 1000
      -> TRAINS on the full N and TESTS on a large fixed held-out pool.
      -> R^2 is stable.

So at the same "N", TransRF (a) saw fewer training points and (b) was scored
on a tiny noisy test set. Both depress + destabilize its R^2 and EXAGGERATE
the gap to TabPFN.

This script fixes it by running BOTH methods under ONE protocol:
  train on the full N target points, evaluate on a large fixed held-out pool.
It also re-runs TransRF under the OLD (buggy) protocol so we can quantify how
much of the apparent gap was the protocol.

Target: dataset_TBCM.csv   Source: dataset_BCM.csv   Outputs: F0, SPL.

Outputs:
  results/transfer_vs_tabpfn_fair.json
  figs/transfer_vs_tabpfn_fair.png
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TabPFN backend
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
FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
N_GRID = [10, 20, 50, 100, 200, 500]
N_RUNS = 4
TEST_POOL_SIZE = 500          # large fixed held-out pool (fair protocol)
TABPFN_MAX_TRAIN = 1000
RANDOM_STATE = 42


# ---------- adaptive RF params (matches BM_TransferRF / TBCM_SmallData) ----------
def get_model_params(n_samples):
    if n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(10, n_samples // 10),
                'min_samples_split': max(5, n_samples // 20)}
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    else:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}


# ==================== TransRF (source/target/residual/augmented blend) =========
def transrf_predict(X_train, Y_train, X_test, rf_source, scaler_X_source, rs):
    """Replicates the TransRF blend from TBCM_SmallData.run_rf_experiment,
    but agnostic to how train/test were produced. Returns blended preds."""
    params = get_model_params(len(X_train))

    pred_source = rf_source.predict(scaler_X_source.transform(X_test))
    pred_src_train = rf_source.predict(scaler_X_source.transform(X_train))

    scaler_X = StandardScaler()
    Xtr_sc = scaler_X.fit_transform(X_train)
    Xte_sc = scaler_X.transform(X_test)

    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_target.fit(Xtr_sc, Y_train)
    pred_target = rf_target.predict(Xte_sc)

    rf_res = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_res.fit(X_train, Y_train - pred_src_train)
    pred_residual = pred_source + rf_res.predict(X_test)

    Xtr_aug = np.column_stack([X_train, pred_src_train])
    Xte_aug = np.column_stack([X_test, pred_source])
    rf_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=rs, n_jobs=-1, **params))
    rf_aug.fit(Xtr_aug, Y_train)
    pred_augmented = rf_aug.predict(Xte_aug)

    # learn positive blend weights on the training set (no val split at small N)
    pred_t_tr = rf_target.predict(Xtr_sc)
    pred_r_tr = pred_src_train + rf_res.predict(X_train)
    pred_a_tr = rf_aug.predict(Xtr_aug)

    pred_transrf = np.zeros_like(pred_target)
    for i in range(Y_train.shape[1]):
        stack_tr = np.column_stack([pred_t_tr[:, i], pred_r_tr[:, i], pred_a_tr[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack_tr, Y_train[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        stack_te = np.column_stack([pred_target[:, i], pred_residual[:, i],
                                    pred_augmented[:, i]])
        pred_transrf[:, i] = stack_te @ w
    return pred_transrf


# ==================== TabPFN ====================
def tabpfn_predict(X_train, Y_train, X_test, rs):
    Xtr = X_train[:TABPFN_MAX_TRAIN]
    Ytr = Y_train[:TABPFN_MAX_TRAIN]
    scaler_X = StandardScaler().fit(Xtr)
    Xtr_sc = scaler_X.transform(Xtr)
    Xte_sc = scaler_X.transform(X_test)
    preds = np.zeros((len(X_test), Y_train.shape[1]))
    for i in range(Y_train.shape[1]):
        sy = StandardScaler().fit(Ytr[:, i].reshape(-1, 1))
        ytr = sy.transform(Ytr[:, i].reshape(-1, 1)).ravel()
        reg = _TabPFNRegressor(random_state=rs)
        reg.fit(Xtr_sc, ytr)
        preds[:, i] = sy.inverse_transform(
            reg.predict(Xte_sc).reshape(-1, 1)).ravel()
    return preds


# ==================== protocols ====================
def run_fair(df_tbcm, n, rf_source, scaler_X_source, rs):
    """FAIR: train on full N, test on a large held-out pool. Same split feeds
    both TransRF and TabPFN."""
    df_train = df_tbcm.sample(n=n, random_state=rs)
    pool = df_tbcm.drop(df_train.index)
    df_test = pool.sample(n=min(TEST_POOL_SIZE, len(pool)), random_state=rs)

    Xtr = df_train[FEATURES].values
    Ytr = df_train[TARGETS].values
    Xte = df_test[FEATURES].values
    Yte = df_test[TARGETS].values

    out = {'n_train': len(Xtr), 'n_test': len(Xte)}
    p_trf = transrf_predict(Xtr, Ytr, Xte, rf_source, scaler_X_source, rs)
    out['transrf_f0'] = r2_score(Yte[:, 0], p_trf[:, 0])
    out['transrf_spl'] = r2_score(Yte[:, 1], p_trf[:, 1])
    p_tab = tabpfn_predict(Xtr, Ytr, Xte, rs)
    out['tabpfn_f0'] = r2_score(Yte[:, 0], p_tab[:, 0])
    out['tabpfn_spl'] = r2_score(Yte[:, 1], p_tab[:, 1])
    return out


def run_old_transrf(df_tbcm, n, rf_source, scaler_X_source, rs):
    """OLD (buggy) protocol for TransRF: sample N, split test_size up to 0.4,
    evaluate on the tiny test set. Reproduces TBCM_SmallData behavior."""
    df = df_tbcm.sample(n=n, random_state=rs)
    X = df[FEATURES].values
    Y = df[TARGETS].values
    test_size = max(0.2, min(0.4, 10 / len(df)))
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_size, random_state=rs)
    if len(Xtr) < 5:
        return None
    p_trf = transrf_predict(Xtr, Ytr, Xte, rf_source, scaler_X_source, rs)
    return {'n_train': len(Xtr), 'n_test': len(Xte),
            'transrf_f0': r2_score(Yte[:, 0], p_trf[:, 0]),
            'transrf_spl': r2_score(Yte[:, 1], p_trf[:, 1])}


# ==================== main ====================
def main():
    print("=" * 70)
    print(f"FAIR COMPARISON: TransRF vs TabPFN  (backend: {_TABPFN_BACKEND})")
    print("=" * 70)
    if not _TABPFN_AVAILABLE:
        print("ERROR: TabPFN not installed.")
        return
    if _TABPFN_BACKEND == 'tabpfn-client' and _set_tabpfn_token is not None:
        tok = os.environ.get('TABPFN_TOKEN')
        if tok:
            _set_tabpfn_token(tok)

    # BCM source: the 360,750-row measured male/AE lookup at repo root, loaded
    # from the canonical binary (data_binary.parquet; content-identical to
    # MaleBCM.csv). NOT TBCM/dataset_BCM.csv (a 54k synthetic grid sweep, F0 up
    # to 862 Hz, SPL down to -4 dB) which was the wrong source and caused
    # divergent results.
    src_path = os.path.join(script_dir, '..', 'data_binary.parquet')
    df_bcm = pd.read_parquet(src_path)
    if 'Ps' in df_bcm.columns and 'PS' not in df_bcm.columns:
        df_bcm = df_bcm.rename(columns={'Ps': 'PS'})
    df_bcm = df_bcm.dropna(subset=TARGETS)[FEATURES + TARGETS]
    df_tbcm = pd.read_csv(os.path.join(script_dir, 'dataset_TBCM.csv'), index_col=0)
    df_tbcm = df_tbcm.rename(columns={'Ps': 'PS'}).drop(columns=['PL'])
    print(f"  BCM source: {len(df_bcm)} rows (data_binary.parquet);  TBCM target: {len(df_tbcm)} rows")

    print("  Training BCM source RF...")
    scaler_X_source = StandardScaler()
    Xs = scaler_X_source.fit_transform(df_bcm[FEATURES])
    rf_source = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(Xs, df_bcm[TARGETS])

    results = {'_meta': {'n_grid': N_GRID, 'n_runs': N_RUNS,
                         'test_pool_size': TEST_POOL_SIZE,
                         'target': 'dataset_TBCM.csv', 'source': 'data_binary.parquet',
                         'backend': _TABPFN_BACKEND},
               'fair': {}, 'old_transrf': {}}

    print(f"\n{'N':>5} | {'TransRF(old)':>22} | {'TransRF(fair)':>14} {'TabPFN(fair)':>14}")
    print(f"{'':>5} | {'F0':>10}{'SPL':>11} | {'F0':>7}{'SPL':>7}{'F0':>7}{'SPL':>7}")
    print("-" * 78)

    for n in N_GRID:
        fair_runs, old_runs = [], []
        for r in range(N_RUNS):
            rs = RANDOM_STATE + r
            fair_runs.append(run_fair(df_tbcm, n, rf_source, scaler_X_source, rs))
            o = run_old_transrf(df_tbcm, n, rf_source, scaler_X_source, rs)
            if o:
                old_runs.append(o)

        def agg(runs, key):
            return float(np.mean([x[key] for x in runs])), float(np.std([x[key] for x in runs]))

        fb = {k: agg(fair_runs, k) for k in
              ['transrf_f0', 'transrf_spl', 'tabpfn_f0', 'tabpfn_spl']}
        ob = {k: agg(old_runs, k) for k in ['transrf_f0', 'transrf_spl']}
        results['fair'][str(n)] = {k: {'mean': v[0], 'std': v[1]} for k, v in fb.items()}
        results['fair'][str(n)]['n_train'] = fair_runs[0]['n_train']
        results['fair'][str(n)]['n_test'] = fair_runs[0]['n_test']
        results['old_transrf'][str(n)] = {k: {'mean': v[0], 'std': v[1]} for k, v in ob.items()}
        results['old_transrf'][str(n)]['n_train'] = old_runs[0]['n_train']
        results['old_transrf'][str(n)]['n_test'] = old_runs[0]['n_test']

        print(f"{n:>5} | {ob['transrf_f0'][0]:>10.3f}{ob['transrf_spl'][0]:>11.3f} "
              f"| {fb['transrf_f0'][0]:>7.3f}{fb['transrf_spl'][0]:>7.3f}"
              f"{fb['tabpfn_f0'][0]:>7.3f}{fb['tabpfn_spl'][0]:>7.3f}")

        os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
        with open(os.path.join(script_dir, 'results',
                               'transfer_vs_tabpfn_fair.json'), 'w') as f:
            json.dump(results, f, indent=2)

    _plot(results)
    print("\nDone.")


def _plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('TransRF vs TabPFN on TBCM — fair protocol vs buggy protocol',
                 fontweight='bold')
    ns = N_GRID
    for ax_idx, (t, name) in enumerate([('f0', 'F0'), ('spl', 'SPL')]):
        ax = axes[ax_idx]
        def series(block, key):
            return ([results[block][str(n)][key]['mean'] for n in ns],
                    [results[block][str(n)][key]['std'] for n in ns])
        # TransRF old (buggy)
        m, s = series('old_transrf', f'transrf_{t}')
        ax.plot(ns, m, 'x--', color='darkred', label='TransRF (old/buggy protocol)')
        ax.fill_between(ns, np.array(m)-np.array(s), np.array(m)+np.array(s),
                        alpha=0.1, color='darkred')
        # TransRF fair
        m, s = series('fair', f'transrf_{t}')
        ax.plot(ns, m, 'o-', color='red', label='TransRF (fair protocol)')
        ax.fill_between(ns, np.array(m)-np.array(s), np.array(m)+np.array(s),
                        alpha=0.12, color='red')
        # TabPFN fair
        m, s = series('fair', f'tabpfn_{t}')
        ax.plot(ns, m, 's-', color='C0', label='TabPFN (fair protocol)')
        ax.fill_between(ns, np.array(m)-np.array(s), np.array(m)+np.array(s),
                        alpha=0.12, color='C0')
        ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel('Target samples (N)'); ax.set_ylabel(f'{name} test $R^2$')
        ax.set_title(f'{name}'); ax.grid(alpha=0.3); ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_ylim(-0.5, 1.02); ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    out = os.path.join(script_dir, 'figs', 'transfer_vs_tabpfn_fair.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == '__main__':
    main()
