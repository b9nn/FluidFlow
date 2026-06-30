"""
BM extended — Transfer learning (BCM -> BM) for the physiological outputs.

Mirrors JASA/JASA_TransferRF.py but targets the Beam-Membrane (FEM) model with
the extended output set. BCM (MaleBCM, 360k) is the cheap source; BM is the
expensive target. Six transfer methods evaluated over a small-N sweep, per
output, reporting BOTH R2 and normalized RMSE (mean +/- std over seeds).

Why nRMSE not MAPE: SPL goes negative and PC/CPP cross ~0, so MAPE is ill-
defined; range-normalized RMSE (RMSE / (max-min) on the test pool) is robust
and dimensionless across outputs.

Target:  Beam_Membrane/dataset_BM_extended.csv  (filtered ACFL > 30 -> phonating)
Source:  TBCM/dataset_BCM.csv  (= MaleBCM, has matching ACFL/PC/CPP columns)
Shared features: a_CT, a_TA, a_LCA, PS  ->  F0, SPL, ACFL, PC, CPP

Caveat (ACFL): BM ACFL runs ~2-4x higher than the BCM source (definition / sr
nuance, see BM_EXTENDED_DATASET_METHODOLOGY.md). Treated here as domain shift
that transfer must absorb; flagged in figures.

Output: results/bm_ext_transfer.json
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))

FEATURES = ['a_CT', 'a_TA', 'a_LCA', 'PS']
TARGETS = ['F0', 'SPL', 'ACFL', 'PC', 'CPP']
N_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 5
METHODS = ['source_only', 'target_only', 'residual', 'augmented',
           'simple_ens', 'transrf']
ACFL_FILTER = 30.0        # phonating-row filter (matches BCM-style ACFL > 30)
TEST_POOL_SIZE = 1000
SOURCE_MAX_ROWS = 60000   # cap BCM source (full 360k blows memory)


def get_model_params(n_samples):
    """Adaptive RF complexity (copied from JASA/TBCM transfer scripts)."""
    if n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(10, n_samples // 10),
                'min_samples_split': max(5, n_samples // 20)}
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    elif n_samples < 500:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}
    elif n_samples < 1000:
        return {'n_estimators': 100, 'max_depth': 10,
                'min_samples_leaf': 2, 'min_samples_split': 2}
    else:
        return {'n_estimators': 200, 'max_depth': None,
                'min_samples_leaf': 1, 'min_samples_split': 2}


def _nrmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / rng if rng > 0 else float('nan')


def run_experiment(df_full, n_target, rf_source, scaler_X_source, random_state):
    """One transfer experiment at a fixed training size, single output.
    Train on all n_target rows; evaluate on a large disjoint held-out pool."""
    df_train = df_full.sample(n=min(n_target, len(df_full)), random_state=random_state)
    if len(df_train) < 5:
        return None
    df_remainder = df_full.drop(df_train.index)
    df_test_pool = df_remainder.sample(
        n=min(TEST_POOL_SIZE, len(df_remainder)), random_state=random_state)

    X_train = df_train[FEATURES]
    y_train = df_train['_y'].values
    X_test = df_test_pool[FEATURES]
    y_test = df_test_pool['_y'].values

    n_train = len(X_train)
    params = get_model_params(n_train)
    res = {'n_train': n_train}

    def source_pred(X):
        return rf_source.predict(scaler_X_source.transform(X))

    def record(name, pred):
        res[f'{name}_r2'] = r2_score(y_test, pred)
        res[f'{name}_nrmse'] = _nrmse(y_test, pred)

    # Source only (zero-shot BCM)
    pred_src_test = source_pred(X_test)
    record('source_only', pred_src_test)

    # Target only
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    rf_t = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_t.fit(X_train_sc, y_train)
    pred_t = rf_t.predict(X_test_sc)
    record('target_only', pred_t)

    # Residual correction
    pred_src_train = source_pred(X_train)
    rf_r = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_r.fit(X_train.values, y_train - pred_src_train)
    pred_res = pred_src_test + rf_r.predict(X_test.values)
    record('residual', pred_res)

    # Feature augmentation
    X_tr_aug = np.column_stack([X_train.values, pred_src_train])
    X_te_aug = np.column_stack([X_test.values, pred_src_test])
    rf_a = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_a.fit(X_tr_aug, y_train)
    pred_aug = rf_a.predict(X_te_aug)
    record('augmented', pred_aug)

    # Simple ensemble
    record('simple_ens', 0.3 * pred_src_test + 0.7 * pred_t)

    # TransRF (non-negative weights learned in-sample on training preds)
    pred_t_tr = rf_t.predict(X_train_sc)
    pred_r_tr = pred_src_train + rf_r.predict(X_train.values)
    pred_a_tr = rf_a.predict(X_tr_aug)
    stack_tr = np.column_stack([pred_t_tr, pred_r_tr, pred_a_tr])
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(stack_tr, y_train)
    w = reg.coef_
    if w.sum() == 0:
        w = np.array([1 / 3, 1 / 3, 1 / 3])
    pred_trf = np.column_stack([pred_t, pred_res, pred_aug]) @ w
    record('transrf', pred_trf)
    res['weights'] = w.tolist()
    return res


def train_source(df_bcm, target):
    if len(df_bcm) > SOURCE_MAX_ROWS:
        df_bcm = df_bcm.sample(n=SOURCE_MAX_ROWS, random_state=42)
    X = df_bcm[FEATURES]
    y = df_bcm[target].values
    sc = StandardScaler()
    rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                               random_state=42, n_jobs=4)
    rf.fit(sc.fit_transform(X), y)
    return rf, sc


def load_bm():
    p = os.path.join(script_dir, 'dataset_BM_extended.csv')
    df = pd.read_csv(p).rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=FEATURES + TARGETS)
    df = df[df['ACFL'] > ACFL_FILTER].reset_index(drop=True)
    return df


def main():
    print("=" * 70)
    print("BM EXTENDED Component 2 — Transfer BCM -> BM (F0,SPL,ACFL,PC,CPP)")
    print("=" * 70)

    df_bcm = pd.read_csv(os.path.join(script_dir, '..', 'TBCM', 'dataset_BCM.csv'),
                         index_col=0).rename(columns={'Ps': 'PS'})
    print(f"  BCM source rows: {len(df_bcm)}")
    df_bm = load_bm()
    print(f"  BM target rows (ACFL>{ACFL_FILTER:.0f}): {len(df_bm)}")
    print(f"  BM a_LCA range [{df_bm.a_LCA.min():.2f},{df_bm.a_LCA.max():.2f}] "
          f"vs BCM a_LCA [{df_bcm.a_LCA.min():.2f},{df_bcm.a_LCA.max():.2f}] "
          f"(extrapolation expected)")

    all_results = {}
    for target in TARGETS:
        print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")
        rf_source, scaler_src = train_source(df_bcm, target)
        chk = df_bm.sample(n=min(2000, len(df_bm)), random_state=42)
        pred_chk = rf_source.predict(scaler_src.transform(chk[FEATURES]))
        rho, _ = spearmanr(chk[target].values, pred_chk)
        print(f"  source zero-shot on BM: Spearman rho={rho:.3f}, "
              f"R2={r2_score(chk[target].values, pred_chk):.3f}")

        df_t = df_bm.copy()
        df_t['_y'] = df_t[target].values
        target_rows = []
        for n in N_GRID:
            runs = [run_experiment(df_t, n, rf_source, scaler_src, 42 + r)
                    for r in range(N_RUNS)]
            runs = [r for r in runs if r is not None]
            if not runs:
                continue
            row = {'n_samples': n,
                   'n_train': int(round(np.mean([r['n_train'] for r in runs])))}
            for m in METHODS:
                for metric in ('r2', 'nrmse'):
                    vals = [r[f'{m}_{metric}'] for r in runs]
                    row[f'{m}_{metric}_mean'] = float(np.nanmean(vals))
                    row[f'{m}_{metric}_std'] = float(np.nanstd(vals))
            row['weights'] = np.mean([r['weights'] for r in runs], axis=0).tolist()
            target_rows.append(row)
            print(f"  N={n:>4}  target R2={row['target_only_r2_mean']:+.3f}  "
                  f"transrf R2={row['transrf_r2_mean']:+.3f}  "
                  f"transrf nRMSE={row['transrf_nrmse_mean']:.3f}")
        all_results[target] = target_rows

    out_path = os.path.join(script_dir, 'results', 'bm_ext_transfer.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'_meta': {'features': FEATURES, 'targets': TARGETS,
                             'n_grid': N_GRID, 'n_runs': N_RUNS, 'methods': METHODS,
                             'acfl_filter': ACFL_FILTER, 'source': 'MaleBCM',
                             'target': 'dataset_BM_extended.csv'},
                   **all_results}, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == '__main__':
    main()
