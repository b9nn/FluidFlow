"""
Component 2 — Transfer learning (BCM -> JASA) for the new outputs.

Replicates the prior RF transfer pipeline (TBCM/TBCM_TransferRF.py) but targets
the clinically-richer outputs MFDR, ACFL, PC (F0/SPL already covered in earlier
work). BCM is the cheap source model; JASA is the target. Six transfer methods
are evaluated over a small-data N sweep, per target.

Shared features: a_CT, a_TA, PS  ->  MFDR, ACFL, PC

Methods (mirror PROJECT_GUIDE.md): source-only, target-only, residual correction,
feature augmentation, simple ensemble (0.3/0.7), TransRF (learned weights).

Output: results/rf_transfer.json
  { target: [ {n_samples, <method>_r2_mean, <method>_r2_std, ...}, ... ] }
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
TARGETS = ['F0', 'SPL', 'MFDR', 'ACFL', 'PC']
# Matches the TabPFN single-target grid at the low end (starts at N=5) so the two
# curves begin at the same point; extends past TabPFN's 200 cap up to 500.
N_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 5
METHODS = ['source_only', 'target_only', 'residual', 'augmented',
           'simple_ens', 'transrf']


def get_model_params(n_samples):
    """Adaptive RF complexity (copied from TBCM_TransferRF.py)."""
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


TEST_POOL_SIZE = 1000  # match TabPFN's held-out test pool size


def run_experiment(df_full, n_target, rf_source, scaler_X_source, random_state):
    """One transfer experiment at a fixed training size, single target.

    Trains on all n_target rows and evaluates on a large disjoint pool (up to
    TEST_POOL_SIZE rows) drawn from df_full outside the training budget, matching
    TabPFN's evaluation design. TransRF weights are learned in-sample (no val
    split) so the sweep can start at the same small N as TabPFN — mirrors
    TBCM/TBCM_SmallData.run_rf_experiment.
    """
    # Draw the n_target-row training budget.
    df_train = df_full.sample(n=min(n_target, len(df_full)), random_state=random_state)
    if len(df_train) < 5:
        return None

    # Disjoint test pool from the remainder (matches TabPFN's split_train_test).
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

    # --- Source only (zero-shot BCM) ---
    pred_src_test = source_pred(X_test)
    res['source_only_r2'] = r2_score(y_test, pred_src_test)

    # --- Target only ---
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    rf_t = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_t.fit(X_train_sc, y_train)
    pred_t = rf_t.predict(X_test_sc)
    res['target_only_r2'] = r2_score(y_test, pred_t)

    # --- Residual correction ---
    pred_src_train = source_pred(X_train)
    rf_r = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_r.fit(X_train.values, y_train - pred_src_train)
    pred_res = pred_src_test + rf_r.predict(X_test.values)
    res['residual_r2'] = r2_score(y_test, pred_res)

    # --- Feature augmentation ---
    X_tr_aug = np.column_stack([X_train.values, pred_src_train])
    X_te_aug = np.column_stack([X_test.values, pred_src_test])
    rf_a = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    rf_a.fit(X_tr_aug, y_train)
    pred_aug = rf_a.predict(X_te_aug)
    res['augmented_r2'] = r2_score(y_test, pred_aug)

    # --- Simple ensemble ---
    pred_simple = 0.3 * pred_src_test + 0.7 * pred_t
    res['simple_ens_r2'] = r2_score(y_test, pred_simple)

    # --- TransRF (non-negative weights learned in-sample on training preds) ---
    pred_t_tr = rf_t.predict(X_train_sc)
    pred_r_tr = pred_src_train + rf_r.predict(X_train.values)
    pred_a_tr = rf_a.predict(X_tr_aug)

    stack_tr = np.column_stack([pred_t_tr, pred_r_tr, pred_a_tr])
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(stack_tr, y_train)
    w = reg.coef_
    # Use the MSE-optimal raw coefs directly. Normalising post-hoc changes the
    # prediction scale (stack_test @ (w/sum(w)) != stack_test @ w) without any
    # guarantee of improving the result; fall back to uniform only if all coefs are 0.
    if w.sum() == 0:
        w = np.array([1/3, 1/3, 1/3])

    stack_test = np.column_stack([pred_t, pred_res, pred_aug])
    pred_trf = stack_test @ w
    res['transrf_r2'] = r2_score(y_test, pred_trf)
    res['weights'] = w.tolist()
    return res


SOURCE_MAX_ROWS = 60000  # cap BCM source (full 360k blows memory); ~prior scale


def train_source(df_bcm, target):
    """Train one BCM source RF for a single target on (subsampled) BCM data."""
    if len(df_bcm) > SOURCE_MAX_ROWS:
        df_bcm = df_bcm.sample(n=SOURCE_MAX_ROWS, random_state=42)
    X = df_bcm[FEATURES]
    y = df_bcm[target].values
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                               random_state=42, n_jobs=4)
    rf.fit(X_sc, y)
    return rf, sc


def main():
    print("=" * 70)
    print("JASA Component 2 — Transfer Learning BCM -> JASA (MFDR, ACFL, PC)")
    print("=" * 70)

    df_bcm = pd.read_csv(os.path.join(script_dir, '..', 'TBCM', 'dataset_BCM.csv'),
                         index_col=0).rename(columns={'Ps': 'PS'})
    print(f"  BCM source rows: {len(df_bcm)}")

    jp = os.path.join(script_dir, 'dataset_JASA.parquet')
    df_jasa = pd.read_parquet(jp) if os.path.exists(jp) else pd.read_csv(
        os.path.join(script_dir, 'dataset_JASA.csv'))
    df_jasa = df_jasa.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    print(f"  JASA target rows: {len(df_jasa)}")

    all_results = {}
    for target in TARGETS:
        print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")
        rf_source, scaler_src = train_source(df_bcm, target)

        # Sanity: Spearman of BCM predictions vs JASA truth.
        chk = df_jasa.sample(n=min(5000, len(df_jasa)), random_state=42)
        pred_chk = rf_source.predict(scaler_src.transform(chk[FEATURES]))
        rho, _ = spearmanr(chk[target].values, pred_chk)
        r2_zero = r2_score(chk[target].values, pred_chk)
        print(f"  source self/zero-shot: Spearman rho={rho:.3f}, zero-shot R2={r2_zero:.3f}")

        df_t = df_jasa.copy()
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
                vals = [r[f'{m}_r2'] for r in runs]
                row[f'{m}_r2_mean'] = float(np.mean(vals))
                row[f'{m}_r2_std'] = float(np.std(vals))
            row['weights'] = np.mean([r['weights'] for r in runs], axis=0).tolist()
            target_rows.append(row)
            print(f"  N={n:>4} (n_train={row['n_train']:>4})  "
                  f"target={row['target_only_r2_mean']:+.3f}  "
                  f"transrf={row['transrf_r2_mean']:+.3f}  "
                  f"featAug={row['augmented_r2_mean']:+.3f}")
        all_results[target] = target_rows

    out_path = os.path.join(script_dir, 'results', 'rf_transfer.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'_meta': {'features': FEATURES, 'targets': TARGETS,
                             'n_grid': N_GRID, 'n_runs': N_RUNS,
                             'methods': METHODS}, **all_results}, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == '__main__':
    main()
