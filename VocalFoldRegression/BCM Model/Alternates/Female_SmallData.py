"""
Male BCM -> Female BCM: RF transfer at small-N target sizes.

Mirrors TBCM/TBCM_SmallData.py (which mirrors Beam_Membrane/BM_SmallData.py).
The point of this script is to give Female_Showcase a transfer comparator
that is methodologically apples-to-apples with the GP/TabPFN alternates:
each N value is a REAL retrain of the RF transfer ensemble on exactly N
target samples, evaluated on a fixed-size held-out test pool — not the
"evaluate-fully-trained-model-on-N-test-rows" approach of
ResgressorAnalysis/AllRegressorsTransferComparison.py.

Why this exists (2026-05-14): the older
all_regressors_transfer_comparison.csv was used by Female_Showcase as the
transfer line. That CSV interpreted `sample_size` as the EVALUATION test
set size, with the underlying transfer model trained once on ~64% of all
female rows. At small N the script's `np.random.choice` also picked rows
the transfer model had already seen during training — train/test
contamination that flattered transfer at small N. This script removes
both confounds: N is the training-set size, and the test pool excludes
all training rows.

Methods (same five as TBCM/BM_SmallData):
  source     — pre-fit Male BCM RF predicts directly on Female rows
  target     — fresh RF trained on N female rows
  residual   — residual correction: target - source on training residuals
  augmented  — feature augmentation: X concat with source predictions
  transrf    — convex-combination ensemble of target/residual/augmented,
               weights learned by positive-coef linear regression on the
               training set (mirrors TBCM_SmallData's TransRF)

Output:
  results/rf_transfer_small_n.json
    schema-compatible with TBCM/results/rf_transfer_small_n.json so
    Female_Showcase can load it the same way the TBCM panel does.
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
bcm_model_dir = os.path.dirname(script_dir)

# Match Female_GP.py / Female_TabPFN.py harness
N_TARGETS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_RUNS = 10
SHARED_FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
TEST_POOL_SIZE = 500
ACFL_THRESHOLD = 30


def get_model_params(n_samples):
    """Adaptive RF complexity scaled by training set size.
    Matches Beam_Membrane/BM_TransferRF.py:51 convention."""
    if n_samples < 100:
        return {
            'n_estimators': 20, 'max_depth': 3,
            'min_samples_leaf': max(10, n_samples // 10),
            'min_samples_split': max(5, n_samples // 20),
        }
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    else:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}


def load_female():
    p = os.path.join(bcm_model_dir, 'FemaleBCM.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(f"FemaleBCM.csv not found at {p}")
    df = pd.read_csv(p)
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    if 'ACFL' in df.columns:
        df = df[df['ACFL'] > ACFL_THRESHOLD]
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    needed = SHARED_FEATURES + TARGETS
    return df[needed]


def load_male():
    p = os.path.join(bcm_model_dir, 'MaleBCM.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(f"MaleBCM.csv not found at {p}")
    df = pd.read_csv(p)
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    return df[SHARED_FEATURES + TARGETS]


def fit_source_rf(df_male):
    """Train Male BCM source RF. Matches TBCM_SmallData.py:285."""
    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(df_male[SHARED_FEATURES].values)
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    )
    rf.fit(X_sc, df_male[TARGETS].values)
    return rf, scaler_X


def run_single(n_target, df_female, rf_source, scaler_X_source, random_state):
    """Mirrors TBCM_SmallData.run_rf_experiment but with fixed TEST_POOL_SIZE=500
    instead of train_test_split percentage. Same five RF methods."""
    if n_target >= len(df_female):
        df_train = df_female.copy()
    else:
        df_train = df_female.sample(n=n_target, random_state=random_state)
    test_pool = df_female.drop(df_train.index)
    if len(test_pool) < 50:
        test_pool = df_female
    df_test = test_pool.sample(
        n=min(TEST_POOL_SIZE, len(test_pool)), random_state=random_state
    )

    X_train = df_train[SHARED_FEATURES].values
    Y_train = df_train[TARGETS].values
    X_test = df_test[SHARED_FEATURES].values
    Y_test = df_test[TARGETS].values

    n_train = len(X_train)
    if n_train < 5:
        return None

    params = get_model_params(n_train)

    # Source-only — fully-trained male RF predicting directly on female
    pred_source = rf_source.predict(scaler_X_source.transform(X_test))
    pred_source_train = rf_source.predict(scaler_X_source.transform(X_train))

    # Target-only
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    rf_target = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    )
    rf_target.fit(X_train_sc, Y_train)
    pred_target = rf_target.predict(X_test_sc)

    # Residual correction
    rf_res = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    )
    rf_res.fit(X_train, Y_train - pred_source_train)
    pred_residual = pred_source + rf_res.predict(X_test)

    # Feature augmentation
    X_train_aug = np.column_stack([X_train, pred_source_train])
    X_test_aug = np.column_stack([X_test, pred_source])
    rf_aug = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    )
    rf_aug.fit(X_train_aug, Y_train)
    pred_augmented = rf_aug.predict(X_test_aug)

    # TransRF — convex ensemble weights learned on training set
    pred_t_tr = rf_target.predict(X_train_sc)
    pred_r_tr = pred_source_train + rf_res.predict(X_train)
    pred_a_tr = rf_aug.predict(X_train_aug)

    weights = {}
    for i, name in enumerate(TARGETS):
        stack = np.column_stack([pred_t_tr[:, i], pred_r_tr[:, i], pred_a_tr[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack, Y_train[:, i])
        w = reg.coef_
        weights[name] = w / np.sum(w) if np.sum(w) > 0 else np.array([1 / 3] * 3)

    pred_transrf = np.zeros_like(Y_test, dtype=float)
    for i, name in enumerate(TARGETS):
        stack = np.column_stack([
            pred_target[:, i], pred_residual[:, i], pred_augmented[:, i]
        ])
        pred_transrf[:, i] = stack @ weights[name]

    return {
        'n_train': int(n_train),
        'source_f0': float(r2_score(Y_test[:, 0], pred_source[:, 0])),
        'source_spl': float(r2_score(Y_test[:, 1], pred_source[:, 1])),
        'target_f0': float(r2_score(Y_test[:, 0], pred_target[:, 0])),
        'target_spl': float(r2_score(Y_test[:, 1], pred_target[:, 1])),
        'residual_f0': float(r2_score(Y_test[:, 0], pred_residual[:, 0])),
        'residual_spl': float(r2_score(Y_test[:, 1], pred_residual[:, 1])),
        'augmented_f0': float(r2_score(Y_test[:, 0], pred_augmented[:, 0])),
        'augmented_spl': float(r2_score(Y_test[:, 1], pred_augmented[:, 1])),
        'transrf_f0': float(r2_score(Y_test[:, 0], pred_transrf[:, 0])),
        'transrf_spl': float(r2_score(Y_test[:, 1], pred_transrf[:, 1])),
    }


def main():
    print("=" * 70)
    print("FemaleBCM SMALL-N TRANSFER: Male BCM -> Female BCM (retrain at each N)")
    print("=" * 70)

    df_female = load_female()
    df_male = load_male()
    print(f"  Female (ACFL>{ACFL_THRESHOLD} filter): {len(df_female)} rows")
    print(f"  Male source: {len(df_male)} rows")

    print("\nTraining Male BCM source RF (n_estimators=300)...")
    rf_source, scaler_X_source = fit_source_rf(df_male)
    print("  Done.")

    per_replicate = {}
    for n in N_TARGETS:
        runs = []
        for r in range(N_RUNS):
            res = run_single(n, df_female, rf_source, scaler_X_source,
                             random_state=42 + r)
            if res is not None:
                runs.append(res)
        if not runs:
            continue
        per_replicate[n] = runs
        # Print summary
        avg = {}
        for key in ['target', 'residual', 'augmented', 'transrf']:
            f0 = np.mean([r[f'{key}_f0'] for r in runs])
            spl = np.mean([r[f'{key}_spl'] for r in runs])
            avg[key] = (f0 + spl) / 2
        print(f"  N={n:>4} ({len(runs)} reps) Target={avg['target']:+.3f}  "
              f"Residual={avg['residual']:+.3f}  Augmented={avg['augmented']:+.3f}  "
              f"TransRF={avg['transrf']:+.3f}")

    # Write JSON in TBCM-compatible schema
    payload = {
        '_meta': {
            'domain': 'FemaleBCM',
            'source_domain': 'MaleBCM',
            'n_targets': N_TARGETS,
            'n_runs': N_RUNS,
            'features': SHARED_FEATURES,
            'targets': TARGETS,
            'test_pool_size': TEST_POOL_SIZE,
            'quality_filter': f'ACFL > {ACFL_THRESHOLD}',
            'note': 'Per-N RF transfer (Male BCM -> Female BCM), retrained '
                    'on N target samples each replicate. Mirrors '
                    'TBCM_SmallData.py JSON dump. Generated to replace the '
                    'methodologically flawed evaluation in '
                    'all_regressors_transfer_comparison.csv (which fixed the '
                    'transfer model and varied only the test set size).',
        },
    }
    for method in ['source', 'target', 'residual', 'augmented', 'transrf']:
        block = {}
        for n, runs in sorted(per_replicate.items()):
            block[str(n)] = {
                'r2_f0': [float(r[f'{method}_f0']) for r in runs],
                'r2_spl': [float(r[f'{method}_spl']) for r in runs],
                'n_train': int(runs[0]['n_train']),
            }
        payload[method] = block

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'rf_transfer_small_n.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == '__main__':
    main()
