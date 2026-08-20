"""
BCM -> BM: four-method low-N comparison across five outputs.

Fills the paper's Table 1 / Fig 1 for the four-method framing:
  PR transfer  : source polynomial + target residual correction
  RF transfer  : source RF + target/residual/augmented blend (optimized variant)
  NN transfer  : source MLP pretrained, target fine-tuned (early layers frozen)
  TabPFN       : no source, in-context fit (per-output)

Target : dataset_BM_extended.csv (5000 rows, 5 outputs)
Source : data_binary.parquet (360,750-row BCM), used by the three transfer methods
Outputs: F0, SPL, ACFL, PC, CPP   (ACFL/CPP carry the known source-comparability
         caveat; generated per the agreed scope, flagged in the paper text)

Protocol (matches BM_TransferVsTabPFN_Fair.py): for each N in N_GRID, train on N
target rows, evaluate on a disjoint 500-row held-out pool, over N_RUNS seeds.
Reports R^2 and normalized RMSE per output per method.

Writes results/bm_five_method.json incrementally and figs/bm_five_method_r2.png.
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn

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
OUTPUTS = ['F0', 'SPL', 'ACFL', 'PC', 'CPP']
N_GRID = [10, 20, 50, 100, 200, 500]
N_RUNS = 3
TEST_POOL_SIZE = 500
TABPFN_MAX_TRAIN = 1000
RANDOM_STATE = 42

SRC_SUBSAMPLE_NN = 20000   # subsample source for NN pretrain (speed)
DEVICE = torch.device('cpu')


# ==================== adaptive RF params (matches prior scripts) ====================
def get_model_params(n):
    if n < 100:
        return {'n_estimators': 20, 'max_depth': 3,
                'min_samples_leaf': max(10, n // 10),
                'min_samples_split': max(5, n // 20)}
    elif n < 250:
        return {'n_estimators': 30, 'max_depth': 5,
                'min_samples_leaf': 5, 'min_samples_split': 5}
    else:
        return {'n_estimators': 50, 'max_depth': 8,
                'min_samples_leaf': 3, 'min_samples_split': 3}


# ==================== source models (built once) ====================
class SourceModels:
    """Holds the per-domain (BCM source) pretrained pieces for each transfer method."""
    def __init__(self, df_src):
        self.scaler_X = StandardScaler().fit(df_src[FEATURES].values)
        Xs = self.scaler_X.transform(df_src[FEATURES].values)
        Ys = df_src[OUTPUTS].values

        # RF source (multi-output)
        print('  [source] RF...')
        self.rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        self.rf.fit(Xs, Ys)

        # Polynomial source (degree 3 + Ridge), per output via multi-output
        print('  [source] polynomial...')
        self.poly = PolynomialFeatures(degree=3, include_bias=False)
        Xs_poly = self.poly.fit_transform(Xs)
        self.pr = MultiOutputRegressor(Ridge(alpha=1.0))
        self.pr.fit(Xs_poly, Ys)

        # NN source: standardize Y too (per-output), pretrain a shared-trunk MLP
        print(f'  [source] MLP pretrain (subsample {SRC_SUBSAMPLE_NN})...')
        rng = np.random.RandomState(42)
        idx = rng.choice(len(Xs), min(SRC_SUBSAMPLE_NN, len(Xs)), replace=False)
        self.scalers_Y = [StandardScaler().fit(Ys[:, k].reshape(-1, 1))
                          for k in range(len(OUTPUTS))]
        Ys_sc = np.column_stack([self.scalers_Y[k].transform(Ys[:, k].reshape(-1, 1)).ravel()
                                 for k in range(len(OUTPUTS))])
        self.nn = _make_mlp(len(FEATURES), len(OUTPUTS))
        _train_mlp(self.nn, Xs[idx], Ys_sc[idx], epochs=120, lr=1e-3)


def _make_mlp(idim, odim):
    return nn.Sequential(
        nn.Linear(idim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, odim),
    ).to(DEVICE)


def _train_mlp(model, X, Y, epochs, lr, freeze_trunk=False):
    if freeze_trunk:
        # freeze all but the last Linear layer
        for p in model.parameters():
            p.requires_grad = False
        for p in list(model.children())[-1].parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    Xt = torch.FloatTensor(X).to(DEVICE)
    Yt = torch.FloatTensor(Y).to(DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = mse(model(Xt), Yt)
        loss.backward()
        opt.step()
    model.eval()


# ==================== per-method predictors ====================
def predict_rf(src, X_train, Y_train, X_test):
    """RF transfer: target/residual/augmented blend with positive weights."""
    params = get_model_params(len(X_train))
    pred_src_test = src.rf.predict(src.scaler_X.transform(X_test))
    pred_src_train = src.rf.predict(src.scaler_X.transform(X_train))

    sX = StandardScaler().fit(X_train)
    Xtr, Xte = sX.transform(X_train), sX.transform(X_test)

    rf_t = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1, **params))
    rf_t.fit(Xtr, Y_train)
    p_t = rf_t.predict(Xte)

    rf_r = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1, **params))
    rf_r.fit(X_train, Y_train - pred_src_train)
    p_r = pred_src_test + rf_r.predict(X_test)

    Xtr_a = np.column_stack([X_train, pred_src_train])
    Xte_a = np.column_stack([X_test, pred_src_test])
    rf_a = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1, **params))
    rf_a.fit(Xtr_a, Y_train)
    p_a = rf_a.predict(Xte_a)

    p_t_tr = rf_t.predict(Xtr)
    p_r_tr = pred_src_train + rf_r.predict(X_train)
    p_a_tr = rf_a.predict(Xtr_a)
    out = np.zeros_like(p_t)
    for i in range(Y_train.shape[1]):
        stack_tr = np.column_stack([p_t_tr[:, i], p_r_tr[:, i], p_a_tr[:, i]])
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(stack_tr, Y_train[:, i])
        w = reg.coef_
        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
        out[:, i] = np.column_stack([p_t[:, i], p_r[:, i], p_a[:, i]]) @ w
    return out


def predict_pr(src, X_train, Y_train, X_test):
    """Polynomial transfer: source poly prediction + target residual poly."""
    Xtr_sc = src.scaler_X.transform(X_train)
    Xte_sc = src.scaler_X.transform(X_test)
    pred_src_train = src.pr.predict(src.poly.transform(Xtr_sc))
    pred_src_test = src.pr.predict(src.poly.transform(Xte_sc))

    # target residual model: low-degree poly (degree 2) + Ridge on the residual
    poly_t = PolynomialFeatures(degree=2, include_bias=False)
    sXt = StandardScaler().fit(X_train)
    Ptr = poly_t.fit_transform(sXt.transform(X_train))
    Pte = poly_t.transform(sXt.transform(X_test))
    res = Y_train - pred_src_train
    ridge = MultiOutputRegressor(Ridge(alpha=10.0))
    ridge.fit(Ptr, res)
    return pred_src_test + ridge.predict(Pte)


def predict_nn(src, X_train, Y_train, X_test):
    """NN transfer: clone pretrained source MLP, fine-tune last layer on target."""
    import copy
    model = copy.deepcopy(src.nn)
    Xtr = src.scaler_X.transform(X_train)
    Xte = src.scaler_X.transform(X_test)
    Ytr_sc = np.column_stack([src.scalers_Y[k].transform(Y_train[:, k].reshape(-1, 1)).ravel()
                              for k in range(Y_train.shape[1])])
    # Full-network fine-tune at low LR (warm-started from the source pretrain).
    # Freezing the trunk was too restrictive on the far BM domain; unfreezing lets
    # the network adapt away from the misaligned source features.
    epochs = 200 if len(X_train) < 100 else 150
    _train_mlp(model, Xtr, Ytr_sc, epochs=epochs, lr=3e-4, freeze_trunk=False)
    with torch.no_grad():
        pred_sc = model(torch.FloatTensor(Xte).to(DEVICE)).cpu().numpy()
    return np.column_stack([src.scalers_Y[k].inverse_transform(pred_sc[:, k].reshape(-1, 1)).ravel()
                            for k in range(pred_sc.shape[1])])


def predict_tabpfn(X_train, Y_train, X_test, rs):
    Xtr, Ytr = X_train[:TABPFN_MAX_TRAIN], Y_train[:TABPFN_MAX_TRAIN]
    sX = StandardScaler().fit(Xtr)
    Xtr_sc, Xte_sc = sX.transform(Xtr), sX.transform(X_test)
    preds = np.zeros((len(X_test), Y_train.shape[1]))
    for i in range(Y_train.shape[1]):
        sy = StandardScaler().fit(Ytr[:, i].reshape(-1, 1))
        reg = _TabPFNRegressor(random_state=rs)
        reg.fit(Xtr_sc, sy.transform(Ytr[:, i].reshape(-1, 1)).ravel())
        preds[:, i] = sy.inverse_transform(reg.predict(Xte_sc).reshape(-1, 1)).ravel()
    return preds


# ==================== harness ====================
def nrmse(y_true, y_pred):
    rng = y_true.max() - y_true.min()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / rng) if rng > 0 else np.nan


def main():
    print('=' * 70)
    print(f'BM four-method comparison (backend: {_TABPFN_BACKEND})')
    print('=' * 70)
    if _TABPFN_BACKEND == 'tabpfn-client' and _set_tabpfn_token is not None:
        tok = os.environ.get('TABPFN_TOKEN')
        if tok:
            _set_tabpfn_token(tok)

    df_src = pd.read_parquet(os.path.join(script_dir, '..', 'data_binary.parquet'))
    if 'Ps' in df_src.columns and 'PS' not in df_src.columns:
        df_src = df_src.rename(columns={'Ps': 'PS'})
    df_src = df_src.dropna(subset=OUTPUTS)[FEATURES + OUTPUTS]
    df_bm = pd.read_csv(os.path.join(script_dir, 'dataset_BM_extended.csv'))
    if 'Ps' in df_bm.columns and 'PS' not in df_bm.columns:
        df_bm = df_bm.rename(columns={'Ps': 'PS'})
    df_bm = df_bm.dropna(subset=OUTPUTS)[FEATURES + OUTPUTS]
    print(f'  source {len(df_src)} rows, target {len(df_bm)} rows, outputs {OUTPUTS}')

    print('\nBuilding source models (once)...')
    src = SourceModels(df_src)

    methods = {'PR': predict_pr, 'RF': predict_rf, 'NN': predict_nn}
    results = {'_meta': {'n_grid': N_GRID, 'n_runs': N_RUNS, 'outputs': OUTPUTS,
                         'target': 'dataset_BM_extended.csv', 'source': 'data_binary.parquet',
                         'test_pool_size': TEST_POOL_SIZE, 'metric': 'R2 + nRMSE'},
               'r2': {}, 'nrmse': {}}
    all_methods = list(methods.keys()) + (['TabPFN'] if _TABPFN_AVAILABLE else [])

    for n in N_GRID:
        r2_acc = {m: {o: [] for o in OUTPUTS} for m in all_methods}
        nr_acc = {m: {o: [] for o in OUTPUTS} for m in all_methods}
        for r in range(N_RUNS):
            rs = RANDOM_STATE + r
            df_tr = df_bm.sample(n=n, random_state=rs)
            pool = df_bm.drop(df_tr.index)
            df_te = pool.sample(n=min(TEST_POOL_SIZE, len(pool)), random_state=rs)
            Xtr, Ytr = df_tr[FEATURES].values, df_tr[OUTPUTS].values
            Xte, Yte = df_te[FEATURES].values, df_te[OUTPUTS].values

            for m, fn in methods.items():
                pred = fn(src, Xtr, Ytr, Xte)
                for j, o in enumerate(OUTPUTS):
                    r2_acc[m][o].append(r2_score(Yte[:, j], pred[:, j]))
                    nr_acc[m][o].append(nrmse(Yte[:, j], pred[:, j]))
            if _TABPFN_AVAILABLE:
                pred = predict_tabpfn(Xtr, Ytr, Xte, rs)
                for j, o in enumerate(OUTPUTS):
                    r2_acc['TabPFN'][o].append(r2_score(Yte[:, j], pred[:, j]))
                    nr_acc['TabPFN'][o].append(nrmse(Yte[:, j], pred[:, j]))

        for m in all_methods:
            results['r2'].setdefault(m, {})[str(n)] = {
                o: {'mean': float(np.mean(r2_acc[m][o])), 'std': float(np.std(r2_acc[m][o]))}
                for o in OUTPUTS}
            results['nrmse'].setdefault(m, {})[str(n)] = {
                o: {'mean': float(np.nanmean(nr_acc[m][o])), 'std': float(np.nanstd(nr_acc[m][o]))}
                for o in OUTPUTS}
        line = ' | '.join(f"{m} F0={np.mean(r2_acc[m]['F0']):+.2f}" for m in all_methods)
        print(f'  N={n:>4}  {line}')
        os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
        with open(os.path.join(script_dir, 'results', 'bm_five_method.json'), 'w') as f:
            json.dump(results, f, indent=2)

    _plot(results, all_methods)
    print('\nDone.')


def _plot(results, methods):
    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(4.2 * len(OUTPUTS), 4.5), sharey=True)
    colors = {'PR': 'C4', 'RF': 'C3', 'NN': 'C1', 'TabPFN': 'C0'}
    for ax, o in zip(axes, OUTPUTS):
        for m in methods:
            N = [int(k) for k in results['r2'][m].keys()]
            mean = [results['r2'][m][str(n)][o]['mean'] for n in N]
            ax.plot(N, mean, marker='o', label=m, color=colors.get(m))
        ax.set_xscale('log'); ax.set_title(o); ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_ylim(-0.5, 1.02); ax.grid(alpha=0.3); ax.set_xlabel('N')
    axes[0].set_ylabel('Test $R^2$'); axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('BCM$\\to$BM: four-method low-N comparison (R$^2$ vs N)', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(script_dir, 'figs', 'bm_five_method_r2.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'  saved: {out}')


if __name__ == '__main__':
    main()
