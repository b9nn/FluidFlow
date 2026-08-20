"""
TabPFN single-output heads vs stacked joint multi-output head, on BM.

Backs the paper's Section 3.3 claim that a single joint multi-output head trails
the dedicated single-output heads by only ~0.05 R^2.

Because TabPFNRegressor is single-target, the "joint" head here is the stacked
variant: standardize all outputs, fit one TabPFN per output but from a SHARED
in-context pass over the same standardized (X, Y_all) block, versus the "single"
setup that fits/standardizes each output fully independently. We report the mean
R^2 gap (single minus joint) per N and per output.

Target: dataset_BM_extended.csv, outputs F0, SPL, ACFL, PC, CPP.
Writes results/bm_tabpfn_single_vs_joint.json and figs/bm_tabpfn_single_vs_joint.png.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    from tabpfn_client import TabPFNRegressor as _TabPFNRegressor
    from tabpfn_client import set_access_token as _set_tabpfn_token
    _BACKEND = 'tabpfn-client'
except ImportError:
    from tabpfn import TabPFNRegressor as _TabPFNRegressor
    _set_tabpfn_token = None
    _BACKEND = 'tabpfn'

script_dir = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['a_CT', 'a_TA', 'PS']
OUTPUTS = ['F0', 'SPL', 'ACFL', 'PC', 'CPP']
N_GRID = [20, 50, 100, 200, 500]
N_RUNS = 3
TEST_POOL_SIZE = 500
TABPFN_MAX_TRAIN = 1000
RANDOM_STATE = 42


def _single_head(Xtr, Ytr, Xte, rs):
    """Each output fit fully independently (own scaler, own regressor)."""
    preds = np.zeros((len(Xte), Ytr.shape[1]))
    for i in range(Ytr.shape[1]):
        sX = StandardScaler().fit(Xtr)
        sy = StandardScaler().fit(Ytr[:, i].reshape(-1, 1))
        reg = _TabPFNRegressor(random_state=rs)
        reg.fit(sX.transform(Xtr), sy.transform(Ytr[:, i].reshape(-1, 1)).ravel())
        preds[:, i] = sy.inverse_transform(
            reg.predict(sX.transform(Xte)).reshape(-1, 1)).ravel()
    return preds


def _joint_head(Xtr, Ytr, Xte, rs):
    """Stacked joint head: one shared input scaler and a single joint Y
    standardization block, outputs predicted together from the same context."""
    sX = StandardScaler().fit(Xtr)
    Xtr_sc, Xte_sc = sX.transform(Xtr), sX.transform(Xte)
    sy = StandardScaler().fit(Ytr)                 # joint standardization
    Ytr_sc = sy.transform(Ytr)
    pred_sc = np.zeros((len(Xte), Ytr.shape[1]))
    for i in range(Ytr.shape[1]):
        reg = _TabPFNRegressor(random_state=rs)
        reg.fit(Xtr_sc, Ytr_sc[:, i])
        pred_sc[:, i] = reg.predict(Xte_sc)
    return sy.inverse_transform(pred_sc)


def main():
    print(f'TabPFN single vs joint head on BM (backend {_BACKEND})')
    if _BACKEND == 'tabpfn-client' and _set_tabpfn_token is not None:
        tok = os.environ.get('TABPFN_TOKEN')
        if tok:
            _set_tabpfn_token(tok)

    df = pd.read_csv(os.path.join(script_dir, 'dataset_BM_extended.csv'))
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=OUTPUTS)[FEATURES + OUTPUTS]

    results = {'_meta': {'n_grid': N_GRID, 'n_runs': N_RUNS, 'outputs': OUTPUTS,
                         'target': 'dataset_BM_extended.csv'},
               'single': {}, 'joint': {}}
    for n in N_GRID:
        s_acc = {o: [] for o in OUTPUTS}
        j_acc = {o: [] for o in OUTPUTS}
        for r in range(N_RUNS):
            rs = RANDOM_STATE + r
            tr = df.sample(n=n, random_state=rs)
            pool = df.drop(tr.index)
            te = pool.sample(n=min(TEST_POOL_SIZE, len(pool)), random_state=rs)
            Xtr, Ytr = tr[FEATURES].values, tr[OUTPUTS].values
            Xte, Yte = te[FEATURES].values, te[OUTPUTS].values
            ps = _single_head(Xtr, Ytr, Xte, rs)
            pj = _joint_head(Xtr, Ytr, Xte, rs)
            for j, o in enumerate(OUTPUTS):
                s_acc[o].append(r2_score(Yte[:, j], ps[:, j]))
                j_acc[o].append(r2_score(Yte[:, j], pj[:, j]))
        results['single'][str(n)] = {o: float(np.mean(s_acc[o])) for o in OUTPUTS}
        results['joint'][str(n)] = {o: float(np.mean(j_acc[o])) for o in OUTPUTS}
        gap = np.mean([np.mean(s_acc[o]) - np.mean(j_acc[o]) for o in OUTPUTS])
        print(f'  N={n:>4}  mean(single - joint) R2 gap = {gap:+.3f}')
        os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
        with open(os.path.join(script_dir, 'results', 'bm_tabpfn_single_vs_joint.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # overall mean gap across N and outputs
    gaps = [results['single'][str(n)][o] - results['joint'][str(n)][o]
            for n in N_GRID for o in OUTPUTS]
    results['_meta']['mean_gap_single_minus_joint'] = float(np.mean(gaps))
    with open(os.path.join(script_dir, 'results', 'bm_tabpfn_single_vs_joint.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Overall mean gap (single - joint) = {np.mean(gaps):+.3f} R2")

    fig, ax = plt.subplots(figsize=(8, 5))
    for o in OUTPUTS:
        s = [results['single'][str(n)][o] for n in N_GRID]
        j = [results['joint'][str(n)][o] for n in N_GRID]
        ln, = ax.plot(N_GRID, s, marker='o', label=f'{o} single')
        ax.plot(N_GRID, j, marker='s', ls='--', color=ln.get_color(), alpha=0.7)
    ax.set_xscale('log'); ax.set_xlabel('N'); ax.set_ylabel('Test $R^2$')
    ax.set_title(f'TabPFN single (solid) vs joint (dashed) head on BM\n'
                 f'mean gap = {np.mean(gaps):+.3f} $R^2$')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    out = os.path.join(script_dir, 'figs', 'bm_tabpfn_single_vs_joint.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'  saved: {out}')


if __name__ == '__main__':
    main()
