"""
BM extended — Component 1: TabPFN singles vs joint multi-head on Beam-Membrane.

Mirrors JASA/JASA_TabPFN.py but on the BM extended dataset. Five single-target
TabPFN regressors (F0, SPL, ACFL, PC, CPP) vs one joint multi-head TabPFN
(z-scored targets stacked with a one-hot output-id). Reports BOTH R2 and
normalized RMSE (range-normalized on the test pool) per output, per N, mean over
seeds.

Target: dataset_BM_extended.csv (filtered ACFL > 30 -> phonating rows).
Features: a_CT, a_TA, a_LCA, PS.   Backend: tabpfn-client (cloud, cached token).

Outputs:
  - results/bm_ext_tabpfn.json   per-N per-target {r2, nrmse} for single + multihead
  - figs/bm_ext_multihead_vs_single.png
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
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from tabpfn_client import TabPFNRegressor

script_dir = os.path.dirname(os.path.abspath(__file__))

FEATURES = ['a_CT', 'a_TA', 'a_LCA', 'PS']
TARGETS = ['F0', 'SPL', 'ACFL', 'PC', 'CPP']
# Plotted grid matches the rest of the paper's comparison figures (GRID_MATCH
# in TBCM/Female/BM showcases). Single-output TabPFN runs the full grid; the
# multi-head stacks 5xN rows so it is capped at N<=200 (5x200 = TabPFN's 1000-row
# in-context limit).
N_GRID = [10, 20, 50, 100, 200, 500]
GRID_MATCH = (10, 20, 50, 100, 200, 500)
MULTIHEAD_MAX_N = 200
N_RUNS = 5
TEST_POOL_SIZE = 1000
ACFL_FILTER = 30.0

OUT_PATH = os.path.join(script_dir, 'results', 'bm_ext_tabpfn.json')


def load_bm():
    p = os.path.join(script_dir, 'dataset_BM_extended.csv')
    df = pd.read_csv(p).rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=FEATURES + TARGETS)
    df = df[df['ACFL'] > ACFL_FILTER].reset_index(drop=True)
    return df[FEATURES + TARGETS]


def _nrmse(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse / rng if rng > 0 else float('nan')


def scale_targets(Y_train):
    scalers = [StandardScaler() for _ in range(Y_train.shape[1])]
    Y_sc = np.zeros_like(Y_train, dtype=float)
    for i in range(Y_train.shape[1]):
        Y_sc[:, i] = scalers[i].fit_transform(Y_train[:, i].reshape(-1, 1)).ravel()
    return scalers, Y_sc


def predict_single(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, rs):
    out_dim = Y_train_sc.shape[1]
    preds = np.zeros((X_test_sc.shape[0], out_dim))
    for i in range(out_dim):
        reg = TabPFNRegressor(random_state=rs)
        reg.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = reg.predict(X_test_sc)
        preds[:, i] = scalers_Y[i].inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    return preds


def predict_multihead(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, rs):
    n_train = X_train_sc.shape[0]
    out_dim = Y_train_sc.shape[1]
    eye = np.eye(out_dim)
    X_stack = np.vstack([np.hstack([X_train_sc, np.tile(eye[i], (n_train, 1))])
                         for i in range(out_dim)])
    y_stack = np.concatenate([Y_train_sc[:, i] for i in range(out_dim)])
    reg = TabPFNRegressor(random_state=rs)
    reg.fit(X_stack, y_stack)
    n_test = X_test_sc.shape[0]
    preds = np.zeros((n_test, out_dim))
    for i in range(out_dim):
        X_test_aug = np.hstack([X_test_sc, np.tile(eye[i], (n_test, 1))])
        pred_sc = reg.predict(X_test_aug)
        preds[:, i] = scalers_Y[i].inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    return preds


def run_single_replicate(df, n_target, rs, do_multi=True):
    df_train = df.sample(n=n_target, random_state=rs)
    test_pool = df.drop(df_train.index)
    df_test = test_pool.sample(n=min(TEST_POOL_SIZE, len(test_pool)), random_state=rs)

    sc_X = StandardScaler()
    X_train_sc = sc_X.fit_transform(df_train[FEATURES].values)
    X_test_sc = sc_X.transform(df_test[FEATURES].values)
    Y_train = df_train[TARGETS].values
    Y_test = df_test[TARGETS].values
    scalers_Y, Y_train_sc = scale_targets(Y_train)

    preds_s = predict_single(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, rs)

    def metrics(preds):
        return {t: {'r2': float(r2_score(Y_test[:, i], preds[:, i])),
                    'nrmse': _nrmse(Y_test[:, i], preds[:, i])}
                for i, t in enumerate(TARGETS)}
    if not do_multi:
        return metrics(preds_s), None
    preds_m = predict_multihead(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, rs)
    return metrics(preds_s), metrics(preds_m)


def _save(results):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    meta = {'features': FEATURES, 'targets': TARGETS, 'n_grid': N_GRID,
            'n_runs': N_RUNS, 'test_pool': TEST_POOL_SIZE, 'acfl_filter': ACFL_FILTER}
    with open(OUT_PATH, 'w') as f:
        json.dump({'_meta': meta, **results}, f, indent=2)


def _load_existing():
    results = {'single': {t: {} for t in TARGETS},
               'multihead': {t: {} for t in TARGETS}}
    if not os.path.exists(OUT_PATH):
        return results
    with open(OUT_PATH) as f:
        prior = json.load(f)
    for kind in ('single', 'multihead'):
        for t in TARGETS:
            results[kind][t] = prior.get(kind, {}).get(t, {})
    return results


def _n_done(results, n):
    """Min completed replicates across all targets for N (resume support)."""
    counts = []
    for t in TARGETS:
        rec = results['single'][t].get(str(n), {})
        counts.append(len(rec.get('r2', [])))
    return min(counts) if counts else 0


def _replicate_with_retry(df, n, rs, do_multi=True, max_tries=5):
    import time
    for attempt in range(max_tries):
        try:
            return run_single_replicate(df, n, rs, do_multi=do_multi)
        except Exception as e:
            if attempt == max_tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"    replicate N={n} rs={rs} failed ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)


def _append(results, kind, t, n, metric_dict):
    rec = results[kind][t].setdefault(str(n), {'r2': [], 'nrmse': []})
    rec['r2'].append(metric_dict['r2'])
    rec['nrmse'].append(metric_dict['nrmse'])


def main():
    print("=" * 70)
    print("BM EXTENDED Component 1 — TabPFN singles vs multi-head (tabpfn-client)")
    print("=" * 70)
    df = load_bm()
    print(f"  rows (ACFL>{ACFL_FILTER:.0f}): {len(df)}  features: {FEATURES}  targets: {TARGETS}")

    results = _load_existing()
    for n in N_GRID:
        start_r = _n_done(results, n)
        if start_r >= N_RUNS:
            print(f"  N={n:>4}  skip (complete)")
            continue
        do_multi = n <= MULTIHEAD_MAX_N
        for r in range(start_r, N_RUNS):
            m_s, m_m = _replicate_with_retry(df, n, 42 + r, do_multi=do_multi)
            for t in TARGETS:
                _append(results, 'single', t, n, m_s[t])
                if m_m is not None:
                    _append(results, 'multihead', t, n, m_m[t])
            _save(results)
        s_avg = np.mean([np.mean(results['single'][t][str(n)]['r2']) for t in TARGETS])
        if do_multi:
            m_avg = np.mean([np.mean(results['multihead'][t][str(n)]['r2']) for t in TARGETS])
            print(f"  N={n:>4}  single avg R2 = {s_avg:+.3f}   multihead avg R2 = {m_avg:+.3f}")
        else:
            print(f"  N={n:>4}  single avg R2 = {s_avg:+.3f}   (multi-head skipped, N>{MULTIHEAD_MAX_N})")

    print(f"\nResults written to: {OUT_PATH}")
    plot(results)


def plot(results):
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                         'legend.fontsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = {'F0': '#d97706', 'SPL': '#0f766e', 'ACFL': '#dc2626',
              'PC': '#2563eb', 'CPP': '#7c3aed'}
    fig, ax = plt.subplots(figsize=(11, 6.8))

    def curve(block, t):
        ns = sorted(int(k) for k in block[t].keys() if int(k) in GRID_MATCH)
        return ns, [float(np.mean(block[t][str(n)]['r2'])) for n in ns]

    for t in TARGETS:
        ns_s, ys_s = curve(results['single'], t)
        ns_m, ys_m = curve(results['multihead'], t)
        ax.plot(ns_s, ys_s, color=colors[t], marker='o', ls='-', lw=2.2, ms=6, label=f'{t} (single)')
        ax.plot(ns_m, ys_m, color=colors[t], marker='s', ls='--', lw=1.8, ms=5, alpha=0.85,
                label=f'{t} (multi-head)')
    ns_s = sorted(int(k) for k in results['single']['F0'].keys() if int(k) in GRID_MATCH)
    ns_m = sorted(int(k) for k in results['multihead']['F0'].keys() if int(k) in GRID_MATCH)
    avg_s = [float(np.mean([np.mean(results['single'][t][str(n)]['r2']) for t in TARGETS])) for n in ns_s]
    avg_m = [float(np.mean([np.mean(results['multihead'][t][str(n)]['r2']) for t in TARGETS])) for n in ns_m]
    ax.plot(ns_s, avg_s, color='black', marker='o', ls='-', lw=3.0, ms=8, label='AVG (single)', zorder=6)
    ax.plot(ns_m, avg_m, color='black', marker='s', ls='--', lw=2.6, ms=7, label='AVG (multi-head)', zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('Number of BM training samples (log scale)')
    ax.set_ylabel('R²  (per output, original scale)')
    ax.set_title('BM: single TabPFN per output vs one joint multi-head TabPFN', fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_ylim(top=1.02)
    ax.legend(loc='lower right', ncol=2, framealpha=0.95, fontsize=8)
    fig.tight_layout()
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'bm_ext_multihead_vs_single.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    main()
