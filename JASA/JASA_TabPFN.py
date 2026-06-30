"""
Component 1 — TabPFN regressors on JASA data: singles vs joint multi-head.

Five single-target TabPFN regressors (F0, SPL, MFDR, ACFL, PC) versus one
joint "multi-head" TabPFN that predicts all five as a single continuous output.
The multi-head is built by z-scoring each target, stacking the five targets into
one long y-column, and tagging each block with a one-hot output-id feature, so a
single TabPFNRegressor learns all outputs at once (no separate model per target).

No models are trained beyond TabPFN's in-context fit — every regressor is a
TabPFN wrapper. Mirrors the harness/scaling in TBCM/TBCM_TabPFN.py.

Backend: `tabpfn-client` (cloud), reusing the cached access token — same backend
the prior TBCM/BM TabPFN scripts used.

Outputs:
  - results/tabpfn_multihead.json   per-N, per-target R2 for singles + multihead
  - figs/jasa_multihead_vs_single.png
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

from tabpfn_client import TabPFNRegressor  # cloud backend (cached token)


script_dir = os.path.dirname(os.path.abspath(__file__))

FEATURES = ['a_CT', 'a_TA', 'a_LCA', 'PS']
TARGETS = ['F0', 'SPL', 'MFDR', 'ACFL', 'PC']
# 5xN stacked train for the multi-head must stay <= TabPFN's 1000-row cap, so the
# core comparison grid stops at N=200 (5x200=1000).
N_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200]
N_RUNS = 5
TEST_POOL_SIZE = 1000


def load_jasa():
    p = os.path.join(script_dir, 'dataset_JASA.parquet')
    df = pd.read_parquet(p) if os.path.exists(p) else pd.read_csv(
        os.path.join(script_dir, 'dataset_JASA.csv'))
    needed = FEATURES + TARGETS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"JASA CSV missing columns: {missing}. Has: {list(df.columns)}")
    return df.dropna(subset=needed).reset_index(drop=True)[needed]


def split_train_test(df, n_target, random_state):
    """Sample n_target training rows; hold out a disjoint test pool."""
    df_train = df.sample(n=n_target, random_state=random_state)
    test_pool = df.drop(df_train.index)
    df_test = test_pool.sample(
        n=min(TEST_POOL_SIZE, len(test_pool)), random_state=random_state)
    return df_train, df_test


def scale_inputs(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def scale_targets(Y_train):
    """Per-target StandardScaler list + z-scored training targets."""
    scalers = [StandardScaler() for _ in range(Y_train.shape[1])]
    Y_sc = np.zeros_like(Y_train, dtype=float)
    for i in range(Y_train.shape[1]):
        Y_sc[:, i] = scalers[i].fit_transform(Y_train[:, i].reshape(-1, 1)).ravel()
    return scalers, Y_sc


def predict_single(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, random_state):
    """One TabPFNRegressor per target (independent models)."""
    out_dim = Y_train_sc.shape[1]
    preds = np.zeros((X_test_sc.shape[0], out_dim))
    for i in range(out_dim):
        reg = TabPFNRegressor(random_state=random_state)
        reg.fit(X_train_sc, Y_train_sc[:, i])
        pred_sc = reg.predict(X_test_sc)
        preds[:, i] = scalers_Y[i].inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    return preds


def predict_multihead(X_train_sc, Y_train_sc, X_test_sc, scalers_Y, random_state):
    """One TabPFN over stacked [X | one-hot(output_id)] -> z-scored target value."""
    n_train, n_feat = X_train_sc.shape
    out_dim = Y_train_sc.shape[1]
    eye = np.eye(out_dim)

    # Stack: for each target i, all training rows tagged with output-id i.
    X_stack = np.vstack([
        np.hstack([X_train_sc, np.tile(eye[i], (n_train, 1))])
        for i in range(out_dim)
    ])
    y_stack = np.concatenate([Y_train_sc[:, i] for i in range(out_dim)])

    reg = TabPFNRegressor(random_state=random_state)
    reg.fit(X_stack, y_stack)

    n_test = X_test_sc.shape[0]
    preds = np.zeros((n_test, out_dim))
    for i in range(out_dim):
        X_test_aug = np.hstack([X_test_sc, np.tile(eye[i], (n_test, 1))])
        pred_sc = reg.predict(X_test_aug)
        preds[:, i] = scalers_Y[i].inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    return preds


def run_single_replicate(df, n_target, random_state):
    df_train, df_test = split_train_test(df, n_target, random_state)
    X_train_sc, X_test_sc = scale_inputs(df_train[FEATURES].values,
                                         df_test[FEATURES].values)
    Y_train = df_train[TARGETS].values
    Y_test = df_test[TARGETS].values
    scalers_Y, Y_train_sc = scale_targets(Y_train)

    preds_single = predict_single(X_train_sc, Y_train_sc, X_test_sc, scalers_Y,
                                  random_state)
    preds_multi = predict_multihead(X_train_sc, Y_train_sc, X_test_sc, scalers_Y,
                                    random_state)

    r2_single = {t: float(r2_score(Y_test[:, i], preds_single[:, i]))
                 for i, t in enumerate(TARGETS)}
    r2_multi = {t: float(r2_score(Y_test[:, i], preds_multi[:, i]))
                for i, t in enumerate(TARGETS)}
    return r2_single, r2_multi


OUT_PATH = os.path.join(script_dir, 'results', 'tabpfn_multihead.json')


def _save(results):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    meta = {'features': FEATURES, 'targets': TARGETS, 'n_grid': N_GRID,
            'n_runs': N_RUNS, 'test_pool': TEST_POOL_SIZE}
    with open(OUT_PATH, 'w') as f:
        json.dump({'_meta': meta, **results}, f, indent=2)


def _load_existing():
    """Resume support: reload prior results and report which N are complete."""
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


def _replicate_with_retry(df, n, random_state, max_tries=5):
    """Survive transient cloud/network drops: retry a replicate with backoff."""
    import time
    for attempt in range(max_tries):
        try:
            return run_single_replicate(df, n, random_state)
        except Exception as e:
            if attempt == max_tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"    replicate N={n} rs={random_state} failed ({type(e).__name__}); "
                  f"retry in {wait}s")
            time.sleep(wait)


def main():
    print("=" * 70)
    print("JASA Component 1 — TabPFN singles vs multi-head (tabpfn-client)")
    print("=" * 70)
    df = load_jasa()
    print(f"  rows: {len(df)}   features: {FEATURES}   targets: {TARGETS}")

    results = _load_existing()

    for n in N_GRID:
        # Skip N already complete for every target (resume after a network drop).
        done = all(len(results['single'][t].get(str(n), [])) >= N_RUNS
                   and len(results['multihead'][t].get(str(n), [])) >= N_RUNS
                   for t in TARGETS)
        if done:
            print(f"  N={n:>4}  skip (already complete)")
            continue

        # Seed per-N lists from any previously saved partial runs.
        single_runs = {t: list(results['single'][t].get(str(n), [])) for t in TARGETS}
        multi_runs = {t: list(results['multihead'][t].get(str(n), [])) for t in TARGETS}
        start_r = min(len(single_runs[t]) for t in TARGETS)
        for r in range(start_r, N_RUNS):
            r2_s, r2_m = _replicate_with_retry(df, n, 42 + r)
            for t in TARGETS:
                single_runs[t].append(r2_s[t])
                multi_runs[t].append(r2_m[t])
            # Checkpoint after each replicate so a later failure doesn't lose work.
            for t in TARGETS:
                results['single'][t][str(n)] = single_runs[t]
                results['multihead'][t][str(n)] = multi_runs[t]
            _save(results)
        s_avg = np.mean([np.mean(single_runs[t]) for t in TARGETS])
        m_avg = np.mean([np.mean(multi_runs[t]) for t in TARGETS])
        print(f"  N={n:>4}  single avg R2 = {s_avg:+.3f}   multihead avg R2 = {m_avg:+.3f}")

    print(f"\nResults written to: {OUT_PATH}")
    plot(results)


def plot(results):
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    colors = {'F0': '#d97706', 'SPL': '#0f766e', 'MFDR': '#7c3aed',
              'ACFL': '#dc2626', 'PC': '#2563eb'}

    fig, ax = plt.subplots(figsize=(11, 6.8))

    def curve(block, target):
        ns = sorted(int(k) for k in block[target].keys())
        means = [float(np.mean(block[target][str(n)])) for n in ns]
        return ns, means

    for t in TARGETS:
        ns, ys_s = curve(results['single'], t)
        _, ys_m = curve(results['multihead'], t)
        ax.plot(ns, ys_s, color=colors[t], marker='o', ls='-', lw=2.2, ms=6,
                label=f'{t} (single)')
        ax.plot(ns, ys_m, color=colors[t], marker='s', ls='--', lw=1.8, ms=5,
                alpha=0.85, label=f'{t} (multi-head)')

    # Bold black 5-target average pair.
    ns = sorted(int(k) for k in results['single']['F0'].keys())
    avg_s = [float(np.mean([np.mean(results['single'][t][str(n)]) for t in TARGETS]))
             for n in ns]
    avg_m = [float(np.mean([np.mean(results['multihead'][t][str(n)]) for t in TARGETS]))
             for n in ns]
    ax.plot(ns, avg_s, color='black', marker='o', ls='-', lw=3.0, ms=8,
            label='AVG (single)', zorder=6)
    ax.plot(ns, avg_m, color='black', marker='s', ls='--', lw=2.6, ms=7,
            label='AVG (multi-head)', zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('Number of training samples (log scale)')
    ax.set_ylabel('R²  (per target, on original scale)')
    ax.set_title('JASA: single TabPFN per output vs one joint multi-head TabPFN',
                 fontweight='bold')
    ax.axhline(0, color='#000', lw=0.6, alpha=0.4)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_ylim(top=1.02)
    ax.legend(loc='lower right', ncol=2, framealpha=0.95, fontsize=8)

    fig.tight_layout()
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, 'jasa_multihead_vs_single.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    main()
