"""
JASA multidimensional TBCM dataset — TabPFN experiments for the paper.

Driven by the 2026-06-04 advisor meeting (Fathom recap). Three experiments
on Jesus's enriched TBCM data (dataset_JASA.csv, 80,751 rows):

  A. MULTI-DIMENSIONAL FEATURES
     Predict 6 clinical outputs (not just F0/SPL) and track R^2 vs train
     size N for each. Shows TabPFN extends gracefully past the bi-dim case.

  B. MISSING DATA (structured sparsity)
     Hold out the high-activation corner (a_CT > 0.7 AND a_TA > 0.7),
     train only on the rest, evaluate extrapolation INTO the corner.
     Mimics clinical data that never samples extreme muscle activations.
     A same-size random holdout is included as an in-distribution reference.

  C. COMPUTATIONAL TIME vs DATA SIZE
     Wall-clock fit + predict(100 points) time for TabPFN vs RandomForest
     (the prior method) across N. Answers the "measure inference time for
     100 estimations; compare vs prior methods" action item.

Conventions (CLAUDE.md):
  - Inputs fixed to [a_CT, a_TA, PS]; richer columns (a_LCA, PL, ...) dropped.
  - Per-domain input scaler + per-output StandardScalers. Never shared.
  - random_state=42, seeds 42 + run_idx.
  - TabPFN train truncated to TABPFN_MAX_TRAIN (cloud model cap).

Outputs:
  results/jasa_tabpfn_results.json     (all numbers, incremental)
  figs/jasa_multidim_r2.png            (Exp A)
  figs/jasa_missing_data_r2.png        (Exp B, corner vs random)
  figs/jasa_missing_data_heatmap.png   (Exp B, F0 surface + held-out corner)
  figs/jasa_compute_time.png           (Exp C)

Run:
  python JASA_TabPFN_Experiments.py            # all three
  python JASA_TabPFN_Experiments.py --exp A    # one experiment (A/B/C)
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ---- TabPFN backend selection: prefer cloud client, fall back to local ----
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
RESULTS_DIR = os.path.join(script_dir, 'results')
FIGS_DIR = os.path.join(script_dir, 'figs')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'jasa_tabpfn_results.json')

FEATURES = ['a_CT', 'a_TA', 'PS']
OUTPUTS = ['F0', 'SPL', 'ACFL', 'PC', 'MFDR', 'CPP']
OUTPUT_LABELS = {
    'F0': 'F0 (Hz)', 'SPL': 'SPL (dB)', 'ACFL': 'AC flow',
    'PC': 'collision pressure', 'MFDR': 'MFDR', 'CPP': 'CPP',
}

RANDOM_STATE = 42
TABPFN_MAX_TRAIN = 1000

# Corner holdout definition (Exp B)
CORNER_CT = 0.7
CORNER_TA = 0.7

# N grids. Train size never exceeds 1000 (small-data regime + TabPFN cap);
# the full 80k dataset is only a pool to draw train/test samples from.
N_GRID_ACCURACY = [10, 50, 100, 200, 500, 1000]
N_RUNS_ACCURACY = 2
N_GRID_TIME = [10, 50, 100, 200, 500, 1000]
N_RUNS_TIME = 2
TEST_POOL_SIZE = 300
PREDICT_BATCH = 100  # "100 estimations" from the action item


# ==================== DATA ====================

def load_data():
    path = os.path.join(script_dir, 'dataset_JASA.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset_JASA.csv not found at {path}")
    df = pd.read_csv(path)
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    needed = FEATURES + OUTPUTS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"JASA CSV missing columns: {missing}")
    return df.dropna(subset=needed).reset_index(drop=True)[needed]


# ==================== SHARED MODEL HELPERS ====================

def _ensure_tabpfn_auth():
    if _TABPFN_BACKEND == 'tabpfn-client':
        token = os.environ.get('TABPFN_TOKEN')
        if token and _set_tabpfn_token is not None:
            _set_tabpfn_token(token)


def scale_train_test(df_train, df_test, outputs):
    """Per-domain input scaler + per-output StandardScalers. Returns scaled
    arrays and the fitted per-output scalers (for inverse-transforming preds)."""
    X_train = df_train[FEATURES].values
    X_test = df_test[FEATURES].values
    scaler_X = StandardScaler().fit(X_train)
    Xtr = scaler_X.transform(X_train)
    Xte = scaler_X.transform(X_test)

    Ytr = np.zeros((len(df_train), len(outputs)))
    scalers_Y = []
    for j, o in enumerate(outputs):
        s = StandardScaler().fit(df_train[o].values.reshape(-1, 1))
        Ytr[:, j] = s.transform(df_train[o].values.reshape(-1, 1)).ravel()
        scalers_Y.append(s)
    return Xtr, Xte, Ytr, scalers_Y


def tabpfn_fit_predict(Xtr, ytr_scaled, Xte, scaler_y, random_state):
    """One TabPFN regressor for one (scaled) output. Returns preds in
    ORIGINAL units."""
    if Xtr.shape[0] > TABPFN_MAX_TRAIN:
        Xtr = Xtr[:TABPFN_MAX_TRAIN]
        ytr_scaled = ytr_scaled[:TABPFN_MAX_TRAIN]
    reg = _TabPFNRegressor(random_state=random_state)
    reg.fit(Xtr, ytr_scaled)
    pred_sc = reg.predict(Xte)
    return scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


# ==================== RESULTS I/O ====================

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


# ==================== EXPERIMENT A: MULTI-DIMENSIONAL ====================

def experiment_A(df, results):
    print("\n" + "=" * 70)
    print("EXPERIMENT A — Multi-dimensional features (R^2 vs N)")
    print("=" * 70)
    block = {o: {'N': [], 'mean': [], 'std': []} for o in OUTPUTS}

    for n in N_GRID_ACCURACY:
        per_feat_runs = {o: [] for o in OUTPUTS}
        for r in range(N_RUNS_ACCURACY):
            seed = RANDOM_STATE + r
            df_train = df.sample(n=min(n, len(df)), random_state=seed)
            test_pool = df.drop(df_train.index)
            df_test = test_pool.sample(
                n=min(TEST_POOL_SIZE, len(test_pool)), random_state=seed)
            Xtr, Xte, Ytr, scalers_Y = scale_train_test(df_train, df_test, OUTPUTS)
            for j, o in enumerate(OUTPUTS):
                preds = tabpfn_fit_predict(Xtr, Ytr[:, j], Xte, scalers_Y[j], seed)
                per_feat_runs[o].append(r2_score(df_test[o].values, preds))
        line = [f"N={n:>4}"]
        for o in OUTPUTS:
            arr = np.array(per_feat_runs[o])
            block[o]['N'].append(n)
            block[o]['mean'].append(float(arr.mean()))
            block[o]['std'].append(float(arr.std()))
            line.append(f"{o}={arr.mean():+.3f}")
        print("  " + "  ".join(line))
        results['experiment_A'] = block
        save_results(results)

    _plot_A(block)
    return block


def _plot_A(block):
    fig, ax = plt.subplots(figsize=(9, 6))
    for o in OUTPUTS:
        N = np.array(block[o]['N'])
        m = np.array(block[o]['mean'])
        s = np.array(block[o]['std'])
        ax.plot(N, m, marker='o', label=OUTPUT_LABELS[o])
        ax.fill_between(N, m - s, m + s, alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel('Training samples (N)')
    ax.set_ylabel('Test $R^2$')
    ax.set_title('TabPFN on JASA TBCM — multi-dimensional output, $R^2$ vs N')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_ylim(-0.05, 1.02)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(FIGS_DIR, 'jasa_multidim_r2.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


# ==================== EXPERIMENT B: MISSING DATA ====================

def experiment_B(df, results):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT B — Missing data (corner a_CT>{CORNER_CT} & a_TA>{CORNER_TA})")
    print("=" * 70)

    corner_mask = (df['a_CT'] > CORNER_CT) & (df['a_TA'] > CORNER_TA)
    df_corner = df[corner_mask]
    df_rest = df[~corner_mask]
    print(f"  corner (held out): {len(df_corner)} rows;  train pool: {len(df_rest)} rows")

    # Fixed corner test set
    corner_test = df_corner.sample(
        n=min(TEST_POOL_SIZE, len(df_corner)), random_state=RANDOM_STATE)

    block = {
        'corner': {o: {'N': [], 'mean': [], 'std': []} for o in OUTPUTS},
        'random': {o: {'N': [], 'mean': [], 'std': []} for o in OUTPUTS},
    }

    for n in N_GRID_ACCURACY:
        corner_runs = {o: [] for o in OUTPUTS}
        random_runs = {o: [] for o in OUTPUTS}
        for r in range(N_RUNS_ACCURACY):
            seed = RANDOM_STATE + r

            # --- corner extrapolation: train on rest, test on corner ---
            df_train = df_rest.sample(n=min(n, len(df_rest)), random_state=seed)
            Xtr, Xte, Ytr, scalers_Y = scale_train_test(df_train, corner_test, OUTPUTS)
            for j, o in enumerate(OUTPUTS):
                preds = tabpfn_fit_predict(Xtr, Ytr[:, j], Xte, scalers_Y[j], seed)
                corner_runs[o].append(r2_score(corner_test[o].values, preds))

            # --- random in-distribution reference (same test size) ---
            df_tr2 = df.sample(n=min(n, len(df)), random_state=seed + 1000)
            pool2 = df.drop(df_tr2.index)
            df_te2 = pool2.sample(
                n=min(len(corner_test), len(pool2)), random_state=seed + 1000)
            Xtr2, Xte2, Ytr2, scalersY2 = scale_train_test(df_tr2, df_te2, OUTPUTS)
            for j, o in enumerate(OUTPUTS):
                preds = tabpfn_fit_predict(Xtr2, Ytr2[:, j], Xte2, scalersY2[j], seed)
                random_runs[o].append(r2_score(df_te2[o].values, preds))

        line = [f"N={n:>4}"]
        for o in OUTPUTS:
            ca, ra = np.array(corner_runs[o]), np.array(random_runs[o])
            block['corner'][o]['N'].append(n)
            block['corner'][o]['mean'].append(float(ca.mean()))
            block['corner'][o]['std'].append(float(ca.std()))
            block['random'][o]['N'].append(n)
            block['random'][o]['mean'].append(float(ra.mean()))
            block['random'][o]['std'].append(float(ra.std()))
            line.append(f"{o}:cor={ca.mean():+.2f}/rnd={ra.mean():+.2f}")
        print("  " + " ".join(line))
        results['experiment_B'] = block
        save_results(results)

    _plot_B(block)
    _plot_B_heatmap(df, df_rest, corner_mask)
    return block


def _plot_B(block):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, o in zip(axes.ravel(), OUTPUTS):
        for cond, ls, color in [('corner', '-', 'C3'), ('random', '--', 'C0')]:
            N = np.array(block[cond][o]['N'])
            m = np.array(block[cond][o]['mean'])
            s = np.array(block[cond][o]['std'])
            ax.plot(N, m, marker='o', ls=ls, color=color,
                    label=('corner (extrapolate)' if cond == 'corner'
                           else 'random (in-dist.)'))
            ax.fill_between(N, m - s, m + s, alpha=0.15, color=color)
        ax.set_xscale('log')
        ax.set_title(OUTPUT_LABELS[o])
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_ylim(-0.5, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel('Training samples (N)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Test $R^2$')
    fig.suptitle(f'TabPFN missing-data: extrapolate into held-out corner '
                 f'(a_CT>{CORNER_CT} & a_TA>{CORNER_TA}) vs random holdout',
                 fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'jasa_missing_data_r2.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


def _plot_B_heatmap(df, df_rest, corner_mask):
    """F0 surface from a model trained on the non-corner region, with the
    held-out corner outlined — visualizes extrapolation quality."""
    GRID = 50
    ps_val = float(df['PS'].median())
    a_ct = np.linspace(df['a_CT'].min(), df['a_CT'].max(), GRID)
    a_ta = np.linspace(df['a_TA'].min(), df['a_TA'].max(), GRID)
    TA, CT = np.meshgrid(a_ta, a_ct)
    grid_X = np.column_stack([CT.ravel(), TA.ravel(), np.full(CT.size, ps_val)])

    df_train = df_rest.sample(n=min(1000, len(df_rest)), random_state=RANDOM_STATE)
    scaler_X = StandardScaler().fit(df_train[FEATURES].values)
    sy = StandardScaler().fit(df_train['F0'].values.reshape(-1, 1))
    Xtr = scaler_X.transform(df_train[FEATURES].values)
    ytr = sy.transform(df_train['F0'].values.reshape(-1, 1)).ravel()
    grid_sc = scaler_X.transform(grid_X)

    try:
        reg = _TabPFNRegressor(random_state=RANDOM_STATE)
        reg.fit(Xtr, ytr)
        pred = sy.inverse_transform(reg.predict(grid_sc).reshape(-1, 1)).ravel()
    except Exception as e:
        print(f"  heatmap skipped (TabPFN failed: {e})")
        return
    Z = pred.reshape(GRID, GRID)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, origin='lower', extent=(a_ta[0], a_ta[-1], a_ct[0], a_ct[-1]),
                   aspect='auto', interpolation='bilinear')
    # outline the held-out corner
    ax.axhline(CORNER_CT, color='red', lw=1.5, ls='--')
    ax.axvline(CORNER_TA, color='red', lw=1.5, ls='--')
    ax.add_patch(plt.Rectangle((CORNER_TA, CORNER_CT),
                               a_ta[-1] - CORNER_TA, a_ct[-1] - CORNER_CT,
                               fill=False, edgecolor='red', lw=2.5, hatch='//'))
    ax.text(CORNER_TA + 0.02, CORNER_CT + 0.02, 'held-out\ncorner',
            color='red', fontsize=10, fontweight='bold', va='bottom')
    ax.set_xlabel('$a_{TA}$', fontsize=13)
    ax.set_ylabel('$a_{CT}$', fontsize=13)
    ax.set_title('F0 surface — TabPFN trained OUTSIDE the corner\n'
                 f'(N=1000, PS={ps_val:.0f}); red box = extrapolation region')
    plt.colorbar(im, ax=ax, label='F0 (Hz)')
    out = os.path.join(FIGS_DIR, 'jasa_missing_data_heatmap.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


# ==================== EXPERIMENT C: COMPUTE TIME ====================

def experiment_C(df, results):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT C — Compute time vs N (predict {PREDICT_BATCH} points)")
    print("=" * 70)
    print("  NOTE: tabpfn-client times include network latency to the cloud API.")

    block = {
        'tabpfn': {'N': [], 'fit_s': [], 'predict_s': []},
        'rf':     {'N': [], 'fit_s': [], 'predict_s': []},
    }
    target = 'F0'  # single output for clean timing

    for n in N_GRID_TIME:
        tp_fit, tp_pred, rf_fit, rf_pred = [], [], [], []
        for r in range(N_RUNS_TIME):
            seed = RANDOM_STATE + r
            df_train = df.sample(n=min(n, len(df)), random_state=seed)
            pool = df.drop(df_train.index)
            df_test = pool.sample(n=PREDICT_BATCH, random_state=seed)

            scaler_X = StandardScaler().fit(df_train[FEATURES].values)
            sy = StandardScaler().fit(df_train[target].values.reshape(-1, 1))
            Xtr = scaler_X.transform(df_train[FEATURES].values)
            ytr = sy.transform(df_train[target].values.reshape(-1, 1)).ravel()
            Xte = scaler_X.transform(df_test[FEATURES].values)

            # RandomForest (prior method)
            t0 = time.perf_counter()
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(Xtr, ytr)
            t1 = time.perf_counter()
            rf.predict(Xte)
            t2 = time.perf_counter()
            rf_fit.append(t1 - t0)
            rf_pred.append(t2 - t1)

            # TabPFN
            ntr = min(n, TABPFN_MAX_TRAIN)
            t0 = time.perf_counter()
            reg = _TabPFNRegressor(random_state=seed)
            reg.fit(Xtr[:ntr], ytr[:ntr])
            t1 = time.perf_counter()
            reg.predict(Xte)
            t2 = time.perf_counter()
            tp_fit.append(t1 - t0)
            tp_pred.append(t2 - t1)

        block['tabpfn']['N'].append(n)
        block['tabpfn']['fit_s'].append(float(np.mean(tp_fit)))
        block['tabpfn']['predict_s'].append(float(np.mean(tp_pred)))
        block['rf']['N'].append(n)
        block['rf']['fit_s'].append(float(np.mean(rf_fit)))
        block['rf']['predict_s'].append(float(np.mean(rf_pred)))
        print(f"  N={n:>4}  TabPFN fit={np.mean(tp_fit):6.2f}s "
              f"pred={np.mean(tp_pred):6.2f}s | "
              f"RF fit={np.mean(rf_fit):6.3f}s pred={np.mean(rf_pred):6.4f}s")
        results['experiment_C'] = block
        save_results(results)

    _plot_C(block)
    return block


def _plot_C(block):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, title in [(axes[0], 'fit_s', 'Fit time'),
                           (axes[1], 'predict_s',
                            f'Predict time ({PREDICT_BATCH} points)')]:
        for method, color, label in [('tabpfn', 'C1', 'TabPFN (cloud)'),
                                     ('rf', 'C2', 'RandomForest')]:
            N = np.array(block[method]['N'])
            t = np.array(block[method][key])
            ax.plot(N, t, marker='o', color=color, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Training samples (N)')
        ax.set_ylabel('Wall-clock seconds')
        ax.set_title(title)
        ax.grid(alpha=0.3, which='both')
        ax.legend()
    fig.suptitle('TabPFN vs RandomForest — compute time vs data size '
                 '(JASA TBCM)', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'jasa_compute_time.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['A', 'B', 'C', 'all'], default='all')
    args = parser.parse_args()

    print("=" * 70)
    print(f"JASA TabPFN experiments  (backend: {_TABPFN_BACKEND})")
    print("=" * 70)
    if not _TABPFN_AVAILABLE:
        print("ERROR: TabPFN not installed. pip install tabpfn-client")
        return
    _ensure_tabpfn_auth()
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()
    print(f"  rows: {len(df)}   inputs: {FEATURES}   outputs: {OUTPUTS}")

    results = load_results()
    results['_meta'] = {
        'dataset': 'dataset_JASA.csv', 'n_rows': len(df),
        'features': FEATURES, 'outputs': OUTPUTS,
        'corner': {'a_CT>': CORNER_CT, 'a_TA>': CORNER_TA},
        'tabpfn_max_train': TABPFN_MAX_TRAIN, 'backend': _TABPFN_BACKEND,
    }
    save_results(results)

    if args.exp in ('A', 'all'):
        experiment_A(df, results)
    if args.exp in ('B', 'all'):
        experiment_B(df, results)
    if args.exp in ('C', 'all'):
        experiment_C(df, results)

    print(f"\nDone. Results: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
