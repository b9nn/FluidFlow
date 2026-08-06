"""
TBCM TabPFN heatmaps at increasing data fractions.

Notebook-style imshow+contour heatmaps for F0 and SPL showing how TabPFN's
predicted surface evolves from tiny (N=10) to large (N=500) training sets.
Ground truth is an RF trained on all available TBCM data.

Outputs (in figs/):
  tbcm_tabpfn_heatmap_N<N>.svg   — 2x3 per fraction (GT | pred | abs error)
  tbcm_tabpfn_heatmaps_all.svg   — 2x(1+n_fracs) combined view
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
FIGS_DIR = os.path.join(script_dir, 'figs')

FEATURES = ['a_CT', 'a_TA', 'PS']
TARGETS = ['F0', 'SPL']
GRID_RES = 50
RANDOM_STATE = 42

N_FRACTIONS = [10, 50, 200, 500]

LEVELS_F0  = [100, 150, 200, 250, 300, 350]
LEVELS_SPL = [60, 70, 80, 90, 100]
VMIN_F0,  VMAX_F0  = 80,  385
VMIN_SPL, VMAX_SPL = 55,  110


# ==================== DATA ====================

def load_data():
    path = os.path.join(script_dir, 'dataset_TBCM.csv')
    df = pd.read_csv(path, index_col=0)
    if 'Ps' in df.columns and 'PS' not in df.columns:
        df = df.rename(columns={'Ps': 'PS'})
    df = df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    return df[FEATURES + TARGETS]


# ==================== GRID ====================

def build_grid(df):
    ps_value = float(df['PS'].median())
    a_ct = np.linspace(df['a_CT'].min(), df['a_CT'].max(), GRID_RES)
    a_ta = np.linspace(df['a_TA'].min(), df['a_TA'].max(), GRID_RES)
    TA_mesh, CT_mesh = np.meshgrid(a_ta, a_ct)
    grid_X = np.column_stack([CT_mesh.ravel(), TA_mesh.ravel(),
                               np.full(CT_mesh.size, ps_value)])
    return TA_mesh, CT_mesh, grid_X, a_ct, a_ta, ps_value


# ==================== MODELS ====================

def predict_rf_gt(df, grid_X):
    X = df[FEATURES].values
    Y = df[TARGETS].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    grid_sc = scaler.transform(grid_X)
    preds = np.zeros((grid_X.shape[0], 2))
    for i in range(2):
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                   random_state=RANDOM_STATE)
        rf.fit(X_sc, Y[:, i])
        preds[:, i] = rf.predict(grid_sc)
    return preds


def _ensure_tabpfn_auth():
    if _TABPFN_BACKEND == 'tabpfn-client':
        token = os.environ.get('TABPFN_TOKEN')
        if token and _set_tabpfn_token is not None:
            _set_tabpfn_token(token)


def predict_tabpfn(df, n_train, grid_X):
    _ensure_tabpfn_auth()
    df_train = df.sample(n=min(n_train, len(df)), random_state=RANDOM_STATE)
    X_train = df_train[FEATURES].values
    Y_train = df_train[TARGETS].values

    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(X_train)
    grid_sc = scaler_X.transform(grid_X)

    scalers_Y = [StandardScaler(), StandardScaler()]
    Y_sc = np.zeros_like(Y_train, dtype=float)
    for i in range(2):
        Y_sc[:, i] = scalers_Y[i].fit_transform(
            Y_train[:, i].reshape(-1, 1)).ravel()

    preds = np.zeros((grid_X.shape[0], 2))
    for i in range(2):
        reg = _TabPFNRegressor(random_state=RANDOM_STATE)
        reg.fit(X_sc, Y_sc[:, i])
        preds[:, i] = scalers_Y[i].inverse_transform(
            reg.predict(grid_sc).reshape(-1, 1)).ravel()
    return preds


# ==================== PLOTTING ====================

def _draw_panel(ax, Z, a_ct, a_ta, levels, vmin, vmax, title):
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                   extent=(a_ta[0], a_ta[-1], a_ct[0], a_ct[-1]),
                   vmin=vmin, vmax=vmax, aspect='auto')
    TA_mesh, CT_mesh = np.meshgrid(a_ta, a_ct)
    CS = ax.contour(TA_mesh, CT_mesh, Z, levels=levels, colors='k')
    ax.clabel(CS, inline=True, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('$a_{TA}$', fontsize=14)
    ax.set_ylabel('$a_{CT}$', fontsize=14)
    plt.colorbar(im, ax=ax)
    return im


def _draw_error_panel(ax, Z_ref, Z_pred, a_ct, a_ta, title):
    err = np.abs(Z_ref - Z_pred)
    im = ax.imshow(err, interpolation='bilinear', origin='lower',
                   extent=(a_ta[0], a_ta[-1], a_ct[0], a_ct[-1]),
                   aspect='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('$a_{TA}$', fontsize=14)
    ax.set_ylabel('$a_{CT}$', fontsize=14)
    plt.colorbar(im, ax=ax)


def plot_per_n(gt_preds, n, tab_preds, a_ct, a_ta):
    Z_f0_gt  = gt_preds[:, 0].reshape(GRID_RES, GRID_RES)
    Z_spl_gt = gt_preds[:, 1].reshape(GRID_RES, GRID_RES)
    Z_f0     = tab_preds[:, 0].reshape(GRID_RES, GRID_RES)
    Z_spl    = tab_preds[:, 1].reshape(GRID_RES, GRID_RES)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'TBCM — TabPFN at N={n}', fontsize=14, fontweight='bold')

    _draw_panel(axs[0, 0], Z_f0_gt, a_ct, a_ta, LEVELS_F0, VMIN_F0, VMAX_F0,
                'Contour lines for data $f_0$')
    _draw_panel(axs[0, 1], Z_f0, a_ct, a_ta, LEVELS_F0, VMIN_F0, VMAX_F0,
                f'Contour lines for predicted $f_0$ TabPFN (N={n})')
    _draw_error_panel(axs[0, 2], Z_f0_gt, Z_f0, a_ct, a_ta,
                      '$f_0$ absolute error (Hz)')

    _draw_panel(axs[1, 0], Z_spl_gt, a_ct, a_ta, LEVELS_SPL, VMIN_SPL, VMAX_SPL,
                'Contour lines for data SPL')
    _draw_panel(axs[1, 1], Z_spl, a_ct, a_ta, LEVELS_SPL, VMIN_SPL, VMAX_SPL,
                f'Contour lines for predicted SPL TabPFN (N={n})')
    _draw_error_panel(axs[1, 2], Z_spl_gt, Z_spl, a_ct, a_ta,
                      'SPL absolute error (dB)')

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, f'tbcm_tabpfn_heatmap_N{n}.svg')
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')


def plot_combined(gt_preds, tab_preds_list, a_ct, a_ta):
    n_cols = 1 + len(N_FRACTIONS)
    fig, axs = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9), squeeze=False)

    Z_f0_gt  = gt_preds[:, 0].reshape(GRID_RES, GRID_RES)
    Z_spl_gt = gt_preds[:, 1].reshape(GRID_RES, GRID_RES)
    _draw_panel(axs[0, 0], Z_f0_gt, a_ct, a_ta, LEVELS_F0, VMIN_F0, VMAX_F0,
                'Ground truth $f_0$')
    _draw_panel(axs[1, 0], Z_spl_gt, a_ct, a_ta, LEVELS_SPL, VMIN_SPL, VMAX_SPL,
                'Ground truth SPL')

    for col, (n, preds) in enumerate(zip(N_FRACTIONS, tab_preds_list), start=1):
        Z_f0  = preds[:, 0].reshape(GRID_RES, GRID_RES)
        Z_spl = preds[:, 1].reshape(GRID_RES, GRID_RES)
        _draw_panel(axs[0, col], Z_f0, a_ct, a_ta, LEVELS_F0, VMIN_F0, VMAX_F0,
                    f'TabPFN $f_0$ (N={n})')
        _draw_panel(axs[1, col], Z_spl, a_ct, a_ta, LEVELS_SPL, VMIN_SPL, VMAX_SPL,
                    f'TabPFN SPL (N={n})')

    fig.suptitle('TBCM — TabPFN at increasing training set sizes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'tbcm_tabpfn_heatmaps_all.svg')
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')


# ==================== MAIN ====================

def main():
    print('=' * 70)
    print(f'TBCM TabPFN Heatmaps  (backend: {_TABPFN_BACKEND})')
    print('=' * 70)

    if not _TABPFN_AVAILABLE:
        print('\nERROR: TabPFN not installed. pip install tabpfn-client')
        return

    os.makedirs(FIGS_DIR, exist_ok=True)

    print('\nLoading TBCM data...')
    df = load_data()
    print(f'  rows: {len(df)}')
    print(f'  F0:  [{df["F0"].min():.1f}, {df["F0"].max():.1f}] Hz')
    print(f'  SPL: [{df["SPL"].min():.1f}, {df["SPL"].max():.1f}] dB')

    TA_mesh, CT_mesh, grid_X, a_ct, a_ta, ps_val = build_grid(df)
    print(f'  grid PS = {ps_val:.1f} (median),  resolution = {GRID_RES}x{GRID_RES}')

    print('\nBuilding ground truth (RF on all data)...')
    gt_preds = predict_rf_gt(df, grid_X)

    tab_preds_list = []
    for n in N_FRACTIONS:
        print(f'\nFitting TabPFN at N={n}...')
        try:
            preds = predict_tabpfn(df, n, grid_X)
            tab_preds_list.append(preds)
            print(f'  F0 pred range: [{preds[:, 0].min():.1f}, {preds[:, 0].max():.1f}]')
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            tab_preds_list.append(np.full((grid_X.shape[0], 2), np.nan))

    print('\nGenerating per-N figures (2x3 notebook style)...')
    for n, preds in zip(N_FRACTIONS, tab_preds_list):
        plot_per_n(gt_preds, n, preds, a_ct, a_ta)

    print('\nGenerating combined figure...')
    plot_combined(gt_preds, tab_preds_list, a_ct, a_ta)

    print('\nDone.')


if __name__ == '__main__':
    main()
