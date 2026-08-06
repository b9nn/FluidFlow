"""
Muscle-activation x F0 heatmaps comparing GP, TabPFN, and best transfer.
Answers Jesus's 2026-05-12 advisor-meeting question: do the new methods
replicate the nonlinear (a_CT, a_TA) -> F0 trend, not just hit high R^2?

For each domain (BM, Female BCM; TBCM if data is present):
  - Train GP and TabPFN at N=50 with random_state=42 (the small-N regime
    where alternates' advantage is most visible for BM).
  - Train a lightweight TransRF baseline at the same N using BCM source +
    target residual + augmented features (mirrors BM_TransferRF method).
  - Sweep (a_CT, a_TA) on a 50x50 grid at PS = dataset median.
  - Predict F0; render heatmaps side by side.
  - Overlay the 50 training points as scatter, colored by their true F0.
  - Mismatch of scatter-color vs surface-color = method fails the
    nonlinear-trend check (per Jesus's design).

Outputs: Beam_Membrane/figs/heatmap_<domain>_F0.png
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids tkinter thread crash on Windows
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS_DIR = os.path.join(os.path.dirname(__file__), 'figs')
GRID_RES = 50
N_TRAIN = 50
RANDOM_STATE = 42
TARGET = 'F0'


def fit_gp_one_target(X_train_sc, y_train_sc):
    kernel = (C(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=3, random_state=42)
    gp.fit(X_train_sc, y_train_sc)
    return gp


def fit_tabpfn_one_target(X_train_sc, y_train_sc):
    try:
        from tabpfn_client import TabPFNRegressor
    except ImportError:
        from tabpfn import TabPFNRegressor
    r = TabPFNRegressor(random_state=42)
    r.fit(X_train_sc, y_train_sc)
    return r


def build_grid(df, ps_value):
    a_ct = np.linspace(df['a_CT'].min(), df['a_CT'].max(), GRID_RES)
    a_ta = np.linspace(df['a_TA'].min(), df['a_TA'].max(), GRID_RES)
    AC, AT = np.meshgrid(a_ct, a_ta)
    grid_X = np.column_stack([AC.ravel(), AT.ravel(),
                              np.full(AC.size, ps_value)])
    return AC, AT, grid_X


def fit_transrf(source_df_path, X_train, y_train, grid_X):
    """Lightweight TransRF: BCM source + target residual + augmented.
    Predicts F0 only (one-target). Returns grid predictions in original scale.
    Source training is the bottleneck (~5-10 sec for ~50k samples).
    """
    src = pd.read_csv(source_df_path)
    if 'Ps' in src.columns and 'PS' not in src.columns:
        src = src.rename(columns={'Ps': 'PS'})
    src = src.dropna(subset=['F0', 'SPL'])[['a_CT', 'a_TA', 'PS', 'F0']]
    X_src = src[['a_CT', 'a_TA', 'PS']].values
    y_src = src['F0'].values

    scaler_X_src = StandardScaler().fit(X_src)
    rf_src = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1)
    rf_src.fit(scaler_X_src.transform(X_src), y_src)

    pred_src_train = rf_src.predict(scaler_X_src.transform(X_train))
    residual = y_train - pred_src_train
    rf_res = RandomForestRegressor(n_estimators=20, max_depth=3,
                                    min_samples_leaf=5, random_state=42)
    rf_res.fit(X_train, residual)

    pred_src_grid = rf_src.predict(scaler_X_src.transform(grid_X))
    pred_res_grid = rf_res.predict(grid_X)
    return pred_src_grid + pred_res_grid


def heatmap_for_domain(df, name, source_df_path, transfer_label):
    df = df.sample(n=min(len(df), 2000), random_state=RANDOM_STATE)
    df_train = df.sample(n=N_TRAIN, random_state=RANDOM_STATE)
    X_train = df_train[['a_CT', 'a_TA', 'PS']].values
    y_train = df_train[TARGET].values

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train_sc = scaler_X.transform(X_train)
    y_train_sc = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    ps_value = float(df['PS'].median())
    AC, AT, grid_X = build_grid(df, ps_value)
    grid_sc = scaler_X.transform(grid_X)

    print(f'  [{name}] fitting GP...')
    gp = fit_gp_one_target(X_train_sc, y_train_sc)
    pred_gp = scaler_y.inverse_transform(
        gp.predict(grid_sc).reshape(-1, 1)).ravel()

    print(f'  [{name}] fitting TabPFN...')
    try:
        tab = fit_tabpfn_one_target(X_train_sc, y_train_sc)
        pred_tab = scaler_y.inverse_transform(
            tab.predict(grid_sc).reshape(-1, 1)).ravel()
        have_tab = True
    except Exception as e:
        print(f'  [{name}] TabPFN failed ({e!r}); panel skipped.')
        pred_tab = None
        have_tab = False

    print(f'  [{name}] fitting TransRF (source: {os.path.basename(source_df_path)})...')
    try:
        pred_xfer = fit_transrf(source_df_path, X_train, y_train, grid_X)
        have_xfer = True
    except Exception as e:
        print(f'  [{name}] transfer baseline failed ({e!r}); panel skipped.')
        pred_xfer = None
        have_xfer = False

    panels = [('GP', pred_gp.reshape(GRID_RES, GRID_RES))]
    if have_tab:
        panels.append(('TabPFN', pred_tab.reshape(GRID_RES, GRID_RES)))
    if have_xfer:
        panels.append((transfer_label, pred_xfer.reshape(GRID_RES, GRID_RES)))

    # Shared color limits across panels for visual comparability.
    vmin = min(p.min() for _, p in panels)
    vmax = max(p.max() for _, p in panels)
    vmin = min(vmin, float(y_train.min()))
    vmax = max(vmax, float(y_train.max()))

    fig, axes = plt.subplots(1, len(panels),
                              figsize=(4.8 * len(panels), 4.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, surf) in zip(axes, panels):
        im = ax.pcolormesh(AC, AT, surf, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.scatter(df_train['a_CT'], df_train['a_TA'],
                   c=df_train[TARGET], cmap='viridis',
                   vmin=vmin, vmax=vmax, edgecolor='white', s=40, lw=0.8)
        ax.set_xlabel('a_CT')
        ax.set_title(f'{name}: {label}\n(PS={ps_value:.0f})',
                     fontweight='bold', fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('a_TA')
    fig.colorbar(im, ax=axes, label='F0 (Hz)', shrink=0.85)
    fig.suptitle(f'F0 surface -- {name}, N={N_TRAIN} training points '
                 '(scatter = true F0)',
                 fontweight='bold', y=1.02)

    out_path = os.path.join(FIGS_DIR, f'heatmap_{name}_F0.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    print('=' * 70)
    print('CROSS-DOMAIN F0 HEATMAPS')
    print('=' * 70)
    os.makedirs(FIGS_DIR, exist_ok=True)

    male_bcm_path = os.path.join(ROOT, 'VocalFoldRegression', 'BCM Model',
                                  'MaleBCM.csv')

    # BM
    bm_path = os.path.join(ROOT, 'Beam_Membrane', 'dataset_BM.csv')
    if os.path.exists(bm_path):
        bm = pd.read_csv(bm_path)
        if 'Ps' in bm.columns and 'PS' not in bm.columns:
            bm = bm.rename(columns={'Ps': 'PS'})
        bm = bm.dropna(subset=['F0', 'SPL'])[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']]
        heatmap_for_domain(bm, 'BM', male_bcm_path,
                           transfer_label='BCM->BM TransRF (lightweight)')
    else:
        print('  BM dataset not found -- skipping')

    # TBCM (gated on data availability)
    tbcm_path = os.path.join(ROOT, 'TBCM', 'dataset_TBCM.csv')
    if os.path.exists(tbcm_path):
        tbcm = pd.read_csv(tbcm_path, index_col=0)
        if 'Ps' in tbcm.columns and 'PS' not in tbcm.columns:
            tbcm = tbcm.rename(columns={'Ps': 'PS'})
        tbcm = tbcm.dropna(subset=['F0', 'SPL'])[
            ['a_CT', 'a_TA', 'PS', 'F0', 'SPL']]
        heatmap_for_domain(tbcm, 'TBCM', male_bcm_path,
                           transfer_label='BCM->TBCM TransRF (lightweight)')
    else:
        print('  TBCM dataset not found -- skipping (gated on data arrival)')

    # Female BCM
    fem_path = os.path.join(ROOT, 'VocalFoldRegression', 'BCM Model',
                             'FemaleBCM.csv')
    if os.path.exists(fem_path):
        fem = pd.read_csv(fem_path)
        if 'Ps' in fem.columns and 'PS' not in fem.columns:
            fem = fem.rename(columns={'Ps': 'PS'})
        if 'ACFL' in fem.columns:
            fem = fem[fem['ACFL'] > 30]
        fem = fem.dropna(subset=['F0', 'SPL'])[
            ['a_CT', 'a_TA', 'PS', 'F0', 'SPL']]
        heatmap_for_domain(fem, 'FemaleBCM', male_bcm_path,
                           transfer_label='Male->Female RF transfer (lightweight)')
    else:
        print('  Female BCM dataset not found -- skipping')


if __name__ == '__main__':
    main()
