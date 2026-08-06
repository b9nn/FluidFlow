"""
BCM->BM: TabPFN vs TransRF prediction-surface heatmaps over (a_CT, a_TA),
across small training sizes N. Shows WHERE each method gets the surface right
as N grows, vs the ground-truth surface (RF on all BM data).

Reuses the exact model logic from BM_TransferVsTabPFN_Fair.py so these
surfaces correspond to the R^2 numbers in that comparison:
  - TabPFN: train on N BM points (no source), in-context regression
  - TransRF: BCM source (data_binary.parquet, 360k) + target blend
  - Ground truth: RF trained on all 5000 BM points

Grid: 50x50 sweep of a_CT x a_TA at PS = BM median. One figure per output
(F0, SPL). Rows = [TabPFN, TransRF]; columns = [Ground truth, N=10, 25, 50, 100].

Outputs:
  figs/bm_tabpfn_vs_transrf_heatmap_F0.png
  figs/bm_tabpfn_vs_transrf_heatmap_SPL.png
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import BM_TransferVsTabPFN_Fair as M  # reuse transrf_predict / tabpfn_predict

script_dir = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(script_dir, 'figs')
FEATURES = M.FEATURES          # ['a_CT','a_TA','PS']
TARGETS = M.TARGETS            # ['F0','SPL']
N_SIZES = [10, 25, 50, 100]
GRID_RES = 50
RANDOM_STATE = 42


def load():
    bcm = pd.read_parquet(os.path.join(script_dir, '..', 'data_binary.parquet'))
    if 'Ps' in bcm.columns and 'PS' not in bcm.columns:
        bcm = bcm.rename(columns={'Ps': 'PS'})
    bcm = bcm.dropna(subset=TARGETS)[FEATURES + TARGETS]
    bm = pd.read_csv(os.path.join(script_dir, 'dataset_BM.csv'))
    if 'Ps' in bm.columns and 'PS' not in bm.columns:
        bm = bm.rename(columns={'Ps': 'PS'})
    bm = bm.dropna(subset=TARGETS)[FEATURES + TARGETS]
    return bcm, bm


def build_grid(bm):
    ps = float(bm['PS'].median())
    a_ct = np.linspace(bm['a_CT'].min(), bm['a_CT'].max(), GRID_RES)
    a_ta = np.linspace(bm['a_TA'].min(), bm['a_TA'].max(), GRID_RES)
    TA, CT = np.meshgrid(a_ta, a_ct)               # shape (n_ct, n_ta)
    grid = np.column_stack([CT.ravel(), TA.ravel(), np.full(CT.size, ps)])
    return a_ct, a_ta, grid, ps


def gt_surface(bm, grid):
    sx = StandardScaler().fit(bm[FEATURES])
    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(sx.transform(bm[FEATURES]), bm[TARGETS])
    return rf.predict(sx.transform(grid))          # (n_grid, 2)


def draw(ax, Z, a_ct, a_ta, vmin, vmax, levels, title):
    im = ax.imshow(Z, origin='lower', extent=(a_ta[0], a_ta[-1], a_ct[0], a_ct[-1]),
                   vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
    TA, CT = np.meshgrid(a_ta, a_ct)
    try:
        cs = ax.contour(TA, CT, Z, levels=levels, colors='k', linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=7)
    except Exception:
        pass
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$a_{TA}$', fontsize=11)
    ax.set_ylabel('$a_{CT}$', fontsize=11)
    return im


def main():
    print("=" * 70)
    print(f"BM heatmaps: TabPFN vs TransRF  (backend: {M._TABPFN_BACKEND})")
    print("=" * 70)
    if not M._TABPFN_AVAILABLE:
        print("ERROR: TabPFN not installed.")
        return
    os.makedirs(FIGS_DIR, exist_ok=True)

    bcm, bm = load()
    print(f"  BCM source: {len(bcm)} rows;  BM target: {len(bm)} rows")
    a_ct, a_ta, grid, ps = build_grid(bm)
    print(f"  grid PS = {ps:.0f} (median),  {GRID_RES}x{GRID_RES}")

    # source RF for TransRF
    print("  training BCM source RF (360k)...")
    scaler_X_source = StandardScaler().fit(bcm[FEATURES])
    rf_source = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1))
    rf_source.fit(scaler_X_source.transform(bcm[FEATURES]), bcm[TARGETS])

    print("  ground-truth surface (RF on all BM)...")
    gt = gt_surface(bm, grid)                       # (n_grid, 2)

    # predictions per N (each call returns both outputs)
    tab_by_n, trf_by_n = {}, {}
    for n in N_SIZES:
        print(f"  N={n}: TransRF + TabPFN on grid...")
        tr = bm.sample(n=n, random_state=RANDOM_STATE)
        Xtr, Ytr = tr[FEATURES].values, tr[TARGETS].values
        trf, _ = M.transrf_predict(Xtr, Ytr, grid, rf_source, scaler_X_source, RANDOM_STATE)
        tab = M.tabpfn_predict(Xtr, Ytr, grid, RANDOM_STATE)
        trf_by_n[n] = trf
        tab_by_n[n] = tab

    # one figure per output
    contour_levels = {
        'F0': [150, 200, 250, 300, 350, 400],
        'SPL': [40, 60, 80, 100, 120],
    }
    for oi, out in enumerate(TARGETS):
        Zgt = gt[:, oi].reshape(GRID_RES, GRID_RES)
        # shared color limits from GT (robust to outliers)
        vmin, vmax = np.percentile(Zgt, 2), np.percentile(Zgt, 98)
        lv = contour_levels[out]

        ncols = 1 + len(N_SIZES)
        fig, axs = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.6), squeeze=False)
        fig.suptitle(f'BCM$\\to$BM {out} surface over $(a_{{CT}}, a_{{TA}})$ at PS={ps:.0f}  '
                     f'— TabPFN (top) vs TransRF (bottom)', fontsize=13, fontweight='bold')

        for row, (label, by_n) in enumerate([('TabPFN', tab_by_n), ('TransRF', trf_by_n)]):
            # col 0: ground truth
            im = draw(axs[row, 0], Zgt, a_ct, a_ta, vmin, vmax, lv,
                      f'Ground truth {out}' if row == 0 else f'Ground truth {out}')
            for j, n in enumerate(N_SIZES, start=1):
                Z = by_n[n][:, oi].reshape(GRID_RES, GRID_RES)
                draw(axs[row, j], Z, a_ct, a_ta, vmin, vmax, lv,
                     f'{label}  N={n}')
            axs[row, 0].set_ylabel(f'{label}\n$a_{{CT}}$', fontsize=11)

        fig.subplots_adjust(right=0.92)
        cbar = fig.colorbar(im, ax=axs, shrink=0.6, location='right')
        cbar.set_label(f'{out}', fontsize=11)
        out_path = os.path.join(FIGS_DIR, f'bm_tabpfn_vs_transrf_heatmap_{out}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
