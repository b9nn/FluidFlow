"""
TBCM JASA motor-control maps under sparse data.

Replicates JASA Fig. 6 (Parra et al.) motor-control maps for the TBCM model,
but produced by ML regressors fit on SPARSE data (N = 10 / 100 / 1000),
comparing TabPFN vs TransferRF against the REAL TBCM ODE ground truth.

Faithfulness notes (this task replicates the JASA paper, so it follows the
paper's conventions rather than the repo's default Ps->PS rename):
  * The paper's pressure input is the COMMANDED lung pressure, varied in 50 Pa
    steps -> that is the `PL` column here (NOT the realized `Ps`). Models train
    on [a_CT, a_TA, PL] and the maps are the PL = 1000 Pa (1 kPa) slice, exactly
    as the paper fixed "PS at 1 kPa" for its Fig. 3 / Fig. 6 heatmaps.
  * Ground truth is the actual ODE simulation reshaped onto the (a_CT, a_TA)
    grid at PL = 1000 (1519 rows, 40x40 with ~81 non-converged "holes" left
    blank) -- not a surrogate.
  * Axes follow JASA Figs 3 & 6: a_TA on x, a_CT on y.

Outputs (TBCM/figs/):
  tbcm_motor_map_F0.png         rows N x cols {Ground truth, TabPFN, TransferRF}
  tbcm_motor_map_SPL.png        same layout, SPL
  tbcm_motor_map_F0_error.png   rows N x cols {TabPFN, TransferRF}, signed error
  tbcm_motor_map_SPL_error.png  same layout, SPL
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive; avoids tkinter thread crash on Windows
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(script_dir, 'figs')

N_TARGETS = [10, 100, 1000]            # exactly as specified in the 2026-05-28 meeting
PL_SLICE = 1000                         # 1 kPa -- the paper's fixed-pressure map slice
RANDOM_STATE = 42
FEATURES = ['a_CT', 'a_TA', 'PL']       # paper's pressure input is commanded PL
TARGETS = ['F0', 'SPL']
TABPFN_MAX_TRAIN = 1000

# Iso-output contour levels (JASA-style). F0 adapted to TBCM 80-400 Hz range.
LEVELS = {'F0': [100, 150, 200, 250, 300, 350], 'SPL': [70, 80, 90, 100]}
CMAPS = {'F0': 'viridis', 'SPL': 'magma'}
UNITS = {'F0': 'Hz', 'SPL': 'dB'}


# ==================== ADAPTIVE RF COMPLEXITY ====================
# Mirrors Beam_Membrane/BM_TransferRF.py:51 get_model_params (Convention 4).
def get_model_params(n_samples):
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


# ==================== DATA ====================
def load_tbcm():
    p = os.path.join(script_dir, 'dataset_TBCM.csv')
    df = pd.read_csv(p, index_col=0)
    return df.dropna(subset=TARGETS).reset_index(drop=True)


def load_bcm_source():
    p = os.path.join(script_dir, 'dataset_BCM.csv')
    df = pd.read_csv(p, index_col=0)
    return df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)


def ground_truth_grid(df):
    """Reshape the REAL ODE rows at PL=1000 onto the (a_CT, a_TA) grid.
    Returns AC, AT meshgrids (a_TA varies along axis-1) and masked F0/SPL
    surfaces with NaN at the non-converged 'holes'."""
    sub = df[df['PL'] == PL_SLICE]
    a_ct = np.sort(sub['a_CT'].unique())
    a_ta = np.sort(sub['a_TA'].unique())
    AC, AT = np.meshgrid(a_ct, a_ta, indexing='ij')  # shape (n_ct, n_ta)
    gt = {}
    for t in TARGETS:
        piv = sub.pivot_table(index='a_CT', columns='a_TA', values=t)
        piv = piv.reindex(index=a_ct, columns=a_ta)
        gt[t] = np.ma.masked_invalid(piv.values)
    grid_X = np.column_stack([AC.ravel(), AT.ravel(),
                              np.full(AC.size, float(PL_SLICE))])
    return AC, AT, grid_X, gt, a_ct.size, a_ta.size


# ==================== MODELS ====================
def train_bcm_source(src_df):
    """BCM source RF on [a_CT, a_TA, PL] (reused across all N for TransferRF)."""
    Xs, Ys = src_df[FEATURES].values, src_df[TARGETS].values
    scaler = StandardScaler().fit(Xs)
    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(scaler.transform(Xs), Ys)
    return rf, scaler


def fit_transferrf(rf_src, scaler_src, X_train, Y_train, grid_X, shape):
    """BCM-source RF + target residual RF (adaptive complexity)."""
    pred_src_train = rf_src.predict(scaler_src.transform(X_train))
    params = get_model_params(len(X_train))
    rf_res = MultiOutputRegressor(RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **params))
    rf_res.fit(X_train, Y_train - pred_src_train)
    pred = rf_src.predict(scaler_src.transform(grid_X)) + rf_res.predict(grid_X)
    return {'F0': pred[:, 0].reshape(shape), 'SPL': pred[:, 1].reshape(shape)}


def fit_tabpfn(X_train, Y_train, grid_X, shape):
    """One TabPFN regressor per output, per-output scaled (Convention 1)."""
    if os.environ.get('TABPFN_FORCE_LOCAL'):
        from tabpfn import TabPFNRegressor
    else:
        try:
            from tabpfn_client import TabPFNRegressor
            try:
                from tabpfn_client import set_access_token
                tok = os.environ.get('TABPFN_TOKEN')
                if tok:
                    set_access_token(tok)
            except Exception:
                pass
        except ImportError:
            from tabpfn import TabPFNRegressor

    if X_train.shape[0] > TABPFN_MAX_TRAIN:
        X_train = X_train[:TABPFN_MAX_TRAIN]
        Y_train = Y_train[:TABPFN_MAX_TRAIN]

    scaler_X = StandardScaler().fit(X_train)
    Xs, Gs = scaler_X.transform(X_train), scaler_X.transform(grid_X)
    out = {}
    for i, t in enumerate(TARGETS):
        sc = StandardScaler().fit(Y_train[:, [i]])
        reg = TabPFNRegressor(random_state=RANDOM_STATE)
        reg.fit(Xs, sc.transform(Y_train[:, [i]]).ravel())
        pred = sc.inverse_transform(reg.predict(Gs).reshape(-1, 1)).ravel()
        out[t] = pred.reshape(shape)
    return out


# ==================== PLOTTING (JASA style: surface + iso-contours) ====================
def _panel(ax, AC, AT, surf, cmap, vmin, vmax, levels, title):
    # a_TA on x, a_CT on y (matches JASA Figs 3 & 6)
    im = ax.pcolormesh(AT, AC, surf, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='nearest')
    cs = ax.contour(AT, AC, surf, levels=levels, colors='k', linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=11, fmt='%d')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(r'$a_{TA}$', fontsize=16)
    ax.set_ylabel(r'$a_{CT}$', fontsize=16)
    ax.tick_params(labelsize=13)
    return im


def _finite_minmax(surfs):
    vals = np.concatenate([np.asarray(s)[np.isfinite(np.asarray(s))].ravel()
                           for s in surfs])
    return float(vals.min()), float(vals.max())


def plot_value_matrix(AC, AT, gt, method_surfs, target, ps_label):
    methods = ['TabPFN', 'TransferRF']
    cmap, levels, unit = CMAPS[target], LEVELS[target], UNITS[target]

    surfs = [gt[target]] + [method_surfs[n][m][target] for n in N_TARGETS
                            for m in methods if method_surfs[n][m] is not None]
    vmin, vmax = _finite_minmax(surfs)

    fig, axes = plt.subplots(len(N_TARGETS), 3,
                             figsize=(13, 4.2 * len(N_TARGETS)))
    im = None
    for r, n in enumerate(N_TARGETS):
        im = _panel(axes[r, 0], AC, AT, gt[target], cmap, vmin, vmax, levels,
                    'Ground truth (TBCM ODE)' if r == 0 else '')
        axes[r, 0].set_ylabel(f'N = {n}\n' + r'$a_{CT}$', fontsize=16)
        for c, m in enumerate(methods, start=1):
            surf = method_surfs[n][m]
            ax = axes[r, c]
            if surf is None:
                ax.text(0.5, 0.5, f'{m}\nunavailable', ha='center',
                        va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            _panel(ax, AC, AT, surf[target], cmap, vmin, vmax, levels,
                   m if r == 0 else '')

    fig.suptitle(f'TBCM motor-control map - {target} ({unit})   {ps_label}   '
                 f'sparse-data replication of JASA Fig. 6\n'
                 f'columns: ground truth | TabPFN | TransferRF    '
                 f'rows: N training samples',
                 fontsize=17, fontweight='bold', y=1.0)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label=f'{target} ({unit})')
    cb.set_label(f'{target} ({unit})', fontsize=15)
    cb.ax.tick_params(labelsize=12)
    out = os.path.join(FIGS_DIR, f'tbcm_motor_map_{target}.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def plot_error_matrix(AC, AT, gt, method_surfs, target, ps_label):
    methods = ['TabPFN', 'TransferRF']
    levels = LEVELS[target]
    gt_surf = gt[target]
    valid = ~np.ma.getmaskarray(gt_surf)        # cells where real GT exists

    errs = []
    for n in N_TARGETS:
        for m in methods:
            s = method_surfs[n][m]
            if s is not None:
                e = np.where(valid, s[target] - gt_surf.filled(np.nan), np.nan)
                errs.append(e)
    amax = float(np.nanmax([np.nanmax(np.abs(e)) for e in errs])) if errs else 1.0

    fig, axes = plt.subplots(len(N_TARGETS), 2,
                             figsize=(10, 4.2 * len(N_TARGETS)))
    im = None
    for r, n in enumerate(N_TARGETS):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            surf = method_surfs[n][m]
            if surf is None:
                ax.text(0.5, 0.5, f'{m}\nunavailable', ha='center',
                        va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            err = np.ma.masked_invalid(
                np.where(valid, surf[target] - gt_surf.filled(np.nan), np.nan))
            rmse = float(np.sqrt(np.nanmean(err.filled(np.nan) ** 2)))
            im = ax.pcolormesh(AT, AC, err, cmap='RdBu_r',
                               vmin=-amax, vmax=amax, shading='nearest')
            cs = ax.contour(AT, AC, gt_surf, levels=levels, colors='k',
                            linewidths=0.5, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%d')
            title = f'{m}\nRMSE {rmse:.1f} {UNITS[target]}' if r == 0 \
                else f'{m}  (RMSE {rmse:.1f} {UNITS[target]})'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel(r'$a_{TA}$')
            if c == 0:
                ax.set_ylabel(f'N = {n}\n' + r'$a_{CT}$', fontsize=11)

    fig.suptitle(f'TBCM motor-control map ERROR - {target}   '
                 f'signed (prediction - ground truth), {ps_label}\n'
                 f'GT iso-{target} contours overlaid; blanks = non-converged holes',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.colorbar(im, ax=axes, shrink=0.6,
                 label=f'{target} error ({UNITS[target]})')
    out = os.path.join(FIGS_DIR, f'tbcm_motor_map_{target}_error.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ==================== IMAGE-LEVEL ERROR METRICS ====================
def write_error_metrics(gt, method_surfs):
    """Image-level MAE + RMSE over all valid grid cells, per N x method x target.
    Promotes the per-panel error (already shown on the error maps) to a saved
    quantitative summary: results/motor_map_error_metrics.json + a table figure.
    """
    methods = ['TabPFN', 'TransferRF']
    table = {}
    print("\n  image-level map error (over valid (a_CT,a_TA) grid cells):")
    print(f"    {'target':>6} {'N':>5} {'method':>11} {'MAE':>10} {'RMSE':>10}")
    for target in TARGETS:
        gt_vals = gt[target].filled(np.nan)
        valid = ~np.ma.getmaskarray(gt[target])
        table[target] = {}
        for n in N_TARGETS:
            table[target][str(n)] = {}
            for m in methods:
                s = method_surfs[n].get(m)
                if s is None:
                    continue
                err = (s[target] - gt_vals)[valid]
                err = err[np.isfinite(err)]
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                table[target][str(n)][m] = {'mae': mae, 'rmse': rmse,
                                            'n_cells': int(err.size)}
                print(f"    {target:>6} {n:>5} {m:>11} "
                      f"{mae:>8.2f}{UNITS[target]} {rmse:>8.2f}{UNITS[target]}")

    out = os.path.join(script_dir, 'results', 'motor_map_error_metrics.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"  wrote {out}")

    # Compact table figure: one row per (target, N), cols = MAE/RMSE per method.
    headers = ['Output', 'N', 'MAE TabPFN', 'RMSE TabPFN', 'MAE TransRF', 'RMSE TransRF']
    cell_text = []
    for target in TARGETS:
        u = UNITS[target]
        for n in N_TARGETS:
            rec = table[target][str(n)]
            tb = rec.get('TabPFN', {})
            tf = rec.get('TransferRF', {})
            cell_text.append([
                target, str(n),
                f"{tb.get('mae', float('nan')):.1f} {u}", f"{tb.get('rmse', float('nan')):.1f} {u}",
                f"{tf.get('mae', float('nan')):.1f} {u}", f"{tf.get('rmse', float('nan')):.1f} {u}",
            ])
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.42 * len(cell_text)))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellLoc='center',
                   loc='center', colColours=['#0f172a'] * len(headers))
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.get_text().set_color('white'); cell.get_text().set_fontweight('bold')
        # bold the winner (lower error) between TabPFN/TransRF per metric
    ax.set_title('TBCM motor-map image-level error (MAE / RMSE over all grid cells)\n'
                 'lower is better; TabPFN vs TransRF at N = 10 / 100 / 1000',
                 fontweight='bold', fontsize=12, pad=12)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'tbcm_motor_map_error_table.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


# ==================== MAIN ====================
def main():
    print('=' * 70)
    print('TBCM MOTOR-CONTROL MAPS (real ODE ground truth @ PL=1000)')
    print('=' * 70)
    os.makedirs(FIGS_DIR, exist_ok=True)

    df = load_tbcm()
    src_df = load_bcm_source()
    print(f'  TBCM rows: {len(df)}   BCM source rows: {len(src_df)}')

    AC, AT, grid_X, gt, n_ct, n_ta = ground_truth_grid(df)
    shape = (n_ct, n_ta)
    n_holes = int(np.ma.getmaskarray(gt['F0']).sum())
    ps_label = f'PL = {PL_SLICE} Pa (1 kPa)'
    print(f'  GT grid: {n_ct}x{n_ta} at {ps_label}; {n_holes} non-converged holes')
    print(f'  GT F0 [{gt["F0"].min():.1f}, {gt["F0"].max():.1f}]  '
          f'SPL [{gt["SPL"].min():.1f}, {gt["SPL"].max():.1f}]')

    print('  training BCM source RF (shared across N)...')
    rf_src, scaler_src = train_bcm_source(src_df)

    method_surfs = {}
    for n in N_TARGETS:
        sub = df.sample(n=n, random_state=RANDOM_STATE)  # shared subset (Conv. 3)
        X_train, Y_train = sub[FEATURES].values, sub[TARGETS].values
        method_surfs[n] = {}

        print(f'  N={n:>4}: TransferRF...')
        method_surfs[n]['TransferRF'] = fit_transferrf(
            rf_src, scaler_src, X_train, Y_train, grid_X, shape)

        print(f'  N={n:>4}: TabPFN...')
        try:
            method_surfs[n]['TabPFN'] = fit_tabpfn(X_train, Y_train, grid_X, shape)
        except Exception as e:
            print(f'        TabPFN failed ({e!r}); panel skipped.')
            method_surfs[n]['TabPFN'] = None

    print('  rendering figures...')
    for target in TARGETS:
        plot_value_matrix(AC, AT, gt, method_surfs, target, ps_label)
        plot_error_matrix(AC, AT, gt, method_surfs, target, ps_label)

    write_error_metrics(gt, method_surfs)

    print('=' * 70)
    print('DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
