"""
JASA TBCM — single-axis structured-missing experiment.

Extends Experiment B (corner holdout) by sweeping HOW MUCH of one muscle
activation axis is withheld, separately for a_CT and a_TA.

For each axis and each missing fraction f in {5, 10, 25, 50}%:
  - Hold out the top-f fraction of rows by that axis value (the highest
    activations) as the TEST region.
  - Train TabPFN on N=1000 samples drawn ONLY from the kept (lower) region.
  - Evaluate R^2 on the held-out slice, for all 6 clinical outputs.
  - The other axis is left untouched (only one muscle's high range is missing).

Larger f => the model must extrapolate further past its training boundary.

Reuses helpers from JASA_TabPFN_Experiments.py (data loader, per-output
scaling, TabPFN fit/predict). In-distribution R^2 at N=1000 from Experiment A
is overlaid as a dotted ceiling per output, when available.

Design choices (see DECISIONS 2026-06-16 / -06-18):
  - HIGH end held out (consistent with the high-corner experiment).
  - "% missing" = fraction of ROWS by axis value (so 50% = top half).

Outputs:
  results/jasa_axis_missing_results.json
  figs/jasa_axis_missing_aCT.png
  figs/jasa_axis_missing_aTA.png
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import JASA_TabPFN_Experiments as J  # shared helpers + config

AXES = ['a_CT', 'a_TA']
MISSING_FRACTIONS = [0.05, 0.10, 0.25, 0.50]
# Two train sizes: if the error is the same at 300 and 1000, the missing-region
# inaccuracy is structural (extrapolation), not a small-sample artifact.
N_TRAIN_LIST = [300, 1000]
N_RUNS = 2
TEST_POOL_SIZE = 300
N_LINESTYLE = {300: '--', 1000: '-'}

RESULTS_PATH = os.path.join(J.RESULTS_DIR, 'jasa_axis_missing_results.json')


def _indist_ceiling():
    """In-distribution R^2 at N=1000 per output, from Experiment A if present."""
    if not os.path.exists(J.RESULTS_PATH):
        return {}
    with open(J.RESULTS_PATH) as f:
        data = json.load(f)
    expA = data.get('experiment_A', {})
    out = {}
    for o in J.OUTPUTS:
        if o in expA and expA[o].get('mean'):
            out[o] = expA[o]['mean'][-1]  # N=1000 is last
    return out


def run_axis(df, axis, n_train):
    print("\n" + "=" * 70)
    print(f"AXIS = {axis}  N_train = {n_train}  (hold out top fraction by {axis})")
    print("=" * 70)
    block = {o: {'frac': [], 'mean': [], 'std': []} for o in J.OUTPUTS}

    for f in MISSING_FRACTIONS:
        thresh = float(np.quantile(df[axis].values, 1.0 - f))
        test_region = df[df[axis] > thresh]
        train_region = df[df[axis] <= thresh]
        per_feat = {o: [] for o in J.OUTPUTS}

        for r in range(N_RUNS):
            seed = J.RANDOM_STATE + r
            df_test = test_region.sample(
                n=min(TEST_POOL_SIZE, len(test_region)), random_state=seed)
            df_train = train_region.sample(
                n=min(n_train, len(train_region)), random_state=seed)
            Xtr, Xte, Ytr, scalers_Y = J.scale_train_test(
                df_train, df_test, J.OUTPUTS)
            for j, o in enumerate(J.OUTPUTS):
                preds = J.tabpfn_fit_predict(Xtr, Ytr[:, j], Xte, scalers_Y[j], seed)
                per_feat[o].append(J.r2_score(df_test[o].values, preds))

        line = [f"{int(f*100):>2}% missing (>{thresh:.3f}), test={len(test_region)}"]
        for o in J.OUTPUTS:
            arr = np.array(per_feat[o])
            block[o]['frac'].append(f)
            block[o]['mean'].append(float(arr.mean()))
            block[o]['std'].append(float(arr.std()))
            line.append(f"{o}={arr.mean():+.2f}")
        print("  " + "  ".join(line))

    return block


def plot_axis(blocks_by_n, axis, ceiling):
    """blocks_by_n: {n_train: block}. Overlays each N (solid=1000, dashed=300)
    per output so a flat gap between the two N lines => not a small-sample
    artifact."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    pct = [int(f * 100) for f in MISSING_FRACTIONS]
    for i, o in enumerate(J.OUTPUTS):
        color = f'C{i}'
        for n_train, block in blocks_by_n.items():
            m = np.array(block[o]['mean'])
            ax.plot(pct, m, marker='o', color=color,
                    ls=N_LINESTYLE.get(n_train, '-'))
        if o in ceiling:
            ax.axhline(ceiling[o], color=color, ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(f'% of {axis} data withheld (top slice)')
    ax.set_ylabel('Test $R^2$ on held-out slice')
    ax.set_title(f'TabPFN extrapolation vs amount of {axis} missing\n'
                 f'solid=N1000, dashed=N300, dotted=in-dist ceiling (Exp A)')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(pct)

    # Two legends: color=output, linestyle=N
    color_handles = [Line2D([0], [0], color=f'C{i}', marker='o', lw=2,
                            label=J.OUTPUT_LABELS[o])
                     for i, o in enumerate(J.OUTPUTS)]
    style_handles = [Line2D([0], [0], color='k', ls=N_LINESTYLE[n], lw=2,
                            label=f'N={n}') for n in blocks_by_n]
    leg1 = ax.legend(handles=color_handles, loc='lower left', fontsize=9,
                     title='output')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='upper right', fontsize=9,
              title='train size')
    ax.grid(alpha=0.3)
    out = os.path.join(J.FIGS_DIR, f'jasa_axis_missing_{axis}.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    print("=" * 70)
    print(f"JASA single-axis missing experiment  (backend: {J._TABPFN_BACKEND})")
    print("=" * 70)
    if not J._TABPFN_AVAILABLE:
        print("ERROR: TabPFN not installed. pip install tabpfn-client")
        return
    J._ensure_tabpfn_auth()
    os.makedirs(J.FIGS_DIR, exist_ok=True)
    os.makedirs(J.RESULTS_DIR, exist_ok=True)

    df = J.load_data()
    ceiling = _indist_ceiling()
    print(f"  rows: {len(df)}   ceiling (Exp A N=1000) loaded for "
          f"{len(ceiling)}/{len(J.OUTPUTS)} outputs")

    results = {'_meta': {
        'axes': AXES, 'missing_fractions': MISSING_FRACTIONS,
        'n_train_list': N_TRAIN_LIST, 'n_runs': N_RUNS,
        'test_pool_size': TEST_POOL_SIZE,
        'held_out_end': 'high', 'fraction_basis': 'rows_by_axis_value',
        'indist_ceiling_N1000': ceiling,
    }}

    for axis in AXES:
        blocks_by_n = {}
        for n_train in N_TRAIN_LIST:
            block = run_axis(df, axis, n_train)
            blocks_by_n[n_train] = block
            results.setdefault(axis, {})[str(n_train)] = block
            with open(RESULTS_PATH, 'w') as fh:
                json.dump(results, fh, indent=2)
        plot_axis(blocks_by_n, axis, ceiling)

    print(f"\nDone. Results: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
