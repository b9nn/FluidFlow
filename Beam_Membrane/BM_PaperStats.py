"""
BM (far target) master statistics run for the paper -- Table 1 and Figure 1.

Seven configurations under one protocol, plus source-alone:

    PR_base / RF_base / NN_base     regressor fit on the N target rows alone
    PR_trans / RF_trans / NN_trans  optimized transfer (identical procedure)
    TabPFN                          no source, in-context fit
    Source                          BCM source applied to BM with no adaptation

Protocol (paper Methods): N in {10,20,50,100,200,500}; every method trains on the
SAME N target rows and is evaluated on the SAME disjoint held-out pool of up to
1000 rows; mean +/- s.d. over 5 seeds. Metrics: R^2 and range-normalised RMSE.

Target : dataset_BM_extended.csv filtered to phonating rows (ACFL > 30) -> 4647 rows
Source : data_binary.parquet (360,750-row BCM)
Outputs: F0, SPL, ACFL, PC, CPP

Writes results/bm_paper_stats.json incrementally.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_methods as PM  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)

FEATURES = ["a_CT", "a_TA", "PS"]
OUTPUTS = ["F0", "SPL", "ACFL", "PC", "CPP"]
N_GRID = [10, 20, 50, 100, 200, 500]
N_SEEDS = 5
TEST_POOL = 1000
RANDOM_STATE = 42
ACFL_PHONATION_THRESHOLD = 30.0

METHODS = ["PR_base", "RF_base", "NN_base",
           "PR_trans", "RF_trans", "NN_trans",
           "TabPFN", "Source"]

OUT_JSON = os.path.join(SCRIPT_DIR, "results", "bm_paper_stats.json")


def load_data():
    src = pd.read_parquet(os.path.join(REPO, "data_binary.parquet"))
    if "Ps" in src.columns and "PS" not in src.columns:
        src = src.rename(columns={"Ps": "PS"})
    src = src.dropna(subset=OUTPUTS)[FEATURES + OUTPUTS].reset_index(drop=True)

    bm = pd.read_csv(os.path.join(SCRIPT_DIR, "dataset_BM_extended.csv"))
    if "Ps" in bm.columns and "PS" not in bm.columns:
        bm = bm.rename(columns={"Ps": "PS"})
    n_all = len(bm)
    bm = bm[bm["ACFL"] > ACFL_PHONATION_THRESHOLD]
    bm = bm.dropna(subset=OUTPUTS)[FEATURES + OUTPUTS].reset_index(drop=True)
    print(f"  source {len(src)} rows | target {len(bm)}/{n_all} phonating rows")
    return src, bm


def main():
    use_tabpfn = "--no-tabpfn" not in sys.argv
    t_start = time.time()
    print("=" * 72)
    print("BM paper statistics: 7 configurations + source-alone")
    print("=" * 72)

    src, bm = load_data()

    if use_tabpfn:
        ok = PM.enable_tabpfn()
        print(f"  TabPFN cloud auth: {'OK' if ok else 'FAILED'}")
        if not ok:
            print("  !! continuing WITHOUT TabPFN")
            use_tabpfn = False

    Xs, Ys = src[FEATURES].values, src[OUTPUTS].values

    print("\nFitting BCM source models (once, reused for every N and seed)...")
    source_models = {}
    for fam in ["PR", "RF", "NN"]:
        t0 = time.time()
        source_models[fam] = PM.SOURCE_FAMILIES[fam].fit(Xs, Ys, n_hint=len(Xs))
        print(f"  [{fam}] source fit in {time.time() - t0:.1f}s")

    results = {
        "_meta": {
            "n_grid": N_GRID,
            "n_seeds": N_SEEDS,
            "test_pool": TEST_POOL,
            "outputs": OUTPUTS,
            "features": FEATURES,
            "target": "dataset_BM_extended.csv",
            "target_filter": f"ACFL > {ACFL_PHONATION_THRESHOLD}",
            "target_rows": int(len(bm)),
            "source": "data_binary.parquet",
            "source_rows": int(len(src)),
            "blend_weights": "out-of-fold stacking, K=min(5,N), non-negative, normalised",
            "rf_params_schedule": {str(n): PM.get_model_params(n) for n in N_GRID},
            "poly_degree": PM.PolyFamily.DEGREE,
            "nn": {"hidden": PM.NNFamily.HIDDEN, "epochs": PM.NNFamily.EPOCHS,
                   "lr": PM.NNFamily.LR, "weight_decay": PM.NNFamily.WEIGHT_DECAY,
                   "source_epochs": PM.NNSourceFamily.EPOCHS,
                   "source_lr": PM.NNSourceFamily.LR,
                   "source_subsample": PM.NNSourceFamily.SUBSAMPLE},
        },
        "r2": {m: {} for m in METHODS},
        "nrmse": {m: {} for m in METHODS},
        "weights": {f: {} for f in ["PR", "RF", "NN"]},
    }

    for n in N_GRID:
        print(f"\n--- N = {n} ---")
        acc_r2 = {m: {o: [] for o in OUTPUTS} for m in METHODS}
        acc_nr = {m: {o: [] for o in OUTPUTS} for m in METHODS}
        wacc = {f: [] for f in ["PR", "RF", "NN"]}

        for s in range(N_SEEDS):
            rs = RANDOM_STATE + s
            tr = bm.sample(n=n, random_state=rs)
            pool = bm.drop(tr.index)
            te = pool.sample(n=min(TEST_POOL, len(pool)), random_state=rs)
            Xtr, Ytr = tr[FEATURES].values, tr[OUTPUTS].values
            Xte, Yte = te[FEATURES].values, te[OUTPUTS].values

            preds = {}
            for fam in ["PR", "RF", "NN"]:
                tp, bp, w = PM.optimized_transfer(
                    PM.FAMILIES[fam], Xtr, Ytr, Xte, source_models[fam], seed=rs)
                preds[f"{fam}_trans"] = tp
                preds[f"{fam}_base"] = bp
                wacc[fam].append(w.tolist())

            preds["Source"] = source_models["RF"].predict(Xte)

            if use_tabpfn:
                t0 = time.time()
                preds["TabPFN"] = PM.tabpfn_predict(Xtr, Ytr, Xte, seed=rs)
                print(f"    seed {s}: TabPFN {time.time() - t0:.0f}s", flush=True)

            for m, p in preds.items():
                for j, o in enumerate(OUTPUTS):
                    acc_r2[m][o].append(PM.r2(Yte[:, j], p[:, j]))
                    acc_nr[m][o].append(PM.nrmse(Yte[:, j], p[:, j]))

        for m in METHODS:
            if not acc_r2[m][OUTPUTS[0]]:
                continue
            results["r2"][m][str(n)] = {
                o: {"mean": float(np.mean(acc_r2[m][o])),
                    "std": float(np.std(acc_r2[m][o], ddof=1)) if len(acc_r2[m][o]) > 1 else 0.0,
                    "runs": [float(v) for v in acc_r2[m][o]]}
                for o in OUTPUTS}
            results["nrmse"][m][str(n)] = {
                o: {"mean": float(np.nanmean(acc_nr[m][o])),
                    "std": float(np.nanstd(acc_nr[m][o], ddof=1)) if len(acc_nr[m][o]) > 1 else 0.0,
                    "runs": [float(v) for v in acc_nr[m][o]]}
                for o in OUTPUTS}
        for f in ["PR", "RF", "NN"]:
            results["weights"][f][str(n)] = np.mean(np.array(wacc[f]), axis=0).tolist()

        shown = [m for m in METHODS if str(n) in results["r2"][m]]
        print("    F0 R2:  " + "  ".join(
            f"{m}={results['r2'][m][str(n)]['F0']['mean']:+.3f}" for m in shown))

        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(results, fh, indent=2)

    results["_meta"]["elapsed_s"] = time.time() - t_start
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDone in {time.time() - t_start:.0f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
