"""
Activation-map reconstruction -- paper Figure 2 (and the BM companion claim).

TBCM is sampled on a complete 40 x 40 (a_CT x a_TA) grid at PL = 1250 Pa
(1600 cells), which gives a REAL ODE ground-truth motor-control map. We draw N
cells at random, fit each method on those N, and predict the full 1600-cell map.
Image-level error is measured over all 1600 cells.

Produces:
  figs/tbcm_motor_map_F0.png    3 x 3: ODE truth | TabPFN | optimized transfer
                                rows N = 10 / 100 / 1000        (paper Fig. 2)
  figs/tbcm_motor_map_error.png image-level nRMSE and R^2 vs N
  ../Beam_Membrane/figs/bm_motor_map_F0.png   BM companion, dense full-data reference
  results/motor_map_results.json
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_methods as PM  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
BM_DIR = os.path.join(REPO, "Beam_Membrane")

FEATURES = ["a_CT", "a_TA", "PS"]
PL_SLICE = 1250
FIG_NS = [10, 100, 1000]
SWEEP_NS = [10, 25, 50, 100, 200, 400, 800, 1000, 1600]
N_SEEDS = 3
RANDOM_STATE = 42

OUT_JSON = os.path.join(SCRIPT_DIR, "results", "motor_map_results.json")


def load_source(outputs):
    src = pd.read_parquet(os.path.join(REPO, "data_binary.parquet"))
    if "Ps" in src.columns and "PS" not in src.columns:
        src = src.rename(columns={"Ps": "PS"})
    return src.dropna(subset=outputs)[FEATURES + outputs].reset_index(drop=True)


def tbcm_grid():
    d = pd.read_csv(os.path.join(SCRIPT_DIR, "dataset_TBCM.csv"), index_col=0)
    d = d.rename(columns={"Ps": "PS"})
    g = d[d["PL"] == PL_SLICE].copy()
    ct = np.sort(g["a_CT"].unique())
    ta = np.sort(g["a_TA"].unique())
    assert len(g) == len(ct) * len(ta), f"grid not complete: {len(g)}"
    g = g.sort_values(["a_TA", "a_CT"]).reset_index(drop=True)
    return g, ct, ta


def bm_grid(n_side=40):
    """BM is randomly sampled, so the reference map is a dense model fit on all
    phonating rows, evaluated on a regular grid at the median subglottal pressure."""
    d = pd.read_csv(os.path.join(BM_DIR, "dataset_BM_extended.csv"))
    if "Ps" in d.columns and "PS" not in d.columns:
        d = d.rename(columns={"Ps": "PS"})
    d = d[d["ACFL"] > 30][FEATURES + ["F0"]].reset_index(drop=True)
    dense = PM.RFFamily().fit(d[FEATURES].values, d[["F0"]].values, n_hint=len(d))
    ct = np.linspace(d["a_CT"].min(), d["a_CT"].max(), n_side)
    ta = np.linspace(d["a_TA"].min(), d["a_TA"].max(), n_side)
    CT, TA = np.meshgrid(ct, ta)
    ps = np.full(CT.size, float(d["PS"].median()))
    X = np.column_stack([CT.ravel(), TA.ravel(), ps])
    y = dense.predict(X).ravel()
    g = pd.DataFrame({"a_CT": X[:, 0], "a_TA": X[:, 1], "PS": X[:, 2], "F0": y})
    return g, ct, ta, d


def run_domain(name, grid, ct, ta, source_df, train_pool, use_tabpfn):
    """Fit every method on N cells and reconstruct the full map."""
    out = {"sweep": {}, "maps": {}, "grid": {"a_CT": ct.tolist(), "a_TA": ta.tolist()}}
    Xg, yg = grid[FEATURES].values, grid["F0"].values
    out["truth"] = yg.tolist()

    src_model = PM.RFFamily().fit(source_df[FEATURES].values,
                                  source_df[["F0"]].values,
                                  n_hint=len(source_df))

    for n in SWEEP_NS:
        if n > len(train_pool):
            continue
        acc = {"TabPFN": [], "Transfer": [], "Baseline": []}
        for s in range(N_SEEDS):
            rs = RANDOM_STATE + s
            tr = train_pool.sample(n=n, random_state=rs)
            Xtr, Ytr = tr[FEATURES].values, tr[["F0"]].values

            tp, bp, _ = PM.optimized_transfer(PM.RFFamily(), Xtr, Ytr, Xg,
                                              src_model, seed=rs)
            preds = {"Transfer": tp[:, 0], "Baseline": bp[:, 0]}
            if use_tabpfn:
                preds["TabPFN"] = PM.tabpfn_predict(Xtr, Ytr, Xg, seed=rs)[:, 0]

            for k, p in preds.items():
                acc[k].append((PM.nrmse(yg, p), PM.r2(yg, p)))
                if s == 0 and n in FIG_NS:
                    out["maps"].setdefault(str(n), {})[k] = p.tolist()

        out["sweep"][str(n)] = {
            k: {"nrmse": float(np.mean([v[0] for v in vals])),
                "r2": float(np.mean([v[1] for v in vals]))}
            for k, vals in acc.items() if vals}
        line = "  ".join(f"{k} nRMSE={out['sweep'][str(n)][k]['nrmse']:.3f}"
                         for k in out["sweep"][str(n)])
        print(f"  [{name}] N={n:>4}  {line}", flush=True)
    return out


def plot_maps(res, ct, ta, title, path, unit="Hz"):
    ns = [n for n in FIG_NS if str(n) in res["maps"]]
    cols = ["TabPFN", "Transfer"]
    have = [c for c in cols if c in res["maps"][str(ns[0])]]
    truth = np.array(res["truth"]).reshape(len(ta), len(ct))
    vmin, vmax = np.nanmin(truth), np.nanmax(truth)

    fig, axes = plt.subplots(len(ns), 1 + len(have),
                             figsize=(3.5 * (1 + len(have)), 3.1 * len(ns)))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(ns):
        for j, key in enumerate(["__truth__"] + have):
            ax = axes[i, j]
            if key == "__truth__":
                Z, lab = truth, "ground truth"
            else:
                Z = np.array(res["maps"][str(n)][key]).reshape(len(ta), len(ct))
                e = res["sweep"][str(n)][key]
                lab = f"{key}  (nRMSE {e['nrmse']:.3f})"
            im = ax.pcolormesh(ct, ta, Z, cmap="viridis", vmin=vmin, vmax=vmax,
                               shading="auto")
            ax.contour(ct, ta, Z, levels=8, colors="w", linewidths=0.4, alpha=0.6)
            ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"N = {n}\n$a_{{TA}}$", fontsize=9)
            if i == len(ns) - 1:
                ax.set_xlabel("$a_{CT}$", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, shrink=0.7, label=f"$F_0$ ({unit})")
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_error(res, path, title):
    ns = sorted(int(k) for k in res["sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"TabPFN": "C0", "Transfer": "C3", "Baseline": "C7"}
    for key in ["TabPFN", "Transfer", "Baseline"]:
        vals = [res["sweep"][str(n)].get(key) for n in ns]
        if not any(vals):
            continue
        axes[0].plot(ns, [v["nrmse"] for v in vals], marker="o",
                     label=key, color=colors[key])
        axes[1].plot(ns, [v["r2"] for v in vals], marker="o",
                     label=key, color=colors[key])
    for ax, lab in zip(axes, ["image-level nRMSE", "image-level $R^2$"]):
        ax.set_xscale("log")
        ax.set_xlabel("N (simulations used)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_ylim(-0.2, 1.02)
    axes[1].axhline(0, color="k", lw=0.6, ls=":")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    use_tabpfn = "--no-tabpfn" not in sys.argv
    t0 = time.time()
    if use_tabpfn:
        use_tabpfn = PM.enable_tabpfn()
        print(f"TabPFN cloud auth: {'OK' if use_tabpfn else 'FAILED'}")

    src = load_source(["F0"])
    results = {"_meta": {"pl_slice": PL_SLICE, "fig_ns": FIG_NS,
                         "sweep_ns": SWEEP_NS, "n_seeds": N_SEEDS,
                         "source": "data_binary.parquet"}}

    print("\nTBCM motor map (real 40x40 ODE ground truth, PL=1250)")
    g, ct, ta = tbcm_grid()
    print(f"  grid {len(ct)} x {len(ta)} = {len(g)} cells")
    results["tbcm"] = run_domain("TBCM", g, ct, ta, src, g, use_tabpfn)
    os.makedirs(os.path.join(SCRIPT_DIR, "figs"), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    plot_maps(results["tbcm"], ct, ta,
              "Activation-map reconstruction (TBCM, real ODE ground truth)",
              os.path.join(SCRIPT_DIR, "figs", "tbcm_motor_map_F0.png"))
    plot_error(results["tbcm"],
               os.path.join(SCRIPT_DIR, "figs", "tbcm_motor_map_error.png"),
               "TBCM activation-map reconstruction error vs N")
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nBM motor map (dense full-data reference)")
    gb, ctb, tab, bm_rows = bm_grid()
    results["bm"] = run_domain("BM", gb, ctb, tab, src, bm_rows, use_tabpfn)
    os.makedirs(os.path.join(BM_DIR, "figs"), exist_ok=True)
    plot_maps(results["bm"], ctb, tab,
              "Activation-map reconstruction (BM, dense full-data reference)",
              os.path.join(BM_DIR, "figs", "bm_motor_map_F0.png"))
    plot_error(results["bm"],
               os.path.join(BM_DIR, "figs", "bm_motor_map_error.png"),
               "BM activation-map reconstruction error vs N")

    results["_meta"]["elapsed_s"] = time.time() - t0
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
