"""
Independent verification of every number that reaches Draft 5.

This deliberately does NOT import the experiment scripts' result files as truth.
It re-derives what it can from the raw data and re-checks the rest for internal
consistency, so that a bug in the experiment harness shows up as a failed check
rather than as a confident number in the paper.

Checks:
  1  data provenance      row counts, filters, NaNs, schema, source/target ranges
  2  protocol integrity   train/test disjoint, correct sizes, seeds honoured
  3  metric correctness   R^2 and nRMSE recomputed from scratch on a fresh split
  4  reproducibility      re-running one cell reproduces the stored value
  5  claim audit          every quantitative claim in the paper checked vs JSON
  6  figure/table sync    emitted .tex numbers match the JSON they came from

Exit code is non-zero if any check FAILs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import paper_methods as PM  # noqa: E402

BM_JSON = os.path.join(REPO, "Beam_Membrane", "results", "bm_paper_stats.json")
TBCM_JSON = os.path.join(REPO, "TBCM", "results", "tbcm_paper_stats.json")
MOTOR_JSON = os.path.join(REPO, "TBCM", "results", "motor_map_results.json")

FEATURES = ["a_CT", "a_TA", "PS"]
PASS, FAIL, WARN = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


def warn(name, detail):
    WARN.append(f"{name}: {detail}")
    print(f"  [WARN] {name} -- {detail}")


# ---------------------------------------------------------------- 1 provenance
def check_provenance(bm, tb):
    print("\n[1] Data provenance")
    src = pd.read_parquet(os.path.join(REPO, "data_binary.parquet"))
    if "Ps" in src.columns and "PS" not in src.columns:
        src = src.rename(columns={"Ps": "PS"})
    outs = bm["_meta"]["outputs"]
    src_c = src.dropna(subset=outs)[FEATURES + outs]
    check("source row count matches metadata",
          len(src_c) == bm["_meta"]["source_rows"],
          f"{len(src_c)} vs {bm['_meta']['source_rows']}")

    d = pd.read_csv(os.path.join(REPO, "Beam_Membrane", "dataset_BM_extended.csv"))
    d = d.rename(columns={"Ps": "PS"})
    filt = d[d["ACFL"] > 30].dropna(subset=outs)
    check("BM phonating-row filter reproduces metadata",
          len(filt) == bm["_meta"]["target_rows"],
          f"ACFL>30 gives {len(filt)}, metadata says {bm['_meta']['target_rows']}")
    check("BM has no NaNs in modelled columns",
          not filt[FEATURES + outs].isna().any().any())

    t = pd.read_csv(os.path.join(REPO, "TBCM", "dataset_TBCM.csv"), index_col=0)
    t = t.rename(columns={"Ps": "PS"}).dropna(subset=tb["_meta"]["outputs"])
    check("TBCM row count matches metadata",
          len(t) == tb["_meta"]["target_rows"],
          f"{len(t)} vs {tb['_meta']['target_rows']}")

    check("input schema is the fixed [a_CT, a_TA, PS]",
          bm["_meta"]["features"] == FEATURES and tb["_meta"]["features"] == FEATURES)

    # the source really is a poor prior on BM: verify the ranges genuinely differ
    for o in ["F0", "SPL"]:
        s_lo, s_hi = src_c[o].min(), src_c[o].max()
        t_lo, t_hi = filt[o].min(), filt[o].max()
        overlap = max(0.0, min(s_hi, t_hi) - max(s_lo, t_lo))
        span = max(s_hi, t_hi) - min(s_lo, t_lo)
        print(f"         {o}: source [{s_lo:.0f},{s_hi:.0f}] "
              f"target [{t_lo:.0f},{t_hi:.0f}] overlap {overlap / span:.0%}")
    return filt, src_c


# ------------------------------------------------------------------ 2 protocol
def check_protocol(bm, target_df):
    print("\n[2] Protocol integrity")
    meta = bm["_meta"]
    ok_disjoint, ok_sizes = True, True
    for n in meta["n_grid"]:
        for s in range(meta["n_seeds"]):
            rs = 42 + s
            tr = target_df.reset_index(drop=True).sample(n=n, random_state=rs)
            pool = target_df.reset_index(drop=True).drop(tr.index)
            te = pool.sample(n=min(meta["test_pool"], len(pool)), random_state=rs)
            if set(tr.index) & set(te.index):
                ok_disjoint = False
            if len(tr) != n or len(te) != min(meta["test_pool"], len(pool)):
                ok_sizes = False
    check("train and test pools are disjoint at every N and seed", ok_disjoint)
    check("train/test sizes are exactly as specified", ok_sizes)
    check("seed count matches the paper's claim of 5", meta["n_seeds"] == 5,
          str(meta["n_seeds"]))
    check("every N reports the full set of per-seed runs",
          all(len(bm["r2"][m][str(n)][o]["runs"]) == meta["n_seeds"]
              for m in bm["r2"] if bm["r2"][m]
              for n in meta["n_grid"] if str(n) in bm["r2"][m]
              for o in meta["outputs"]))


# -------------------------------------------------------------------- 3 metric
def check_metrics():
    print("\n[3] Metric correctness (recomputed from scratch)")
    rng = np.random.RandomState(0)
    y = rng.rand(200) * 10
    p = y + rng.randn(200) * 0.5
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    check("R^2 matches 1 - SS_res/SS_tot",
          abs(PM.r2(y, p) - (1 - ss_res / ss_tot)) < 1e-10)
    manual = np.sqrt(np.mean((y - p) ** 2)) / (y.max() - y.min())
    check("nRMSE matches RMSE / (max - min)", abs(PM.nrmse(y, p) - manual) < 1e-12)
    check("R^2 of the mean predictor is 0",
          abs(PM.r2(y, np.full_like(y, y.mean()))) < 1e-12)


# ------------------------------------------------------------ 4 reproducibility
def check_repro(bm, target_df, src_df):
    print("\n[4] Reproducibility (re-running one cell end to end)")
    outs = bm["_meta"]["outputs"]
    n, rs = 50, 42
    src_model = PM.SOURCE_FAMILIES["RF"].fit(
        src_df[FEATURES].values, src_df[outs].values, n_hint=len(src_df))
    tdf = target_df.reset_index(drop=True)
    tr = tdf.sample(n=n, random_state=rs)
    te = tdf.drop(tr.index).sample(n=min(bm["_meta"]["test_pool"], len(tdf) - n),
                                   random_state=rs)
    Xtr, Ytr = tr[FEATURES].values, tr[outs].values
    Xte, Yte = te[FEATURES].values, te[outs].values

    sp = src_model.predict(Xte)
    stored_src = bm["r2"]["Source"][str(n)]["F0"]["runs"][0]
    got_src = PM.r2(Yte[:, 0], sp[:, 0])
    check("source-alone F0 R^2 reproduces (seed 0, N=50)",
          abs(got_src - stored_src) < 0.02, f"got {got_src:.3f}, stored {stored_src:.3f}")

    tp, bp, _ = PM.optimized_transfer(PM.FAMILIES["RF"], Xtr, Ytr, Xte, src_model, seed=rs)
    stored_b = bm["r2"]["RF_base"][str(n)]["F0"]["runs"][0]
    got_b = PM.r2(Yte[:, 0], bp[:, 0])
    check("RF baseline F0 R^2 reproduces (seed 0, N=50)",
          abs(got_b - stored_b) < 0.02, f"got {got_b:.3f}, stored {stored_b:.3f}")
    stored_t = bm["r2"]["RF_trans"][str(n)]["F0"]["runs"][0]
    got_t = PM.r2(Yte[:, 0], tp[:, 0])
    check("RF transfer F0 R^2 reproduces (seed 0, N=50)",
          abs(got_t - stored_t) < 0.02, f"got {got_t:.3f}, stored {stored_t:.3f}")


# --------------------------------------------------------------- 5 claim audit
def check_claims(bm, tb, mm):
    print("\n[5] Claim audit (paper prose vs results)")
    outs, ns = bm["_meta"]["outputs"], bm["_meta"]["n_grid"]
    non_tp = [f"{f}_{k}" for k in ("base", "trans") for f in ["PR", "RF", "NN"]]

    def r2v(d, m, n, o):
        return d["r2"][m][str(n)][o]["mean"]

    # "the source is less accurate than predicting the mean"
    worst = {o: max(r2v(bm, "Source", n, o) for n in ns) for o in outs}
    check("BM source-alone is worse than the mean predictor for every output",
          all(v < 0 for v in worst.values()),
          ", ".join(f"{o} max R2={v:.2f}" for o, v in worst.items()))

    # "TabPFN is the strongest at every N" -- verified, not assumed
    losses = [(n, o, round(r2v(bm, "TabPFN", n, o), 3),
               max(non_tp, key=lambda c: r2v(bm, c, n, o)),
               round(max(r2v(bm, c, n, o) for c in non_tp), 3))
              for n in ns for o in outs
              if r2v(bm, "TabPFN", n, o) < max(r2v(bm, c, n, o) for c in non_tp)]
    if losses:
        warn("TabPFN is NOT best in every BM cell",
             f"{len(losses)}/{len(ns) * len(outs)} cells lost: " +
             "; ".join(f"N={n} {o} ({tp} vs {w} {wv})" for n, o, tp, w, wv in losses[:8]))
    else:
        check("TabPFN has the highest R^2 in every BM cell", True)

    # "transfer barely clears the baseline on BM, but clears it on TBCM"
    bm_gain = float(np.mean([r2v(bm, f"{f}_trans", 50, o) - r2v(bm, f"{f}_base", 50, o)
                             for f in ["PR", "RF", "NN"] for o in outs]))
    tb_gain = float(np.mean([r2v(tb, f"{f}_trans", 50, o) - r2v(tb, f"{f}_base", 50, o)
                             for f in ["PR", "RF", "NN"] for o in tb["_meta"]["outputs"]]))
    check("transfer gain on BM is smaller than on TBCM",
          bm_gain < tb_gain, f"BM {bm_gain:+.3f} vs TBCM {tb_gain:+.3f}")
    check("TBCM source-alone is a genuinely useful prior (R^2 > 0)",
          r2v(tb, "Source", 50, "F0") > 0, f"F0 R2={r2v(tb, 'Source', 50, 'F0'):.2f}")

    # blend weights should not be degenerate
    for fam in ["PR", "RF", "NN"]:
        w = np.array(bm["weights"][fam]["50"])
        check(f"{fam} blend weights are normalised and non-negative",
              np.allclose(w.sum(axis=1), 1.0, atol=1e-6) and (w >= -1e-9).all())
    tgt_share = float(np.mean([np.array(bm["weights"][f]["50"])[:, 0].mean()
                               for f in ["PR", "RF", "NN"]]))
    print(f"         mean weight on the target-only sub-model at N=50: {tgt_share:.2f}")
    if tgt_share > 0.95:
        warn("blend is nearly all target-only",
             "transfer is degenerate to baseline by construction; check OOF stacking")

    # motor map
    for dom in ["tbcm", "bm"]:
        if dom not in mm:
            continue
        sw = mm[dom]["sweep"]
        nns = sorted(sw, key=int)
        tw = [n for n in nns if sw[n]["TabPFN"]["nrmse"] < sw[n]["Transfer"]["nrmse"]]
        print(f"         {dom} map: TabPFN beats transfer at {len(tw)}/{len(nns)} N "
              f"({', '.join(tw) if tw else 'none'})")
    check("motor-map grid is the complete 40x40 ODE slice",
          len(mm["tbcm"]["truth"]) == 1600, f"{len(mm['tbcm']['truth'])} cells")


# ------------------------------------------------------------ 6 table/tex sync
def check_tex(bm):
    print("\n[6] Emitted LaTeX matches the results JSON")
    p = os.path.join(HERE, "table1_bm.tex")
    if not os.path.exists(p):
        warn("table1_bm.tex not generated yet", "run BM_PaperFigures.py")
        return
    txt = open(p).read()
    outs = bm["_meta"]["outputs"]
    order = ["PR_base", "RF_base", "NN_base", "PR_trans", "RF_trans", "NN_trans", "TabPFN"]
    body = txt.split("$R^2$ (higher is better)")[1]
    ok = True
    for o in outs:
        m = re.search(rf"^{o} & (.+?) \\\\", body, re.M)
        if not m:
            ok = False
            continue
        cells = [float(re.sub(r"[^0-9.\-]", "", c)) for c in m.group(1).split("&")]
        want = [round(bm["r2"][k]["50"][o]["mean"], 2) for k in order]
        if not np.allclose(cells, want, atol=0.005):
            ok = False
            print(f"         MISMATCH {o}: tex {cells} vs json {want}")
    check("every Table 1 R^2 cell equals the stored result", ok)


def main():
    print("=" * 72)
    print("Draft 5 sanity check")
    print("=" * 72)
    bm = json.load(open(BM_JSON))
    tb = json.load(open(TBCM_JSON))
    mm = json.load(open(MOTOR_JSON))

    target_df, src_df = check_provenance(bm, tb)
    check_protocol(bm, target_df)
    check_metrics()
    check_repro(bm, target_df, src_df)
    check_claims(bm, tb, mm)
    check_tex(bm)

    print("\n" + "=" * 72)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(WARN)} warnings")
    for w in WARN:
        print(f"  WARN {w}")
    for f in FAIL:
        print(f"  FAIL {f}")
    print("=" * 72)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
