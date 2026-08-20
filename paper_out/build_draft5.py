"""
Build FF_Draft_5.tex from the template by substituting every @@TOKEN@@ with a
value computed from the experiment result JSONs, then compile it to PDF.

The point of this indirection is that no statistic in the paper is typed by hand.
If a number appears in Draft 5, it was computed here from a results file. Any
token left unsubstituted is a hard error, so a missing statistic cannot silently
become prose.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
BM_JSON = os.path.join(REPO, "Beam_Membrane", "results", "bm_paper_stats.json")
TBCM_JSON = os.path.join(REPO, "TBCM", "results", "tbcm_paper_stats.json")
MOTOR_JSON = os.path.join(REPO, "TBCM", "results", "motor_map_results.json")

N_STAR = 50
FAMILIES = ["PR", "RF", "NN"]
NON_TABPFN = [f"{f}_{k}" for k in ("base", "trans") for f in FAMILIES]


def load(path, what):
    if not os.path.exists(path):
        sys.exit(f"ERROR missing {what}: {path}")
    with open(path) as fh:
        return json.load(fh)


def r2(d, m, n, o):
    return d["r2"][m][str(n)][o]["mean"]


def f2(v):
    return f"{v:.2f}"


def signed(v):
    return f"{v:+.2f}"


def best_of(d, kind, n, o):
    vals = {f: r2(d, f"{f}_{kind}", n, o) for f in FAMILIES}
    f = max(vals, key=vals.get)
    return vals[f], f


def mean_gain(d, n, outs):
    return float(np.mean([r2(d, f"{f}_trans", n, o) - r2(d, f"{f}_base", n, o)
                          for f in FAMILIES for o in outs]))


def tabpfn_margin(d, n, outs):
    return float(np.mean([r2(d, "TabPFN", n, o) - max(r2(d, c, n, o) for c in NON_TABPFN)
                          for o in outs]))


def coverage_phrase(d, outs):
    """How often TabPFN is the best configuration, across N, outputs, and both
    metrics. Stated exactly rather than asserted."""
    ns = [n for n in d["_meta"]["n_grid"] if str(n) in d["r2"]["TabPFN"]]
    total = wins = 0
    for n in ns:
        for o in outs:
            total += 1
            if r2(d, "TabPFN", n, o) >= max(r2(d, c, n, o) for c in NON_TABPFN):
                wins += 1
            total += 1
            tp = d["nrmse"]["TabPFN"][str(n)][o]["mean"]
            if tp <= min(d["nrmse"][c][str(n)][o]["mean"] for c in NON_TABPFN):
                wins += 1
    if wins == total:
        return "every training size and every output", wins, total
    frac = f"{wins} of {total} (training size $\\times$ output $\\times$ metric) comparisons"
    return frac, wins, total


def map_phrase(mm, dom):
    """Where TabPFN's activation-map error beats optimized transfer."""
    sw = mm[dom]["sweep"]
    ns = sorted(sw, key=int)
    wins = [n for n in ns if "TabPFN" in sw[n]
            and sw[n]["TabPFN"]["nrmse"] < sw[n]["Transfer"]["nrmse"]]
    if not wins:
        return "no training size", None
    if len(wins) == len(ns):
        return "every training size", int(wins[0])
    return (f"{len(wins)} of the {len(ns)} training sizes tested "
            f"($N \\ge {min(int(w) for w in wins)}$)"), int(min(int(w) for w in wins))


def first_n_reaching(mm, dom, thresh=0.9):
    sw = mm[dom]["sweep"]
    good = [int(n) for n in sorted(sw, key=int)
            if "TabPFN" in sw[n] and sw[n]["TabPFN"]["r2"] >= thresh]
    return good[0] if good else None


def tabpfn_from_n(d, o="F0"):
    """Smallest N such that TabPFN is the best configuration for this output at
    that N and every larger N tested. Stated, not assumed."""
    ns = [n for n in d["_meta"]["n_grid"] if str(n) in d["r2"]["TabPFN"]]
    best_at = [n for n in ns
               if r2(d, "TabPFN", n, o) >= max(r2(d, c, n, o) for c in NON_TABPFN)]
    for i, n in enumerate(ns):
        if all(m in best_at for m in ns[i:]):
            return n
    return None


def worst_deficit(*datasets, method="TabPFN"):
    """Largest amount by which `method` trails the best configuration overall,
    across every dataset, output and N. For the transfer family, `method` is the
    best of the three transfer variants at each cell."""
    everyone = NON_TABPFN + ["TabPFN"]
    worst = 0.0
    for d in datasets:
        for n in d["_meta"]["n_grid"]:
            if str(n) not in d["r2"]["TabPFN"]:
                continue
            for o in d["_meta"]["outputs"]:
                best = max(r2(d, c, n, o) for c in everyone)
                if method == "TabPFN":
                    mine = r2(d, "TabPFN", n, o)
                else:
                    mine = max(r2(d, f"{f}_trans", n, o) for f in FAMILIES)
                worst = max(worst, best - mine)
    return worst


def main():
    bm = load(BM_JSON, "BM results")
    tb = load(TBCM_JSON, "TBCM results")
    mm = load(MOTOR_JSON, "motor-map results")
    sd = load(os.path.join(HERE, "source_diagnostic.json"), "source diagnostic")

    bo = bm["_meta"]["outputs"]
    to = tb["_meta"]["outputs"]
    meta = bm["_meta"]
    nn = meta["nn"]

    # Describe the ACTUAL breakpoints of get_model_params, not merely the values
    # it happens to take at the tested grid points.
    import paper_methods as PM_  # noqa: N813
    bands, prev = [], None
    for n in range(1, 2001):
        p = PM_.get_model_params(n)
        key = (p["max_depth"], p["min_samples_leaf"])
        if key != prev:
            bands.append([n, p])
            prev = key
    parts = []
    for i, (lo, p) in enumerate(bands):
        depth = p["max_depth"] if p["max_depth"] else "unlimited"
        rng = (f"$N<{bands[i + 1][0]}$" if i + 1 < len(bands) else f"$N\\ge{lo}$")
        parts.append(f"{rng}: depth {depth}, min leaf {p['min_samples_leaf']}")
    sched_txt = "; ".join(parts)

    cov, wins, total = coverage_phrase(bm, bo)
    tb_map_txt, _ = map_phrase(mm, "tbcm")
    bm_map_txt, _ = map_phrase(mm, "bm")
    tb_map_n = first_n_reaching(mm, "tbcm")
    bm_map_n = first_n_reaching(mm, "bm")
    n0 = bm["_meta"]["n_grid"][0]

    aim3_outs = [o for o in bo if o not in ("F0", "SPL")]
    aim3_wins = sum(
        1 for o in bo
        if r2(bm, "TabPFN", N_STAR, o) >= max(r2(bm, c, N_STAR, o) for c in NON_TABPFN))

    # How does optimized transfer place against the full method set on each
    # target? This is the decision-relevant form of the alignment claim: not
    # "how much does the source add" but "is transfer the method to reach for".
    ALL_CFG = NON_TABPFN + ["TabPFN"]

    def _trans_standing(d, ns):
        outs = d["_meta"]["outputs"]
        ranks, wins, cells = [], 0, 0
        for n in ns:
            for o in outs:
                bt = max(r2(d, f"{f}_trans", n, o) for f in FAMILIES)
                ordered = sorted((r2(d, c, n, o) for c in ALL_CFG), reverse=True)
                ranks.append(1 + ordered.index(bt))
                cells += 1
                if bt > r2(d, "TabPFN", n, o):
                    wins += 1
        return float(np.mean(ranks)), wins, cells

    LOW_N = [n for n in bm["_meta"]["n_grid"] if n <= 50]
    tb_rank_lo, tb_win_lo, tb_cells_lo = _trans_standing(tb, LOW_N)
    bm_rank_lo, bm_win_lo, bm_cells_lo = _trans_standing(bm, LOW_N)
    _, bm_win_all, bm_cells_all = _trans_standing(bm, bm["_meta"]["n_grid"])
    bm_rank_best = min(_trans_standing(bm, [n])[0] for n in bm["_meta"]["n_grid"])

    # Does any source-quality statistic predict realised transfer gain? Reported
    # in the Discussion with its p-value precisely because it does not reach
    # significance at this sample size.
    from scipy.stats import spearmanr as _sp
    pairs = []
    for d, key, outs in [(bm, "BM", bo), (tb, "TBCM", to)]:
        for o in outs:
            g = float(np.mean([r2(d, f"{f}_trans", n, o) - r2(d, f"{f}_base", n, o)
                               for f in FAMILIES for n in [10, 20, 50]]))
            pairs.append((sd[key]["outputs"][o]["unadapted_r2"],
                          sd[key]["outputs"][o]["spearman"], g))
    rho_u, p_u = _sp([x[0] for x in pairs], [x[2] for x in pairs])
    rho_r, p_r = _sp([x[1] for x in pairs], [x[2] for x in pairs])

    tok = {
        "DATE": dt.date.today().strftime("%B %-d, %Y"),
        "N_PAIRS": str(len(pairs)),
        "TBCM_TRANS_RANK_LO": f"{tb_rank_lo:.1f}",
        "BM_TRANS_RANK_LO": f"{bm_rank_lo:.1f}",
        "BM_TRANS_RANK_BEST": f"{bm_rank_best:.1f}",
        "TBCM_TRANS_WINS_LO": f"{tb_win_lo} of {tb_cells_lo}",
        "BM_TRANS_WINS_LO": f"{bm_win_lo} of {bm_cells_lo}",
        "BM_TRANS_WINS_ALL": f"{bm_win_all} of {bm_cells_all}",
        "RHO_UNADAPTED": f"{rho_u:+.2f}",
        "P_UNADAPTED": f"{p_u:.2f}",
        "RHO_RANK": f"{rho_r:+.2f}",
        "P_RANK": f"{p_r:.2f}",
        "SOURCE_ROWS": f"{meta['source_rows']:,}",
        "BM_ROWS": f"{meta['target_rows']:,}",
        "TBCM_ROWS": f"{tb['_meta']['target_rows']:,}",
        "N_GRID": ", ".join(str(n) for n in meta["n_grid"]),
        "TEST_POOL": str(meta["test_pool"]),
        "N_SEEDS": str(meta["n_seeds"]),
        "POLY_DEGREE": str(meta["poly_degree"]),
        "RF_TREES": str(meta["rf_params_schedule"][str(meta["n_grid"][0])]["n_estimators"]),
        "RF_SCHEDULE": sched_txt,
        "NN_HIDDEN": str(nn["hidden"]),
        "NN_EPOCHS": str(nn["epochs"]),
        "NN_LR": f"{nn['lr']:g}",
        "NN_WD": f"{nn['weight_decay']:g}",
        "NN_SRC_SUB": f"{nn['source_subsample']:,}",
        "NN_SRC_EPOCHS": str(nn["source_epochs"]),
        "NN_SRC_LR": f"{nn['source_lr']:g}",

        "BM_SOURCE_F0": f"{r2(bm, 'Source', N_STAR, 'F0'):.1f}",
        "TBCM_SOURCE_F0": f2(r2(tb, "Source", N_STAR, "F0")),
        "BM_BASE_F0": f2(best_of(bm, "base", N_STAR, "F0")[0]),
        "BM_TRANS_F0": f2(best_of(bm, "trans", N_STAR, "F0")[0]),
        "TBCM_BASE_F0": f2(best_of(tb, "base", N_STAR, "F0")[0]),
        "TBCM_TRANS_F0": f2(best_of(tb, "trans", N_STAR, "F0")[0]),
        "BM_TABPFN_F0": f2(r2(bm, "TabPFN", N_STAR, "F0")),
        "TBCM_TABPFN_F0": f2(r2(tb, "TabPFN", N_STAR, "F0")),
        "BM_TABPFN_ACFL": f2(r2(bm, "TabPFN", N_STAR, "ACFL")),
        "BM_TABPFN_PC": f2(r2(bm, "TabPFN", N_STAR, "PC")),
        "BM_TABPFN_CPP": f2(r2(bm, "TabPFN", N_STAR, "CPP")),
        "BM_GAIN_N50": signed(mean_gain(bm, N_STAR, bo)),
        "BM_GAIN_N500": signed(mean_gain(bm, meta["n_grid"][-1], bo)),
        "BM_TABPFN_MARGIN_N50": signed(tabpfn_margin(bm, N_STAR, bo)),
        "BM_TABPFN_COVERAGE": cov,
        "TBCM_MAP_N": str(tb_map_n) if tb_map_n else "the smallest tested",
        "TBCM_MAP_WINS": tb_map_txt,
        "BM_MAP_WINS": bm_map_txt,
        "BM_MAP_N": str(bm_map_n) if bm_map_n else "?",
        "TBCM_GAIN_N50": signed(mean_gain(tb, N_STAR, to)),
        "BM_TABPFN_FROM_N": str(tabpfn_from_n(bm)),
        "TBCM_TABPFN_FROM_N": str(tabpfn_from_n(tb)),
        "TBCM_TRANS_F0_N10": f2(best_of(tb, "trans", n0, "F0")[0]),
        "TBCM_TABPFN_F0_N10": f2(r2(tb, "TabPFN", n0, "F0")),
        "BM_NN_BASE_F0_N10": f2(r2(bm, "NN_base", n0, "F0")),
        "BM_TABPFN_F0_N10": f2(r2(bm, "TabPFN", n0, "F0")),
        "TABPFN_WORST_DEFICIT": f"{worst_deficit(bm, tb):.2f}",
        "BM_F0_RHO": f"{sd['BM']['outputs']['F0']['spearman']:.2f}",
        "BM_MIN_RHO": f"{min(v['spearman'] for v in sd['BM']['outputs'].values()):.2f}",
        "TBCM_GAIN_MEAN": f"{np.mean([mean_gain(tb, n, to) for n in tb['_meta']['n_grid']]):+.2f}",
        "BM_GAIN_MEAN": f"{np.mean([mean_gain(bm, n, bo) for n in bm['_meta']['n_grid']]):+.2f}",
        "BM_TRANS_GAIN_N50_3": f"{best_of(bm, 'trans', 50, 'F0')[0] - best_of(bm, 'base', 50, 'F0')[0]:+.3f}",
        "BM_F0_GAIN_LO": f"{np.mean([r2(bm, f'{f}_trans', n, 'F0') - r2(bm, f'{f}_base', n, 'F0') for f in FAMILIES for n in [10, 20, 50]]):+.2f}",
        "TBCM_RESID_W": f"{np.mean([np.array(tb['weights'][f][str(n)]).mean(axis=0)[1] for f in FAMILIES for n in tb['_meta']['n_grid']]):.0%}".replace('%', r'\%'),
        "BM_TGT_W_LO": f"{np.mean([np.array(bm['weights'][f]['10']).mean(axis=0)[0] for f in FAMILIES]):.2f}",
        "BM_TGT_W_HI": f"{np.mean([np.array(bm['weights'][f]['500']).mean(axis=0)[0] for f in FAMILIES]):.2f}",
        "TRANSFER_WORST_DEFICIT": f"{worst_deficit(bm, tb, method='trans'):.2f}",
        "BM_BASE_F0_3": f"{best_of(bm, 'base', N_STAR, 'F0')[0]:.3f}",
        "BM_TRANS_F0_3": f"{best_of(bm, 'trans', N_STAR, 'F0')[0]:.3f}",
        "BM_F0_DELTA": f"{best_of(bm, 'trans', N_STAR, 'F0')[0] - best_of(bm, 'base', N_STAR, 'F0')[0]:+.3f}",
        "AIM3_TABPFN_WINS": {5: "all five", 4: "four", 3: "three",
                             2: "two", 1: "one", 0: "none"}[aim3_wins],
        "AIM3_GAIN": signed(mean_gain(bm, N_STAR, aim3_outs)),
    }

    with open(os.path.join(HERE, "FF_Draft_5.tex.template")) as fh:
        tex = fh.read()
    for k, v in tok.items():
        tex = tex.replace(f"@@{k}@@", str(v))

    leftover = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", tex)))
    if leftover:
        sys.exit(f"ERROR unsubstituted tokens: {leftover}")

    out_tex = os.path.join(HERE, "FF_Draft_5.tex")
    with open(out_tex, "w") as fh:
        fh.write(tex)
    print(f"  wrote {out_tex}")
    with open(os.path.join(HERE, "substitutions.json"), "w") as fh:
        json.dump(tok, fh, indent=2)
    print(f"  wrote substitutions.json ({len(tok)} values)")
    print(f"  TabPFN best in {wins}/{total} BM comparisons")

    for _ in range(2):
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", "FF_Draft_5.tex"],
                           cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        sys.exit("ERROR pdflatex failed")
    print(f"  compiled {os.path.join(HERE, 'FF_Draft_5.pdf')}")


if __name__ == "__main__":
    main()
