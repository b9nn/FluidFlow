"""
Paper Figure 1 + LaTeX tables, generated directly from the results JSON.

Nothing in this file hard-codes a number: every value in the figures and in the
emitted .tex comes from results/bm_paper_stats.json and
../TBCM/results/tbcm_paper_stats.json. Re-running the experiments and re-running
this script is the only way the paper's numbers change.

Outputs:
  figs/bm_ext_metrics_vs_n.png      paper Figure 1
  paper_out/table1_bm.tex           paper Table 1
  paper_out/table2_fidelity.tex     paper Table 2
  paper_out/numbers.json            every number quoted in the prose
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO, "paper_out")

BM_JSON = os.path.join(SCRIPT_DIR, "results", "bm_paper_stats.json")
TBCM_JSON = os.path.join(REPO, "TBCM", "results", "tbcm_paper_stats.json")
MOTOR_JSON = os.path.join(REPO, "TBCM", "results", "motor_map_results.json")

# Okabe-Ito derived, validated for CVD separation (deutan/protan/tritan).
COLORS = {"PR": "#CC79A7", "RF": "#D55E00", "NN": "#E69F00",
          "TabPFN": "#0072B2", "Source": "#666666"}
LABELS = {"PR_base": "PR, target alone", "RF_base": "RF, target alone",
          "NN_base": "NN, target alone", "PR_trans": "PR, transfer",
          "RF_trans": "RF, transfer", "NN_trans": "NN, transfer",
          "TabPFN": "TabPFN (no source)", "Source": "source alone"}
TABLE_ORDER = ["PR_base", "RF_base", "NN_base",
               "PR_trans", "RF_trans", "NN_trans", "TabPFN"]


def load(path, what):
    if not os.path.exists(path):
        sys.exit(f"missing {what}: {path}\nRun the experiment script first.")
    with open(path) as fh:
        return json.load(fh)


def style(method):
    """Colour encodes the regressor family, line style encodes the variant, so
    identity never rests on colour alone."""
    if method == "TabPFN":
        return dict(color=COLORS["TabPFN"], ls="-", lw=2.4, marker="o", ms=6, zorder=5)
    if method == "Source":
        return dict(color=COLORS["Source"], ls=":", lw=1.6, marker="", zorder=2)
    fam, kind = method.split("_")
    return dict(color=COLORS[fam], lw=1.8, ms=5, zorder=3,
                ls="-" if kind == "trans" else "--",
                marker="s" if kind == "trans" else "^")


def figure1(bm):
    outputs = bm["_meta"]["outputs"]
    ns = bm["_meta"]["n_grid"]
    methods = [m for m in TABLE_ORDER if bm["r2"].get(m)]

    fig, axes = plt.subplots(2, len(outputs), figsize=(3.05 * len(outputs), 6.4),
                             sharex=True)
    for col, out in enumerate(outputs):
        for row, metric in enumerate(["r2", "nrmse"]):
            ax = axes[row, col]
            for m in methods:
                xs = [n for n in ns if str(n) in bm[metric][m]]
                mu = np.array([bm[metric][m][str(n)][out]["mean"] for n in xs])
                sd = np.array([bm[metric][m][str(n)][out]["std"] for n in xs])
                st = style(m)
                ax.plot(xs, mu, **st)
                ax.fill_between(xs, mu - sd, mu + sd, color=st["color"],
                                alpha=0.13, lw=0, zorder=1)
            ax.set_xscale("log")
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(out, fontsize=11, fontweight="bold")
                ax.set_ylim(-0.55, 1.03)
                ax.axhline(0, color="k", lw=0.7, ls=":", zorder=1)
                src = bm["r2"]["Source"]
                if src:
                    vals = [src[str(n)][out]["mean"] for n in ns if str(n) in src]
                    ax.annotate(f"source alone\n$R^2$ = {np.mean(vals):.1f}",
                                xy=(0.04, 0.045), xycoords="axes fraction",
                                fontsize=7, color=COLORS["Source"], va="bottom")
            else:
                ax.set_xlabel("N (target simulations)", fontsize=9)
                ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Test $R^2$", fontsize=10)
    axes[1, 0].set_ylabel("nRMSE", fontsize=10)

    handles = [Line2D([], [], label=LABELS[m], **style(m)) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("BCM $\\rightarrow$ BM (far target): accuracy and error versus training size",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    path = os.path.join(SCRIPT_DIR, "figs", "bm_ext_metrics_vs_n.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def fmt(v, best=False):
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if best else s


def table1(bm, n_star=50):
    """Table 1: R^2 and nRMSE on BM at N = n_star for all seven configurations."""
    outputs = bm["_meta"]["outputs"]
    methods = [m for m in TABLE_ORDER if str(n_star) in bm["r2"].get(m, {})]
    key = str(n_star)

    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Aim 1. Accuracy ($R^2$, upper block) and range-normalised RMSE "
        f"(lower block) on the far BM target at $N={n_star}$ target simulations, "
        f"mean over {bm['_meta']['n_seeds']} seeds. Best per row in bold. The "
        f"\\emph{{target alone}} columns are each regressor family fit on the $N$ "
        f"target rows with no source; \\emph{{optimized transfer}} adds the BCM "
        f"source through the blended procedure of Sec.~2.2.}}",
        "\\label{tab:aim1}",
        "\\begin{tabular}{l|ccc|ccc|c}", "\\toprule",
        "& \\multicolumn{3}{c|}{Fit on target alone} & \\multicolumn{3}{c|}{Optimized transfer} & No source \\\\",
        "Output & PR & RF & NN & PR & RF & NN & TabPFN \\\\", "\\midrule",
        f"\\multicolumn{{8}}{{l}}{{\\textit{{$R^2$ (higher is better)}}}} \\\\",
    ]
    for o in outputs:
        vals = [bm["r2"][m][key][o]["mean"] for m in methods]
        b = int(np.argmax(vals))
        lines.append(f"{o} & " + " & ".join(
            fmt(v, i == b) for i, v in enumerate(vals)) + " \\\\")
    lines += ["\\midrule",
              "\\multicolumn{8}{l}{\\textit{nRMSE (lower is better)}} \\\\"]
    for o in outputs:
        vals = [bm["nrmse"][m][key][o]["mean"] for m in methods]
        b = int(np.argmin(vals))
        lines.append(f"{o} & " + " & ".join(
            fmt(v, i == b) for i, v in enumerate(vals)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    path = os.path.join(OUT_DIR, "table1_bm.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  saved: {path}")


def table2(bm, tbcm, n_star=50):
    """Table 2: F0 R^2 at N = n_star as the target moves from aligned to far."""
    key = str(n_star)

    def row(label, d, out="F0"):
        src = d["r2"]["Source"][key][out]["mean"]
        base = max(d["r2"][f"{f}_base"][key][out]["mean"] for f in ["PR", "RF", "NN"])
        tr = max(d["r2"][f"{f}_trans"][key][out]["mean"] for f in ["PR", "RF", "NN"])
        tp = d["r2"]["TabPFN"][key][out]["mean"]
        best = max(base, tr, tp)
        cells = [f"{src:.2f}" if src > -10 else f"{src:.1f}",
                 fmt(base, base == best), fmt(tr, tr == best), fmt(tp, tp == best)]
        return f"{label} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Aim 2. Source--target fidelity. $F_0$ accuracy ($R^2$) at "
        f"$N={n_star}$ as the target moves from an aligned lumped model (TBCM) to a "
        f"structurally distinct continuum model (BM). \\emph{{Baseline}} is the best "
        f"of the three regressor families fit on the target alone; "
        f"\\emph{{opt.\\ transfer}} is the best of the three transfer variants. "
        f"Transfer is only worth its source when it beats the baseline column. A "
        f"negative source-alone value means the unadapted BCM source is less "
        f"accurate than predicting the mean output.}}",
        "\\label{tab:aim2}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Target (fidelity) & Source alone & Baseline & Opt.\\ transfer & TabPFN \\\\",
        "\\midrule",
        row("TBCM (aligned lumped)", tbcm),
        row("BM (far continuum)", bm),
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    path = os.path.join(OUT_DIR, "table2_fidelity.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  saved: {path}")


def prose_numbers(bm, tbcm, n_star=50):
    """Every number quoted in the Results prose, so the text can be checked
    against a machine-generated source rather than retyped."""
    key = str(n_star)
    num = {"n_star": n_star,
           "bm_rows": bm["_meta"]["target_rows"],
           "tbcm_rows": tbcm["_meta"]["target_rows"],
           "source_rows": bm["_meta"]["source_rows"],
           "n_seeds": bm["_meta"]["n_seeds"],
           "test_pool": bm["_meta"]["test_pool"]}

    for tag, d, outs in [("bm", bm, bm["_meta"]["outputs"]),
                         ("tbcm", tbcm, tbcm["_meta"]["outputs"])]:
        for o in outs:
            for m in TABLE_ORDER + ["Source"]:
                if key in d["r2"].get(m, {}):
                    num[f"{tag}_{m}_{o}_r2_N{n_star}"] = round(
                        d["r2"][m][key][o]["mean"], 4)
        # best-of-family summaries
        for kind in ["base", "trans"]:
            for o in outs:
                vals = {f: d["r2"][f"{f}_{kind}"][key][o]["mean"]
                        for f in ["PR", "RF", "NN"] if key in d["r2"][f"{f}_{kind}"]}
                if vals:
                    bf = max(vals, key=vals.get)
                    num[f"{tag}_best_{kind}_{o}_N{n_star}"] = round(vals[bf], 4)
                    num[f"{tag}_best_{kind}_{o}_N{n_star}_family"] = bf

    # transfer gain over baseline, averaged across outputs, per N
    for tag, d, outs in [("bm", bm, bm["_meta"]["outputs"]),
                         ("tbcm", tbcm, tbcm["_meta"]["outputs"])]:
        gains = {}
        for n in d["_meta"]["n_grid"]:
            k = str(n)
            if k not in d["r2"]["RF_trans"]:
                continue
            g = [d["r2"][f"{f}_trans"][k][o]["mean"] - d["r2"][f"{f}_base"][k][o]["mean"]
                 for f in ["PR", "RF", "NN"] for o in outs]
            gains[k] = round(float(np.mean(g)), 4)
        num[f"{tag}_mean_transfer_gain_by_N"] = gains

    # TabPFN margin over the best non-TabPFN configuration, per N
    for tag, d, outs in [("bm", bm, bm["_meta"]["outputs"]),
                         ("tbcm", tbcm, tbcm["_meta"]["outputs"])]:
        marg = {}
        for n in d["_meta"]["n_grid"]:
            k = str(n)
            if k not in d["r2"].get("TabPFN", {}):
                continue
            m = [d["r2"]["TabPFN"][k][o]["mean"] -
                 max(d["r2"][c][k][o]["mean"] for c in TABLE_ORDER if c != "TabPFN")
                 for o in outs]
            marg[k] = round(float(np.mean(m)), 4)
        num[f"{tag}_tabpfn_margin_by_N"] = marg
        num[f"{tag}_tabpfn_wins_all_N"] = all(v > 0 for v in marg.values())

    if os.path.exists(MOTOR_JSON):
        mm = load(MOTOR_JSON, "motor maps")
        for dom in ["tbcm", "bm"]:
            if dom not in mm:
                continue
            sw = mm[dom]["sweep"]
            num[f"{dom}_motor_map"] = {
                n: {k: round(v["nrmse"], 4) for k, v in sw[n].items()} for n in sw}
            wins = [n for n in sorted(sw, key=int)
                    if "TabPFN" in sw[n] and sw[n]["TabPFN"]["nrmse"] < sw[n]["Transfer"]["nrmse"]]
            num[f"{dom}_motor_map_tabpfn_better_at_N"] = wins
            good = [int(n) for n in sorted(sw, key=int)
                    if "TabPFN" in sw[n] and sw[n]["TabPFN"]["r2"] >= 0.9]
            num[f"{dom}_motor_map_tabpfn_r2_ge_0.9_from_N"] = min(good) if good else None

    path = os.path.join(OUT_DIR, "numbers.json")
    with open(path, "w") as fh:
        json.dump(num, fh, indent=2)
    print(f"  saved: {path}")
    return num


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bm = load(BM_JSON, "BM results")
    tbcm = load(TBCM_JSON, "TBCM results")
    print("Generating paper artefacts...")
    figure1(bm)
    table1(bm)
    table2(bm, tbcm)
    num = prose_numbers(bm, tbcm)
    print(f"\n  TabPFN best at every N on BM:   {num['bm_tabpfn_wins_all_N']}")
    print(f"  TabPFN best at every N on TBCM: {num['tbcm_tabpfn_wins_all_N']}")
    print(f"  mean transfer gain over baseline, BM:   {num['bm_mean_transfer_gain_by_N']}")
    print(f"  mean transfer gain over baseline, TBCM: {num['tbcm_mean_transfer_gain_by_N']}")


if __name__ == "__main__":
    main()
