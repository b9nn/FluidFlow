"""
What makes a source model useful for transfer?

Draft 5's Discussion originally claimed that a source's *unadapted accuracy* on
the target predicts whether transfer will help. Our own results falsify that:

  TBCM SPL   unadapted R^2 = -7.09   yet transfer gains up to +0.34
  BM   SPL   unadapted R^2 = -2.23   yet transfer gains ~0

The BM source is *more* accurate in raw R^2 and *less* useful. R^2 conflates two
different failures: miscalibration (wrong scale/offset, which residual correction
and feature augmentation can undo) and structural disagreement (the source ranks
operating points wrongly, which nothing downstream can undo).

This script separates them, per output and per domain:

  Spearman rho        rank agreement -- is the ordering right?
  affine-corrected R2 R^2 after the best least-squares rescale a*src+b, i.e. the
                      ceiling a residual/augmentation sub-model could reach if it
                      only had to fix scale and offset

Writes results/source_diagnostic.json.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import paper_methods as PM  # noqa: E402

FEATURES = ["a_CT", "a_TA", "PS"]
OUT_JSON = os.path.join(HERE, "source_diagnostic.json")


def load_source(outputs):
    s = pd.read_parquet(os.path.join(REPO, "data_binary.parquet"))
    if "Ps" in s.columns and "PS" not in s.columns:
        s = s.rename(columns={"Ps": "PS"})
    return s.dropna(subset=outputs)[FEATURES + outputs].reset_index(drop=True)


def load_target(name):
    if name == "BM":
        t = pd.read_csv(os.path.join(REPO, "Beam_Membrane", "dataset_BM_extended.csv"))
        t = t.rename(columns={"Ps": "PS"})
        t = t[t["ACFL"] > 30]
        outs = ["F0", "SPL", "ACFL", "PC", "CPP"]
    else:
        t = pd.read_csv(os.path.join(REPO, "TBCM", "dataset_TBCM.csv"), index_col=0)
        t = t.rename(columns={"Ps": "PS"})
        outs = ["F0", "SPL"]
    return t.dropna(subset=outs)[FEATURES + outs].reset_index(drop=True), outs


def main():
    results = {}
    for dom in ["BM", "TBCM"]:
        tgt, outs = load_target(dom)
        src = load_source(outs)
        model = PM.RFFamily().fit(src[FEATURES].values, src[outs].values,
                                  n_hint=len(src))
        P, Y = model.predict(tgt[FEATURES].values), tgt[outs].values

        print(f"\n=== {dom}: unadapted BCM source on {len(tgt)} target rows ===")
        print(f"{'output':<7}{'R2':>10}{'Spearman':>10}{'Pearson':>10}{'affineR2':>10}")
        results[dom] = {"n_target_rows": int(len(tgt)), "outputs": {}}
        for j, o in enumerate(outs):
            raw = PM.r2(Y[:, j], P[:, j])
            rho = float(spearmanr(P[:, j], Y[:, j]).statistic)
            pea = float(pearsonr(P[:, j], Y[:, j])[0])
            a, b = np.polyfit(P[:, j], Y[:, j], 1)
            aff = PM.r2(Y[:, j], a * P[:, j] + b)
            print(f"{o:<7}{raw:>10.2f}{rho:>10.3f}{pea:>10.3f}{aff:>10.3f}")
            results[dom]["outputs"][o] = {
                "unadapted_r2": raw, "spearman": rho, "pearson": pea,
                "affine_corrected_r2": aff,
                "source_range": [float(P[:, j].min()), float(P[:, j].max())],
                "target_range": [float(Y[:, j].min()), float(Y[:, j].max())],
            }

    # the headline contrast, computed rather than asserted
    bm, tb = results["BM"]["outputs"], results["TBCM"]["outputs"]
    results["_contrast_spl"] = {
        "note": "TBCM SPL has the worse unadapted R2 but the better recoverable signal",
        "bm_spl_unadapted_r2": bm["SPL"]["unadapted_r2"],
        "tbcm_spl_unadapted_r2": tb["SPL"]["unadapted_r2"],
        "bm_spl_affine_r2": bm["SPL"]["affine_corrected_r2"],
        "tbcm_spl_affine_r2": tb["SPL"]["affine_corrected_r2"],
        "unadapted_r2_ranks_correctly": tb["SPL"]["unadapted_r2"] > bm["SPL"]["unadapted_r2"],
        "affine_r2_ranks_correctly": tb["SPL"]["affine_corrected_r2"] > bm["SPL"]["affine_corrected_r2"],
    }
    c = results["_contrast_spl"]
    print("\n--- SPL contrast ---")
    print(f"  unadapted R2 picks the more useful source: {c['unadapted_r2_ranks_correctly']}")
    print(f"  affine-corrected R2 picks it:              {c['affine_r2_ranks_correctly']}")

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
