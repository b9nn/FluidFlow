# Team TODO

Master task list. Owner field is `brian`, `callum`, `shared`, or `tbd`. Status field is `backlog`, `in-progress`, `review`, or `done`.

When a row reaches `done` and survives a sync cycle, migrate it to `../docs/MILESTONES.md` and remove from this list.

## Active

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 1 | Explore non-transfer alternate methods for BM (BCM→BM without first stage) | brian | in-progress | P1 | Brian's main thread post-hiatus. Goal: see if direct BM modeling beats transfer at certain sample regimes. TabPFN + GP scripts (`BM_TabPFN.py`, `BM_GP.py`) ingested from feature/fem into dev on 2026-05-31. |
| 14 | FemaleBCM TabPFN heatmap analysis at data fractions | callum | in-progress | P1 | Notebook-style (imshow+contour) heatmaps for F0 and SPL comparing TabPFN at N=10/50/200/500 vs RF ground truth. Script: `VocalFoldRegression/BCM Model/Alternates/BCM_TabPFN_Heatmaps.py`. TabPFN auth needed to run. |
| 15 | JASA TBCM TabPFN experiments (multi-dim features, missing data, compute time) | callum | in-progress | P1 | Jesus's enriched TBCM data (`TBCM/dataset_JASA.csv`, 80k rows). Three experiments from 2026-06-04 advisor mtg: (A) 6 clinical outputs R² vs N, (B) high-corner extrapolation vs random holdout, (C) TabPFN vs RF compute time vs N. Script: `TBCM/JASA_TabPFN_Experiments.py`. N≤1000. See DECISIONS 2026-06-16. |
| 2 | TBCM→BM two-stage transfer experiment | callum | in-progress | P1 | Use TBCM as cheap-but-closer-to-BM intermediate. Compare against BCM→BM direct. Script done: `BM_TransferRF_TBCM.py` (8 methods incl chain). Chain TransRF wins at most fractions. Needs review. |
| 16 | Draft 5: generate all missing paper statistics, figures and tables | callum | review | P1 | Aug-20 deadline from 2026-08-06 advisor mtg. Delivered: 7-config protocol on BM+TBCM (5 seeds, N=10-500), Table 1, Table 2, Fig 1 (`bm_ext_metrics_vs_n.png`), Fig 2 (`tbcm_motor_map_F0.png`), Sec 2.2 hyperparams, Sec 3.3 written. `paper_out/FF_Draft_5.pdf` + `sanity_check.py` (23 pass / 0 fail). NOTE: rerun contradicts Draft 4's "TabPFN wins on both targets" - transfer wins on TBCM at N<=50. See DECISIONS 2026-08-20. |
| 3 | Extend NN partial-freezing transfer to BM and TBCM | brian | backlog | P2 | Brian's NN strategy from VocalFoldRegression has only been used on female BCM. Add as additional method in `BM_Summary.py` / `TBCM_Summary.py` |
| 4 | Extend PR (degree 4–5 + Ridge) transfer to BM and TBCM | brian | backlog | P2 | Same — bring PR into the new domains |
| 5 | Verify reproducibility of Callum's BM/TBCM results | brian | backlog | P1 | Run `BM_Summary.py` and `TBCM_Summary.py` locally; compare against `results/*.json` |
| 6 | Paper write-up draft — domain gap & transfer story | shared | backlog | P2 | Combine BCM→TBCM (easy case) and BCM→BM (hard case) into a "transfer for expensive simulators" narrative |
| 7 | Decide fate of `VocalFoldRegression/Beam+Membrane_ForSean/` (Sean's MATLAB) | brian | backlog | P2 | Currently untracked locally. Either get Sean's consent to vendor, or document as required-external-dep |
| 12 | BCM→BM domain-gap bridging: calibrated source, input-space filtering, uncertainty features | callum | in-progress | P1 | Three new approaches to improve small-data transfer: (1) linear pre-alignment of BCM outputs, (2) retrain BCM on Ps∈[600,1000] only, (3) RF tree variance as extra features. Compare at N=10–500 |
| 13 | BCM→BM small-data regime analysis and best-practice recommendations | callum | in-progress | P1 | Synthesize results from `BM_SmallData.py` + new methods into actionable guidance: which method to use at which sample count |

## Maybe-later

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 8 | Female BM / female TBCM transfer | tbd | backlog | P2 | Only if female datasets exist for those models |
| 9 | Glottal area integration as additional feature | tbd | backlog | P2 | `glottal_area/` scripts are dormant |
| 10 | OpenIFEM coupling for full FSI training data | tbd | backlog | P2 | Replaces reduced-order BM with full FSI |
| 11 | Hyperparameter sweep tracker (CSV log or MLflow) | tbd | backlog | P2 | Once we're running enough variations to need it |
