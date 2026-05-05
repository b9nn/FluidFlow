# Team TODO

Master task list. Owner field is `brian`, `callum`, `shared`, or `tbd`. Status field is `backlog`, `in-progress`, `review`, or `done`.

When a row reaches `done` and survives a sync cycle, migrate it to `../docs/MILESTONES.md` and remove from this list.

## Active

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 1 | Explore non-transfer alternate methods for BM (BCM→BM without first stage) | brian | review | P1 | Code complete (5 phases) — `Beam_Membrane/BM_Alternates.py` with GP+PINN+TabPFN, `BM_Summary.py` extended. Real-data run pending in #12 |
| 12 | Run `BM_Alternates.py` on real `dataset_BM.csv` and land actual R² results | brian | backlog | P1 | Blocker: need BM data on local machine + `TABPFN_TOKEN` for TabPFN. Once run, regenerate `bm_alternates.png` and migrate #1 + #12 to MILESTONES |
| 2 | TBCM→BM two-stage transfer experiment | tbd | backlog | P1 | Use TBCM as cheap-but-closer-to-BM intermediate. Compare against BCM→BM direct. Decide owner at next 1pm |
| 3 | Extend NN partial-freezing transfer to BM and TBCM | brian | backlog | P2 | Brian's NN strategy from VocalFoldRegression has only been used on female BCM. Add as additional method in `BM_Summary.py` / `TBCM_Summary.py` |
| 4 | Extend PR (degree 4–5 + Ridge) transfer to BM and TBCM | brian | backlog | P2 | Same — bring PR into the new domains |
| 5 | Verify reproducibility of Callum's BM/TBCM results | brian | backlog | P1 | Run `BM_Summary.py` and `TBCM_Summary.py` locally; compare against `results/*.json` |
| 6 | Paper write-up draft — domain gap & transfer story | shared | backlog | P2 | Combine BCM→TBCM (easy case) and BCM→BM (hard case) into a "transfer for expensive simulators" narrative |
| 7 | Decide fate of `VocalFoldRegression/Beam+Membrane_ForSean/` (Sean's MATLAB) | brian | backlog | P2 | Currently untracked locally. Either get Sean's consent to vendor, or document as required-external-dep |

## Maybe-later

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 8 | Female BM / female TBCM transfer | tbd | backlog | P2 | Only if female datasets exist for those models |
| 9 | Glottal area integration as additional feature | tbd | backlog | P2 | `glottal_area/` scripts are dormant |
| 10 | OpenIFEM coupling for full FSI training data | tbd | backlog | P2 | Replaces reduced-order BM with full FSI |
| 11 | Hyperparameter sweep tracker (CSV log or MLflow) | tbd | backlog | P2 | Once we're running enough variations to need it |
