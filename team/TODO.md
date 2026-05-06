# Team TODO

Master task list. Owner field is `brian`, `callum`, `shared`, or `tbd`. Status field is `backlog`, `in-progress`, `review`, or `done`.

When a row reaches `done` and survives a sync cycle, migrate it to `../docs/MILESTONES.md` and remove from this list.

## Active

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 1 | Explore non-transfer alternate methods for BM (BCM→BM without first stage) | brian | done | P1 | Code + real-data results landed 2026-05-05. TabPFN dominates Callum's transfer methods by +0.20 to +0.47 avg R² at small N. See `docs/MILESTONES.md` and `Beam_Membrane/figs/bm_alternates.png` |
| 12 | Run `BM_Alternates.py` on real `dataset_BM.csv` and land actual R² results | brian | done | P1 | Done 2026-05-05. `alternates_results.json` committed; `bm_alternates.png` regenerated |
| 13 | Re-run `BM_SmallData.py` and dump its results to JSON for tighter head-to-head | brian | backlog | P2 | The MILESTONES table compares alternates against PROJECT_GUIDE.md's small-data numbers, but those aren't committed as JSON. Tightens the apples-to-apples comparison |
| 14 | Investigate MonoMLP underperformance at N≤10 — monotonicity prior fighting data | brian | backlog | P2 | MonoMLP R² collapses to negative at N=5 and N=10. Try: λ ramp-up schedule (low at start of training), skip priors below N=20, or relax to λ=0.01 |
| 15 | Upgrade MonoMLP toward an actual PINN if/when feasible | brian | backlog | P3 | Three options if we want to go beyond sign constraints: (a) Titze-style algebraic relations (e.g. F0² ∝ longitudinal stress), (b) generic Laplacian smoothness penalty, (c) sign constraints on second derivatives. Real PDE residuals would require reaching into the BM FEM solver, which we can't |
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
