# Team TODO

Master task list. Owner field is `brian`, `callum`, `shared`, or `tbd`. Status field is `backlog`, `in-progress`, `review`, or `done`.

When a row reaches `done` and survives a sync cycle, migrate it to `../docs/MILESTONES.md` and remove from this list.

## Active

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 1 | Explore non-transfer alternate methods for BM (BCM→BM without first stage) | brian | done | P1 | Code + real-data results landed 2026-05-05. TabPFN dominates Callum's transfer methods by +0.20 to +0.47 avg R² at small N. See `docs/MILESTONES.md` and `Beam_Membrane/figs/bm_alternates.png` |
| 12 | Run `BM_Alternates.py` on real `dataset_BM.csv` and land actual R² results | brian | done | P1 | Done 2026-05-05. `alternates_results.json` committed; `bm_alternates.png` regenerated |
| 13 | Re-run `BM_SmallData.py` and dump its results to JSON for tighter head-to-head | brian | backlog | P2 | The MILESTONES table compares alternates against PROJECT_GUIDE.md's small-data numbers, but those aren't committed as JSON. Tightens the apples-to-apples comparison |
| 14 | Investigate MonoMLP underperformance at N≤10 — monotonicity prior fighting data | brian | done | P2 | Closed 2026-05-06: MonoMLP removed from active code (mid-tier, didn't add to story). See `docs/DECISIONS.md` 2026-05-06 removal entry |
| 15 | Build a real PDE-residual PINN over BM governing equations | brian | backlog | P3 | The PDEs are now extracted in `docs/BM_GOVERNING_EQUATIONS.md`. Multi-week project: port stages 2/3/5 to PyTorch, use NN to predict displacement fields with PDE-residual losses at collocation points, derive F0/SPL post-hoc. Different deliverable from GP/TabPFN — physics surrogate that generalizes OOD and gives gradients |
| 2 | TBCM→BM two-stage transfer experiment | tbd | backlog | P1 | Use TBCM as cheap-but-closer-to-BM intermediate. Compare against BCM→BM direct. Decide owner at next 1pm |
| 3 | Extend NN partial-freezing transfer to BM and TBCM | brian | backlog | P2 | Brian's NN strategy from VocalFoldRegression has only been used on female BCM. Add as additional method in `BM_Summary.py` / `TBCM_Summary.py` |
| 4 | Extend PR (degree 4–5 + Ridge) transfer to BM and TBCM | brian | backlog | P2 | Same — bring PR into the new domains |
| 5 | Verify reproducibility of Callum's BM/TBCM results | brian | backlog | P1 | Run `BM_Summary.py` and `TBCM_Summary.py` locally; compare against `results/*.json` |
| 6 | Paper write-up draft — domain gap & transfer story | shared | backlog | P2 | Combine BCM→TBCM (easy case) and BCM→BM (hard case) into a "transfer for expensive simulators" narrative |
| 7 | Decide fate of `VocalFoldRegression/Beam+Membrane_ForSean/` (Sean's MATLAB) | brian | backlog | P2 | Currently untracked locally. Either get Sean's consent to vendor, or document as required-external-dep |
| 16 | N=20 panel in BM bootstrap boxplot | brian | done | P2 | Done 2026-05-12. Commit `4d5b747`. At N=20 GP/TabPFN medians ≈ 0.38 vs best-transfer (Feature Aug) at 0.054 |
| 17 | Cross-domain GP/TabPFN: TBCM + Female BCM (2026-05-12 advisor ask) | brian | review | P1 | DONE 2026-05-13. All three panels real. Female: commits `50837fa`, `23351e7`. TBCM: commits `0d40466` GP, `5c45242` TabPFN, `31c118f` transfer JSON. Cross-domain fig regen `94fe23d`. Cross-domain headline refined: alternates dominate BCM→BM (gap +0.40 at N=50), TabPFN edges transfer on Male→Female (gap +0.06 at N=100, converges by N=500), and on well-aligned BCM→TBCM transfer essentially ties TabPFN at N=500 (0.972 vs 0.972). |
| 17b | TBCM cross-domain work (data-gated): TBCM_GP, TBCM_TabPFN, TBCM_SmallData JSON dump | brian | done | P1 | DONE 2026-05-13. Plan Tasks 2 / 3 / 5.5 all landed: commits `0d40466`, `5c45242`, `31c118f`. Dataset `TBCM/dataset_TBCM.csv` (43,102 rows) arrived locally, stays gitignored per CSV rule. Cross-domain figure regenerated in `94fe23d`. |
| 18 | Muscle-activation x F0 heatmaps (Jesus 2026-05-12 ask) | brian | review | P1 | DONE 2026-05-13. All three panels present: `heatmap_BM_F0.png`, `heatmap_FemaleBCM_F0.png`, `heatmap_TBCM_F0.png`. Used matplotlib.use('Agg') after tkinter crash on Windows. |
| 18b | TBCM F0 heatmap (data-gated) | brian | done | P1 | DONE 2026-05-13. `Beam_Membrane/figs/heatmap_TBCM_F0.png` committed in `94fe23d`. No code change needed — existing `BM_Heatmaps.py` handles TBCM when CSV exists. |
| 19 | Group email draft — Sean/Jesus/Emiro/Matias follow-up | brian | review | P1 | Draft committed 2026-05-12 (`1fdd6e7`). TBCM numbers now unblocked as of 2026-05-13 (#17b done). Brian still needs to: (a) fill the `<FILL>` placeholders with real TBCM cross-domain R² (now available — TransRF N=500 = 0.972 ≈ TabPFN N=500 = 0.972), (b) strip "Draft notes" block, (c) attach `cross_domain_alternates.png` + 3 heatmap PNGs, (d) send. |

## Maybe-later

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 8 | Female BM / female TBCM transfer | tbd | backlog | P2 | Only if female datasets exist for those models |
| 9 | Glottal area integration as additional feature | tbd | backlog | P2 | `glottal_area/` scripts are dormant |
| 10 | OpenIFEM coupling for full FSI training data | tbd | backlog | P2 | Replaces reduced-order BM with full FSI |
| 11 | Hyperparameter sweep tracker (CSV log or MLflow) | tbd | backlog | P2 | Once we're running enough variations to need it |
