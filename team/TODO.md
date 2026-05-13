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
| 17 | Cross-domain GP/TabPFN: TBCM + Female BCM (2026-05-12 advisor ask) | brian | review | P1 | Female portion DONE (commits `50837fa` Female_GP, `23351e7` Female_TabPFN, `fe6b642` cross-domain figure). TBCM portion gated → #17b. Surprise finding: TabPFN catches transfer at N≈75 for well-aligned Male→Female, vs N=10 for misaligned BCM→BM |
| 17b | TBCM cross-domain work (data-gated): TBCM_GP, TBCM_TabPFN, TBCM_SmallData JSON dump | brian | backlog | P1 | Plan Tasks 2, 3, 5.5. All three gated on `dataset_TBCM.csv` not being on this clone. Ask Callum at next 1pm sync. Once unblocked, also regenerate cross-domain figure (`Beam_Membrane/BM_CrossDomain.py`) — TBCM panel currently shows "no data yet" |
| 18 | Muscle-activation x F0 heatmaps (Jesus 2026-05-12 ask) | brian | review | P1 | BM + Female DONE (commit `38bfe81`). TBCM heatmap deferred → #18b. Used matplotlib.use('Agg') after tkinter crash on Windows |
| 18b | TBCM F0 heatmap (data-gated) | brian | backlog | P1 | Mirror of BM/Female panels in `Beam_Membrane/BM_Heatmaps.py`; script already handles TBCM if `dataset_TBCM.csv` exists. Pure data-gated, no new code needed |
| 19 | Group email draft — Sean/Jesus/Emiro/Matias follow-up | brian | done | P1 | Done 2026-05-12. Commit `1fdd6e7`. Draft at `docs/superpowers/email-draft-2026-05-12-vocal-fold-update.md`. Strip "Draft notes" block before sending. Awaiting TBCM numbers before final send |

## Maybe-later

| # | Task | Owner | Status | Priority | Notes |
|---|---|---|---|---|---|
| 8 | Female BM / female TBCM transfer | tbd | backlog | P2 | Only if female datasets exist for those models |
| 9 | Glottal area integration as additional feature | tbd | backlog | P2 | `glottal_area/` scripts are dormant |
| 10 | OpenIFEM coupling for full FSI training data | tbd | backlog | P2 | Replaces reduced-order BM with full FSI |
| 11 | Hyperparameter sweep tracker (CSV log or MLflow) | tbd | backlog | P2 | Once we're running enough variations to need it |
