# Team Board

Visual snapshot of where work is. Source-of-truth fields are in [`TODO.md`](TODO.md); this is the kanban view.

Format per row: `#N — Task — owner — Pn`. Click through to TODO for full notes.

---

## Backlog

- **#2** — TBCM→BM two-stage transfer experiment — `tbd` — `P1`
- **#3** — Extend NN partial-freezing transfer to BM and TBCM — `brian` — `P2`
- **#4** — Extend PR (degree 4–5 + Ridge) transfer to BM and TBCM — `brian` — `P2`
- **#5** — Verify reproducibility of Callum's BM/TBCM results — `brian` — `P1`
- **#6** — Paper write-up draft — domain gap & transfer story — `shared` — `P2`
- **#7** — Decide fate of Sean's MATLAB folder — `brian` — `P2`
- **#8** — Female BM / female TBCM transfer — `tbd` — `P2`
- **#9** — Glottal area integration as additional feature — `tbd` — `P2`
- **#10** — OpenIFEM coupling for full FSI training data — `tbd` — `P2`
- **#11** — Hyperparameter sweep tracker — `tbd` — `P2`
- **#13** — Re-run `BM_SmallData.py` and dump JSON for tighter head-to-head — `brian` — `P2`
- **#15** — Build a real PDE-residual PINN over BM governing equations — `brian` — `P3`
- **#16** — N=20 panel in BM bootstrap boxplot — `brian` — `P2` _(2026-05-12 advisor follow-up, plan pending approval)_
- **#17** — Cross-domain GP/TabPFN: TBCM + Female BCM — `brian` — `P1` _(2026-05-12 advisor follow-up, plan pending approval)_
- **#17b** — Dump TBCM small-N transfer to JSON — `brian` — `P1` _(Sub-task of #17, plan Task 5.5)_
- **#18** — Muscle-activation x F0 heatmaps — `brian` — `P1` _(2026-05-12 advisor follow-up, plan pending approval)_
- **#19** — Group email draft — `brian` — `P1` _(2026-05-12 advisor follow-up, plan pending approval)_

## In Progress

_(empty — pick up #2, #5, or #13/#14 follow-ups)_

## In Review

_(empty)_

## Recently Done

- **2026-05-12 (later)** — Extended alternates to N=500 (was N≤100). `BM_GP.py` and `BM_TabPFN.py` now run at N ∈ {…, 150, 200, 300, 500}, with per-N JSON merge so existing N values are preserved. Showcase figures regenerated: sim-budget R²≥0.7 panel now resolves (TabPFN N=111, GP N=114 vs transfer N=396–463); bootstrap fig adds N=200, N=500 panels. Even at N=500 alternates lead transfer by +0.17 R² — `brian`
- **2026-05-12** — Showcase figures for advisor / paper. New `Beam_Membrane/BM_Showcase.py` produces three figures: `bm_showcase_headline.png` (avg R² vs N with annotated +0.47 gap), `bm_showcase_sim_budget.png` (TabPFN N=32 vs transfer N=165 to hit R²=0.5), `bm_showcase_bootstrap.png` (10-replicate boxplots beating transfer at N≥20). README + MILESTONES updated — `brian`
- **2026-05-06 (later)** — Split `BM_Alternates.py` into `Beam_Membrane/BM_GP.py` and `Beam_Membrane/BM_TabPFN.py`. Self-contained per-method-family files matching Callum's convention; both merge into the shared `alternates_results.json`. Delete merged file. Update README quick-start and glossary pointers — `brian`
- **2026-05-06** — Code cleanup: MonoMLP method removed from `BM_Alternates.py` (mid-tier, didn't add to GP/TabPFN story); JSON key dropped; figure regenerated; torch import no longer needed for alternates. Closes TODO #14. Real PDE-residual PINN scoped as TODO #15 (separate project, equations extracted in `docs/BM_GOVERNING_EQUATIONS.md`). BM governing equations doc added — `brian`
- **2026-05-05** — **TODO #1 + #12 done.** Non-transfer alternates ran on real BM data. Headline: TabPFN at N=50 hits avg R²=0.66, beating Callum's best transfer (Feature Aug, R²=0.19) by +0.47. GP is comparable. MonoMLP struggles at N≤10. Real result, possibly publishable. See `docs/MILESTONES.md` and `Beam_Membrane/figs/bm_alternates.png` — `brian`
- **2026-05-05** — Non-transfer alternates code complete: GP, MonoMLP (3 monotonicity priors), TabPFN; `BM_Alternates.py` (~440 lines), `BM_Summary.py` extended — `brian`
- **2026-05-04** — Set up `/team/` shared agile folder (TODO, BOARD, MEETING_NOTES) and link from CLAUDE.md — `brian`
- **2026-05-03** — Merge Callum's `origin/main` into `feature/fem`, reconcile `/docs/` to reflect new structure, push to main — `brian`
- **2026-05-02** — Callum's PR #1 merged: `Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`. 6 RF transfer methods + 3 autoencoder methods, adaptive RF complexity, real measured R² results — `callum`
- **2026-05-02** — Replace `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with `/CLAUDE.md` + `/README.md` + `/docs/{ARCHITECTURE,MILESTONES,ROADMAP,GLOSSARY,DECISIONS}.md` — `brian`

> Items here will migrate to `../docs/MILESTONES.md` after the next sync cycle.
