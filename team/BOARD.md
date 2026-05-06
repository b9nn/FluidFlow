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
- **#14** — Investigate MonoMLP underperformance at N≤10 (priors fighting data) — `brian` — `P2`
- **#15** — Upgrade MonoMLP toward an actual PINN if feasible — `brian` — `P3`

## In Progress

_(empty — pick up #2, #5, or #13/#14 follow-ups)_

## In Review

_(empty)_

## Recently Done

- **2026-05-05** — **TODO #1 + #12 done.** Non-transfer alternates ran on real BM data. Headline: TabPFN at N=50 hits avg R²=0.66, beating Callum's best transfer (Feature Aug, R²=0.19) by +0.47. GP is comparable. MonoMLP struggles at N≤10. Real result, possibly publishable. See `docs/MILESTONES.md` and `Beam_Membrane/figs/bm_alternates.png` — `brian`
- **2026-05-05** — Non-transfer alternates code complete: GP, MonoMLP (3 monotonicity priors), TabPFN; `BM_Alternates.py` (~440 lines), `BM_Summary.py` extended — `brian`
- **2026-05-04** — Set up `/team/` shared agile folder (TODO, BOARD, MEETING_NOTES) and link from CLAUDE.md — `brian`
- **2026-05-03** — Merge Callum's `origin/main` into `feature/fem`, reconcile `/docs/` to reflect new structure, push to main — `brian`
- **2026-05-02** — Callum's PR #1 merged: `Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`. 6 RF transfer methods + 3 autoencoder methods, adaptive RF complexity, real measured R² results — `callum`
- **2026-05-02** — Replace `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with `/CLAUDE.md` + `/README.md` + `/docs/{ARCHITECTURE,MILESTONES,ROADMAP,GLOSSARY,DECISIONS}.md` — `brian`

> Items here will migrate to `../docs/MILESTONES.md` after the next sync cycle.
