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

## In Progress

- **#1** — Explore non-transfer alternate methods for BM (BCM→BM without first stage) — `brian` — `P1`
  - Phase 1/5 done: GP method (Matern 2.5 + WhiteKernel)
  - Phase 2/5 done: PINN — MLP `[3→32→32→2]` + monotonicity penalty with **3** priors (`∂F0/∂a_CT ≥ 0`, `∂SPL/∂PS ≥ 0`, `∂F0/∂PS ≥ 0`), λ=0.1, early stopping
  - Both smoke-tested on synthetic data; needs real BM data to land actual results
  - Next: Phase 3 (TabPFN), Phase 4 (BM_Summary integration), Phase 5 (wrap-up)

## In Review

_(empty)_

## Recently Done

- **2026-05-04** — Set up `/team/` shared agile folder (TODO, BOARD, MEETING_NOTES) and link from CLAUDE.md — `brian`
- **2026-05-03** — Merge Callum's `origin/main` into `feature/fem`, reconcile `/docs/` to reflect new structure, push to main — `brian`
- **2026-05-02** — Callum's PR #1 merged: `Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`. 6 RF transfer methods + 3 autoencoder methods, adaptive RF complexity, real measured R² results — `callum`
- **2026-05-02** — Replace `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with `/CLAUDE.md` + `/README.md` + `/docs/{ARCHITECTURE,MILESTONES,ROADMAP,GLOSSARY,DECISIONS}.md` — `brian`

> Items here will migrate to `../docs/MILESTONES.md` after the next sync cycle.
