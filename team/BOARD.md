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

- **#12** — Run `BM_Alternates.py` on real `dataset_BM.csv` and land actual R² results — `brian` — `P1`
  - BM data being copied to local machine. TabPFN backend swapped to `tabpfn-client` (cloud, no model-weight download). Auth: one-time `init()` browser login or `$env:TABPFN_TOKEN`

## In Review

- **#1** — Explore non-transfer alternate methods for BM — `brian` — `P1`
  - All 5 phases complete. Code: GP + PINN + TabPFN in `BM_Alternates.py`; integration in `BM_Summary.py`. Decisions logged. Awaits real-data run (TODO #12) to produce actual R² numbers. Migrate to Recently Done once #12 lands.

## Recently Done

- **2026-05-05** — Non-transfer alternates code complete: GP, PINN (3 monotonicity priors), TabPFN; `BM_Alternates.py` (~430 lines), `BM_Summary.py` extended, `bm_alternates.png` figure. Awaits real-data run (#12 in backlog) — `brian` (TODO #1 → `review`)
- **2026-05-04** — Set up `/team/` shared agile folder (TODO, BOARD, MEETING_NOTES) and link from CLAUDE.md — `brian`
- **2026-05-03** — Merge Callum's `origin/main` into `feature/fem`, reconcile `/docs/` to reflect new structure, push to main — `brian`
- **2026-05-02** — Callum's PR #1 merged: `Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`. 6 RF transfer methods + 3 autoencoder methods, adaptive RF complexity, real measured R² results — `callum`
- **2026-05-02** — Replace `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with `/CLAUDE.md` + `/README.md` + `/docs/{ARCHITECTURE,MILESTONES,ROADMAP,GLOSSARY,DECISIONS}.md` — `brian`

> Items here will migrate to `../docs/MILESTONES.md` after the next sync cycle.
