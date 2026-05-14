# Team Board

Visual snapshot of where work is. Source-of-truth fields are in [`TODO.md`](TODO.md); this is the kanban view.

Format per row: `#N — Task — owner — Pn`. Click through to TODO for full notes.

---

## Backlog

- **#2** — TBCM→BM two-stage transfer experiment — `tbd` — `P1`
- **#3** — Extend NN partial-freezing transfer to BM and TBCM — `ben` — `P2`
- **#4** — Extend PR (degree 4–5 + Ridge) transfer to BM and TBCM — `ben` — `P2`
- **#5** — Verify reproducibility of Callum's BM/TBCM results — `ben` — `P1`
- **#6** — Paper write-up draft — domain gap & transfer story — `shared` — `P2`
- **#7** — Decide fate of Sean's MATLAB folder — `ben` — `P2`
- **#8** — Female BM / female TBCM transfer — `tbd` — `P2`
- **#9** — Glottal area integration as additional feature — `tbd` — `P2`
- **#10** — OpenIFEM coupling for full FSI training data — `tbd` — `P2`
- **#11** — Hyperparameter sweep tracker — `tbd` — `P2`
- **#13** — Re-run `BM_SmallData.py` and dump JSON for tighter head-to-head — `ben` — `P2`
- **#15** — Build a real PDE-residual PINN over BM governing equations — `ben` — `P3`

## In Progress

_(empty)_

## In Review

- **#17** — Cross-domain GP/TabPFN: TBCM + Female BCM — `ben` — `P1` _(all three panels real; cross-domain headline refined)_
- **#18** — Muscle-activation x F0 heatmaps — `ben` — `P1` _(BM + Female + TBCM all done)_
- **#19** — Group email draft — `ben` — `P1` _(draft done; awaits Ben filling TBCM numbers, attachments, send)_

## Recently Done

- **2026-05-13** — TBCM cross-domain unblock. Dataset arrived locally (43,102 rows, gitignored). Three commits: `0d40466` TBCM_GP (mirrors BM_GP.py), `5c45242` TBCM_TabPFN (mirrors BM_TabPFN.py — included one-off fix to a 0-byte tabpfn-client cache corrupted by overnight computer reset), `31c118f` TBCM_SmallData JSON dump (Plan Task 5.5). Cross-domain figure regenerated `94fe23d` with TBCM F0 heatmap. Refined cross-domain story: TBCM TransRF at N=500 hits 0.972 — within 0.001 of TabPFN; gap at N=100 only +0.07. Confirms thesis "alternates dominate when source-target alignment is poor (BCM→BM gap +0.40 at N=50); transfer competes and at large N matches when alignment is good (BCM→TBCM, Male→Female)". Closes #17b, #18b — `ben`

- **2026-05-12 (later 3)** — Implemented 2026-05-12 advisor follow-up: N=20 boxplot panel (`4d5b747`), Female_GP (`50837fa`) and Female_TabPFN (`23351e7`) on female BCM, cross-domain figure (`fe6b642`), BM + Female F0 heatmaps (`38bfe81`), group email draft (`1fdd6e7`). TBCM portions remain queued (#17b, #18b) pending dataset arrival. Refined story finding: TabPFN catches Male→Female RF transfer at N≈75 and dominates from N=100, even though that transfer is well-aligned — `ben`

- **2026-05-12 (later)** — Extended alternates to N=500 (was N≤100). `BM_GP.py` and `BM_TabPFN.py` now run at N ∈ {…, 150, 200, 300, 500}, with per-N JSON merge so existing N values are preserved. Showcase figures regenerated: sim-budget R²≥0.7 panel now resolves (TabPFN N=111, GP N=114 vs transfer N=396–463); bootstrap fig adds N=200, N=500 panels. Even at N=500 alternates lead transfer by +0.17 R² — `ben`
- **2026-05-12** — Showcase figures for advisor / paper. New `Beam_Membrane/BM_Showcase.py` produces three figures: `bm_showcase_headline.png` (avg R² vs N with annotated +0.47 gap), `bm_showcase_sim_budget.png` (TabPFN N=32 vs transfer N=165 to hit R²=0.5), `bm_showcase_bootstrap.png` (10-replicate boxplots beating transfer at N≥20). README + MILESTONES updated — `ben`
- **2026-05-06 (later)** — Split `BM_Alternates.py` into `Beam_Membrane/BM_GP.py` and `Beam_Membrane/BM_TabPFN.py`. Self-contained per-method-family files matching Callum's convention; both merge into the shared `alternates_results.json`. Delete merged file. Update README quick-start and glossary pointers — `ben`
- **2026-05-06** — Code cleanup: MonoMLP method removed from `BM_Alternates.py` (mid-tier, didn't add to GP/TabPFN story); JSON key dropped; figure regenerated; torch import no longer needed for alternates. Closes TODO #14. Real PDE-residual PINN scoped as TODO #15 (separate project, equations extracted in `docs/BM_GOVERNING_EQUATIONS.md`). BM governing equations doc added — `ben`
- **2026-05-05** — **TODO #1 + #12 done.** Non-transfer alternates ran on real BM data. Headline: TabPFN at N=50 hits avg R²=0.66, beating Callum's best transfer (Feature Aug, R²=0.19) by +0.47. GP is comparable. MonoMLP struggles at N≤10. Real result, possibly publishable. See `docs/MILESTONES.md` and `Beam_Membrane/figs/bm_alternates.png` — `ben`
- **2026-05-05** — Non-transfer alternates code complete: GP, MonoMLP (3 monotonicity priors), TabPFN; `BM_Alternates.py` (~440 lines), `BM_Summary.py` extended — `ben`
- **2026-05-04** — Set up `/team/` shared agile folder (TODO, BOARD, MEETING_NOTES) and link from CLAUDE.md — `ben`
- **2026-05-03** — Merge Callum's `origin/main` into `feature/fem`, reconcile `/docs/` to reflect new structure, push to main — `ben`
- **2026-05-02** — Callum's PR #1 merged: `Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`. 6 RF transfer methods + 3 autoencoder methods, adaptive RF complexity, real measured R² results — `callum`
- **2026-05-02** — Replace `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with `/CLAUDE.md` + `/README.md` + `/docs/{ARCHITECTURE,MILESTONES,ROADMAP,GLOSSARY,DECISIONS}.md` — `ben`

> Items here will migrate to `../docs/MILESTONES.md` after the next sync cycle.
