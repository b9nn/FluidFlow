# 1 o'clock Meeting Notes

Append-only. Newest at the top. Cadence: a few times a week.

Format per entry:
```
## YYYY-MM-DD
**Attendees:** brian, callum
**Decisions:**
- ...
**Action items:** (with owner)
- [task] — owner
**Blockers:** (or "none")
- ...
```

---

## 2026-05-04 (later) — Brian solo (planning + phase 1 of #1)
**Attendees:** brian
**Decisions:**
- Locked in scope of TODO #1 ("alternate methods"): three methods — GP, physics-informed MLP, TabPFN — at N ∈ [5, 10, 20, 30, 50, 75, 100], head-to-head with Callum's TransRF / Feature Aug on the same test split.
- Output lands as new rows in `BM_Summary.py`'s comparison.
- Spec written: `docs/superpowers/specs/2026-05-04-bm-alternates-design.md`.

**Action items:**
- Phase 1 (GP scaffold) — done — `brian`
- Phase 2 (PINN with monotonicity priors) — next — `brian`
- Run BM_Alternates.py against real `dataset_BM.csv` to land Phase 1 actual R² numbers — `brian` (blocked on having data on this machine)

**Blockers:**
- BM dataset not on this clone. Phase 1 was smoke-tested on synthetic data; real numbers pending data load.

## 2026-05-04 — Brian solo (post-hiatus catch-up)
**Attendees:** brian
**Decisions:**
- Adopt `/team/` folder as shared agile coordination layer. Both Claude workflows (Brian's and Callum's) will read it via root `CLAUDE.md`.
- Cadence: a few times a week, no fixed schedule.
- Track ownership per-task (`brian`, `callum`, `shared`, `tbd`) rather than by domain.
- Brian's main thread coming off hiatus: explore alternate methods to BM that don't go through the BCM→BM transfer first (TODO #1).
- TBCM→BM two-stage transfer is on the radar as a separate experiment (TODO #2). Owner to be picked at next 1pm.
- Keep Callum's `PROJECT_GUIDE.md` as his hands-on guide for `Beam_Membrane/` and `TBCM/`. Don't fold into `/docs/` — separate purpose.

**Action items:**
- Verify reproducibility of Callum's BM/TBCM results locally before extending — `brian` (TODO #5)
- Pick up TODO #1 (alternate methods) — `brian`
- Discuss TBCM→BM ownership and scope at next 1pm — `shared`

**Blockers:** none.

**Notes for Callum's next session:** Brian merged `origin/main` into `feature/fem` (clean fast-forward, no force) and pushed `main` up with the doc reconcile commits. No code touched — only `/docs/`, `/CLAUDE.md`, and the new `/team/` folder. Please skim `/team/` in your next Claude session; the root `CLAUDE.md` now points there as the shared coordination layer.
