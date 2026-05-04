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

## 2026-05-04 — Callum async (responding to Brian's onboarding notes)
**Attendees:** callum
**Decisions:**
- Callum takes ownership of #2 (TBCM→BM two-stage transfer) — was considering this as natural follow-up to BCM→BM work. TBCM is closer to BM than BCM is, so it could be a better intermediate.
- #5 (verify reproducibility) — go ahead, Brian. Results were verified during the repo cleanup session but independent confirmation is good practice.
- Added two in-progress tasks: #12 (three new domain-gap bridging approaches) and #13 (small-data best-practice recommendations). These were the active threads before the repo cleanup/handoff.

**Action items:**
- Implement calibrated source, input-space filtering, and uncertainty features for BCM→BM small-data transfer — `callum` (#12)
- Synthesize small-data results into actionable recommendations — `callum` (#13)
- TBCM→BM two-stage transfer scoping — `callum` (#2, after #12 completes)

**Blockers:** none.

**Notes for Brian:** Re the TBCM→BM question — yes, I'd been thinking about it. The logic: TBCM shares the lumped-element physics with BCM but we showed BCM→TBCM transfer is very strong (R²=0.96 at 5%). If we can also get TBCM→BM to work, the chain BCM→TBCM→BM might outperform direct BCM→BM since TBCM is a closer intermediate. Will scope this properly after finishing the current domain-gap bridging work (#12).

---

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
