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

## 2026-05-06 (later) — Brian solo (file split)
**Attendees:** brian
**Decisions:**
- Split `BM_Alternates.py` into `BM_GP.py` and `BM_TabPFN.py`. Self-contained per-method-family files matching Callum's existing convention. Both still write to the same `alternates_results.json` so `BM_Summary.py` is unchanged.

**Action items:** none — code-organization-only change.

**Blockers:** none.

## 2026-05-06 — Brian solo (cleanup, BM equations extracted)
**Attendees:** brian
**Decisions:**
- **Removed MonoMLP** (formerly "PINN") from active alternates code. Mid-tier result, didn't add to the GP/TabPFN story; kept the torch dep around for one second-rate baseline. JSON key dropped, figure regenerated, TODO #14 closed.
- **Extracted BM governing equations** from Sean's MATLAB into `docs/BM_GOVERNING_EQUATIONS.md`. Two coupled PDEs (membrane + beam) with full BC/IC, plus the constitutive algebra, aerodynamic Bernoulli model, WRA acoustic propagation, and post-hoc F0/SPL extraction. ~Citations to MATLAB line numbers throughout.
- **Real PINN scoped as TODO #15.** Different deliverable than GP/TabPFN — a PDE-residual physics surrogate that generalizes OOD and gives gradients for inverse problems. Multi-week project.

**Action items:**
- Decide whether to actually pursue #15 vs other research threads — `brian` (and discuss with Callum at next 1pm)
- Pick up another P1 item: #2 (TBCM→BM transfer), #5 (verify Callum's results), or #13 (BM_SmallData JSON dump)

**Blockers:** none.

## 2026-05-05 (later, again) — Brian solo (real-data run; TODO #1 + #12 closed)
**Attendees:** brian
**Decisions:**
- BM data extracted from `~/Downloads/dataset_BM.zip` to `Beam_Membrane/dataset_BM.csv`. 5,000 rows, no NaN, ranges match Callum's spec.
- `BM_Alternates.py` ran end-to-end (~3 min total). `BM_Summary.py` regenerated all three figures.
- **Headline result:** non-transfer alternates beat Callum's transfer methods at every N ∈ [10, 100]. TabPFN especially. Best gain: +0.47 R² at N=50. See `MILESTONES.md` 2026-05-05 entry for the table.
- This reframes the transfer story — at small N, BCM source is a liability, not an asset. TBCM→BM (TODO #2) might still help since TBCM is physics-closer; worth scoping.
- Two follow-up TODOs created (#13: dump JSON for `BM_SmallData.py` to tighten comparison; #14: investigate MonoMLP failure at N≤10).
- TODO #1 + #12 marked `done`. Migrated to Recently Done in BOARD; will roll into MILESTONES at next sync.

**Action items:**
- Discuss TBCM→BM ownership (#2) at next 1pm with Callum given the new framing — `shared`
- Pick up #13 or #14 next, or pivot to #2 — `brian`

**Blockers:** none.

## 2026-05-05 (later) — Brian solo (TabPFN backend swap, BM data inbound)
**Attendees:** brian
**Decisions:**
- TabPFN backend swap from `tabpfn` (local, license-gated) to `tabpfn-client` (cloud, account-gated). Code now imports `tabpfn-client` first and falls back to `tabpfn` if only that is installed. Logged to DECISIONS.
- Auth helper added: reads `TABPFN_TOKEN` env var if set, else relies on cached interactive login from `tabpfn_client.init()`.
- Add `.env*` to `.gitignore` for safety even though no `.env` is currently in the project.
- TODO #12 status moves from `backlog` → `in-progress` since BM data is being copied locally.

**Action items:**
- Run `python -c "from tabpfn_client import init; init()"` once to do browser login — `brian`
- Drop `dataset_BM.csv` into `Beam_Membrane/` and run `python Beam_Membrane/BM_Alternates.py` — `brian`
- Then run `python Beam_Membrane/BM_Summary.py` to regenerate `bm_alternates.png` with real numbers — `brian`
- After numbers land: migrate TODO #1 + #12 to MILESTONES — `brian`

**Blockers:** none — data inbound, code unblocked.

## 2026-05-05 — Brian solo (phases 2–5 of #1, code complete)
**Attendees:** brian
**Decisions:**
- MonoMLP encodes 3 monotonicity priors: `∂F0/∂a_CT`, `∂SPL/∂PS`, and `∂F0/∂PS` (chest-voice physiology). λ = 0.1 default. Logged to DECISIONS.
- TabPFN ≥ 7.x license + token requirement turned out to be a setup blocker — handled with graceful skip; documented in code and in TODO #12.
- TODO #1 status moves from `in-progress` to `review` since the implementation work is complete; the actual real-data run is split out as a new TODO #12 that's the prerequisite for migrating both rows to MILESTONES.

**Action items:**
- Phase 5 (DECISIONS, MILESTONES, BOARD/TODO migration) — done — `brian`
- TODO #12 (real-data run) — backlog — `brian` — needs `dataset_BM.csv` on local machine + `TABPFN_TOKEN`
- At next 1pm with Callum: walk through `bm_alternates.png` once #12 has actual numbers, decide whether the alternates change his transfer story

**Blockers:**
- BM data + TabPFN license/token (both deferred to #12).

## 2026-05-04 (later) — Brian solo (planning + phase 1 of #1)
**Attendees:** brian
**Decisions:**
- Locked in scope of TODO #1 ("alternate methods"): three methods — GP, physics-informed MLP, TabPFN — at N ∈ [5, 10, 20, 30, 50, 75, 100], head-to-head with Callum's TransRF / Feature Aug on the same test split.
- Output lands as new rows in `BM_Summary.py`'s comparison.
- Spec written: `docs/superpowers/specs/2026-05-04-bm-alternates-design.md`.

**Action items:**
- Phase 1 (GP scaffold) — done — `brian`
- Phase 2 (MonoMLP with 3 monotonicity priors: `∂F0/∂a_CT`, `∂SPL/∂PS`, `∂F0/∂PS`) — done — `brian`
- Phase 3 (TabPFN integration) — done — `brian`
- Phase 4 (BM_Summary integration) — next — `brian`
- Phase 5 (DECISIONS + MILESTONES + close TODO #1) — pending — `brian`
- One-time: accept TabPFN license at https://ux.priorlabs.ai, set `TABPFN_TOKEN` env var — `brian`
- Run BM_Alternates.py against real `dataset_BM.csv` to land actual R² numbers — `brian` (blocked on having data on this machine)

**Blockers:**
- BM dataset not on this clone. GP+MonoMLP smoke-tested on synthetic data; TabPFN couldn't be smoke-tested (needs license + token). Real numbers pending data + license.

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
