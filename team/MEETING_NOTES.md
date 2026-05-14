# 1 o'clock Meeting Notes

Append-only. Newest at the top. Cadence: a few times a week.

Format per entry:
```
## YYYY-MM-DD
**Attendees:** ben, callum
**Decisions:**
- ...
**Action items:** (with owner)
- [task] — owner
**Blockers:** (or "none")
- ...
```

---

## 2026-05-13 — Ben solo (TBCM cross-domain unblock — #17b + #18b)

**Attendees:** ben

**Decisions:**
- TBCM dataset (`TBCM/dataset_TBCM.csv`, 43,102 rows, gitignored per CSV
  convention) arrived locally between 2026-05-12 night and 2026-05-13
  morning. GP commit `0d40466` landed late 2026-05-12 (after the advisor
  follow-up tier). TabPFN run that night was interrupted at N=100 because
  the local `tabpfn_client/.tabpfn/dataset_cache` file got truncated to 0
  bytes during the user-machine reset, raising `JSONDecodeError` on next
  import. Resolution: delete the 0-byte cache (auth `config` untouched);
  the idempotent `_existing_complete_ns` skip logic in TBCM_TabPFN.py
  picked up exactly at N=150 and filled {150, 200, 300, 500} cleanly.
- Plan Task 5.5 (TBCM_SmallData JSON dump) executed in parallel and
  produced `TBCM/results/rf_transfer_small_n.json` — per-N replicate RF
  transfer scores schema-matched to `alternates_results.json` for
  `BM_CrossDomain.py` to consume.
- Cross-domain figure regenerated. TBCM panel now real (was "no data
  yet"). Heatmap pipeline re-run end-to-end; `heatmap_TBCM_F0.png`
  produced for the first time. `heatmap_FemaleBCM_F0.png` regen is pixel
  jitter, kept for consistency.
- **Refined cross-domain story (the actual deliverable for Matias):**
  alignment quality of source→target is the load-bearing variable.
    * **BCM→BM (poor alignment, Ps 10-2010 vs 600-1000):** alternates
      dominate by +0.47 R² at N=50; TabPFN still leads by +0.17 at N=500.
    * **Male→Female BCM (good alignment, same physics, demographic shift):**
      TabPFN catches transfer at N≈75; dominates from N=100.
    * **BCM→TBCM (same physics family, geometry difference):** TransRF at
      N=500 hits 0.972 — within 0.001 of TabPFN. At N=100 the gap is
      only +0.07 in TabPFN's favor. Transfer competes and at large N
      essentially ties the alternates.
  Headline framing: "alternates win at small N when alignment is poor;
  transfer wins or ties at large N when alignment is good." Cleaner
  thesis than "alternates always win at small N."

**Commits landed (in order, all on `feature/fem`):**
- `0d40466` — TBCM_GP (landed late 2026-05-12 23:15 EDT; carried into
  this entry because cross-domain panel only completed today)
- `5c45242` — TBCM_TabPFN (filled missing N values 150/200/300/500
  after cache fix)
- `31c118f` — TBCM_SmallData JSON dump + regenerated small_data plot
- `94fe23d` — cross_domain_alternates regen + TBCM F0 heatmap +
  FemaleBCM heatmap re-write

**Action items:**
- Fill `<FILL>` placeholders in the group-email draft with TBCM cross-
  domain R² numbers (now have them: TransRF 0.86 vs TabPFN 0.93 at
  N=100; TransRF 0.972 ≈ TabPFN 0.972 at N=500). Strip "Draft notes"
  block. Attach `cross_domain_alternates.png` and the three F0 heatmaps.
  Send to Sean/Jesus/Emiro/Matias — `ben`
- At next 1pm: walk Callum through the refined three-domain story.
  The "transfer ties at large N when aligned" finding partially vindicates
  his original framing (transfer helps for expensive simulators) while
  preserving the alternates-win-at-small-N finding from BM — `shared`

**Blockers:** none. All four 2026-05-12 advisor follow-ups now fully landed.

---

## 2026-05-12 (later 3) — Ben solo (advisor follow-up implementation)

**Attendees:** ben

**Decisions:**
- Implemented all four 2026-05-12 advisor-meeting follow-ups except the TBCM
  components (which remain data-gated). Plan was approved with amendments
  (added Task 5.5 for TBCM JSON dump, fixed Task 6 loaders for the actual
  CSV/JSON schemas on disk) before execution.
- Order executed: plan-commit → N=20 boxplot → Female_GP → Female_TabPFN
  → cross-domain figure → F0 heatmaps → email draft → these docs.
- **Surprise finding (refines headline story):** For the well-aligned
  Male→Female BCM domain pair, the existing RF transfer baseline holds at
  r²_avg ≈ 0.72 across N=25..843 — an _extremely_ strong transfer baseline.
  Yet TabPFN catches up by N≈75 (r²_avg 0.72) and dominates from N=100
  (0.79 vs 0.73), reaching 0.97 at N=500. GP alone _doesn't_ catch up.
  TabPFN's pretrained prior is the load-bearing component.
- Refined paper thesis: **alternates win when source-target alignment is
  poor; TabPFN catches transfer by N≈75 even when alignment is good.**
  Different framing than "alternates always win at small N." Documented in
  the email draft as the leading question for Matias.
- Female BCM convention: applied `ACFL > 30` quality filter (matches
  Ben's existing `FemaleRFTransfer.py:24`) and capped `TEST_POOL_SIZE = 500`
  (dataset is 1195 rows post-filter; needed to keep ≥700 for the test pool
  at N=500).
- Heatmap: had to set `matplotlib.use('Agg')` explicitly before pyplot
  import — default tkinter backend crashes on Windows with
  `Tcl_AsyncDelete: async handler deleted by the wrong thread` during the
  second figure write. Documented inline.

**Commits landed (in order, all on `feature/fem`):**
- `4a81b9c` — plan + team doc updates
- `4d5b747` — Task 1: N=20 boxplot panel
- `50837fa` — Task 4: Female_GP
- `23351e7` — Task 5: Female_TabPFN
- `fe6b642` — Task 6: cross-domain figure (BM + Female; TBCM placeholder)
- `38bfe81` — Task 7: F0 heatmaps (BM + Female)
- `1fdd6e7` — Task 8: email draft

**Action items:**
- Send email after sanity-checking attachments and once TBCM portion lands
  (or send now noting TBCM as a separate follow-up) — `ben`
- Ask Callum for `dataset_TBCM.csv` at next 1pm sync to unblock #17b + #18b — `ben`
- Wait for Matias's reply on publication framing — `ben`

**Blockers:**
- TBCM dataset for #17b and #18b. Female BCM (1195 rows post-filter) and
  BM (5000 rows) both fully unblocked through this cycle.

---

## 2026-05-12 (later 2) — Ben solo (plan for 2026-05-12 advisor follow-ups)
**Attendees:** ben
**Decisions:**
- 2026-05-12 advisor sync ("Vocal Fold ML Update", Fathom recording 145795554)
  generated four Ben-Gladney follow-ups. Plan written to
  `docs/superpowers/plans/2026-05-12-cross-domain-alternates.md`, pending Ben's approval.
- Execution order (Ben's call): N=20 boxplot panel → cross-domain runs
  (TBCM + Female BCM) → muscle-activation × F0 heatmaps → group email draft.
  Rationale: N=20 already exists in alternates JSON (verified), so adding the
  panel is a one-line fix that lets later cross-domain plots share the data
  point without recomputation.
- **Pre-flight blocker:** TBCM dataset CSV is not in the repo and not in
  `TBCM/`. Need to locate via Callum or `~/Downloads/` before Tasks 2/3 run.
  Female BCM CSV is committed at `VocalFoldRegression/BCM Model/FemaleBCM.csv` —
  unblocked.
- **3-way comparison locked per panel** (Ben's 2026-05-12 clarification): each
  cross-domain panel plots GP, TabPFN, AND the existing best transfer method
  for that target. Alternates scripts only produce the GP/TabPFN side; the
  transfer comparator is read from existing repo artifacts. Sources:
    - **BM:** existing `SMALL_N_TRANSFER` dict + `rf_transfer_results.json`
    - **TBCM:** NEW `rf_transfer_small_n.json` produced by Task 5.5
      (`TBCM_SmallData.py` already runs RF transfer at the right N grid;
      just needs a JSON dump appended). The existing
      `rf_transfer_results.json` is fraction-based (smallest n=1379) — no
      overlap with N=5..500 alternates regime, unusable here.
    - **Female BCM:** `ResgressorAnalysis/figs/all_regressors_transfer_comparison.csv`
      filtered to `regressor == 'RF'`, `r2_avg` column. RF-only per Ben's
      2026-05-12 decision (not NN, not PR — keeps the comparator clean).
- **Heatmap design lock:** scatter the 50 training points (true F0) over each
  method's predicted F0 surface at fixed `PS = median`. No FEM-on-grid
  "ground truth" — would take ~8 min × 2500 cells per domain. Misalignment of
  dot color vs surface color is the nonlinear-trend check.
- **Per-domain script convention:** new TBCM_GP/TBCM_TabPFN and
  Female_GP/Female_TabPFN scripts are self-contained copies of the BM
  originals (~80 lines duplicated per domain), matching Callum's existing
  per-domain pattern rather than introducing a shared engine module.
- Email is a markdown draft committed to the repo, not auto-sent. Ben sends
  from his client after sanity-checking attachments and replacing the
  `<FILL>` placeholders with real cross-domain R² gaps.

**Action items:**
- Ben approves plan, then implementation proceeds task-by-task per the plan doc — `ben`
- Pre-flight: locate TBCM dataset CSV (ask Callum at next sync or check `~/Downloads/`) — `ben`
- Plan execution will produce TODO #16/#17/#18/#19; rows added to `team/TODO.md`
  + `team/BOARD.md` as `backlog` for now — `ben`

**Blockers:**
- TBCM dataset (resolution gated on locating local copy or asking Callum at 1pm sync).

---

## 2026-05-12 (later) — Ben solo (extended alternates to N=500)
**Attendees:** ben
**Decisions:**
- Extended `BM_GP.py` and `BM_TabPFN.py` from `N ≤ 100` to `N ∈ {…, 150, 200, 300, 500}`. Changed merge logic in both scripts to per-N: existing JSON entries are preserved, only new N values are fit. Added a `_existing_complete_ns` guard that skips N values already at full `N_RUNS` replicates — re-running is now idempotent and cheap.
- Result: alternates lead transfer by **+0.17 R²** even at N=500 (TabPFN 0.91 vs TransRF 0.74). At N=200 the gap is +0.26 R². The small-N dominance is not a small-N artifact — it persists through the regime where Callum's transfer was supposed to start winning.
- Sim-budget figure is much stronger now: R²≥0.7 panel resolves (TabPFN N=111, GP N=114 vs transfer N=396–463), so the "~5× sample efficiency" claim holds at both R²=0.5 and R²=0.7 thresholds. No more "not reached in tested range" caveat to explain away.
- Bootstrap figure extended to 5 panels (N=10/50/100/200/500). Visible tightening of distributions at large N gives a clean robustness story — the gain over transfer is monotonic and the variance shrinks as expected.

**Action items:**
- Share regenerated figures with Callum at next 1pm — `ben`
- TODO #13 is even more valuable now that we have alternates across the full N=200/500 range — needed for like-for-like boxplot comparison vs transfer — `ben`
- New question to raise with Callum: where (if anywhere) does transfer actually win? `rf_transfer_results.json` has transfer up to N=3200 (frac=1.0). Worth checking whether the curves ever cross — if not, "transfer doesn't help for BCM→BM at any N" becomes a stronger thesis than "transfer doesn't help at small N." — `shared`

**Blockers:** none.

## 2026-05-12 — Ben solo (showcase figures)
**Attendees:** ben
**Decisions:**
- Built presentation-quality showcase figures for the alternates-vs-transfer headline: `Beam_Membrane/BM_Showcase.py` produces three figures (`bm_showcase_headline.png`, `bm_showcase_sim_budget.png`, `bm_showcase_bootstrap.png`). Intended for advisor briefings and paper draft. Standalone from `BM_Summary.py` — different audience, different framing.
- Sim-budget framing: cast the small-N win as "BM simulations required to hit accuracy thresholds." TabPFN/GP need ~5× fewer expensive sims than transfer methods to reach R²≥0.5. Strongest single number for non-ML audiences (every expensive sim costs ~8 min).
- Bootstrap fig uses all 10 replicates per N for GP/TabPFN (we have per-replicate data) and a dashed reference line for Callum's best transfer mean at each N (we only have means in `PROJECT_GUIDE.md`). Carries the same TODO #13 caveat as the 2026-05-05 result.

**Action items:**
- Share figures at next 1pm with Callum — `ben`
- TODO #13 (dump `BM_SmallData.py` to JSON) gets more valuable now — would let the showcase boxplot fig include transfer distributions, not just means — `ben`

**Blockers:** none.

---

## 2026-05-07 — Advisor briefing prep (notes for shared meeting)

**Purpose:** condensed handoff for advisors / non-day-to-day stakeholders. Full notes are in this section so they can be cut-and-pasted into a shared doc or email if needed.

### Recap of where we were

Project goal: replace expensive Beam-Membrane (BM) MATLAB simulations with a fast ML model that maps muscle activations + subglottal pressure → fundamental frequency (F0) and sound pressure level (SPL). Each MATLAB BM simulation is slow (~8 min), so we want a model that predicts accurately from a small number of expensive simulations.

Prior state (Callum's PR, 2026-05-02): six transfer-learning methods that pretrain on cheap BCM simulations (~54,000 samples) and adapt to expensive BM simulations. Best transfer method (TransRF) achieved R² ≈ 0.28 at N=100 BM samples on average.

### What we did this round

Ben asked: does transfer learning actually help here, or could a simpler non-transfer method match or beat it? Tested two non-transfer baselines on the same data:

1. **Gaussian Process (GP)** — classical Bayesian method that fits a smooth function to BM data alone, no BCM involved
2. **TabPFN** — pretrained foundation model (a transformer, similar in spirit to a language model but for tables); makes predictions in one forward pass without per-task fitting

Both run on Callum's data, same sample sizes, same evaluation protocol. **No BCM source data used.**

### Findings

Average R² (F0+SPL, 10 bootstrap runs each) on a 1,000-row held-out BM test pool:

| Training samples | Best transfer (Callum) | GP (ours) | TabPFN (ours) | Gain over transfer |
|---|---|---|---|---|
| 10  | 0.08 | 0.19 | 0.27 | **+0.20** |
| 20  | 0.05 | 0.38 | 0.38 | **+0.33** |
| 30  | 0.19 | 0.44 | 0.47 | **+0.29** |
| 50  | 0.19 | 0.60 | **0.66** | **+0.47** |
| 75  | 0.29 | 0.67 | **0.69** | **+0.40** |
| 100 | 0.28 | 0.67 | 0.67 | **+0.39** |

**Headline:** non-transfer methods beat transfer at every small data size we tested, with gains of 0.20 to 0.47 R². TabPFN at N=50 matches what Callum's best transfer needed N=200 to achieve.

### Why transfer didn't help here

1. **Domain mismatch.** BCM operates over Ps `[10, 2010]` Pa; BM over `[600, 1000]`. Source-only R² on BM ≈ −2 — actively misleading.
2. **Strong generic priors beat weak domain priors at small N.** TabPFN's pretrained prior (millions of synthetic regression problems) plus 50 BM examples beats transfer methods leveraging 54k BCM examples that are pointing the wrong direction.

### Practical implications

- Fast/accurate BM acoustic predictions: simplest path is now "run 50 BM sims, fit TabPFN." The BCM step is unnecessary.
- Simulation budget: TabPFN at N=50 (R²=0.66) is good enough as a screening surrogate.
- Transfer learning research: transfer's value depends critically on source-target alignment. BCM↔BM is misaligned; TBCM↔BM (matching Ps range, same physics family) might work where BCM didn't.

### Caveats

1. Transfer numbers come from Callum's `BM_SmallData.py` results in `PROJECT_GUIDE.md`, run separately from our alternates. Same protocol, not literally the same test rows. Tightening: TODO #13.
2. Tested at N ≤ 100 only. Transfer may catch up at larger N (Callum's published results show TransRF improving toward R²=0.59 at N=200).
3. TabPFN is a cloud-API service. If we ship a model that depends on it, we depend on Prior Labs' uptime. Local-install fallback exists.

### Next steps (decision at next 1pm)

A. **TBCM → BM two-stage transfer** (~1–2 weeks): tests whether physics-aligned transfer beats the alternates.
B. **Real PDE-residual PINN over BM equations** (~3–4 weeks): different *kind* of model — physics surrogate that generalizes OOD and gives gradients. PDEs already extracted from Sean's MATLAB in `docs/BM_GOVERNING_EQUATIONS.md`.

### Anticipated advisor questions (with prepared answers)

See bottom of this entry for full Q&A. Twelve questions covering: result robustness, why transfer didn't help, what TabPFN actually is, reproducibility, project direction, cost, timeline.

---

#### Q&A — short version

**Q: Was Callum's transfer work wasted?** No. Same infrastructure (data, harness, evaluation). The finding is "transfer doesn't beat strong generic priors at small N for *this specific domain pair*", not "transfer is bad." Different pair (TBCM→BM) might still win.

**Q: Why is TabPFN so much better than transfer? Sounds too good.** Two factors compound: (1) TabPFN was pretrained on millions of synthetic tabular problems, strong built-in priors; (2) BCM source operates at 5× the Ps range of BM, so the source signal is misleading. Remove misleading signal + add strong generic prior = simpler approach wins.

**Q: How robust is the +0.47 gap at N=50?** 10 bootstrap replicates per (method, N). Std per replicate is 0.13–0.18. 0.47 is ~3× std — well outside noise. Can rerun with more replicates if needed.

**Q: What is TabPFN, in plain language?** A neural network trained once by Prior Labs on millions of synthetic regression problems, then frozen. To predict on a new dataset, give it your training rows + test rows; it outputs predictions in one forward pass without further training. Like an LLM doing in-context learning, but for tables.

**Q: Why do GPs work at small N?** Bayesian model fitting a probability distribution over smooth functions. Only ~3 hyperparameters; can't really overfit. Textbook small-N method.

**Q: Could we use polynomial regression / small neural networks instead?** Tested earlier in Ben's female-BCM work. Polynomials overfit catastrophically at small N; small NNs need more data. GP and TabPFN are specifically designed for the small-N regime.

**Q: Should we abandon transfer learning?** No. (a) TBCM→BM might work since it's physics-aligned. (b) At larger N transfer's benefit grows. (c) Callum's transfer infrastructure is the test bed. We're augmenting, not replacing.

**Q: How does this affect publication?** Strengthens it. Original story: "transfer learning helps for expensive simulators." New story: "transfer's value depends on source-target alignment; here's a head-to-head against strong generic baselines that establishes when transfer is and isn't useful." More rigorous.

**Q: Is TabPFN reproducible?** TabPFN v2 is in a Nature 2025 paper. Weights publicly available. Cloud API is convenient but not required — local-install fallback. Can pin to a specific version.

**Q: How long did this take?** ~3 days of focused work: GP and TabPFN methods, real-data run, doc reorganization, BM equation extraction.

**Q: What's the next milestone?** Either TBCM→BM transfer (~1–2 weeks) or real PINN (~3–4 weeks). Decide at next 1pm.

**Q: Do we need more BM simulations from Sean?** For regression: probably not — TabPFN at N=50 is already strong. For PINN: yes, ideally with broader sampling of input space. Open question.

## 2026-05-06 (later) — Ben solo (file split)
**Attendees:** ben
**Decisions:**
- Split `BM_Alternates.py` into `BM_GP.py` and `BM_TabPFN.py`. Self-contained per-method-family files matching Callum's existing convention. Both still write to the same `alternates_results.json` so `BM_Summary.py` is unchanged.

**Action items:** none — code-organization-only change.

**Blockers:** none.

## 2026-05-06 — Ben solo (cleanup, BM equations extracted)
**Attendees:** ben
**Decisions:**
- **Removed MonoMLP** (formerly "PINN") from active alternates code. Mid-tier result, didn't add to the GP/TabPFN story; kept the torch dep around for one second-rate baseline. JSON key dropped, figure regenerated, TODO #14 closed.
- **Extracted BM governing equations** from Sean's MATLAB into `docs/BM_GOVERNING_EQUATIONS.md`. Two coupled PDEs (membrane + beam) with full BC/IC, plus the constitutive algebra, aerodynamic Bernoulli model, WRA acoustic propagation, and post-hoc F0/SPL extraction. ~Citations to MATLAB line numbers throughout.
- **Real PINN scoped as TODO #15.** Different deliverable than GP/TabPFN — a PDE-residual physics surrogate that generalizes OOD and gives gradients for inverse problems. Multi-week project.

**Action items:**
- Decide whether to actually pursue #15 vs other research threads — `ben` (and discuss with Callum at next 1pm)
- Pick up another P1 item: #2 (TBCM→BM transfer), #5 (verify Callum's results), or #13 (BM_SmallData JSON dump)

**Blockers:** none.

## 2026-05-05 (later, again) — Ben solo (real-data run; TODO #1 + #12 closed)
**Attendees:** ben
**Decisions:**
- BM data extracted from `~/Downloads/dataset_BM.zip` to `Beam_Membrane/dataset_BM.csv`. 5,000 rows, no NaN, ranges match Callum's spec.
- `BM_Alternates.py` ran end-to-end (~3 min total). `BM_Summary.py` regenerated all three figures.
- **Headline result:** non-transfer alternates beat Callum's transfer methods at every N ∈ [10, 100]. TabPFN especially. Best gain: +0.47 R² at N=50. See `MILESTONES.md` 2026-05-05 entry for the table.
- This reframes the transfer story — at small N, BCM source is a liability, not an asset. TBCM→BM (TODO #2) might still help since TBCM is physics-closer; worth scoping.
- Two follow-up TODOs created (#13: dump JSON for `BM_SmallData.py` to tighten comparison; #14: investigate MonoMLP failure at N≤10).
- TODO #1 + #12 marked `done`. Migrated to Recently Done in BOARD; will roll into MILESTONES at next sync.

**Action items:**
- Discuss TBCM→BM ownership (#2) at next 1pm with Callum given the new framing — `shared`
- Pick up #13 or #14 next, or pivot to #2 — `ben`

**Blockers:** none.

## 2026-05-05 (later) — Ben solo (TabPFN backend swap, BM data inbound)
**Attendees:** ben
**Decisions:**
- TabPFN backend swap from `tabpfn` (local, license-gated) to `tabpfn-client` (cloud, account-gated). Code now imports `tabpfn-client` first and falls back to `tabpfn` if only that is installed. Logged to DECISIONS.
- Auth helper added: reads `TABPFN_TOKEN` env var if set, else relies on cached interactive login from `tabpfn_client.init()`.
- Add `.env*` to `.gitignore` for safety even though no `.env` is currently in the project.
- TODO #12 status moves from `backlog` → `in-progress` since BM data is being copied locally.

**Action items:**
- Run `python -c "from tabpfn_client import init; init()"` once to do browser login — `ben`
- Drop `dataset_BM.csv` into `Beam_Membrane/` and run `python Beam_Membrane/BM_Alternates.py` — `ben`
- Then run `python Beam_Membrane/BM_Summary.py` to regenerate `bm_alternates.png` with real numbers — `ben`
- After numbers land: migrate TODO #1 + #12 to MILESTONES — `ben`

**Blockers:** none — data inbound, code unblocked.

## 2026-05-05 — Ben solo (phases 2–5 of #1, code complete)
**Attendees:** ben
**Decisions:**
- MonoMLP encodes 3 monotonicity priors: `∂F0/∂a_CT`, `∂SPL/∂PS`, and `∂F0/∂PS` (chest-voice physiology). λ = 0.1 default. Logged to DECISIONS.
- TabPFN ≥ 7.x license + token requirement turned out to be a setup blocker — handled with graceful skip; documented in code and in TODO #12.
- TODO #1 status moves from `in-progress` to `review` since the implementation work is complete; the actual real-data run is split out as a new TODO #12 that's the prerequisite for migrating both rows to MILESTONES.

**Action items:**
- Phase 5 (DECISIONS, MILESTONES, BOARD/TODO migration) — done — `ben`
- TODO #12 (real-data run) — backlog — `ben` — needs `dataset_BM.csv` on local machine + `TABPFN_TOKEN`
- At next 1pm with Callum: walk through `bm_alternates.png` once #12 has actual numbers, decide whether the alternates change his transfer story

**Blockers:**
- BM data + TabPFN license/token (both deferred to #12).

## 2026-05-04 (later) — Ben solo (planning + phase 1 of #1)
**Attendees:** ben
**Decisions:**
- Locked in scope of TODO #1 ("alternate methods"): three methods — GP, physics-informed MLP, TabPFN — at N ∈ [5, 10, 20, 30, 50, 75, 100], head-to-head with Callum's TransRF / Feature Aug on the same test split.
- Output lands as new rows in `BM_Summary.py`'s comparison.
- Spec written: `docs/superpowers/specs/2026-05-04-bm-alternates-design.md`.

**Action items:**
- Phase 1 (GP scaffold) — done — `ben`
- Phase 2 (MonoMLP with 3 monotonicity priors: `∂F0/∂a_CT`, `∂SPL/∂PS`, `∂F0/∂PS`) — done — `ben`
- Phase 3 (TabPFN integration) — done — `ben`
- Phase 4 (BM_Summary integration) — next — `ben`
- Phase 5 (DECISIONS + MILESTONES + close TODO #1) — pending — `ben`
- One-time: accept TabPFN license at https://ux.priorlabs.ai, set `TABPFN_TOKEN` env var — `ben`
- Run BM_Alternates.py against real `dataset_BM.csv` to land actual R² numbers — `ben` (blocked on having data on this machine)

**Blockers:**
- BM dataset not on this clone. GP+MonoMLP smoke-tested on synthetic data; TabPFN couldn't be smoke-tested (needs license + token). Real numbers pending data + license.

## 2026-05-04 — Ben solo (post-hiatus catch-up)
**Attendees:** ben
**Decisions:**
- Adopt `/team/` folder as shared agile coordination layer. Both Claude workflows (Ben's and Callum's) will read it via root `CLAUDE.md`.
- Cadence: a few times a week, no fixed schedule.
- Track ownership per-task (`ben`, `callum`, `shared`, `tbd`) rather than by domain.
- Ben's main thread coming off hiatus: explore alternate methods to BM that don't go through the BCM→BM transfer first (TODO #1).
- TBCM→BM two-stage transfer is on the radar as a separate experiment (TODO #2). Owner to be picked at next 1pm.
- Keep Callum's `PROJECT_GUIDE.md` as his hands-on guide for `Beam_Membrane/` and `TBCM/`. Don't fold into `/docs/` — separate purpose.

**Action items:**
- Verify reproducibility of Callum's BM/TBCM results locally before extending — `ben` (TODO #5)
- Pick up TODO #1 (alternate methods) — `ben`
- Discuss TBCM→BM ownership and scope at next 1pm — `shared`

**Blockers:** none.

**Notes for Callum's next session:** Ben merged `origin/main` into `feature/fem` (clean fast-forward, no force) and pushed `main` up with the doc reconcile commits. No code touched — only `/docs/`, `/CLAUDE.md`, and the new `/team/` folder. Please skim `/team/` in your next Claude session; the root `CLAUDE.md` now points there as the shared coordination layer.
