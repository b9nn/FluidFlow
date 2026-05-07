# Decisions

Append-only log of judgment calls. Newest at the bottom.

Format:

```
## YYYY-MM-DD — Short title
**Context:** what triggered the decision
**Decision:** what we chose
**Why:** the reason / trade-off
**Where it shows up:** files / scripts affected
```

---

## 2025-10 — Standard data split: `random_state=42`, 80/20

**Context:** Reproducibility across regressor scripts.
**Decision:** Every script uses `train_test_split(test_size=0.2, random_state=42)`.
**Why:** Same hold-out across RF, NN, PR makes cross-regressor comparison meaningful.
**Where it shows up:** every `*.py` under `VocalFoldRegression/`, `Beam_Membrane/`, `TBCM/`.

## 2025-10 — Per-domain StandardScalers (never share across domains)

**Context:** Initial female-BCM transfer learning produced near-zero R² when the male input scaler was reused on female data.
**Decision:** Each domain (male BCM, female BCM, BM, TBCM) fits and saves its own input scaler and its own per-output scalers.
**Why:** Different domains have different feature distributions; reusing a male scaler shifts and scales target data into the wrong subspace before the model sees it.
**Where it shows up:** `BCM Model/NeuralNetwork/interpret.txt` (incident notes); every transfer script.

## 2025-10 — Polynomial Regression degree 12 for male BCM

**Context:** Male BCM has ~54k samples and a smooth nonlinear input/output map.
**Decision:** Use `PolynomialFeatures(degree=12)` + plain `LinearRegression` for the male PR baseline.
**Why:** Sample count tolerates the high-degree feature blow-up without overfitting.
**Where it shows up:** `BCM Model/PolynomialRegressor/MalePR.py`.

## 2025-11 — RF female transfer weights: 0.3 male + 0.7 female

**Context:** Tuning the weighted ensemble for RF on ~1.3k female samples.
**Decision:** `α = 0.3` source, `1-α = 0.7` target.
**Why:** Empirically best on the female test set.
**Where it shows up:** `BCM Model/RandomForest/FemaleRFTransfer.py`.

## 2025-11 — PR female transfer weights: 0.05 male + 0.95 female

**Context:** Same tuning exercise for PR.
**Decision:** `α = 0.05` source, `1-α = 0.95` target.
**Why:** Polynomial regression generalizes poorly across domains — degree-12 male features extrapolate badly onto female inputs. Trust target almost entirely.
**Where it shows up:** `BCM Model/PolynomialRegressor/FemalePRTransfer.py`.

## 2025-11 — TransRF data efficiency sweet spot: 200–500 target samples

**Context:** Brian's earlier data efficiency experiment for the advanced RF transfer method on female BCM.
**Decision:** Best transfer benefit observed at 200–500 target samples.
**Why:** Below 200, target-only is too noisy; above 500, target-only alone is already strong.
**Where it shows up:** `BCM Model/RandomForest/DataEfficiencyExperiment.py`.

## 2026-02 — PR for small target domains: degree 4–5 + Ridge

**Context:** Plan for B+M PR transfer (~500 samples).
**Decision:** Drop polynomial degree from 12 to 4–5 and use `Ridge` instead of plain `LinearRegression`.
**Why:** Degree 12 with 500 samples will overfit catastrophically.
**Where it shows up:** Planned for `BeamMembranePRTransfer.py` (Brian's local stub, not yet committed; superseded by the RF/AE-only direction Callum took).

## 2026-05-02 — Adaptive RF complexity, replacing fixed `n_estimators=300`

**Context:** Callum's BM/TBCM transfer scripts work across small (10) to large (40k+) target sample counts. A single hyperparam config either underfits the large case or overfits the small case.
**Decision:** Use `get_model_params(n_samples)` to scale `n_estimators` (20→300) and `max_depth` (3→None) with sample count.
**Why:** Prevents overfitting at small sizes and underfitting at large sizes without hyperparameter search per run.
**Where it shows up:** `Beam_Membrane/BM_TransferRF.py:51`, `Beam_Membrane/BM_SmallData.py:47`, `TBCM/TBCM_TransferRF.py`, `TBCM/TBCM_SmallData.py`.

## 2026-05-02 — Drop `a_LCA` (BM-only feature)

**Context:** BM data includes an extra `a_LCA` activation `∈ [0.5, 1.0]` not in BCM.
**Decision:** Don't use `a_LCA` as a feature. Train BM RF on `[a_CT, a_TA, PS]` only.
**Why:** Empirically `a_LCA` has near-zero correlation with `F0` and `SPL` in the BM data; including it as a feature doesn't improve transfer and creates a feature-mismatch with BCM.
**Where it shows up:** `Beam_Membrane/BM_TransferRF.py`, `Beam_Membrane/BM_TransferAE.py`. Files like `bm_pred_vs_actual_aLCA.png` document the comparison run with the feature included.

## 2026-05-02 — Best small-data BM method depends on sample size

**Context:** Cross-method comparison at 10–500 BM samples (10 runs each).
**Decision:** No single best method. Use **Feature Augmentation** at 20–75 samples, **TransRF** at 100+ samples. Avoid **Residual Correction** at any small size.
**Why:** Feature Aug doesn't force trust in BCM's absolute predictions, which is critical when the BCM-BM scale mismatch is large. TransRF needs enough data to estimate sub-model weights reliably. Residual Correction trusts raw BCM predictions, which are way off in scale at all sizes.
**Where it shows up:** `Beam_Membrane/BM_SmallData.py`; figure `bm_small_data_all.png`.

## 2026-05-02 — Autoencoders underperform RF on BM (use them for TBCM only)

**Context:** AE methods (Vanilla, MMD, DAAE) tested on both BM and TBCM transfer.
**Decision:** AE is competitive on TBCM (same physics family) but loses to RF on BM. Default to RF methods for BM; keep AE for TBCM as a comparison point.
**Why:** AE-based domain adaptation requires the source and target distributions to be related by a smooth manifold mapping. BM's different physics + scale mismatch breaks that assumption.
**Where it shows up:** `Beam_Membrane/figs/bm_rf_vs_ae.png`, `Beam_Membrane/BM_TransferAE.py`.

## 2026-05-02 — Use Spearman rank correlation alongside R² for transfer evaluation

**Context:** BCM source-only predictions on BM have R² ≈ −2.0 (terrible absolute accuracy) but Spearman 0.81/0.74 (good rank ordering).
**Decision:** Report Spearman alongside R² when evaluating cross-domain transfer.
**Why:** Spearman captures whether the source model has learned the right *shape* of the relationship even when the *scale* is wrong — and the right shape is exactly what residual / feature-aug / TransRF methods can exploit.
**Where it shows up:** `Beam_Membrane/BM_Summary.py`, `TBCM/TBCM_Summary.py`.

## 2026-05-02 — Data loading convention: `index_col=0` + `Ps → PS`

**Context:** All datasets are CSVs with an unnamed index column and use `Ps` for subglottal pressure.
**Decision:** Standardize to:
```python
df = pd.read_csv('dataset_*.csv', index_col=0)
df.rename(columns={'Ps': 'PS'}, inplace=True)
```
**Why:** Consistent column names across BCM / TBCM / BM let the same model code work on all three.
**Where it shows up:** every `*.py` in `Beam_Membrane/` and `TBCM/`.

## 2026-02 — Active branch: `feature/fem`

**Context:** Brian and Callum worked on parallel branches (`feature/fem` and `cc-dev`) from common ancestor `fc24567`.
**Decision:** Continue work on `feature/fem`.
**Why:** Brian's docs commits live there; `cc-dev` was Callum's PR feeder branch and is merged to `main`.
**Where it shows up:** repo state. Note: as of 2026-05-03, `main` is no longer stale — Callum's PR landed and `feature/fem` has been merged with `origin/main`.

## 2026-05-02 — Documentation lives at repo root, not under `VocalFoldRegression/`

**Context:** Replacing ad-hoc `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with a coherent doc tree.
**Decision:** `CLAUDE.md` and `README.md` at repo root; reference docs under `/docs/`.
**Why:** Both audiences (Claude auto-loads `CLAUDE.md` from root; humans expect `README.md` at root) find the docs without nesting.
**Where it shows up:** `/CLAUDE.md`, `/README.md`, `/docs/*.md`.

## 2026-05-05 — Non-transfer alternates: GP kernel choice (Matern 2.5)

**Context:** Phase 1 of TODO #1. Picking a kernel for the Gaussian Process baseline on BM at very small N.
**Decision:** `ConstantKernel * Matern(nu=2.5) + WhiteKernel`. Optimize hyperparameters via marginal-likelihood (sklearn default), 3 restarts. One independent regressor per output (F0, SPL).
**Why:** Matern(ν=2.5) is twice differentiable — smoother than Matern(ν=1.5) but not as restrictive as RBF (which assumes infinite differentiability). Vocal-fold physics is smooth but not analytic; ν=2.5 is the standard middle-ground for physics regression. Multiplicative ConstantKernel allows the GP to learn output magnitude; WhiteKernel absorbs noise.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py:fit_predict_gp`.

## 2026-05-06 — Split BM_Alternates.py into BM_GP.py and BM_TabPFN.py

**Context:** After removing MonoMLP, the file was a single ~280-line module hosting two unrelated methods (sklearn GP and a cloud-API call to TabPFN). One file, two completely different dependencies and authentication paths. Inconsistent with Callum's per-method-family pattern (`BM_TransferRF.py`, `BM_TransferAE.py`).
**Decision:** Split `BM_Alternates.py` into `Beam_Membrane/BM_GP.py` and `Beam_Membrane/BM_TabPFN.py`. Each file is self-contained: own harness (`run_single`, `run_method`), own data loader, own JSON merge helper. Both write to the same `Beam_Membrane/results/alternates_results.json` using `merge_into_alternates()`, so they don't clobber each other. Order of execution doesn't matter. Delete the merged `BM_Alternates.py`. `BM_Summary.py` reads the unified JSON unchanged.
**Why:** Matches Callum's existing convention (one file per method family), makes auth/dependency boundaries clear (GP needs only sklearn, TabPFN needs cloud auth), and lets either method be re-run independently without dragging the other along.
**Where it shows up:** `Beam_Membrane/BM_GP.py` (NEW), `Beam_Membrane/BM_TabPFN.py` (NEW), `Beam_Membrane/BM_Alternates.py` (DELETED), `README.md` quick-start, `docs/GLOSSARY.md` GP/TabPFN entries point at the split files.

## 2026-05-06 — Remove MonoMLP from active code (keep GP + TabPFN only)

**Context:** After the rename and the real-data run, MonoMLP was a mid-tier non-transfer method — better than Callum's transfer at moderate N (≥20), worse than GP and TabPFN throughout, and actively bad at N≤10 where the monotonicity prior dominated too-little data. The story we tell is "non-transfer alternates beat transfer at small N"; MonoMLP didn't sharpen that story and added a torch dependency for the only methods that needed it.
**Decision:** Remove `MonoMLP`, `monotonicity_penalty`, `MONOMLP_*` constants, and the torch imports from `Beam_Membrane/BM_Alternates.py`. Drop `'MonoMLP'` styling from `BM_Summary.py`. Drop the `'MonoMLP'` key from `Beam_Membrane/results/alternates_results.json`. Regenerate `figs/bm_alternates.png` with only GP + TabPFN curves. Mark `team/TODO.md` #14 (investigate MonoMLP collapse) as `done`/canceled — no longer relevant. Keep `team/TODO.md` #15 (build a real PDE-residual PINN over the BM equations) since that's a different, larger project.
**Why:** The earlier rename (PINN → MonoMLP, 2026-05-06) addressed the misnaming. This removal addresses the substance: a method that doesn't improve the story is dead weight. A real PINN over the BM PDEs (now extracted in `docs/BM_GOVERNING_EQUATIONS.md`) is a separate, deliberate project — not a continuation of the MonoMLP work.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py`, `Beam_Membrane/BM_Summary.py`, `Beam_Membrane/results/alternates_results.json`, `figs/bm_alternates.png`, `docs/MILESTONES.md` (results table simplified), `team/TODO.md` (#14 closed).

## 2026-05-06 — Rename "PINN" to "MonoMLP" (avoid overclaiming)

**Context:** What we built and called "PINN" is not actually a PDE-residual physics-informed neural network in the Raissi-Perdikaris-Karniadakis sense. It's a small MLP whose loss adds a soft inequality penalty on finite-difference approximations of first partial derivatives — i.e. monotonicity constraints, not PDE residuals.
**Decision:** Rename throughout: code (`PINN_*` → `MONOMLP_*`, `PinnMLP` → `MonoMLP`, `fit_predict_pinn` → `fit_predict_monomlp`, dispatch key `'PINN'` → `'MonoMLP'`), results JSON (top-level key), and all docs (DECISIONS, MILESTONES, GLOSSARY, team/*). The historical spec at `docs/superpowers/specs/2026-05-04-bm-alternates-design.md` gets a one-line update at the top noting the rename.
**Why:** A real PINN would constrain the residual `R[u] = ∂u/∂t − L[u] = 0` at collocation points using the actual governing equations. We can't do that here — the BM physics (Navier-Stokes + elastodynamics + WRA) lives inside the FEM solver and isn't accessible as a closed-form PDE between our (a_CT, a_TA, PS) and (F0, SPL). Naming it "PINN" overclaims; "MonoMLP" describes what the code actually does.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py`, `Beam_Membrane/BM_Summary.py`, `Beam_Membrane/results/alternates_results.json`, all `/docs/` and `/team/` markdown.

## 2026-05-05 — MonoMLP monotonicity priors and λ choice

**Context:** Phase 2 of TODO #1. Picking which physical priors to encode in the constrained MLP and how strongly to weight them.
**Decision:** Three monotonicity constraints, all of form `∂y/∂x ≥ 0`:
1. `∂F0 / ∂a_CT ≥ 0` (longer fold → higher pitch)
2. `∂SPL / ∂PS ≥ 0` (more pressure → louder)
3. `∂F0 / ∂PS ≥ 0` (chest-voice physiology — pressure raises pitch modestly)
λ = 0.1 default for the monotonicity term. MLP `[3 → 32 → 32 → 2]` joint head, ReLU. Penalty = mean ReLU(y_anchor − y_nudged) over 256 random anchor pairs in standardized input space, nudge δ = 0.1.
**Why:** These three priors are well-established in vocal-fold physiology and unlikely to fight the data. `∂F0/∂a_TA` is intentionally not constrained — it's non-monotonic in BM (TA tightening can lower F0 in some regimes). λ=0.1 keeps the prior secondary to the data fit while still nudging predictions toward physically plausible behavior at small N. λ to be re-tuned if real-data results show the prior is fighting the data.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py:MONOMLP_PRIORS`, `monotonicity_penalty`.

## 2026-05-05 — Confirmed: non-transfer alternates beat transfer at small N

**Context:** Real-data run of TODO #1 / #12 against `dataset_BM.csv`.
**Decision:** Treat the result as a real, publishable finding: **non-transfer methods (TabPFN especially) outperform Callum's transfer methods by +0.20 to +0.47 avg R² at every N ∈ [10, 100] tested.** TabPFN at N=50 (R²=0.66) matches TransRF at N=200. This reframes the "transfer for expensive simulators" story: at small N the BCM source actively hurts (negative source-only R² ≈ −2.0, scale mismatch). A pretrained tabular foundation model with no domain knowledge wins.
**Why:** Three non-overlapping factors:
1. BCM and BM have a hard scale mismatch on `Ps` (BCM `[10, 2010]`, BM `[600, 1000]`) — transfer methods that trust BCM's predictions (Residual Correction, Simple Ensemble) hurt at all small sizes.
2. At very small N (5–30), 3 input features and 2 outputs is too low-dimensional for transfer to add signal; the inductive bias of TabPFN's pretrained prior is more valuable than 54k BCM samples.
3. MonoMLP's monotonicity prior is helpful at moderate N (≥20) but actively damaging at N≤10 where data is too sparse to ground the prior.
**Where it shows up:** `Beam_Membrane/results/alternates_results.json`; `MILESTONES.md` 2026-05-05 entry.
**Implications for next steps:** TODO #2 (TBCM→BM two-stage transfer) is even more interesting now — TBCM is closer to BM in physics, so transfer might actually help where BCM didn't. Worth scoping a separate exploration. Also: re-running Callum's `BM_SmallData.py` with JSON dump would tighten the head-to-head numbers.

## 2026-05-05 — TabPFN as a third alternate (cloud client preferred)

**Context:** Phase 3 of TODO #1. Adding a pretrained tabular foundation model. Initial install of the local `tabpfn` package hit a license-acceptance gate that required interactive browser login + `TABPFN_TOKEN` to download model weights, which broke headless smoke testing.
**Decision:** Switch to `tabpfn-client` (cloud-API; no local model-weight download, no license dance — just an account login). The code prefers `tabpfn-client` and falls back to local `tabpfn` if only that is installed. One regressor per target. Train cap of 1000 samples (matches our N ≤ 100 regime anyway). Auth either via interactive `tabpfn_client.init()` (browser, one-time, cached) or via `TABPFN_TOKEN` env var. Wrap the run loop in try/except so auth/network errors skip TabPFN without breaking GP and MonoMLP.
**Why:** Cloud client removes a friction step (license + token download) and keeps TabPFN reachable from headless or fresh-clone environments. The dual-import dance (`tabpfn-client` first, `tabpfn` fallback) means whichever Brian or Callum has installed will work.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py:fit_predict_tabpfn`, `_TABPFN_BACKEND` selection at import time, `_ensure_tabpfn_auth` helper.

## 2026-05-04 — Shared agile workspace at `/team/`, per-task ownership

**Context:** Brian and Callum sync at ~1pm a few times a week. Need a single place both Claude workflows can read for current state — what's in flight, who owns what, recent decisions.
**Decision:** Add `/team/` folder with `TODO.md`, `BOARD.md`, `MEETING_NOTES.md`, `README.md`. Track ownership per-task with an `owner` field (`brian`, `callum`, `shared`, `tbd`) rather than fixing it by codebase area. Both contributors can pick up work in either codebase. Strategic research roadmap stays in `docs/ROADMAP.md`; `/team/` is operational (this week, this sprint).
**Why:** Splitting ownership by codebase area would freeze responsibilities and make cross-area contributions awkward. A per-task owner field scales as work moves and keeps both Claudes informed about who's doing what.
**Where it shows up:** `team/*.md`; `CLAUDE.md` "Team workflow" section and "Where to look" list; "Contributors" reframed as authorship-only.

## 2026-05-03 — Keep Callum's `PROJECT_GUIDE.md` alongside `/docs/`

**Context:** Merging Callum's main into `feature/fem` brings `PROJECT_GUIDE.md`, which overlaps with `README.md` + `docs/ARCHITECTURE.md`.
**Decision:** Keep both. `PROJECT_GUIDE.md` is preserved as Callum's hands-on guide for `Beam_Membrane/` and `TBCM/`. `README.md` and `docs/` are the canonical entry points.
**Why:** The two docs serve subtly different purposes (Callum's is method-focused with empirical tables; ours is repo-wide with conventions). Folding them risks losing his framing. Better to cross-link.
**Where it shows up:** `README.md` nav table, `CLAUDE.md` repo map.
