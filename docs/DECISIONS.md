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

## 2026-05-05 — PINN monotonicity priors and λ choice

**Context:** Phase 2 of TODO #1. Picking which physical priors to encode in the physics-informed MLP and how strongly to weight them.
**Decision:** Three monotonicity constraints, all of form `∂y/∂x ≥ 0`:
1. `∂F0 / ∂a_CT ≥ 0` (longer fold → higher pitch)
2. `∂SPL / ∂PS ≥ 0` (more pressure → louder)
3. `∂F0 / ∂PS ≥ 0` (chest-voice physiology — pressure raises pitch modestly)
λ = 0.1 default for the monotonicity term. MLP `[3 → 32 → 32 → 2]` joint head, ReLU. Penalty = mean ReLU(y_anchor − y_nudged) over 256 random anchor pairs in standardized input space, nudge δ = 0.1.
**Why:** These three priors are well-established in vocal-fold physiology and unlikely to fight the data. `∂F0/∂a_TA` is intentionally not constrained — it's non-monotonic in BM (TA tightening can lower F0 in some regimes). λ=0.1 keeps the prior secondary to the data fit while still nudging predictions toward physically plausible behavior at small N. λ to be re-tuned if real-data results show the prior is fighting the data.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py:PINN_PRIORS`, `monotonicity_penalty`.

## 2026-05-05 — TabPFN as a third alternate, with license caveat

**Context:** Phase 3 of TODO #1. Adding a pretrained tabular foundation model.
**Decision:** Use TabPFN ≥ 7.x (`pip install tabpfn`) with one regressor per output. Train cap of 1000 samples (TabPFN's effective limit). Document the one-time license + `TABPFN_TOKEN` setup as a setup task in `team/TODO.md`. Wrap the run loop in try/except so license/network errors skip TabPFN gracefully without breaking GP and PINN.
**Why:** TabPFN delivers strong small-N tabular regression with no hyperparameter tuning. The 1000-sample cap is fine — our regime is N ≤ 100. The license requirement was unexpected but is a one-time setup; graceful skip means TabPFN being unavailable doesn't block the GP/PINN comparison.
**Where it shows up:** `Beam_Membrane/BM_Alternates.py:fit_predict_tabpfn`, `_TABPFN_AVAILABLE` guard in `main()`.

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
