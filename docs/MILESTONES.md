# Milestones

Append-only history. Newest first. Dates from `git log` unless noted.

## 2026-05-05 — Non-transfer alternate methods for BM (TODO #1, code complete)

Three non-transfer baselines for BCM→BM at very small N (5–100 samples), built and wired into `BM_Summary.py`:

- **GP** — Gaussian Process with `ConstantKernel * Matern(2.5) + WhiteKernel`, independent regressor per output, marginal-likelihood hyperparameter optimization
- **PINN** — Physics-informed MLP `[3→32→32→2]`, joint head, MSE + 0.1·monotonicity penalty over 3 priors (`∂F0/∂a_CT`, `∂SPL/∂PS`, `∂F0/∂PS` all ≥ 0)
- **TabPFN** — Pretrained tabular foundation model, one regressor per output, capped at 1000 train samples (license + token required)

Adds `Beam_Membrane/BM_Alternates.py` (~430 lines), extends `BM_Summary.py` to read `alternates_results.json` and emit `figs/bm_alternates.png` comparing alternates against Callum's TransRF / Feature Aug / Target Only references. End-to-end integration smoke-tested with synthetic JSONs. **Pending:** real-data run (BM dataset not on this clone) — tracked as `team/TODO.md` #12.

## 2026-05-03 — Merged Callum's PR #1 into `feature/fem`

Pulled `origin/main` (commit `c8349ee`) into `feature/fem`. Doc tree updated to reflect the new top-level layout (`Beam_Membrane/`, `TBCM/`, `archive/`, `PROJECT_GUIDE.md`). No code changes from Brian — purely a merge + doc reconcile.

## 2026-05-02 — Documentation consolidation (Brian)

Replaced `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with the `/docs/` tree plus root `CLAUDE.md` and `README.md`. Establishes a single source of truth for both Claude sessions and human collaborators. Commits `e690dd6`, `ebb78af`.

## 2026-05-02 — Callum's PR #1 merged to main: BCM → BM and BCM → TBCM transfer

Massive 96-file PR (`ed5f1f6`, +11.6k lines) landing two new top-level subprojects:

- **`Beam_Membrane/`** — BCM → BM transfer with 6 RF methods (Source Only, Target Only, Residual Correction, Feature Augmentation, Simple Ensemble, TransRF Ensemble) and 3 autoencoder methods (Vanilla AE, MMD AE, DAAE). MATLAB data generation (`Generate_BM_Dataset.m`, 5,000 BM samples). JSON results, comparison figures.
- **`TBCM/`** — BCM → TBCM transfer with the same RF + AE method suite plus waveform feature extraction.
- **`archive/`** — old exploratory scripts moved aside (`PhysicsInformedTransfer.py`, `GaussianProcessTransfer.py`, beam-membrane v1/v2 diagnostics, etc.).
- **`PROJECT_GUIDE.md`** — Callum's onboarding doc with empirical results and run instructions.
- Adaptive RF complexity (`get_model_params(n_samples)`) replaces fixed `n_estimators=300` for small-data targets.

Key empirical findings from this PR:
- BCM → TBCM: TransRF R² > 0.96 even at 5% of target data.
- BCM → BM: harder. Feature Augmentation wins at 20–75 samples; TransRF wins at 100+. 200 samples is the sweet spot (+0.09 R² over target-only).
- Autoencoders underperform RF on BM but are competitive on TBCM.
- `a_LCA` (BM-only) drops out — near-zero correlation with outputs.

Co-authored with Claude Opus 4.6.

## 2026-02-19 — Plan of action written for B+M transfer learning (Brian)

5-phase plan: data gen → exploration → baselines → transfer → analysis. Folded into [`ROADMAP.md`](ROADMAP.md) on 2026-05-02. Commit `1d6b348`. (Plan was largely realized by Callum's PR a couple months later.)

## 2026-02-12 — Callum scaffolds `Beam_Membrane/` on `cc-dev`

`Beam_Membrane/PhysicsInformedTransfer.py` — 681-line first pass at BM transfer using physics-informed RF. Later moved to `archive/` in PR #1. Commit `4115763`.

## 2026-02-12 — Transfer learning repairs + physics-based experiments (Brian)

Repaired prior transfer methods; added experimental physics-informed RF transfer (`PhysicsInformedTransfer.py`). Commit `fc24567`. This is the common ancestor of `feature/fem` and `cc-dev`.

## 2026-01-12 — Callum onboarded for new transfer learning approach

Bulk code dump in preparation for Callum picking up transfer-learning experiments per advisors Jesus and Emiro. Commit `c718828`.

## 2025-11-20 — Bulk experimental dump (Brian)

Multiple alternative transfer methods committed (`AlternativeTransferMethods.py`, `ImprovedTransferLearning.py`, `TransferLearningExperiment.py`, `FemaleRFEnsambleLearning.py`, etc.). Commit `9ecc1ab`.

## 2025-11-06 — Neural Network transfer learning baseline (Brian)

Partial layer freezing implementation (`FemaleNNTransfer.py`). Commit `1913f35`.

## 2025-11-05 — RF + PR transfer learning for female BCM (Brian)

Weighted-ensemble transfer for both regressors (`FemaleRFTransfer.py`, `FemalePRTransfer.py`). Commit `56a313e`.

## 2025-10-24 — Female BCM compatibility (Brian)

Initial female-domain support across regressors. Commit `fe03dc4`.

## 2025-10-24 — Polynomial Regression baseline (Brian, male BCM)

Degree-12 PR with `MultiOutputRegressor(LinearRegression)` over `PolynomialFeatures` (`MalePR.py`). Commit `6e224f9`.

## 2025-10-23/24 — Neural Network baseline (Brian, male BCM)

Sequential `[512, 256, 128, 64, 2]` with L2 + Dropout; saved as `standard_model.keras`. Commits `0ff9d8a`, `9b027df`, `6328398`.

## 2025-10-16 — Random Forest baseline (Brian, male BCM)

`MultiOutputRegressor(RandomForest)`, `n_estimators=300`, hyperparam search via `GridSearchCV`. Saved as `RF_BCM.pkl`. Commit `1a3af93`.

## 2025-10-16 — Glottal area extraction script (Brian)

`glottal_area/` scripts for processing `VF_Left_data.csv` / `VF_Right_data.csv`. Commit `def15e2`. (`integrate.py` later moved to `archive/top_level/glottal_area/`.)
