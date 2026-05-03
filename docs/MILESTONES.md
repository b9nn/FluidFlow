# Milestones

Append-only history. Newest first. Dates from `git log` unless noted.

## 2026-05 — Documentation consolidation

Replaced `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with this `/docs/` tree plus root `CLAUDE.md` and `README.md`. Establishes a single source of truth for both Claude sessions and human collaborators.

## 2026-05 — B+M training data generated *(assumed for planning)*

~500 valid samples after NaN drop. Output at `VocalFoldRegression/Beam+Membrane Model/Data_Membrane_Beam_Model.txt`.

## 2026-02-19 — Plan of action written for B+M transfer learning

5-phase plan: data gen → exploration → baselines → transfer → analysis. Folded into [`ROADMAP.md`](ROADMAP.md). Commit `1d6b348`.

## 2026-02-12 — Beam+Membrane file structure created

`VocalFoldRegression/Beam+Membrane Model/` scaffolded with `load_bm_data.py` and three regressor stubs (`BeamMembraneRFTransfer.py`, `BeamMembraneNNTransfer.py`, `BeamMembranePRTransfer.py`) plus `RegressorAnalysis/BMTransferComparison.py`. Branch `feature/fem` opened. Commit `4115763`.

## 2026-02-12 — Transfer learning repairs + physics-based experiments

Repaired prior transfer methods; added new experimental approaches including a physics-informed RF transfer (`PhysicsInformedTransfer.py`). Commit `fc24567`.

## 2026-01-12 — Callum onboarded for new transfer learning approach

Bulk code dump in preparation for Callum picking up transfer-learning experiments per advisors Jesus and Emiro. Commit `c718828`.

## 2025-11-20 — Bulk experimental dump

Multiple alternative transfer methods committed (`AlternativeTransferMethods.py`, `ImprovedTransferLearning.py`, `TransferLearningExperiment.py`, `FemaleRFEnsambleLearning.py`, etc.). Commit `9ecc1ab`.

## 2025-11-06 — Neural Network transfer learning baseline

Partial layer freezing implementation (`FemaleNNTransfer.py`). Commit `1913f35`.

## 2025-11-05 — RF + PR transfer learning for female BCM

Weighted-ensemble transfer for both regressors (`FemaleRFTransfer.py`, `FemalePRTransfer.py`). Commit `56a313e`.

## 2025-10-24 — Female BCM compatibility

Initial female-domain support across regressors. Commit `fe03dc4`.

## 2025-10-24 — Polynomial Regression baseline (male BCM)

Degree-12 PR with `MultiOutputRegressor(LinearRegression)` over `PolynomialFeatures` (`MalePR.py`). Commit `6e224f9`.

## 2025-10-23/24 — Neural Network baseline (male BCM)

Sequential `[512, 256, 128, 64, 2]` with L2 + Dropout; saved as `standard_model.keras`. Commits `0ff9d8a`, `9b027df`, `6328398`.

## 2025-10-16 — Random Forest baseline (male BCM)

`MultiOutputRegressor(RandomForest)`, `n_estimators=300`, hyperparam search via `GridSearchCV`. Saved as `RF_BCM.pkl`. Commit `1a3af93`.

## 2025-10-16 — Glottal area extraction script

`glottal_area/` scripts for processing `VF_Left_data.csv` / `VF_Right_data.csv`. Commit `def15e2`.
