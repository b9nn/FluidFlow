# Milestones

Append-only history. Newest first. Dates from `git log` unless noted.

## 2026-05-05 — Non-transfer alternate methods for BM (TODO #1 + #12 done)

Three non-transfer baselines for BCM→BM at very small N (5–100 samples) ran end-to-end on real `dataset_BM.csv` (5,000 samples, no NaN). Outputs at `Beam_Membrane/results/alternates_results.json` and `Beam_Membrane/figs/bm_alternates.png`.

**Methods:**
- **GP** — `ConstantKernel * Matern(2.5) + WhiteKernel`, independent per output, marginal-likelihood optimization
- **TabPFN** — Pretrained tabular foundation model via `tabpfn-client` (cloud), one regressor per output, capped at 1000 train samples

**Results (avg R² over F0 and SPL, 10 bootstrap runs each):**

| N | GP | TabPFN | Callum's best transfer (PROJECT_GUIDE.md table) | Best alternate gain |
|---|---|---|---|---|
| 10  | 0.19 | **0.27** | TransRF 0.08 | **+0.20** |
| 20  | 0.38 | 0.38 | Feature Aug 0.05 | **+0.33** |
| 30  | 0.44 | **0.47** | Feature Aug 0.19 | **+0.29** |
| 50  | 0.60 | **0.66** | Feature Aug 0.19 | **+0.47** |
| 75  | 0.67 | **0.69** | Feature Aug 0.29 | **+0.40** |
| 100 | 0.67 | 0.67 | TransRF 0.28 | **+0.39** |

**Headline:** non-transfer methods (TabPFN especially) **dominate transfer at small N** — gains of 0.20 to 0.47 R². TabPFN at N=50 (R²=0.66) matches what Callum's TransRF needed N=200 to achieve (R²=0.59). GP is competitive with TabPFN throughout. A third method (originally "PINN", later "MonoMLP") was implemented and then removed 2026-05-06 — it was a mid-tier baseline that didn't add to the GP/TabPFN story; see `docs/DECISIONS.md` for the rationale.

**Caveats:** Callum's "best transfer" column comes from the table in `PROJECT_GUIDE.md` — `BM_SmallData.py` results that aren't dumped to JSON, so this isn't run-on-the-same-test-pool. Same harness shape (sub-sample of N rows, separate test pool of 1000) but different invocations. Re-running `BM_SmallData.py` and dumping its JSON would tighten the comparison; tracked in roadmap.

Code: `Beam_Membrane/BM_Alternates.py` (~440 lines), `BM_Summary.py` extended. Five-phase landing on 2026-05-04 / 05.

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
