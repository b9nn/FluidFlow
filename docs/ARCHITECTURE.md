# Architecture

## Pipeline

```
[a_CT, a_TA, PS]  →  StandardScaler(input)  →  Regressor  →  StandardScaler⁻¹  →  [F0, SPL]
                                                              (per-output, per-domain)
```

Inputs and outputs are scaled separately per domain. **Three scalers per domain:** one for inputs, one for `F0`, one for `SPL`. Never reuse scalers across domains.

## Domains

| Domain | File | Size | Range notes |
|---|---|---|---|
| Male BCM | `VocalFoldRegression/BCM Model/MaleBCM.csv` | ~54k | Source. `F0 ∈ [55, 862]`, `SPL ∈ [−4, 91]`, `Ps ∈ [10, 2010]` |
| Female BCM | `VocalFoldRegression/BCM Model/FemaleBCM.csv` | ~1.3k | Filtered `ACFL > 30` |
| TBCM (Triangular BCM) | `TBCM/dataset_TBCM.csv` | ~43k | Same physics family as BCM. `F0 ∈ [80, 384]`, `SPL ∈ [56, 109]`. Has extra `PL` column we drop |
| BM (Beam-Membrane FEM) | `Beam_Membrane/dataset_BM.csv` | ~5,000 | Different physics. `F0 ∈ [116, 426]`, `SPL ∈ [−54, 129]`, `Ps ∈ [600, 1000]`. Has extra `a_LCA` column we drop |

## Top-level layout

```
/
├── VocalFoldRegression/   ← Brian's male/female BCM work (RF, NN, PR)
├── Beam_Membrane/         ← BCM→BM transfer (Callum) + non-transfer alternates (Brian: GP, TabPFN)
├── TBCM/                  ← Callum: BCM→TBCM transfer (RF + AE + waveform)
├── archive/               ← Old experiments parked here in PR #1
├── team/                  ← Shared agile workspace (TODO, BOARD, MEETING_NOTES)
└── docs/, CLAUDE.md, README.md, PROJECT_GUIDE.md
```

## Regressors and methods, by era

### Era 1 — Brian's `VocalFoldRegression/` (male → female BCM)

| Regressor | Source script | Source artifact | Target script | Transfer |
|---|---|---|---|---|
| Random Forest | `BCM Model/RandomForest/MaleRF.py` | `RF_BCM.pkl` | `FemaleRFTransfer.py` | Weighted ensemble `0.3·src + 0.7·tgt` |
| Neural Network | `BCM Model/NeuralNetwork/MaleNN.py` | `standard_model.keras` | `FemaleNNTransfer.py` | Partial layer freezing, sweep `N ∈ {2,4,5,6}` |
| Polynomial | `BCM Model/PolynomialRegressor/MalePR.py` | `firstPR` | `FemalePRTransfer.py` | Weighted ensemble `0.05·src + 0.95·tgt` |

**Cross-regressor analysis:** `BCM Model/ResgressorAnalysis/AllRegressorsTransferComparison.py` runs all three on the female test set with bootstrap sampling.

### Era 2 — Callum's `Beam_Membrane/` and `TBCM/` (BCM → BM, BCM → TBCM)

Six labeled RF methods + three autoencoder methods. Same input schema `[a_CT, a_TA, PS]`, BCM as source, target chosen per directory.

#### RF transfer methods (in `*_TransferRF.py`)

| # | Method | Description |
|---|---|---|
| 1 | **Source Only** | Apply BCM RF directly (zero-shot) |
| 2 | **Target Only** | Train RF on target alone (no transfer baseline) |
| 3 | **Residual Correction** | RF predicts `y_target − BCM_pred(x)` |
| 4 | **Feature Augmentation** | Input is `[x; BCM_pred(x)]` |
| 5 | **Simple Ensemble** | Fixed `0.3·src + 0.7·tgt` |
| 6 | **TransRF Ensemble** | Learned per-output weights over methods 2/3/4 via `LinearRegression(positive=True)` on a held-out validation slice |

**Adaptive RF complexity** (`get_model_params(n_samples)` in `BM_TransferRF.py:51`):
- `< 100` samples: `n_estimators=20, max_depth=3`
- `100–250`: `n_estimators=30, max_depth=5`
- scales up to `n_estimators=300, max_depth=None` for the largest sets.

#### Autoencoder transfer methods (in `*_Autoencoder.py`)

Architecture: Encoder `3 → 64 → 32 → 16` → Decoder `16 → 32 → 64 → 3` + Predictor `16 → 32 → 16 → 2` + Discriminator `16 → 16 → 8 → 1`. Implemented in PyTorch.

| # | Method | Description |
|---|---|---|
| A | **Vanilla AE** | Train encoder + decoder + predictor on BCM, fine-tune predictor on target |
| B | **MMD AE** | Shared encoder with Maximum Mean Discrepancy penalty aligning latent distributions |
| C | **DAAE** | Domain-Adversarial AE — encoder fools a domain discriminator via gradient reversal |

#### TBCM-only — Waveform features

`TBCM_WaveformFeatures.py` extracts per-cycle waveform features from `.mat` files; `TBCM_WaveformTransfer.py` evaluates whether richer features close the BCM→TBCM gap. Result enriched dataset: `TBCM/dataset_TBCM_enriched.csv`.

### Era 3 — Brian's non-transfer alternates (BM only)

Two methods that **don't use BCM source data at all** — the question they answer is "do strong generic priors beat domain-specific transfer at small N?" Per file, matching Callum's per-method-family convention:

| Method | File | Library | Per-task fitting? |
|---|---|---|---|
| **Gaussian Process** | `Beam_Membrane/BM_GP.py` | `sklearn` | Yes (marginal-likelihood, ~1 s) |
| **TabPFN** | `Beam_Membrane/BM_TabPFN.py` | `tabpfn-client` (cloud) → `tabpfn` (local fallback) | No (one forward pass per row) |

**GP details:** kernel `ConstantKernel * Matern(ν=2.5) + WhiteKernel`. Independent regressor per output (F0, SPL). Hyperparameters via marginal-likelihood with 3 random restarts.

**TabPFN details:** pretrained transformer for tabular regression. Single-output → two regressors (one per target). Train cap of 1,000 samples (matches our N≤100 regime). Cloud client preferred; auth via one-time browser login (`tabpfn_client.init()`) or `TABPFN_TOKEN` env var.

Both files:
- Mirror Callum's harness: N ∈ [5, 10, 20, 30, 50, 75, 100], 10 bootstrap runs, 1,000-row test pool
- Drop `a_LCA` for shared-feature parity with BCM
- Per-sub-sample scalers
- Merge into shared `Beam_Membrane/results/alternates_results.json` via `merge_into_alternates()` — order-independent
- `BM_Summary.py` reads the unified JSON and emits `figs/bm_alternates.png`

**Removed methods:** an earlier "PINN" / "MonoMLP" (small MLP with monotonicity penalties on first partials) was implemented and removed 2026-05-06 — mid-tier results, didn't sharpen the GP/TabPFN story. A real PDE-residual PINN over the BM equations is scoped separately as `team/TODO.md` #15; PDEs extracted in `docs/BM_GOVERNING_EQUATIONS.md`.

## Key empirical results

### Non-transfer alternates vs transfer (BCM → BM, small N)

Average R² over F0+SPL on a 1,000-row held-out BM test pool, 10 bootstrap runs each:

| N | Best transfer (Callum) | GP (Brian) | TabPFN (Brian) | Best-alternate gain |
|---|---|---|---|---|
| 10  | TransRF 0.08 | 0.19 | **0.27** | +0.20 |
| 20  | Feature Aug 0.05 | 0.38 | 0.38 | +0.33 |
| 30  | Feature Aug 0.19 | 0.44 | **0.47** | +0.29 |
| 50  | Feature Aug 0.19 | 0.60 | **0.66** | +0.47 |
| 75  | Feature Aug 0.29 | 0.67 | **0.69** | +0.40 |
| 100 | TransRF 0.28 | 0.67 | 0.67 | +0.39 |

**Headline:** non-transfer alternates beat transfer at every small-N point tested. TabPFN at N=50 (R²=0.66) matches what TransRF needed N=200 to achieve. Two factors: (1) BCM→BM scale mismatch on `Ps` (`[10, 2010]` vs `[600, 1000]`) makes BCM source predictions actively misleading; (2) TabPFN's pretrained prior is stronger than 54k BCM samples worth of misaligned signal at small N.

> Caveat: transfer numbers come from Callum's `BM_SmallData.py` results in `PROJECT_GUIDE.md` (separate runs, same harness shape). Tightening this comparison is `team/TODO.md` #13.

### Original transfer-only results (from Callum's PR #1)

### BCM → TBCM (same physics family)

| Fraction | Target-Only R² | TransRF R² | Gain |
|---|---|---|---|
| 5% (1,379) | 0.956 | 0.962 | +0.006 |
| 10% | 0.978 | 0.981 | +0.003 |
| 25% | 0.987 | 0.990 | +0.003 |
| 100% | 0.992 | 0.993 | +0.001 |

Spearman: F0 = 0.88, SPL = 0.75. RF methods dominate AE. Transfer gain is small because target-only is already excellent (same physics family).

### BCM → BM (different physics)

| Fraction | Target-Only R² | TransRF R² | Gain |
|---|---|---|---|
| 5% (160) | 0.493 | 0.516 | +0.023 |
| 25% (800) | 0.777 | 0.800 | +0.023 |
| 50% (1,600) | 0.866 | 0.872 | +0.006 |

Spearman: F0 = 0.81, SPL = 0.74. Source-only R² ≈ −2.0 (scale mismatch). AE underperforms RF on BM.

### BCM → BM small-data regime

At 10–500 BM samples (10 runs each):

- **20–75 samples:** Feature Augmentation wins
- **100+ samples:** TransRF wins
- **200 samples:** sweet spot (+0.09 R² over target-only)
- **Residual Correction** hurts at all small sizes (trusts raw BCM predictions despite scale mismatch)

## Data conventions

- **CSV loading:** `pd.read_csv(..., index_col=0)`, then `df.rename(columns={'Ps': 'PS'}, inplace=True)`.
- **Schema:** input columns `a_CT, a_TA, PS`; output columns `F0, SPL`. Drop extra columns (`PL` for TBCM, `a_LCA` for BM — has near-zero correlation with outputs).
- **NaN handling (BM):** drop rows where `F0` or `SPL` is NaN (physically invalid configs).
- **Scalers:** per-domain, always.
- **Splits:** `train_test_split(test_size=0.2, random_state=42)`.

## Beam-Membrane source (MATLAB)

Two MATLAB pipelines coexist:

- **Callum's `Beam_Membrane/Generate_BM_Dataset.m`** — current. 5,000 samples saved incrementally. Uses FEM solver `Membrane_Beam_Solver_MyImplementation2`; requires `PhonationModelsCode2/` on the MATLAB path.
- **Sean's `Beam+Membrane_ForSean/Randomly_Generating_Data_Membrane_Beam_Model.m`** — older, untracked locally on `feature/fem`. Has `Membrane_Beam_Solver.m`, `WRA_Solver.m`, `MuscleControlModel`-flavored posturing scripts. Not currently used.
