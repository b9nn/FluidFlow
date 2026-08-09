# FluidFlow: Transfer Learning for Vocal Fold Models

## Project Overview

This project investigates **transfer learning** between vocal fold simulation models of varying complexity and cost. The core idea: train a machine learning model (Random Forest or autoencoder) on a large dataset from a **cheap, simple model**, then use that knowledge to reduce the number of expensive simulations needed from a **complex, costly model**.

**Input parameters** (shared across all models):
- `a_CT` — Cricothyroid muscle activation (0–1)
- `a_TA` — Thyroarytenoid muscle activation (0–1)
- `Ps` — Subglottal pressure (Pa)

**Output targets** (predicted by all models):
- `F0` — Fundamental frequency (Hz)
- `SPL` — Sound pressure level (dB)

---

## The Three Models

### BCM — Body-Cover Model (Source)
- **Type:** Lumped-element (mass-spring)
- **Cost:** Very cheap — 54,103 samples available
- **Inputs:** `a_CT, a_TA, Ps`
- **Outputs:** `F0, SPL`
- **Ranges:** F0 ∈ [55, 862] Hz, SPL ∈ [−4, 91] dB, Ps ∈ [10, 2010] Pa
- **Role:** Always the **source** model for transfer learning

### TBCM — Triangular Body-Cover Model (Target 1)
- **Type:** Lumped-element (same physics family as BCM, different geometry)
- **Cost:** Moderate — 43,102 samples available
- **Inputs:** `a_CT, a_TA, PL, Ps` (we drop `PL` for transfer since BCM doesn't have it)
- **Outputs:** `F0, SPL`
- **Ranges:** F0 ∈ [80, 384] Hz, SPL ∈ [56, 109] dB
- **Role:** Target model — transfer from BCM is very effective here (same physics family)

### BM — Beam-Membrane Model (Target 2)
- **Type:** Finite-element (FEM, ~560k timesteps per simulation)
- **Cost:** Very expensive — 5,000 samples generated overnight in MATLAB
- **Inputs:** `a_CT, a_TA, Ps, a_LCA` (extra input not in BCM)
- **Outputs:** `F0, SPL`
- **Ranges:** F0 ∈ [116, 426] Hz, SPL ∈ [−54, 129] dB, Ps ∈ [600, 1000] Pa
- **Role:** Target model — transfer from BCM is harder (different physics, scale mismatch)
- **Note:** `a_LCA` (lateral cricoarytenoid activation, 0.5–1.0) is unique to BM but has near-zero correlation with F0/SPL, so adding it as a feature doesn't help transfer

---

## Transfer Learning Methods

All methods use the BCM as the source model and a fraction of target data for adaptation. The key metric is R² (coefficient of determination) evaluated on held-out target data.

### RF-Based Methods (implemented in `*_TransferRF.py`)

| # | Method | Description |
|---|--------|-------------|
| 1 | **Source Only** | Apply BCM RF directly to target data (zero-shot, no adaptation) |
| 2 | **Target Only** | Train RF on target data alone (no transfer, baseline) |
| 3 | **Residual Correction** | Train RF to learn the error pattern: `target_truth − BCM_prediction` |
| 4 | **Feature Augmentation** | Use BCM predictions as extra input features alongside the original inputs |
| 5 | **Simple Ensemble** | Fixed 0.3/0.7 weighted average of source and target predictions |
| 6 | **TransRF Ensemble** | Learned per-output weights over target-only, residual, and augmented sub-models using `LinearRegression(positive=True)` on a validation set |

**Adaptive RF complexity:** The function `get_model_params(n_samples)` scales tree depth and count based on available training data to prevent overfitting at small sizes. This is critical for the small-data regime.

### Autoencoder-Based Methods (implemented in `*_Autoencoder.py`)

| # | Method | Description |
|---|--------|-------------|
| A | **Vanilla AE** | Train encoder+decoder+predictor on BCM, fine-tune predictor on target |
| B | **MMD AE** | Shared encoder with Maximum Mean Discrepancy penalty to align latent distributions |
| C | **DAAE** | Domain-Adversarial AE — encoder fools a domain discriminator via gradient reversal |

**Architecture:** Encoder(3→64→32→16) → Decoder(16→32→64→3) + Predictor(16→32→16→2) + Discriminator(16→16→8→1)

---

## Key Results

### BCM → TBCM (same physics family — strong transfer)

| Fraction | Target-Only R² | TransRF R² | Transfer Gain |
|----------|---------------|-----------|---------------|
| 5% (1,379 samples) | 0.956 | 0.962 | +0.006 |
| 10% | 0.978 | 0.981 | +0.003 |
| 25% | 0.987 | 0.990 | +0.003 |
| 100% | 0.992 | 0.993 | +0.001 |

- Spearman rank correlation: F0 = 0.88, SPL = 0.75
- TransRF wins at all fractions; RF methods dominate autoencoders
- Transfer gain is modest because target-only is already excellent (same physics family)

### BCM → BM (different physics — harder transfer)

| Fraction | Target-Only R² | TransRF R² | Transfer Gain |
|----------|---------------|-----------|---------------|
| 5% (160 samples) | 0.493 | 0.516 | +0.023 |
| 10% (320) | 0.745 | 0.719 | −0.026 |
| 25% (800) | 0.777 | 0.800 | +0.023 |
| 50% (1,600) | 0.866 | 0.872 | +0.006 |
| 100% (3,200) | 0.854 | 0.865 | +0.011 |

- Spearman rank correlation: F0 = 0.81, SPL = 0.74 (good structural agreement)
- Source-only R² is deeply negative (~−2.0) due to scale mismatch (BCM Ps range 10–2010 vs BM 600–1000)
- Autoencoders underperform RF methods on BM (unlike TBCM)

### BCM → BM Small Data (the real use case)

At truly small target sizes (10–500 BM samples), tested with 10 runs each:

| N samples | Target-Only | Feature Aug | TransRF | Best Method |
|-----------|------------|-------------|---------|-------------|
| 10 | 0.040 | 0.071 | 0.075 | TransRF |
| 20 | −0.040 | 0.054 | 0.019 | Feature Aug |
| 30 | 0.143 | 0.187 | 0.171 | Feature Aug |
| 50 | 0.118 | 0.193 | 0.188 | Feature Aug |
| 75 | 0.222 | 0.294 | 0.250 | Feature Aug |
| 100 | 0.211 | 0.278 | 0.280 | TransRF |
| 200 | 0.494 | 0.584 | 0.586 | TransRF |
| 500 | 0.719 | 0.716 | 0.739 | TransRF |

- **Feature Augmentation** wins at 20–75 samples (gives the model BCM predictions as signal without forcing trust in absolute values)
- **TransRF** wins at 100+ samples (has enough data to learn optimal sub-model weights)
- **Residual Correction** hurts at all small sizes (trusts raw BCM predictions which are way off in scale)
- **Vanilla AE** struggles on BM (unlike TBCM)
- **200 samples** is the sweet spot — biggest transfer gain (+0.09 R²)

---

## Directory Structure

```
FluidFlow/
│
├── PROJECT_GUIDE.md              ← This file
├── .gitignore
├── MaleBCM.csv                   ← Full male BCM dataset (54k samples)
├── FemaleBCM.csv                 ← Female BCM dataset
│
├── Beam_Membrane/                ← BCM → BM transfer experiments
│   ├── BM_TransferRF.py          ← 6 RF transfer methods at varying fractions
│   ├── BM_TransferAE.py          ← 3 autoencoder approaches
│   ├── BM_Summary.py             ← Cross-method comparison (RF vs AE)
│   ├── BM_SmallData.py           ← Small data experiment (10–500 samples)
│   ├── Generate_BM_Dataset.m     ← MATLAB script to generate BM simulations
│   ├── dataset_BCM.csv           ← BCM data for BM experiments
│   ├── dataset_BM.csv            ← 5,000 BM simulations (generated by MATLAB)
│   ├── results/                  ← JSON results files
│   │   ├── rf_transfer_results.json
│   │   ├── rf_transfer_results_aLCA.json
│   │   └── autoencoder_results.json
│   └── figs/                     ← Output plots
│       ├── bm_rf_transfer.png        ← R² vs fraction (6 RF methods)
│       ├── bm_ae_results.png         ← R² vs fraction (3 AE methods)
│       ├── bm_comparison.png         ← RF vs AE head-to-head
│       ├── bm_small_data_all.png     ← Small data: all methods
│       └── bm_small_data_comparison.png ← Small data: transfer gain
│
├── TBCM/                         ← BCM → TBCM transfer experiments
│   ├── TBCM_TransferRF.py        ← 6 RF transfer methods
│   ├── TBCM_Autoencoder.py       ← 3 autoencoder approaches
│   ├── TBCM_Summary.py           ← Cross-method comparison
│   ├── TBCM_SmallData.py         ← Small data experiment
│   ├── TBCM_WaveformTransfer.py  ← Waveform feature experiment
│   ├── TBCM_WaveformFeatures.py  ← Waveform feature extraction from .mat files
│   ├── plot_small_data.py        ← Quick small-data results plotter
│   ├── dataset_BCM.csv           ← BCM data for TBCM experiments
│   ├── dataset_TBCM.csv          ← TBCM simulation data (43k samples)
│   ├── dataset_TBCM_enriched.csv ← TBCM + extracted waveform features
│   ├── results/
│   │   ├── rf_transfer_results.json
│   │   └── autoencoder_results.json
│   └── figs/
│       ├── tbcm_rf_transfer.png
│       ├── tbcm_ae_results.png
│       ├── tbcm_comparison.png
│       ├── tbcm_small_data.png
│       └── tbcm_waveform_experiment.png
│
├── VocalFoldRegression/          ← Original male → female BCM transfer
│   └── BCM Model/
│       ├── RandomForest/
│       │   ├── MaleRF.py                      ← Baseline RF on male BCM
│       │   ├── FemaleRF.py                    ← Baseline RF on female BCM
│       │   ├── FemaleRFTransfer.py            ← Male → Female RF transfer
│       │   ├── FemaleRFEnsambleLearning.py    ← Ensemble transfer approach
│       │   ├── TransferLearningExperiment.py  ← Data efficiency experiment
│       │   ├── DataEfficiencyExperiment.py    ← Min samples needed analysis
│       │   ├── ImprovedTransferLearning.py    ← Enhanced transfer methods
│       │   ├── glide.py                       ← Jacobian sensitivity analysis
│       │   └── heatmap.py                     ← Output heatmaps (CT vs TA)
│       ├── Neural Network/                    ← NN baselines and transfer
│       ├── Polynomial Regressor/              ← Polynomial baselines
│       └── Regressor Analysis/                ← Cross-regressor comparison
│
├── transrf_env/                  ← Python 3.13 virtual environment
│
└── archive/                      ← Exploratory scripts moved here for clarity
    ├── beam_membrane/            ← Old BM v1/v2 explorations & diagnostics
    └── random_forest/            ← Dead-end approaches (physics-informed, GP)
```

---

## How to Run Experiments

### Environment Setup

```bash
cd /path/to/FluidFlow          # wherever you cloned the repo

# Activate the virtual environment (Python 3.13)
source transrf_env/bin/activate        # Windows: transrf_env\Scripts\activate

# Key packages: scikit-learn 1.8, torch 2.11, numpy, scipy, pandas, matplotlib
```

> `transrf_env` is a local venv name, not something in the repo — create your
> own with `python -m venv transrf_env` and install the packages above. The
> `./transrf_env/bin/python3` prefix in the commands below is the same thing;
> plain `python` works if your environment is already active.

### Running Scripts

All scripts use `MPLBACKEND=Agg` to avoid display issues in headless/terminal mode:

```bash
# BCM → BM: RF transfer methods
MPLBACKEND=Agg ./transrf_env/bin/python3 Beam_Membrane/BM_TransferRF.py

# BCM → BM: Autoencoder methods
MPLBACKEND=Agg ./transrf_env/bin/python3 Beam_Membrane/BM_TransferAE.py

# BCM → BM: Cross-method summary
MPLBACKEND=Agg ./transrf_env/bin/python3 Beam_Membrane/BM_Summary.py

# BCM → BM: Small data experiment (10–500 samples)
MPLBACKEND=Agg ./transrf_env/bin/python3 Beam_Membrane/BM_SmallData.py

# BCM → TBCM: Same pattern
MPLBACKEND=Agg ./transrf_env/bin/python3 TBCM/TBCM_TransferRF.py
MPLBACKEND=Agg ./transrf_env/bin/python3 TBCM/TBCM_Autoencoder.py
MPLBACKEND=Agg ./transrf_env/bin/python3 TBCM/TBCM_Summary.py
```

### Generating BM Data (MATLAB)

If you need to regenerate the BM simulation dataset:

1. Open MATLAB R2025b
2. Navigate to `/tmp/bm_code/Beam+Membrane-Sean/`
3. Run `Generate_BM_Dataset.m` (generates 5,000 samples, saves incrementally)
4. Copy output: `cp dataset_BM_clean.csv /path/to/FluidFlow/Beam_Membrane/dataset_BM.csv`

The MATLAB script calls `Membrane_Beam_Solver_MyImplementation2` (FEM solver) and requires `PhonationModelsCode2/` on the MATLAB path.

---

## Key Code Patterns

### Adaptive RF Complexity (`get_model_params`)

All transfer scripts use this function to scale model complexity with available data:

```python
# BM_TransferRF.py:51-72, BM_SmallData.py:47-68
def get_model_params(n_samples):
    if n_samples < 100:
        return {'n_estimators': 20, 'max_depth': 3, ...}
    elif n_samples < 250:
        return {'n_estimators': 30, 'max_depth': 5, ...}
    # ... scales up to n_estimators=300, max_depth=None for large datasets
```

### TransRF Weight Learning

The TransRF ensemble learns per-output weights over three sub-models using constrained linear regression:

```python
# BM_TransferRF.py:130-155
# Stack predictions from target-only, residual, and feature-augmented models
# Learn weights via LinearRegression(positive=True) to ensure non-negative combination
```

### Data Loading Convention

All scripts load CSVs with `index_col=0` and rename `Ps` → `PS`:

```python
# Standard pattern across all scripts
df = pd.read_csv('dataset_BCM.csv', index_col=0)
df.rename(columns={'Ps': 'PS'}, inplace=True)
```

---

## Important Findings

1. **Transfer works best in the small-data regime** — the whole point is reducing the number of expensive simulations needed. At 200 BM samples, transfer gives +9% R² over target-only.

2. **Feature Augmentation is the safest bet at very small sizes** (20–75 samples) — it passes BCM predictions as extra inputs without assuming they're accurate in absolute terms.

3. **TransRF wins with more data** (100+ samples) — it has enough data to learn optimal weights over sub-models.

4. **Residual Correction fails when there's a large scale mismatch** — it trusts raw BCM predictions, which are way off for BM (BCM Ps range 10–2010 vs BM 600–1000).

5. **Autoencoders underperform RF on BM** but are competitive on TBCM — suggests AE-based domain adaptation works better when source and target share the same physics family.

6. **Spearman rank correlation > R²** for evaluating transfer potential — BCM→BM has Spearman 0.81/0.74 (good structural agreement) despite R² of −3.0/−0.8 (terrible absolute accuracy).

7. **The `a_LCA` feature unique to BM has near-zero correlation with outputs** — adding it as an extra input doesn't improve transfer performance.

---

## Historical Context

The project evolved through three phases:

1. **Male → Female BCM** (`VocalFoldRegression/`): Proved that transfer learning between vocal fold models works. Tested RF, NN, and polynomial regressors. RF with TransRF ensemble emerged as the best approach.

2. **BCM → TBCM** (`TBCM/`): Extended to a different lumped-element model. Very successful (R²>0.96 at 5% data). Also tested autoencoder domain adaptation and waveform features.

3. **BCM → BM** (`Beam_Membrane/`): Extended to a finite-element model. Harder due to physics gap. Focused on small-data regime (10–500 samples). Feature Augmentation and TransRF are the winning methods depending on sample count.
