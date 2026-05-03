# Architecture

## Pipeline

```
[a_CT, a_TA, PS]  →  StandardScaler(input)  →  Regressor  →  StandardScaler⁻¹  →  [F0, SPL]
                                                              (per-output, per-domain)
```

Inputs and outputs are scaled separately per domain. **Three scalers per domain:** one for inputs, one for `F0`, one for `SPL`.

## Domains

| Domain | Source file | Size | Notes |
|---|---|---|---|
| Male BCM | `MaleBCM.csv` | ~90,000 | Source of truth for pretraining |
| Female BCM | `FemaleBCM.csv` | ~1,331 | Filtered to `ACFL > 30` |
| Beam+Membrane (B+M) | `Data_Membrane_Beam_Model.txt` | ~500 valid | After NaN drop; physically more accurate FE model |

## Regressor matrix

| Regressor | Source script | Source artifact | Target script (BCM→F) | Transfer strategy |
|---|---|---|---|---|
| Random Forest | `BCM Model/RandomForest/MaleRF.py` | `RF_BCM.pkl` | `FemaleRFTransfer.py` | Weighted ensemble (`0.3·source + 0.7·target`); advanced TransRF in `DataEfficiencyExperiment.py` |
| Neural Network | `BCM Model/NeuralNetwork/MaleNN.py` | `standard_model.keras` | `FemaleNNTransfer.py` | Partial layer freezing — clone source, freeze first N layers, fine-tune remainder |
| Polynomial Regression | `BCM Model/PolynomialRegressor/MalePR.py` | `firstPR` | `FemalePRTransfer.py` | Weighted ensemble (`0.05·source + 0.95·target`) |

**RF architecture:** `MultiOutputRegressor(RandomForestRegressor(n_estimators=300, ...))`, hyperparams via `GridSearchCV`.
**NN architecture:** `Sequential([Dense(512), Dense(256), Dense(128), Dense(64), Dense(2)])` with L2 regularization and `Dropout(0.1)`.
**PR architecture:** `MultiOutputRegressor(LinearRegression())` over `PolynomialFeatures(degree=12)` for male BCM. Smaller domains use `degree=4–5` + `Ridge`.

## Transfer strategies

### Weighted ensemble (RF, PR)

```
y_pred = α · model_source.predict(x) + (1-α) · model_target.predict(x)
```

- RF: `α = 0.3` (target dominates; source still adds signal)
- PR: `α = 0.05` (PR generalizes poorly across domains; trust target almost entirely)

### Partial layer freezing (NN)

1. Clone the source `keras` model.
2. Freeze the first N layers (`layer.trainable = False`).
3. Fit on target data with a low learning rate.
4. Sweep `N ∈ {2, 4, 5, 6}`; pick by mean R² across F0 and SPL.

### TransRF (advanced RF)

Three sub-models fit on target data:

1. **Target-only** RF.
2. **Residual correction** RF — predicts `y_target - source_pred(x)`.
3. **Feature-augmentation** RF — input is `[x; source_pred(x)]`.

Final prediction is a learned convex combination of the three; weights estimated via K-fold CV. See `BCM Model/RandomForest/DataEfficiencyExperiment.py` — finds **200–500 target samples** is the sweet spot.

## Cross-regressor analysis

`BCM Model/ResgressorAnalysis/AllRegressorsTransferComparison.py` runs all three transfer models against a held-out target test set, with bootstrap sampling across varying training-sample counts. Outputs comparison CSV + figures. The B+M analog lives at `Beam+Membrane Model/RegressorAnalysis/BMTransferComparison.py`.

## Data layer

- **Schema:** input columns `a_CT, a_TA, PS`; output columns `F0, SPL`. The B+M `.txt` adds three additional activations (`a_LCA, a_IA, a_PCA`) — only `a_CT, a_TA, PS` are used as features for parity with BCM.
- **NaN handling (B+M):** drop rows where any of `F0, SPL` is NaN — these are physically invalid configurations (beam in compression, negative mucosa stress).
- **Scalers:** see [`DECISIONS.md`](DECISIONS.md) entry on per-domain scalers. Per-domain, always.
- **Splits:** `train_test_split(test_size=0.2, random_state=42)` everywhere.

## Beam+Membrane model (target domain)

Sean's MATLAB FE model in `VocalFoldRegression/Beam+Membrane_ForSean/`:

- 3 tissue layers (TA muscle + ligament + mucosa) vs BCM's 2-layer lumped model
- Distributed beam with flexural rigidity + 2D membrane wave equation
- Posturing simulated via `MuscleControlModel` (larynx kinematics)
- WRA acoustic solver for SPL

Same free inputs as BCM: `a_CT ∈ [0,1]`, `a_TA ∈ [0,1]`, `PS ∈ [300, 1000] Pa`. Same outputs `F0, SPL`. Generation script: `Randomly_Generating_Data_Membrane_Beam_Model.m`. Set `N_s ≈ 700` to overshoot the ~500-valid target.
