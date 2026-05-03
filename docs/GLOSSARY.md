# Glossary

## Domain

- **BCM (Body-Cover Model)** — Two-layer lumped-mass vocal-fold model. Source domain for the male dataset.
- **Beam+Membrane (B+M)** — Three-layer FE vocal-fold model with distributed beam (flexural rigidity) and 2D membrane wave equation. Sean's MATLAB code in `VocalFoldRegression/Beam+Membrane_ForSean/`.
- **F0** — Fundamental frequency (Hz). One of the two predicted outputs.
- **FE model** — Finite Element model.
- **Glottal area** — Area of the opening between the vocal folds; an indirect measurement of vibration. Scripts in `glottal_area/`.
- **MuscleControlModel** — Component of the B+M MATLAB code that simulates posturing (larynx kinematics) from activation inputs.
- **Posturing** — Larynx geometry adjustment driven by extrinsic and intrinsic muscle activations.
- **SPL** — Sound Pressure Level (dB). One of the two predicted outputs.
- **Subglottal pressure (PS)** — Air pressure below the vocal folds (Pa). Drives oscillation; one of the three input features. Range used: `[300, 1000] Pa`.
- **Vocal fold** — Tissue structure in the larynx that vibrates to produce voice.
- **WRA acoustic solver** — Wave Reflection Analog acoustic solver used in the B+M model to compute SPL.

## Inputs / activations

- **a_CT** — Cricothyroid muscle activation. Range `[0, 1]`. Used as a feature.
- **a_IA** — Interarytenoid muscle activation. Present in B+M data; not used as a feature.
- **a_LCA** — Lateral cricoarytenoid muscle activation. B+M only; not a feature.
- **a_PCA** — Posterior cricoarytenoid muscle activation. B+M only; not a feature.
- **a_TA** — Thyroarytenoid muscle activation. Range `[0, 1]`. Used as a feature.
- **PS** — Subglottal pressure (see Domain).

## Methods

- **K-fold CV** — K-fold cross-validation. Used in TransRF to learn ensemble weights.
- **Partial layer freezing** — NN transfer-learning strategy: clone source model, set `trainable=False` on the first N layers, fine-tune the rest.
- **Polynomial regression (PR)** — Linear regression over polynomial features. Degree 12 for male BCM; degree 4–5 + Ridge for small-data targets.
- **Random Forest (RF)** — `MultiOutputRegressor(RandomForestRegressor)`. Source uses `n_estimators=300`.
- **Residual correction** — TransRF sub-model that learns `y - source_pred(x)`.
- **Ridge regularization** — L2-penalized linear regression. Used with PR on small-data domains.
- **StandardScaler** — Zero-mean unit-variance scaler. Always per-domain (see DECISIONS).
- **TransRF** — Custom RF transfer-learning method combining target-only, residual-correction, and feature-augmentation sub-models with K-fold-learned weights.
- **Weighted ensemble** — `α · source.predict(x) + (1-α) · target.predict(x)`.

## Datasets / files

- **`MaleBCM.csv`** — Male BCM source dataset, ~90k samples.
- **`FemaleBCM.csv`** — Female BCM target dataset, ~1.3k samples after `ACFL > 30` filter.
- **`Data_Membrane_Beam_Model.txt`** — B+M generated dataset; 8 columns `a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL`. Some rows NaN.
- **`BCMlookuptable.mat`** — MATLAB lookup table the regressors are replacing.
- **`data_binary.parquet`** — Cached preprocessed BCM dataset.
- **`RF_BCM.pkl`** — Saved male BCM Random Forest.
- **`standard_model.keras`** — Saved male BCM Neural Network.
- **`x_scaler_BCM.pkl`** — Saved male input scaler.

## People

- **Brian Gladney** — Lead, owns regressor and transfer-learning code.
- **Callum** — Collaborator on transfer-learning experiments (joined 2026-01).
