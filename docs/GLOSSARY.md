# Glossary

## Domain

- **AE** — Autoencoder. Encoder + decoder neural network used for domain adaptation in `*_Autoencoder.py`.
- **BCM (Body-Cover Model)** — Two-layer lumped-mass vocal-fold model. **Always the source domain** for transfer experiments.
- **BM (Beam-Membrane)** — Three-layer FE vocal-fold model with distributed beam (flexural rigidity) and 2D membrane wave equation. Expensive (~560k timesteps per simulation). Sean's MATLAB code `Beam+Membrane_ForSean/`; Callum's data-generation script `Beam_Membrane/Generate_BM_Dataset.m`.
- **DAAE** — Domain-Adversarial Autoencoder. AE variant where the encoder is trained to fool a domain discriminator via gradient reversal, forcing domain-invariant latent representations.
- **DAAE / DANN-style** — see DAAE.
- **F0** — Fundamental frequency (Hz). Output target.
- **FE / FEM** — Finite Element / Finite Element Model.
- **Glottal area** — Area of the opening between the vocal folds; indirect measurement of vibration. Scripts in `glottal_area/`.
- **MMD** — Maximum Mean Discrepancy. Kernel-based distribution-similarity penalty added to the AE loss to align source and target latent distributions (in `MMD AE`).
- **MuscleControlModel** — Component of Sean's BM MATLAB code that simulates posturing (larynx kinematics) from activation inputs.
- **Posturing** — Larynx geometry adjustment driven by extrinsic and intrinsic muscle activations.
- **Spearman rank correlation** — Rank-based correlation coefficient. Used as an alternative to R² when source predictions have the right *shape* but wrong *scale* (BCM→BM has Spearman 0.81/0.74 despite negative R²).
- **SPL** — Sound Pressure Level (dB). Output target.
- **Subglottal pressure (PS / Ps)** — Air pressure below the vocal folds (Pa). Drives oscillation; one of the three input features. BCM range `[10, 2010]` Pa; BM range `[600, 1000]` Pa (scale mismatch).
- **TBCM (Triangular Body-Cover Model)** — Lumped-element vocal-fold model, same physics family as BCM, different geometry. Cheap to simulate. Used as a "sanity check" target — same physics family means transfer is easy here.
- **Vocal fold** — Tissue structure in the larynx that vibrates to produce voice.
- **WRA acoustic solver** — Wave Reflection Analog acoustic solver used in BM to compute SPL.

## Inputs / activations

- **a_CT** — Cricothyroid muscle activation. Range `[0, 1]`. Feature.
- **a_IA** — Interarytenoid activation. Present in some MATLAB outputs; not a feature.
- **a_LCA** — Lateral cricoarytenoid activation. **BM only**, range `[0.5, 1.0]`. Has near-zero correlation with `F0` / `SPL` so we drop it (see DECISIONS).
- **a_PCA** — Posterior cricoarytenoid activation. Not a feature.
- **a_TA** — Thyroarytenoid activation. Range `[0, 1]`. Feature.
- **PL** — Lung pressure. **TBCM only**, dropped for transfer parity with BCM.
- **PS** — Subglottal pressure (see Domain). Note: the source CSVs use `Ps`; scripts rename to `PS` after loading.

## Methods

- **Adaptive RF complexity** — `get_model_params(n_samples)` scales `n_estimators` and `max_depth` by available data to prevent overfitting at small sizes. Pattern in `Beam_Membrane/BM_TransferRF.py:51`.
- **Gaussian Process (GP)** — Bayesian non-parametric regressor. Places a prior over smooth functions (encoded by a kernel) and conditions on observed data via marginal-likelihood optimization. Used as a non-transfer alternate in `BM_Alternates.py` with `ConstantKernel * Matern(2.5) + WhiteKernel`.
- **MLP (Multi-Layer Perceptron)** — Standard feedforward neural network: a stack of fully-connected (dense) layers with nonlinearities between them. Our `MonoMLP` is `[3 → 32 → 32 → 2]` with ReLU activations, ~1,200 parameters total.
- **MonoMLP (Monotonicity-constrained MLP)** — Small MLP whose loss added a soft inequality penalty on finite-difference approximations of first partial derivatives. Used briefly as a non-transfer alternate (originally mis-labeled "PINN") and removed 2026-05-06 — mid-tier results, didn't sharpen the GP/TabPFN story. Kept in this glossary for git-archaeology purposes only. A real PDE-residual PINN over the BM equations is tracked separately as `team/TODO.md` #15.
- **Feature Augmentation** — RF transfer method 4. Input is `[x; BCM_pred(x)]` — concatenate the source-model prediction onto the feature vector. Wins at 20–75 BM samples.
- **Gradient reversal** — Backprop trick used in DAAE. Multiplies gradients by `−1` between encoder and discriminator so optimizing the discriminator's loss makes the encoder produce domain-invariant features.
- **K-fold CV** — K-fold cross-validation. Used to learn TransRF ensemble weights.
- **Partial layer freezing** — NN transfer method (Brian's): clone source model, set `trainable=False` on the first N layers, fine-tune the rest. Sweep `N ∈ {2, 4, 5, 6}`.
- **Polynomial regression (PR)** — Linear regression over polynomial features. Degree 12 for male BCM; degree 4–5 + Ridge for small-data targets.
- **Random Forest (RF)** — `MultiOutputRegressor(RandomForestRegressor)`. With adaptive complexity in newer scripts.
- **Residual Correction** — RF transfer method 3. RF predicts `y_target − BCM_pred(x)`. Hurts on BM at small sample sizes (trusts raw BCM predictions despite scale mismatch).
- **Ridge regularization** — L2-penalized linear regression.
- **Simple Ensemble** — RF transfer method 5. Fixed `0.3·source + 0.7·target`. Same idea as Brian's male→female RF transfer.
- **Source Only** — RF transfer method 1. Apply BCM model directly to target inputs (zero-shot, no adaptation).
- **StandardScaler** — Zero-mean unit-variance scaler. Always per-domain.
- **Target Only** — RF transfer method 2. Train RF on target alone — the no-transfer baseline.
- **TabPFN (Tabular Prior-Fitted Network)** — Pretrained transformer for tabular regression. Pretrained once by Prior Labs on millions of synthetic tabular problems; at inference, performs in-context learning on `(X_train, y_train)` and predicts `y_test` in a single forward pass — no per-task fitting loop. Used as a non-transfer alternate in `BM_Alternates.py`. Capped at ~1,000 train samples. Two install paths: `tabpfn-client` (cloud) preferred; local `tabpfn` is fallback.
- **TransRF Ensemble** — RF transfer method 6. Learned per-output convex combination of methods 2/3/4 via `LinearRegression(positive=True)`. Wins at 100+ BM samples.
- **Vanilla AE** — Autoencoder method A. Train encoder + decoder + predictor on BCM, fine-tune predictor on target.
- **Waveform features** — Per-cycle features extracted from time-domain waveforms in `.mat` files. TBCM-specific. Pipeline in `TBCM/TBCM_WaveformFeatures.py` → enriched dataset `TBCM/dataset_TBCM_enriched.csv`.
- **Weighted ensemble** — `α · source.predict(x) + (1-α) · target.predict(x)`. Used in Brian's RF (α=0.3) and PR (α=0.05) female transfer.

## Datasets / files

- **`MaleBCM.csv`** — Male BCM source dataset, ~54k samples.
- **`FemaleBCM.csv`** — Female BCM target dataset, filtered `ACFL > 30`.
- **`Beam_Membrane/dataset_BM.csv`** — BM dataset, ~5,000 samples generated by `Generate_BM_Dataset.m`.
- **`Beam_Membrane/dataset_BCM.csv`** — Local BCM copy used by BM scripts.
- **`TBCM/dataset_TBCM.csv`** — TBCM dataset, ~43k samples.
- **`TBCM/dataset_TBCM_enriched.csv`** — TBCM + waveform features.
- **`Data_Membrane_Beam_Model.txt`** — Sean's older BM output format (`a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL`); not used by current scripts.
- **`BCMlookuptable.mat`** — MATLAB lookup table the regressors are replacing.
- **`RF_BCM.pkl`** — Saved male BCM Random Forest.
- **`standard_model.keras`** — Saved male BCM Neural Network.
- **`x_scaler_BCM.pkl`** — Saved male input scaler.
- **`Beam_Membrane/results/rf_transfer_results.json`** — All BM RF method R² numbers across fractions.
- **`Beam_Membrane/results/autoencoder_results.json`** — All BM AE method R² numbers.
- **`TBCM/results/*.json`** — Same for TBCM.

## People

- **Brian Gladney** — Lead. Owns `VocalFoldRegression/` (male/female BCM, RF/NN/PR baselines and transfer).
- **Callum Camazzola** — Joined 2026-01. Owns `Beam_Membrane/` and `TBCM/` (BCM → BM and BCM → TBCM transfer; RF + AE methods).
- **Sean** — Authored the original `Beam+Membrane_ForSean/` MATLAB FE solver.
- **Jesus, Emiro** — Advisors who directed the transfer-learning push that started 2026-01.
