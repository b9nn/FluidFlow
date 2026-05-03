# Roadmap

No hard deadlines — phases are ordered, not dated. Each phase ends with a measurable artifact written to disk.

## Now — Phase 2: Beam+Membrane data exploration & validation

**Goal:** confirm the generated B+M data is usable before training anything.

- Run `VocalFoldRegression/Beam+Membrane Model/load_bm_data.py` to parse the `.txt`.
- Report NaN attrition rate. If valid samples < 300, regenerate with larger `N_s`.
- Distribution plots for `a_CT, a_TA, PS, F0, SPL`. Look for dead zones (regions where everything is NaN).
- Scatter plots: `F0 vs a_CT`, `F0 vs a_TA`, `F0 vs PS` (and same for SPL).
- Side-by-side comparison: B+M vs male BCM distributions for the shared columns.

**Done when:** a short exploration notebook (or script + figures) is committed under `Beam+Membrane Model/`.

## Next — Phase 3: B+M baselines (no transfer)

**Goal:** establish floor metrics that transfer learning has to beat.

- **RF baseline:** small-data hyperparams (lower `n_estimators`, `min_samples_leaf ≥ 2`). Record R² and MAE for F0 and SPL.
- **NN baseline:** smaller architecture, train from scratch. Heavy regularization expected.
- **PR baseline:** `degree=4–5` + `Ridge(alpha)`. Sweep `alpha`.
- Hold out a fixed B+M test set with `random_state=42` so all later experiments compare apples-to-apples.

**Done when:** a baseline metrics CSV is committed.

## Then — Phase 4: Male BCM → B+M transfer learning

Apply the three strategies validated on female BCM, adapted for very small target data.

- **RF transfer (`BeamMembraneRFTransfer.py`):** TransRF with target-only + residual correction + feature augmentation; weights via K-fold CV. Compare vs source-only, target-only, simple ensemble.
- **NN transfer (`BeamMembraneNNTransfer.py`):** partial freezing from `standard_model.keras`; sweep `N ∈ {2, 4, 5, 6}`; pick best by mean R².
- **PR transfer (`BeamMembranePRTransfer.py`):** B+M-specific PR (degree 4–5 + Ridge) ensembled with male PR. Grid search `(degree, alpha, α_ensemble)`.
- **Cross-regressor comparison (`BMTransferComparison.py`):** side-by-side on the same B+M test set with bootstrap sampling.

**Done when:** transfer metrics CSV + comparison figure committed.

## Then — Phase 5: analysis & write-up

- Quantify the BCM↔B+M domain gap. Where does the male model fail on B+M? (residual maps, scatter against truth.)
- Physical interpretation: what does the B+M model capture that BCM doesn't, and where does that show up in the errors?
- Update [`MILESTONES.md`](MILESTONES.md) with results and figures.

**Done when:** results section drafted; figures saved under `figs/`.

## Maybe-later

- Glottal-area integration as an additional input/output feature.
- OpenIFEM coupling for full FSI training data (replaces reduced-order B+M).
- Hyperparameter sweep tracker (e.g. local MLflow or a CSV log).
- Female B+M transfer (once a female-tuned B+M dataset exists).
