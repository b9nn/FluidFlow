# Fluid Flow — Vocal Fold Regression

Fast ML regressors that predict vocal-fold acoustic outputs (fundamental frequency `F0`, sound pressure level `SPL`) from three motor inputs: cricothyroid activation `a_CT`, thyroarytenoid activation `a_TA`, and subglottal pressure `PS`. Replaces slow physics-based lookup tables and supports transfer learning across vocal-fold model families (BCM → Beam+Membrane).

## Quick start

```bash
# Python deps
pip install scikit-learn tensorflow pandas numpy matplotlib joblib scipy

# Train the male BCM Random Forest baseline
python "VocalFoldRegression/BCM Model/RandomForest/MaleRF.py"

# Train the male BCM Neural Network baseline
python "VocalFoldRegression/BCM Model/NeuralNetwork/MaleNN.py"

# Transfer-learn to female BCM (RF, weighted ensemble)
python "VocalFoldRegression/BCM Model/RandomForest/FemaleRFTransfer.py"

# Beam+Membrane data load (after MATLAB has produced Data_Membrane_Beam_Model.txt)
python "VocalFoldRegression/Beam+Membrane Model/load_bm_data.py"
```

Datasets sit next to the model scripts:

- `VocalFoldRegression/BCM Model/MaleBCM.csv` — ~90k samples
- `VocalFoldRegression/BCM Model/FemaleBCM.csv` — ~1.3k samples after `ACFL > 30` filter
- `VocalFoldRegression/Beam+Membrane Model/Data_Membrane_Beam_Model.txt` — ~500 valid samples after NaN drop

## Documentation

| Doc | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, regressor matrix, transfer strategies |
| [`docs/MILESTONES.md`](docs/MILESTONES.md) | Dated history of what's shipped |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased plan for what's next |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Domain terms, methods, file references |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Append-only judgment log |
| [`CLAUDE.md`](CLAUDE.md) | Conventions and entry point for Claude Code sessions |

## Branches

Active work lives on `feature/fem`. `main` is stale — do not pull from it without intent.

## Contributors

- Brian Gladney
- Callum
