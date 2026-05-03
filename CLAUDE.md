# CLAUDE.md

> Auto-loaded by Claude Code in every session. Read this first.

## Project mission

Map vocal-fold motor inputs (`a_CT`, `a_TA`, `PS`) to acoustic outputs (`F0` Hz, `SPL` dB) using ML regressors. Train on a cheap source model (BCM) and **transfer-learn** to expensive targets (TBCM, Beam-Membrane FEM) so we need fewer expensive simulations. Currently three target domains: female BCM, TBCM, BM.

Active branch: `feature/fem`. As of 2026-05-02 it is **merged with `origin/main`** (Callum's PR #1). Push when ready.

## Repo map

| Path | Status | What it is |
|---|---|---|
| `Beam_Membrane/` | **primary (Callum)** | BCM → BM transfer experiments. RF + autoencoder methods, MATLAB data gen, results JSON, figures |
| `TBCM/` | **primary (Callum)** | BCM → TBCM transfer experiments. RF + autoencoder + waveform features |
| `VocalFoldRegression/` | **primary (Brian)** | Original male/female BCM transfer (RF, NN, polynomial regressor) |
| `archive/` | reference | Old experimental scripts and figures Callum moved aside in PR #1 |
| `docs/` | docs | Architecture, milestones, roadmap, glossary, decisions |
| `PROJECT_GUIDE.md` | docs (Callum) | Callum's standalone onboarding doc — kept alongside `/docs/`, focused on his transfer-learning methods |
| `glottal_area/` | aux | Vocal-fold area extraction scripts (`integrate.py` archived) |
| `OpenIFEM/` | aux | External FE/FSI solver, vendored — not actively used |

> Local-only (untracked, third-party): `VocalFoldRegression/Beam+Membrane_ForSean/` (Sean's upstream MATLAB FE solver — kept on your machine, not vendored into the repo without consent).

## Stack

- **Python**: `scikit-learn`, `torch` (autoencoders), `tensorflow` / `keras` (legacy NN), `pandas`, `numpy`, `matplotlib`, `joblib`, `scipy`
- **MATLAB**: BM data generation. Two scripts coexist:
  - Callum's `Beam_Membrane/Generate_BM_Dataset.m` (current — 5,000 samples, FEM solver `Membrane_Beam_Solver_MyImplementation2`)
  - Sean's `VocalFoldRegression/Beam+Membrane_ForSean/Randomly_Generating_Data_Membrane_Beam_Model.m` (older, untracked locally)

## Hard conventions — obey these

1. **Per-domain StandardScalers.** Each domain (male, female, BM, TBCM) fits its own input scaler and its own per-output scalers (one for `F0`, one for `SPL`). Never reuse a scaler across domains — it destroyed early female-BCM transfer results (`VocalFoldRegression/BCM Model/NeuralNetwork/interpret.txt`).
2. **Schema is fixed.** Inputs always `[a_CT, a_TA, PS]`. Outputs always `[F0, SPL]`. Don't reorder. (BM has an extra `a_LCA` column we **drop** — see DECISIONS.)
3. **Reproducibility.** `random_state=42`, `test_size=0.2` train/test split — match every existing script.
4. **Adaptive RF complexity.** For new RF transfer scripts, scale `n_estimators` and `max_depth` by sample count using a `get_model_params(n_samples)` function (pattern in `Beam_Membrane/BM_TransferRF.py:51`). Don't hardcode `n_estimators=300` for small-data targets.
5. **Data loading convention** (Callum's pattern, used across `Beam_Membrane/` and `TBCM/`):
   ```python
   df = pd.read_csv('dataset_*.csv', index_col=0)
   df.rename(columns={'Ps': 'PS'}, inplace=True)
   ```
6. **Don't blob-commit binaries.** `.pkl`, `.keras`, `.parquet` are committed selectively. `.csv`, `*.txt`, `*.mat`, `*.ipynb` are gitignored — use `git add -f` only when needed and you'll see why in `.gitignore`.
7. **Append a line to `docs/DECISIONS.md`** whenever you make a non-obvious judgment call (hyperparam choice, weighting, transfer strategy variant, scope change).

## Interaction conventions

- When you need to ask Brian multiple-choice or preference questions, use the `AskUserQuestion` tool ("planning format" — rendered as chips). Don't ask A/B/C/D in plain text. Inline prose questions are fine only for genuinely open-ended prompts.

## Where to look

- System design, regressor matrix, transfer methods → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- What's been done (dated) → [`docs/MILESTONES.md`](docs/MILESTONES.md)
- What's next → [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Term lookup → [`docs/GLOSSARY.md`](docs/GLOSSARY.md)
- Why we chose X → [`docs/DECISIONS.md`](docs/DECISIONS.md)
- Callum's hands-on guide for `Beam_Membrane/` and `TBCM/` → [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)

## Contributors

- **Brian Gladney** — lead. Owns `VocalFoldRegression/` (male/female BCM, RF/NN/PR baselines and transfer).
- **Callum Camazzola** — collaborator (joined 2026-01). Owns `Beam_Membrane/` and `TBCM/` (BCM → BM and BCM → TBCM transfer; RF and autoencoder methods).
