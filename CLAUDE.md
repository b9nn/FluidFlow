# CLAUDE.md

> Auto-loaded by Claude Code in every session. Read this first.

## Project mission

Map vocal-fold motor inputs (`a_CT`, `a_TA`, `PS`) to acoustic outputs (`F0` Hz, `SPL` dB) using ML regressors. Goal: replace slow physics-based lookup tables with fast, trainable models, and transfer-learn across domains (male BCM → female BCM → Beam+Membrane).

Active branch: `feature/fem`. **Do not pull from `origin/main`** — it is stale.

## Repo map

| Path | Status | What it is |
|---|---|---|
| `VocalFoldRegression/` | **primary** | All ML work (BCM + B+M, three regressors, transfer learning) |
| `VocalFoldRegression/Beam+Membrane_ForSean/` | aux | Sean's MATLAB FE model — generates B+M training data |
| `glottal_area/` | aux | Vocal-fold area extraction scripts (separate scope) |
| `OpenIFEM/` | aux | External FE/FSI solver, vendored — not actively used |
| `figs/` | aux | Top-level scratch figures |
| `docs/` | docs | Architecture, milestones, roadmap, glossary, decisions |

## Stack

- **Python**: `scikit-learn`, `tensorflow` / `keras`, `pandas`, `numpy`, `matplotlib`, `joblib`, `scipy`
- **MATLAB**: B+M data generation (`Beam+Membrane_ForSean/Randomly_Generating_Data_Membrane_Beam_Model.m`)

## Hard conventions — obey these

1. **Per-domain StandardScalers.** Each domain (male, female, B+M) fits its own input scaler and its own per-output scalers (one for `F0`, one for `SPL`). Never use a male scaler on female or B+M data — it has destroyed past experiments (see `VocalFoldRegression/BCM Model/NeuralNetwork/interpret.txt`).
2. **Schema is fixed.** Inputs always `[a_CT, a_TA, PS]`. Outputs always `[F0, SPL]`. Don't reorder.
3. **Reproducibility.** `random_state=42`, `test_size=0.2` train/test split — match every existing script.
4. **Don't blob-commit binaries.** `.pkl`, `.keras`, `.parquet` are committed selectively. Check the surrounding folder before adding new ones.
5. **Append a line to `docs/DECISIONS.md`** whenever you make a non-obvious judgment call (hyperparam choice, weighting, transfer strategy variant).

## Interaction conventions

- When you need to ask Brian multiple-choice or preference questions, use the `AskUserQuestion` tool ("planning format" — rendered as chips). Don't ask A/B/C/D in plain text. Inline prose questions are fine only for genuinely open-ended prompts.

## Where to look

- System design + regressor matrix → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- What's been done (dated) → [`docs/MILESTONES.md`](docs/MILESTONES.md)
- What's next → [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Term lookup → [`docs/GLOSSARY.md`](docs/GLOSSARY.md)
- Why we chose X → [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Contributors

- Brian Gladney — lead
- Callum — transfer learning collaborator
