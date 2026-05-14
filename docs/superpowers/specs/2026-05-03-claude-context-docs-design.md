# Claude + Human Context Docs — Design

**Date:** 2026-05-03
**Author:** Ben (with Claude)
**Branch:** feature/fem
**Status:** Approved — ready for implementation plan

---

## 1. Goal

Create a small, durable set of documentation files that:

1. Let any future Claude session pick up full project context fast (machine-readable entry point + structured references).
2. Let a human collaborator (currently Callum, anyone joining later) onboard quickly.
3. Replace the ad-hoc `VocalFoldRegression/PROJECT_CONTEXT.md` and `VocalFoldRegression/PLAN_OF_ACTION.md` with a single coherent doc tree at the repo root.

Scope is `VocalFoldRegression/` only. `glottal_area/`, `OpenIFEM/`, and `Beam+Membrane_ForSean/` are mentioned by name but not deeply documented.

## 2. Constraints & Decisions

| Decision | Choice | Why |
|---|---|---|
| Audience | Both Claude and humans, separate entry points | Claude reads `CLAUDE.md` automatically; humans read `README.md` |
| Doc location | Repo root: `/CLAUDE.md`, `/README.md`, `/docs/` | Standard Claude Code convention; both audiences find it instantly |
| Existing docs | Fold into new structure, then delete | Avoid drift between two sources of truth |
| Tone | Terse + technical (bullets, tables) | Matches existing `PROJECT_CONTEXT.md`; faster to write & maintain |
| Milestones | Both completed log + forward roadmap | Captures momentum and direction |
| Decision log | Single `DECISIONS.md` (append-only) | Lighter than ADR-per-file; promote later if it grows past ~20 entries |
| Deadlines | None | No paper, defense, or sprint pressure |
| Team | Ben + Callum named | Only contributors |
| Sensitivity | Nothing redacted | Repo treated as share-safe |
| State assumption | B+M Phase 1 (data gen) is done | Per user instruction "pretend we have the data" |

## 3. File Tree

```
/CLAUDE.md                     ← machine entry point (auto-loaded)
/README.md                     ← human entry point
/docs/
  ARCHITECTURE.md              ← system design + 3-regressor matrix + transfer strategies
  MILESTONES.md                ← dated history, append-only
  ROADMAP.md                   ← forward-looking phases
  GLOSSARY.md                  ← domain + method + people
  DECISIONS.md                 ← append-only judgment log
/docs/superpowers/specs/
  2026-05-03-claude-context-docs-design.md   ← this file
```

## 4. File Contents

### 4.1 `/CLAUDE.md`

Machine entry point. Concise, scannable, opinionated.

Sections:

- **Project mission** (one paragraph): map vocal-fold motor inputs (a_CT, a_TA, PS) to acoustic outputs (F0, SPL) using ML regressors; replace slow lookup tables; transfer-learn across domains (male BCM → female BCM → B+M).
- **Repo map**: which folders matter (`VocalFoldRegression/`), which are auxiliary (`glottal_area/`, `OpenIFEM/`, `Beam+Membrane_ForSean/`).
- **Stack**: Python (sklearn, TensorFlow/Keras, pandas, numpy), MATLAB for B+M data generation.
- **Hard conventions Claude must obey**:
  - StandardScaler **per-domain** — never share scalers across male/female/B+M (lesson from `NeuralNetwork/interpret.txt`).
  - Input order is always `[a_CT, a_TA, PS]`; output order is always `[F0, SPL]`. Don't reorder.
  - `random_state=42`, 80/20 train/test split — match existing scripts.
  - Don't blindly commit `.pkl`, `.keras`, `.parquet` — check existing patterns and `.gitignore`.
  - When making a non-obvious judgment call, append a line to `docs/DECISIONS.md`.
- **Pointers**: ARCHITECTURE for system design, MILESTONES + ROADMAP for state, GLOSSARY for terms, DECISIONS for the log.
- **Active branch note**: work on `feature/fem`; `main` is stale.

### 4.2 `/README.md`

Human entry point. 3-sentence what-this-is, quick start, nav table.

Sections:

- **What this is**: 3 sentences — vocal-fold regression project, what we predict, why it matters (replaces a slow lookup table; supports transfer to physiologically realistic models).
- **Quick start**: clone, Python env hint (`pip install scikit-learn tensorflow pandas numpy matplotlib`), where data lives (`VocalFoldRegression/BCM Model/MaleBCM.csv`, etc.), how to run a baseline (e.g. `python "VocalFoldRegression/BCM Model/RandomForest/MaleRF.py"`).
- **Nav**: table linking ARCHITECTURE, MILESTONES, ROADMAP, GLOSSARY, DECISIONS, CLAUDE.md.
- **Contributors**: Ben Gladney, Callum.

### 4.3 `/docs/ARCHITECTURE.md`

System design at a moderate depth — enough that a new contributor can locate themselves in 5 minutes.

Sections:

- **Pipeline diagram** (ASCII):
  ```
  [a_CT, a_TA, PS]  →  StandardScaler(input)  →  Regressor  →  StandardScaler⁻¹  →  [F0, SPL]
  ```
- **Domains**: source = male BCM (~90k samples); targets = female BCM (~1.3k after ACFL filter), B+M (~500 valid after NaN drop).
- **Regressor matrix** (table): RF / NN / PR — architecture, hyperparams, source script, target script, transfer strategy, transfer script.
- **Transfer strategies**:
  - Weighted ensemble (RF, PR): `α·source_pred + (1-α)·target_pred`; α tuned per regressor (0.3/0.7 for RF, 0.05/0.95 for PR).
  - Partial layer freezing (NN): clone source model, freeze first N layers, fine-tune rest.
  - TransRF (advanced RF): target-only + residual correction + feature augmentation; K-fold learned weights.
- **Data layer**: schema, NaN handling for B+M, scaler discipline, where each artifact lives.
- **Cross-regressor analysis**: where the comparison scripts live and what they emit.

### 4.4 `/docs/MILESTONES.md`

Dated history, append-only. Approximate dates from git log where possible.

Format: `## YYYY-MM — Title` followed by 1–3 bullets.

Seed entries (dates derived from commit history; mark approximate where uncertain):

- ~2025 mid — Glottal area extraction script (`glottal_area/`)
- ~2025 mid — Random Forest baseline (Male BCM)
- ~2025 mid — Neural Network baseline (Male BCM)
- ~2025 mid — Polynomial Regression baseline (Male BCM)
- ~2025 late — Female BCM compatibility via transfer learning (NN partial freezing)
- ~2025 late — RF female transfer (weighted ensemble, 0.3/0.7)
- ~2025 late — PR female transfer (weighted ensemble, 0.05/0.95)
- ~2025 late — TransRF data efficiency experiment (200–500 sample sweet spot)
- ~2026 early — Cross-regressor transfer comparison (`AllRegressorsTransferComparison.py`)
- 2026-04 — Beam+Membrane file structure created (`Beam+Membrane Model/`, `feature/fem` branch)
- 2026-05 — Plan of action written (1d6b348)
- **2026-05 (assumed)** — B+M data generation complete (per design assumption)

Each entry should answer: what shipped, where the artifact lives.

### 4.5 `/docs/ROADMAP.md`

Forward-looking. Phased, no hard dates.

Sections:

- **Now — Phase 2: B+M data exploration & validation**
  - Run `load_bm_data.py`; check NaN attrition, distributions, dead zones.
  - Compare B+M F0/SPL distributions to male BCM.
- **Next — Phase 3: B+M baselines**
  - RF baseline (small-data hyperparams).
  - NN baseline (smaller architecture, train from scratch).
  - PR baseline (degree 4–5 + Ridge).
  - Establish floor metrics (R², MAE) for F0 and SPL.
- **Then — Phase 4: Male BCM → B+M transfer**
  - RF: TransRF with K-fold learned weights.
  - NN: partial layer freezing (test 2/4/5/6 frozen layers).
  - PR: B+M-specific (degree 4–5 + Ridge) ensembled with male PR; grid search degree × α.
  - Cross-regressor comparison + bootstrap evaluation.
- **Then — Phase 5: analysis + write-up**
  - Quantify domain gap (BCM vs B+M).
  - Document where source-only fails (residual maps).
  - Update `MILESTONES.md` and figures for paper/presentation.
- **Maybe-later**
  - Glottal-area integration as an additional feature/output.
  - OpenIFEM coupling (full FSI rather than reduced-order B+M).
  - Hyperparameter tuning sweeps logged with a tracker.

### 4.6 `/docs/GLOSSARY.md`

Sorted alphabetically within each section.

Sections:

- **Domain**: BCM, Beam+Membrane (B+M), FE model, glottal area, F0, MuscleControlModel, posturing, SPL, subglottal pressure, vocal fold, WRA acoustic solver.
- **Inputs / activations**: a_CT, a_IA, a_LCA, a_PCA, a_TA, PS.
- **Methods**: K-fold CV, partial layer freezing, polynomial regression, random forest, residual correction, Ridge regularization, StandardScaler, TransRF, weighted ensemble.
- **Datasets / files**: `MaleBCM.csv`, `FemaleBCM.csv`, `Data_Membrane_Beam_Model.txt`, `BCMlookuptable.mat`, `data_binary.parquet`.
- **People**: Ben Gladney (lead), Callum (collaborator on transfer learning).

Each entry: term — one-sentence definition — optional pointer to where it appears in code.

### 4.7 `/docs/DECISIONS.md`

Append-only log. Format per entry:

```
## YYYY-MM-DD — Short title
**Context:** what triggered the decision
**Decision:** what we chose
**Why:** the reason / trade-off
**Where it shows up:** files / scripts affected
```

Seed entries (dates approximate; mark unknown where needed):

1. **Per-domain scalers** — Never share StandardScalers between male, female, or B+M data. Each domain fits its own. Source: `NeuralNetwork/interpret.txt`.
2. **PR degree by data size** — Degree 12 for male BCM (~90k samples); degree 4–5 + Ridge for small-data targets like B+M (~500 samples).
3. **RF female transfer weights** — 0.3 male + 0.7 female after empirical tuning.
4. **PR female transfer weights** — 0.05 male + 0.95 female (PR generalizes worse across domains).
5. **TransRF sweet spot** — 200–500 target samples; below that, target-only is too noisy; above that, residual correction stops mattering.
6. **Active branch is `feature/fem`** — `main` is stale; do not pull from it without intent.
7. **Standard split** — `random_state=42`, 80/20 train/test for reproducibility across regressors.

## 5. Migration Plan

1. Write all 7 new files (CLAUDE.md, README.md, ARCHITECTURE.md, MILESTONES.md, ROADMAP.md, GLOSSARY.md, DECISIONS.md).
2. Verify content from old docs is fully covered:
   - `PROJECT_CONTEXT.md` → CLAUDE.md (mission, conventions) + ARCHITECTURE.md (regressor matrix, transfer strategies, directory) + GLOSSARY.md (terms).
   - `PLAN_OF_ACTION.md` → ROADMAP.md (phases 2–5).
3. Delete `VocalFoldRegression/PROJECT_CONTEXT.md` and `VocalFoldRegression/PLAN_OF_ACTION.md`.
4. Commit on `feature/fem` with a message describing the doc consolidation.

## 6. Out of Scope

- Documenting `OpenIFEM/`, `glottal_area/`, `Beam+Membrane_ForSean/` beyond a one-line mention.
- Generating new figures or rerunning experiments.
- Writing CI/CD config, infra docs, or deployment docs.
- ADR-style decision records (deferred until DECISIONS.md grows past ~20 entries).
- Pulling from `origin/main` (user explicitly excluded).

## 7. Success Criteria

- A new Claude session opening this repo gets the full picture from `CLAUDE.md` alone (with pointers to drill in).
- A new human collaborator can run a baseline script in under 10 minutes using only `README.md` + `docs/`.
- No content from `PROJECT_CONTEXT.md` or `PLAN_OF_ACTION.md` is lost in the migration.
- Doc tree contains zero placeholders, TODOs, or contradictions on commit.
