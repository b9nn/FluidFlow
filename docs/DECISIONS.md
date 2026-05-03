# Decisions

Append-only log of judgment calls. Newest at the bottom.

Format:

```
## YYYY-MM-DD — Short title
**Context:** what triggered the decision
**Decision:** what we chose
**Why:** the reason / trade-off
**Where it shows up:** files / scripts affected
```

---

## 2025-10 — Standard data split: `random_state=42`, 80/20

**Context:** Reproducibility across regressor scripts.
**Decision:** Every script uses `train_test_split(test_size=0.2, random_state=42)`.
**Why:** Same hold-out across RF, NN, PR makes cross-regressor comparison meaningful.
**Where it shows up:** every `*.py` under `BCM Model/` and `Beam+Membrane Model/`.

## 2025-10 — Per-domain StandardScalers (never share across domains)

**Context:** Initial female-BCM transfer learning produced near-zero R² when the male input scaler was reused on female data.
**Decision:** Each domain (male, female, B+M) fits and saves its own input scaler and its own per-output scalers (one for F0, one for SPL).
**Why:** Different domains have different feature distributions; reusing a male scaler shifts and scales female data into the wrong subspace before the model sees it.
**Where it shows up:** `BCM Model/NeuralNetwork/interpret.txt` (incident notes); every transfer script.

## 2025-10 — Polynomial Regression degree 12 for male BCM

**Context:** Male BCM has ~90k samples and a smooth nonlinear input/output map.
**Decision:** Use `PolynomialFeatures(degree=12)` + plain `LinearRegression` for the male PR baseline.
**Why:** Sample count tolerates the high-degree feature blow-up without overfitting; gives PR a fair chance against RF/NN.
**Where it shows up:** `BCM Model/PolynomialRegressor/MalePR.py`.

## 2025-11 — RF female transfer weights: 0.3 male + 0.7 female

**Context:** Tuning the weighted ensemble for RF on ~1.3k female samples.
**Decision:** `α = 0.3` source, `1-α = 0.7` target.
**Why:** Empirically best on the female test set; target dominates but source still adds signal.
**Where it shows up:** `BCM Model/RandomForest/FemaleRFTransfer.py`.

## 2025-11 — PR female transfer weights: 0.05 male + 0.95 female

**Context:** Same tuning exercise for PR.
**Decision:** `α = 0.05` source, `1-α = 0.95` target.
**Why:** Polynomial regression generalizes poorly across domains — degree-12 male features extrapolate badly onto female inputs. Trust target almost entirely.
**Where it shows up:** `BCM Model/PolynomialRegressor/FemalePRTransfer.py`.

## 2025-11 — TransRF sweet spot: 200–500 target samples

**Context:** Data efficiency experiment for the advanced RF transfer method.
**Decision:** Best transfer benefit observed at 200–500 target samples.
**Why:** Below 200, target-only is too noisy to constrain the residual sub-model; above 500, target-only alone is already strong and source contribution stops mattering.
**Where it shows up:** `BCM Model/RandomForest/DataEfficiencyExperiment.py`.

## 2026-02 — PR for small target domains: degree 4–5 + Ridge

**Context:** Planning the B+M transfer for PR (~500 samples).
**Decision:** Drop polynomial degree from 12 to 4–5 and use `Ridge` instead of plain `LinearRegression`.
**Why:** Degree 12 with 500 samples will overfit catastrophically. Lower degree + L2 penalty matches the data scale.
**Where it shows up:** `Beam+Membrane Model/PolynomialRegressor/BeamMembranePRTransfer.py` (planned).

## 2026-02 — Active branch: `feature/fem`; `main` is stale

**Context:** Local development outpaced `origin/main`; pulling from main would clobber active work.
**Decision:** Treat `feature/fem` as the source of truth. Do not rebase onto or merge from main without intent.
**Why:** Avoids accidental loss of in-progress B+M scaffolding.
**Where it shows up:** repo state.

## 2026-05 — Documentation lives at repo root, not under `VocalFoldRegression/`

**Context:** Replacing ad-hoc `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with a coherent doc tree.
**Decision:** `CLAUDE.md` and `README.md` at repo root; reference docs under `/docs/`.
**Why:** Both audiences (Claude auto-loads `CLAUDE.md` from root; humans expect `README.md` at root) find the docs without nesting. Single source of truth avoids drift between two paths.
**Where it shows up:** `/CLAUDE.md`, `/README.md`, `/docs/*.md`.
