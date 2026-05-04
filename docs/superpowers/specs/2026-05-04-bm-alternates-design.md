# Non-Transfer Alternate Methods for BM — Design

**Date:** 2026-05-04
**Author:** Brian (with Claude)
**Branch:** feature/fem
**Tracks:** `team/TODO.md` #1
**Status:** Approved — proceeding to implementation

---

## Goal

Establish non-transfer baselines for BCM→BM that match or beat Callum's transfer methods in the very-small-N regime (5–100 samples). If a non-transfer method matches transfer here, transfer's value drops; if not, the case for transfer strengthens. Either result is publishable.

## Constraints (set during brainstorm)

- **Sample regime:** N ∈ [5, 10, 20, 30, 50, 75, 100]
- **Test set:** identical to Callum's BM_SmallData (`random_state=42`, `test_size=0.2`, fixed once)
- **Replicates:** 10 bootstrap runs per (method, n) with seeds `42 + run_idx`
- **Schema:** `[a_CT, a_TA, PS] → [F0, SPL]` — drop `a_LCA` per existing convention
- **Per-domain scalers** — fit on each sub-sample
- **Comparison target:** Callum's `TransRF` and `Feature Augmentation` curves from `rf_transfer_results.json`

## Methods

### M1 — Gaussian Process (GP)

- `sklearn.gaussian_process.GaussianProcessRegressor`, no new dep
- Kernel: `ConstantKernel * Matern(nu=2.5) + WhiteKernel`
- Independent regressor per output (F0 and SPL trained separately)
- Standardize inputs and outputs per sub-sample; never reuse a scaler
- Optimize kernel hyperparameters via marginal-likelihood (sklearn default)

### M2 — Physics-Informed MLP (PINN)

- `torch`, small MLP `[3 → 32 → 32 → 2]`, ReLU activations
- Joint `[F0, SPL]` head; standardize outputs per sub-sample
- Loss: `MSE(y_pred, y_true) + λ · monotonicity_penalty`
- **Priors encoded** (confirmed strong, low-disagreement):
  - `∂F0 / ∂a_CT ≥ 0` (longer fold → higher pitch)
  - `∂SPL / ∂PS ≥ 0` (more pressure → louder)
- **Priors deferred** to phase-2 discussion: `∂F0 / ∂PS` (mixed sign in BM data)
- Monotonicity penalty: sample random pairs `(x_i, x_j)` from a Sobol grid over the input cube; compute `relu(-(f(x_high) - f(x_low)))` for each constraint
- λ default = 0.1; sweep if needed
- Optimizer: Adam, lr=1e-3, max 2000 epochs with early stopping on a held-out 10% slice

### M3 — TabPFN

- `pip install tabpfn` — new dep, document in DECISIONS
- Single-output → two TabPFN regressors (one for F0, one for SPL)
- Cap at 1000 train samples (TabPFN limit; matches our regime)
- No hyperparameters to tune; just fit and predict

## File plan

```
Beam_Membrane/
  BM_Alternates.py                ← NEW. ~400 lines. All three methods + harness
  results/alternates_results.json ← NEW output
  figs/bm_alternates.png          ← NEW figure
  BM_Summary.py                   ← MODIFIED to read alternates JSON
```

## Output schema

`alternates_results.json`:
```json
{
  "GP":     {"5":  {"r2_f0": [r1,...,r10], "r2_spl": [r1,...,r10]}, "10": {...}, ...},
  "PINN":   {...},
  "TabPFN": {...}
}
```

## Figure

`bm_alternates.png` — two panels (F0, SPL).
- X axis: `n_samples` (log)
- Y axis: R² (mean across 10 runs, ± 1 std as shaded band)
- Lines: GP, PINN, TabPFN
- Dashed reference lines: Callum's TransRF and Feature Aug curves loaded from `rf_transfer_results.json`

## BM_Summary.py extension

- Load `alternates_results.json` if present
- Add alternate-method rows to the existing comparison figure / CSV
- Don't break if the file isn't present yet (run-order independence)

## Phasing

1. **Phase 1** — Scaffold `BM_Alternates.py` with experiment harness, GP method, run on 5/10/20/50/100, commit.
2. **Phase 2** — Add PINN. Confirm monotonicity priors with Brian. Tune λ. Commit.
3. **Phase 3** — Add TabPFN. Document new dep in DECISIONS. Commit.
4. **Phase 4** — Extend `BM_Summary.py` to read alternates. Regenerate comparison figure. Commit.
5. **Phase 5** — DECISIONS entries (kernel choice, physics priors, TabPFN cap), MILESTONES update, mark TODO #1 → `done`, migrate to MILESTONES, mention in MEETING_NOTES. Commit.

## Out of scope

- Method tuning beyond what fits the harness (no Bayesian-optimization sweeps).
- Transfer variants of these methods (no `GP-Transfer` etc. — that's a separate exploration).
- Different test splits or bootstrap counts than Callum's.
- TBCM equivalent — only BM in scope here.
- Updating `archive/random_forest/GaussianProcessTransfer.py` — fresh implementation, treat archived version as reference only.

## Success criteria

- Three methods land with R² curves in `bm_alternates.png` for N ∈ [5, 10, 20, 30, 50, 75, 100]
- `BM_Summary.py` includes alternate methods in comparison output
- DECISIONS log captures the kernel choice, physics priors, and any λ that was kept
- No same-test-set drift — confirmed by re-running Callum's `BM_SmallData.py` and matching his published numbers within bootstrap variance
