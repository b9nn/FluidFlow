# PAPER WIREFRAME — TabPFN as a pretrained low-N surrogate for vocal-fold models

**Owners:** Ben + Callum · **Rewritten:** 2026-06-29 (supervisor restructuring) · **For review by:** Sean, Matías, Jesus, Emiro
**Supervisor feedback driving this draft:** `docs/feedback/2026-06-23-supervisor-feedback.md` (private).

> **Status tags:** **READY** (figure/result exists), **TO-RUN** (doable now, no external dep), **PROVISIONAL** (exists but caveated), **DISCUSS** (open with advisors). Reply by X-ing blocks to cut and ?-ing anything to reframe.

---

## Framing (the supervisor's arc)

TabPFN as the **natural continuation of the surrogate → transfer-learning line**:

1. *Prior paper:* ML models can replace expensive biomechanical simulations for the **forward problem**, but accuracy **degrades at small N** — which is what motivated transfer learning.
2. *This paper's question:* **Can TabPFN act as a pretrained surrogate for vocal-fold models in the low-N regime, while competing with transfer-learning strategies?**

Both methods are "pretrained," which is the honest way to put them on equal footing:
- **TabPFN** — pretrained once by Prior Labs on millions of synthetic tabular problems (frozen; in-context fit, no per-task training).
- **TransRF** — "pretrained" on the cheap **BCM source model**, then adapted with N target rows.

The variable that decides the contest is **source–target alignment**: when the cheap source is a good prior, transfer competes; when it isn't, TabPFN wins outright.

## Working title (pick / edit)
- A. *"Pretrained tabular transformers as low-N surrogates for vocal-fold models: a no-source alternative to transfer learning"*
- B. *"TabPFN for sparse-data emulation of vocal-fold biomechanics, with physiological outputs"*

---

## Section flow

```
1. Intro (surrogate -> transfer -> TabPFN)
2. Methods
   2a. Physical vocal-fold models (BM main, TBCM secondary)
   2b. Low-N regression strategies (TabPFN, TransRF)
3. RESULT 1: metrics vs N        (R^2 + nRMSE, BCM->BM)        [headline]
4. RESULT 2: activation-map reconstruction  (+ image-level MAE/RMSE)
5. RESULT 3: runtime             (adaptation vs inference)
6. RESULT 4: richer outputs      (multi-output ACFL/PC/CPP; single-head ablation)
7. RESULT 5: structured missing-modality   (sensor/output dropout)
8. Discussion -> inverse-problem outlook (exploratory)
9. Conclusion
```

---

## 1. Introduction — *no figure*
- Physical voice models (BCM, TBCM, Beam–Membrane) are accurate but **expensive**; ML surrogates need many simulations and **degrade at small N** (prior paper).
- Transfer learning helps but needs a trained **source model + target adaptation**, and only pays off when source↔target are aligned.
- **Enter TabPFN:** a pretrained tabular foundation model that needs **no source and no training** — just the few target rows. Question: does it reduce/replace the need for transfer in the low-N regime?

## 2. Methods

### 2a. Physical vocal-fold models — *schematic table* — **READY**
- **Beam–Membrane (BM)** — *main case.* Finite-element / continuum-like; the expensive, structurally distinct target. **Far** from the BCM source (different physics) → the regime where the low-N question bites hardest.
- **TBCM** — *secondary case.* Lumped-element, same family as BCM; carries the **gridded** `(a_CT × a_TA)` activation maps (real ODE ground truth) → used for the map-reconstruction result.
- **BCM** — the cheap **source** (360k samples) for all transfer.
- *Supplement:* male→female BCM. **BCM→TBCM transfer:** DISCUSS — likely cut to keep the message focused on BM-main.

### 2b. Low-N regression strategies — *no figure*
- **TabPFN** (pretrained tabular transformer; in-context fit, per-output scaling; single-head per output and a stacked **multi-head** for joint outputs).
- **TransRF** (BCM source RF + residual/feature-augmented sub-models, non-negative ensemble weights).

---

## 3. RESULT 1 — metrics vs training size — **READY** *(headline)*
**Claim:** On BCM→BM (far domain), TabPFN beats transfer at every N, on both accuracy and error, and the gap is largest in the low-N regime the paper is about.
- **Fig 1** `Beam_Membrane/figs/bm_ext_metrics_vs_n.png` — 2×5 grid: **R² (top) and normalized RMSE (bottom)** vs N for F0, SPL, ACFL, PC, CPP. TabPFN vs TransRF vs target-only, **mean±std over 5 seeds**, grid `[10,20,50,100,200,500]` (consistent with all other figures).
- Headline numbers (TabPFN single / TransRF), R² @ N=50 → N=200:
  - F0 0.91/0.70 → 0.98/0.87 · SPL 0.82/0.53 → 0.93/0.77 · ACFL* 0.97/0.64 → 0.99/0.92 · PC 0.76/0.47 → 0.94/0.68 · CPP 0.46/0.28 → 0.60/0.49.
- *Why nRMSE not MAPE:* SPL goes negative and PC/CPP cross ~0 → MAPE ill-defined; range-normalized RMSE is robust and dimensionless.

## 4. RESULT 2 — activation-map reconstruction — **READY (TBCM)** / **DISCUSS (BM)**
**Claim:** From as few as N=10–100 samples, the model reconstructs the full JASA-style `(a_CT × a_TA)` motor-control map — the *physiological structure* of the control space, not just point accuracy.
- **Fig 2** `TBCM/figs/tbcm_motor_map_{F0,SPL}.png` (+ `_error`) — rows N{10,100,1000} × cols {ODE ground truth | TabPFN | TransRF}.
- **NEW (supervisor):** add an **image-level error metric** — MAE/RMSE over all grid cells — as a summary number per map panel (the error maps already carry per-panel RMSE; promote to a table). **TO-RUN.**
- **Structural note:** TBCM is the map hero because it is a **gridded** model with real ODE ground truth. **BM is randomly sampled (no grid)** → a true-ground-truth BM map isn't possible; a full-data surrogate map could stand in but must be labeled as such. DISCUSS whether BM gets a surrogate map or maps stay TBCM-only.

## 5. RESULT 3 — runtime: adaptation vs inference — **READY (cloud)** / **TO-RUN (local)**
**Claim:** TabPFN is practical, and crucially needs **no source model and no source data** — the per-domain pretraining transfer requires is eliminated.
- **Fig 3** `Beam_Membrane/figs/bm_ext_runtime.png` — stacked bars per method, **adaptation (per task) vs inference**, at N=200, 5 outputs.
- TransRF: **279 s one-time source-RF pretrain** + 5.0 s adapt + 1.9 s infer. TabPFN: **0 s** per-domain pretrain + ~7.8 s adapt + ~15.6 s infer.
- **Caveat / TO-RUN:** TabPFN inference here is the **cloud client incl. network round-trip** — a deployment number, not compute. **Local TabPFN timing pending** (license-gated weight download; needs a one-time interactive accept).

## 6. RESULT 4 — richer physiological outputs — **READY** *(ACFL provisional)*
**Claim:** TabPFN extends from F0/SPL to physiological outputs (**ACFL, PC, CPP**) at no extra machinery; a single joint **multi-head** delivers all outputs at once.
- **Fig 4** `Beam_Membrane/figs/bm_ext_multihead_vs_single.png` — single-output vs multi-head TabPFN, R² vs N. Multi-head trails singles by ~0.05; **lead with multi-output** (richer-outputs-for-free story), single-head as **ablation**. Multi-head capped at N=200 (5×N ≤ TabPFN's 1000-row limit).
- **PROVISIONAL (ACFL):** BM ACFL runs ~2–4× the BCM source — a definition/`sr` nuance (Bernoulli source flow vs tract-loaded), pending reconciliation against Parra's exact ACFL routine. **PC and CPP are scale-consistent** → the safe physiological headliners. See `Beam_Membrane/BM_EXTENDED_DATASET_METHODOLOGY.md`.

## 7. RESULT 5 — structured missing-modality — **TO-RUN** *(supervisor reframe)*
**Claim:** Reframed from random deletion to **clinically structured missing data**: a whole input/output *modality* (acoustic, aerodynamic, or physiological) is absent in X% of rows — mimicking a sensor/measurement not available for a subject.
- **Fig 5** (to generate): degrade by dropping a modality column in X% of training rows; report accuracy vs X for TabPFN (its probabilistic head degrades gracefully) vs transfer.
- Bridge to clinical relevance; sets up the inverse problem.

## 8. Discussion → inverse-problem outlook — *no figure* — **DISCUSS**
- Forward: outputs from (Ps, muscle activations). **Inverse:** estimate control/physiological variables from observable features (ACC, acoustic, aerodynamic) — where missing-modality is most natural.
- TabPFN could connect three prior lines: ML surrogate (forward), deterministic Ps/collision/activation estimation, and probabilistic PBNN.
- **Open (supervisor, "discuss tomorrow"):** inverse as an *exploratory section if clean*, or the seed of a **second paper**. Kept as outlook here, not a main result.

## 9. Conclusion — *no figure*
- A pretrained tabular transformer is a practical, **no-source, no-train** low-N surrogate for expensive voice models — competitive with, and on far domains better than, purpose-built transfer learning, and it extends naturally to physiological outputs.

---

## Figure inventory
| # | Figure | File | Status |
|---|--------|------|--------|
| 1 | Metrics vs N (R²+nRMSE), BCM→BM | `Beam_Membrane/figs/bm_ext_metrics_vs_n.png` | READY |
| 2 | TBCM motor-control maps (+image-level error) | `TBCM/figs/tbcm_motor_map_*.png` | READY / error-metric TO-RUN |
| 3 | Runtime adaptation vs inference | `Beam_Membrane/figs/bm_ext_runtime.png` | READY (cloud) / local TO-RUN |
| 4 | Multi-output vs single-head TabPFN | `Beam_Membrane/figs/bm_ext_multihead_vs_single.png` | READY |
| 5 | Structured missing-modality | — | TO-RUN |

## Decisions for the advisors (the "put the X" list)
1. **Title** A / B.
2. **BCM→TBCM transfer** — cut entirely, or keep as supplement? (recommend cut.)
3. **BM activation map** — surrogate-reference map for BM, or keep maps TBCM-only? (recommend TBCM-only, honest about the grid.)
4. **ACFL** — include provisional, or hold until reconciled with Parra's definition? (PC/CPP carry the physiological story regardless.)
5. **Inverse problem** — exploratory section now, or second paper?
