# Roadmap

No hard deadlines — phases are ordered, not dated. Each phase ends with a measurable artifact written to disk.

> **State as of 2026-05-07:** TODO #1 (non-transfer alternates) closed — GP and TabPFN both ran on real BM data and beat all of Callum's transfer methods at small N. Headline result documented in `docs/MILESTONES.md` 2026-05-05 entry. Decision point coming at the next 1pm sync: which of Phase A / Phase B below to scope next.

## Decision point — pick one of A / B at next 1pm

### Phase A — TBCM → BM two-stage transfer (TODO #2)

**Question:** does *physics-aligned* transfer beat the alternates? BCM→BM failed because of the `Ps` scale mismatch. TBCM has the same `Ps` range as BM and the same physics family, so transfer might actually help here.

**Plan:**
- Train RF on TBCM (~43k samples) as the source
- Use BCM as a control source (run parallel)
- Apply Callum's 6 RF methods + 3 AE methods with TBCM as source
- Add TabPFN with TBCM features added (TabPFN-with-source-features as a baseline)
- Compare against Brian's GP + TabPFN non-transfer baselines

**Effort:** ~1–2 weeks. Reuses existing harness; mostly data plumbing.

**Done when:** updated `bm_alternates.png` + summary CSV showing TBCM→BM curves alongside BCM→BM and non-transfer.

### Phase B — Real PDE-residual PINN over BM equations (TODO #15)

**Question:** can we build a physics surrogate that generalizes outside the BM training distribution and provides gradients for inverse problems? Different *kind* of model than GP/TabPFN — not a marginal R² improvement, a different deliverable.

**Plan:**
- Port BM Stage 2 (constitutive algebra) and Stage 3 (coupled membrane + beam PDE) from MATLAB to PyTorch using the equations extracted in `docs/BM_GOVERNING_EQUATIONS.md`
- Predict displacement fields `w(x, y, t)` and `w_b(x, t)` with neural networks
- Compute PDE residuals at collocation points via autograd
- Encode boundary and initial conditions as additional loss terms
- Derive F0 (FFT on `Am(t)`) and SPL (RMS via simplified WRA or call-out) from the predicted fields
- Connect inputs `(a_CT, a_TA, PS, eps0, Theta_G)` — call MATLAB once per training sample for `(eps0, Theta_G)` since posturing is black-box

**Effort:** ~3–4 weeks of focused work. Substantial engineering project.

**Done when:** PINN predicts F0/SPL on a held-out BM set with R² ≥ 0.5 *and* generalizes to one OOD test condition (e.g., a `Ps` value outside the training range). Then we can compare its OOD performance vs GP/TabPFN, which can't extrapolate.

## Then — Phase C: Paper write-up (TODO #6)

Whichever of A or B happens first, the write-up should fold in:

- BCM→BM and BCM→TBCM transfer (Callum's full results)
- Non-transfer alternates (GP + TabPFN) — the surprise headline
- Domain-gap quantification: when does transfer help, when does it hurt?
- Whichever follow-up was pursued (TBCM→BM transfer or real PINN)

**Done when:** results section drafted, figures paper-ready.

## Maybe-later

- Female BM / female TBCM transfer (if those datasets exist)
- Glottal-area integration as additional input/output feature
- OpenIFEM coupling for full FSI training data (replaces reduced-order BM)
- Hyperparameter sweep tracker (e.g., a small CSV log or MLflow)
- Decide fate of `VocalFoldRegression/Beam+Membrane_ForSean/` — Sean's upstream MATLAB FE solver (currently untracked locally; needs his consent before vendoring or a `data/README.md` documenting it as an external dependency). Tracked as `team/TODO.md` #7
- Re-run Callum's `BM_SmallData.py` with JSON dump for tighter alternates-vs-transfer head-to-head — `team/TODO.md` #13

## What's already done (was in this roadmap before)

- Phase 0 (digest Callum's PR + decide direction) — closed 2026-05-04
- Original "Phase B/C" (NN/PR coverage of BM/TBCM, push small-N) — superseded by the non-transfer alternates direction; remaining bits are TODO #3 and #4 in backlog
