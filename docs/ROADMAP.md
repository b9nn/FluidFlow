# Roadmap

No hard deadlines — phases are ordered, not dated. Each phase ends with a measurable artifact written to disk.

> **State as of 2026-05-03:** Callum's PR #1 already shipped working BCM→BM and BCM→TBCM transfer pipelines with measured results. The "build the transfer pipelines" phase is done. Remaining work is analysis, NN/PR coverage, and write-up.

## Now — Phase A: Digest Callum's results, decide next direction

**Goal:** Brian (returning from hiatus) reviews Callum's empirical results and picks the next research thread.

- Read `PROJECT_GUIDE.md` end-to-end.
- Run one or more of Callum's scripts locally to verify reproducibility (`BM_Summary.py` is a fast sanity check).
- Open `Beam_Membrane/results/rf_transfer_results.json` and `TBCM/results/rf_transfer_results.json` to inspect raw R² numbers.
- Decide: which of Phases B–D below to prioritize.

**Done when:** decision recorded in `DECISIONS.md`.

## Next options (any order)

### Phase B — Extend NN/PR transfer to BM and TBCM

The new BM/TBCM work is RF + autoencoder only. Brian's NN (partial freezing) and PR (weighted ensemble) strategies haven't been tested on these domains.

- Apply NN partial-freezing to BCM→TBCM (likely easy win — same physics family).
- Apply NN partial-freezing to BCM→BM (likely hard — scale mismatch).
- Apply PR ensemble (degree 4–5 + Ridge) to both. Expect PR to struggle on BM specifically.
- Add results to `BM_Summary.py` / `TBCM_Summary.py` as additional rows.

**Done when:** updated cross-regressor comparison includes NN and PR.

### Phase C — Push the small-data regime further

Callum's small-data sweep covers 10–500 BM samples. The interesting question is what happens below 10 (3, 5, 8 samples) — physically motivated priors might dominate at that scale.

- Re-run `BM_SmallData.py` with `n ∈ {3, 5, 8}` and many more bootstrap runs.
- Try a physics-informed prior on the residual model.
- Document where target-only collapses to near-zero R².

**Done when:** updated small-data figure committed.

### Phase D — Write-up / paper draft

- Quantify the BCM↔BM and BCM↔TBCM domain gaps with measured numbers.
- Physical interpretation: what does BM capture that BCM doesn't, and where does that show up in the errors?
- Generate paper-ready figures from `Beam_Membrane/figs/` and `TBCM/figs/`.
- Prose narrative around "transfer for expensive simulators" pitch.

**Done when:** results section drafted; figures finalized.

## Maybe-later

- Female BM / female TBCM transfer (if those datasets exist).
- Glottal-area integration as an additional input/output feature.
- OpenIFEM coupling for full FSI training data (replaces reduced-order BM).
- Hyperparameter sweep tracker (e.g., a small CSV log or a tracker like MLflow).
- Reconcile or delete Brian's local-only `VocalFoldRegression/Beam+Membrane Model/` (pre-Callum scaffolding) and `Beam+Membrane_ForSean/` (Sean's MATLAB).
