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
**Where it shows up:** every `*.py` under `VocalFoldRegression/`, `Beam_Membrane/`, `TBCM/`.

## 2025-10 — Per-domain StandardScalers (never share across domains)

**Context:** Initial female-BCM transfer learning produced near-zero R² when the male input scaler was reused on female data.
**Decision:** Each domain (male BCM, female BCM, BM, TBCM) fits and saves its own input scaler and its own per-output scalers.
**Why:** Different domains have different feature distributions; reusing a male scaler shifts and scales target data into the wrong subspace before the model sees it.
**Where it shows up:** `BCM Model/NeuralNetwork/interpret.txt` (incident notes); every transfer script.

## 2025-10 — Polynomial Regression degree 12 for male BCM

**Context:** Male BCM has ~54k samples and a smooth nonlinear input/output map.
**Decision:** Use `PolynomialFeatures(degree=12)` + plain `LinearRegression` for the male PR baseline.
**Why:** Sample count tolerates the high-degree feature blow-up without overfitting.
**Where it shows up:** `BCM Model/PolynomialRegressor/MalePR.py`.

## 2025-11 — RF female transfer weights: 0.3 male + 0.7 female

**Context:** Tuning the weighted ensemble for RF on ~1.3k female samples.
**Decision:** `α = 0.3` source, `1-α = 0.7` target.
**Why:** Empirically best on the female test set.
**Where it shows up:** `BCM Model/RandomForest/FemaleRFTransfer.py`.

## 2025-11 — PR female transfer weights: 0.05 male + 0.95 female

**Context:** Same tuning exercise for PR.
**Decision:** `α = 0.05` source, `1-α = 0.95` target.
**Why:** Polynomial regression generalizes poorly across domains — degree-12 male features extrapolate badly onto female inputs. Trust target almost entirely.
**Where it shows up:** `BCM Model/PolynomialRegressor/FemalePRTransfer.py`.

## 2025-11 — TransRF data efficiency sweet spot: 200–500 target samples

**Context:** Brian's earlier data efficiency experiment for the advanced RF transfer method on female BCM.
**Decision:** Best transfer benefit observed at 200–500 target samples.
**Why:** Below 200, target-only is too noisy; above 500, target-only alone is already strong.
**Where it shows up:** `BCM Model/RandomForest/DataEfficiencyExperiment.py`.

## 2026-02 — PR for small target domains: degree 4–5 + Ridge

**Context:** Plan for B+M PR transfer (~500 samples).
**Decision:** Drop polynomial degree from 12 to 4–5 and use `Ridge` instead of plain `LinearRegression`.
**Why:** Degree 12 with 500 samples will overfit catastrophically.
**Where it shows up:** Planned for `BeamMembranePRTransfer.py` (Brian's local stub, not yet committed; superseded by the RF/AE-only direction Callum took).

## 2026-05-02 — Adaptive RF complexity, replacing fixed `n_estimators=300`

**Context:** Callum's BM/TBCM transfer scripts work across small (10) to large (40k+) target sample counts. A single hyperparam config either underfits the large case or overfits the small case.
**Decision:** Use `get_model_params(n_samples)` to scale `n_estimators` (20→300) and `max_depth` (3→None) with sample count.
**Why:** Prevents overfitting at small sizes and underfitting at large sizes without hyperparameter search per run.
**Where it shows up:** `Beam_Membrane/BM_TransferRF.py:51`, `Beam_Membrane/BM_SmallData.py:47`, `TBCM/TBCM_TransferRF.py`, `TBCM/TBCM_SmallData.py`.

## 2026-05-02 — Drop `a_LCA` (BM-only feature)

**Context:** BM data includes an extra `a_LCA` activation `∈ [0.5, 1.0]` not in BCM.
**Decision:** Don't use `a_LCA` as a feature. Train BM RF on `[a_CT, a_TA, PS]` only.
**Why:** Empirically `a_LCA` has near-zero correlation with `F0` and `SPL` in the BM data; including it as a feature doesn't improve transfer and creates a feature-mismatch with BCM.
**Where it shows up:** `Beam_Membrane/BM_TransferRF.py`, `Beam_Membrane/BM_TransferAE.py`. Files like `bm_pred_vs_actual_aLCA.png` document the comparison run with the feature included.

## 2026-05-02 — Best small-data BM method depends on sample size

**Context:** Cross-method comparison at 10–500 BM samples (10 runs each).
**Decision:** No single best method. Use **Feature Augmentation** at 20–75 samples, **TransRF** at 100+ samples. Avoid **Residual Correction** at any small size.
**Why:** Feature Aug doesn't force trust in BCM's absolute predictions, which is critical when the BCM-BM scale mismatch is large. TransRF needs enough data to estimate sub-model weights reliably. Residual Correction trusts raw BCM predictions, which are way off in scale at all sizes.
**Where it shows up:** `Beam_Membrane/BM_SmallData.py`; figure `bm_small_data_all.png`.

## 2026-05-02 — Autoencoders underperform RF on BM (use them for TBCM only)

**Context:** AE methods (Vanilla, MMD, DAAE) tested on both BM and TBCM transfer.
**Decision:** AE is competitive on TBCM (same physics family) but loses to RF on BM. Default to RF methods for BM; keep AE for TBCM as a comparison point.
**Why:** AE-based domain adaptation requires the source and target distributions to be related by a smooth manifold mapping. BM's different physics + scale mismatch breaks that assumption.
**Where it shows up:** `Beam_Membrane/figs/bm_rf_vs_ae.png`, `Beam_Membrane/BM_TransferAE.py`.

## 2026-05-02 — Use Spearman rank correlation alongside R² for transfer evaluation

**Context:** BCM source-only predictions on BM have R² ≈ −2.0 (terrible absolute accuracy) but Spearman 0.81/0.74 (good rank ordering).
**Decision:** Report Spearman alongside R² when evaluating cross-domain transfer.
**Why:** Spearman captures whether the source model has learned the right *shape* of the relationship even when the *scale* is wrong — and the right shape is exactly what residual / feature-aug / TransRF methods can exploit.
**Where it shows up:** `Beam_Membrane/BM_Summary.py`, `TBCM/TBCM_Summary.py`.

## 2026-05-02 — Data loading convention: `index_col=0` + `Ps → PS`

**Context:** All datasets are CSVs with an unnamed index column and use `Ps` for subglottal pressure.
**Decision:** Standardize to:
```python
df = pd.read_csv('dataset_*.csv', index_col=0)
df.rename(columns={'Ps': 'PS'}, inplace=True)
```
**Why:** Consistent column names across BCM / TBCM / BM let the same model code work on all three.
**Where it shows up:** every `*.py` in `Beam_Membrane/` and `TBCM/`.

## 2026-02 — Active branch: `feature/fem`

**Context:** Brian and Callum worked on parallel branches (`feature/fem` and `cc-dev`) from common ancestor `fc24567`.
**Decision:** Continue work on `feature/fem`.
**Why:** Brian's docs commits live there; `cc-dev` was Callum's PR feeder branch and is merged to `main`.
**Where it shows up:** repo state. Note: as of 2026-05-03, `main` is no longer stale — Callum's PR landed and `feature/fem` has been merged with `origin/main`.

## 2026-05-02 — Documentation lives at repo root, not under `VocalFoldRegression/`

**Context:** Replacing ad-hoc `VocalFoldRegression/PROJECT_CONTEXT.md` and `PLAN_OF_ACTION.md` with a coherent doc tree.
**Decision:** `CLAUDE.md` and `README.md` at repo root; reference docs under `/docs/`.
**Why:** Both audiences (Claude auto-loads `CLAUDE.md` from root; humans expect `README.md` at root) find the docs without nesting.
**Where it shows up:** `/CLAUDE.md`, `/README.md`, `/docs/*.md`.

## 2026-05-04 — Shared agile workspace at `/team/`, per-task ownership

**Context:** Brian and Callum sync at ~1pm a few times a week. Need a single place both Claude workflows can read for current state — what's in flight, who owns what, recent decisions.
**Decision:** Add `/team/` folder with `TODO.md`, `BOARD.md`, `MEETING_NOTES.md`, `README.md`. Track ownership per-task with an `owner` field (`brian`, `callum`, `shared`, `tbd`) rather than fixing it by codebase area. Both contributors can pick up work in either codebase. Strategic research roadmap stays in `docs/ROADMAP.md`; `/team/` is operational (this week, this sprint).
**Why:** Splitting ownership by codebase area would freeze responsibilities and make cross-area contributions awkward. A per-task owner field scales as work moves and keeps both Claudes informed about who's doing what.
**Where it shows up:** `team/*.md`; `CLAUDE.md` "Team workflow" section and "Where to look" list; "Contributors" reframed as authorship-only.

## 2026-05-03 — Keep Callum's `PROJECT_GUIDE.md` alongside `/docs/`

**Context:** Merging Callum's main into `feature/fem` brings `PROJECT_GUIDE.md`, which overlaps with `README.md` + `docs/ARCHITECTURE.md`.
**Decision:** Keep both. `PROJECT_GUIDE.md` is preserved as Callum's hands-on guide for `Beam_Membrane/` and `TBCM/`. `README.md` and `docs/` are the canonical entry points.
**Why:** The two docs serve subtly different purposes (Callum's is method-focused with empirical tables; ours is repo-wide with conventions). Folding them risks losing his framing. Better to cross-link.
**Where it shows up:** `README.md` nav table, `CLAUDE.md` repo map.

## 2026-06-16 — JASA TBCM TabPFN experiments: scope & design choices

**Context:** Jesus shared the enriched multidimensional TBCM dataset (`JASA_data.csv`, 80,751 rows) per the 2026-06-04 advisor meeting. Three experiments requested: multi-dimensional features, missing-data extrapolation, compute-time vs data size.
**Decision:**
- Inputs kept to the fixed schema `[a_CT, a_TA, PS]`; richer columns (`a_LCA`, `PL`, `PCA`, `PGO`) dropped. Dimensionality expansion applied to **outputs** only.
- Output set = 6 clinical features: `F0, SPL, ACFL (AC flow), PC (collision pressure), MFDR, CPP`. All NaN-free; matches the meeting's named metrics.
- Missing-data holdout = high-activation corner `a_CT > 0.7 AND a_TA > 0.7` (6,364 rows, ~7.9%). A same-size random holdout is reported alongside as an in-distribution reference, to isolate the extrapolation penalty.
- Train size capped at 1000 (small-data regime + TabPFN cloud cap `TABPFN_MAX_TRAIN=1000`); the 80k rows are only a sampling pool, never a training set. N grid `[10,50,100,200,500,1000]`, 2 seeds, 300-row test set — trimmed to keep the cloud-API run ~10 min.
**Why:** `ACFL`→AC flow and `PC`→collision pressure are the meeting's explicit clinical targets; the corner holdout mimics clinical data that never samples extreme muscle activations. Runtime is dominated by per-call network latency to the tabpfn-client cloud API, not by dataset size, so the lever is call count (seeds × N × features), not row count.
**Where it shows up:** `TBCM/JASA_TabPFN_Experiments.py`, `TBCM/dataset_JASA.csv`, `TBCM/results/jasa_tabpfn_results.json`, `TBCM/figs/jasa_*.png`.

## 2026-08-20 — Draft 5 statistics run: protocol, blend weights, RF schedule

**Context:** Draft 4 shipped with placeholder numbers in Table 1 (all three baseline columns carried the same value per row), no target-alone baseline anywhere in the codebase, and no figure for `bm_ext_metrics_vs_n.png` or `tbcm_motor_map_F0.png`. Callum owned "finalize all results and figures by Aug 20" from the 2026-08-06 advisor meeting.
**Decision:**
- **One protocol for all seven configurations.** N ∈ {10,20,50,100,200,500}, 5 seeds, disjoint held-out pool of up to 1000 rows, R² and range-normalised RMSE — matching what Draft 4's Methods section already claimed. Superseded `BM_FiveMethod.py` (3 seeds, 500-row pool, no phonation filter).
- **BM target filtered to `ACFL > 30`** → 4,647 of 5,000 rows, matching the count Draft 4 states. Non-phonating rows have spurious F0.
- **The optimized transfer procedure is identical across PR, RF and NN**: source model + target-only + residual-correction + feature-augmentation sub-models, blended by non-negative weights. Previously only RF used the blend; PR was residual-only and NN was a fine-tune, so the draft's claim that "the same procedure" was applied to all three was not true of the code.
- **Blend weights are fit on out-of-fold (K = min(5,N)) sub-model predictions**, not in-sample.
- **Recalibrated `get_model_params`**: 300 trees throughout, capacity controlled by depth (3/6/10/unlimited at N < 50 / 250 / 1000 / above). The previous schedule set `min_samples_leaf = max(10, n//10)`, which forces every tree to a stump at N ≤ 100 and made the RF baseline a mean-predictor in the low-N regime the paper is about.
**Why:** The paper's central claim is "transfer barely beats the target-alone baseline on BM". In-sample blend weights let the target-only sub-model win weight by memorising the N training rows, which would have produced that result as an artifact of the blending rather than as a finding. Out-of-fold stacking gives the source-dependent sub-models a fair chance to earn weight — and they still don't help on BM (measured mean weight on the target-only sub-model at N=50 is 0.45, i.e. not degenerate). The result survives the stronger test.
**Where it shows up:** `paper_methods.py`, `Beam_Membrane/BM_PaperStats.py`, `TBCM/TBCM_PaperStats.py`, `TBCM/TBCM_MotorMaps.py`, `Beam_Membrane/BM_PaperFigures.py`, `paper_out/`.

## 2026-08-20 — Draft 4's "TabPFN wins on both targets" does not survive the rerun

**Context:** Draft 4 asserts TabPFN "is the strongest method at low N on both the aligned (TBCM) and the higher-fidelity (BM) target".
**Decision:** Report the measured result instead: TabPFN is best on BM from N=50 upward and on TBCM from N=100 upward. On TBCM at N=10–50 the optimized transfer methods beat it (F0 R² 0.93 vs 0.78 at N=10). Draft 5 states this and reframes the takeaway as *insensitivity to the fidelity gap* rather than uniform superiority.
**Why:** The exception is not a problem for the paper — it is the alignment principle operating. Transfer wins exactly where a well-aligned source exists and target data is scarcest, and loses that advantage entirely on BM. Claiming uniform superiority would have been contradicted by our own TBCM numbers, which Jesús and Emiro can now reproduce from the repo.
**Where it shows up:** `paper_out/FF_Draft_5.tex` abstract, Sec. 3.1, Sec. 3.2, Conclusion; `paper_out/sanity_check.py` check [5] reports every cell where TabPFN is not best.

## 2026-08-20 — `BM_TabPFN_SingleVsJoint.py` is degenerate; the "~0.05 gap" claim has no experiment

**Context:** The Draft-2 figure map lists a Sec 3.3 claim that a joint multi-output TabPFN head trails single-output heads by ~0.05 R².
**Decision:** Treat the claim as unsupported and leave it out of Draft 5 (Draft 4 had already dropped it).
**Why:** `_joint_head` standardises Y with a single `StandardScaler` over the whole output block, but `StandardScaler` standardises each column independently, so the "joint" head is mathematically identical to the "single" head. `results/bm_tabpfn_single_vs_joint.json` confirms it: the two are bit-for-bit equal and the recorded `mean_gap_single_minus_joint` is exactly 0.0. A real joint-head test needs a genuinely multi-output estimator, which `TabPFNRegressor` is not.
**Where it shows up:** `Beam_Membrane/BM_TabPFN_SingleVsJoint.py`, `Beam_Membrane/results/bm_tabpfn_single_vs_joint.json`.
