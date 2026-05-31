# PLAN — JASA motor-control maps for TBCM under sparse data

**Owner:** ben (Callum can pick up BM/Female extension + GT-source revisit)
**Created:** 2026-05-31
**Driver:** 2026-05-28 "Vocal Fold ML Update" sync — pivot from headline R² to *replicating JASA motor-control maps* under sparse data.
**Deliverable:** heatmaps for TBCM comparing TabPFN vs TransferRF vs ground truth at N = 10 / 100 / 1000, F0 + SPL, before next Thursday.
**Scope this pass:** make the figures. No docs/board/email/DECISIONS updates, no result caching.

---

## 1. Goal

Replicate the JASA motor-control maps — an `(a_CT × a_TA)` activation plane at fixed `PS`, colored by an acoustic output with black iso-frequency contour lines — for the **TBCM** model, but produced by ML regressors fit on **sparse** data (N = 10 / 100 / 1000). Compare **TabPFN vs TransferRF vs ground truth**, for **F0** and **SPL**, plus signed-error maps. The novelty vs the JASA paper: doing it from a handful of samples (the clinical-data challenge).

---

## 2. The map template already exists — port `Regressors-BCM.ipynb` cells 32–40

The working motor-control-map generator is **cells 32–40** of `VocalFoldRegression/BCM Model/Regressors-BCM.ipynb` (NOT cell 44, which is unrelated fo-glide gesture tracking).

- **Cell 32:** `df2 = df2[df2['Ps']==1010]` — PS fixed by filtering the dataset to a grid value.
- **Cell 35:** grid from `df2['a_CT'].unique()` × `df2['a_TA'].unique()` via `meshgrid`; ground truth `Z1_l/Z2_l` = dataset F0/SPL reshaped onto that grid.
- **Cells 36–38:** per-model maps, JASA style — `imshow(interpolation='bilinear', origin='lower', extent=...)` + black `contour(levels=...)` with `clabel`, laid out 2×3 as `[data | predicted | abs-error]` for `[F0 ; SPL]`. F0 levels `[100,200,300,400,500,600]`, SPL `arange(65,90,5)`.
- **Cells 39–40:** combined `[dataset | RF | NN | PR]` figure, saved `Figs/heatmap_BCM.svg`.

**Carry over:** imshow+contour+clabel look, `[ground-truth | method | error]` layout. **Adapt** F0 contour levels to TBCM's 80–400 Hz range, e.g. `[100,150,200,250,300,350]`.

> BCM works by reshaping the dataset because its lookup table is a full `a_CT×a_TA` grid at each PS. **TBCM is randomly sampled in PS** (43,102 unique PS; a_CT/a_TA gridded at 40 levels) — so dataset-reshape GT is unavailable and we use a dense surrogate (§5).

---

## 3. Map machinery already exists — generalize `Beam_Membrane/BM_Heatmaps.py`

`BM_Heatmaps.py` already renders a side-by-side `[GP | TabPFN | TransRF]` F0 heatmap at fixed N on a 50×50 `a_CT×a_TA` grid at PS=median, with per-method scalers, training-point scatter, shared color scale, and the `Agg` backend. Reuse directly:

| Need | In `BM_Heatmaps.py` | Action |
|---|---|---|
| Headless backend | `matplotlib.use('Agg')` `:27` | keep |
| Grid build | `build_grid(df, ps_value)` `:63-69` | keep (50×50) |
| TabPFN one-target | `fit_tabpfn_one_target` `:53-60` + per-output scaler inverse `:106-124` | keep |
| TransRF on grid | `fit_transrf` `:72-97` (BCM source RF + residual RF) | **extend to SPL too**; drive residual RF via `get_model_params(N)` from `Beam_Membrane/BM_TransferRF.py:51-70` (no hardcoded trees) |
| Shared vmin/vmax + scatter | `:146-167` | keep |

---

## 4. New file: `TBCM/TBCM_MotorMaps.py` → `TBCM/figs/`

1. **Load TBCM** (`TBCM_TabPFN.load_tbcm` pattern): `read_csv('dataset_TBCM.csv', index_col=0)`, `Ps→PS`, drop `PL`, schema `[a_CT,a_TA,PS]→[F0,SPL]`.
2. **Ground truth:** `MultiOutputRegressor(RandomForestRegressor(n_estimators=200))` on all 43k rows (scaled); predict F0+SPL on the 50×50 grid at `PS=median (≈1418.8)`.
3. **Per N ∈ {10,100,1000}:** `df.sample(n=N, random_state=42)` drawn **once, shared** by both methods; per-domain per-output scalers on the subset only; produce TabPFN and TransferRF F0/SPL grid surfaces (inverse-transformed to Hz/dB). Source for TransferRF = `TBCM/dataset_BCM.csv`.
4. **Render** (imshow + black iso-contours + clabel):
   - `tbcm_motor_map_F0.png` — rows N{10,100,1000} × cols {Ground truth, TabPFN, TransferRF}, viridis, shared F0 scale, iso-F0 contours, N points scattered.
   - `tbcm_motor_map_SPL.png` — same, magma.
   - `tbcm_motor_map_F0_error.png` / `_SPL_error.png` — rows N × cols {TabPFN, TransferRF}, **signed** error (pred−GT), diverging `RdBu_r` centered 0, per-panel RMSE, GT iso-contours overlaid. **Surfaces Emiro's dynamic-range observation** (where in 80–400 Hz TabPFN's edge over TransferRF shrinks).

---

## 5. Decisions resolved (Ben, 2026-05-31)

| Decision | Choice |
|---|---|
| GT source | Dense RF surrogate over 43k rows (TBCM is PS-random → no dataset-reshape) |
| PS slice | Single fixed `PS = median ≈ 1418.8` |
| Methods | Ground truth · TabPFN · TransferRF (GP omitted to keep figure focused) |

Non-blocking external: which exact JASA figure (ask Jesus — adapt contour levels/PS to match); TabPFN N=1000 cloud vs local (harness tries both).

---

## 6. Conventions honored
Per-domain/per-output scalers on the N-subset only (1) · drop `PL`, fixed schema (2) · `random_state=42`, shared N-draw across methods (3) · `get_model_params(N)`, no hardcoded 300 (4) · `index_col=0` + `Ps→PS` (5) · PNGs in `TBCM/figs/`, gitignored unless `git add -f` (6).

---

## 7. Verification
`python TBCM/TBCM_MotorMaps.py` writes 4 PNGs. Down each column, TabPFN/TransferRF maps sharpen toward the GT column as N→1000; GT F0 spans ~80–400 Hz, SPL ~56–109 dB; per-panel RMSE drops with N; TabPFN RMSE ≤ TransferRF at small N.

---

## 8. Out of scope this pass (for Callum / later)
Notebook packaging; BM/Female panels (the script generalizes — `BM_Heatmaps` already loops domains); email draft; DECISIONS/board/MILESTONES; `.npz`/JSON caching; callable-simulator GT (revisit if Sean/Callum wire the TBCM solver into Python).

---

## 9. Verified facts (2026-05-31)
- `TBCM/dataset_TBCM.csv`: 43,102 rows, cols `a_CT,a_TA,PL,Ps,F0,SPL`; a_CT/a_TA 40 levels each (0.025–1), PS random 448–2160 (median 1418.8); F0 80–383.5, SPL 55.5–109.3.
- `TBCM/dataset_BCM.csv` present (TransferRF source).
- Map template = cells 32–40 (cell 44 unrelated). Machinery base = `BM_Heatmaps.py`.
