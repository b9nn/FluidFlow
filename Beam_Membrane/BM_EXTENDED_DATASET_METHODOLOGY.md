# BM Extended Dataset — Methodology & Provenance

**File:** `dataset_BM_extended.csv` (5,000 rows)
**Generated:** 2026-06-28 (MATLAB pipeline on Callum's machine)
**Supersedes:** `dataset_BM.csv` (same rows + params + F0/SPL; adds 3 physiological outputs)

---

## 1. Why this dataset exists

The paper was restructured to make **BCM→BM the primary case** (supervisor
feedback). For that, the Beam-Membrane (BM) target needs **physiologically /
clinically relevant outputs**, not just F0 and SPL. `dataset_BM.csv` only had
F0 and SPL, so we regenerated it with three additional output columns:

| Output | Meaning | Why it matters |
|---|---|---|
| **PC** | Peak collision (contact) pressure | Tissue-stress / vocal-hyperfunction marker; clinically central |
| **ACFL** | AC glottal flow (peak-to-peak of glottal volume flow) | Aerodynamic effort / efficiency measure |
| **CPP** | Cepstral peak prominence | Standard voice-quality / dysphonia metric |
| *(PC_loc)* | Location of peak collision pressure along the fold | Optional; included since it was free |

These mirror the clinical outputs already present on the **BCM source**
(`MaleBCM.csv` / `data_binary.parquet`, the Parra/JASA measures), so that
BCM→BM transfer compares like-for-like quantities.

## 2. The key design decision: row-aligned regeneration (not re-sampling)

The original `dataset_BM.csv` was generated with **unseeded `rand()`**, so the
exact 5,000 (a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps) points **cannot be reproduced
by re-randomizing**. If we had re-sampled, the new outputs would correspond to
*different* operating points and would not line up with any existing
transfer/TabPFN result computed on the original rows.

**Therefore we read `dataset_BM.csv` and looped over its exact parameter rows**,
re-running the deterministic model on each, and appended the new outputs. This
guarantees the extended dataset is **1:1 row-aligned** with all prior BM
results — every existing figure and R² number remains valid, just with extra
columns.

## 3. Pipeline

Model: Sean's Beam-Membrane vocal-fold FE model (April 2026 version).
Per row, the exact same pipeline that produced the original F0/SPL:

```
params row  ->  Posturing_Simulation2(ActVec)            % posturing (eps0, Theta_G)
            ->  Membrane_Beam_Parameters4(...)           % stretched-fold material/geometry
            ->  Membrane_Beam_Solver_MyImplementation2() % dynamic FE solve -> Am(t), p_contact(t)
            ->  Measures_Estimates_Ext(...)              % steady-state measures over t in [0.5, 1.0] s
```

Fixed config (unchanged from the original run): `T=1 s`, `c=350 m/s`,
`factor=4`, `dt=(0.0025/c)/factor`, steady-state window `t1=0.5, t2=1.0`,
`sr=1.2`, `Nx=20, Ny=10`, `Theta_conv=0.05`. Ps ∈ [600,1000] Pa; muscle
activations as stored per row (a_PCA fixed 0, a_IA = a_LCA).

Scripts (in `BM_Extended_Run/`, copies tracked in `Beam_Membrane/`):
- `Generate_BM_Dataset_Extended.m` — reads CSV, loops exact rows, writes output, runs sanity checks.
- `Measures_Estimates_Ext.m` — non-destructive extension of Sean's `Measures_Estimates` (adds ACFL, CPP; PC was already computed but discarded).
- `CPP_Estimate.m` — cepstral-peak-prominence helper.

## 4. How each new output is computed

- **PC** — already computed inside `Measures_Estimates` as `p_contact_max`
  (max of `p_contact` over the steady-state window); it was simply never
  returned. Promoted to an output. Units as produced by the solver's contact
  model. `PC_loc` = `p_contact_max_location`, normalized fold position [0,1].

- **ACFL** — peak-to-peak of the glottal volume flow over steady state. The
  glottal volume flow is the Bernoulli source flow that drives the WRA
  aeroacoustic solver: `Ug = sqrt(2·Ps/ρ)·(sr·Am)` with ρ=1.14 kg/m³.
  `ACFL = max(Ug_ss) − min(Ug_ss)`.
  **Unit conversion:** `Am` is in m² → `Ug` in m³/s; the BCM source reports AC
  flow in **cm³/s (mL/s)**, so ACFL is multiplied by `1e6` (`ACFL_UNIT`) to
  match. Without this, ACFL came out ~10⁶ too small.

- **CPP** — from the radiated pressure `pout` (already built for SPL via
  `WRA_Solver`). Power cepstrum `real(ifft(log(|fft(pout_ss)|²)))`, peak in the
  voiced quefrency band (F0 ∈ [60,500] Hz → quefrency [1/500, 1/60] s), minus a
  linear-regression baseline across that band, on a dB scale (Hillenbrand-style).

## 5. Validation (run on all 5,000 rows)

- **Row alignment:** params (a_CT, a_TA, Ps) reproduce with **max|diff| = 0** —
  exact 1:1 alignment to `dataset_BM.csv`.
- **Pipeline reproduces F0/SPL:** max |ΔF0| ≈ 2.0 Hz (one-spectral-bin
  quantization on a few borderline rows; FFT frequency resolution), max |ΔSPL|
  ≈ 0.43 dB. Confirms the regenerated pipeline matches the original.
- **Physicality:** PC ≥ 0 and ACFL ≥ 0 on all rows.
- **Ranges vs BCM source** (`ACFL∈[30,1190]`, `PC∈[0,4000]`, `CPP∈[−12,46]`):

  | Output | BM (this dataset) | BCM source | Note |
  |---|---|---|---|
  | F0  | [116, 426], med 150 | — | matches original |
  | SPL | [−54, 129], med 116 | — | matches original; negative = non-phonating |
  | PC  | [0, 2433], med 863 | [0, 4000] | in range |
  | CPP | [3.8, 52.1], med 21.4 | [−12, 46] med 24 | scale matches well |
  | ACFL| [0, 2668], med 1153 | [30, 1190] med 305 | **runs ~2–4× higher (see caveats)** |

## 6. Consistency caveats (read before transfer use)

For BCM→BM transfer to be meaningful, source and target outputs must be the
**same physical quantity in the same units**. Status:

- **PC, CPP** — units and scale are consistent with the BCM source. CPP median
  (21.4) is very close to BCM's (24).
- **ACFL** — now in correct units (cm³/s) but its phonating range runs **~2–4×
  higher than BCM**. This is *not* a unit error; it is one of:
  (a) a genuine model difference (BM is a different solver), which transfer is
      designed to absorb as domain shift; or
  (b) a definitional nuance — here ACFL is peak-to-peak of the **Bernoulli
      source flow including the separation ratio `sr=1.2`**; if the BCM/Parra
      ACFL uses the tract-loaded glottal flow or omits `sr`, that explains part
      of the gap. **To be confirmed against Parra's exact ACFL code** before
      treating ACFL as definitionally identical across domains.
- **CPP algorithm** — implemented as a standard Hillenbrand power cepstrum; the
  scale matches BCM, but the exact algorithm should still be confirmed against
  the Parra/JASA CPP routine for full rigor.

## 7. Non-phonating rows

~330 rows have SPL < 60 dB (down to −54 dB) and ACFL ≈ 0 — these are
non/weakly-phonating operating points (no sustained oscillation). They are
**kept** to preserve row alignment, but downstream analysis may filter them:
**4,647 / 5,000 rows pass the BCM-style `ACFL > 30` filter**. F0 on
non-phonating rows is a spurious spectral-peak artifact and should not be
trusted.

## 8. Columns

`a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL, ACFL, PC, CPP, PC_loc`

(`PC_loc` is optional — drop it for an exact F0/SPL/ACFL/PC/CPP deliverable.)

## 9. Reproduction

Run folder: `~/Downloads/BM_Extended_Run/` (self-contained: Sean's April-2026
solver code + `PhonationModelsCode2/` + the 3 scripts above + `dataset_BM.csv`).
In MATLAB (Signal Processing Toolbox required for `findpeaks`/`rms`):

```matlab
cd '~/Downloads/BM_Extended_Run'
Generate_BM_Dataset_Extended    % MAX_ROWS = Inf for the full 5000-row run
```

Output: `dataset_BM_extended.csv` (saved incrementally every 50 rows;
~hours for the full run; no auto-resume).
