# Email draft — Vocal Fold ML follow-up (2026-05-12 meeting + 2026-05-13/14 follow-ups)

**To:** Sean Peterson; Jesus Parra <jesus.parrap@sansano.usm.cl>; Emiro Ibarra <emiro.ibarra@sansano.usm.cl>; Matias Zanartu <matias.zanartu@usm.cl>
**Cc:** Callum Camazzola <callumcamazzola@gmail.com>
**Subject:** Vocal Fold ML — non-transfer alternates beat transfer in 2 of 3 domains, tie in the third (cross-domain validation + heatmaps)

Hi all,

Thanks for the call on the 12th. Quick recap of what landed in the days after, plus the follow-ups Jesus and Emiro asked for.

## Headline

GP and TabPFN re-run on all three target domains. The result is more nuanced than the original "alternates beat transfer at small N" finding and I think it makes for a stronger paper.

**Alternates dominate when source-target alignment is poor or moderate. Transfer ties the alternates only when source and target are in the same physics family (BCM→TBCM).**

Three-domain summary (avg R² of F0+SPL, mean over 10 bootstrap replicates per N, GP/TabPFN trained on N target samples; transfer methods retrained on N target samples on top of the full source pretraining):

| N | BM (BCM→BM) |  | Female (Male→Female BCM) |  | TBCM (BCM→TBCM) |  |
|---|---|---|---|---|---|---|
|   | TabPFN | best transfer | TabPFN | best transfer | TabPFN | best transfer |
| 10 | 0.27 | 0.08 | 0.22 | -0.13 | 0.62 | -0.22 |
| 50 | **0.66** | 0.19 | 0.63 | **0.40** | 0.88 | 0.78 |
| 100 | 0.67 | 0.28 | 0.79 | 0.64 | 0.93 | 0.86 |
| 500 | 0.91 | 0.74 | 0.97 | 0.84 | **0.97** | **0.97** |

- **BM (Beam-Membrane FEM)** — alternates dominate at every N. TabPFN at N=50 = 0.66 vs best BCM→BM transfer = 0.19 (gap +0.47). Even at N=500 the gap is still +0.17. _(Same headline as the 2026-05-12 call.)_
- **Female BCM (Male→Female transfer)** — alternates dominate at every N. TabPFN starts at 0.03 vs RF transfer at -0.15 at N=5, gap narrows but never closes within tested range. At N=500 TabPFN = 0.97 vs RF transfer = 0.84.
- **TBCM** — alternates lead at small N; transfer catches and ties at N=500 (both 0.972). At N=100 the gap is +0.07 in TabPFN's favor; at N=50 it's +0.10.

## Refined thesis (different from what I said on the call)

Source-target alignment quality is the load-bearing variable for how fast transfer catches up:

| Alignment | Pair | What happens |
|---|---|---|
| Poor (Ps range mismatch, different physics) | BCM→BM | Alternates dominate at every N; transfer catches slowly and never closes within N≤500. |
| Medium (same physics, demographic shift) | Male→Female BCM | Alternates dominate at every N; transfer catches faster than BM but still doesn't close within N≤500. |
| Tight (same physics family, geometry shift only) | BCM→TBCM | Alternates lead at small N; transfer catches and ties at N=500. |

This is a cleaner story than "alternates always win at small N." It identifies _when_ the BCM-style source pretraining is worth the investment: only when the source-target pair is in the same physics family. Otherwise a strong generic prior (TabPFN) plus N target samples wins.

## Methodology note (important — fixed an artifact yesterday)

The Female BCM numbers above reflect a methodology fix landed 2026-05-14 (commit `c0873a4`). The earlier version of the Female RF transfer comparator was evaluating a single fully-trained transfer model on N random test rows — which included rows the transfer model had been trained on. That gave an artificially flat ~0.72 R² across all N and made it look like transfer held an edge at the smallest N. The corrected approach (`Female_SmallData.py`) retrains the full transfer ensemble on N target samples per replicate, evaluated on a 500-row held-out test pool. Mirrors what we already do for BM and TBCM.

The corrected numbers tell the cleaner story above. Flag if you want the methodology details — happy to walk through.

## What I added since the call (Jesus + Emiro's requests)

1. **Cross-domain replication** — BM/Female/TBCM all done. Single figure `cross_domain_alternates.png` (3 panels side-by-side, attached).
2. **N=20 panel** added to the BM bootstrap-robustness figure (`bm_showcase_bootstrap.png`). At N=20 GP/TabPFN medians ≈ 0.38 vs best-transfer at 0.05 — alternates lead is established by N=20 with comfortable margin.
3. **Muscle-activation × F0 heatmaps** at fixed `PS = median` for each domain. Scatter of training points (true F0, color-coded) overlays each method's predicted F0 surface. Methods that replicate the nonlinear `(a_CT, a_TA) → F0` trend land scatter colors on the surface; methods that distort the shape produce visible mismatch. Attached: `heatmap_BM_F0.png`, `heatmap_FemaleBCM_F0.png`, `heatmap_TBCM_F0.png`.
4. **Per-domain showcase sets** for the BM/TBCM/Female cases — line chart, sample-budget bar chart, bootstrap boxplots, and a per-N head-to-head table for each. Useful for a deeper look beyond the 3-panel summary. Attached for BM (4 figs); happy to send the TBCM/Female sets if useful.

## Implementation status

- Everything committed on `feature/fem`. All three domains' alternates code lives in per-domain folders matching Callum's convention:
  - BM: `Beam_Membrane/{BM_GP,BM_TabPFN,BM_SmallData,BM_Showcase,BM_CrossDomain,BM_Heatmaps}.py`
  - TBCM: `TBCM/{TBCM_GP,TBCM_TabPFN,TBCM_SmallData,TBCM_Showcase}.py`
  - Female: `VocalFoldRegression/BCM Model/Alternates/{Female_GP,Female_TabPFN,Female_SmallData,Female_Showcase}.py`
- All data files (BM 5,000 rows, TBCM 43,102 rows, Female 1,195 post-`ACFL>30` filter) loaded and producing results.

## Why alternates work where transfer didn't (BM and Female cases)

- **Domain mismatch (BM).** BCM operates over Ps [10, 2010] Pa; BM over [600, 1000]. Source-only R² on BM ≈ −2 — actively misleading. Transfer methods that trust BCM predictions hurt at small N.
- **Strong generic priors beat misaligned domain priors at small N.** TabPFN's pretrained prior (millions of synthetic regression problems) plus 50 target examples beats transfer methods that lean on a misaligned source.
- **Demographic shift isn't enough alignment (Female).** Same physics, same input/output schema — but enough distributional shift that even retrained transfer can't beat TabPFN.

## References (attached or linked)

- **TabPFN Nature paper (2025):** *Accurate predictions on small data with a tabular foundation model* (the paper Sean shared).
- **Gaussian Process textbook:** Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press 2006 — open-access PDF at http://www.gaussianprocess.org/gpml/
- **sklearn GP docs:** https://scikit-learn.org/stable/modules/gaussian_process.html
- **TabPFN client (cloud-API we use):** https://github.com/PriorLabs/tabpfn-client

## Attached

- `cross_domain_alternates.png` — 3-panel R² vs N across BM, TBCM, Female BCM
- `bm_showcase_headline.png` — BM line chart with annotated +0.47 R² gap at N=50
- `bm_showcase_sim_budget.png` — BM simulations needed to hit R²=0.5 and 0.7 thresholds
- `bm_showcase_bootstrap.png` — 6-panel boxplots including N=20
- `heatmap_BM_F0.png`, `heatmap_FemaleBCM_F0.png`, `heatmap_TBCM_F0.png`
- TabPFN Nature paper (Sean's forward, re-attached for convenience)

## Open questions

1. **Publication framing.** On the call, Sean and Matias both leaned toward bundling transfer + alternates as one paper with transfer as the baseline. The corrected Female result reinforces that — alternates win in 2 of 3 domains and tie in the third, with a clean explanatory variable (alignment quality). Matias, anything you'd push back on with that framing?
2. **Higher-fidelity validation.** Anyone have a higher-fidelity FEM than BM, or access to clinical F0/SPL data, we could run as a fourth domain? Would test whether the alignment-quality thesis holds when we move from synthetic to real.

Happy to discuss at the next sync or async — replies welcome.

Ben
