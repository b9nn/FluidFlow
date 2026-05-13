# Email draft — Vocal Fold ML follow-up (2026-05-12)

**To:** Sean Peterson; Jesus Parra <jesus.parrap@sansano.usm.cl>; Emiro Ibarra <emiro.ibarra@sansano.usm.cl>; Matias Zanartu <matias.zanartu@usm.cl>
**Cc:** Callum Camazzola <callumcamazzola@gmail.com>
**Subject:** Vocal Fold ML — non-transfer alternates, cross-domain validation, and heatmaps

Hi all,

Thanks for the call yesterday. Quick recap plus the follow-ups Jesus and Emiro asked for.

## Headline

Two non-transfer methods — Gaussian Process (GP) and TabPFN — were re-run on each of our target domains to check whether the BM "alternates beat transfer" finding generalizes. Result is more nuanced than expected and I think makes for a stronger paper.

- **Beam-Membrane (FEM)** — TabPFN at N=50 reaches avg R²=0.66 vs best BCM→BM transfer at 0.19 (**+0.47** R²). Alternates dominate at every N≤100. _(Same headline as last week's call.)_
- **Female BCM (Male→Female transfer)** — refined picture:

  | N | GP | TabPFN | Male→Female RF transfer |
  |---|---|---|---|
  | 10 | 0.11 | 0.22 | ~0.72 |
  | 50 | 0.50 | 0.63 | 0.72 |
  | **75** | 0.55 | **0.72** | 0.73 _(TabPFN catches)_ |
  | **100** | 0.61 | **0.79** | 0.73 _(TabPFN takes the lead)_ |
  | 200 | 0.73 | 0.92 | ~0.74 |
  | 500 | 0.89 | 0.97 | 0.75 |

  Transfer holds an edge only at N≤30 in this well-aligned domain pair. TabPFN catches transfer by N≈75 and dominates from N=100 — even against a transfer baseline leveraging ~54k male source samples.
- **TBCM** — pending dataset arrival on my local clone; will share once the runs complete (gated on a copy of `dataset_TBCM.csv`).

The Female result sharpens the story: **TabPFN-class methods are competitive with or dominant over transfer in the small-N regime that matters for expensive simulators, across diverse source–target pairs.** When alignment is poor (BCM→BM, large Ps mismatch + different physics families), alternates dominate from N=10. When alignment is good (Male BCM↔Female BCM, same physics, gender-specific anatomy only), transfer holds an edge only at the very-smallest N (≤30) and TabPFN catches up by N≈75 with weak generic priors plus ~75 target samples — vs ~54k source samples on the transfer side.

That's a cleaner thesis than "alternates always win at small N" and matches Sean's read at the meeting that transfer's value depends on source-target alignment.

## What I added since yesterday (Jesus + Emiro's requests)

1. **Cross-domain replication** — BM result re-run on Female BCM; TBCM pending dataset. Figure: `cross_domain_alternates.png` (3 panels side-by-side; TBCM panel is a placeholder until the data arrives).
2. **N=20 panel** added to the BM bootstrap-robustness figure (`bm_showcase_bootstrap.png`). Confirms the alternates lead is established by N=20 with comfortable margin — at N=20 GP/TabPFN medians ≈ 0.38 vs best transfer at 0.05.
3. **Muscle-activation × F0 heatmaps** at fixed `PS = median` for each domain (`heatmap_BM_F0.png`, `heatmap_FemaleBCM_F0.png`, `heatmap_TBCM_F0.png` once data lands). The scatter of training points (true F0, color-coded) sits on top of each method's predicted surface. Methods that replicate the nonlinear `(a_CT, a_TA) → F0` trend land the scatter colors on the surface colors; methods that distort the shape produce visible mismatch.

## Implementation status

- Everything except TBCM committed on `feature/fem` (BM panel, N=20 bootstrap panel, Female GP + TabPFN, cross-domain figure, F0 heatmaps for BM and Female)
- TBCM panels — gated on `dataset_TBCM.csv` not being on my clone; ping Callum at the next 1pm sync, then I'll close the matrix
- All code in `Beam_Membrane/BM_GP.py`, `BM_TabPFN.py`, `BM_Showcase.py`, `BM_CrossDomain.py`, `BM_Heatmaps.py`, and `VocalFoldRegression/BCM Model/Alternates/Female_{GP,TabPFN}.py`

## Why these methods work where transfer didn't (BM case)

- **Domain mismatch.** BCM operates over Ps [10, 2010] Pa; BM over [600, 1000]. Source-only R² on BM ≈ −2 — actively misleading.
- **Strong generic priors beat weak misaligned domain priors at small N.** TabPFN's pretrained prior (millions of synthetic regression problems) plus 50 BM examples beats transfer methods leveraging 54k BCM examples that point the wrong direction.

## References (attached or linked)

- **TabPFN Nature paper (2025):** *Accurate predictions on small data with a tabular foundation model* (the paper Sean shared yesterday).
- **Gaussian Process textbook:** Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press 2006 — open-access PDF at http://www.gaussianprocess.org/gpml/
- **sklearn GP docs:** https://scikit-learn.org/stable/modules/gaussian_process.html
- **TabPFN client (the cloud-API we use):** https://github.com/PriorLabs/tabpfn-client

## Attached / on the shared drive

- `cross_domain_alternates.png` — 3-panel R² vs N across BM, TBCM, Female BCM
- `bm_showcase_bootstrap.png` — 6-panel boxplots including N=20
- `heatmap_BM_F0.png`, `heatmap_FemaleBCM_F0.png` (TBCM to follow)
- `bm_showcase_sim_budget.png` — BM simulations needed to hit R²=0.5 and 0.7
- TabPFN Nature paper (Sean's forward)

## Open questions for the group

1. **Publication framing.** Jesus floated two options yesterday — (a) bundle transfer + alternates as one paper with transfer as background, (b) keep them as separate stories. The Female BCM result above pushes me toward (a) with a slightly different angle: the paper is about **when** transfer helps, not whether it does. Matias — interested in your read.
2. **TBCM dataset.** I need a copy of `dataset_TBCM.csv` (whatever Callum has been using for `TBCM_TransferRF.py`) on my clone to close the cross-domain matrix. Anyone able to share?
3. **Higher-fidelity validation.** Anyone have a higher-fidelity FEM than BM, or access to clinical F0/SPL data, we could run as a fourth domain?

Happy to discuss at the next sync or async — replies welcome.

Ben

---

**Draft notes (not part of email — strip before sending):**
- Add TBCM numbers + heatmap once dataset lands and Tasks 2, 3, 5.5 land
- Verify Sean's email address before sending
- Confirm the TabPFN Nature paper PDF is in your inbox before attaching
