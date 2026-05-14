# Email draft — Vocal Fold ML follow-up (2026-05-12 meeting + 2026-05-13/14 follow-ups)

**To:** Sean Peterson; Jesus Parra <jesus.parrap@sansano.usm.cl>; Emiro Ibarra <emiro.ibarra@sansano.usm.cl>; Matias Zanartu <matias.zanartu@usm.cl>
**Cc:** Callum Camazzola <callumcamazzola@gmail.com>
**Subject:** Vocal Fold ML — TabPFN wins at every N in all 3 target domains; transfer only catches at TBCM N=500

Hi all,

Thanks for the call on the 12th. Quick recap of what landed in the days after plus the follow-ups Jesus and Emiro asked for. Figures attached for the full story.

## Quick primer on the two alternate methods

Both replace the transfer-learning pipeline entirely. They train directly on a small number of target-domain simulations and produce F0/SPL predictions without ever using BCM as a source.

**Gaussian Process (GP).** Classical Bayesian regression — fits a probability distribution over smooth functions that pass through your training points and interpolates new ones. Three hyperparameters (signal strength, smoothness, noise) tuned by marginal-likelihood maximization. Almost impossible to overfit. Textbook small-N method, well-understood, no neural networks involved. We use a Matérn-2.5 kernel + WhiteKernel; one independent GP per output (F0, SPL).

**TabPFN (Tabular Prior-data Fitted Network).** A transformer that was pretrained ONCE by Prior Labs on millions of synthetic tabular regression problems, then frozen. To predict on a new dataset, you give it your training rows + the test inputs in a single forward pass and it returns predictions — no per-task fitting, no gradient steps. Like a language model doing in-context learning, but for tables. Published in Nature 2025 (paper attached). We use the cloud-API version (`tabpfn-client`) so we don't depend on local GPU weights.

The reason both work where transfer learning struggled: TabPFN brings a strong generic prior built from millions of tabular problems; GP brings the smoothness assumption that vocal-fold physics actually satisfies. Neither needs the BCM source data, so neither suffers from BCM↔target mismatch (different Ps ranges, different physics families, etc.).

## Headline

GP and TabPFN re-run on all three target domains. The result is more nuanced than the original "alternates beat transfer at small N" finding from the 12th, and I think it makes for a stronger paper.

**TabPFN wins at every N tested (5 to 500) in all three domains.** The single point where transfer matches it is TBCM at N=500 — i.e. the tightest source-target alignment (BCM↔TBCM, same physics family) and the largest N we tested. Everywhere else, alternates lead.

- **BM (Beam-Membrane FEM).** TabPFN wins at every N. The N=50 gap is +0.47 R²; even at N=500 it's still +0.17. _(Same headline as the 2026-05-12 call.)_
- **Female BCM (Male→Female transfer).** TabPFN wins at every N. RF transfer trained on N target samples starts deeply negative at N=5 and climbs slowly; TabPFN already leads at N=5 and never gives up the lead. At N=500 the gap is +0.13.
- **TBCM.** TabPFN wins at every N from 5 to 300. At N=500 TabPFN and TransRF tie almost exactly (both 0.972). This is the only place where the BCM source pretraining stays useful at large N — and it's the case where source and target are in the same physics family (BCM and TBCM are both lumped-element body-cover models).

## Refined thesis (different from what I said on the call)

Source-target alignment quality is the load-bearing variable. The tighter the alignment, the faster transfer catches up to the alternates as N grows — but in our setup, only BCM↔TBCM is tight enough for transfer to ever match TabPFN within the tested N range.

- Poor alignment (BCM→BM, Ps range mismatch + different physics): transfer never catches — gap +0.17 R² persists at N=500.
- Medium alignment (Male→Female BCM, same physics, demographic shift): transfer never catches — gap +0.13 at N=500.
- Tight alignment (BCM→TBCM, same physics family, geometry shift only): transfer matches TabPFN at N=500.

This is a cleaner story than "alternates always win at small N." It identifies _when_ the BCM-style source pretraining is worth the investment: only when the source-target pair is in the same physics family does it stay competitive at large N. Otherwise a strong generic prior plus N target samples wins outright.

## Methodology note (important — fixed an artifact yesterday)

The Female BCM numbers above reflect a methodology fix landed 2026-05-14. The earlier version of the Female RF transfer comparator was evaluating a single fully-trained transfer model on N random test rows — which included rows the transfer model had been trained on. That gave an artificially flat ~0.72 R² across all N and made it look like transfer held an edge at the smallest N. The corrected approach retrains the full transfer ensemble on N target samples per replicate and evaluates on a held-out test pool — mirroring what we already do for BM and TBCM. The corrected numbers tell the cleaner story above. Happy to walk through if you want the details.

## What I added since the call (Jesus + Emiro's requests)

1. **Cross-domain replication.** BM/Female/TBCM all done — figure attached.
2. **N=20 panel** added to the BM bootstrap-robustness figure. At N=20 GP/TabPFN medians ≈ 0.38 vs best transfer at 0.05 — the alternates lead is established by N=20 with comfortable margin.
3. **Muscle-activation × F0 heatmaps** for each domain, at fixed `PS = median`. Training points (true F0, color-coded) overlay each method's predicted F0 surface. Methods that replicate the nonlinear `(a_CT, a_TA) → F0` trend land scatter colors on the surface; methods that distort the shape produce visible mismatch. Three heatmaps attached.
4. **Per-domain showcase sets** for BM, TBCM, and Female — line chart, sample-budget bar chart, bootstrap boxplots, and a per-N head-to-head table. Happy to send the TBCM/Female sets too if useful.

## Why alternates work where transfer didn't (BM and Female cases)

- **Domain mismatch (BM).** BCM operates over Ps [10, 2010] Pa; BM over [600, 1000]. Source-only R² on BM ≈ −2 — actively misleading. Transfer methods that trust BCM predictions hurt at small N.
- **Strong generic priors beat misaligned domain priors at small N.** TabPFN's pretrained prior (millions of synthetic regression problems) plus 50 target examples beats transfer methods that lean on a misaligned source.
- **Demographic shift isn't enough alignment (Female).** Same physics, same input/output schema — but enough distributional shift that even retrained transfer can't beat TabPFN.

## References

- **TabPFN Nature paper (2025):** *Accurate predictions on small data with a tabular foundation model* (the paper Sean shared, re-attached).
- **Gaussian Process textbook:** Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press 2006 — open-access PDF at http://www.gaussianprocess.org/gpml/
- **sklearn GP docs:** https://scikit-learn.org/stable/modules/gaussian_process.html
- **TabPFN client (cloud-API we use):** https://github.com/PriorLabs/tabpfn-client

## Open questions

1. **Publication framing.** On the call, Sean and Matias both leaned toward bundling transfer + alternates as one paper with transfer as the baseline. The corrected Female result reinforces that — alternates win in all three domains, with a clean explanatory variable (alignment quality) for when transfer catches up. Matias, anything you'd push back on with that framing?
2. **Higher-fidelity validation.** Anyone have a higher-fidelity FEM than BM, or access to clinical F0/SPL data, we could run as a fourth domain? Would test whether the alignment-quality thesis holds when we move from synthetic to real.

Happy to discuss at the next sync or async — replies welcome.

Ben
