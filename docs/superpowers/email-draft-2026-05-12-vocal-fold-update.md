# Email draft — Vocal Fold ML follow-up (2026-05-12 meeting + 2026-05-13/14 follow-ups)

**To:** Sean Peterson; Jesus Parra <jesus.parrap@sansano.usm.cl>; Emiro Ibarra <emiro.ibarra@sansano.usm.cl>; Matias Zanartu <matias.zanartu@usm.cl>
**Cc:** Callum Camazzola <callumcamazzola@gmail.com>
**Subject:** Vocal Fold ML — TabPFN wins at every N in all 3 target domains

Hi all,

Quick follow-up to the call on the 12th. Figures attached.

**The two alternates, briefly.** Both replace the transfer pipeline entirely and train directly on a small number of target-domain simulations. **GP** is classical Bayesian regression — fits a smooth function to your training points with three hyperparameters, near-impossible to overfit. **TabPFN** is a transformer that Prior Labs pretrained on millions of synthetic tabular problems, then froze (Nature 2025); you hand it your training rows + test inputs in one forward pass and it returns predictions, no per-task fitting. Neither uses BCM as a source, so neither suffers from BCM↔target mismatch.

**Headline.** TabPFN wins at every N tested (5 to 500) in all three domains. The only point where transfer matches it is TBCM at N=500 — the tightest source-target alignment (BCM↔TBCM, same physics family) and the largest N we tested.

- **BM (BCM→BM):** alternates dominate at every N; gap +0.47 R² at N=50, still +0.17 at N=500.
- **Female BCM (Male→Female):** alternates dominate at every N; gap +0.13 at N=500.
- **TBCM (BCM→TBCM):** alternates lead at small N; transfer ties at N=500 (both 0.972).

**Thesis.** Source-target alignment quality determines how fast transfer catches up. In our setup, only the BCM↔TBCM pair is aligned tightly enough for transfer to ever match TabPFN within tested N.

**Methodology note.** Female numbers reflect a fix landed 2026-05-14 — the previous comparator was evaluating a single fully-trained transfer model on N test rows that included its own training rows. The corrected approach retrains transfer on N target samples per replicate, mirroring what we do for BM and TBCM. Happy to walk through.

**References.** TabPFN Nature paper attached. GP textbook: Rasmussen & Williams, *Gaussian Processes for ML* (open-access at gaussianprocess.org/gpml). TabPFN client: github.com/PriorLabs/tabpfn-client.

**Open questions:**
1. Matias — does the bundled-paper framing (TL as baseline, alternates as headline, alignment quality as the explanatory variable) still feel right?
2. Anyone have a higher-fidelity FEM than BM, or clinical F0/SPL data, we could add as a fourth domain?

Happy to discuss async.

Ben
