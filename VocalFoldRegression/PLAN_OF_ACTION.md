# Beam+Membrane Transfer Learning — Plan of Action

```
====================================================================
PHASE 1: DATA GENERATION (MATLAB)
====================================================================

  [1.1] Open Randomly_Generating_Data_Membrane_Beam_Model.m
         |
         v
  [1.2] Change N_s = 2  -->  N_s = 700
        (overshoot to ~700 because some samples will be NaN)
         |
         v
  [1.3] Run in MATLAB
        - Each sample simulates 1 second of vocal fold dynamics
        - Expect 40 min to 4 hrs depending on hardware
         |
         v
  [1.4] Output: Data_Membrane_Beam_Model.txt
        - 8 columns: a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL
        - Some rows will have NaN (physically invalid configs)
         |
         v
  [1.5] Sanity check the output
        - How many valid (non-NaN) rows? Target: ~500+
        - If too few valid rows (<300), increase N_s and re-run
        - Eyeball F0 and SPL ranges — do they look physically reasonable?


====================================================================
PHASE 2: DATA EXPLORATION & VALIDATION
====================================================================

        ** Don't skip this — it's between your step 1 and 2 **

  [2.1] Run load_bm_data.py to parse the .txt file
        - Confirms column names parse correctly
        - Reports NaN attrition rate
        - Prints feature/target ranges
         |
         v
  [2.2] Quick exploratory analysis
        - Distribution plots: a_CT, a_TA, PS, F0, SPL
        - Are there dead zones? (regions where everything is NaN)
        - Scatter: F0 vs a_CT, F0 vs a_TA, F0 vs PS (same for SPL)
        - Compare distributions to male BCM data — how different?
         |
         v
  [2.3] Check for data quality issues
        - Any extreme outliers in F0 or SPL?


====================================================================
PHASE 3: BASELINE MODELS (B+M data only, no transfer)
====================================================================

  [3.1] Random Forest baseline
        - Train RF on B+M data alone (adaptive hyperparams for small data)
        - Record: R² and MAE for F0 and SPL
         |
         v
  [3.2] Neural Network baseline
        - Train NN from scratch on B+M data (smaller architecture)
        - Record: R² and MAE for F0 and SPL
         |
         v
  [3.3] Polynomial Regression baseline
        - Train PR on B+M data (degree 4-5, Ridge regularization)
        - Record: R² and MAE for F0 and SPL
         |
         v
  [3.4] Establish the floor
        - These baseline numbers are what transfer learning needs to BEAT
        - If baselines are already very good (R² > 0.95), transfer may
          not add much — that's fine, it's still worth testing
        - If baselines are poor (R² < 0.5), transfer learning has a
          bigger opportunity to help


====================================================================
PHASE 4: TRANSFER LEARNING (Male BCM --> Beam+Membrane)
====================================================================

  [4.1] Random Forest transfer (BeamMembraneRFTransfer.py)
        - TransRF: target-only + residual correction + feature augmentation
        - Learns optimal weights via K-fold CV
        - Compare: source-only, target-only, simple ensemble, TransRF
         |
         v
  [4.2] Neural Network transfer (BeamMembraneNNTransfer.py)
        - Partial layer freezing from pretrained male model
        - Tests freeze configs: 2, 4, 5, 6 frozen layers
        - Picks best by average R²
         |
         v
  [4.3] Polynomial Regression transfer (BeamMembranePRTransfer.py)
        - B+M-specific PR (reduced degree + Ridge) + male PR ensemble
        - Grid search over degree and alpha
         |
         v
  [4.4] Cross-regressor comparison (BMTransferComparison.py)
        - Side-by-side: RF vs NN vs PR transfer on same test set
        - Bootstrap evaluation for statistical robustness
        - Generates comparison CSV + visualization


====================================================================
PHASE 5: ANALYSIS & REPORTING
====================================================================

  [5.1] Key question: Did transfer learning actually help?
        - Compare Phase 3 baselines vs Phase 4 transfer results
        - If yes: by how much? Which regressor benefited most?
        - If no: the domain gap (BCM vs B+M) may be too large
         |
         v
  [5.2] Investigate the domain gap
        - How do male BCM predictions perform on B+M data? (source-only)
        - Where does the male model fail? (scatter plots, residuals)
        - Physical interpretation: what's different about B+M physics?
         |
         v
  [5.3] Document findings
        - Update PROJECT_CONTEXT.md with results
        - Figures for paper/presentation


====================================================================
```

## Summary

You had the right 3 phases. The one thing worth adding is **Phase 2 (data exploration)** between generation and baselines. It's a quick sanity check that catches problems early — like if the MATLAB script produced mostly NaN, or if F0 values are unrealistic, or if the parameter space has dead zones. Better to catch that before training models on bad data.

Everything else maps to your intuition:
- **Step 1** = Phase 1 (generate data)
- **Step 2** = Phase 3 (baselines on B+M only)
- **Step 3** = Phase 4 (transfer learning from male BCM)
