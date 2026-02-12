"""
Validate Female Simulation Data
================================
Run this after MATLAB generates the female data to check if the results
are physically realistic before using them for transfer learning.

Checks:
1. F0 and SPL ranges (are they in the right ballpark for females?)
2. Failure rate (how many simulations produced NaN?)
3. Comparison with existing female data (if available)
4. Comparison with male data (females should have higher F0)
5. Physical relationships (does F0 increase with a_CT? does SPL increase with Ps?)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==================== CONFIGURATION ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# new MATLAB-generated female data
NEW_FEMALE_PATH = '/tmp/Female_Membrane_Beam_Model.txt'

# existing datasets for comparison (optional)
EXISTING_FEMALE_PATH = os.path.join(SCRIPT_DIR, 'Female_binary.parquet')
MALE_PATH = os.path.join(SCRIPT_DIR, 'MaleBCM.csv')

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'validation-figs')

# expected ranges from literature
FEMALE_F0_RANGE = (100, 500)   # Hz — normal female phonation
MALE_F0_RANGE = (60, 300)      # Hz — normal male phonation
SPL_RANGE = (40, 130)          # dB — normal phonation


# ==================== LOAD DATA ====================
def load_matlab_output(path):
    """Load raw MATLAB output including NaN rows"""
    df = pd.read_csv(path, delim_whitespace=True)

    # clean column names
    renamed = {}
    for col in df.columns:
        clean = col.strip()
        if 'Ps' in clean or 'PS' in clean:
            renamed[col] = 'PS'
        elif 'F0' in clean:
            renamed[col] = 'F0'
        elif 'SPL' in clean:
            renamed[col] = 'SPL'
        elif 'a_CT' in clean:
            renamed[col] = 'a_CT'
        elif 'a_TA' in clean:
            renamed[col] = 'a_TA'
        else:
            renamed[col] = clean

    df = df.rename(columns=renamed)
    return df


def main():
    print("=" * 70)
    print("FEMALE SIMULATION DATA VALIDATION")
    print("=" * 70)

    # --- load new data ---
    if not os.path.exists(NEW_FEMALE_PATH):
        print(f"\nERROR: File not found: {NEW_FEMALE_PATH}")
        print("Run the MATLAB simulation first.")
        sys.exit(1)

    df_raw = load_matlab_output(NEW_FEMALE_PATH)
    print(f"\nLoaded {len(df_raw)} total rows from MATLAB output")
    print(f"Columns: {df_raw.columns.tolist()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_passed = True

    # ==================== CHECK 1: FAILURE RATE ====================
    print("\n" + "=" * 50)
    print("CHECK 1: SIMULATION FAILURE RATE")
    print("=" * 50)

    n_total = len(df_raw)
    n_nan = df_raw[['F0', 'SPL']].isna().any(axis=1).sum()
    n_valid = n_total - n_nan
    failure_rate = n_nan / n_total * 100

    print(f"  Total simulations:  {n_total}")
    print(f"  Valid (non-NaN):    {n_valid}")
    print(f"  Failed (NaN):       {n_nan}")
    print(f"  Failure rate:       {failure_rate:.1f}%")

    if failure_rate > 80:
        print("  FAIL — >80% of simulations failed. Parameters are likely wrong.")
        all_passed = False
    elif failure_rate > 50:
        print("  WARNING — >50% failure rate. Parameters may need adjustment.")
        all_passed = False
    elif failure_rate > 20:
        print("  CAUTION — >20% failure rate. Worth investigating but may be OK.")
    else:
        print("  PASS — failure rate is reasonable.")

    # filter to valid rows
    df = df_raw[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
    df = df[(df['F0'] > 0) & (df['SPL'] > 0)].reset_index(drop=True)
    print(f"\n  Working with {len(df)} valid samples after filtering")

    if len(df) < 10:
        print("\n  FAIL — too few valid samples to analyse. Check MATLAB parameters.")
        sys.exit(1)

    # ==================== CHECK 2: F0 RANGE ====================
    print("\n" + "=" * 50)
    print("CHECK 2: F0 RANGE (Is it in the female range?)")
    print("=" * 50)

    f0_min, f0_max = df['F0'].min(), df['F0'].max()
    f0_mean, f0_median = df['F0'].mean(), df['F0'].median()

    print(f"  F0 min:     {f0_min:.1f} Hz")
    print(f"  F0 max:     {f0_max:.1f} Hz")
    print(f"  F0 mean:    {f0_mean:.1f} Hz")
    print(f"  F0 median:  {f0_median:.1f} Hz")
    print(f"  Expected female range: {FEMALE_F0_RANGE[0]}-{FEMALE_F0_RANGE[1]} Hz")

    if f0_median < 50:
        print("  FAIL — F0 is far too low for female. Something is wrong.")
        all_passed = False
    elif f0_median < FEMALE_F0_RANGE[0]:
        print("  WARNING — median F0 is below typical female range.")
        print("  This might indicate male-like parameters are still active.")
    elif f0_median > FEMALE_F0_RANGE[1]:
        print("  WARNING — median F0 is above typical female range.")
    else:
        print("  PASS — F0 is in the expected female range.")

    # ==================== CHECK 3: SPL RANGE ====================
    print("\n" + "=" * 50)
    print("CHECK 3: SPL RANGE")
    print("=" * 50)

    spl_min, spl_max = df['SPL'].min(), df['SPL'].max()
    spl_mean = df['SPL'].mean()

    print(f"  SPL min:    {spl_min:.1f} dB")
    print(f"  SPL max:    {spl_max:.1f} dB")
    print(f"  SPL mean:   {spl_mean:.1f} dB")
    print(f"  Expected range: {SPL_RANGE[0]}-{SPL_RANGE[1]} dB")

    if spl_mean < SPL_RANGE[0] or spl_mean > SPL_RANGE[1]:
        print("  WARNING — SPL mean is outside typical range.")
        all_passed = False
    else:
        print("  PASS — SPL is in a realistic range.")

    # ==================== CHECK 4: PHYSICAL RELATIONSHIPS ====================
    print("\n" + "=" * 50)
    print("CHECK 4: PHYSICAL RELATIONSHIPS")
    print("=" * 50)

    # F0 should increase with a_CT (CT stretches folds → higher pitch)
    low_ct = df[df['a_CT'] < 0.3]['F0'].median()
    high_ct = df[df['a_CT'] > 0.7]['F0'].median()

    if not np.isnan(low_ct) and not np.isnan(high_ct):
        print(f"  F0 at low a_CT (<0.3):   {low_ct:.1f} Hz")
        print(f"  F0 at high a_CT (>0.7):  {high_ct:.1f} Hz")
        if high_ct > low_ct:
            print("  PASS — F0 increases with CT activation (expected).")
        else:
            print("  WARNING — F0 does NOT increase with CT activation.")
            print("  This is unexpected and may indicate a problem.")
    else:
        print("  SKIP — not enough data at extreme CT values.")

    # SPL should increase with Ps (more pressure → louder)
    low_ps = df[df['PS'] < 500]['SPL'].median()
    high_ps = df[df['PS'] > 700]['SPL'].median()

    if not np.isnan(low_ps) and not np.isnan(high_ps):
        print(f"\n  SPL at low Ps (<500):    {low_ps:.1f} dB")
        print(f"  SPL at high Ps (>700):   {high_ps:.1f} dB")
        if high_ps > low_ps:
            print("  PASS — SPL increases with subglottal pressure (expected).")
        else:
            print("  WARNING — SPL does NOT increase with pressure.")
    else:
        print("  SKIP — not enough data at extreme Ps values.")

    # ==================== CHECK 5: COMPARE WITH EXISTING FEMALE DATA ====================
    print("\n" + "=" * 50)
    print("CHECK 5: COMPARISON WITH EXISTING FEMALE DATA")
    print("=" * 50)

    if os.path.exists(EXISTING_FEMALE_PATH):
        try:
            df_existing = pd.read_parquet(EXISTING_FEMALE_PATH)
            df_existing = df_existing[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
            df_existing = df_existing[(df_existing['F0'] > 0) & (df_existing['SPL'] > 0)]

            print(f"  Existing female samples: {len(df_existing)}")
            print(f"\n  {'Metric':<20} {'Existing':<15} {'New MATLAB':<15} {'Difference':<15}")
            print("  " + "-" * 65)

            for col in ['F0', 'SPL']:
                existing_mean = df_existing[col].mean()
                new_mean = df[col].mean()
                diff_pct = (new_mean - existing_mean) / existing_mean * 100
                print(f"  {col + ' mean':<20} {existing_mean:<15.1f} {new_mean:<15.1f} {diff_pct:>+.1f}%")

                existing_std = df_existing[col].std()
                new_std = df[col].std()
                diff_pct_std = (new_std - existing_std) / existing_std * 100
                print(f"  {col + ' std':<20} {existing_std:<15.1f} {new_std:<15.1f} {diff_pct_std:>+.1f}%")

            print("\n  NOTE: Some difference is expected — the existing data may come")
            print("  from a different simulation model (BCM vs beam+membrane).")
            print("  Large differences (>50%) warrant investigation.")

        except Exception as e:
            print(f"  Could not load existing female data: {e}")
    else:
        print("  No existing female data found for comparison. Skipping.")

    # ==================== CHECK 6: COMPARE WITH MALE DATA ====================
    print("\n" + "=" * 50)
    print("CHECK 6: COMPARISON WITH MALE DATA")
    print("=" * 50)

    if os.path.exists(MALE_PATH):
        try:
            df_male = pd.read_csv(MALE_PATH)
            df_male = df_male[['a_CT', 'a_TA', 'PS', 'F0', 'SPL']].dropna()
            df_male = df_male[(df_male['F0'] > 0) & (df_male['SPL'] > 0)]

            male_f0_mean = df_male['F0'].mean()
            female_f0_mean = df['F0'].mean()

            print(f"  Male F0 mean:    {male_f0_mean:.1f} Hz")
            print(f"  Female F0 mean:  {female_f0_mean:.1f} Hz")

            if female_f0_mean > male_f0_mean:
                ratio = female_f0_mean / male_f0_mean
                print(f"  PASS — Female F0 is {ratio:.1f}x higher than male (expected ~1.5-2x).")
                if ratio < 1.2:
                    print("  NOTE: Ratio is lower than typical. May still be OK for this model.")
                elif ratio > 3:
                    print("  WARNING: Ratio is unusually high. Worth checking parameters.")
            else:
                print("  FAIL — Female F0 is LOWER than male. This is wrong.")
                print("  The geometry changes may not be taking effect.")
                all_passed = False

        except Exception as e:
            print(f"  Could not load male data: {e}")
    else:
        print("  No male data found for comparison. Skipping.")

    # ==================== VISUALISATION ====================
    print("\n" + "=" * 50)
    print("GENERATING VALIDATION PLOTS")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Female Simulation Data Validation', fontsize=14, fontweight='bold')

    # 1. F0 distribution
    ax = axes[0, 0]
    ax.hist(df['F0'], bins=50, alpha=0.7, color='coral', label='New MATLAB')
    if os.path.exists(EXISTING_FEMALE_PATH):
        try:
            df_ex = pd.read_parquet(EXISTING_FEMALE_PATH)
            df_ex = df_ex[['F0']].dropna()
            df_ex = df_ex[df_ex['F0'] > 0]
            ax.hist(df_ex['F0'], bins=50, alpha=0.5, color='blue', label='Existing female')
        except:
            pass
    ax.set_xlabel('F0 (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('F0 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SPL distribution
    ax = axes[0, 1]
    ax.hist(df['SPL'], bins=50, alpha=0.7, color='coral', label='New MATLAB')
    if os.path.exists(EXISTING_FEMALE_PATH):
        try:
            df_ex = pd.read_parquet(EXISTING_FEMALE_PATH)
            df_ex = df_ex[['SPL']].dropna()
            df_ex = df_ex[df_ex['SPL'] > 0]
            ax.hist(df_ex['SPL'], bins=50, alpha=0.5, color='blue', label='Existing female')
        except:
            pass
    ax.set_xlabel('SPL (dB)')
    ax.set_ylabel('Count')
    ax.set_title('SPL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. F0 vs a_CT (should trend upward)
    ax = axes[0, 2]
    scatter = ax.scatter(df['a_CT'], df['F0'], c=df['a_TA'], cmap='viridis',
                         alpha=0.3, s=10)
    ax.set_xlabel('a_CT')
    ax.set_ylabel('F0 (Hz)')
    ax.set_title('F0 vs CT Activation\n(coloured by a_TA)')
    plt.colorbar(scatter, ax=ax, label='a_TA')
    ax.grid(True, alpha=0.3)

    # 4. SPL vs Ps (should trend upward)
    ax = axes[1, 0]
    ax.scatter(df['PS'], df['SPL'], alpha=0.3, s=10, color='green')
    ax.set_xlabel('Ps (Pa)')
    ax.set_ylabel('SPL (dB)')
    ax.set_title('SPL vs Subglottal Pressure')
    ax.grid(True, alpha=0.3)

    # 5. F0 vs SPL scatter
    ax = axes[1, 1]
    ax.scatter(df['F0'], df['SPL'], alpha=0.3, s=10, color='purple')
    ax.set_xlabel('F0 (Hz)')
    ax.set_ylabel('SPL (dB)')
    ax.set_title('F0 vs SPL')
    ax.grid(True, alpha=0.3)

    # 6. Input distributions
    ax = axes[1, 2]
    ax.hist(df['a_CT'], bins=30, alpha=0.5, label='a_CT', color='red')
    ax.hist(df['a_TA'], bins=30, alpha=0.5, label='a_TA', color='blue')
    ax.hist(df['PS'] / 1000, bins=30, alpha=0.5, label='Ps/1000', color='green')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Input Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'female_data_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()

    # ==================== FINAL VERDICT ====================
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: All checks passed. Data looks reasonable.")
        print("Proceed with transfer learning experiment.")
    else:
        print("OVERALL: Some checks FAILED. Review the warnings above.")
        print("Consider consulting your colleague before using this data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
