"""
Extract waveform features from TBCM simulation .mat files.

For each simulation, extracts features from:
  - Pout (output acoustic pressure): spectral shape, harmonic structure
  - Ut (glottal flow): flow shape, open/close dynamics
  - a_g (glottal area): open quotient, amplitude, regularity
  - Pcol (collision pressure): contact patterns
  - X (tissue displacement): displacement amplitude, asymmetry

Matches each sim to the CSV using (a_CT, a_TA, PL) and produces an
enriched dataset with the original columns plus waveform features.
"""

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import os
import glob
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
MAT_DIR = '/tmp/tbcm_full/Simu_TBCM/Signal'


def extract_cycle_features(waveform, min_distance=5):
    """Extract per-cycle features from a periodic waveform (steady-state portion)."""
    # Use last 60% to skip transient
    n = len(waveform)
    steady = waveform[int(n * 0.4):]

    if len(steady) < 30:
        return {}

    # Find peaks and troughs
    peaks, _ = signal.find_peaks(steady, distance=min_distance)
    troughs, _ = signal.find_peaks(-steady, distance=min_distance)

    if len(peaks) < 3 or len(troughs) < 2:
        return {}

    # Peak-to-peak amplitudes
    peak_vals = steady[peaks]
    trough_vals = steady[troughs]
    amplitudes = peak_vals[:-1] - trough_vals[:len(peak_vals)-1] if len(trough_vals) >= len(peak_vals)-1 else np.array([])

    # Period regularity (jitter-like)
    periods = np.diff(peaks)
    mean_period = np.mean(periods)

    features = {
        'mean_period_samples': mean_period,
        'n_cycles': len(peaks),
    }

    if len(periods) > 1:
        # Jitter: cycle-to-cycle period variation
        features['jitter_abs'] = np.mean(np.abs(np.diff(periods)))
        features['jitter_rel'] = features['jitter_abs'] / mean_period if mean_period > 0 else 0

    if len(amplitudes) > 1:
        features['mean_amplitude'] = np.mean(amplitudes)
        # Shimmer: cycle-to-cycle amplitude variation
        features['shimmer_abs'] = np.mean(np.abs(np.diff(amplitudes)))
        features['shimmer_rel'] = features['shimmer_abs'] / features['mean_amplitude'] if features['mean_amplitude'] > 0 else 0

    # Waveform shape
    features['rms'] = np.sqrt(np.mean(steady ** 2))
    features['skewness'] = skew(steady)
    features['kurtosis'] = kurtosis(steady)
    features['crest_factor'] = np.max(np.abs(steady)) / features['rms'] if features['rms'] > 0 else 0

    return features


def extract_glottal_features(a_g):
    """Extract glottal area features: open quotient, speed quotient."""
    n = len(a_g)
    steady = a_g[int(n * 0.4):]

    if len(steady) < 30:
        return {}

    features = {}

    # Open quotient: fraction of cycle where glottis is open
    # Glottis is "open" when a_g > some threshold
    min_area = np.min(steady)
    max_area = np.max(steady)
    if max_area - min_area < 1e-6:
        return {}

    threshold = min_area + 0.1 * (max_area - min_area)
    open_fraction = np.mean(steady > threshold)
    features['open_quotient'] = open_fraction

    # Find individual cycles via peaks
    peaks, _ = signal.find_peaks(steady, distance=5)
    if len(peaks) < 3:
        return features

    # Speed quotient: ratio of opening time to closing time per cycle
    speed_quotients = []
    for i in range(len(peaks) - 1):
        cycle = steady[peaks[i]:peaks[i+1]]
        if len(cycle) < 4:
            continue
        min_idx = np.argmin(cycle)
        opening_time = min_idx  # samples from trough to peak of next
        closing_time = len(cycle) - min_idx
        if closing_time > 0:
            speed_quotients.append(opening_time / closing_time)

    if speed_quotients:
        features['speed_quotient'] = np.mean(speed_quotients)
        features['speed_quotient_std'] = np.std(speed_quotients)

    # Glottal area amplitude variation
    features['ag_amplitude'] = max_area - min_area
    features['ag_mean'] = np.mean(steady)
    features['ag_std'] = np.std(steady)

    return features


def extract_spectral_features(waveform):
    """Extract spectral shape features (harmonic ratios, spectral tilt)."""
    n = len(waveform)
    steady = waveform[int(n * 0.4):]

    if len(steady) < 64:
        return {}

    # FFT
    windowed = steady * np.hanning(len(steady))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum = spectrum / (np.max(spectrum) + 1e-10)  # normalize

    if len(spectrum) < 10:
        return {}

    # Find fundamental peak
    peaks, properties = signal.find_peaks(spectrum, height=0.1, distance=3)
    if len(peaks) < 1:
        return {}

    f0_bin = peaks[0]
    features = {}

    # Harmonic-to-noise ratio approximation
    # Energy at harmonics vs total
    harmonic_energy = 0
    total_energy = np.sum(spectrum ** 2)
    n_harmonics = 0
    for h in range(1, 8):
        h_bin = f0_bin * h
        if h_bin >= len(spectrum) - 2:
            break
        # Sum energy in small window around harmonic
        lo = max(0, h_bin - 1)
        hi = min(len(spectrum), h_bin + 2)
        harmonic_energy += np.sum(spectrum[lo:hi] ** 2)
        n_harmonics += 1

    if total_energy > 0 and n_harmonics > 0:
        features['hnr_approx'] = 10 * np.log10(harmonic_energy / (total_energy - harmonic_energy + 1e-10))

    # Harmonic ratios (H1-H2, H1-H4)
    harmonic_amps = []
    for h in range(1, 6):
        h_bin = f0_bin * h
        if h_bin < len(spectrum) - 1:
            harmonic_amps.append(spectrum[h_bin])
        else:
            harmonic_amps.append(0)

    if harmonic_amps[0] > 0:
        if harmonic_amps[1] > 0:
            features['h1_h2_ratio'] = harmonic_amps[0] / harmonic_amps[1]
        if harmonic_amps[3] > 0:
            features['h1_h4_ratio'] = harmonic_amps[0] / harmonic_amps[3]

    # Spectral tilt (slope of log spectrum)
    log_spec = np.log10(spectrum[1:] + 1e-10)
    freqs = np.arange(1, len(log_spec) + 1)
    if len(log_spec) > 2:
        slope, _ = np.polyfit(np.log10(freqs), log_spec, 1)
        features['spectral_tilt'] = slope

    # Spectral centroid (normalized)
    freq_bins = np.arange(len(spectrum))
    total = np.sum(spectrum)
    if total > 0:
        features['spectral_centroid'] = np.sum(freq_bins * spectrum) / total / len(spectrum)

    return features


def extract_collision_features(pcol):
    """Extract collision pressure features."""
    n = len(pcol)
    steady = pcol[int(n * 0.4):]

    features = {}
    features['collision_max'] = np.max(steady)
    features['collision_mean'] = np.mean(steady[steady > 0]) if np.any(steady > 0) else 0
    features['collision_fraction'] = np.mean(steady > 0)  # fraction of time in collision

    return features


def extract_tissue_features(X):
    """Extract tissue displacement features from X (6 DOF x time)."""
    n = X.shape[1]
    steady = X[:, int(n * 0.4):]

    features = {}
    # Amplitude of each DOF
    for dof in range(min(6, X.shape[0])):
        features[f'tissue_amp_dof{dof}'] = np.max(steady[dof]) - np.min(steady[dof])

    # Phase difference between DOFs (mucosal wave indicator)
    if X.shape[0] >= 2:
        # Cross-correlation between first two DOFs
        d0 = steady[0] - steady[0].mean()
        d1 = steady[1] - steady[1].mean()
        if np.std(d0) > 0 and np.std(d1) > 0:
            corr = np.correlate(d0, d1, mode='full')
            corr = corr / (np.std(d0) * np.std(d1) * len(d0))
            lag = np.argmax(corr) - len(d0) + 1
            features['tissue_phase_lag_01'] = lag

    return features


def process_one_sim(data, sim_idx):
    """Extract all features for one simulation."""
    features = {}

    # Acoustic pressure features
    pout = data['Pout'][sim_idx]
    pout_feats = extract_cycle_features(pout)
    for k, v in pout_feats.items():
        features[f'pout_{k}'] = v

    pout_spec = extract_spectral_features(pout)
    for k, v in pout_spec.items():
        features[f'pout_{k}'] = v

    # Glottal flow features
    ut = data['Ut'][sim_idx]
    ut_feats = extract_cycle_features(ut)
    for k, v in ut_feats.items():
        features[f'flow_{k}'] = v

    # Glottal area features
    ag = data['a_g'][sim_idx]
    ag_feats = extract_glottal_features(ag)
    for k, v in ag_feats.items():
        features[f'ag_{k}'] = v

    ag_cycle = extract_cycle_features(ag)
    for k, v in ag_cycle.items():
        features[f'ag_{k}'] = v

    # Collision features
    pcol = data['Pcol'][sim_idx]
    col_feats = extract_collision_features(pcol)
    for k, v in col_feats.items():
        features[f'col_{k}'] = v

    # Tissue displacement features
    X = data['X'][sim_idx]  # (6, 4411)
    tissue_feats = extract_tissue_features(X)
    for k, v in tissue_feats.items():
        features[k] = v

    return features


def main():
    print("=" * 70)
    print("WAVEFORM FEATURE EXTRACTION: TBCM")
    print("=" * 70)

    # Load CSV for matching
    df_csv = pd.read_csv(os.path.join(script_dir, 'dataset_TBCM.csv'), index_col=0)
    print(f"CSV: {len(df_csv)} rows")

    # Process all .mat files
    mat_files = sorted(glob.glob(os.path.join(MAT_DIR, 'Data_*.mat')))
    print(f"Found {len(mat_files)} .mat files")

    all_rows = []
    matched = 0
    unmatched = 0

    for fi, mat_path in enumerate(mat_files):
        fname = os.path.basename(mat_path)
        try:
            f = h5py.File(mat_path, 'r')
            data = f['Data']
            act = data['act'][()]
            pl = data['PL'][()].flatten()
            n_sims = act.shape[0]

            for sim_idx in range(n_sims):
                a_ct = act[sim_idx, 3]
                a_ta = act[sim_idx, 4]
                p = pl[sim_idx]

                # Match to CSV
                mask = ((np.abs(df_csv.a_CT - a_ct) < 0.001) &
                        (np.abs(df_csv.a_TA - a_ta) < 0.001) &
                        (np.abs(df_csv.PL - p) < 1))
                csv_matches = df_csv[mask]

                if len(csv_matches) != 1:
                    unmatched += 1
                    continue

                csv_row = csv_matches.iloc[0]

                # Extract waveform features
                features = process_one_sim(data, sim_idx)

                # Combine with CSV data
                row = {
                    'a_CT': csv_row.a_CT,
                    'a_TA': csv_row.a_TA,
                    'PL': csv_row.PL,
                    'Ps': csv_row.Ps,
                    'F0': csv_row.F0,
                    'SPL': csv_row.SPL,
                }
                row.update(features)
                all_rows.append(row)
                matched += 1

            f.close()
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            continue

        if (fi + 1) % 20 == 0:
            print(f"  Processed {fi+1}/{len(mat_files)} files, {matched} matched so far")

    print(f"\nTotal matched: {matched}, unmatched: {unmatched}")

    # Build DataFrame
    df_enriched = pd.DataFrame(all_rows)
    print(f"\nEnriched dataset: {df_enriched.shape}")
    print(f"Columns ({len(df_enriched.columns)}):")

    # Group columns
    base_cols = ['a_CT', 'a_TA', 'PL', 'Ps', 'F0', 'SPL']
    feat_cols = [c for c in df_enriched.columns if c not in base_cols]
    print(f"  Base: {base_cols}")
    print(f"  Waveform features ({len(feat_cols)}):")
    for c in sorted(feat_cols):
        valid = df_enriched[c].notna().sum()
        print(f"    {c}: {valid}/{len(df_enriched)} valid, "
              f"range=[{df_enriched[c].min():.4f}, {df_enriched[c].max():.4f}]")

    # Save
    out_path = os.path.join(script_dir, 'dataset_TBCM_enriched.csv')
    df_enriched.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Correlation analysis: which features correlate with F0/SPL?
    print("\n" + "=" * 70)
    print("CORRELATION WITH F0 AND SPL")
    print("=" * 70)
    print(f"\n{'Feature':<35} {'Corr(F0)':>10} {'Corr(SPL)':>10} {'Independent?':>12}")
    print("-" * 70)

    for c in sorted(feat_cols):
        col = df_enriched[c].dropna()
        if len(col) < 10:
            continue
        valid_idx = col.index
        corr_f0 = np.corrcoef(col, df_enriched.loc[valid_idx, 'F0'])[0, 1]
        corr_spl = np.corrcoef(col, df_enriched.loc[valid_idx, 'SPL'])[0, 1]
        # "Independent" if it has low correlation with BOTH F0 and SPL
        independent = abs(corr_f0) < 0.5 and abs(corr_spl) < 0.5
        marker = "<-- NEW INFO" if independent else ""
        print(f"  {c:<33} {corr_f0:>+10.3f} {corr_spl:>+10.3f} {marker:>12}")

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
