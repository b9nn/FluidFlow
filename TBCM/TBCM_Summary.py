"""
Cross-Method Comparison: RF Transfer vs Autoencoder Domain Adaptation (BCM -> TBCM)

Loads results from TBCM_TransferRF.py and TBCM_Autoencoder.py, produces
unified comparison figures and console output.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("CROSS-METHOD COMPARISON: BCM -> TBCM")
    print("=" * 70)

    # Load results
    rf_path = os.path.join(script_dir, 'results', 'rf_transfer_results.json')
    ae_path = os.path.join(script_dir, 'results', 'autoencoder_results.json')

    with open(rf_path) as f:
        rf_results = json.load(f)
    with open(ae_path) as f:
        ae_results = json.load(f)

    print(f"\nLoaded RF results: {len(rf_results)} fractions")
    print(f"Loaded AE results: {len(ae_results)} fractions")

    # Build unified data structure
    # Match fractions between RF and AE
    rf_fracs = {r['frac']: r for r in rf_results}
    ae_fracs = {r['frac']: r for r in ae_results}
    common_fracs = sorted(set(rf_fracs.keys()) & set(ae_fracs.keys()))

    print(f"Common fractions: {common_fracs}")

    # All methods to compare
    all_methods = {
        # RF methods
        'Source Only': lambda rf, ae: (rf['source_only_f0_r2_mean'], rf['source_only_spl_r2_mean']),
        'Target Only': lambda rf, ae: (rf['target_only_f0_r2_mean'], rf['target_only_spl_r2_mean']),
        'Residual': lambda rf, ae: (rf['residual_f0_r2_mean'], rf['residual_spl_r2_mean']),
        'Augmented': lambda rf, ae: (rf['augmented_f0_r2_mean'], rf['augmented_spl_r2_mean']),
        'Simple Ens': lambda rf, ae: (rf['simple_ens_f0_r2_mean'], rf['simple_ens_spl_r2_mean']),
        'TransRF': lambda rf, ae: (rf['transrf_f0_r2_mean'], rf['transrf_spl_r2_mean']),
        # AE methods
        'Vanilla AE': lambda rf, ae: (ae['vanilla_f0_r2_mean'], ae['vanilla_spl_r2_mean']),
        'MMD AE': lambda rf, ae: (ae['mmd_f0_r2_mean'], ae['mmd_spl_r2_mean']),
        'DAAE': lambda rf, ae: (ae['daae_f0_r2_mean'], ae['daae_spl_r2_mean']),
    }

    # Build data arrays
    n_samples_list = []
    method_data = {name: {'f0': [], 'spl': []} for name in all_methods}

    for frac in common_fracs:
        rf = rf_fracs[frac]
        ae = ae_fracs[frac]
        n_samples_list.append(rf['n_samples'])

        for name, fn in all_methods.items():
            f0_r2, spl_r2 = fn(rf, ae)
            method_data[name]['f0'].append(f0_r2)
            method_data[name]['spl'].append(spl_r2)

    # ==================== CONSOLE OUTPUT ====================
    print("\n" + "=" * 70)
    print("UNIFIED RESULTS TABLE")
    print("=" * 70)

    # Header
    header = f"{'Frac':>5} {'N':>6}"
    for name in all_methods:
        header += f" {name:>11}"
    print(f"\nAverage R2 (F0+SPL)/2:")
    print(header)
    print("-" * len(header))

    for i, frac in enumerate(common_fracs):
        row = f"{frac*100:>4.0f}% {n_samples_list[i]:>6}"
        for name in all_methods:
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            row += f" {avg:>11.3f}"
        print(row)

    # Transfer gain table (vs target-only)
    print("\n\nTRANSFER GAIN vs TARGET-ONLY:")
    print("-" * 90)
    transfer_methods = [n for n in all_methods if n not in ('Source Only', 'Target Only')]
    header2 = f"{'Frac':>5} {'N':>6}"
    for name in transfer_methods:
        header2 += f" {name:>11}"
    print(header2)
    print("-" * len(header2))

    for i, frac in enumerate(common_fracs):
        tgt_avg = (method_data['Target Only']['f0'][i] + method_data['Target Only']['spl'][i]) / 2
        row = f"{frac*100:>4.0f}% {n_samples_list[i]:>6}"
        for name in transfer_methods:
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            gain = avg - tgt_avg
            row += f" {gain:>+11.4f}"
        print(row)

    # Winner per fraction
    print("\n\nWINNER PER FRACTION:")
    print("-" * 60)
    for i, frac in enumerate(common_fracs):
        best_name = None
        best_score = -float('inf')
        for name in all_methods:
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            if avg > best_score:
                best_score = avg
                best_name = name
        print(f"  {frac*100:>4.0f}% ({n_samples_list[i]:>6} samples): "
              f"{best_name:<12} (avg R2={best_score:.3f})")

    # ==================== VISUALIZATION ====================
    print("\nGenerating comparison plots...")
    figs_dir = os.path.join(script_dir, 'figs')

    # Color/style scheme
    plot_config = {
        'Source Only':  ('gray', '--', 'D', 5),
        'Target Only':  ('blue', '-', 'o', 6),
        'Residual':     ('purple', '-', 's', 5),
        'Augmented':    ('green', '-', '^', 5),
        'Simple Ens':   ('orange', '--', 'v', 5),
        'TransRF':      ('red', '-', 'o', 7),
        'Vanilla AE':   ('steelblue', '-.', 'D', 5),
        'MMD AE':       ('darkorange', '-.', 's', 5),
        'DAAE':         ('darkred', '-.', '^', 6),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('BCM -> TBCM Transfer Learning: All Methods Compared',
                 fontsize=16, fontweight='bold')

    # --- Top-left: F0 R2 vs samples ---
    ax = axes[0, 0]
    for name in all_methods:
        color, ls, marker, ms = plot_config[name]
        ax.plot(n_samples_list, method_data[name]['f0'],
                marker=marker, linestyle=ls, color=color,
                linewidth=2, markersize=ms, label=name)
    ax.set_xlabel('Number of Training Samples', fontsize=11)
    ax.set_ylabel('R2 for F0', fontsize=11)
    ax.set_title('F0 Prediction', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Top-right: SPL R2 vs samples ---
    ax = axes[0, 1]
    for name in all_methods:
        color, ls, marker, ms = plot_config[name]
        ax.plot(n_samples_list, method_data[name]['spl'],
                marker=marker, linestyle=ls, color=color,
                linewidth=2, markersize=ms, label=name)
    ax.set_xlabel('Number of Training Samples', fontsize=11)
    ax.set_ylabel('R2 for SPL', fontsize=11)
    ax.set_title('SPL Prediction', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Best method per fraction (bar chart) ---
    ax = axes[1, 0]
    bar_names = []
    bar_scores = []
    bar_colors = []
    for i, frac in enumerate(common_fracs):
        best_name = None
        best_score = -float('inf')
        for name in all_methods:
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            if avg > best_score:
                best_score = avg
                best_name = name
        bar_names.append(f"{frac*100:.0f}%")
        bar_scores.append(best_score)
        bar_colors.append(plot_config[best_name][0])

    bars = ax.bar(bar_names, bar_scores, color=bar_colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Target Data Fraction', fontsize=11)
    ax.set_ylabel('Best Average R2', fontsize=11)
    ax.set_title('Best Method per Fraction', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add method name labels on bars
    for i, frac in enumerate(common_fracs):
        best_name = None
        best_score = -float('inf')
        for name in all_methods:
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            if avg > best_score:
                best_score = avg
                best_name = name
        ax.text(i, best_score + 0.01, best_name, ha='center', va='bottom',
                fontsize=7, rotation=45, fontweight='bold')

    # --- Bottom-right: Transfer gain heatmap ---
    ax = axes[1, 1]
    transfer_names = [n for n in all_methods if n not in ('Source Only', 'Target Only')]
    gain_matrix = np.zeros((len(transfer_names), len(common_fracs)))

    for i, frac in enumerate(common_fracs):
        tgt_avg = (method_data['Target Only']['f0'][i] + method_data['Target Only']['spl'][i]) / 2
        for j, name in enumerate(transfer_names):
            avg = (method_data[name]['f0'][i] + method_data[name]['spl'][i]) / 2
            gain_matrix[j, i] = avg - tgt_avg

    im = ax.imshow(gain_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-max(abs(gain_matrix.min()), abs(gain_matrix.max())),
                   vmax=max(abs(gain_matrix.min()), abs(gain_matrix.max())))
    ax.set_xticks(range(len(common_fracs)))
    ax.set_xticklabels([f"{f*100:.0f}%" for f in common_fracs])
    ax.set_yticks(range(len(transfer_names)))
    ax.set_yticklabels(transfer_names, fontsize=8)
    ax.set_xlabel('Target Data Fraction', fontsize=11)
    ax.set_title('Transfer Gain vs Target-Only (R2)', fontsize=12, fontweight='bold')

    # Add text annotations
    for j in range(len(transfer_names)):
        for i in range(len(common_fracs)):
            val = gain_matrix[j, i]
            color = 'white' if abs(val) > 0.5 * max(abs(gain_matrix.min()), abs(gain_matrix.max())) else 'black'
            ax.text(i, j, f"{val:+.3f}", ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='R2 Gain')

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'tbcm_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
