"""
BCM -> Beam-Membrane: Cross-Method Comparison Summary

Loads results from RF transfer and autoencoder experiments,
produces unified comparison plots and tables.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("BCM -> BEAM-MEMBRANE: CROSS-METHOD COMPARISON")
    print("=" * 70)

    figs_dir = os.path.join(script_dir, 'figs')
    results_dir = os.path.join(script_dir, 'results')

    # Load RF results
    rf_path = os.path.join(results_dir, 'rf_transfer_results.json')
    if not os.path.exists(rf_path):
        print(f"  RF results not found: {rf_path}")
        print("  Run BM_TransferRF.py first.")
        return
    with open(rf_path) as f:
        rf_results = json.load(f)
    print(f"  Loaded RF results: {len(rf_results)} fractions")

    # Load AE results
    ae_path = os.path.join(results_dir, 'autoencoder_results.json')
    if not os.path.exists(ae_path):
        print(f"  AE results not found: {ae_path}")
        print("  Run BM_TransferAE.py first.")
        return
    with open(ae_path) as f:
        ae_results = json.load(f)
    print(f"  Loaded AE results: {len(ae_results)} fractions")

    # Build unified method list
    all_methods = {
        # RF methods
        'Source Only':    {'key': 'source_only', 'source': 'rf',
                          'color': 'gray', 'ls': '--', 'marker': 's'},
        'Target Only':    {'key': 'target_only', 'source': 'rf',
                          'color': 'blue', 'ls': '-', 'marker': 'o'},
        'Residual':       {'key': 'residual', 'source': 'rf',
                          'color': 'purple', 'ls': '-', 'marker': '^'},
        'Feature Aug':    {'key': 'augmented', 'source': 'rf',
                          'color': 'green', 'ls': '-', 'marker': 'v'},
        'Simple Ens':     {'key': 'simple_ens', 'source': 'rf',
                          'color': 'orange', 'ls': '--', 'marker': 'D'},
        'TransRF':        {'key': 'transrf', 'source': 'rf',
                          'color': 'red', 'ls': '-', 'marker': '*'},
        # AE methods
        'Vanilla AE':     {'key': 'vanilla', 'source': 'ae',
                          'color': 'steelblue', 'ls': ':', 'marker': 'p'},
        'MMD AE':         {'key': 'mmd', 'source': 'ae',
                          'color': 'darkorange', 'ls': ':', 'marker': 'h'},
        'DAAE':           {'key': 'daae', 'source': 'ae',
                          'color': 'darkred', 'ls': ':', 'marker': 'X'},
    }

    def get_r2(method_info, frac, target='avg'):
        """Get R2 value for a method at a given fraction."""
        key = method_info['key']
        source = method_info['source']
        results = rf_results if source == 'rf' else ae_results

        for r in results:
            if abs(r['frac'] - frac) < 0.001:
                if target == 'avg':
                    f0 = r.get(f'{key}_f0_r2_mean', None)
                    spl = r.get(f'{key}_spl_r2_mean', None)
                    if f0 is not None and spl is not None:
                        return (f0 + spl) / 2
                elif target == 'f0':
                    return r.get(f'{key}_f0_r2_mean', None)
                elif target == 'spl':
                    return r.get(f'{key}_spl_r2_mean', None)
        return None

    def get_r2_std(method_info, frac, target='avg'):
        """Get R2 std for a method at a given fraction."""
        key = method_info['key']
        source = method_info['source']
        results = rf_results if source == 'rf' else ae_results

        for r in results:
            if abs(r['frac'] - frac) < 0.001:
                if target == 'avg':
                    f0_s = r.get(f'{key}_f0_r2_std', 0)
                    spl_s = r.get(f'{key}_spl_r2_std', 0)
                    return (f0_s + spl_s) / 2
                elif target == 'f0':
                    return r.get(f'{key}_f0_r2_std', 0)
                elif target == 'spl':
                    return r.get(f'{key}_spl_r2_std', 0)
        return 0

    # Get fractions from RF results
    fractions = [r['frac'] for r in rf_results]
    n_samples_rf = [r['n_samples'] for r in rf_results]

    # ==================== CONSOLE SUMMARY ====================
    print("\n" + "=" * 70)
    print("UNIFIED COMPARISON: Average R2 (F0+SPL)/2")
    print("=" * 70)

    header = f"{'Frac':>5} {'N':>6}"
    for name in all_methods:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))

    for i, frac in enumerate(fractions):
        n = n_samples_rf[i]
        line = f"{frac*100:>4.0f}% {n:>6}"
        for name, info in all_methods.items():
            val = get_r2(info, frac)
            if val is not None:
                line += f" {val:>12.3f}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # Best method per fraction
    print("\n\nBEST METHOD PER DATA SIZE:")
    print("-" * 60)
    for i, frac in enumerate(fractions):
        best_name = None
        best_val = -float('inf')
        for name, info in all_methods.items():
            val = get_r2(info, frac)
            if val is not None and val > best_val:
                best_val = val
                best_name = name
        n = n_samples_rf[i]
        print(f"  {frac*100:>4.0f}% ({n:>5} samples): "
              f"{best_name:<15} (avg R2={best_val:.3f})")

    # Transfer gain table
    print("\n\nTRANSFER GAIN OVER TARGET-ONLY:")
    print("-" * 70)
    header2 = f"{'Frac':>5}"
    transfer_methods = [n for n in all_methods
                        if n not in ['Source Only', 'Target Only']]
    for name in transfer_methods:
        header2 += f" {name:>12}"
    print(header2)
    for frac in fractions:
        baseline = get_r2(all_methods['Target Only'], frac)
        if baseline is None:
            continue
        line = f"{frac*100:>4.0f}%"
        for name in transfer_methods:
            val = get_r2(all_methods[name], frac)
            if val is not None:
                gain = val - baseline
                line += f" {gain:>+12.4f}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # ==================== VISUALIZATION ====================
    print("\n\nGenerating comparison plots...")

    # --- Plot: 2x2 comprehensive comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('BCM -> Beam-Membrane: Complete Transfer Learning Comparison',
                 fontsize=16, fontweight='bold')

    # Top-left: F0 R2 vs samples (all methods)
    ax = axes[0, 0]
    for name, info in all_methods.items():
        vals = [get_r2(info, f, 'f0') for f in fractions]
        stds = [get_r2_std(info, f, 'f0') for f in fractions]
        if any(v is not None for v in vals):
            valid = [(n, v, s) for n, v, s in zip(n_samples_rf, vals, stds)
                     if v is not None]
            ns, vs, ss = zip(*valid)
            ax.plot(ns, vs, marker=info['marker'], label=name,
                    color=info['color'], linestyle=info['ls'],
                    linewidth=2, markersize=7)
            ax.fill_between(ns,
                            [v - s for v, s in zip(vs, ss)],
                            [v + s for v, s in zip(vs, ss)],
                            alpha=0.08, color=info['color'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('R2')
    ax.set_title('F0 Prediction - All Methods')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Top-right: SPL R2 vs samples
    ax = axes[0, 1]
    for name, info in all_methods.items():
        vals = [get_r2(info, f, 'spl') for f in fractions]
        stds = [get_r2_std(info, f, 'spl') for f in fractions]
        if any(v is not None for v in vals):
            valid = [(n, v, s) for n, v, s in zip(n_samples_rf, vals, stds)
                     if v is not None]
            ns, vs, ss = zip(*valid)
            ax.plot(ns, vs, marker=info['marker'], label=name,
                    color=info['color'], linestyle=info['ls'],
                    linewidth=2, markersize=7)
            ax.fill_between(ns,
                            [v - s for v, s in zip(vs, ss)],
                            [v + s for v, s in zip(vs, ss)],
                            alpha=0.08, color=info['color'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('R2')
    ax.set_title('SPL Prediction - All Methods')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Best method per fraction (bar chart)
    ax = axes[1, 0]
    bar_fracs = [f"{f*100:.0f}%" for f in fractions]
    bar_colors = []
    bar_vals = []
    bar_labels = []
    for frac in fractions:
        best_name = None
        best_val = -float('inf')
        for name, info in all_methods.items():
            val = get_r2(info, frac)
            if val is not None and val > best_val:
                best_val = val
                best_name = name
        bar_vals.append(best_val)
        bar_labels.append(best_name)
        bar_colors.append(all_methods[best_name]['color'] if best_name else 'gray')

    bars = ax.bar(bar_fracs, bar_vals, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    for bar, label in zip(bars, bar_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                label, ha='center', va='bottom', fontsize=7, rotation=15)
    ax.set_xlabel('Target Data Fraction')
    ax.set_ylabel('Best Average R2')
    ax.set_title('Best Method per Data Size')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Transfer gain heatmap
    ax = axes[1, 1]
    gain_methods = transfer_methods
    gain_matrix = []
    for name in gain_methods:
        row = []
        for frac in fractions:
            baseline = get_r2(all_methods['Target Only'], frac)
            val = get_r2(all_methods[name], frac)
            if baseline is not None and val is not None:
                row.append(val - baseline)
            else:
                row.append(0)
        gain_matrix.append(row)

    gain_matrix = np.array(gain_matrix)
    im = ax.imshow(gain_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-max(0.05, np.abs(gain_matrix).max()),
                   vmax=max(0.05, np.abs(gain_matrix).max()))
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f*100:.0f}%" for f in fractions])
    ax.set_yticks(range(len(gain_methods)))
    ax.set_yticklabels(gain_methods, fontsize=8)
    ax.set_xlabel('Target Data Fraction')
    ax.set_title('Transfer Gain over Target-Only (R2 difference)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate heatmap cells
    for i in range(len(gain_methods)):
        for j in range(len(fractions)):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > np.abs(gain_matrix).max() * 0.6 else 'black'
            ax.text(j, i, f"{val:+.3f}", ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'bm_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # --- Plot 2: RF vs AE head-to-head ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('RF vs Autoencoder: Head-to-Head (BCM -> BM)',
                  fontsize=14, fontweight='bold')

    key_methods = ['TransRF', 'Vanilla AE', 'Target Only']

    for name in key_methods:
        info = all_methods[name]
        vals_f0 = [get_r2(info, f, 'f0') for f in fractions]
        vals_spl = [get_r2(info, f, 'spl') for f in fractions]
        valid_f0 = [(n, v) for n, v in zip(n_samples_rf, vals_f0)
                    if v is not None]
        valid_spl = [(n, v) for n, v in zip(n_samples_rf, vals_spl)
                     if v is not None]

        if valid_f0:
            ns, vs = zip(*valid_f0)
            axes2[0].plot(ns, vs, 'o-', label=name, color=info['color'],
                         linewidth=2.5, markersize=8)
        if valid_spl:
            ns, vs = zip(*valid_spl)
            axes2[1].plot(ns, vs, 'o-', label=name, color=info['color'],
                         linewidth=2.5, markersize=8)

    axes2[0].set_xlabel('Number of Training Samples')
    axes2[0].set_ylabel('R2 for F0')
    axes2[0].set_title('F0: TransRF vs Vanilla AE vs Baseline')
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_xlabel('Number of Training Samples')
    axes2[1].set_ylabel('R2 for SPL')
    axes2[1].set_title('SPL: TransRF vs Vanilla AE vs Baseline')
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(figs_dir, 'bm_rf_vs_ae.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
