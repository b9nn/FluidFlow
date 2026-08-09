"""Quick plot of small-data results — reuses data from TBCM_SmallData.py output."""
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Results from the experiment (avg R2 over 5 runs)
N = [10, 20, 30, 50, 75, 100, 200, 500]

# Average R2 = (F0 + SPL) / 2
target_only = [-1.149, -1.010, -1.877, 0.352, 0.548, 0.622, 0.850, 0.938]
residual    = [-2.295, -0.691, -1.062, 0.426, 0.500, 0.512, 0.837, 0.908]
augmented   = [-1.149, -1.010, -1.877, 0.387, 0.621, 0.677, 0.834, 0.935]
transrf     = [-0.378, -0.072, -0.601, 0.593, 0.692, 0.748, 0.873, 0.943]
vanilla_ae  = [-0.483,  0.065, -0.101, 0.637, 0.676, 0.731, 0.788, 0.825]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle('BCM → TBCM Transfer: Small Target Data Regime',
             fontsize=15, fontweight='bold', y=0.98)

# ---- Left panel: all methods ----
ax = axes[0]
ax.plot(N, target_only, 'o-', color='blue', linewidth=2, markersize=7, label='Target Only (RF)')
ax.plot(N, residual,    's-', color='purple', linewidth=1.5, markersize=5, label='Residual Correction')
ax.plot(N, augmented,   '^-', color='green', linewidth=1.5, markersize=5, label='Feature Augmentation')
ax.plot(N, transrf,     'D-', color='red', linewidth=2.5, markersize=7, label='TransRF')
ax.plot(N, vanilla_ae,  'o--', color='steelblue', linewidth=2.5, markersize=7, label='Vanilla AE')

ax.axhline(y=0, color='black', linestyle=':', alpha=0.4, linewidth=1)
ax.set_xlabel('Number of Target Samples', fontsize=12)
ax.set_ylabel('Average R² (F0 + SPL) / 2', fontsize=12)
ax.set_title('All Transfer Methods', fontsize=13)
ax.set_xscale('log')
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N])
ax.set_ylim([-2.5, 1.05])
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Shade the "AE wins" region
ax.axvspan(10, 65, alpha=0.06, color='steelblue')
ax.text(32, -2.3, 'AE wins', ha='center', fontsize=10, color='steelblue',
        fontstyle='italic', fontweight='bold')
ax.axvspan(65, 600, alpha=0.06, color='red')
ax.text(200, -2.3, 'TransRF wins', ha='center', fontsize=10, color='red',
        fontstyle='italic', fontweight='bold')

# ---- Right panel: TransRF vs Vanilla AE head-to-head ----
ax2 = axes[1]

ax2.plot(N, transrf, 'D-', color='red', linewidth=2.5, markersize=8, label='TransRF')
ax2.plot(N, vanilla_ae, 'o--', color='steelblue', linewidth=2.5, markersize=8, label='Vanilla AE')
ax2.plot(N, target_only, 'o-', color='blue', linewidth=1.5, markersize=5,
         alpha=0.4, label='Target Only (baseline)')

# Highlight winner at each point
for i, n in enumerate(N):
    if vanilla_ae[i] > transrf[i]:
        ax2.annotate(f'AE +{vanilla_ae[i]-transrf[i]:.2f}',
                     (n, vanilla_ae[i]), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=8, color='steelblue',
                     fontweight='bold')
    else:
        ax2.annotate(f'RF +{transrf[i]-vanilla_ae[i]:.2f}',
                     (n, transrf[i]), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=8, color='red',
                     fontweight='bold')

ax2.axhline(y=0, color='black', linestyle=':', alpha=0.4, linewidth=1)
ax2.axvline(x=65, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.text(65, 0.95, 'crossover\n~60 samples', ha='center', fontsize=9,
         color='gray', fontstyle='italic')

ax2.set_xlabel('Number of Target Samples', fontsize=12)
ax2.set_ylabel('Average R² (F0 + SPL) / 2', fontsize=12)
ax2.set_title('TransRF vs Vanilla AE (Head-to-Head)', fontsize=13)
ax2.set_xscale('log')
ax2.set_xticks(N)
ax2.set_xticklabels([str(n) for n in N])
ax2.set_ylim([-1.0, 1.1])
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(script_dir, 'figs', 'tbcm_small_data_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")
plt.close()
