import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================
# Input data
# ============================
methods = ["Baseline", "TL", "EWC", "Replay buffer"]

# Keep only the three datasets under evaluation
results = {
    "UAV-Bonn": [77.89, 88.18, 88.02, 88.6],
    "UAV-Zurich": [73.04, 84.45, 80.81, 86.1],
    "UGV-Bonn": [55.94, 79.15, 77.3, 77.27],
}

metric_name = "mIoU only(%)"

# ============================
# Plot settings
# ============================
datasets = list(results.keys())
colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes = axes.ravel()

x = np.arange(len(methods))

for ax, dataset in zip(axes, datasets):
    scores = results[dataset]
    bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_title(dataset, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, fontweight='bold')   # Bold x-tick labels
    ax.set_ylim(0, 100)
    ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')   # Bold y-label
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels above bars
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Create legend with bold text
legend_handles = [
    Patch(facecolor=colors[i], edgecolor='black', label=methods[i]) for i in range(len(methods))
]

# Add figure title and legend
fig.suptitle(f"Comparison across methods ({metric_name})", fontsize=15, fontweight='bold')
legend = fig.legend(
    handles=legend_handles, loc='lower center', ncol=4, fontsize=10, frameon=False
)

# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Adjust layout
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.15, wspace=0.25)

# Save and display
# plt.savefig("comparison_three_panel_bold.png", dpi=300, bbox_inches="tight")
plt.show()
