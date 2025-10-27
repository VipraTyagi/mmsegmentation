import matplotlib.pyplot as plt
import numpy as np

# Datasets in order
datasets = ['PhenoBench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']

# Example mean IoU scores (replace with actual values)
crop_iou_means = [95.11, 87.20, 69.12, 85.1]
weed_iou_means = [70.40, 49.01, 53.7, 77.4]

# Example standard deviations (replace with actual values)
crop_iou_std = [2.4, 1.26, 0.26, 2.0]
weed_iou_std = [2.46, 13.05, 1.74, 2.8]

# Colors from colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

# Create 2x1 panel (two rows, one column)
fig, axes = plt.subplots(2, 1, figsize=(10, 10))



bar_width = 0.4  

# --- Crop IoU Chart ---
bars1 = axes[0].bar(datasets, crop_iou_means, color=colors, edgecolor='black', width=bar_width)
axes[0].set_title('Crop IoU Comparison Across Datasets when model is trained on PhenoBench Data only', fontsize=12)
axes[0].set_ylabel('Crop IoU (%)', fontsize=12)
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', linestyle='--')

# Add mean ± SD labels above bars
for bar, mean, sd in zip(bars1, crop_iou_means, crop_iou_std):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{mean:.1f} ± {sd:.1f}', ha='center', va='bottom', fontsize=10)

# --- Weed IoU Chart ---
bars2 = axes[1].bar(datasets, weed_iou_means, color=colors, edgecolor='black',  width=bar_width)
axes[1].set_title('Weed IoU Comparison Across Datasets when model is trained on PhenoBench Data only', fontsize=14)
axes[1].set_ylabel('Weed IoU (%)', fontsize=12)
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', linestyle='--')

# Add mean ± SD labels above bars
for bar, mean, sd in zip(bars2, weed_iou_means, weed_iou_std):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{mean:.1f} ± {sd:.1f}', ha='center', va='bottom', fontsize=10)

# Add a common X label
fig.text(0.5, 0.04, 'Datasets', ha='center', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
