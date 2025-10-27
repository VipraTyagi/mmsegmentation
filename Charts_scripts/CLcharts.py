import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))

# Model A: Trained on PhenoBench
crop_A = [95.11, 87.20, 69.12, 54.68]
weed_A = [70.40, 49.01, 53.70, 14.40]
miou_A = [88.30, 77.89, 73.04, 55.94]

# Model B: Trained on PhenoBench → UAV Zurich
crop_B = [78.26, 77.49, 84.34, 30.02]
weed_B = [23.50, 50.62, 72.70, 7.33]
miou_B = [66.31, 74.21, 84.45, 44.59]

# Model C: Continual learning on 5% Pheno + trained on Zurich
crop_C = [95.30, 88.84, 84.24, 41.77]
weed_C = [70.99, 59.16, 75.54, 6.70]
miou_C = [88.57, 82.17, 86.10, 48.68]

# -----------------------------
# Styling
# -----------------------------
bar_width = 0.25
color_A = '#2E8B57'  # teal
color_B = '#1f4e79'  # navy
color_C = '#B22222'  # crimson
label_A = 'Trained on PhenoBench'
label_B = 'TL (PhenoBench → UAV Zurich)'
label_C = 'CL (5% Pheno) → Zurich'

def add_labels(ax, bars, means):
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom', fontsize=8, clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop top, Weed bottom)
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 10))

# Crop
ax = axes[0]
b1 = ax.bar(x - bar_width, crop_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             crop_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, crop_C, bar_width, color=color_C, edgecolor='black', label=label_C)
ax.set_title('Crop IoU Comparison Across Datasets', fontsize=14)
ax.set_ylabel('Crop IoU (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(fontsize=9, loc='upper right', frameon=True)
add_labels(ax, b1, crop_A)
add_labels(ax, b2, crop_B)
add_labels(ax, b3, crop_C)

# Weed
ax = axes[1]
b1 = ax.bar(x - bar_width, weed_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             weed_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, weed_C, bar_width, color=color_C, edgecolor='black', label=label_C)
ax.set_title('Weed IoU Comparison Across Datasets', fontsize=14)
ax.set_ylabel('Weed IoU (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(fontsize=9, loc='upper right', frameon=True)
add_labels(ax, b1, weed_A)
add_labels(ax, b2, weed_B)
add_labels(ax, b3, weed_C)

fig1.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UAV Zurich), and CL (5% Pheno) → Zurich',
          ha='center', fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# -----------------------------
# Figure 2: mIoU comparison
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(x - bar_width, miou_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             miou_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, miou_C, bar_width, color=color_C, edgecolor='black', label=label_C)

ax.set_title('mIoU Comparison Across Datasets', fontsize=14)
ax.set_ylabel('Mean Intersection over Union (mIoU) (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(fontsize=9, loc='upper right', frameon=True)

add_labels(ax, b1, miou_A)
add_labels(ax, b2, miou_B)
add_labels(ax, b3, miou_C)

fig2.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UAV Zurich), and CL (5% Pheno) → Zurich',
          ha='center', fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
