import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only, in fixed order)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.25

# Model A: Trained on PhenoBench
crop_A = [95.11, 87.20, 69.12, 54.68]
weed_A = [70.40, 49.01, 53.70, 14.40]
miou_A = [88.30, 77.89, 73.04, 55.94]

# Model B: TL (PhenoBench → UGV Bonn)  [reordered to match datasets above]
crop_B = [35.80, 44.09, 19.89, 90.83]
weed_B = [3.14, 13.41, 14.26, 49.44]
miou_B = [42.59, 49.16, 41.13, 79.15]

# Model C: Trained on UGV Bonn with λ = 16.25  [reordered to match datasets above]
crop_C = [83.61, 69.34, 36.58, 87.84]
weed_C = [25.01, 28.13, 35.13, 44.96]
miou_C = [68.95, 64.14, 55.11, 77.53]

# -----------------------------
# Styling (bold fonts for PNG clarity)
# -----------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9
})

# Color-blind friendly palette
color_A = '#1b9e77'   # green (PhenoBench)
color_B = '#d95f02'   # orange (TL → UGV Bonn)
color_C = '#7570b3'   # purple (UGV Bonn λ=16.25)

label_A = 'Trained on PhenoBench'
label_B = 'TL (PhenoBench → UGV Bonn)'
label_C = 'CL on UGV Bonn (λ = 16.25)'

def add_labels(ax, bars, means):
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop top, Weed bottom)
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Crop IoU ---
ax = axes[0]
b1 = ax.bar(x - bar_width, crop_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             crop_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, crop_C, bar_width, color=color_C, edgecolor='black', label=label_C)

ax.set_title('Crop IoU Comparison Across Datasets')
ax.set_ylabel('Crop IoU (%)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
# place legend where it won't cover labels
ax.legend(loc='upper center', frameon=True)

add_labels(ax, b1, crop_A)
add_labels(ax, b2, crop_B)
add_labels(ax, b3, crop_C)

# --- Weed IoU ---
ax = axes[1]
b1 = ax.bar(x - bar_width, weed_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             weed_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, weed_C, bar_width, color=color_C, edgecolor='black', label=label_C)

ax.set_title('Weed IoU Comparison Across Datasets')
ax.set_ylabel('Weed IoU (%)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, weed_A)
add_labels(ax, b2, weed_B)
add_labels(ax, b3, weed_C)

fig1.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UGV Bonn), and CL on UGV Bonn (λ = 16.25)',
          ha='center', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# -----------------------------
# Figure 2: mIoU comparison
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(x - bar_width, miou_A, bar_width, color=color_A, edgecolor='black', label=label_A)
b2 = ax.bar(x,             miou_B, bar_width, color=color_B, edgecolor='black', label=label_B)
b3 = ax.bar(x + bar_width, miou_C, bar_width, color=color_C, edgecolor='black', label=label_C)

ax.set_title('mIoU Comparison Across Datasets')
ax.set_ylabel('Mean Intersection over Union (mIoU) (%)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, miou_A)
add_labels(ax, b2, miou_B)
add_labels(ax, b3, miou_C)

fig2.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UGV Bonn), and CL on UGV Bonn (λ = 16.25)',
          ha='center', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
