import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only, fixed order)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.32

# EWC (Zurich, λ = 0.4)
crop_EWC = [87.96, 88.31, 79.28, 32.53]
weed_EWC = [61.27, 61.68, 64.31, 9.28]
miou_EWC = [82.58, 82.83, 80.51, 46.32]

# CL (5% Pheno → Zurich)
crop_CL = [95.30, 88.84, 84.24, 41.77]
weed_CL = [70.99, 59.16, 75.54, 6.70]
miou_CL = [88.57, 82.17, 86.10, 48.68]

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

# Colors
color_EWC = '#006D77'  # dark cyan
color_CL  = '#B23A48'  # brick red

label_EWC = 'EWC (λ = 0.4, Zurich)'
label_CL  = 'CL (5% Pheno → Zurich)'

def add_labels(ax, bars, means):
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop + Weed)
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 10))
fig1.suptitle('Classwise IoU: EWC (λ = 0.4, Zurich) vs CL (5% Pheno → Zurich)',
              fontsize=15, fontweight='bold')

# Crop
ax = axes[0]
b1 = ax.bar(x - bar_width/2, crop_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, crop_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Crop Class IoU Across Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, crop_EWC)
add_labels(ax, b2, crop_CL)

# Weed
ax = axes[1]
b1 = ax.bar(x - bar_width/2, weed_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, weed_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Weed Class IoU Across Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Datasets Under Evaluation', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, weed_EWC)
add_labels(ax, b2, weed_CL)

plt.show()

# -----------------------------
# Figure 2: mIoU comparison
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(x - bar_width/2, miou_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, miou_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)

ax.set_title('mIoU: EWC (λ = 0.4, Zurich) vs CL (5% Pheno → Zurich)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Datasets Under Evaluation', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, miou_EWC)
add_labels(ax, b2, miou_CL)


plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
