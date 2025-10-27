import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only, fixed order)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.32

# EWC (UGV Bonn, λ = 16.25)
crop_EWC = [83.61, 69.34, 36.58, 87.84]
weed_EWC = [25.01, 28.13, 35.13, 44.96]
miou_EWC = [68.95, 64.14, 55.11, 77.53]

# CL (5% Pheno → UGV Bonn)
crop_CL = [90.73, 83.46, 56.65, 88.98]
weed_CL = [43.31, 55.03, 34.12, 43.03]
miou_CL = [77.64, 78.86, 61.86, 77.27]

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

# Colors (clear, contrasting, print-friendly)
color_EWC = '#1B4965'  # Deep Blue
color_CL  = '#BB3E03'  # Burnt Orange

label_EWC = 'EWC (λ = 16.25, UGV Bonn)'
label_CL  = 'CL (5% Pheno → UGV Bonn)'

def add_labels(ax, bars, means):
    """Add value labels above bars"""
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop + Weed)
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 10))
fig1.suptitle('Classwise IoU: EWC (λ = 16.25, UGV Bonn) vs CL (5% Pheno → UGV Bonn)',
              fontsize=15, fontweight='bold')

# --- Crop IoU ---
ax = axes[0]
b1 = ax.bar(x - bar_width/2, crop_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, crop_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Crop Class IoU Across Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper center', frameon=True)
add_labels(ax, b1, crop_EWC)
add_labels(ax, b2, crop_CL)

# --- Weed IoU ---
ax = axes[1]
b1 = ax.bar(x - bar_width/2, weed_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, weed_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Weed Class IoU Across Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Datasets Under Evaluation', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, weed_EWC)
add_labels(ax, b2, weed_CL)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# -----------------------------
# Figure 2: mIoU comparison
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - bar_width/2, miou_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, miou_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)

ax.set_title('mIoU: EWC (λ = 16.25, UGV Bonn) vs CL (5% Pheno → UGV Bonn)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Datasets Under Evaluation', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, miou_EWC)
add_labels(ax, b2, miou_CL)



plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
