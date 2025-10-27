import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only, fixed order)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.24  # slim bars for tight pairing

# EWC (λ = 0.6, UAV Bonn)
crop_EWC = [91.52, 93.00, 59.39, 46.42]
weed_EWC = [52.12, 72.13, 46.66, 1.49]
miou_EWC = [80.88, 88.02, 67.48, 48.30]

# CL (5% Pheno → UAV Bonn)
crop_CL = [94.91, 92.97, 68.75, 77.60]
weed_CL = [68.97, 73.89, 48.69, 8.20]
miou_CL = [87.75, 88.60, 71.30, 61.63]

# -----------------------------
# Styling
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

# Colors (clear and distinct)
color_EWC = '#1B4965'  # deep blue
color_CL  = '#BB3E03'  # burnt orange

label_EWC = 'EWC (λ = 0.6, UAV Bonn)'
label_CL  = 'CL (5% Pheno → UAV Bonn)'

def add_labels(ax, bars, means):
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop + Weed) — paired bars per dataset
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 9))
fig1.suptitle('Classwise IoU: EWC vs CL (5% Pheno) on UAV Bonn',
              fontsize=15, fontweight='bold')

# Crop IoU
ax = axes[0]
b1 = ax.bar(x - bar_width/2, crop_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, crop_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Crop Class IoU Across Datasets')
ax.set_ylabel('IoU (%)')
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, crop_EWC)
add_labels(ax, b2, crop_CL)

# Weed IoU
ax = axes[1]
b1 = ax.bar(x - bar_width/2, weed_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, weed_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)
ax.set_title('Weed Class IoU Across Datasets')
ax.set_ylabel('IoU (%)')
ax.set_xlabel('Datasets Under Evaluation')
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, weed_EWC)
add_labels(ax, b2, weed_CL)

# Caption
# fig1.text(0.5, 0.02,
#           'Each dataset shows two bars: left = EWC ( λ = 0.6, UAV Bonn), right = CL (5% Pheno → UAV Bonn)',
#           ha='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# -----------------------------
# Figure 2: mIoU — paired bars per dataset
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - bar_width/2, miou_EWC, bar_width, color=color_EWC, edgecolor='black', label=label_EWC)
b2 = ax.bar(x + bar_width/2, miou_CL,  bar_width, color=color_CL,  edgecolor='black', label=label_CL)

ax.set_title('Mean IoU: EWC(λ = 0.6, UAV Bonn) vs CL(5% Pheno) on UAV Bonn')
ax.set_ylabel('Mean IoU (%)')
ax.set_xlabel('Datasets Under Evaluation')
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, miou_EWC)
add_labels(ax, b2, miou_CL)

# fig2.text(0.5, 0.02,
#           'Each dataset shows two bars: left = EWC (UAV Bonn, λ = 0.6), right = CL (5% Pheno → UAV Bonn)',
#           ha='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
