import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (means only, fixed order)
# -----------------------------
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.22  # slim and professional look

# PhenoBench model
crop_PB = [95.11, 87.20, 69.12, 54.68]
weed_PB = [70.40, 49.01, 53.70, 14.40]
miou_PB = [88.30, 77.89, 73.04, 55.94]

# TL (PhenoBench → UAV Bonn)
crop_TL = [78.93, 93.21, 47.85, 26.03]
weed_TL = [30.35, 74.09, 50.55, 3.09]
miou_TL = [65.75, 88.18, 64.64, 41.34]

# CL (UAV Bonn with λ = 0.6)
crop_CL = [91.52, 93.00, 59.39, 46.42]
weed_CL = [52.12, 72.13, 46.66, 1.49]
miou_CL = [80.88, 88.02, 67.48, 48.30]

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

# Colors (new, distinct)
color_PB = '#6A0DAD'   # Deep Purple
color_TL = '#FF7F50'   # Coral
color_CL = '#4682B4'   # Steel Blue

label_PB = 'PhenoBench'
label_TL = 'TL (PhenoBench → UAV Bonn)'
label_CL = 'CL on UAV Bonn(λ = 0.6)'

def add_labels(ax, bars, means):
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', clip_on=False)

# -----------------------------
# Figure 1: Classwise IoU (Crop + Weed)
# -----------------------------
fig1, axes = plt.subplots(2, 1, figsize=(12, 9))
# fig1.suptitle('Classwise IoU: PhenoBench vs TL (PhenoBench → UAV Bonn) vs CL (UAV Bonn, λ = 0.6)',
#               fontsize=15, fontweight='bold')

# Crop
ax = axes[0]
b1 = ax.bar(x - bar_width, crop_PB, bar_width, color=color_PB, edgecolor='black', label=label_PB)
b2 = ax.bar(x,             crop_TL, bar_width, color=color_TL, edgecolor='black', label=label_TL)
b3 = ax.bar(x + bar_width, crop_CL, bar_width, color=color_CL, edgecolor='black', label=label_CL)
ax.set_title('Crop Class IoU Across Datasets')
ax.set_ylabel('IoU (%)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, crop_PB)
add_labels(ax, b2, crop_TL)
add_labels(ax, b3, crop_CL)

# Weed
ax = axes[1]
b1 = ax.bar(x - bar_width, weed_PB, bar_width, color=color_PB, edgecolor='black', label=label_PB)
b2 = ax.bar(x,             weed_TL, bar_width, color=color_TL, edgecolor='black', label=label_TL)
b3 = ax.bar(x + bar_width, weed_CL, bar_width, color=color_CL, edgecolor='black', label=label_CL)
ax.set_title('Weed Class IoU Across Datasets')
ax.set_ylabel('IoU (%)')
ax.set_xlabel('Datasets Under Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
add_labels(ax, b1, weed_PB)
add_labels(ax, b2, weed_TL)
add_labels(ax, b3, weed_CL)

# Caption inside figure
fig1.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UAV Bonn), and CL (UAV Bonn, λ = 0.6)',
          ha='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# -----------------------------
# Figure 2: mIoU comparison
# -----------------------------
fig2, ax = plt.subplots(figsize=(12, 5.5))

b1 = ax.bar(x - bar_width, miou_PB, bar_width, color=color_PB, edgecolor='black', label=label_PB)
b2 = ax.bar(x,             miou_TL, bar_width, color=color_TL, edgecolor='black', label=label_TL)
b3 = ax.bar(x + bar_width, miou_CL, bar_width, color=color_CL, edgecolor='black', label=label_CL)

ax.set_title('mIoU Comparison Across Datasets')
ax.set_ylabel('Mean IoU (%)')
ax.set_xlabel('Datasets Under Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

add_labels(ax, b1, miou_PB)
add_labels(ax, b2, miou_TL)
add_labels(ax, b3, miou_CL)

# Caption inside figure
fig2.text(0.5, 0.02,
          'Datasets under evaluation on three models: PhenoBench, TL (PhenoBench → UAV Bonn), and CL (UAV Bonn, λ = 0.6)',
          ha='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
