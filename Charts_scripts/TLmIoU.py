import matplotlib.pyplot as plt
import numpy as np

# Datasets under evaluation (consistent order)
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.32

# ===== Model trained on PhenoBench =====
miou_pheno = [88.30, 77.89, 73.04, 55.94]
miou_pheno_sd = [0.93, 4.29, 0.38, 17.06]

# ===== Model trained on PhenoBench → UGV Bonn =====
miou_pb_ugv = [42.59, 49.16, 41.13, 79.15]
miou_pb_ugv_sd = [1.59, 12.6, 8.5, 2.48]

# ===== Colors =====
color_pheno = '#2E8B57'   # deep teal
color_pb_ugv = '#B22222'  # muted crimson

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
bars1 = ax.bar(x - bar_width/2, miou_pheno, bar_width,
               label='Trained on PhenoBench', color=color_pheno, edgecolor='black')
bars2 = ax.bar(x + bar_width/2, miou_pb_ugv, bar_width,
               label='Trained on PhenoBench → UGV Bonn', color=color_pb_ugv, edgecolor='black')

# Title and labels
ax.set_title('mIoU Comparison Across Datasets', fontsize=14)
ax.set_ylabel('Mean Intersection over Union (mIoU) (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Small legend inside chart (top-right)
ax.legend(fontsize=9, loc='upper right', frameon=True)

# Add mean ± SD labels above bars
for bar, mean, sd in zip(bars1, miou_pheno, miou_pheno_sd):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{mean:.2f} ± {sd:.2f}', ha='center', va='bottom', fontsize=9, clip_on=False)
for bar, mean, sd in zip(bars2, miou_pb_ugv, miou_pb_ugv_sd):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{mean:.2f} ± {sd:.2f}', ha='center', va='bottom', fontsize=9, clip_on=False)

# Caption below the figure
fig.text(0.5, 0.02,
         'Datasets under evaluation on models trained on PhenoBench and PhenoBench → UGV Bonn',
         ha='center', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
