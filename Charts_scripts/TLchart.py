import matplotlib.pyplot as plt
import numpy as np

# Use bold font globally
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Datasets under evaluation (consistent order)
datasets = ['Phenobench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
x = np.arange(len(datasets))
bar_width = 0.32  # slightly narrower bars

# ===== Model trained on PhenoBench =====
crop_pheno =    [95.11, 87.20, 69.12, 54.68]
crop_pheno_sd = [0.34,  1.26,  0.26,  31.21]
weed_pheno =    [70.40, 49.01, 53.70, 14.40]
weed_pheno_sd = [2.46,  13.05, 1.74,  18.75]

# ===== Model trained on PhenoBench → UGV Bonn =====
crop_pb_ugv =    [35.80, 44.09, 19.89, 90.83]
crop_pb_ugv_sd = [5.24,  25.40, 14.65, 0.70]
weed_pb_ugv =    [3.14,  13.41, 14.26, 49.44]
weed_pb_ugv_sd = [0.46,  8.20, 8.34, 10.18]

# ===== Colors (teal and crimson) =====
color_pheno = '#2E8B57'   # deep teal
color_pb_ugv = '#B22222'  # muted crimson

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# -------- Crop IoU Chart --------
bars1 = axes[0].bar(x - bar_width/2, crop_pheno, bar_width,
                    label='Trained on PhenoBench', color=color_pheno, edgecolor='black')
bars2 = axes[0].bar(x + bar_width/2, crop_pb_ugv, bar_width,
                    label='Trained on PhenoBench → UGV Bonn', color=color_pb_ugv, edgecolor='black')

axes[0].set_title('Crop IoU Comparison Across Datasets', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Crop IoU (%)', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets, fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 105)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Add legend
axes[0].legend(fontsize=10, loc='upper center', frameon=True, prop={'weight':'bold'})

# Add mean ± SD labels (bold text)
for bar, mean, sd in zip(bars1, crop_pheno, crop_pheno_sd):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{mean:.2f} ± {sd:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
for bar, mean, sd in zip(bars2, crop_pb_ugv, crop_pb_ugv_sd):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{mean:.2f} ± {sd:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)

# -------- Weed IoU Chart --------
bars3 = axes[1].bar(x - bar_width/2, weed_pheno, bar_width,
                    label='Trained on PhenoBench', color=color_pheno, edgecolor='black')
bars4 = axes[1].bar(x + bar_width/2, weed_pb_ugv, bar_width,
                    label='Trained on PhenoBench → UGV Bonn', color=color_pb_ugv, edgecolor='black')

axes[1].set_title('Weed IoU Comparison Across Datasets', fontsize=15, fontweight='bold')
axes[1].set_ylabel('Weed IoU (%)', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets, fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 105)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# Add legend
axes[1].legend(fontsize=10, loc='upper right', frameon=True, prop={'weight':'bold'})

# Add mean ± SD labels (bold)
for bar, mean, sd in zip(bars3, weed_pheno, weed_pheno_sd):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{mean:.2f} ± {sd:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
for bar, mean, sd in zip(bars4, weed_pb_ugv, weed_pb_ugv_sd):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{mean:.2f} ± {sd:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)

# Bold caption below the figure
fig.text(0.5, 0.02,
         'Datasets under evaluation on models trained on PhenoBench and PhenoBench → UGV Bonn',
         ha='center', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
