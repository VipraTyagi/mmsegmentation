import matplotlib.pyplot as plt
import numpy as np

# Datasets and corresponding mean mIoU scores
datasets = ['PhenoBench', 'UAV Bonn', 'UAV Zurich', 'UGV Bonn']
miou_scores = [88.3, 77.89, 73.04, 81.27]

# Standard deviations for each dataset (replace with actual values)
std_devs = [0.93, 4.29, 0.38, 2.0]

# Generate colors from a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
bar_width = 0.4  
plt.figure(figsize=(10, 6))
bars = plt.bar(datasets, miou_scores, color=colors, edgecolor='black', width=bar_width)

# Axis labels and title
plt.xlabel('Datasets under Evaluation', fontsize=12)
plt.ylabel('mIoU', fontsize=12)
plt.title('Comparison of mIoU Scores Across Datasets (Mean ± SD)', fontsize=14)

# Add value labels on top of each bar showing mean ± SD
for bar, mean, sd in zip(bars, miou_scores, std_devs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{mean:.2f} ± {sd:.2f}',
             ha='center', va='bottom', fontsize=10)

plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--')
plt.show()
