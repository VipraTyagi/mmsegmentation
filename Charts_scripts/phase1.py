import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import re

# Parse strings like "88,3 ± 0.93" to floats (value part only)
def parse_value(s: str):
    s = str(s).strip().replace(',', '.')
    if '±' in s:
        s = s.split('±')[0].strip()
    m = re.search(r'[-+]?\d*\.?\d+', s)
    return float(m.group(0)) if m else None

# Datasets (order used across all plots)
datasets = ["Phenobench", "UGV Bonn", "UAV Zurich", "UAV Bonn"]

# Raw values
miou_raw = {
    "Phenobench": "88,3 ± 0.93",
    "UGV Bonn": "55,94 ± 17,06",
    "UAV Zurich": "73,04 ± 0,38",
    "UAV Bonn": "77.89 ± 4.29",
}

crop_iou_raw = {
    "Phenobench": "95.11 ±  0,34",
    "UGV Bonn": "54,68 ± 31,21",
    "UAV Zurich": "69.12 ± 0,26",
    "UAV Bonn": "87.20 ± 1,26",
}

weed_iou_raw = {
    "Phenobench": "70.40 ± 2.46",
    "UGV Bonn": "14,4 ± 18,75",
    "UAV Zurich": "53,7± 1.74",
    "UAV Bonn": "49.01 ± 13.05",
}

# Convert to float lists (value only)
miou = [parse_value(miou_raw[d]) for d in datasets]
crop_iou = [parse_value(crop_iou_raw[d]) for d in datasets]
weed_iou = [parse_value(weed_iou_raw[d]) for d in datasets]

# Consistent colors for datasets
tab10 = cm.get_cmap('tab10')
color_map = {
    "Phenobench": tab10(0),
    "UGV Bonn": tab10(1),
    "UAV Zurich": tab10(2),
    "UAV Bonn": tab10(3),
}

# Common x positions
x = np.arange(len(datasets))

# ---------- 1) mIoU (everything bold, legend included) ----------
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.subplots_adjust(right=0.8)  # leave room for legend outside
bars1 = ax1.bar(x, miou, color=[color_map[d] for d in datasets])

ax1.set_title("mIoU (model trained on Phenobench)", fontweight='bold')
ax1.set_ylabel("mIoU (%)", fontweight='bold')
ax1.set_xlabel("Dataset under evaluation", fontweight='bold')
ax1.set_xticks(x, datasets, fontweight='bold')
for label in ax1.get_yticklabels():
    label.set_fontweight('bold')

for rect in bars1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.5, f"{h:.2f}",
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax1.legend(bars1, datasets, title="Dataset",
           prop={'weight': 'bold'},
           loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()

# ---------- 2) Crop IoU (separate figure) ----------
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.subplots_adjust(right=0.8)
bars2 = ax2.bar(x, crop_iou, color=[color_map[d] for d in datasets])

ax2.set_title("Crop IoU (model trained on Phenobench) ", fontweight='bold')
ax2.set_ylabel("IoU (%)", fontweight='bold')
ax2.set_xlabel("Dataset under evaluation", fontweight='bold')
ax2.set_xticks(x, datasets, fontweight='bold')

for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

for rect in bars2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, h + 0.5, f"{h:.2f}",
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax2.legend(bars2, datasets, title="Dataset",
           loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()

# ---------- 3) Weed IoU (separate figure) ----------
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.subplots_adjust(right=0.8)
bars3 = ax3.bar(x, weed_iou, color=[color_map[d] for d in datasets])

ax3.set_title("Weed IoU (model trained on Phenobench)", fontweight='bold')
ax3.set_ylabel("IoU (%)", fontweight='bold')
ax3.set_xlabel("Dataset under evaluation", fontweight='bold')
ax3.set_xticks(x, datasets, fontweight='bold')

for label in ax3.get_yticklabels():
    label.set_fontweight('bold')

for rect in bars3:
    h = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2.0, h + 0.5, f"{h:.2f}",
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax3.legend(bars3, datasets, title="Dataset",
           loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()
