#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === CONFIG: your real dataset paths & names ===
MASK_PATH   = "/home/vipra/Thesis/Semantic_Segmentation/data/cropandweed/vwg-1347-0006.png"

CLASS_NAMES = {
    4: "Background",
    1: "Crop",
    2: "Weed",
    3: "vegetation"
}

def visualize_mask_only(mask_path, class_names=None, random_seed=0):
    # Load mask (single‐channel)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask at {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]

    # Find unique IDs & pixel counts
    unique_ids, counts = np.unique(mask, return_counts=True)
    print(f"Found {len(unique_ids)} distinct IDs: {unique_ids.tolist()}")
    for uid, cnt in zip(unique_ids, counts):
        name = class_names.get(uid, f"ID {uid}") if class_names else f"ID {uid}"
        print(f"  {name} ({uid}): {cnt} pixels")

    # Assign each ID a reproducible random color
    rng = np.random.default_rng(random_seed)
    colors = {uid: tuple(map(int, rng.integers(0, 256, size=3))) for uid in unique_ids}

    # Build RGB color mask
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for uid, col in colors.items():
        color_mask[mask == uid] = col

    # Save raw color mask
    out_raw = os.path.splitext(mask_path)[0] + "_color_mask.png"
    cv2.imwrite(out_raw, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved raw color mask to: {out_raw}")

    # Prepare legend entries
    legend_handles = [
        Patch(color=np.array(col)/255.0, label=f"{class_names.get(uid, f'ID {uid}')} ({uid})")
        for uid, col in colors.items()
    ]

    # Plot and save with legend
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(color_mask)
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.15, 0))
    plt.tight_layout()

    out_legend = os.path.splitext(mask_path)[0] + "_color_mask_legend.png"
    plt.savefig(out_legend, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved color mask with legend to: {out_legend}")

if __name__ == "__main__":
    visualize_mask_only(MASK_PATH, CLASS_NAMES)
