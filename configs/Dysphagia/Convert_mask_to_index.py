import os
import numpy as np
from PIL import Image

IN_DIR = "/home/vipra/Thesis/Semantic_Segmentation/data/residue_valleculae_100_masks/SegmentationClass/Residue_valleculae_100"
OUT_DIR = "/home/vipra/Thesis/Semantic_Segmentation/data/residue_valleculae_100_masks/Masks"
os.makedirs(OUT_DIR, exist_ok=True)

BOLUS_RGB = (250, 250, 55)

for fname in os.listdir(IN_DIR):
    if not fname.endswith(".png"):
        continue

    rgb = np.array(Image.open(os.path.join(IN_DIR, fname)).convert("RGB"))
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    bolus = (
        (rgb[:, :, 0] == BOLUS_RGB[0]) &
        (rgb[:, :, 1] == BOLUS_RGB[1]) &
        (rgb[:, :, 2] == BOLUS_RGB[2])
    )

    mask[bolus] = 1
    Image.fromarray(mask).save(os.path.join(OUT_DIR, fname))

print("Conversion complete")
