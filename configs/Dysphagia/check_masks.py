
import os
import shutil
import numpy as np
from PIL import Image

IMG_DIR = "data/Residue_valleculae_100"
REAL_MASK_DIR = "data/residue_valleculae_100_masks/Masks"

OUT_MASK_DIR = "data/swallow_full/masks"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

real_masks = set(f for f in os.listdir(REAL_MASK_DIR) if f.endswith(".png"))

count_real = 0
count_empty = 0

for img in os.listdir(IMG_DIR):
    if not img.endswith(".jpg"):
        continue

    base = img.replace(".jpg", "")
    mask_name = base + ".png"
    out_path = os.path.join(OUT_MASK_DIR, mask_name)

    if mask_name in real_masks:
        shutil.copy(
            os.path.join(REAL_MASK_DIR, mask_name),
            out_path
        )
        count_real += 1
    else:
        img_path = os.path.join(IMG_DIR, img)
        w, h = Image.open(img_path).size
        empty = np.zeros((h, w), dtype=np.uint8)
        Image.fromarray(empty).save(out_path)
        count_empty += 1

print("Real masks:", count_real)
print("Empty masks created:", count_empty)

