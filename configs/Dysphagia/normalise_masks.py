import numpy as np
from PIL import Image
from pathlib import Path

# CHANGE NOTHING BELOW UNLESS PATHS ARE DIFFERENT
MASK_ROOT = Path("masks")
OUT_ROOT = Path("masks_binary")
OUT_ROOT.mkdir(exist_ok=True)

# All bolus colors used by you + your friend
BOLUS_COLORS = [
    (250, 250, 55),   # your bolus
    (22, 248, 240),   # friend's bolus
]

for subdir in ["RV_masks", "RP_masks", "Normal_masks"]:
    src_dir = MASK_ROOT / subdir
    dst_dir = OUT_ROOT / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in src_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path))
        out = np.zeros(mask.shape[:2], dtype=np.uint8)

        for color in BOLUS_COLORS:
            match = np.all(mask == color, axis=-1)
            out[match] = 1

        Image.fromarray(out).save(dst_dir / mask_path.name)

print("All masks converted to binary (0=background, 1=bolus)")
