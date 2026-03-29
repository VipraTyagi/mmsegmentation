import os
import numpy as np
from PIL import Image

BOLUS_COLOR = (42,125,209)   # from your labelmap.txt

in_dir = "data/raw/masks_color"
out_dir = "data/processed/masks_binary"
os.makedirs(out_dir, exist_ok=True)

for f in os.listdir(in_dir):
    img = np.array(Image.open(os.path.join(in_dir,f)).convert("RGB"))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    bolus = (img[:,:,0]==42)&(img[:,:,1]==125)&(img[:,:,2]==209)
    mask[bolus] = 1

    Image.fromarray(mask).save(os.path.join(out_dir,f))
    print(f"Converted {f}")