import os
import numpy as np
from PIL import Image

# Set your paths (prefer Linux path in WSL)
img_name = "bonirob_2016-05-12-10-26-12_6_frame273.png"
images_dir = "/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/images"
masks_dir  = "/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/semantics"

mask_path = os.path.join(masks_dir, os.path.splitext(img_name)[0] + ".png")
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Mask not found: {mask_path}")

# Color → class-name map (tune if your colors differ)
COLOR_MAP = {
    (0, 0, 0):   "Background",
    (0, 255, 0): "Crop",
    (255, 0, 0): "Weed",
    # add more if your masks use other colors
}

def pack_rgb(arr):
    """Pack HxWx3 uint8 to a single int per pixel."""
    arr = arr.astype(np.uint32)
    return (arr[...,0] << 16) | (arr[...,1] << 8) | arr[...,2]

def unpack_rgb(v):
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)

# Load mask
im = Image.open(mask_path)
arr = np.array(im)

# If single-channel, you only have one class ID space; if 3-channel, pack colors
if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
    ids, counts = np.unique(arr.reshape(-1), return_counts=True)
    print(f"Mask is single-channel. Unique IDs: {ids.tolist()}")
    for i,c in zip(ids.tolist(), counts.tolist()):
        name = {0: "Background", 1: "Crop", 2: "Weed", 3: "Vegetation"}.get(i, "Unknown")
        print(f"ID {i}: {name}  ({c} px)")
else:
    # RGB color-coded
    packed = pack_rgb(arr[...,:3])
    vals, counts = np.unique(packed.reshape(-1), return_counts=True)
    print("Mask is RGB color-coded. Unique colors:")
    for v,cnt in zip(vals.tolist(), counts.tolist()):
        rgb = unpack_rgb(int(v))
        name = COLOR_MAP.get(rgb, "Unknown")
        print(f"{rgb}: {name}  ({cnt} px)")
