import os
import numpy as np
from PIL import Image

# Paths
img_path = '/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/images/_2016-05-27-10-26-48_5_frame0.png'
mask_dir = '/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/semantics'

# Mapping (update if you know the correct IDs/names)
CLASS_NAMES = {
    0: "Background",
    1: "Crop",
    2: "Weed",
    3: "Vegetation"
}

# Find corresponding mask
base = os.path.splitext(os.path.basename(img_path))[0]
mask_path = os.path.join(mask_dir, base + ".png")
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Mask not found for {img_path}")

# Load mask
mask = np.array(Image.open(mask_path))

# Get unique label values
unique_vals = np.unique(mask)

# Print results
print(f"Mask path: {mask_path}")
print(f"Unique raw label values: {unique_vals}")

for val in unique_vals:
    class_name = CLASS_NAMES.get(val, "Unknown")
    print(f"Label {val}: {class_name}")
