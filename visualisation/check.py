import os
import cv2
import warnings
from PIL import Image, UnidentifiedImageError

# Don’t let PIL silently load truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

data_root = '/home/vipra/Thesis/Semantic_Segmentation/data/cropandweed/SugarBeet_Weed_Vegetation_Soil'
img_dir = os.path.join(data_root, 'images/train')
bad = []

# Turn any PIL warnings into exceptions
warnings.filterwarnings('error')

for fn in sorted(os.listdir(img_dir)):
    full = os.path.join(img_dir, fn)

    # 1) Check that the file exists & is not completely unreadable
    img = cv2.imread(full, cv2.IMREAD_COLOR)
    if img is None:
        bad.append((fn, "cv2.imread returned None"))
        continue

    # 2) Force a full decode with PIL, catching truncated or corrupt images
    try:
        with Image.open(full) as im:
            im.load()   # this is where a truncated JPEG will error out
    except (UnidentifiedImageError, OSError, Warning) as e:
        bad.append((fn, str(e)))

if bad:
    print("🚩 Found corrupted or unreadable images:")
    for fn, reason in bad:
        print(f"  • {fn}: {reason}")
    print(f"\nTotal bad files: {len(bad)}")
else:
    print("✅ All images are present and fully decodable!")
