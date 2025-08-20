import os
import shutil
import random

# Dataset paths
DATASET_DIR = "/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn"
OUTPUT_DIR = "/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1) < 1e-6, "Splits must sum to 1"

# Source directories
images_dir = os.path.join(DATASET_DIR, "images")
masks_dir = os.path.join(DATASET_DIR, "semantics")  # <-- RGB masks

# Output directories
train_images_dir = os.path.join(OUTPUT_DIR, "images", "train")
train_masks_dir = os.path.join(OUTPUT_DIR, "semantics", "train")
val_images_dir = os.path.join(OUTPUT_DIR, "images", "val")
val_masks_dir = os.path.join(OUTPUT_DIR, "semantics", "val")
test_images_dir = os.path.join(OUTPUT_DIR, "images", "test")
test_masks_dir = os.path.join(OUTPUT_DIR, "semantics", "test")

# Create output directories
for folder in [train_images_dir, train_masks_dir,
               val_images_dir, val_masks_dir,
               test_images_dir, test_masks_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Compute splits
total_files = len(image_files)
train_split = int(total_files * TRAIN_RATIO)
val_split = int(total_files * (TRAIN_RATIO + VAL_RATIO))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# Function to copy images and masks
def copy_files(file_list, src_img_dir, src_mask_dir, tgt_img_dir, tgt_mask_dir):
    for file_name in file_list:
        img_src = os.path.join(src_img_dir, file_name)
        mask_src = os.path.join(src_mask_dir, file_name)  # Same filename for mask
        img_dest = os.path.join(tgt_img_dir, file_name)
        mask_dest = os.path.join(tgt_mask_dir, file_name)

        if os.path.exists(img_src) and os.path.exists(mask_src):
            shutil.copy(img_src, img_dest)
            shutil.copy(mask_src, mask_dest)
        else:
            print(f"⚠ Missing image or mask for {file_name}")

# Copy files
copy_files(train_files, images_dir, masks_dir, train_images_dir, train_masks_dir)
copy_files(val_files, images_dir, masks_dir, val_images_dir, val_masks_dir)
copy_files(test_files, images_dir, masks_dir, test_images_dir, test_masks_dir)

print(f"✅ Dataset split complete:")
print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
