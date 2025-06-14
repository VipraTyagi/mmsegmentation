import os
import shutil
import random
from collections import defaultdict

# Paths (adjust to your environment)
source_root = "/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/PhenoBench/train"
destination_root = "/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/phenobenchCL/train5"

# Selection criteria: date (MM-DD) -> number of samples
date_selection = {
    "05-15": 23,
    "05-26": 23,
    "06-05": 24,  # example; adjust as needed
}

# Step 1: Group image filenames by date prefix
def collect_images_by_date(images_folder):
    grouped = defaultdict(list)
    for fname in os.listdir(images_folder):
        if fname.endswith(".png"):
            date = fname.split("_", 1)[0]  # extracts 'MM-DD'
            grouped[date].append(fname)
    return grouped

# Step 2: Copy sampled images and their semantics
def copy_images_and_semantics(images_by_date, src_root, dst_root, selection_map):
    for date_prefix, count in selection_map.items():
        files = images_by_date.get(date_prefix, [])
        if not files:
            print(f"No images found for date {date_prefix}")
            continue

        selected = random.sample(files, min(count, len(files)))
        print(f"Selected {len(selected)} files from {date_prefix}")

        for fname in selected:
            for sub in ["images", "semantics"]:
                src = os.path.join(src_root, sub, fname)
                dst_dir = os.path.join(dst_root, sub)
                dst = os.path.join(dst_dir, fname)

                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: missing {src}")

# Run the process
images_folder = os.path.join(source_root, "images")
images_by_date = collect_images_by_date(images_folder)
copy_images_and_semantics(images_by_date, source_root, destination_root, date_selection)
