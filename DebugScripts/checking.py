#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def clean_zone_files(dir_path: Path):
    """Delete any files containing ':Zone.Identifier' in their name."""
    for f in dir_path.iterdir():
        if f.is_file() and ":Zone.Identifier" in f.name:
            print(f"Removing zone file: {f.name}")
            f.unlink()

def gather_basenames(dir_path: Path):
    """Return set of all file stems (i.e. names without extension) in dir."""
    return {f.stem for f in dir_path.iterdir() if f.is_file()}

def move_unmatched(files, target_dir: Path):
    """Move each Path in files into target_dir, creating it if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = target_dir / f.name
        print(f"Moving {f.name} → {target_dir.relative_to(base_dir)}/")
        shutil.move(str(f), str(dest))

if __name__ == "__main__":
    base_dir = Path("/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn")
    images_dir = base_dir / "images"
    semantics_dir = base_dir / "semantics"
    unmatched_images_dir = base_dir / "unmatched/images"
    unmatched_semantics_dir = base_dir / "unmatched/semantics"

    # Step 1: remove Zone.Identifier files
    clean_zone_files(images_dir)
    clean_zone_files(semantics_dir)

    # Step 2: gather basenames and counts
    image_files = [f for f in images_dir.iterdir() if f.is_file()]
    semantics_files = [f for f in semantics_dir.iterdir() if f.is_file()]

    image_basenames = {f.stem for f in image_files}
    semantics_basenames = {f.stem for f in semantics_files}

    total_images = len(image_files)
    total_semantics = len(semantics_files)

    # Step 3: find unmatched
    unmatched_images = [f for f in image_files if f.stem not in semantics_basenames]
    unmatched_semantics = [f for f in semantics_files if f.stem not in image_basenames]

    # Step 4: report
    print(f"Total images (post-clean):    {total_images}")
    print(f"Total semantics (post-clean): {total_semantics}")
    print(f"Images without mask:          {len(unmatched_images)}")
    print(f"Masks without image:          {len(unmatched_semantics)}")

    # Step 5: move unmatched into separate subfolders
    move_unmatched(unmatched_images, unmatched_images_dir)
    move_unmatched(unmatched_semantics, unmatched_semantics_dir)

    print("All done.")
