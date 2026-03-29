from pathlib import Path
import random
import numpy as np
from PIL import Image

# Paths
IMAGES_DIR = Path("images_all")
MASKS_DIR = Path("masks_all")
SPLITS_DIR = Path("splits")
SPLITS_DIR.mkdir(exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# Collect all samples
samples = []

for img_path in IMAGES_DIR.iterdir():
    if not img_path.is_file():
        continue

    mask_path = MASKS_DIR / f"{img_path.stem}.png"
    if not mask_path.exists():
        raise RuntimeError(f"Missing mask for {img_path.name}")

    mask = np.array(Image.open(mask_path))
    has_bolus = bool(np.any(mask == 1))

    samples.append({
        "name": img_path.stem,
        "has_bolus": has_bolus
    })

# Separate positives and negatives
positives = [s for s in samples if s["has_bolus"]]
negatives = [s for s in samples if not s["has_bolus"]]

random.seed(42)
random.shuffle(positives)
random.shuffle(negatives)

def split_list(items):
    n = len(items)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

pos_train, pos_val, pos_test = split_list(positives)
neg_train, neg_val, neg_test = split_list(negatives)

train = pos_train + neg_train
val = pos_val + neg_val
test = pos_test + neg_test

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

def write_split(name, items):
    with open(SPLITS_DIR / name, "w") as f:
        for item in items:
            f.write(item["name"] + "\n")

write_split("train.txt", train)
write_split("val.txt", val)
write_split("test.txt", test)

print("===== SPLIT SUMMARY =====")
print(f"Total samples     : {len(samples)}")
print(f"Train             : {len(train)}")
print(f"Val               : {len(val)}")
print(f"Test              : {len(test)}")
print()
print("Bolus distribution:")
print(f"  Train positives : {sum(s['has_bolus'] for s in train)}")
print(f"  Val positives   : {sum(s['has_bolus'] for s in val)}")
print(f"  Test positives  : {sum(s['has_bolus'] for s in test)}")
