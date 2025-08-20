# save as /home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/scan_labels.py
import os, collections, numpy as np
from PIL import Image

MASKS = "/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/semantics"
EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def pack_rgb(arr):
    arr = arr.astype(np.uint32)
    return (arr[...,0] << 16) | (arr[...,1] << 8) | arr[...,2]

def load_mask_norm(path):
    im = Image.open(path)
    mode = im.mode
    arr = np.array(im)
    if mode == "P":
        ids = arr.astype(np.int64)
        kind = "id"
    elif mode.startswith("I;16") or mode == "I":
        a16 = arr.astype(np.uint16)
        vals = a16.reshape(-1)
        if vals.size and (np.mean((vals % 256) == 0) > 0.95):
            ids = (a16 // 256).astype(np.int64)  # normalize id*256 -> id
            kind = "id16norm"
        else:
            ids = a16.astype(np.int64)
            kind = "id16raw"
    elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        ids = arr[...,0].astype(np.int64) if arr.ndim == 3 else arr.astype(np.int64)
        kind = "id"
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        ids = pack_rgb(arr[...,:3]).astype(np.int64)
        kind = "rgb"
    else:
        ids = arr.reshape(-1).astype(np.int64)
        kind = "unknown"
    return ids, kind, mode, arr.shape

files = [os.path.join(MASKS,f) for f in sorted(os.listdir(MASKS)) if os.path.splitext(f)[1].lower() in EXTS]
if not files:
    print("No mask files found.")
    raise SystemExit

mode_counts = collections.Counter()
kind_counts = collections.Counter()
global_ids = collections.Counter()
rgb_examples = collections.Counter()  # packed int -> count
non_binary_files = []

for p in files:
    ids, kind, mode, shape = load_mask_norm(p)
    mode_counts[mode] += 1
    kind_counts[kind] += 1
    u, c = np.unique(ids, return_counts=True)
    if kind == "rgb":
        for val, cnt in zip(u.tolist(), c.tolist()):
            rgb_examples[val] += cnt
    else:
        for val, cnt in zip(u.tolist(), c.tolist()):
            global_ids[val] += cnt
        # record files that have values other than {0,255}
        if not set(u.tolist()).issubset({0, 255}):
            non_binary_files.append((os.path.basename(p), u.tolist()))

print("\n=== Dataset summary ===")
print(f"Total masks: {len(files)}")
print("PIL modes:", dict(mode_counts))
print("Kinds:", dict(kind_counts))

if kind_counts.get("rgb",0) > 0:
    print("\nUnique RGB colors across dataset (top 20 by pixels):")
    top = rgb_examples.most_common(20)
    for k,cnt in top:
        r = (k >> 16) & 255; g = (k >> 8) & 255; b = k & 255
        print(f"  #{k:06X} -> ({r},{g},{b}) pixels={cnt}")

if global_ids:
    print("\nUnique integer IDs across dataset (sorted):")
    ids_sorted = sorted(global_ids.items(), key=lambda x: x[0])
    preview = ", ".join([f"{k}:{v}" for k,v in ids_sorted[:15]])
    print(preview + (" ..." if len(ids_sorted)>15 else ""))

if non_binary_files:
    print("\nFiles with values other than {0,255}:")
    for name, vals in non_binary_files[:25]:
        print(f"  {name} -> {sorted(set(vals))}")
    if len(non_binary_files) > 25:
        print(f"  ... and {len(non_binary_files)-25} more")
else:
    print("\nEvery scanned mask only contains {0,255}. No third label found in these files.")
