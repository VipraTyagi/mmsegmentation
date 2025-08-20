#!/usr/bin/env python3
import numpy as np
import torch
from collections import defaultdict, Counter

from mmseg.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmseg.registry import DATASETS

# ====== set your config and split ======
CFG = "/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation/configs/thesisdata/ugvbonn.py"
SPLIT = "val"   # choose: "train", "val", or "test"
MAX_SAMPLES = 500   # or an integer like 200 for a quick check
# =======================================

def pick_dataset_cfg(cfg, split):
    if split == "train":
        return cfg.train_dataloader['dataset']
    elif split == "val":
        return cfg.val_dataloader['dataset']
    elif split == "test":
        return cfg.test_dataloader['dataset']
    else:
        raise ValueError("SPLIT must be one of train, val, test")

def main():
    cfg = Config.fromfile(CFG)
    ds_cfg = pick_dataset_cfg(cfg, SPLIT)
    dataset = DATASETS.build(ds_cfg)

    classes = tuple(dataset.metainfo.get('classes', ()))
    palette = dataset.metainfo.get('palette', None)
    ignore_index = dataset.metainfo.get('ignore_index', 255)

    print("\n=== Dataset metainfo ===")
    print("Split         :", SPLIT)
    print("Num samples   :", len(dataset))
    print("Classes       :", classes)
    print("Ignore index  :", ignore_index)
    if palette is not None:
        print("Palette       :", [tuple(map(int, p)) for p in palette])

    # Show pipeline stages so you can confirm ConvertRGBMaskToLabelID is present
    try:
        stages = [t.__class__.__name__ for t in dataset.pipeline.transforms]
        print("Pipeline      :", " -> ".join(stages))
    except Exception:
        pass

    n = len(dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(dataset))

    # Aggregates
    pixel_totals = Counter()          # label_id -> pixel count
    file_presence = defaultdict(int)  # label_id -> num files containing it
    seen_labels = set()

    for i in range(n):
        if i % 100 == 0:
            print(f"processed {i}/{n}")
        sample = dataset[i]  # after pipeline; returns dict with 'data_samples'
        ds = sample['data_samples']
        # gt_sem_seg.data is a tensor of shape (1, H, W) or (H, W)
        gt = ds.gt_sem_seg.data
        if isinstance(gt, torch.Tensor):
            gt = gt.squeeze().cpu().numpy().astype(np.int64)
        else:
            gt = np.asarray(gt).squeeze().astype(np.int64)

        uniq, counts = np.unique(gt, return_counts=True)
        seen_labels.update(uniq.tolist())
        for u, c in zip(uniq, counts):
            pixel_totals[int(u)] += int(c)
        for u in uniq:
            file_presence[int(u)] += 1

    # Print summary
    print("\n=== Labels after pipeline (what the model sees) ===")
    all_labels = sorted(seen_labels)
    print("Unique label ids seen:", all_labels)

    # Pretty table
    print(f"\n{'ID':>4} | {'Class name':<12} | {'Files w/ id':>10} | {'Pixels':>12}")
    print("-" * 48)
    for lid in all_labels:
        name = None
        if lid == ignore_index:
            name = "ignore"
        elif 0 <= lid < len(classes):
            name = classes[lid]
        else:
            name = "unknown"
        print(f"{lid:>4} | {name:<12} | {file_presence[lid]:10d} | {pixel_totals[lid]:12d}")

    # Optional: per-class totals restricted to valid class ids
    valid_ids = [i for i in range(len(classes))]
    valid_totals = {i: pixel_totals.get(i, 0) for i in valid_ids}
    print("\nPer-class pixel totals (valid ids only):")
    for i in valid_ids:
        print(f"  {i}: {classes[i]} -> {valid_totals[i]}")

if __name__ == "__main__":
    main()
