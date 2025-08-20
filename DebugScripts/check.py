# /home/vipra/Thesis/Semantic_Segmentation/tools/triage_weeds_after_pipeline.py
import os, random
import numpy as np
import torch
from PIL import Image
from copy import deepcopy

from mmseg.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmseg.registry import DATASETS

# IMPORTANT: ensure the converter class is registered
try:
    from mmseg.datasets.transforms import ConvertRGBMaskToLabelID  # noqa: F401
except Exception:
    ConvertRGBMaskToLabelID = None  # ok if already registered elsewhere

CFG = "/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation/configs/thesisdata/ugvbonn.py"

SEED = 7
MAX_RAW_SCAN = 200   # scan only this many train entries for raw weed presence
MAX_CHECK    = 50    # among weed-positive indices, verify at most this many
SHUFFLE_RAW  = True  # shuffle before taking the 200-sample subset

def rgb_to_ids(rgb):
    ids = np.full(rgb.shape[:2], 255, np.uint8)
    ids[(rgb == [0,   0,   0]).all(-1)] = 0
    ids[(rgb == [0, 255,   0]).all(-1)] = 1
    ids[(rgb == [255, 0,   0]).all(-1)] = 2
    return ids

def get_seg_path(ds, i):
    info = ds.get_data_info(i)
    return (
        info.get('seg_map_path')
        or info.get('seg_map')
        or (info.get('ann', {}) or {}).get('seg_map')
    )

def ids_present_from_dataset(ds, i):
    """Return sorted unique ids from the post-pipeline gt for sample i."""
    s = ds[i]['data_samples']
    gt = s.gt_sem_seg.data
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze().cpu().numpy()
    else:
        gt = np.asarray(gt).squeeze()
    return np.unique(gt).tolist()

def main():
    cfg = Config.fromfile(CFG)

    # Build ACTUAL train dataset (with your train pipeline)
    ds_cfg_train = cfg.train_dataloader['dataset']
    ds_train = DATASETS.build(ds_cfg_train)
    N = len(ds_train)

    # === Quick check: exactly your snippet adapted to ds_train ===
    try:
        s = ds_train[0]['data_samples']
        gt = s.gt_sem_seg.data.squeeze().cpu().numpy()
        print("Quick check train[0] ids present:", np.unique(gt))
    except Exception as e:
        print("Quick check train[0] failed:", repr(e))

    # Subset for raw scan
    all_idxs = list(range(N))
    if SHUFFLE_RAW:
        random.Random(SEED).shuffle(all_idxs)
    subset = all_idxs[:min(MAX_RAW_SCAN, N)]
    print(f"Scanning RAW masks for weed on {len(subset)} of {N} train entries...")

    # Find indices with weed in RAW masks (no pipeline)
    weed_idxs = []
    for i in subset:
        p = get_seg_path(ds_train, i)
        if p is None:
            continue
        arr = np.array(Image.open(p))
        ids = arr.astype(np.uint8) if arr.ndim == 2 else rgb_to_ids(np.array(Image.open(p).convert('RGB')))
        if (ids == 2).any():
            weed_idxs.append(i)
    print(f"Found {len(weed_idxs)} weed-positive entries in the scanned subset.")
    if not weed_idxs:
        print("No weed found in the sampled subset. Increase MAX_RAW_SCAN or change SEED.")
        return

    # Build a NO-AUG dataset that only converts RGB->IDs (isolates mapping)
    ds_cfg_noaug = deepcopy(ds_cfg_train)
    ds_cfg_noaug['pipeline'] = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ConvertRGBMaskToLabelID', ignore_index=255),
        dict(type='PackSegInputs'),
    ]
    ds_noaug = DATASETS.build(ds_cfg_noaug)

    # === Quick check: same snippet for ds_noaug ===
    try:
        s2 = ds_noaug[0]['data_samples']
        gt2 = s2.gt_sem_seg.data.squeeze().cpu().numpy()
        print("Quick check noaug[0] ids present:", np.unique(gt2))
    except Exception as e:
        print("Quick check noaug[0] failed:", repr(e))

    # Pick a small set of weed-positive indices to verify after both pipelines
    rng = random.Random(SEED)
    rng.shuffle(weed_idxs)
    weed_idxs = weed_idxs[:min(MAX_CHECK, len(weed_idxs))]

    kept_noaug, gone_noaug = [], []
    kept_train, gone_train = [], []

    # Aggregate which IDs appear across the checked set
    agg_ids_noaug = set()
    agg_ids_train = set()

    for i in weed_idxs:
        ids_noaug = ids_present_from_dataset(ds_noaug, i)
        ids_train = ids_present_from_dataset(ds_train, i)

        agg_ids_noaug.update(ids_noaug)
        agg_ids_train.update(ids_train)

        if 2 in ids_noaug:
            kept_noaug.append(i)
        else:
            gone_noaug.append(i)

        if 2 in ids_train:
            kept_train.append(i)
        else:
            gone_train.append(i)

    print(f"\nChecked {len(weed_idxs)} weed-positive entries (same indices):")
    print(f"  After NO-AUG + ConvertRGBMaskToLabelID : weed present in {len(kept_noaug)} | missing in {len(gone_noaug)}")
    print(f"  After TRAIN pipeline                    : weed present in {len(kept_train)} | missing in {len(gone_train)}")
    print(f"\nAggregate ids after NO-AUG   : {sorted(agg_ids_noaug)}")
    print(f"Aggregate ids after TRAIN    : {sorted(agg_ids_train)}")

    # Show a few examples and per-sample id sets so you can inspect
    def show_examples(tag, dsA, dsB, idxs, limit=5):
        print(f"\nExamples {tag}:")
        for i in idxs[:limit]:
            p = get_seg_path(ds_train, i)
            idsA = ids_present_from_dataset(dsA, i)
            idsB = ids_present_from_dataset(dsB, i)
            print(" ", p)
            print("    ids NO-AUG :", idsA)
            print("    ids TRAIN  :", idsB)

    show_examples("NO-AUG MISSING (mapping problem)", ds_noaug, ds_train, gone_noaug)
    show_examples("TRAIN MISSING (augmentation problem)", ds_noaug, ds_train, gone_train)

    # === Extra: check the first weed-positive index deterministically ===
    try:
        idx = weed_idxs[0]
        s_train = ds_train[idx]['data_samples']
        s_noaug = ds_noaug[idx]['data_samples']
        gt_train = s_train.gt_sem_seg.data.squeeze().cpu().numpy()
        gt_noaug = s_noaug.gt_sem_seg.data.squeeze().cpu().numpy()
        print(f"\nFirst weed idx {idx}: ids train={np.unique(gt_train)} | ids noaug={np.unique(gt_noaug)}")
    except Exception as e:
        print("\nExtra check failed:", repr(e))

if __name__ == "__main__":
    main()
