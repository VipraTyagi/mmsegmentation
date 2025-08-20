#!/usr/bin/env python3
# triage_localize_dropout_v2.py
#
# Pinpoint the first transform in the TRAIN pipeline that causes class 2 to disappear.
# Handles stages where GT is not yet available without crashing.

import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image

from mmseg.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmseg.registry import DATASETS

try:
    from mmseg.datasets.transforms import ConvertRGBMaskToLabelID  # noqa: F401
except Exception:
    ConvertRGBMaskToLabelID = None

# --------------- Settings --------------- #
CFG = "/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation/configs/thesisdata/ugvbonn.py"
CLASS_ID = 2
SEED = 7
MAX_RAW_SCAN = 200
MAX_TRACE = 10
SHUFFLE_RAW = True
PRINT_PARAMS = True
# --------------------------------------- #

def seed_for_index(base_seed: int, i: int) -> None:
    s = base_seed + int(i)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
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

def extract_ids_from_data_sample(s):
    """Return sorted unique ids from SegDataSample or None if GT not present yet."""
    try:
        seg = s.gt_sem_seg  # may raise if _gt_sem_seg not set
    except Exception:
        return None
    if seg is None:
        return None
    gt = getattr(seg, 'data', None)
    if gt is None:
        return None
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze().detach().cpu().numpy()
    else:
        gt = np.asarray(gt).squeeze()
    return np.unique(gt).tolist()

def ids_present_from_dataset(ds, i):
    """Deterministic fetch of ids; returns list or None if GT not available at this stage."""
    seed_for_index(SEED, i)
    s = ds[i]['data_samples']
    return extract_ids_from_data_sample(s)

def build_noaug_dataset(ds_cfg_train):
    ds_cfg = deepcopy(ds_cfg_train)
    ds_cfg['pipeline'] = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ConvertRGBMaskToLabelID', ignore_index=255),
        dict(type='PackSegInputs'),
    ]
    return DATASETS.build(ds_cfg)

def pipeline_list(ds_cfg_train):
    return deepcopy(ds_cfg_train['pipeline'])

def ensure_packseginputs(pipe):
    has_pack = any(step.get('type') == 'PackSegInputs' for step in pipe)
    if not has_pack:
        pipe.append(dict(type='PackSegInputs'))
    return pipe

def build_stage_dataset(ds_cfg_train, k):
    base = deepcopy(ds_cfg_train)
    full = pipeline_list(ds_cfg_train)
    pipe = full[:k]
    pipe = ensure_packseginputs(pipe)
    base['pipeline'] = pipe
    return DATASETS.build(base), pipe

def print_pipeline_overview(ds_cfg_train):
    full = pipeline_list(ds_cfg_train)
    print("\nTRAIN pipeline overview:")
    for i, t in enumerate(full, 1):
        print(f"{i:02d}. {t.get('type', 'Unknown')}")

def find_weed_positive_indices_raw(ds_train, limit, shuffle=True):
    N = len(ds_train)
    all_idxs = list(range(N))
    if shuffle:
        random.Random(SEED).shuffle(all_idxs)
    subset = all_idxs[:min(limit, N)]
    print(f"Scanning RAW masks for weed on {len(subset)} of {N} train entries...")
    weed_idxs = []
    for i in subset:
        p = get_seg_path(ds_train, i)
        if not p:
            continue
        arr = np.array(Image.open(p))
        if arr.ndim == 2:
            ids = arr.astype(np.uint8)
        else:
            ids = rgb_to_ids(np.array(Image.open(p).convert('RGB')))
        if (ids == CLASS_ID).any():
            weed_idxs.append(i)
    print(f"Found {len(weed_idxs)} weed-positive entries in the scanned subset.")
    return weed_idxs

def classify_kept_vs_gone(ds_noaug, ds_train, idxs, class_id):
    kept, gone = [], []
    for i in idxs:
        ids_noaug = ids_present_from_dataset(ds_noaug, i)
        ids_train = ids_present_from_dataset(ds_train, i)
        # ids_noaug should not be None with the fixed no-aug pipeline
        if ids_noaug is None:
            continue
        if ids_train is None:
            # If full TRAIN returns None, treat as gone for triage
            gone.append(i)
        elif class_id in ids_noaug and class_id not in ids_train:
            gone.append(i)
        else:
            kept.append(i)
    return kept, gone

def trace_dropout_for_index(ds_cfg_train, idx, class_id, print_params=True):
    """Stage-by-stage trace for one index.
    Returns (first_irreversible_k, first_irreversible_name) or (None, None).
    Prints the whole pipeline; marks temporary vs irreversible drops.
    """
    full = pipeline_list(ds_cfg_train)
    stage_names = [t.get('type', 'Unknown') for t in full]
    ids_by_stage = []

    print(f"\n=== Stage-by-stage ids for sample index {idx} ===")
    # Collect ids after each stage
    for k in range(1, len(full) + 1):
        ds_k, _ = build_stage_dataset(ds_cfg_train, k)
        seed_for_index(SEED, idx)
        ids_k = ids_present_from_dataset(ds_k, idx)  # list or None
        ids_by_stage.append(ids_k)
        tcfg = full[k - 1]
        tname = tcfg.get('type', 'Unknown')
        if ids_k is None:
            print(f"{k:02d}. {tname:>24} -> GT not available yet")
        else:
            print(f"{k:02d}. {tname:>24} -> {ids_k}")
            if print_params:
                compact = {kk: vv for kk, vv in tcfg.items() if kk != 'transforms'}
                print(f"     params: {compact}")

    # Decide where to start evaluating presence:
    # Prefer to start at or after ConvertRGBMaskToLabelID if present.
    start_k = index_of_stage_type(full, "ConvertRGBMaskToLabelID")
    if start_k is None:
        # else start from the first stage where the class is actually present
        start_k = first_stage_with_class(ids_by_stage, class_id)
    else:
        # include the mapping stage itself
        pass
    if start_k is None:
        print("Result: class never appears in any stage, cannot localize a drop.")
        return None, None

    # Find the first irreversible drop: first k >= start_k where class is absent
    # and never reappears in any later stage.
    first_irrev = None
    for k in range(start_k, len(ids_by_stage)):
        ids_k = ids_by_stage[k]
        if ids_k is None:
            continue
        if class_id not in ids_k:
            later_has = any(
                (ids_j is not None) and (class_id in ids_j)
                for ids_j in ids_by_stage[k + 1 :]
            )
            if not later_has:
                first_irrev = k
                break

    if first_irrev is None:
        print("Result: class present at the end of the pipeline (no irreversible drop).")
        return None, None

    fail_name = stage_names[first_irrev]
    print(f"Result: first irreversible drop at stage {first_irrev + 1} [{fail_name}].")
    return first_irrev + 1, fail_name
def index_of_stage_type(pipeline_steps, type_name: str):
    """Return zero-based index of the first step whose 'type' matches type_name, else None."""
    for i, t in enumerate(pipeline_steps):
        if t.get('type') == type_name:
            return i
    return None


def first_stage_with_class(ids_by_stage, class_id: int):
    """Return zero-based index of first stage where class_id appears."""
    for i, ids in enumerate(ids_by_stage):
        if ids is not None and class_id in ids:
            return i
    return None
def summarize_failures(fail_records):
    from collections import Counter
    names = [name for (_, _, name) in fail_records if name is not None]
    if not names:
        print("\nNo first-failure transform to summarize.")
        return
    c = Counter(names)
    print("\n=== Summary of first failing transform across traced indices ===")
    for name, cnt in c.most_common():
        print(f"{name:>24} : {cnt}")

def main():
    cfg = Config.fromfile(CFG)
    ds_cfg_train = cfg.train_dataloader['dataset']
    ds_train = DATASETS.build(ds_cfg_train)
    ds_noaug = build_noaug_dataset(ds_cfg_train)

    # Determinism sanity check
    try:
        seed_for_index(SEED, 0)
        ids1 = ids_present_from_dataset(ds_train, 0)
        seed_for_index(SEED, 0)
        ids2 = ids_present_from_dataset(ds_train, 0)
        print("Determinism check on train[0]:", ids1, "|", ids2)
    except Exception as e:
        print("Determinism check failed:", repr(e))

    print_pipeline_overview(ds_cfg_train)

    weed_idxs = find_weed_positive_indices_raw(ds_train, MAX_RAW_SCAN, SHUFFLE_RAW)
    if not weed_idxs:
        print("No weed found in the sampled subset. Increase MAX_RAW_SCAN or change SEED.")
        return

    kept, gone = classify_kept_vs_gone(ds_noaug, ds_train, weed_idxs, CLASS_ID)
    print(f"\nChecked {len(weed_idxs)} weed-positive entries (same indices):")
    print(f"  After NO-AUG + ConvertRGBMaskToLabelID : expected present")
    print(f"  After TRAIN pipeline                    : weed kept in {len(kept)} | missing in {len(gone)}")

    if not gone:
        print("\nNo indices lost class after TRAIN in the scanned subset.")
        return

    rng = random.Random(SEED)
    rng.shuffle(gone)
    to_trace = gone[:min(MAX_TRACE, len(gone))]
    print(f"\nTracing up to {len(to_trace)} indices where TRAIN dropped class {CLASS_ID}...")

    fail_records = []
    for idx in to_trace:
        k, name = trace_dropout_for_index(ds_cfg_train, idx, CLASS_ID, print_params=PRINT_PARAMS)
        fail_records.append((idx, k, name))

    summarize_failures(fail_records)

    print("\n=== First failing stage per traced index ===")
    for idx, k, name in fail_records:
        if k is None:
            print(f"idx {idx:6d} : class still present after full pipeline")
        else:
            print(f"idx {idx:6d} : first missing at stage {k:02d} [{name}]")

if __name__ == "__main__":
    main()
