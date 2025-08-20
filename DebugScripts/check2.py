import os, numpy as np
from PIL import Image
from mmseg.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmseg.registry import DATASETS, TRANSFORMS

 

CFG  = "/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation/configs/thesisdata/ugvbonn.py"
USE_TEST_SPLIT = False

cfg = Config.fromfile(CFG)
ds_cfg = (cfg.test_dataloader if USE_TEST_SPLIT else cfg.val_dataloader)['dataset']
dataset = DATASETS.build(ds_cfg)

to_ids = TRANSFORMS.build(dict(type='ConvertRGBMaskToLabelID'))

tot = np.zeros(3, dtype=np.int64)
for i in range(len(dataset)):
    info = dataset.get_data_info(i)
    mask_path = info['seg_map_path']
    rgb = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8)
    ids = to_ids.transform({'gt_seg_map': rgb})['gt_seg_map']
    for c in (0,1,2):
        tot[c] += int((ids == c).sum())

print("\nGT pixel totals on this split (after conversion):")
print(f"background: {tot[0]}")
print(f"sugarbeet:  {tot[1]}")
print(f"weed:       {tot[2]}")
