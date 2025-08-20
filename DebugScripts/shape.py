from mmengine.config import Config
from data.ugvbonn.check import CFG
from mmseg.registry import DATASETS

cfg = Config.fromfile(CFG)
val_ds = DATASETS.build(cfg.val_dataloader['dataset'])
s = val_ds[3]['data_samples']
gt = s.gt_sem_seg.data.squeeze().cpu().numpy()
print("img_shape:", s.img_shape, "ori_shape:", s.ori_shape)
print("gt shape :", gt.shape)  # must equal s.img_shape
