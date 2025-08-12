# mmseg/datasets/ugvbonn_dataset.py
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class UgvBonnDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'sugarbeet', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('reduce_zero_label', False)
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs
        )

    def get_seg_map(self, idx):
        """Load RGB mask and convert to integer label map."""
        seg_map_path = self.data_list[idx]['seg_map_path']
        mask_rgb = np.array(Image.open(seg_map_path).convert('RGB'))

        # Mapping RGB colors to class IDs
        color_to_id = {
            (0, 0, 0): 0,       # background
            (0, 255, 0): 1,     # crop
            (255, 0, 0): 2      # weed
        }

        mask_id = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for color, class_id in color_to_id.items():
            mask_id[np.all(mask_rgb == color, axis=-1)] = class_id

        return mask_id
