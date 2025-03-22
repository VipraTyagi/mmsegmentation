# mmseg/datasets/zurich_dataset.py
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio

@DATASETS.register_module()
class ZurichDataset(BaseSegDataset):
    """UAV Zurich Agricultural Dataset"""
    
    METAINFO = dict(
    classes=('background', 'crop', 'weed'),  
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        # Ensure `reduce_zero_label` is explicitly set
        kwargs.setdefault('reduce_zero_label', False)
        
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs
        )

    def load_data_list(self):
        """Load data list from YAML-specified structure"""
        data_list = []
        img_dir = self.data_prefix['img_path']
        ann_dir = self.data_prefix['seg_map_path']

        # Get sorted list of image files
        img_files = sorted(fileio.list_dir_or_file(
            dir_path=img_dir,
            list_dir=False,
            suffix=self.img_suffix,
            recursive=True
        ))

        # Pair images with annotations using YAML's 1:1 filename mapping
        for img_file in img_files:
            data_info = {
                'img_path': osp.join(img_dir, img_file),
                'seg_map_path': osp.join(ann_dir, img_file),
                'seg_fields': [],
                'reduce_zero_label': False  # Ensure it's always present
            }
            data_list.append(data_info)
            
        return data_list

# def load_annotations(self, img_path, seg_map_path):
#     """Ensure segmentation maps are correctly loaded and remap labels."""
#     gt_seg_map = np.array(Image.open(seg_map_path)).astype(np.uint16)  # Ensure it's read as uint16

#     # Debug: Print original mask values
#     # print(f"Original mask values for {seg_map_path}: {np.unique(gt_seg_map)}")

#   # Fix incorrect label mapping
#     gt_seg_map[gt_seg_map == 10000] = 1  # Convert 10000 → Crop (1)
#     gt_seg_map[gt_seg_map == 2] = 2      # Weed remains 2
#     gt_seg_map[gt_seg_map > 2] = 0       # Ensure no unknown values (set to soil)

#     # Debug: Print fixed mask values
#     print(f"Fixed mask values for {seg_map_path}: {np.unique(gt_seg_map)}")

#     return {
#         'filename': img_path,
#         'gt_seg_map': gt_seg_map.astype(np.uint8)  # Convert back to uint8
    # }
