import os.path as osp
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio

@DATASETS.register_module()
class PhenobenchDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list

    def load_annotations(self, img_path, seg_map_path):
        """Load annotation from png file.
        Args:
            img_path (str): Path to image file.
            seg_map_path (str): Path to segmentation png file.
        Returns:
            dict: The dict contains loaded image and semantic segmentation annotations.
        """
        img_info = dict(filename=img_path)
        seg_map = np.array(Image.open(seg_map_path))
        
        # Convert class 3 to 1 (crop) and class 4 to 2 (weed)
        seg_map[seg_map == 3] = 1
        seg_map[seg_map == 4] = 2
        
        img_info['gt_seg_map'] = seg_map
        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)