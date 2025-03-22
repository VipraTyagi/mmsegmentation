import os.path as osp
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio

@DATASETS.register_module()
class CropAndWeedDataset(BaseSegDataset):
    METAINFO = dict(
    classes=('background', 'sugarbeet', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [ 255,0, 0]]
)


    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
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
                seg_map = osp.basename(img)[:-4]+self.seg_map_suffix
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

        # Load segmentation mask with explicit uint8 type
        seg_map = np.array(Image.open(seg_map_path).convert("L"), dtype=np.uint8)

        # Debugging: Check original unique values
        print("Before Remapping - Unique values:", np.unique(seg_map))

        # Create a copy to prevent modifying the original mask
        seg_map_fixed = seg_map.copy()

        # Define correct label mapping (manually replacing values)
        seg_map_fixed[seg_map == 255] = 0   # Background → 0
        seg_map_fixed[seg_map == 1] = 1     # Crop (Sugarbeet) → 1
        seg_map_fixed[seg_map == 2] = 2     # Weed → 2

        # Debugging: Check if values are correctly mapped
        print("After Remapping - Unique values:", np.unique(seg_map_fixed))

        # Store corrected segmentation map
        img_info['gt_seg_map'] = seg_map_fixed.astype(np.uint8)  # Ensure dtype

        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)