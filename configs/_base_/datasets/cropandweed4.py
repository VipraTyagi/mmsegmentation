# dataset settings
dataset_type = 'CropAndWeedfourDataset'
data_root = '/home/vipra/Thesis/Semantic_Segmentation/data/cropandweed/SugarBeet_Weed_Vegetation_Soil'

# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'sugarbeet', 'weed', 'vegetation'),
   palette=[[0, 0, 0], [0, 255, 0], [ 255,0, 0], [0, 0, 255]] 
)


train_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations'),   
    dict(type='CropAndWeedMapping'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CropAndWeedMapping'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks_remap/train'),
    #    ann_file='splits/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        
        data_prefix=dict(
             img_path='images/val',
            seg_map_path='masks_remap/val'),
            #  ann_file='splits/val.txt',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='masks_remap/test'),
        # ann_file='splits/test.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mFscore'], classwise=True,F1_score=True)
test_evaluator = val_evaluator