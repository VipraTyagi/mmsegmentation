# dataset settings
dataset_type = 'CustomSegDataset'
data_root = '/mmsegmentation/data/Sugarbeets Naeem'

# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsFromNPZ'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotationsFromNPZ'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_meta,
        data_prefix=dict(
            img_path='main_camera/rect',
            seg_map_path='main_camera_annotations/semantic_segmentation'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_meta,
        data_prefix=dict(
            img_path='main_camera/rect',
            seg_map_path='main_camera_annotations/semantic_segmentation'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator