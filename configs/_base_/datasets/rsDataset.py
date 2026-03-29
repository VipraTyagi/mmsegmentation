# dataset settings
dataset_type = 'rsDataset' 

data_root = '/home/vipra/Thesis/Semantic_Segmentation/data/Dysphagia_Final'

# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'Bolus'),  # Match original dataset
    palette=[[0, 0, 0], [255, 0, 0]]
)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
     dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.95),
    dict(type='PackSegInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CenterCrop', crop_size=(512, 512)),
    dict(type='PackSegInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# Data loaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
    img_path='images_all',
    seg_map_path='masks_all'
),
            ann_file='splits/train.txt',
            pipeline=train_pipeline,
            metainfo=dataset_meta,
            reduce_zero_label=False
        )
    )



val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
                img_path='images_all',
            seg_map_path='masks_all'),
        ann_file='splits/val.txt',
        pipeline=test_pipeline,
        metainfo=dataset_meta,
        reduce_zero_label=False,
      )   
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
                  img_path='images_all',
            seg_map_path='masks_all'),
        ann_file='splits/test.txt',
        pipeline=test_pipeline,
        metainfo=dataset_meta,
        reduce_zero_label=False))

# Evaluators
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)

