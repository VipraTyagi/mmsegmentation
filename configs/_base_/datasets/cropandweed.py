# dataset settings
dataset_type = 'CropAndWeedDataset'
data_root = '/home/vipra/Thesis/Semantic_Segmentation/data/cropandweed'



# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'sugarbeet', 'weed'),
   palette=[[0, 0, 0], [0, 255, 0], [ 255,0, 0]]
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            img_path='Sugarbeet/train/images',
            seg_map_path='Sugarbeet/train/masks'),
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
             img_path='Sugarbeet/val/images',
            seg_map_path='Sugarbeet/val/masks'),
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
            img_path='Sugarbeet/test/images',
            seg_map_path='Sugarbeet/test/masks'),
        # ann_file='splits/test.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator