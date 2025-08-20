# dataset settings
dataset_type = 'mixed_data' 

# data_root = '/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/PhenoBench/'

# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'sugarbeet', 'weed'),  # Match original dataset
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
)


# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
     dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),                            # ← load mask first
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='CenterCrop', crop_size=(512, 512)),           # now both are 512×512
    dict(type='PackSegInputs')
]

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
      dataset = dict(
        type='ConcatDataset',
        datasets=[
            # ------- Domain-1  (PhenoBench) -----------------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/phenobenchCL',
                data_prefix=dict(
                    img_path='train/images',      
                    seg_map_path='train/semantics'  # sub-folder with masks
                ),
                pipeline        = train_pipeline,    # ← use SAME pipeline
                metainfo        = dataset_meta,
                reduce_zero_label=False
            ),
            # ------- Domain-2  (Zurich) ---------------------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn',
                data_prefix=dict(
                    img_path='images/train',         
                    seg_map_path='masks/train'        # separate train split, point there
                ),
                pipeline        = train_pipeline,
                metainfo        = dataset_meta,
                reduce_zero_label=False)
            # ,
            # dict(
            #     type='mixed_data',
            #     data_root='/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn',
            #     data_prefix=dict(
            #         img_path='images/train',         
            #         seg_map_path='masks/train'        # separate train split, point there
            #     ),
            #     pipeline        = train_pipeline,
            #     metainfo        = dataset_meta,
            #     reduce_zero_label=False
            # )
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
     dataset=dict(
        type='ConcatDataset',
        datasets=[
            # ---------- Domain 1  (PhenoBench) ----------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/PhenoBench',
                data_prefix=dict(
                    img_path='val/images',
                    seg_map_path='val/semantics'          # update if mask path differs
                ),
                pipeline=test_pipeline,
                metainfo=dataset_meta,
                reduce_zero_label=False
            ),
            # ---------- Domain 2  (Zurich) --------------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn',
                data_prefix=dict(
                    img_path='images/val',
                    seg_map_path='masks/val'
                ),
                pipeline=test_pipeline,
                metainfo=dataset_meta,
                reduce_zero_label=False
            )
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
     dataset=dict(
        type='ConcatDataset',
        datasets=[
            # ---------- Domain 1  (PhenoBench) ----------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/PhenoBench/PhenoBench',
                data_prefix=dict(
                    img_path='test/images',
                    seg_map_path='test/semantics'          # update if mask path differs
                ),
                pipeline=test_pipeline,
                metainfo=dataset_meta,
                reduce_zero_label=False
            ),
            # ---------- Domain 2  (Zurich) --------------
            dict(
                type='mixed_data',
                data_root='/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn',
                data_prefix=dict(
                    img_path='images/val',
                    seg_map_path='masks/val'
                ),
                pipeline=test_pipeline,
                metainfo=dataset_meta,
                reduce_zero_label=False
            )
        ]))

# Evaluators
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


