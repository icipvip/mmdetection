_base_ = '../configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py'

dataset_type = 'CocoDataset'
classes = ('vehicle',)
data_root = 'data/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/labels_cocoformat.json',
        img_prefix=data_root + 'train/images/'
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/labels_cocoformat.json',
        img_prefix=data_root + 'test/images/'
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/labels_cocoformat.json',
        img_prefix=data_root + 'test/images/'
    )
)

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

total_epochs = 5

optimizer = dict(
    lr=0.005
)

lr_config = dict(
    step=[3, 4]
)

test_cfg = dict(
    score_thr=0.4
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(
    create_symlink=False
)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth'
work_dir = 'vipcup/logdir'
