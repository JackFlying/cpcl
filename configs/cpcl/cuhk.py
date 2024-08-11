_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid_norm_unsu.py',
    '../_base_/datasets/coco_reid_unsup.py',
    '../_base_/schedules/schedule_1x_reid_norm_base.py', '../_base_/default_runtime.py'
]
TEST = False
USE_PART_FEAT = False
GLOBAL_WEIGHT = 0.8
UNCERTAINTY = False
USE_GFN = False
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        strides=(1, 2, 2, 1),
        num_stages=4,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNs16C45add',
        in_channels=[512, 1024, 2048],
        out_channels=1024,
        use_dconv=True,
        kernel1=True),
    roi_head=dict(
        type='ReidRoIHeadCPCL',
        use_gfn=USE_GFN,
        use_RoI_Align_feat=False,
        use_part_feat=USE_PART_FEAT,
        scene_emb_size=56,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='DeformRoIPoolPack',
                output_size=tuple([14, 6]),
                output_channels=1024),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            in_channels=1024,
            testing=TEST,
            type='CPCLHead',
            id_num=55272,
            rcnn_bbox_bn=False,
            cluster_top_percent=0.6,
            instance_top_percent=1.0,
            use_quaduplet_loss=True,
            use_cluster_hard_loss=True,
            norm_type='l2norm',    # ['l2norm', 'protonorm', 'batchnorm']
            use_bn_affine=False,
            update_method='max_iou',    # ['momentum', 'iou', 'max_iou', 'momentum_max_iou']
            use_max_IoU_bbox=False,
            momentum=0.2,
            cluster_mean_method='time_consistency',    # ['naive', 'intra_cluster', 'time_consistency', 'intra_cluster_time_consistency', 'latest']
            tc_winsize=500,
            decay_weight=-0.0005,
            tc_method='linear', # ['linear', 'concave_function', 'convex_function']
            intra_cluster_T=0.1,
            use_part_feat=USE_PART_FEAT,
            global_weight= GLOBAL_WEIGHT if USE_PART_FEAT else 1,
            triplet_instance_weight=1,
            num_features=256,
            margin=0.3,
            loss_bbox=dict(type='L1Loss', loss_weight=1),
            loss_reid=dict(loss_weight=1.0),
        )
    )
)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', 
        img_scale=[(667, 400),(1000, 600), (1333, 800), (1500,900), (1666, 1000), (2000, 1200)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        # img_scale=(1666, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
query_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        # img_scale=(1666, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['proposals']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline, query_test_pipeline=None),
    train_cluster=dict(pipeline=val_pipeline, query_test_pipeline=None),
    val=dict(pipeline=test_pipeline, query_test_pipeline=None),
    test=dict(pipeline=test_pipeline,
        query_test_pipeline=query_test_pipeline,
    ))
# optimizer_config = dict(_delete_=True, grad_clip=None)

# optimizer
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2242,
    warmup_ratio=1.0 / 200,
    step=[16, 22])
total_epochs = 26

SPCL=True
PSEUDO_LABELS = dict(
    freq=1, # epochs
    use_outliers=True,
    norm_feat=True,
    norm_center=True,
    SpCL=False,
    cluster='FINCH_context_SpCL_Plus',
    eps=[0.68, 0.7, 0.72],
    min_samples=4, # for dbscan
    dist_metric='jaccard',
    k1=30, # for jaccard distance
    k2=6, # for jaccard distance
    search_type=0, # 0,1,2 for GPU, 3 for CPU (work for faiss)
    cluster_num=None,
    iters=2,
    lambda_scene=0,    # default: 0.3
    lambda_person=0.1,
    context_method='zero',    # sum, max, zero
    threshold=0.5,
    use_post_process=False,
    filter_threshold=0.2,
    use_crop=False,
    use_k_reciprocal_nearest=False,
    K=10,
    part_feat=dict(use_part_feat=USE_PART_FEAT, 
                    global_weight=GLOBAL_WEIGHT,
                    uncertainty=UNCERTAINTY,
                    uncertainty_threshold=0.5,
                    global_weights=[1.0, 0.9]
                    ),
    inter_cluster=dict(
                    use_inter_cluster=True,
                    T=1,
                    )
)
# fp16 = dict(loss_scale=512.)
workflow = [('train', 1)]
evaluation = dict(start=30, interval=30, metric='bbox')
testing = TEST
save_features = True
restart = False