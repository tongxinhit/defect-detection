_base_ = ['/home/tongxin/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py']
# model settings
find_unused_parameters=True
temp=0.5
alpha_fgd=0.0016
beta_fgd=0.0008
gamma_fgd=0.008
lambda_fgd=0.000008
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = '/home/tongxin/mmdetection/work_dir_aidot/yolox_l_1/best_bbox_mAP_epoch_188.pth',
    init_student = True,
    distill_cfg = [dict(student_module = 'neck.out_convs.0.conv', 
                         teacher_module = 'neck.out_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 96,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.1.conv',
                         teacher_module = 'neck.out_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 96,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.2.conv',
                         teacher_module = 'neck.out_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 96,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),

                   ]
    )
student_cfg = '/home/tongxin/mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py'
teacher_cfg = '/home/tongxin/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,)