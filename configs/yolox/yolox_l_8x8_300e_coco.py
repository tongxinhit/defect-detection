_base_ = './yolox_s_8x8_300e_coco.py'
# _base_ = './yolox_s_wdist.py'
# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    # neck =[
    #         dict(
    #             type='YOLOXPAFPN',
    #             in_channels=[256, 512, 1024],
    #             out_channels=256,
    #             num_csp_blocks=3),
    #         dict(
    #             type='BFP',
    #             in_channels=256,
    #             num_levels=3,
    #             refine_level=2,
    #             refine_type='non_local')
    #       ],
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))
