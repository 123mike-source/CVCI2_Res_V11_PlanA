"""GeoBEV planning frontend; the pretrained IRWM encoder is separate."""

GEOBEV_VERSION = 9
GEOBEV_FORMAT = 'cvci2res_geobev_dualframe_integrated_v9'
GEOBEV_PROFILE = 'geobev_pretraining_cvci2_planner_fresh_v8_res_v1'
GEOBEV_SHA256 = 'd7076e95611b2057ee1612e2e7b896c88a5c703374e6c1935d4f54786ffa19cb'
CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')


def architecture_config(depth_loss_weight=.1):
    return dict(schema='geobev_dualframe_camera_v1', cameras=list(CAMERAS),
                image_size=[256, 704], image_channels='official_pil_imnormalize_to_rgb',
                scales=[16, 8], depth_range=[1., 60., .5],
                native_range=[-51.2, -51.2, 51.2, 51.2], native_interval=.4,
                native_axis_order='row_y_col_x',
                planner_range=[-32., -32., 32., 32.], planner_axis_order='row_x_col_y',
                planner_size=[64, 64], query_size=[8, 8], channels=256,
                history_frames=1, history_gradients=False,
                temporal_alignment='sensor_to_current_ego_before_rc_sampling',
                lidar_feature_input=False, batch_norm='fixed_pretrained_statistics',
                geobev_sha256=GEOBEV_SHA256, depth_loss_weight=float(depth_loss_weight))
