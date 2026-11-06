import torch.nn as nn
from mmdet.models import builder


class MVXTwoStageDetector(nn.Module):
    """Minimal image-only base required by Drive_OccWorld/BEVFormer."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.img_backbone = builder.build_backbone(img_backbone) if img_backbone is not None else None
        self.img_neck = builder.build_neck(img_neck) if img_neck is not None else None
        if pts_bbox_head is not None:
            pts_bbox_head = pts_bbox_head.copy()
            pts_bbox_head.update(
                train_cfg=train_cfg.get("pts") if train_cfg is not None else None,
                test_cfg=test_cfg.get("pts") if test_cfg is not None else None,
            )
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        else:
            self.pts_bbox_head = None
        self.pts_voxel_layer = None
        self.pts_voxel_encoder = None
        self.pts_middle_encoder = None
        self.pts_fusion_layer = None
        self.pts_backbone = None
        self.pts_neck = None
        self.img_roi_head = None
        self.img_rpn_head = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_img_neck = self.img_neck is not None
        self.with_pts_bbox = self.pts_bbox_head is not None
