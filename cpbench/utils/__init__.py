"""Configuration composition, dataset path resolution, BEV geometry helpers."""

from .config import load_config
from .geometry import (boxes_to_corners_bev, nms_bev, pose6_to_matrix,
                       rotated_iou_bev, standup_iou_bev, transform_boxes)
from .paths import (DEFAULT_DATA_ROOT, ENV_VAR, data_root, dataset_root,
                    describe_source, missing_root_message,
                    require_dataset_root)

__all__ = ["load_config", "boxes_to_corners_bev", "standup_iou_bev",
           "rotated_iou_bev", "nms_bev", "transform_boxes", "pose6_to_matrix",
           "data_root", "dataset_root", "require_dataset_root",
           "describe_source", "missing_root_message", "ENV_VAR",
           "DEFAULT_DATA_ROOT"]
