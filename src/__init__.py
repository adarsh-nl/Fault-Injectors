"""Fault-Injectors toolkit. Top-level names resolve lazily (PEP 562) so that
importing a submodule (e.g. `src.datasets.base`) does not drag in the
plotting and PLY stack. A LiDAR-only training run must not require matplotlib.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "load_image": ".data_loaders",
    "load_lidar": ".data_loaders",
    "load_pose_griffin": ".data_loaders",
    "load_calib_griffin": ".data_loaders",
    "load_sensor_extrinsic": ".data_loaders",
    "load_labels_for_frame": ".data_loaders",
    "parse_label_txt": ".data_loaders",
    "get_file_lists": ".data_loaders",
    "project_lidar_to_image": ".transforms",
    "project_ego_to_img": ".transforms",
    "ego_box_corners_3d": ".transforms",
    "ann_to_ego_corners_bev": ".transforms",
    "ego_points_to_world": ".transforms",
    "plot_surround_cameras": ".visualisation",
    "plot_bev": ".visualisation",
    "plot_front_view": ".visualisation",
    "plot_fusion": ".visualisation",
    "plot_bev_with_boxes": ".visualisation",
    "plot_boxes_on_image": ".visualisation",
    "CAT_COLORS": ".visualisation",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)
