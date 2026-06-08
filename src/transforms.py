"""
transforms.py
-------------
Coordinate frame transforms and LiDAR-to-image projection for Griffin.

Frame conventions
-----------------
ENU (world)   : X=East, Y=North, Z=Up  — LiDAR points stored here
Ego (vehicle) : X=forward, Y=left, Z=up — annotations stored here
Camera        : X=right, Y=down, Z=forward (depth) — OpenCV convention

Pipeline for LiDAR -> image
----------------------------
  pts_ENU
    -> T_ENU_to_ego       (inv of pose: ego position in ENU)
    -> T_ego_to_sensor    (inv of extrinsic: sensor position in ego)
    -> K                  (pinhole projection, Z=depth)
    -> (u, v) pixels

Pipeline for annotations -> image (already in ego frame)
----------------------------------------------------------
  pts_ego
    -> T_ego_to_sensor
    -> K
    -> (u, v) pixels
"""

import numpy as np


def project_lidar_to_image(pts_ENU, K, T_ego_to_sensor, T_ENU_to_ego,
                            img_h, img_w, max_depth=80.0):
    """
    Project LiDAR points (ENU world frame) onto an image plane.

    Parameters
    ----------
    pts_ENU        : np.ndarray (N, 3)  LiDAR points in ENU world frame.
    K              : np.ndarray (3, 3)  Camera intrinsic matrix.
    T_ego_to_sensor: np.ndarray (4, 4)  Ego -> sensor transform.
    T_ENU_to_ego   : np.ndarray (4, 4)  ENU -> ego transform (inv pose).
    img_h, img_w   : int                Image dimensions.
    max_depth      : float              Maximum depth to keep (metres).

    Returns
    -------
    np.ndarray (M, 3)  columns: u (pixel col), v (pixel row), depth (m)
    """
    N = pts_ENU.shape[0]

    # Combine ENU -> ego -> sensor into a single 4x4
    T = T_ego_to_sensor @ T_ENU_to_ego

    pts_h = np.hstack([pts_ENU, np.ones((N, 1))])
    pts_s = (T @ pts_h.T).T[:, :3]

    # Depth = Z in sensor frame; keep only points in front of camera
    depth = pts_s[:, 2]
    mask  = depth > 0.1
    pts_s = pts_s[mask]
    depth = depth[mask]

    # Pinhole projection
    proj = (K @ pts_s.T).T
    u    = proj[:, 0] / depth
    v    = proj[:, 1] / depth

    # Image bounds filter
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (depth < max_depth)
    return np.stack([u[valid], v[valid], depth[valid]], axis=1)


def project_ego_to_img(pts_ego, K, T_ego_to_sensor, img_h, img_w):
    """
    Project ego-frame points onto the image plane.

    Used for projecting bounding box corners (which are in ego frame,
    not ENU frame). The pose transform is NOT applied here.

    Parameters
    ----------
    pts_ego        : np.ndarray (N, 3)  Points in ego frame.
    K              : np.ndarray (3, 3)  Camera intrinsic matrix.
    T_ego_to_sensor: np.ndarray (4, 4)  Ego -> sensor transform.
    img_h, img_w   : int                Image dimensions.

    Returns
    -------
    np.ndarray (N, 3)  columns: u, v, depth
                        u=-9999 or depth<=0 means point is behind camera.
    """
    N     = pts_ego.shape[0]
    pts_s = (T_ego_to_sensor @ np.hstack([pts_ego, np.ones((N, 1))]).T).T[:, :3]
    depth = pts_s[:, 2]
    proj  = (K @ pts_s.T).T
    u = np.where(depth > 0.01, proj[:, 0] / np.maximum(depth, 1e-6), -9999.0)
    v = np.where(depth > 0.01, proj[:, 1] / np.maximum(depth, 1e-6), -9999.0)
    return np.stack([u, v, depth], axis=1)


def ego_box_corners_3d(ann):
    """
    Compute the 8 corners of a 3D bounding box in the ego frame.

    Parameters
    ----------
    ann : dict  Annotation dict with keys x,y,z,l,w,h,yaw (yaw in degrees).

    Returns
    -------
    np.ndarray (8, 3)  Box corners in ego frame.
    """
    cx, cy, cz = ann['x'], ann['y'], ann['z']
    l, w, h    = ann['l'], ann['w'], ann['h']
    yaw        = np.radians(ann.get('yaw', 0.0))

    cy_r, sy_r = np.cos(yaw), np.sin(yaw)
    R3 = np.array([[cy_r, -sy_r, 0],
                   [sy_r,  cy_r, 0],
                   [0,     0,    1]])

    dx, dy, dz = l / 2, w / 2, h / 2
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
    ])
    return (R3 @ corners.T).T + np.array([cx, cy, cz])


def ann_to_ego_corners_bev(ann):
    """
    Compute the 4 BEV footprint corners of a box in ego frame.

    Parameters
    ----------
    ann : dict  Annotation dict with keys x,y,l,w,yaw (yaw in degrees).

    Returns
    -------
    np.ndarray (4, 2)  BEV corners (x, y).
    """
    cx, cy = ann['x'], ann['y']
    l, w   = ann['l'], ann['w']
    yaw    = np.radians(ann.get('yaw', 0.0))
    cy_r, sy_r = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2, w / 2
    corners = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
    R2 = np.array([[cy_r, -sy_r], [sy_r, cy_r]])
    return (R2 @ corners.T).T + np.array([cx, cy])
