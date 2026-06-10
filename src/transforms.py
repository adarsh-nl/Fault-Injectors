"""
transforms.py
-------------
Coordinate frame transforms and projection for Griffin.

IMPORTANT — frame of the LiDAR points
-------------------------------------
Griffin LiDAR .ply points are stored in the EGO (vehicle) frame: the car sits
at the origin, X=forward, Y=left, Z=up. They are NOT in the ENU world frame.
(The large coordinate ranges seen in some scans are just far-field returns in a
large scene, still expressed relative to the ego.)

Consequences:
  - Projecting LiDAR onto a camera goes EGO -> SENSOR -> IMAGE directly.
    No pose transform is involved.
  - Annotations are also in the ego frame, so they use the same path.
  - The pose (T_ENU_to_ego / its inverse) is only needed when you want to place
    ego-frame data into the world frame, e.g. a world-frame 3D viewer.

Frame conventions
-----------------
ENU (world)   : X=East, Y=North, Z=Up
Ego (vehicle) : X=forward, Y=left, Z=up   <- LiDAR points and annotations live here
Camera        : X=right, Y=down, Z=forward (depth), OpenCV convention
"""

import numpy as np


def project_lidar_to_image(pts_ego, K, T_ego_to_sensor, img_h, img_w, max_depth=80.0):
    """
    Project LiDAR points (EGO frame) onto an image plane.

    Pipeline: ego -> sensor (via T_ego_to_sensor) -> pixel (via K, Z=depth).
    No pose transform, because the points are already ego-centred.

    Parameters
    ----------
    pts_ego        : np.ndarray (N, 3)  LiDAR points in ego frame.
    K              : np.ndarray (3, 3)  Camera intrinsic matrix.
    T_ego_to_sensor: np.ndarray (4, 4)  Ego -> sensor transform.
    img_h, img_w   : int                Image dimensions.
    max_depth      : float              Max depth to keep (metres).

    Returns
    -------
    np.ndarray (M, 3)  columns: u (pixel col), v (pixel row), depth (m)
    """
    N = pts_ego.shape[0]
    pts_s = (T_ego_to_sensor @ np.hstack([pts_ego, np.ones((N, 1))]).T).T[:, :3]

    depth = pts_s[:, 2]
    mask  = depth > 0.1
    pts_s, depth = pts_s[mask], depth[mask]

    proj = (K @ pts_s.T).T
    u = proj[:, 0] / depth
    v = proj[:, 1] / depth

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (depth < max_depth)
    return np.stack([u[valid], v[valid], depth[valid]], axis=1)


def project_ego_to_img(pts_ego, K, T_ego_to_sensor, img_h, img_w):
    """
    Project ego-frame points onto the image plane, KEEPING all points (no
    in-image filtering) so callers can preserve connectivity, e.g. drawing the
    12 edges of a 3D box. Points behind the camera get depth <= 0.

    Parameters
    ----------
    pts_ego        : np.ndarray (N, 3)
    K              : np.ndarray (3, 3)
    T_ego_to_sensor: np.ndarray (4, 4)
    img_h, img_w   : int

    Returns
    -------
    np.ndarray (N, 3)  columns: u, v, depth
                        depth <= 0 means the point is behind the camera.
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
    8 corners of a 3D bounding box in the ego frame.

    Parameters
    ----------
    ann : dict  with keys x, y, z, l, w, h, yaw (yaw in degrees).

    Returns
    -------
    np.ndarray (8, 3)
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
    4 BEV footprint corners of a box in ego frame.

    Parameters
    ----------
    ann : dict  with keys x, y, l, w, yaw (yaw in degrees).

    Returns
    -------
    np.ndarray (4, 2)
    """
    cx, cy = ann['x'], ann['y']
    l, w   = ann['l'], ann['w']
    yaw    = np.radians(ann.get('yaw', 0.0))
    cy_r, sy_r = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2, w / 2
    corners = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
    R2 = np.array([[cy_r, -sy_r], [sy_r, cy_r]])
    return (R2 @ corners.T).T + np.array([cx, cy])


def ego_points_to_world(pts_ego, T_ego_to_ENU):
    """
    Lift ego-frame points into the ENU world frame.

    Used by the world-frame 3D viewer, where the car drives through a fixed
    scene. Not used for camera projection.

    Parameters
    ----------
    pts_ego      : np.ndarray (N, 3)  points in ego frame.
    T_ego_to_ENU : np.ndarray (4, 4)  ego -> ENU transform (inverse of pose).

    Returns
    -------
    np.ndarray (N, 3)  points in ENU world frame.
    """
    N = pts_ego.shape[0]
    return (T_ego_to_ENU @ np.hstack([pts_ego, np.ones((N, 1))]).T).T[:, :3]