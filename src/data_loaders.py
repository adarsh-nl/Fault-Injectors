"""
data_loaders.py
---------------
All file I/O for the Griffin dataset.
Handles images, LiDAR point clouds, poses, calibration, and labels.
"""

import os
import glob
import json
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation as Rot


# ── File list helpers ──────────────────────────────────────────────────────

def get_file_lists(veh_root, drone_root=None):
    """
    Build sorted file lists for all data types.

    Parameters
    ----------
    veh_root  : str  Path to vehicle-side root directory.
    drone_root: str  Path to drone-side root directory (optional).

    Returns
    -------
    dict with keys:
        veh_fronts, veh_backs, veh_lefts, veh_rights,
        lidar_plys, pose_files, label_files,
        drone_fronts, drone_backs, drone_lefts, drone_rights, drone_bottoms
    """
    def _glob(base, *parts):
        return sorted(glob.glob(os.path.join(base, *parts)))

    lists = {
        'veh_fronts' : _glob(veh_root, 'camera', 'front',      '*.png'),
        'veh_backs'  : _glob(veh_root, 'camera', 'back',       '*.png'),
        'veh_lefts'  : _glob(veh_root, 'camera', 'left',       '*.png'),
        'veh_rights' : _glob(veh_root, 'camera', 'right',      '*.png'),
        'lidar_plys' : _glob(veh_root, 'lidar',  'lidar_top',  '*.ply'),
        'pose_files' : _glob(veh_root, 'pose',                 '*.json'),
        'label_files': _glob(veh_root, 'label',                '*.txt'),
    }

    if drone_root:
        lists.update({
            'drone_fronts'  : _glob(drone_root, 'camera', 'front',  '*.png'),
            'drone_backs'   : _glob(drone_root, 'camera', 'back',   '*.png'),
            'drone_lefts'   : _glob(drone_root, 'camera', 'left',   '*.png'),
            'drone_rights'  : _glob(drone_root, 'camera', 'right',  '*.png'),
            'drone_bottoms' : _glob(drone_root, 'camera', 'bottom', '*.png'),
        })

    return lists


# ── Image loader ───────────────────────────────────────────────────────────

def load_image(path):
    """
    Load an image as an (H, W, 3) uint8 RGB numpy array.

    Parameters
    ----------
    path : str  Path to image file (.png or .jpg).

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8
    """
    return np.array(Image.open(path).convert('RGB'))


# ── LiDAR loader ──────────────────────────────────────────────────────────

def load_lidar(ply_path):
    """
    Load a Griffin LiDAR .ply file.

    Griffin stores LiDAR fields as: x, y, z (ENU world frame) and I (intensity).
    Points are in the ENU world frame, not the ego or sensor frame.

    Parameters
    ----------
    ply_path : str  Path to .ply file.

    Returns
    -------
    np.ndarray  shape (N, 4), dtype float32 — columns: x, y, z, intensity
    """
    ply = PlyData.read(ply_path)
    v   = ply['vertex']
    x   = np.array(v['x'], dtype=np.float32)
    y   = np.array(v['y'], dtype=np.float32)
    z   = np.array(v['z'], dtype=np.float32)
    I   = np.array(v['I'], dtype=np.float32)
    return np.stack([x, y, z, I], axis=1)


# ── Pose loader ───────────────────────────────────────────────────────────

def load_pose_griffin(pose_path):
    """
    Load ego pose and return T_ENU_to_ego (4x4).

    The pose file gives the vehicle's position and orientation in the ENU
    world frame. Griffin uses Euler order 'xyz' (roll, pitch, yaw) as
    documented in space_utils.py.

    Parameters
    ----------
    pose_path : str  Path to pose .json file.

    Returns
    -------
    np.ndarray  shape (4, 4) — T_ENU_to_ego transform matrix
    dict        raw pose data (x, y, z, roll, pitch, yaw, timestamp, velocity)
    """
    with open(pose_path) as f:
        p = json.load(f)

    R_mat = Rot.from_euler(
        'xyz', [p['roll'], p['pitch'], p['yaw']], degrees=True
    ).as_matrix()

    T_ego_to_ENU = np.eye(4)
    T_ego_to_ENU[:3, :3] = R_mat
    T_ego_to_ENU[:3,  3] = [p['x'], p['y'], p['z']]

    return np.linalg.inv(T_ego_to_ENU), p   # T_ENU_to_ego, raw pose dict


# ── Calibration loader ────────────────────────────────────────────────────

def load_calib_griffin(calib_dir, camera='front'):
    """
    Load calibration for one camera.

    The extrinsic matrix in the calibration file is T_sensor_to_ego:
    it transforms a point from camera coordinates into ego coordinates.
    We invert it to get T_ego_to_sensor for projection.

    Parameters
    ----------
    calib_dir : str  Path to the calib/ directory.
    camera    : str  One of 'front', 'back', 'left', 'right'.

    Returns
    -------
    K               : np.ndarray (3, 3) — intrinsic matrix
    T_ego_to_sensor : np.ndarray (4, 4) — ego -> camera transform
    """
    with open(os.path.join(calib_dir, f'{camera}.json')) as f:
        cal = json.load(f)

    K               = np.array(cal['intrinsic'],  dtype=np.float64)
    T_sensor_to_ego = np.array(cal['extrinsic'],  dtype=np.float64)
    return K, np.linalg.inv(T_sensor_to_ego)


# ── Label loader ──────────────────────────────────────────────────────────

def parse_label_txt(label_path):
    """
    Parse a Griffin label .txt file into a list of annotation dicts.

    Each line format (space-separated):
        category  x  y  z  l  w  h  roll  pitch  yaw  track_id  visibility

    where x,y,z is the box centre in ego frame (metres),
    l,w,h are dimensions (metres), roll/pitch/yaw are in degrees,
    and visibility is in [0, 1].

    Parameters
    ----------
    label_path : str  Path to label .txt file.

    Returns
    -------
    list of dict
    """
    anns = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            anns.append({
                'category'  : parts[0],
                'x'         : float(parts[1]),
                'y'         : float(parts[2]),
                'z'         : float(parts[3]),
                'l'         : float(parts[4]),
                'w'         : float(parts[5]),
                'h'         : float(parts[6]),
                'roll'      : float(parts[7]),
                'pitch'     : float(parts[8]),
                'yaw'       : float(parts[9]),
                'id'        : parts[10],
                'visibility': float(parts[11]),
            })
    return anns


def load_labels_for_frame(veh_root, pose_files, frame_idx):
    """
    Load annotations for a given frame index, matched by timestamp.

    Parameters
    ----------
    veh_root   : str   Path to vehicle-side root directory.
    pose_files : list  Sorted list of pose file paths.
    frame_idx  : int   Frame index.

    Returns
    -------
    list of dict  — annotation list (may be empty if no objects in frame)
    """
    label_dir = os.path.join(veh_root, 'label')

    with open(pose_files[frame_idx]) as f:
        ts = int(json.load(f)['timestamp'])

    # Primary: timestamp-based filename
    label_path = os.path.join(label_dir, f'{ts:06d}.txt')
    if os.path.exists(label_path):
        return parse_label_txt(label_path)

    # Fallback: sorted index
    all_labels = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    if all_labels and frame_idx < len(all_labels):
        return parse_label_txt(all_labels[frame_idx])

    return []
