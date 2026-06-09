"""
viser_viewer.py
---------------
Interactive 3D viewer for the Griffin dataset using Viser, in the WORLD frame
(standing on the sidewalk watching the car drive past).

  - The scene is fixed in space.
  - The ego vehicle drives THROUGH the scene as you scrub frames.
  - The LiDAR + 4 camera markers ride along on the car at their true mounting
    offsets, moving and rotating with it.
  - The LiDAR point cloud (the current scan) travels with the car.
  - 3D bounding boxes move with their objects.

Two viewpoints, one choice
--------------------------
EGO frame  : car pinned at the origin, world flows past (sit inside the car).
WORLD frame: car drives through a fixed scene (stand on the sidewalk).  <-- this file

This file uses the WORLD frame so the car, its sensors, and its point cloud all
move together as the vehicle drives — which is how py123d shows a scene over time.

Coordinates are shifted by a reference origin (ego position at the first frame)
so the numbers stay small and near zero.

Usage
-----
    pip install viser
    python viser_viewer.py

Pick a frame range where the car is DRIVING and within ONE scene.
Tick "Camera follows car" to keep the car centred while the world scrolls.

JupyterHub access (no SSH): set USE_SHARE_URL = True.
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import viser

from src import (
    load_lidar, load_pose_griffin, load_labels_for_frame,
    get_file_lists, ego_box_corners_3d, CAT_COLORS,
)


# ── Configuration ──────────────────────────────────────────────────────────

DATASET_ROOT = '../datasets/griffin_50scenes_25m/griffin-release/griffin_50scenes_25m/griffin-release'
VEH = os.path.join(DATASET_ROOT, 'vehicle-side')

FRAME_START = 600     # a range where the car is driving, within one scene
FRAME_END   = 670     # exclusive

POINT_SIZE  = 0.05
HEIGHT_MIN  = -2.0
HEIGHT_MAX  = 8.0

PORT          = 8080
USE_SHARE_URL = False
FOLLOW_DEFAULT = True    # camera tracks the ego by default (py123d style)
FOLLOW_HEIGHT  = 35.0    # camera height above the car when following (metres)


# ── Helpers ────────────────────────────────────────────────────────────────

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def height_colors(z, vmin=HEIGHT_MIN, vmax=HEIGHT_MAX):
    z_norm = np.clip((z - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgb = cm.plasma(z_norm)[:, :3]
    return (rgb * 255).astype(np.uint8)


def load_sensor_extrinsic(calib_dir, sensor):
    """Return T_sensor_to_ego (4x4) for a sensor (camera or lidar_top)."""
    with open(os.path.join(calib_dir, f'{sensor}.json')) as f:
        cal = json.load(f)
    return np.array(cal['extrinsic'], dtype=np.float64)


def matrix_to_pos_wxyz(T):
    """Split a 4x4 transform into (position, wxyz quaternion) for Viser."""
    pos = T[:3, 3]
    quat_xyzw = Rot.from_matrix(T[:3, :3]).as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, wxyz


# ── Sensors (vehicle LiDAR + 4 cameras), mounting poses in ego frame ───────

CALIB_DIR = os.path.join(VEH, 'calib')
SENSORS = {
    'lidar_top': 'LiDAR',
    'front':     'CAM front',
    'back':      'CAM back',
    'left':      'CAM left',
    'right':     'CAM right',
}
T_SENSOR_TO_EGO = {s: load_sensor_extrinsic(CALIB_DIR, s) for s in SENSORS}


# ── Preload the sequence (world frame, shifted by REF) ─────────────────────

print('Loading file lists...')
files = get_file_lists(VEH)

N_FRAMES = FRAME_END - FRAME_START

_, ref_pose = load_pose_griffin(files['pose_files'][FRAME_START])
REF = np.array([ref_pose['x'], ref_pose['y'], ref_pose['z']])
print(f'Reference origin (ENU): {REF}')

FRAMES = []
print(f'Preloading {N_FRAMES} frames ({FRAME_START}..{FRAME_END - 1})...')
for idx in range(FRAME_START, FRAME_END):
    pts_ENU            = load_lidar(files['lidar_plys'][idx])
    T_ENU_to_ego, pose = load_pose_griffin(files['pose_files'][idx])
    anns               = load_labels_for_frame(VEH, files['pose_files'], idx)

    T_ego_to_ENU = np.linalg.inv(T_ENU_to_ego)

    # Points already in ENU; just shift by reference origin
    pts_world = pts_ENU[:, :3] - REF
    colors    = height_colors(pts_ENU[:, 2])   # colour by absolute world height

    # Ego pose in shifted world frame
    ego_pos  = np.array([pose['x'], pose['y'], pose['z']]) - REF
    _, ego_wxyz = matrix_to_pos_wxyz(T_ego_to_ENU)

    # Sensor poses in shifted world frame (they ride on the car)
    sensor_world = {}
    for s in SENSORS:
        T_sensor_to_world = T_ego_to_ENU @ T_SENSOR_TO_EGO[s]
        pos, wxyz = matrix_to_pos_wxyz(T_sensor_to_world)
        sensor_world[s] = (pos - REF, wxyz)

    # Boxes: ego-frame corners -> world -> shift
    seg_pts, seg_col = [], []
    for ann in anns:
        corners_ego = ego_box_corners_3d(ann)
        N = corners_ego.shape[0]
        corners_world = (T_ego_to_ENU @ np.hstack([corners_ego, np.ones((N,1))]).T).T[:, :3] - REF
        rgb = hex_to_rgb(CAT_COLORS.get(ann.get('category', 'car').lower(), '#ffffff'))
        for i, j in BOX_EDGES:
            seg_pts.append([corners_world[i], corners_world[j]])
            seg_col.append([rgb, rgb])
    if seg_pts:
        seg_pts = np.array(seg_pts, dtype=np.float32)
        seg_col = np.array(seg_col, dtype=np.uint8)
    else:
        seg_pts = np.zeros((0, 2, 3), dtype=np.float32)
        seg_col = np.zeros((0, 2, 3), dtype=np.uint8)

    FRAMES.append({
        'pts_world'   : pts_world.astype(np.float32),
        'colors'      : colors,
        'seg_pts'     : seg_pts,
        'seg_col'     : seg_col,
        'ego_pos'     : ego_pos.astype(np.float32),
        'ego_wxyz'    : ego_wxyz.astype(np.float32),
        'sensor_world': sensor_world,
        'pose'        : pose,
        'n_boxes'     : len(anns),
    })
    print(f'  frame {idx}: ego at ({ego_pos[0]:+.1f}, {ego_pos[1]:+.1f})', end='\r')

print(f'\nPreload complete. {N_FRAMES} frames in memory.')

ego_xy = np.array([f['ego_pos'][:2] for f in FRAMES])
travel = np.linalg.norm(ego_xy[-1] - ego_xy[0])
print(f'Ego travels {travel:.1f} m across the sequence.')
if travel < 1.0:
    print('WARNING: the car barely moves — pick a range where it is driving.')
jumps = np.linalg.norm(np.diff(ego_xy, axis=0), axis=1)
if len(jumps) and jumps.max() > 10.0:
    print(f'WARNING: pose jump of {jumps.max():.0f} m — you may cross a scene boundary.')


# ── Start the Viser server ─────────────────────────────────────────────────

if 'server' in globals():
    try:
        server.stop()
    except Exception:
        pass

server = viser.ViserServer(port=PORT)
server.scene.set_up_direction('+z')

# Initial camera: above the middle of the trajectory, looking down
mid = ego_xy.mean(axis=0)
span = max(20.0, np.ptp(ego_xy, axis=0).max())


@server.on_client_connect
def _on_connect(client: viser.ClientHandle) -> None:
    client.camera.position = (float(mid[0]), float(mid[1]) - 0.1, float(span * 1.5))
    client.camera.look_at  = (float(mid[0]), float(mid[1]), 0.0)


gui_frame = server.gui.add_slider('Timestep', min=0, max=N_FRAMES - 1, step=1, initial_value=0)
gui_playing = server.gui.add_checkbox('Playing', initial_value=False)
gui_speed = server.gui.add_slider('Playback FPS', min=1, max=20, step=1, initial_value=10)
gui_point_size = server.gui.add_slider('Point size', min=0.01, max=0.3, step=0.01, initial_value=POINT_SIZE)
gui_follow = server.gui.add_checkbox('Camera follows car', initial_value=FOLLOW_DEFAULT)
gui_show_sensors = server.gui.add_checkbox('Show sensor markers', initial_value=True)
gui_info = server.gui.add_text('Info', initial_value='', disabled=True)


def show_frame(local_idx):
    data = FRAMES[local_idx]

    server.scene.add_point_cloud(
        '/lidar',
        points=data['pts_world'],
        colors=data['colors'],
        point_size=gui_point_size.value,
        point_shape='circle',
    )
    server.scene.add_line_segments(
        '/boxes',
        points=data['seg_pts'],
        colors=data['seg_col'],
        line_width=2.0,
    )

    # Ego origin marker (the vehicle reference point) — moves with the car
    server.scene.add_frame(
        '/ego_origin',
        position=tuple(float(v) for v in data['ego_pos']),
        wxyz=tuple(float(v) for v in data['ego_wxyz']),
        axes_length=2.5, axes_radius=0.12,
    )

    # Sensor markers ride on the car — update their world pose each frame
    for s, label in SENSORS.items():
        pos, wxyz = data['sensor_world'][s]
        is_lidar = (s == 'lidar_top')
        server.scene.add_frame(
            f'/sensors/{s}',
            position=tuple(float(v) for v in pos),
            wxyz=tuple(float(v) for v in wxyz),
            axes_length=1.5 if is_lidar else 1.0,
            axes_radius=0.08 if is_lidar else 0.05,
            visible=gui_show_sensors.value,
        )
        server.scene.add_label(
            f'/sensors/{s}/label',
            text=label,
            position=tuple(float(v) for v in pos),
        )

    # Optional camera follow — keeps the car centred and full-size (py123d style)
    if gui_follow.value:
        ex, ey, _ = data['ego_pos']
        for client in server.get_clients().values():
            client.camera.position = (float(ex), float(ey) - 0.1, FOLLOW_HEIGHT)
            client.camera.look_at  = (float(ex), float(ey), 0.0)

    pose = data['pose']
    abs_frame = FRAME_START + local_idx
    gui_info.value = (
        f"frame {abs_frame}  |  ts={int(pose.get('timestamp', 0))}  |  "
        f"{len(data['pts_world']):,} pts  |  {data['n_boxes']} boxes  |  "
        f"vel={pose.get('velocity', 0):.2f} m/s"
    )


@gui_frame.on_update
def _(_):
    show_frame(gui_frame.value)


@gui_point_size.on_update
def _(_):
    show_frame(gui_frame.value)


@gui_follow.on_update
def _(_):
    show_frame(gui_frame.value)


@gui_show_sensors.on_update
def _(_):
    show_frame(gui_frame.value)


show_frame(0)

if USE_SHARE_URL:
    print('Requesting Viser share URL (may take a few seconds)...')
    print(f'\nShare URL: {server.request_share_url()}')

print(f'\nViser server running on port {PORT}.')
print('Tick "Playing" to animate — the ego, sensors, and LiDAR move through the scene.')
print('Camera follows the car by default, so it stays centred (py123d style).')
print('Press Ctrl+C to stop.\n')

# Playback loop: when "Playing" is ticked, auto-advance the Timestep slider.
# Setting gui_frame.value fires its on_update, which re-renders the frame.
try:
    while True:
        if gui_playing.value:
            next_idx = (gui_frame.value + 1) % N_FRAMES
            gui_frame.value = next_idx          # triggers show_frame via on_update
            time.sleep(1.0 / gui_speed.value)   # playback speed
        else:
            time.sleep(0.05)
except KeyboardInterrupt:
    print('\nShutting down.')
    server.stop()