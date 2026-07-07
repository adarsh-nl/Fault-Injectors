"""
Builders for tiny synthetic dataset trees (Griffin / OPV2V / DAIR-V2X-C)
used by the adapter tests. Each builder writes a minimal but format-correct
directory and returns the info a test needs to assert against.
"""

import json
import os

import numpy as np


# ── PCD writers ─────────────────────────────────────────────────────────────

def write_pcd_ascii(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n'
                'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                'TYPE F F F F\nCOUNT 1 1 1 1\n'
                f'WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                f'POINTS {len(pts)}\nDATA ascii\n')
        for p in pts:
            f.write(' '.join(f'{v:.6f}' for v in p) + '\n')


def write_pcd_binary(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n'
              'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              f'WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {len(pts)}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(pts, dtype='<f4').tobytes())


# ── OPV2V / V2XSet ──────────────────────────────────────────────────────────

def make_opv2v_scenario(root, cav_ids=('641', '650'), n_frames=4, step=2):
    """
    Writes <root>/<cav>/<ts>.yaml + .pcd for each frame. Timestamps are
    '000068', '000070', ... Returns dict with 'timestamps' and 'points'
    (the per-cav base point cloud).
    """
    import yaml as _yaml
    timestamps = [f'{68 + step * i:06d}' for i in range(n_frames)]
    points = {}
    for ci, cav in enumerate(cav_ids):
        cav_dir = os.path.join(root, cav)
        os.makedirs(cav_dir, exist_ok=True)
        pts = np.array([[1.0 + ci, 0, 0, 10], [0, 2.0 + ci, 0, 20],
                        [0, 0, 3.0 + ci, 30]], dtype=np.float32)
        points[cav] = pts
        for fi, ts in enumerate(timestamps):
            params = {
                'ego_speed': 5.0,
                'lidar_pose': [10.0 * ci + fi, 5.0 * ci, 1.9,
                               0.0, 15.0 * ci, 0.0],
                'true_ego_pos': [10.0 * ci + fi, 5.0 * ci, 0.0,
                                 0.0, 15.0 * ci, 0.0],
                'camera0': {
                    'cords': [0, 0, 0, 0, 0, 0],
                    'intrinsic': np.diag([800.0, 800.0, 1.0]).tolist(),
                    'extrinsic': np.eye(4).tolist(),
                },
                'vehicles': {
                    200 + fi: {
                        'angle': [0.0, 30.0, 0.0],
                        'center': [0.0, 0.0, 0.75],
                        'extent': [2.4, 1.06, 0.75],
                        'location': [20.0 + fi, 15.0, 0.0],
                        'speed': 8.0,
                    },
                },
            }
            with open(os.path.join(cav_dir, f'{ts}.yaml'), 'w') as f:
                _yaml.safe_dump(params, f)
            write_pcd_ascii(os.path.join(cav_dir, f'{ts}.pcd'), pts)
    return {'timestamps': timestamps, 'points': points}


# ── Griffin ─────────────────────────────────────────────────────────────────

def make_griffin_tree(root, n_frames=3):
    """
    Writes a minimal vehicle-side tree: front camera pngs, lidar plys,
    poses, labels, calib. Returns dict with 'veh_root' and 'points'.
    """
    from PIL import Image
    from plyfile import PlyData, PlyElement

    veh = os.path.join(root, 'vehicle-side')
    for sub in ('camera/front', 'lidar/lidar_top', 'pose', 'label', 'calib'):
        os.makedirs(os.path.join(veh, sub), exist_ok=True)

    pts = np.array([[5.0, 0.0, -1.1, 7.0], [0.0, 3.0, -1.1, 9.0]],
                   dtype=np.float32)
    T_lidar_to_ego = np.eye(4)
    T_lidar_to_ego[:3, 3] = [0.25, 0.0, 1.10]

    with open(os.path.join(veh, 'calib', 'lidar_top.json'), 'w') as f:
        json.dump({'extrinsic': T_lidar_to_ego.tolist()}, f)
    with open(os.path.join(veh, 'calib', 'front.json'), 'w') as f:
        json.dump({'intrinsic': np.diag([700.0, 700.0, 1.0]).tolist(),
                   'extrinsic': np.eye(4).tolist()}, f)

    for k in range(n_frames):
        ts = k
        Image.new('RGB', (8, 6), (k, 100, 200)).save(
            os.path.join(veh, 'camera', 'front', f'{ts:06d}.png'))

        vert = np.array([tuple(p) for p in pts],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('I', 'f4')])
        PlyData([PlyElement.describe(vert, 'vertex')]).write(
            os.path.join(veh, 'lidar', 'lidar_top', f'{ts:06d}.ply'))

        with open(os.path.join(veh, 'pose', f'{ts:06d}.json'), 'w') as f:
            json.dump({'x': 100.0 + k, 'y': 200.0, 'z': 0.0,
                       'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                       'velocity': 5.0, 'timestamp': ts}, f)
        with open(os.path.join(veh, 'label', f'{ts:06d}.txt'), 'w') as f:
            f.write(f'Car 10.0 2.0 0.5 4.5 1.8 1.6 0 0 45.0 track{k} 0.9\n')
    return {'veh_root': veh, 'points': pts,
            'T_lidar_to_ego': T_lidar_to_ego}


# ── DAIR-V2X-C ──────────────────────────────────────────────────────────────

def make_dair_tree(root, n_frames=2):
    """
    Writes a minimal cooperative-vehicle-infrastructure tree with binary
    pcds. Returns dict with per-side 'points'.
    """
    veh_pts = np.array([[1.0, 1.0, 0.0, 5.0], [2.0, 0.0, 0.5, 6.0]],
                       dtype=np.float32)
    inf_pts = np.array([[8.0, -1.0, 2.0, 3.0]], dtype=np.float32)

    for side in ('vehicle-side', 'infrastructure-side'):
        for sub in ('velodyne', 'image', 'calib/camera_intrinsic'):
            os.makedirs(os.path.join(root, side, sub), exist_ok=True)
    for sub in ('calib/lidar_to_camera', 'calib/lidar_to_novatel',
                'calib/novatel_to_world'):
        os.makedirs(os.path.join(root, 'vehicle-side', sub), exist_ok=True)
    for sub in ('calib/virtuallidar_to_camera', 'calib/virtuallidar_to_world'):
        os.makedirs(os.path.join(root, 'infrastructure-side', sub),
                    exist_ok=True)
    os.makedirs(os.path.join(root, 'cooperative', 'label_world'),
                exist_ok=True)

    def rt(rotation, translation):
        return {'rotation': np.asarray(rotation).tolist(),
                'translation': np.asarray(translation).reshape(3, 1).tolist()}

    pairs = []
    for k in range(n_frames):
        vid, iid = f'{k:06d}', f'{k + 100:06d}'
        write_pcd_binary(os.path.join(root, 'vehicle-side', 'velodyne',
                                      f'{vid}.pcd'), veh_pts)
        write_pcd_binary(os.path.join(root, 'infrastructure-side', 'velodyne',
                                      f'{iid}.pcd'), inf_pts)

        vcal = os.path.join(root, 'vehicle-side', 'calib')
        with open(os.path.join(vcal, 'lidar_to_novatel', f'{vid}.json'), 'w') as f:
            json.dump({'transform': rt(np.eye(3), [1.0, 0.0, 0.5])}, f)
        with open(os.path.join(vcal, 'novatel_to_world', f'{vid}.json'), 'w') as f:
            json.dump(rt(np.eye(3), [1000.0 + k, 2000.0, 40.0]), f)
        with open(os.path.join(vcal, 'camera_intrinsic', f'{vid}.json'), 'w') as f:
            json.dump({'cam_K': np.diag([900.0, 900.0, 1.0]).ravel().tolist()}, f)
        with open(os.path.join(vcal, 'lidar_to_camera', f'{vid}.json'), 'w') as f:
            json.dump(rt(np.eye(3), [0.0, 0.0, 0.0]), f)

        ical = os.path.join(root, 'infrastructure-side', 'calib')
        with open(os.path.join(ical, 'virtuallidar_to_world', f'{iid}.json'), 'w') as f:
            json.dump(rt(np.eye(3), [1050.0, 2000.0, 45.0]), f)
        with open(os.path.join(ical, 'camera_intrinsic', f'{iid}.json'), 'w') as f:
            json.dump({'cam_K': np.diag([900.0, 900.0, 1.0]).ravel().tolist()}, f)
        with open(os.path.join(ical, 'virtuallidar_to_camera', f'{iid}.json'), 'w') as f:
            json.dump(rt(np.eye(3), [0.0, 0.0, 0.0]), f)

        label_rel = f'cooperative/label_world/{vid}.json'
        with open(os.path.join(root, label_rel), 'w') as f:
            json.dump([{'type': 'Car',
                        '3d_dimensions': {'l': 4.2, 'w': 1.8, 'h': 1.5},
                        '3d_location': {'x': 1010.0, 'y': 2005.0, 'z': 40.2},
                        'rotation': float(np.pi / 2)}], f)

        pairs.append({
            'vehicle_pointcloud_path': f'vehicle-side/velodyne/{vid}.pcd',
            'infrastructure_pointcloud_path':
                f'infrastructure-side/velodyne/{iid}.pcd',
            'cooperative_label_path': label_rel,
        })

    with open(os.path.join(root, 'cooperative', 'data_info.json'), 'w') as f:
        json.dump(pairs, f)
    return {'veh_points': veh_pts, 'inf_points': inf_pts}
