"""
dair_v2x.py
-----------
DAIR-V2X-C adapter (the real-world cooperative vehicle-infrastructure
dataset, CVPR 2022). Two agents per sample:

    'vehicle' (ego)   camera + LiDAR + GPS/IMU pose
    'infrastructure'  camera + LiDAR ("virtuallidar") + surveyed pose

Layout expected (root = the cooperative-vehicle-infrastructure directory):

    vehicle-side/
        image/{id}.jpg          velodyne/{id}.pcd
        calib/camera_intrinsic/{id}.json
        calib/lidar_to_camera/{id}.json
        calib/lidar_to_novatel/{id}.json
        calib/novatel_to_world/{id}.json
        data_info.json
    infrastructure-side/
        image/{id}.jpg          velodyne/{id}.pcd
        calib/camera_intrinsic/{id}.json
        calib/virtuallidar_to_camera/{id}.json
        calib/virtuallidar_to_world/{id}.json
        data_info.json
    cooperative/
        data_info.json          (pairs vehicle frame <-> infra frame)
        label_world/{id}.json   (world-frame cooperative 3-D labels)

Conventions mapped to the common model:
    * agent frame  = each side's LiDAR frame (what the .pcd is in).
    * vehicle pose = T_novatel_to_world @ T_lidar_to_novatel
      infra pose   = T_virtuallidar_to_world
    * labels       = cooperative world-frame boxes, attached to the ego
      agent with frame='world'. DAIR stores rotation in radians; converted
      to degrees here.

DAIR-V2X is asynchronous by nature (vehicle and infra frames are matched,
not simultaneous); the matched pairing from cooperative/data_info.json is
what this adapter serves, and fault injectors add error on top of it.
"""

import json
import os

import numpy as np

from .base import AgentFrame, BaseDataset, Box3D, CameraCalib
from .pcd import load_pcd


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _rt_to_T(node):
    """{'rotation': 3x3, 'translation': 3x1} (possibly under 'transform')."""
    if 'transform' in node:
        node = node['transform']
    T = np.eye(4)
    T[:3, :3] = np.asarray(node['rotation'], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(node['translation'], dtype=np.float64).reshape(3)
    return T


def _frame_id(path_str):
    """'.../velodyne/015344.pcd' -> '015344'."""
    return os.path.splitext(os.path.basename(path_str))[0]


class DairV2XDataset(BaseDataset):
    """
    Parameters
    ----------
    root : the cooperative-vehicle-infrastructure directory (the one that
           contains vehicle-side/, infrastructure-side/ and cooperative/).
    """

    name = 'dair-v2x'
    fps = 10.0

    _SIDES = {
        'vehicle': {
            'dir': 'vehicle-side',
            'to_cam': 'lidar_to_camera',
        },
        'infrastructure': {
            'dir': 'infrastructure-side',
            'to_cam': 'virtuallidar_to_camera',
        },
    }

    def __init__(self, root):
        self.root = root
        info_path = os.path.join(root, 'cooperative', 'data_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f'{info_path} not found -- expected the DAIR-V2X-C '
                f'cooperative-vehicle-infrastructure directory')
        self._pairs = _read_json(info_path)

    # -- BaseDataset --------------------------------------------------------

    def agent_ids(self):
        return ['vehicle', 'infrastructure']

    @property
    def ego_id(self):
        return 'vehicle'

    def __len__(self):
        return len(self._pairs)

    def _load_agent(self, agent_id, k, load):
        pair = self._pairs[k]
        side = self._SIDES[agent_id]
        prefix = agent_id if agent_id == 'vehicle' else 'infrastructure'
        pcd_rel = pair.get(f'{prefix}_pointcloud_path')
        img_rel = pair.get(f'{prefix}_image_path')
        if pcd_rel is None and img_rel is None:
            return None
        fid = _frame_id(pcd_rel or img_rel)
        side_root = os.path.join(self.root, side['dir'])
        calib = os.path.join(side_root, 'calib')

        frame = AgentFrame(
            agent_id=agent_id,
            agent_type='vehicle' if agent_id == 'vehicle' else 'infrastructure',
            pose=self._pose(agent_id, calib, fid),
        )

        # camera calibration
        try:
            intr = _read_json(os.path.join(calib, 'camera_intrinsic',
                                           f'{fid}.json'))
            K = np.asarray(intr['cam_K'], dtype=np.float64).reshape(3, 3)
            T_lidar_to_cam = _rt_to_T(_read_json(
                os.path.join(calib, side['to_cam'], f'{fid}.json')))
            frame.cameras['camera'] = CameraCalib(
                K=K, T_cam_to_agent=np.linalg.inv(T_lidar_to_cam),
                name='camera')
        except (FileNotFoundError, KeyError):
            pass

        if 'lidar' in load and pcd_rel:
            pcd_path = os.path.join(self.root, pcd_rel) \
                if not os.path.isabs(pcd_rel) else pcd_rel
            if os.path.exists(pcd_path):
                frame.lidar = load_pcd(pcd_path)

        if 'images' in load and img_rel:
            img_path = os.path.join(self.root, img_rel) \
                if not os.path.isabs(img_rel) else img_rel
            if os.path.exists(img_path):
                from PIL import Image
                frame.images['camera'] = np.array(
                    Image.open(img_path).convert('RGB'))

        if 'labels' in load and agent_id == 'vehicle':
            frame.labels = self._coop_labels(pair)
        return frame

    # -- helpers ------------------------------------------------------------

    def _pose(self, agent_id, calib, fid):
        try:
            if agent_id == 'vehicle':
                T_l2n = _rt_to_T(_read_json(
                    os.path.join(calib, 'lidar_to_novatel', f'{fid}.json')))
                T_n2w = _rt_to_T(_read_json(
                    os.path.join(calib, 'novatel_to_world', f'{fid}.json')))
                return T_n2w @ T_l2n
            return _rt_to_T(_read_json(
                os.path.join(calib, 'virtuallidar_to_world', f'{fid}.json')))
        except (FileNotFoundError, KeyError):
            return None

    def _coop_labels(self, pair):
        label_rel = pair.get('cooperative_label_path')
        if not label_rel:
            return []
        label_path = os.path.join(self.root, label_rel) \
            if not os.path.isabs(label_rel) else label_rel
        if not os.path.exists(label_path):
            return []
        boxes = []
        for obj in _read_json(label_path):
            loc, dim = obj.get('3d_location'), obj.get('3d_dimensions')
            if not loc or not dim:
                continue
            boxes.append(Box3D(
                center=np.array([loc['x'], loc['y'], loc['z']], dtype=np.float64),
                size=np.array([dim['l'], dim['w'], dim['h']], dtype=np.float64),
                yaw=float(np.degrees(float(obj.get('rotation', 0.0)))),
                category=obj.get('type', ''),
                track_id=str(obj.get('track_id', '')),
                frame='world',
            ))
        return boxes
