"""
griffin.py
----------
Griffin (arXiv:2503.06983) adapter: aerial-ground cooperative perception.

Two agents:
    'vehicle' (ego)  4 cameras (front/back/left/right), lidar_top, pose,
                     ego-frame 3-D labels.
    'drone'          up to 5 cameras (front/back/left/right/bottom), pose
                     when a pose/ folder exists on the drone side.

Layout expected under each side's root (see src/download_griffin.py):
    camera/<name>/*.png   lidar/lidar_top/*.ply   pose/*.json
    label/*.txt           calib/<sensor>.json

Conventions mapped to the common model:
    * agent frame  = Griffin ego frame (LiDAR mount already applied by
                     `load_lidar`); labels are ego-frame -> Box3D(frame='agent').
    * pose         = T_agent_to_world from pose/*.json (ENU world frame).
"""

import glob
import json
import os

import numpy as np

from ..data_loaders import (
    load_image, load_lidar, load_pose_griffin, load_calib_griffin,
    parse_label_txt,
)
from .base import AgentFrame, BaseDataset, Box3D, CameraCalib

_VEH_CAMERAS   = ('front', 'back', 'left', 'right')
_DRONE_CAMERAS = ('front', 'back', 'left', 'right', 'bottom')


def _sorted(base, *parts):
    return sorted(glob.glob(os.path.join(base, *parts)))


class GriffinDataset(BaseDataset):
    """
    Parameters
    ----------
    veh_root   : vehicle-side directory.
    drone_root : drone-side directory, or None for vehicle-only.
    cameras    : restrict camera names to load (default: all available).
    """

    name = 'griffin'
    fps = 10.0

    def __init__(self, veh_root, drone_root=None, cameras=None):
        self.roots = {'vehicle': veh_root}
        if drone_root:
            self.roots['drone'] = drone_root
        self._cameras = set(cameras) if cameras else None

        self._files = {}
        for aid, root in self.roots.items():
            cams = _VEH_CAMERAS if aid == 'vehicle' else _DRONE_CAMERAS
            self._files[aid] = {
                'images': {c: _sorted(root, 'camera', c, '*.png') for c in cams
                           if self._cameras is None or c in self._cameras},
                'lidar' : _sorted(root, 'lidar', 'lidar_top', '*.ply'),
                'pose'  : _sorted(root, 'pose', '*.json'),
                'label' : _sorted(root, 'label', '*.txt'),
            }

        n = len(self._files['vehicle']['pose']) or len(
            self._files['vehicle']['lidar'])
        if n == 0:
            raise FileNotFoundError(
                f'no pose/ or lidar/ frames under {veh_root!r} -- is this a '
                f'Griffin vehicle-side directory?')
        self._n = n

    # -- BaseDataset --------------------------------------------------------

    def agent_ids(self):
        return list(self.roots)

    @property
    def ego_id(self):
        return 'vehicle'

    def __len__(self):
        return self._n

    def _load_agent(self, agent_id, k, load):
        files = self._files[agent_id]
        root  = self.roots[agent_id]
        frame = AgentFrame(
            agent_id=agent_id,
            agent_type='vehicle' if agent_id == 'vehicle' else 'drone',
        )

        # pose + timestamp
        if k < len(files['pose']):
            T_world_to_agent, raw = load_pose_griffin(files['pose'][k])
            frame.pose = np.linalg.inv(T_world_to_agent)
            ts = raw.get('timestamp')
            frame.timestamp = float(ts) if ts is not None else None
        elif agent_id != 'vehicle':
            # drone subsets without pose/: still usable for camera faults
            frame.pose = None

        # calibration (cheap, always loaded)
        calib_dir = os.path.join(root, 'calib')
        for cam in files['images']:
            try:
                K, T_agent_to_cam = load_calib_griffin(calib_dir, cam)
                frame.cameras[cam] = CameraCalib(
                    K=K, T_cam_to_agent=np.linalg.inv(T_agent_to_cam), name=cam)
            except (FileNotFoundError, KeyError):
                pass

        if 'lidar' in load and k < len(files['lidar']):
            frame.lidar = load_lidar(files['lidar'][k])

        if 'images' in load:
            for cam, paths in files['images'].items():
                if k < len(paths):
                    frame.images[cam] = load_image(paths[k])

        if 'labels' in load and agent_id == 'vehicle':
            frame.labels = self._labels_for_frame(k)

        if frame.pose is None and frame.lidar is None and not frame.images:
            return None
        return frame

    # -- helpers ------------------------------------------------------------

    def _labels_for_frame(self, k):
        files = self._files['vehicle']
        label_path = None
        if k < len(files['pose']):
            with open(files['pose'][k]) as f:
                ts = int(json.load(f)['timestamp'])
            cand = os.path.join(self.roots['vehicle'], 'label', f'{ts:06d}.txt')
            if os.path.exists(cand):
                label_path = cand
        if label_path is None and k < len(files['label']):
            label_path = files['label'][k]
        if label_path is None:
            return []
        return [
            Box3D(
                center=np.array([a['x'], a['y'], a['z']]),
                size=np.array([a['l'], a['w'], a['h']]),
                yaw=a['yaw'], roll=a['roll'], pitch=a['pitch'],
                category=a['category'], track_id=a['id'], frame='agent',
                extra={'visibility': a['visibility']},
            )
            for a in parse_label_txt(label_path)
        ]
