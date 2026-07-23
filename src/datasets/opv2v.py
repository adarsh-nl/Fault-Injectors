"""
opv2v.py
--------
OPV2V (arXiv:2109.07644) / V2XSet (V2X-ViT, arXiv:2203.10638) adapter.

This is the OpenCOOD on-disk format used by the reference implementations
of V2VNet*, CoBEVT, Where2comm and V2X-ViT, so samples from this adapter
correspond one-to-one with what those models consume.
(*V2VNet as re-implemented in OpenCOOD.)

Layout: one SCENARIO directory containing one folder per agent (CAV),
each folder holding per-timestamp files:

    <scenario>/
        <cav_id>/                     e.g. 641/ ... or -1/ (V2XSet infra)
            000068.yaml               pose, speed, camera calib, GT vehicles
            000068.pcd                LiDAR in the agent's LiDAR frame
            000068_camera0..3.png     4 cameras (optional)

Conventions mapped to the common model:
    * agent frame  = the agent's LiDAR frame (that is what the .pcd is in
      and what `lidar_pose` localises), so pose = x_to_world(lidar_pose)
      with OpenCOOD's exact CARLA angle convention.
    * ego          = the numerically smallest cav id (OpenCOOD sorts and
      takes the first), overridable via `ego_id=`.
    * agent type   = 'infrastructure' when int(cav_id) < 0 (V2XSet
      convention), else 'vehicle'.
    * labels       = the yaml's `vehicles` dict -> world-frame Box3D
      (center = location + center offset, size = 2 * extent, yaw = angle[1]).

Timestamps are taken from the EGO's yaml files; other agents contribute a
frame only where they have the same timestamp on disk.
"""

import glob
import os

import numpy as np

from .base import AgentFrame, BaseDataset, Box3D, CameraCalib
from .pcd import load_pcd

try:
    import yaml
except ImportError:                                          # pragma: no cover
    yaml = None


def x_to_world(pose6):
    """
    OpenCOOD's exact pose -> 4x4 conversion (CARLA convention).
    pose6 = [x, y, z, roll, yaw, pitch] in metres / degrees.
    Returns T_agent_to_world.
    """
    x, y, z, roll, yaw, pitch = pose6
    c_y, s_y = np.cos(np.radians(yaw)),   np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)),  np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

    T = np.identity(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    T[0, 0] = c_p * c_y
    T[0, 1] = c_y * s_p * s_r - s_y * c_r
    T[0, 2] = -c_y * s_p * c_r - s_y * s_r
    T[1, 0] = s_y * c_p
    T[1, 1] = s_y * s_p * s_r + c_y * c_r
    T[1, 2] = -s_y * s_p * c_r + c_y * s_r
    T[2, 0] = s_p
    T[2, 1] = -c_p * s_r
    T[2, 2] = c_p * c_r
    return T


def _load_yaml(path):
    if yaml is None:
        raise ImportError('pyyaml is required for the OPV2V/V2XSet adapter '
                          '(pip install pyyaml)')
    with open(path) as f:
        return yaml.safe_load(f)


class OPV2VDataset(BaseDataset):
    """
    One OPV2V / V2XSet scenario.

    Parameters
    ----------
    scenario_dir : path to one scenario folder (contains cav_id subfolders).
    ego_id       : override the default ego (smallest cav id), as a string.
    max_cams     : number of cameras per agent to expose (default 4).
    """

    name = 'opv2v'
    fps = 10.0

    def __init__(self, scenario_dir, ego_id=None, max_cams=4):
        self.scenario_dir = scenario_dir
        self.max_cams = max_cams

        cav_dirs = sorted(
            (d for d in glob.glob(os.path.join(scenario_dir, '*'))
             if os.path.isdir(d) and _is_int(os.path.basename(d))),
            key=lambda d: int(os.path.basename(d)))
        if not cav_dirs:
            raise FileNotFoundError(
                f'no numeric CAV folders under {scenario_dir!r} -- expected '
                f'an OPV2V/V2XSet scenario directory')
        self._cav_dirs = {os.path.basename(d): d for d in cav_dirs}

        if ego_id is not None:
            if str(ego_id) not in self._cav_dirs:
                raise ValueError(f'ego_id {ego_id!r} not among CAVs '
                                 f'{list(self._cav_dirs)}')
            self._ego = str(ego_id)
        else:
            self._ego = os.path.basename(cav_dirs[0])

        self._timestamps = [
            os.path.basename(p)[:-5] for p in
            sorted(glob.glob(os.path.join(self._cav_dirs[self._ego], '*.yaml')))
            if not os.path.basename(p).startswith('data_protocol')
        ]
        if not self._timestamps:
            raise FileNotFoundError(
                f'ego CAV {self._ego} has no .yaml frames in {scenario_dir!r}')

    # -- BaseDataset --------------------------------------------------------

    def agent_ids(self):
        return list(self._cav_dirs)

    @property
    def ego_id(self):
        return self._ego

    def __len__(self):
        return len(self._timestamps)

    def timestamp_key(self, k):
        """The on-disk timestamp string of ego frame k (e.g. '000068')."""
        return self._timestamps[k]

    def _load_agent(self, agent_id, k, load):
        ts = self._timestamps[k]
        cav_dir = self._cav_dirs[agent_id]
        yaml_path = os.path.join(cav_dir, f'{ts}.yaml')
        if not os.path.exists(yaml_path):
            return None                       # agent absent at this timestamp
        params = _load_yaml(yaml_path)

        speed = params.get('ego_speed')
        frame = AgentFrame(
            agent_id=agent_id,
            agent_type='infrastructure' if int(agent_id) < 0 else 'vehicle',
            timestamp=int(ts) / self.fps if ts.isdigit() else None,
            speed=float(speed) if speed is not None else None,
            pose=x_to_world(params['lidar_pose']),
        )

        for i in range(self.max_cams):
            cam = params.get(f'camera{i}')
            if cam is None:
                continue
            frame.cameras[f'camera{i}'] = CameraCalib(
                K=np.asarray(cam['intrinsic'], dtype=np.float64),
                # yaml 'extrinsic' is lidar->camera; invert to cam->agent
                T_cam_to_agent=np.linalg.inv(
                    np.asarray(cam['extrinsic'], dtype=np.float64)),
                name=f'camera{i}')

        if 'lidar' in load:
            pcd_path = os.path.join(cav_dir, f'{ts}.pcd')
            if os.path.exists(pcd_path):
                frame.lidar = load_pcd(pcd_path)

        if 'images' in load:
            from PIL import Image
            for i in range(self.max_cams):
                img_path = os.path.join(cav_dir, f'{ts}_camera{i}.png')
                if os.path.exists(img_path):
                    frame.images[f'camera{i}'] = np.array(
                        Image.open(img_path).convert('RGB'))

        if 'labels' in load:
            frame.labels = [
                Box3D(
                    center=np.asarray(v['location'], dtype=np.float64)
                           + np.asarray(v.get('center', [0, 0, 0]),
                                        dtype=np.float64),
                    size=2.0 * np.asarray(v['extent'], dtype=np.float64),
                    roll=v['angle'][0], yaw=v['angle'][1], pitch=v['angle'][2],
                    category='vehicle', track_id=str(vid), frame='world',
                    extra={'speed': v.get('speed')},
                )
                for vid, v in (params.get('vehicles') or {}).items()
            ]
        return frame


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class V2XSetDataset(OPV2VDataset):
    """V2XSet uses the OPV2V format verbatim; negative cav ids are infra."""
    name = 'v2xset'
