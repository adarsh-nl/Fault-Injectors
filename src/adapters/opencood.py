"""
opencood.py
-----------
Adapter between OpenCOOD's ``retrieve_base_data`` dict and the canonical
``CooperativeSample`` / ``AgentFrame`` model.

This module deliberately imports **nothing from OpenCOOD**. It knows the
*schema* of the dict that ``BaseDataset.retrieve_base_data`` returns, not the
package that produces it -- which is why the same file serves both
``~/opencood-official`` and ``~/v2xvit-official`` (their ``basedataset.py``
differ only in docstrings and the ``opencood.``/``v2xvit.`` namespace).

The dict, keyed by ``cav_id`` with the ego first::

    {cav_id: {'ego':       bool,
              'time_delay': int,                    # units of 100 ms
              'params':    {'lidar_pose': [x,y,z,roll,yaw,pitch],
                            'ego_speed':  float,
                            'vehicles':   {...},    # world-frame GT
                            'transformation_matrix':     (4,4),
                            'gt_transformation_matrix':  (4,4),
                            'spatial_correction_matrix': (4,4)},
              'lidar_np':  (N, 4) float32}}

Why the transformation matrix is recomputed
-------------------------------------------
``reform_param`` computes ``transformation_matrix = x1_to_x2(cav_pose,
ego_pose)`` *inside* ``retrieve_base_data``, before it returns, and
``get_item_single_car`` projects the point cloud with **that matrix** -- not
with ``params['lidar_pose']``. Perturbing ``lidar_pose`` after the dict is
handed to us is therefore a silent no-op.

So :meth:`OpenCOODAdapter.from_canonical` rebuilds it from the (possibly
corrupted) canonical poses::

    T = inv(ego.pose) @ agent.pose

which is definitionally ``x1_to_x2(cav_pose, ego_pose)``:
``inv(x_to_world(ego)) @ x_to_world(cav)``. Our ``x_to_world``
(:func:`src.datasets.opv2v.x_to_world`) was checked against OpenCOOD's over 200
random poses spanning +-200 m and +-180 deg -- maximum absolute difference
``0.0``, bit-identical -- so with an empty pipeline the round trip is exact,
not approximate.

``gt_transformation_matrix`` is deliberately **not** recomputed: it exists so
that late fusion transforms *ground truth* with the true pose, and GT must be
identical between the clean and faulty runs or the AP delta is not
attributable. ``spatial_correction_matrix`` is ``eye(4)`` whenever
``cur_ego_pose_flag`` is true (all three of our eval configs), so pose noise
cannot reach it either.
"""

from collections import OrderedDict

import numpy as np

from ..datasets.base import AgentFrame, Box3D, CooperativeSample
from ..datasets.opv2v import x_to_world
from .modality import ModalityError, require_all  # noqa: F401 -- ModalityError
#     is re-exported: it was born here and callers import it from this module.
#     The gate itself moved to `modality.py` so mixed-modality adapters
#     (Griffin) can share per-agent semantics; this adapter keeps the strict
#     require-all form, whose observable behaviour is unchanged.

# Fields of `params` that must survive `to_canonical` -> `from_canonical`
# untouched. Asserted rather than assumed: `from_canonical` rebuilds the dict
# the model consumes wholesale, so a field can quietly get dropped there.
_PASSTHROUGH = ('vehicles', 'ego_speed', 'gt_transformation_matrix',
                'spatial_correction_matrix')

_LIDAR_ONLY_HINT = ('OPV2V/V2XSet as loaded by OpenCOOD are LiDAR-only '
                    '(retrieve_base_data never reads the camera PNGs).')


class OpenCOODAdapter:
    """
    Translate one ``retrieve_base_data`` dict to/from a ``CooperativeSample``.

    Parameters
    ----------
    load_labels : bool
        Populate ``AgentFrame.labels`` from ``params['vehicles']``. Default
        False: no injector in the verified set touches labels, ``vehicles``
        is passed straight back untouched, and building ~50 ``Box3D`` per
        agent per frame across 16 workers is pure overhead. The canonical
        model supports labels and the Griffin adapter will want them; this
        flag is where that gets switched on.
    """

    def __init__(self, load_labels=False):
        self.load_labels = load_labels

    # ── OpenCOOD -> canonical ───────────────────────────────────────────

    def to_canonical(self, base_data_dict, frame_index):
        """Build a ``CooperativeSample`` from one ``retrieve_base_data`` dict."""
        ego_id = None
        for cav_id, entry in base_data_dict.items():
            if entry['ego']:
                ego_id = str(cav_id)
                break
        if ego_id is None:
            raise ValueError('no ego agent in base_data_dict')

        sample = CooperativeSample(frame_index=frame_index, ego_id=ego_id)
        for cav_id, entry in base_data_dict.items():
            params = entry['params']
            aid = str(cav_id)
            sample.agents[aid] = AgentFrame(
                agent_id=aid,
                # V2XSet gives roadside units a negative id.
                agent_type='infrastructure' if _is_infra(cav_id) else 'vehicle',
                is_ego=bool(entry['ego']),
                speed=params.get('ego_speed'),
                pose=x_to_world(params['lidar_pose']),
                lidar=entry['lidar_np'],
                labels=(_boxes_from_vehicles(params.get('vehicles') or {})
                        if self.load_labels else []),
            )

        # `intermediate_fusion_dataset.__getitem__` hard-asserts that the ego
        # is the first key. Record the incoming order so `from_canonical` can
        # reproduce it exactly.
        sample.meta['key_order'] = [str(k) for k in base_data_dict]
        sample.meta['frame_index'] = frame_index
        return sample

    # ── canonical -> OpenCOOD ───────────────────────────────────────────

    def from_canonical(self, sample, base_data_dict, write_back_pose=False):
        """
        Rebuild a ``retrieve_base_data`` dict from a (corrupted) sample.

        Agents removed by a fault (AgentDrop) are absent from the result;
        surviving agents keep their original relative order, ego first.

        ``params['lidar_pose']`` is left **clean** by default. After our seam
        it reaches exactly one consumer -- the 70 m ``COM_RANGE`` gate in
        ``intermediate_fusion_dataset.__getitem__`` -- and OpenCOOD's own
        ``add_loc_noise`` also leaves it clean (it perturbs a local copy), so
        this is protocol-faithful. The corrupted pose is recorded under
        ``params['fi_noisy_lidar_pose']`` for logging either way.
        """
        ego_id = sample.ego_id
        if ego_id not in sample.agents:
            raise ValueError('the ego agent was removed from the sample; '
                             'ego must never be dropped')

        ego_pose = sample.agents[ego_id].pose
        inv_ego = np.linalg.inv(ego_pose)

        out = OrderedDict()
        for key in sample.meta['key_order']:
            if key not in sample.agents:
                continue                          # dropped by AgentDrop
            agent = sample.agents[key]
            entry = base_data_dict[_orig_key(base_data_dict, key)]
            params = entry['params']

            # Cheap presence check every sample: the failure mode here is a
            # field silently disappearing from the dict the model consumes.
            # Deep value equality is Gate 1's job (tests/test_gate1.py).
            missing = [f for f in _PASSTHROUGH if f not in params]
            if missing:
                raise AssertionError(
                    'agent {}: pass-through field(s) {} lost in the '
                    'adapter round trip'.format(key, missing))

            # The one derived quantity a fault is allowed to move.
            params['transformation_matrix'] = inv_ego @ agent.pose
            if write_back_pose:
                params['lidar_pose'] = _pose6_from_matrix(
                    agent.pose, params['lidar_pose'])

            entry['lidar_np'] = np.ascontiguousarray(
                agent.lidar, dtype=np.float32)
            out[_orig_key(base_data_dict, key)] = entry

        first = list(out)[0]
        if not out[first]['ego']:
            raise AssertionError(
                'ego is no longer the first key; '
                'IntermediateFusionDataset asserts on this')
        return out

    # ── modality gate ───────────────────────────────────────────────────

    @staticmethod
    def assert_modality(sample, need):
        """
        Structural block on e.g. image injectors over a LiDAR-only dataset.

        Delegates to the shared gate in :mod:`src.adapters.modality` in its
        strict ``require_all`` form: OPV2V/V2XSet are homogeneous, so "some
        agent lacks the modality" and "the dataset lacks it" coincide, and
        the strict form additionally catches malformed samples. Mixed
        datasets (Griffin) use ``require_any`` / ``agents_supporting`` from
        the same module instead -- per-agent capability, one shared
        implementation, no drift.
        """
        require_all(sample, need, hint=_LIDAR_ONLY_HINT)


# ── helpers ─────────────────────────────────────────────────────────────────

def _is_infra(cav_id):
    try:
        return int(cav_id) < 0
    except (TypeError, ValueError):
        return False


def _orig_key(base_data_dict, str_key):
    """Map our str agent id back to the dict's original key type."""
    if str_key in base_data_dict:
        return str_key
    for k in base_data_dict:
        if str(k) == str_key:
            return k
    raise KeyError(str_key)


def _boxes_from_vehicles(vehicles):
    """OPV2V ``vehicles`` dict -> world-frame ``Box3D`` list."""
    boxes = []
    for vid, v in vehicles.items():
        loc = np.asarray(v['location'], dtype=np.float64)
        ctr = loc + np.asarray(v.get('center', [0.0, 0.0, 0.0]),
                               dtype=np.float64)
        extent = np.asarray(v['extent'], dtype=np.float64)
        angle = v['angle']
        boxes.append(Box3D(center=ctr, size=2.0 * extent, yaw=float(angle[1]),
                           roll=float(angle[0]), pitch=float(angle[2]),
                           category='vehicle', track_id=str(vid),
                           frame='world'))
    return boxes


def _pose6_from_matrix(T, template):
    """
    4x4 -> ``[x, y, z, roll, yaw, pitch]`` in OpenCOOD's CARLA convention.

    Inverse of :func:`src.datasets.opv2v.x_to_world`, whose rotation block is
    ``R = Rz(yaw) Ry(pitch) Rx(roll)`` with
    ``R[2,0] = sin(pitch)``, ``R[1,0]/R[0,0] = tan(yaw)``,
    ``R[2,1]/R[2,2] = -tan(roll)``. Only used when ``write_back_pose=True``.
    """
    T = np.asarray(T, dtype=np.float64)
    pitch = np.degrees(np.arcsin(np.clip(T[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    roll = np.degrees(np.arctan2(-T[2, 1], T[2, 2]))
    out = list(template)
    out[0], out[1], out[2] = T[0, 3], T[1, 3], T[2, 3]
    out[3], out[4], out[5] = roll, yaw, pitch
    return out
