# Dataset adapters: one sample model for every dataset

The fault injectors never see dataset-specific formats. Every dataset is
normalised by an adapter in `src/datasets/` into the same **cooperative
sample model**, and everything downstream (fault pipeline, visualisation,
information-quality analysis) consumes that model.

## The sample model

```
CooperativeSample                      # one multi-agent frame at time k
  frame_index : int
  ego_id      : str
  agents      : {agent_id: AgentFrame}
  meta        : dict                   # fps, dropped_agents, ...

AgentFrame                             # one platform's contribution
  agent_id    : str
  agent_type  : 'vehicle' | 'infrastructure' | 'drone'
  is_ego      : bool
  timestamp   : float | None           # seconds
  pose        : (4, 4) T_agent_to_world | None
  lidar       : (N, C>=3) float32 in the AGENT frame | None
  images      : {camera_name: (H, W, 3) uint8}
  cameras     : {camera_name: CameraCalib(K, T_cam_to_agent)}
  labels      : [Box3D]                # frame='agent' or 'world' per box
  faults      : dict                   # log of everything injected
```

Conventions (adapters must guarantee these):

* **Agent frame** — a metric, right-handed frame rigidly attached to the
  platform; all of the agent's sensor data is expressed in it. For OPV2V /
  V2XSet / DAIR-V2X the agent frame *is* the LiDAR frame; for Griffin it is
  the ego frame (LiDAR mount extrinsic already applied).
* **pose** — `T_agent_to_world`. Cross-agent fusion is
  `inv(T_ego_to_world) @ T_agent_to_world`, available as
  `sample.lidar_in_ego_frame(agent_id)`. Pose-error injection corrupts
  exactly this quantity, which is why it degrades fusion.
* Degrees, metres, seconds. Box sizes are full extents `(l, w, h)`.

## Built-in adapters

| name       | class            | datasets / papers served                        |
|------------|------------------|--------------------------------------------------|
| `griffin`  | `GriffinDataset` | Griffin aerial-ground (arXiv:2503.06983)         |
| `opv2v`    | `OPV2VDataset`   | OPV2V — V2VNet re-impl, CoBEVT, Where2comm       |
| `v2xset`   | `V2XSetDataset`  | V2XSet — V2X-ViT (negative cav ids = infra)      |
| `dair-v2x` | `DairV2XDataset` | DAIR-V2X-C real vehicle + infrastructure         |

```python
from src.datasets import load_dataset

ds = load_dataset('opv2v', '/data/opv2v/test/2021_08_18_09_02_56')
ds = load_dataset('v2xset', '/data/v2xset/test/<scenario>')
ds = load_dataset('dair-v2x', '/data/DAIR-V2X/cooperative-vehicle-infrastructure')
ds = load_dataset('griffin', veh_root='.../vehicle-side',
                  drone_root='.../drone-side')

sample = ds.get_sample(0)                       # all agents, frame 0
sample = ds.get_sample(0, load=('lidar',))      # skip images/labels (fast)
pts    = sample.lidar_in_ego_frame('650')       # fusion-side view
```

Each adapter's module docstring documents the exact on-disk layout it
expects and how the dataset's conventions map onto the model
(`src/datasets/opv2v.py`, `dair_v2x.py`, `griffin.py`).

Point clouds are read with a dependency-free PCD reader
(`src/datasets/pcd.py`, ASCII + binary) — no open3d required.

## Adding a new dataset

Subclass `BaseDataset` and implement three methods; register it and every
fault injector, the pipeline and the sweep tooling work on it unchanged.

```python
from src.datasets import BaseDataset, AgentFrame, register_dataset

class MyDataset(BaseDataset):
    name = 'mydata'
    fps  = 10.0

    def __init__(self, root):
        ...                       # index the files, decide the agent list

    def agent_ids(self):
        return ['ego_vehicle', 'rsu_1']

    def __len__(self):
        return self._n_frames

    def _load_agent(self, agent_id, k, load):
        # return an AgentFrame for agent_id at ego-frame k,
        # or None if the agent is absent at k.
        frame = AgentFrame(agent_id=agent_id, agent_type='vehicle')
        frame.pose = ...                          # (4,4) T_agent_to_world
        if 'lidar' in load:
            frame.lidar = ...                     # (N, C) agent frame
        if 'images' in load:
            frame.images['front'] = ...           # (H, W, 3) uint8
        if 'labels' in load:
            frame.labels = [...]                  # Box3D list
        return frame

register_dataset('mydata', MyDataset)
```

Guidelines:

* `_load_agent` must honour `load` — only read the heavy payloads asked
  for. Poses/calib/timestamps are always loaded (faults need them).
* Convert whatever the dataset stores (Euler poses, quaternion poses,
  rotation/translation json) into `T_agent_to_world` **once**, in the
  adapter. Nothing downstream should ever see raw dataset conventions.
* If the ego is not your first agent id, override the `ego_id` property.
* Add a synthetic-tree test like the ones in `src/tests/test_datasets.py`
  (see `src/tests/synthetic.py` for the fixture builders); the adapters
  are tested without any real data downloaded.
