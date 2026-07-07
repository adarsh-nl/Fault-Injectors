"""
datasets
--------
Dataset adapters: every supported dataset is normalised into the same
cooperative sample model (`CooperativeSample` of `AgentFrame`s) so the
fault injectors and pipeline never see dataset-specific formats.

    from src.datasets import load_dataset
    ds = load_dataset('opv2v', scenario_dir='.../2021_08_18_09_02_56')
    sample = ds.get_sample(0)                 # all agents, frame 0
    pts_ego = sample.lidar_in_ego_frame('650')

Supported (see each module for the on-disk layout it expects):
    griffin   GriffinDataset    aerial-ground (vehicle + drone)
    opv2v     OPV2VDataset      OpenCOOD format (V2VNet/CoBEVT/Where2comm)
    v2xset    V2XSetDataset     OPV2V format + infrastructure agents
    dair-v2x  DairV2XDataset    real vehicle + infrastructure pairs

Adding a new dataset = subclassing `BaseDataset` (3 methods) and calling
`register_dataset`; see docs/datasets.md.
"""

from .base import (
    AGENT_TYPES,
    AgentFrame,
    BaseDataset,
    Box3D,
    CameraCalib,
    CooperativeSample,
    transform_points,
)
from .dair_v2x import DairV2XDataset
from .griffin import GriffinDataset
from .opv2v import OPV2VDataset, V2XSetDataset, x_to_world
from .pcd import load_pcd

_REGISTRY = {
    'griffin' : GriffinDataset,
    'opv2v'   : OPV2VDataset,
    'v2xset'  : V2XSetDataset,
    'dair-v2x': DairV2XDataset,
}


def register_dataset(name, cls):
    """Register a BaseDataset subclass under a name for load_dataset()."""
    if not (isinstance(cls, type) and issubclass(cls, BaseDataset)):
        raise TypeError('cls must be a BaseDataset subclass')
    _REGISTRY[name.lower()] = cls


def available_datasets():
    return sorted(_REGISTRY)


def load_dataset(name, *args, **kwargs):
    """Instantiate a registered dataset adapter by name."""
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f'unknown dataset {name!r}; available: '
                         f'{available_datasets()}') from None
    return cls(*args, **kwargs)


__all__ = [
    'AGENT_TYPES', 'AgentFrame', 'BaseDataset', 'Box3D', 'CameraCalib',
    'CooperativeSample', 'transform_points', 'load_pcd', 'x_to_world',
    'GriffinDataset', 'OPV2VDataset', 'V2XSetDataset', 'DairV2XDataset',
    'register_dataset', 'available_datasets', 'load_dataset',
]
