"""Every shipped dataset config must produce a grid the default backbone can
consume. Guards against a config that raises only at model construction time,
i.e. after a job has already queued."""
from pathlib import Path

import pytest

from cpbench.models.encoder import validate_backbone_geometry
from cpbench.utils.config import load_config
from corabench.scripts.common import build_grid

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
DATASETS = [p.stem for p in (CONFIG.parent / "dataset").glob("*.yaml")]


@pytest.mark.parametrize("name", DATASETS)
def test_shipped_dataset_config_matches_backbone_stride(name):
    cfg = load_config(str(CONFIG), [f"dataset={name}"])
    validate_backbone_geometry(build_grid(cfg["dataset"]),
                               tuple(cfg["model"]["block_strides"]))
