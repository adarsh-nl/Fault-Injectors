"""
Shared fixtures for the lgcpbench test suite.

Everything here is CPU-only, seconds-fast, and needs no dataset on disk and
no OpenCOOD install -- the standing requirement that every module runs
independently.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch

from cpbench.data.preprocessing import AnchorGenerator, GridSpec
from lgcpbench.confidence import AreaConfidenceEstimator
from lgcpbench.network import (
    FusionLatencyModel,
    InterferenceModel,
    LatencyModel,
    TransmissionScheduler,
)
from lgcpbench.orchestration import GlobalViewAggregator, LGCPPipeline, RSUController
from lgcpbench.orchestration.pipeline import FrameInput
from lgcpbench.perception import AgentInputs, AreaFeatureMasker, NativeReferenceBackbone
from lgcpbench.perception.decode import AreaBoxDecoder
from lgcpbench.roi import AreaGrid, make_occupancy_estimator
from lgcpbench.selection import GreedyGroupSelector, SelectionAlgorithm

# A small RoI so the whole pipeline runs in milliseconds. Voxel 0.4 m and
# downsample 4 preserve the paper's 1.6 m feature stride, so the area/cell
# arithmetic is identical to OPV2V's -- only the extent is smaller.
#
# The extent is chosen so the pillar canvas (192 x 64) divides evenly by the
# backbone's total stride of 16 (downsample 4 x the 3-level pyramid), the
# same property OPV2V's 704 x 192 canvas has.
TEST_POINT_RANGE = (-38.4, -12.8, -3.0, 38.4, 12.8, 1.0)
TEST_CHANNELS = 8


@pytest.fixture(scope="session")
def spec() -> GridSpec:
    return GridSpec(voxel_size=(0.4, 0.4), point_range=TEST_POINT_RANGE, downsample=4)


@pytest.fixture(scope="session")
def area_grid(spec: GridSpec) -> AreaGrid:
    return AreaGrid.from_grid_spec(spec)


@pytest.fixture(scope="session")
def masker(area_grid: AreaGrid, spec: GridSpec) -> AreaFeatureMasker:
    return AreaFeatureMasker(area_grid, feature_hw=spec.feature_hw)


@pytest.fixture
def backbone(spec: GridSpec) -> NativeReferenceBackbone:
    """A backbone behaving as if trained.

    ``DetectionHead`` initialises its classification bias to -4.59, a
    focal-loss prior that puts sigmoid at ~0.01. That is correct for starting
    training, but it means EVERY area confidence sits below dg, so Algorithm 1
    admits nobody and every area orphans -- which would let orchestration
    tests pass vacuously on empty groups.

    Zeroing the bias gives confidences spread around 0.5, which is what a
    trained detector produces and what the control plane needs to exercise.
    ``test_untrained_backbone_orphans_every_area`` pins the untrained
    behaviour separately, since it is itself a meaningful system state.
    """
    torch.manual_seed(0)
    model = NativeReferenceBackbone(
        grid_hw=spec.grid_hw,
        feature_hw=spec.feature_hw,
        channels=TEST_CHANNELS,
        downsample=spec.downsample,
    )
    with torch.no_grad():
        model.head.cls_head.bias.zero_()
    return model.eval()


@pytest.fixture
def untrained_backbone(spec: GridSpec) -> NativeReferenceBackbone:
    """The as-initialised backbone, focal-loss bias intact."""
    torch.manual_seed(0)
    return NativeReferenceBackbone(
        grid_hw=spec.grid_hw,
        feature_hw=spec.feature_hw,
        channels=TEST_CHANNELS,
        downsample=spec.downsample,
    ).eval()


@pytest.fixture
def cav_positions() -> np.ndarray:
    """Four CAVs spread along the road so areas differ per CAV."""
    return np.array([[-30.0, -4.0], [-10.0, 4.0], [10.0, -4.0], [30.0, 4.0]])


@pytest.fixture
def agent_ids() -> Tuple[str, ...]:
    return ("cav0", "cav1", "cav2", "cav3")


@pytest.fixture
def agents(spec: GridSpec, cav_positions: np.ndarray,
           agent_ids: Tuple[str, ...]) -> AgentInputs:
    """Collated pillar inputs for four CAVs, deterministic."""
    rng = np.random.default_rng(0)
    n_agents = len(agent_ids)
    h0, w0 = spec.grid_hw
    n_pillars = 120
    coords = np.stack(
        [
            rng.integers(0, n_agents, size=n_pillars),
            rng.integers(0, h0, size=n_pillars),
            rng.integers(0, w0, size=n_pillars),
        ],
        axis=1,
    )
    return AgentInputs(
        features=torch.from_numpy(rng.normal(size=(n_pillars, 32, 9))).float(),
        coords=torch.from_numpy(coords).long(),
        num_points=torch.from_numpy(rng.integers(1, 32, size=n_pillars)).long(),
        n_agents=n_agents,
        agent_ids=agent_ids,
        positions=cav_positions,
        ego_index=0,
    )


@pytest.fixture
def gt_boxes() -> np.ndarray:
    """Six objects spread across the RoI, so several areas are occupied."""
    centres = np.array(
        [[-30.0, -4.0], [-12.0, 3.0], [0.0, 0.0], [11.0, -3.0], [25.0, 5.0], [33.0, -6.0]]
    )
    boxes = np.zeros((len(centres), 7), dtype=np.float32)
    boxes[:, :2] = centres
    boxes[:, 2] = -1.0
    boxes[:, 3:6] = (3.9, 1.6, 1.56)
    return boxes


@pytest.fixture
def rsu(area_grid: AreaGrid, spec: GridSpec, cav_positions: np.ndarray,
        agent_ids: Tuple[str, ...]) -> RSUController:
    positions = {aid: tuple(p) for aid, p in zip(agent_ids, cav_positions)}
    scheduler = TransmissionScheduler(
        InterferenceModel(positions, interference_range_m=1e6),
        fusion_model=FusionLatencyModel.for_model("where2comm"),
    )
    return RSUController(
        grid=area_grid,
        occupancy=make_occupancy_estimator("gt"),
        confidence=AreaConfidenceEstimator(area_grid, spec.feature_hw, pooling="max"),
        selection=SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075)),
        scheduler=scheduler,
        latency=LatencyModel(rate_bps=27e6),
        aggregator=GlobalViewAggregator(mode="union"),
    )


@pytest.fixture
def decoder(spec: GridSpec, area_grid: AreaGrid) -> AreaBoxDecoder:
    return AreaBoxDecoder(
        AnchorGenerator(spec), area_grid, spec.feature_hw, score_threshold=0.2
    )


@pytest.fixture
def pipeline(backbone, rsu, masker, decoder) -> LGCPPipeline:
    return LGCPPipeline(backbone=backbone, rsu=rsu, masker=masker, decoder=decoder)


@pytest.fixture
def frame(agents: AgentInputs, gt_boxes: np.ndarray) -> FrameInput:
    return FrameInput(index=0, agents=agents, gt_boxes=gt_boxes)
