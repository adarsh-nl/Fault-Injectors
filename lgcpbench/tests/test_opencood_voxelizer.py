"""
Tests for the OpenCOOD voxeliser and the OpenCOOD data path.

These verify the tensor CONTRACT against what OpenCOOD's models actually
dereference, read from
``opencood/data_utils/pre_processor/sp_voxel_preprocessor.py``:

    preprocess  -> voxel_features (N, max_pts, 4), voxel_coords (N, 3) [z,y,x],
                   voxel_num_points (N,)
    collate     -> coords padded with the agent index in column 0, (N, 4)

They do NOT verify byte-equality with spconv -- see the module docstring for
why voxel ordering differs and why that is irrelevant to the model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.data.preprocessing import GridSpec
from cpbench.data.synthetic import SyntheticCooperativeDataset
from cpbench.faults.bridge import DataFaultBridge
from lgcpbench.data import LGCPDataset, OpenCOODVoxelizer

OPV2V_RANGE = (-140.8, -38.4, -3.0, 140.8, 38.4, 1.0)
SMALL_RANGE = (-38.4, -12.8, -3.0, 38.4, 12.8, 1.0)


@pytest.fixture
def vox() -> OpenCOODVoxelizer:
    return OpenCOODVoxelizer(cav_lidar_range=OPV2V_RANGE)


def _points(n: int = 500, seed: int = 0, rng_range=OPV2V_RANGE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = np.array(rng_range[:3]), np.array(rng_range[3:])
    xyz = rng.uniform(lo, hi, size=(n, 3))
    intensity = rng.uniform(0, 1, size=(n, 1))
    return np.concatenate([xyz, intensity], axis=1).astype(np.float32)


# --------------------------------------------------------------------- #
# the PointPillar degeneracy this implementation relies on
# --------------------------------------------------------------------- #


def test_pointpillar_config_has_exactly_one_z_voxel(vox: OpenCOODVoxelizer) -> None:
    """The whole reason spconv is not needed: voxel_size [0.4, 0.4, 4] over a
    4 m z-range gives one z-voxel, so 3-D voxelisation degenerates to pillar
    voxelisation."""
    assert vox.grid_size.tolist() == [704, 192, 1]


def test_multi_z_voxel_config_is_refused_with_guidance() -> None:
    """A config outside the degenerate case must fail loudly rather than
    silently produce differently-shaped tensors."""
    with pytest.raises(ValueError, match="single z-voxel"):
        OpenCOODVoxelizer(cav_lidar_range=OPV2V_RANGE, voxel_size=(0.4, 0.4, 0.5))


def test_from_grid_spec_derives_the_z_extent() -> None:
    spec = GridSpec((0.4, 0.4), OPV2V_RANGE, 4)
    v = OpenCOODVoxelizer.from_grid_spec(spec)
    assert v.voxel_size == (0.4, 0.4, 4.0)
    assert v.grid_size[2] == 1


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError):
        OpenCOODVoxelizer(cav_lidar_range=(0, 0, 0, 1, 1))
    with pytest.raises(ValueError):
        OpenCOODVoxelizer(cav_lidar_range=OPV2V_RANGE, voxel_size=(0.4, 0.4))


# --------------------------------------------------------------------- #
# preprocess -- the per-CAV contract
# --------------------------------------------------------------------- #


def test_preprocess_shapes_match_opencood(vox: OpenCOODVoxelizer) -> None:
    out = vox.preprocess(_points())
    n = out["voxel_features"].shape[0]
    assert out["voxel_features"].shape == (n, 32, 4)
    assert out["voxel_coords"].shape == (n, 3)
    assert out["voxel_num_points"].shape == (n,)
    assert out["voxel_features"].dtype == np.float32
    assert out["voxel_coords"].dtype == np.int32


def test_coords_are_z_y_x_and_in_range(vox: OpenCOODVoxelizer) -> None:
    """OpenCOOD's coords are [z, y, x] -- the REVERSE of the [x, y, z] index
    order. Getting this backwards would scatter every feature to the wrong
    BEV cell while every shape assertion still passed."""
    coords = vox.preprocess(_points(2000))["voxel_coords"]
    nx, ny, nz = vox.grid_size
    assert coords[:, 0].max() < nz and coords[:, 0].min() >= 0
    assert coords[:, 1].max() < ny and coords[:, 1].min() >= 0
    assert coords[:, 2].max() < nx and coords[:, 2].min() >= 0
    assert np.all(coords[:, 0] == 0)      # single z-voxel


def test_a_known_point_lands_in_the_expected_cell(vox: OpenCOODVoxelizer) -> None:
    """Anchors the coordinate convention to arithmetic, not to itself."""
    pts = np.array([[-140.8 + 0.2, -38.4 + 0.2, -2.0, 0.5]], dtype=np.float32)
    coords = vox.preprocess(pts)["voxel_coords"]
    assert coords.tolist() == [[0, 0, 0]]

    pts = np.array([[-140.8 + 4.2, -38.4 + 2.2, -2.0, 0.5]], dtype=np.float32)
    coords = vox.preprocess(pts)["voxel_coords"]
    assert coords.tolist() == [[0, 5, 10]]   # 2.2/0.4=5 (y), 4.2/0.4=10 (x)


def test_points_sharing_a_cell_share_a_voxel(vox: OpenCOODVoxelizer) -> None:
    pts = np.array(
        [[0.0, 0.0, 0.0, 0.1], [0.1, 0.1, 0.5, 0.2], [0.35, 0.35, -1.0, 0.3]],
        dtype=np.float32,
    )
    out = vox.preprocess(pts)
    assert out["voxel_features"].shape[0] == 1
    assert int(out["voxel_num_points"][0]) == 3


def test_num_points_totals_the_retained_points(vox: OpenCOODVoxelizer) -> None:
    pts = _points(3000, seed=1)
    out = vox.preprocess(pts)
    assert int(out["voxel_num_points"].sum()) == len(pts)


def test_points_outside_the_range_are_dropped(vox: OpenCOODVoxelizer) -> None:
    inside = np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
    outside = np.array([[1e4, 1e4, 1e4, 0.5], [0.0, 0.0, 1e3, 0.5]], dtype=np.float32)
    out = vox.preprocess(np.concatenate([inside, outside]))
    assert out["voxel_features"].shape[0] == 1
    assert int(out["voxel_num_points"].sum()) == 1


def test_empty_cloud_gives_well_formed_empty_tensors(vox: OpenCOODVoxelizer) -> None:
    """A dropped CAV is a legitimate fault outcome, not an error."""
    out = vox.preprocess(np.zeros((0, 4), dtype=np.float32))
    assert out["voxel_features"].shape == (0, 32, 4)
    assert out["voxel_coords"].shape == (0, 3)


def test_max_points_per_voxel_is_respected() -> None:
    v = OpenCOODVoxelizer(cav_lidar_range=OPV2V_RANGE, max_points_per_voxel=4)
    pts = np.tile(np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32), (20, 1))
    out = v.preprocess(pts)
    assert out["voxel_features"].shape[1] == 4
    assert int(out["voxel_num_points"][0]) == 4


def test_max_voxels_truncates_with_a_warning(caplog) -> None:
    v = OpenCOODVoxelizer(cav_lidar_range=OPV2V_RANGE, max_voxels=5)
    out = v.preprocess(_points(500, seed=2))
    assert out["voxel_features"].shape[0] == 5
    assert any("max_voxels" in r.message for r in caplog.records)


def test_short_point_features_are_padded(vox: OpenCOODVoxelizer) -> None:
    """The models are built for 4 features; a 3-column cloud must not crash."""
    out = vox.preprocess(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert out["voxel_features"].shape[-1] == 4


def test_extra_point_features_are_truncated(vox: OpenCOODVoxelizer) -> None:
    out = vox.preprocess(np.zeros((3, 7), dtype=np.float32))
    assert out["voxel_features"].shape[-1] == 4


def test_malformed_input_is_rejected(vox: OpenCOODVoxelizer) -> None:
    with pytest.raises(ValueError):
        vox.preprocess(np.zeros((5,), dtype=np.float32))


def test_preprocess_is_deterministic(vox: OpenCOODVoxelizer) -> None:
    pts = _points(1000, seed=3)
    a, b = vox.preprocess(pts), vox.preprocess(pts)
    assert np.array_equal(a["voxel_coords"], b["voxel_coords"])
    assert np.array_equal(a["voxel_features"], b["voxel_features"])


# --------------------------------------------------------------------- #
# collate -- the batched contract
# --------------------------------------------------------------------- #


def test_collate_prepends_the_agent_index(vox: OpenCOODVoxelizer) -> None:
    """``collate_batch_list`` pads coords with the entry index in column 0, so
    the model receives (N, 4) = [agent, z, y, x]."""
    per_cav = [vox.preprocess(_points(200, seed=s)) for s in range(3)]
    out = vox.collate(per_cav)

    assert out["voxel_coords"].shape[1] == 4
    assert set(out["voxel_coords"][:, 0].tolist()) == {0, 1, 2}
    for i, item in enumerate(per_cav):
        assert int((out["voxel_coords"][:, 0] == i).sum()) == len(item["voxel_coords"])


def test_collate_returns_torch_tensors_of_the_expected_dtypes(vox: OpenCOODVoxelizer) -> None:
    out = vox.collate([vox.preprocess(_points(100))])
    assert out["voxel_features"].dtype == torch.float32
    assert out["voxel_coords"].dtype == torch.int64
    assert out["voxel_num_points"].dtype == torch.int64


def test_collate_concatenates_all_agents(vox: OpenCOODVoxelizer) -> None:
    per_cav = [vox.preprocess(_points(300, seed=s)) for s in range(4)]
    out = vox.collate(per_cav)
    assert out["voxel_features"].shape[0] == sum(
        len(c["voxel_features"]) for c in per_cav
    )


def test_collate_of_nothing_is_well_formed(vox: OpenCOODVoxelizer) -> None:
    out = vox.collate([])
    assert out["voxel_coords"].shape == (0, 4)
    assert out["voxel_features"].shape[1:] == (32, 4)


def test_collate_handles_an_empty_agent(vox: OpenCOODVoxelizer) -> None:
    """An agent dropped by a fault contributes no voxels but must not break
    the agent indexing of the others."""
    per_cav = [
        vox.preprocess(_points(100, seed=0)),
        vox.preprocess(np.zeros((0, 4), dtype=np.float32)),
        vox.preprocess(_points(100, seed=1)),
    ]
    out = vox.collate(per_cav)
    assert set(out["voxel_coords"][:, 0].tolist()) == {0, 2}


# --------------------------------------------------------------------- #
# the dataset path
# --------------------------------------------------------------------- #


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec((0.4, 0.4), SMALL_RANGE, 4)


@pytest.fixture
def adapter() -> SyntheticCooperativeDataset:
    return SyntheticCooperativeDataset(n_frames=3, n_agents=3, n_objects=4, seed=0)


def test_dataset_opencood_backend_populates_extra(adapter, spec) -> None:
    ds = LGCPDataset(adapter, spec, feature_backend="opencood")
    frame, _ = ds[0]
    processed = frame.agents.extra["processed_lidar"]
    assert set(processed) == {"voxel_features", "voxel_coords", "voxel_num_points"}
    assert processed["voxel_coords"].shape[1] == 4


def test_dataset_native_backend_leaves_extra_empty(adapter, spec) -> None:
    frame, _ = LGCPDataset(adapter, spec)[0]
    assert frame.agents.extra == {}
    assert frame.agents.features.shape[0] > 0


def test_backends_voxelise_once_not_twice(adapter, spec) -> None:
    """Producing both layouts would double per-frame preprocessing for no
    benefit, since a backend never reads the other's tensors."""
    native = LGCPDataset(adapter, spec, feature_backend="native")
    oc = LGCPDataset(adapter, spec, feature_backend="opencood")
    assert native.opencood_voxelizer is None
    assert oc.voxelizer is None


def test_agent_index_matches_agent_ids(adapter, spec) -> None:
    """The model splits voxels by column 0, so it must line up with the agent
    order the control plane uses -- otherwise features and decisions refer to
    different CAVs."""
    ds = LGCPDataset(adapter, spec, feature_backend="opencood")
    frame, _ = ds[0]
    coords = frame.agents.extra["processed_lidar"]["voxel_coords"]
    assert int(coords[:, 0].max()) < frame.agents.n_agents


def test_unknown_backend_is_rejected(adapter, spec) -> None:
    with pytest.raises(ValueError, match="feature_backend"):
        LGCPDataset(adapter, spec, feature_backend="nope")


def test_faults_reach_the_opencood_path(adapter, spec) -> None:
    """Plane 1 is backend-agnostic: corruption happens on the sample, before
    either voxeliser runs."""
    bridge = DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 1.5}}}, seed=1
    )
    clean = LGCPDataset(adapter, spec, feature_backend="opencood")
    faulty = LGCPDataset(adapter, spec, bridge=bridge, feature_backend="opencood")

    clean_frame, clean_faults = clean[0]
    faulty_frame, faults = faulty[0]

    assert clean_faults == [] and faults
    assert not np.allclose(clean_frame.agents.positions, faulty_frame.agents.positions)


def test_opencood_dataset_feeds_the_adapter(adapter, spec) -> None:
    """The end-to-end seam: dataset output is exactly what the adapter reads.

    Uses the structural stub, since real OpenCOOD needs the py3.7 environment.
    """
    from lgcpbench.perception.opencood.adapter import OpenCOODBackbone
    from lgcpbench.perception.opencood.stub import StubOpenCOODModel

    ds = LGCPDataset(adapter, spec, feature_backend="opencood")
    frame, _ = ds[0]

    model = StubOpenCOODModel(
        grid_hw=spec.grid_hw, feature_hw=spec.feature_hw, channels=32
    )
    backbone = OpenCOODBackbone(
        model, "point_pillar_where2comm", spec.feature_hw, 32
    ).eval()

    features = backbone.encode(frame.agents)
    assert tuple(features.shape) == (frame.agents.n_agents, 32, *spec.feature_hw)
    assert tuple(backbone.confidence(features).shape[1:]) == (1, *spec.feature_hw)
