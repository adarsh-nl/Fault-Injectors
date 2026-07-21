"""
calibration.py
--------------
Camera miscalibration: perturb intrinsics and extrinsics.

Why this injector has to exist
------------------------------
CoBEVT has no depth network. It lifts image features to BEV by matching unit
ray directions computed from ``K`` and ``T_cam_to_agent`` (see
``fusion/camera_embedding.py``). Those matrices are therefore **tensors on
the attention path**, not preprocessing metadata -- change them and image
content lands somewhere else on the BEV grid.

``src/fault_injectors`` has nothing that perturbs them. A CoBEVT fault
benchmark without this injector could corrupt every input to the model
*except* the mechanism the model is actually built around.

It is also a physically ordinary fault: thermal drift moves focal length,
vibration and minor knocks move mounting pose, and both degrade slowly enough
that no alarm fires. That is precisely the silent-degradation regime this
benchmark exists to measure.

Placement
---------
A **sample stage**: it mutates ``agent.cameras[name]`` before any tensor
exists, keeping the corruption plane's contract that no model code is
fault-aware. It lives in ``cobevtbench`` rather than ``src`` because there is
one consumer; a second camera paper is the trigger to promote it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

AgentScope = Union[str, Sequence[str]]


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """(3, 3) rotation from an axis and angle, via Rodrigues' formula.

    >>> import numpy as np
    >>> R = rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    >>> bool(np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]))
    True
    >>> bool(np.allclose(R @ R.T, np.eye(3)))
    True
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle_rad) * cross
            + (1.0 - np.cos(angle_rad)) * (cross @ cross))


class CalibrationErrorInjector:
    """Perturb camera intrinsics and extrinsics on a cooperative sample.

    Purpose
        Reach CoBEVT's lifting geometry with a physically motivated fault.

    Inputs
    ------
    sigma_focal_px      std of the focal-length error, in pixels (fx, fy
                        perturbed independently -- a real sensor's axes drift
                        separately)
    sigma_principal_px  std of the principal-point error, in pixels
    sigma_translation_m std of the mounting-position error, in metres
    sigma_rotation_deg  std of the mounting-orientation error, in degrees,
                        about a uniformly random axis
    agents              ``"all"`` | ``"non-ego"`` | ``"ego"`` | list of ids.
                        Defaults to ``"all"``: unlike a V2X-link fault, a
                        miscalibrated camera is a property of the vehicle
                        that owns it, and the ego is not exempt.
    cameras             ``"all"`` | ``"one"`` | list of camera names
    seed                master seed

    Outputs
    -------
    The same sample, mutated, with an audit entry under
    ``agent.faults["calibration"]`` that ``DataFaultBridge`` harvests into
    ``injection_summary.csv``.

    Example
    -------
    >>> import numpy as np
    >>> from cpbench.data import SyntheticCameraCooperativeDataset
    >>> adapter = SyntheticCameraCooperativeDataset(
    ...     n_frames=1, n_agents=2, image_size=(16, 16))
    >>> sample = adapter.get_sample(0, load=("images",))
    >>> before = sample.agents["agent1"].cameras["camera0"].K.copy()
    >>> injector = CalibrationErrorInjector(sigma_focal_px=5.0, seed=0)
    >>> _ = injector.apply_to_sample(sample)
    >>> bool(np.allclose(before, sample.agents["agent1"].cameras["camera0"].K))
    False
    >>> sorted(sample.agents["agent1"].faults["calibration"])
    ['cameras', 'max_focal_px', 'max_rotation_deg', 'n_cameras']
    """

    def __init__(self, sigma_focal_px: float = 0.0,
                 sigma_principal_px: float = 0.0,
                 sigma_translation_m: float = 0.0,
                 sigma_rotation_deg: float = 0.0,
                 agents: AgentScope = "all", cameras: Union[str, Sequence[str]] = "all",
                 seed: int = 0) -> None:
        for name, value in (("sigma_focal_px", sigma_focal_px),
                            ("sigma_principal_px", sigma_principal_px),
                            ("sigma_translation_m", sigma_translation_m),
                            ("sigma_rotation_deg", sigma_rotation_deg)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self.sigma_focal_px = float(sigma_focal_px)
        self.sigma_principal_px = float(sigma_principal_px)
        self.sigma_translation_m = float(sigma_translation_m)
        self.sigma_rotation_deg = float(sigma_rotation_deg)
        self.agents = agents
        self.cameras = cameras
        self.rng = np.random.default_rng(seed)

    @property
    def is_active(self) -> bool:
        """False when every sigma is zero -- nothing would be injected."""
        return any((self.sigma_focal_px, self.sigma_principal_px,
                    self.sigma_translation_m, self.sigma_rotation_deg))

    # -- selection ----------------------------------------------------------

    def _target_agents(self, sample) -> List[str]:
        if isinstance(self.agents, str):
            if self.agents == "all":
                return list(sample.agents)
            if self.agents == "ego":
                return [sample.ego_id]
            if self.agents == "non-ego":
                return [a for a in sample.agents if a != sample.ego_id]
            raise ValueError(
                f"unknown agent scope {self.agents!r}; expected 'all', 'ego', "
                "'non-ego' or a list of agent ids")
        return [a for a in self.agents if a in sample.agents]

    def _target_cameras(self, agent) -> List[str]:
        names = sorted(agent.cameras)
        if isinstance(self.cameras, str):
            if self.cameras == "all":
                return names
            if self.cameras == "one":
                return [names[int(self.rng.integers(len(names)))]] if names else []
            raise ValueError(
                f"unknown camera selection {self.cameras!r}; expected 'all', "
                "'one' or a list of camera names")
        return [n for n in self.cameras if n in agent.cameras]

    # -- perturbation -------------------------------------------------------

    def sample_error(self) -> Dict[str, float]:
        """Draw one calibration error. Exposed so a test can pin the draw."""
        return {
            "d_fx": float(self.rng.normal(0.0, self.sigma_focal_px)),
            "d_fy": float(self.rng.normal(0.0, self.sigma_focal_px)),
            "d_cx": float(self.rng.normal(0.0, self.sigma_principal_px)),
            "d_cy": float(self.rng.normal(0.0, self.sigma_principal_px)),
            "rotation_deg": float(self.rng.normal(0.0,
                                                  self.sigma_rotation_deg)),
            "translation_m": float(np.abs(
                self.rng.normal(0.0, self.sigma_translation_m))),
        }

    def perturb_intrinsics(self, K: np.ndarray,
                           error: Dict[str, float]) -> np.ndarray:
        """Apply focal and principal-point error to a copy of ``K``."""
        out = np.array(K, dtype=np.float64, copy=True)
        out[0, 0] += error["d_fx"]
        out[1, 1] += error["d_fy"]
        out[0, 2] += error["d_cx"]
        out[1, 2] += error["d_cy"]
        return out.astype(K.dtype if hasattr(K, "dtype") else np.float32)

    def perturb_extrinsics(self, T: np.ndarray,
                           error: Dict[str, float]) -> np.ndarray:
        """Apply mounting error to a copy of ``T_cam_to_agent``.

        The rotation is applied on the left of the existing one, so it is an
        error in the camera's orientation rather than a re-parameterisation
        of the agent frame.
        """
        out = np.array(T, dtype=np.float64, copy=True)
        if self.sigma_rotation_deg > 0:
            axis = self.rng.normal(size=3)
            R = rotation_from_axis_angle(axis, np.radians(error["rotation_deg"]))
            out[:3, :3] = R @ out[:3, :3]
        if self.sigma_translation_m > 0 and error["translation_m"] > 0:
            direction = self.rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                out[:3, 3] += (direction / norm) * error["translation_m"]
        return out.astype(T.dtype if hasattr(T, "dtype") else np.float32)

    # -- FaultPipeline sample-stage interface -------------------------------

    def apply_to_sample(self, sample, protect_ego: bool = True):
        """Perturb the selected cameras, in place.

        ``protect_ego`` is accepted for the ``FaultPipeline`` sample-stage
        interface but deliberately ignored: which agents are affected is set
        by ``agents=`` at construction. A miscalibrated camera belongs to the
        vehicle that owns it, so exempting the ego by default would make the
        most realistic version of this fault unreachable -- and it is the
        version the paper's own camera experiments target.
        """
        if not self.is_active:
            return sample
        for agent_id in self._target_agents(sample):
            agent = sample.agents[agent_id]
            cameras, max_focal, max_rotation = [], 0.0, 0.0
            for camera in self._target_cameras(agent):
                calib = agent.cameras[camera]
                error = self.sample_error()
                calib.K = self.perturb_intrinsics(calib.K, error)
                if calib.T_cam_to_agent is not None:
                    calib.T_cam_to_agent = self.perturb_extrinsics(
                        calib.T_cam_to_agent, error)
                cameras.append(camera)
                max_focal = max(max_focal, abs(error["d_fx"]),
                                abs(error["d_fy"]))
                max_rotation = max(max_rotation, abs(error["rotation_deg"]))
            if cameras:
                # A flat summary dict, so injection_summary.csv gets real
                # columns rather than a stringified per-camera list. The exact
                # per-camera K delta is recoverable from the tapped
                # input/intrinsics tensor; what the audit trail needs is
                # "which cameras, how much".
                agent.faults["calibration"] = {
                    "cameras": ",".join(cameras), "n_cameras": len(cameras),
                    "max_focal_px": round(max_focal, 4),
                    "max_rotation_deg": round(max_rotation, 4)}
        return sample

    def __repr__(self) -> str:
        return (f"CalibrationErrorInjector(focal={self.sigma_focal_px}px, "
                f"principal={self.sigma_principal_px}px, "
                f"translation={self.sigma_translation_m}m, "
                f"rotation={self.sigma_rotation_deg}deg, "
                f"agents={self.agents!r}, cameras={self.cameras!r})")
