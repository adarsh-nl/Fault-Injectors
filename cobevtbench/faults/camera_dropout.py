"""
camera_dropout.py
-----------------
Blind specific cameras on specific agents.

The paper reproduction
----------------------
CoBEVT section 7.4 drops **all four ego cameras** and reports the model still
reaches 44.3 IoU, because collaborators cover the ego's blind scene. That is
a fault-injection experiment the authors ran themselves, and reproducing it
is the strongest available check that this benchmark's corruption plane is
wired correctly: normally there is no ground truth for "did my injector
actually work", but here there is a published number to hit.

Why not ``MissingModalityInjector``
-----------------------------------
``src.fault_injectors.MissingModalityInjector`` drops images *probabilistically
and per-array*. ``FaultPipeline``'s image stages receive one image at a time
with no indication of which camera or which agent it belongs to, so "drop all
four cameras of the ego, and nothing else" cannot be expressed there.

This is therefore a **sample stage**, which sees the whole scene and can
select by agent and by camera name. For probabilistic single-camera dropout
the existing injector remains the right tool and this one is unnecessary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

FILL_MODES = ("zero", "mean", "noise")


class CameraDropoutInjector:
    """Blind a chosen set of cameras.

    Purpose
        Reproduce the paper's camera-dropout robustness experiment, and make
        partial sensor loss a first-class benchmark condition.

    Inputs
    ------
    agents     ``"ego"`` (paper's experiment) | ``"all"`` | ``"non-ego"`` |
               list of agent ids
    cameras    explicit camera names to drop; overrides ``n_drop``/``p_drop``
    n_drop     drop this many randomly chosen cameras per selected agent.
               ``n_drop >= len(cameras)`` blinds the agent completely, which
               is the paper's condition.
    p_drop     independent per-camera dropout probability; used only when
               neither ``cameras`` nor ``n_drop`` is set
    fill       what a blinded camera returns. ``"zero"`` is a dead sensor,
               ``"mean"`` a washed-out one, ``"noise"`` a disconnected feed.
               They are not equivalent: a zero image is far outside the
               normalised input distribution, so a model can in principle
               learn to *detect* it, whereas plausible-looking noise cannot
               be recognised as invalid.
    seed       master seed

    Outputs
    -------
    The sample, mutated, with an audit entry under
    ``agent.faults["camera_dropout"]``.

    Example
    -------
    >>> import numpy as np
    >>> from cpbench.data import SyntheticCameraCooperativeDataset
    >>> adapter = SyntheticCameraCooperativeDataset(
    ...     n_frames=1, n_agents=2, image_size=(16, 16))
    >>> sample = adapter.get_sample(0, load=("images",))
    >>> injector = CameraDropoutInjector(agents="ego", n_drop=4, seed=0)
    >>> _ = injector.apply_to_sample(sample)
    >>> ego = sample.agents[sample.ego_id]
    >>> [int(ego.images[c].sum()) for c in sorted(ego.images)]
    [0, 0, 0, 0]

    Collaborators are untouched -- that is what the paper's 44.3 IoU relies on:

    >>> other = sample.agents["agent1"]
    >>> bool(sum(int(other.images[c].sum()) for c in other.images) > 0)
    True

    The audit entry is a flat dict, one per affected agent:

    >>> ego.faults["camera_dropout"]["n_dropped"]
    4
    """

    def __init__(self, agents: Union[str, Sequence[str]] = "ego",
                 cameras: Optional[Sequence[str]] = None,
                 n_drop: int = 0, p_drop: float = 0.0,
                 fill: str = "zero", seed: int = 0) -> None:
        if fill not in FILL_MODES:
            raise ValueError(
                f"unknown fill {fill!r}; expected one of {FILL_MODES}")
        if not 0.0 <= p_drop <= 1.0:
            raise ValueError(f"p_drop must be in [0, 1], got {p_drop}")
        if n_drop < 0:
            raise ValueError(f"n_drop must be >= 0, got {n_drop}")
        self.agents = agents
        self.cameras = list(cameras) if cameras is not None else None
        self.n_drop = int(n_drop)
        self.p_drop = float(p_drop)
        self.fill = fill
        self.rng = np.random.default_rng(seed)

    @property
    def is_active(self) -> bool:
        return bool(self.cameras) or self.n_drop > 0 or self.p_drop > 0.0

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

    def _cameras_to_drop(self, agent) -> List[str]:
        available = sorted(agent.images)
        if not available:
            return []
        if self.cameras is not None:
            return [c for c in self.cameras if c in agent.images]
        if self.n_drop > 0:
            count = min(self.n_drop, len(available))
            chosen = self.rng.choice(len(available), size=count, replace=False)
            return [available[int(i)] for i in sorted(chosen)]
        if self.p_drop > 0:
            return [c for c in available
                    if self.rng.random() < self.p_drop]
        return []

    # -- blinding -----------------------------------------------------------

    def _blank(self, image: np.ndarray) -> np.ndarray:
        if self.fill == "zero":
            return np.zeros_like(image)
        if self.fill == "mean":
            return np.full_like(image, int(np.mean(image)))
        noise = self.rng.integers(0, 256, size=image.shape)
        return noise.astype(image.dtype)

    # -- FaultPipeline sample-stage interface -------------------------------

    def apply_to_sample(self, sample, protect_ego: bool = True):
        """Blind the selected cameras, in place.

        ``protect_ego`` is accepted for the sample-stage interface and
        ignored: the paper's own experiment blinds the **ego**, so honouring
        a default that protects it would make the headline condition
        unreachable. Selection is set by ``agents=`` at construction.
        """
        if not self.is_active:
            return sample
        for agent_id in self._target_agents(sample):
            agent = sample.agents[agent_id]
            dropped = self._cameras_to_drop(agent)
            for camera in dropped:
                agent.images[camera] = self._blank(agent.images[camera])
            if dropped:
                # A flat dict, not a list: DataFaultBridge._harvest maps a
                # dict straight to FaultRecord.params (real columns in
                # injection_summary.csv) but stringifies anything else into a
                # single opaque param_value. One agent drops cameras once per
                # frame, so a flat summary loses nothing.
                agent.faults["camera_dropout"] = {
                    "cameras": ",".join(dropped), "n_dropped": len(dropped),
                    "n_available": len(agent.images), "fill": self.fill}
        return sample

    def __repr__(self) -> str:
        return (f"CameraDropoutInjector(agents={self.agents!r}, "
                f"cameras={self.cameras}, n_drop={self.n_drop}, "
                f"p_drop={self.p_drop}, fill={self.fill!r})")
