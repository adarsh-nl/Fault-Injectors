"""
injectors.py
------------
The metadata plane: faults on the V2X metadata V2X-ViT's mechanisms consume.

Why a second plane exists at all
--------------------------------
This repository's standing rule is that faults are *physical* and are applied
to raw data upstream of the model -- one corruption path, in
``src.pipeline.FaultPipeline``, reached through ``cpbench.faults``. That rule
is not relaxed here.

But V2X-ViT is the first model in this repository whose *robustness
mechanisms* consume inputs that travel outside the sensor data: the reported
time delay (consumed by the DPE), the agent-type flag (which selects HMSA's
projections and relation matrices), the pose used only by the feature warp,
and the speed field in the prior. Each is a small metadata field in the V2X
message header, and a stack that delivers the feature payload while
corrupting a header field -- a clock skew, a truncated field, a stale
handshake -- is a routine, physically real event no sensor-level injector
can express. w2cbench's protocol plane established the precedent; this
follows it, and confines itself to the same boundary: these injectors act on
*communicated metadata*, between collation and the forward pass, and every
action becomes a ``FaultRecord`` exactly like a physical one.

What is deliberately not here
-----------------------------
An agent that *lies* about its type or delay is an attack, not a fault, and
modelling it would let an adversarial result be read as a reliability
result. These injectors model transport-layer corruption: the sender
reported truthfully, and the value arrived wrong. The distinction is why
every injector here is symmetric noise or a stuck-at value, never an
optimised perturbation.

The ego row is out of bounds for every injector -- the ego's metadata never
crossed a link -- and :class:`~v2xvitbench.faults.metadata.MetadataFaultBridge`
enforces that centrally rather than trusting each injector to remember.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

#: batch keys the metadata plane may corrupt, by stage name
STAGES = ("time_delay", "agent_types", "poses", "prior")


class MetadataInjector(ABC):
    """One corruption applied to one metadata field of a collated batch.

    Attributes
    ----------
    stage  which field this injector acts on:

           ``"time_delay"``   the reported delay the DPE reads
           ``"agent_types"``  the vehicle/infrastructure flag HMSA routes by
           ``"poses"``        the agent-to-ego transform the STTF warps by
           ``"prior"``        the velocity field of the prior encoding

    Subclasses implement :meth:`apply` on the full ``(B, L, ...)`` tensor
    and may corrupt any slot; the bridge restores the ego row afterwards.
    Return the corrupted tensor and a parameter dict for the audit trail
    (or None when nothing fired, so a probabilistic injector does not record
    a no-op as a fault).
    """

    stage: str = ""
    name: str = ""

    @abstractmethod
    def apply(self, tensor: torch.Tensor, *, generator: torch.Generator,
              **context: Any) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Corrupt `tensor`; return ``(tensor, params or None)``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(stage={self.stage!r})"


def _non_ego_draw(shape: Tuple[int, int], p: float,
                  generator: torch.Generator) -> torch.Tensor:
    """(B, L) bool, True where a non-ego slot is selected with prob `p`."""
    draw = torch.rand(shape, generator=generator) < p
    draw[:, 0] = False
    return draw


class DelayEncodingInjector(MetadataInjector):
    """Corrupt the REPORTED delay, splitting it from the actual staleness.

    Purpose
        The DPE compensates for delay it is told about. This injector makes
        the report wrong, which is the failure mode the paper's asynchronous
        experiment cannot reach: there, delay and report always agree.

    Modes
    -----
    ``zero``   the report says "fresh" regardless of the truth. Run combined
               with a plane-1 latency fault, this isolates the DPE's
               contribution: features stale, encoding says otherwise.
    ``stale``  the report adds ``magnitude_frames`` to the truth: fresh
               features presented as old.
    ``noise``  the report is jittered by +/- ``magnitude_frames`` (uniform),
               clamped at zero -- clock skew rather than a stuck field.

    Inputs
    ------
    mode              one of the above
    magnitude_frames  frames of offset/jitter (unused by ``zero``)
    p_affected        probability, per non-ego agent per batch, of corruption

    Example
    -------
    >>> import torch
    >>> injector = DelayEncodingInjector(mode="zero", p_affected=1.0)
    >>> dts = torch.tensor([[0, 3, 5]])
    >>> out, params = injector.apply(
    ...     dts, generator=torch.Generator().manual_seed(0))
    >>> out.tolist(), params["n_affected"]
    ([[0, 0, 0]], 2)
    """

    stage = "time_delay"
    name = "delay_encoding"

    def __init__(self, mode: str = "zero", magnitude_frames: int = 3,
                 p_affected: float = 1.0) -> None:
        if mode not in ("zero", "stale", "noise"):
            raise ValueError(
                f"unknown delay_encoding mode {mode!r}; expected "
                "'zero', 'stale' or 'noise'")
        if not 0.0 <= float(p_affected) <= 1.0:
            raise ValueError(f"p_affected must be in [0, 1], got {p_affected}")
        self.mode = mode
        self.magnitude_frames = int(magnitude_frames)
        self.p_affected = float(p_affected)

    def apply(self, tensor, *, generator, **context):
        affected = _non_ego_draw(tuple(tensor.shape), self.p_affected,
                                 generator)
        if not bool(affected.any()):
            return tensor, None
        out = tensor.clone()
        if self.mode == "zero":
            out[affected] = 0
        elif self.mode == "stale":
            out[affected] += self.magnitude_frames
        else:  # noise
            jitter = torch.randint(-self.magnitude_frames,
                                   self.magnitude_frames + 1,
                                   tensor.shape, generator=generator)
            out[affected] = (out[affected] + jitter[affected]).clamp_(min=0)
        return out, {"mode": self.mode,
                     "magnitude_frames": self.magnitude_frames,
                     "n_affected": int(affected.sum()),
                     "slots": affected.nonzero().tolist()}


class AgentTypeFlipInjector(MetadataInjector):
    """Flip the vehicle/infrastructure flag HMSA routes by.

    Purpose
        The novel injection surface of this paper: one flipped bit re-routes
        an agent through projections and relation matrices fitted to the
        other sensor class. Measures what the heterogeneity machinery costs
        when its routing input is wrong.

    Inputs
    ------
    p_flip     probability, per non-ego agent per batch, of a flip
    direction  ``"both"`` flips whatever is there; ``"to_infra"`` only
               corrupts vehicles into infrastructure; ``"to_vehicle"`` the
               reverse -- the asymmetric modes matter because V2XSet scenes
               carry at most one real infrastructure agent, so the two flip
               directions have very different base rates.

    Example
    -------
    >>> import torch
    >>> injector = AgentTypeFlipInjector(p_flip=1.0)
    >>> types = torch.tensor([[0, 0, 1]])
    >>> out, params = injector.apply(
    ...     types, generator=torch.Generator().manual_seed(0))
    >>> out.tolist(), params["n_flipped"]
    ([[0, 1, 0]], 2)
    """

    stage = "agent_types"
    name = "type_flip"

    def __init__(self, p_flip: float = 0.5, direction: str = "both") -> None:
        if not 0.0 <= float(p_flip) <= 1.0:
            raise ValueError(f"p_flip must be in [0, 1], got {p_flip}")
        if direction not in ("both", "to_infra", "to_vehicle"):
            raise ValueError(
                f"unknown direction {direction!r}; expected 'both', "
                "'to_infra' or 'to_vehicle'")
        self.p_flip = float(p_flip)
        self.direction = direction

    def apply(self, tensor, *, generator, **context):
        flip = _non_ego_draw(tuple(tensor.shape), self.p_flip, generator)
        if self.direction == "to_infra":
            flip &= tensor == 0
        elif self.direction == "to_vehicle":
            flip &= tensor == 1
        if not bool(flip.any()):
            return tensor, None
        out = tensor.clone()
        out[flip] = 1 - out[flip]
        return out, {"p_flip": self.p_flip, "direction": self.direction,
                     "n_flipped": int(flip.sum()),
                     "slots": flip.nonzero().tolist()}


class CorrectionMatrixInjector(MetadataInjector):
    """Perturb the agent-to-ego transform, and ONLY the transform.

    Purpose
        Distinct from the plane-1 ``pose_error`` on purpose: that one
        corrupts the shared poses upstream, moving the warp AND the ground
        truth's agent alignment together. This one acts after collation, so
        points, labels and everything upstream stay clean and the measured
        effect is purely the STTF's sensitivity to its correction input.

    Inputs
    ------
    sigma_xy           translation noise std, metres (paper protocol: 0-0.5)
    sigma_heading_deg  heading noise std, degrees (paper protocol: 0-1.0)

    Example
    -------
    >>> import torch
    >>> injector = CorrectionMatrixInjector(sigma_xy=0.5, sigma_heading_deg=0)
    >>> T = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    >>> out, params = injector.apply(
    ...     T, generator=torch.Generator().manual_seed(0))
    >>> bool(torch.equal(out[:, 0], T[:, 0]))   # bridge also enforces this
    True
    >>> bool((out[:, 1, :2, 3] != 0).any()), params["n_affected"]
    (True, 1)
    """

    stage = "poses"
    name = "correction_matrix"

    def __init__(self, sigma_xy: float = 0.2,
                 sigma_heading_deg: float = 0.2) -> None:
        if float(sigma_xy) < 0 or float(sigma_heading_deg) < 0:
            raise ValueError("noise standard deviations must be >= 0")
        self.sigma_xy = float(sigma_xy)
        self.sigma_heading_deg = float(sigma_heading_deg)

    def apply(self, tensor, *, generator, **context):
        batch, agents = tensor.shape[:2]
        if agents <= 1 or (self.sigma_xy == 0 and self.sigma_heading_deg == 0):
            return tensor, None
        out = tensor.clone()
        n_affected = 0
        for b in range(batch):
            for l in range(1, agents):
                shift = torch.randn(2, generator=generator) * self.sigma_xy
                angle = (torch.randn(1, generator=generator).item()
                         * math.radians(self.sigma_heading_deg))
                c, s = math.cos(angle), math.sin(angle)
                rot = torch.tensor([[c, -s], [s, c]], dtype=out.dtype)
                out[b, l, :2, :2] = rot @ out[b, l, :2, :2]
                out[b, l, :2, 3] += shift.to(out.dtype)
                n_affected += 1
        return out, {"sigma_xy": self.sigma_xy,
                     "sigma_heading_deg": self.sigma_heading_deg,
                     "n_affected": n_affected}


class PriorNoiseInjector(MetadataInjector):
    """Gaussian noise on the reported speed in the prior encoding.

    The lowest-stakes field -- the reference model only carries velocity as
    context -- included so the sweep can show a metadata field the
    architecture is (presumably) insensitive to, as a control for the two it
    is built around.

    Example
    -------
    >>> import torch
    >>> injector = PriorNoiseInjector(sigma_v=1.0)
    >>> v = torch.zeros(1, 3)
    >>> out, params = injector.apply(
    ...     v, generator=torch.Generator().manual_seed(0))
    >>> bool((out[0, 1:] != 0).any()), out[0, 0].item()
    (True, 0.0)
    """

    stage = "prior"
    name = "prior_noise"

    def __init__(self, sigma_v: float = 0.3) -> None:
        if float(sigma_v) < 0:
            raise ValueError(f"sigma_v must be >= 0, got {sigma_v}")
        self.sigma_v = float(sigma_v)

    def apply(self, tensor, *, generator, **context):
        if self.sigma_v == 0:
            return tensor, None
        noise = torch.randn(tensor.shape, generator=generator) * self.sigma_v
        noise[:, 0] = 0.0
        return tensor + noise.to(tensor.dtype), {
            "sigma_v": self.sigma_v,
            "n_affected": int(tensor.shape[0] * (tensor.shape[1] - 1))}


_INJECTORS = {
    DelayEncodingInjector.name: DelayEncodingInjector,
    AgentTypeFlipInjector.name: AgentTypeFlipInjector,
    CorrectionMatrixInjector.name: CorrectionMatrixInjector,
    PriorNoiseInjector.name: PriorNoiseInjector,
}


def make_metadata_injector(name: str, **params: Any) -> MetadataInjector:
    """Build one metadata injector by config name.

    >>> make_metadata_injector("type_flip", p_flip=0.25).p_flip
    0.25
    >>> make_metadata_injector("bit_rot")
    Traceback (most recent call last):
    ValueError: unknown metadata injector 'bit_rot'; expected one of ['correction_matrix', 'delay_encoding', 'prior_noise', 'type_flip']
    """
    if name not in _INJECTORS:
        raise ValueError(
            f"unknown metadata injector {name!r}; expected one of "
            f"{sorted(_INJECTORS)}")
    return _INJECTORS[name](**params)
