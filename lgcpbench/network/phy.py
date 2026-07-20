"""
phy.py
------
5G-V2X physical layer: path loss, shadowing, SINR, achievable rate.

Paper mapping -- Table I, verbatim
    Frequency band              5.9 GHz
    Bandwidth                   40 MHz
    Number of subchannels Z     5
    Transmission power          23 dBm
    Time slot tau               0.25 ms
    Maximum latency T           100 ms
    Path loss model             128.1 + 36.6 log10(d)
    Shadowing distribution      Log-normal
    Shadowing standard dev.     8 dB
    Noise power                 -114 dBm

    Section VI-C: "A 5G-V2X channel is divided into five 8MHz subchannels."
    and "data transmission is disabled when the achievable transmission rate
    falls below 27 Mbps. Once the rate exceeds this threshold, transmission
    resumes at a fixed rate of 27 Mbps."

The threshold-then-fixed-rate model
    The paper does not use a continuous Shannon rate. It uses a step: below
    27 Mbps the link is unusable, above it the link runs at exactly 27 Mbps.
    We reproduce that, but compute the underlying Shannon rate too, because
    the SINR margin is what a physical-plane fault (distance, interference,
    shadowing) actually moves, and reporting only the stepped rate would hide
    a link sitting one dB above the cliff.

Derived constant: the implied SINR threshold
    27 Mbps over an 8 MHz subchannel is a spectral efficiency of 3.375
    bit/s/Hz, so by Shannon the threshold SINR is 2^3.375 - 1 ~= 9.37 dB.
    Everything about link usability follows from that one number, and it is
    derived here rather than hardcoded, so changing the subchannel bandwidth
    or the rate threshold in config keeps the model self-consistent.

Assumption B6 -- interference range
    The paper never gives a numeric interference range, only the rule. We
    derive it from Table I: the distance at which received power falls to
    noise + threshold SINR. ``RateModel.max_range_m()`` computes it; config
    may override with ``interference_range_m``.

Determinism
    Shadowing is a random variable, but a benchmark must be reproducible: a
    schedule difference has to be attributable to an injected fault, never to
    RNG drift. ``ShadowingModel`` therefore derives a per-link seed from the
    (sender, receiver) pair, so the same link draws the same shadowing every
    time within a run, and re-running a condition reproduces it exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------- #
# Table I defaults
# ---------------------------------------------------------------------- #

DEFAULT_TX_POWER_DBM: float = 23.0
DEFAULT_NOISE_POWER_DBM: float = -114.0
DEFAULT_SHADOWING_STD_DB: float = 8.0
DEFAULT_SUBCHANNELS: int = 5
DEFAULT_SUBCHANNEL_BANDWIDTH_HZ: float = 8e6
DEFAULT_RATE_THRESHOLD_BPS: float = 27e6
DEFAULT_TIME_SLOT_S: float = 0.25e-3
DEFAULT_MAX_LATENCY_S: float = 100e-3

# Path loss 128.1 + 36.6 log10(d_km) -- 3GPP urban macro, d in KILOMETRES.
DEFAULT_PATH_LOSS_INTERCEPT_DB: float = 128.1
DEFAULT_PATH_LOSS_SLOPE_DB: float = 36.6

# Below this separation the log-distance model is not physically meaningful
# (it would predict negative path loss); clamp rather than produce nonsense.
_MIN_DISTANCE_M: float = 1.0


@dataclass(frozen=True)
class PathLossModel:
    """Log-distance path loss: ``intercept + slope * log10(d_km)``.

    Purpose
        Table I's propagation model. Frozen because it is a physical
        constant of the scenario, not per-frame state.

    Inputs   distance in METRES (the rest of the codebase works in metres;
             the conversion to km happens here, once).
    Outputs  path loss in dB.

    Example
    -------
    >>> pl = PathLossModel()
    >>> round(pl.loss_db(100.0), 2)
    91.5
    >>> round(pl.loss_db(1000.0), 2)
    128.1
    """

    intercept_db: float = DEFAULT_PATH_LOSS_INTERCEPT_DB
    slope_db: float = DEFAULT_PATH_LOSS_SLOPE_DB

    def loss_db(self, distance_m: float) -> float:
        d_km = max(float(distance_m), _MIN_DISTANCE_M) / 1000.0
        return self.intercept_db + self.slope_db * math.log10(d_km)

    def loss_db_array(self, distance_m: np.ndarray) -> np.ndarray:
        """Vectorised ``loss_db`` for a full pairwise distance matrix."""
        d_km = np.maximum(np.asarray(distance_m, dtype=np.float64), _MIN_DISTANCE_M) / 1000.0
        return self.intercept_db + self.slope_db * np.log10(d_km)

    def distance_for_loss_m(self, loss_db: float) -> float:
        """Invert the model: separation at which loss reaches ``loss_db``.

        Used to derive the interference range (assumption B6) in closed form
        rather than by search.
        """
        exponent = (loss_db - self.intercept_db) / self.slope_db
        return float(10.0 ** exponent * 1000.0)


class ShadowingModel:
    """Log-normal shadowing, deterministic per link.

    Purpose
        Table I's 8 dB log-normal shadowing. The determinism requirement is
        the design point: a fault study compares a clean run against a
        corrupted one, and if shadowing were freshly random per call, every
        comparison would be contaminated by RNG noise that looks exactly like
        a fault effect.

    Inputs
    ------
    std_db  shadowing standard deviation (8 dB per Table I).
    seed    run seed; combined with the link identity to derive a per-link
            draw.
    enabled set False for a deterministic no-shadowing ablation.

    Outputs
    -------
    ``for_link(a, b)`` -> shadowing loss in dB (may be negative: shadowing is
    zero-mean, so some links are favourably shadowed).

    Symmetry
        The draw is keyed on the UNORDERED pair, so the a->b and b->a links
        see identical shadowing. That matches physical reciprocity and keeps
        the interference graph symmetric, which the scheduler relies on.

    Example
    -------
    >>> s = ShadowingModel(seed=0)
    >>> s.for_link("a", "b") == s.for_link("a", "b")     # deterministic
    True
    >>> s.for_link("a", "b") == s.for_link("b", "a")     # reciprocal
    True
    >>> ShadowingModel(enabled=False).for_link("a", "b")
    0.0
    """

    def __init__(
        self,
        std_db: float = DEFAULT_SHADOWING_STD_DB,
        seed: int = 0,
        enabled: bool = True,
    ) -> None:
        if std_db < 0:
            raise ValueError(f"std_db must be >= 0, got {std_db}")
        self.std_db = float(std_db)
        self.seed = int(seed)
        self.enabled = enabled
        self._cache: dict = {}

    def for_link(self, a: str, b: str) -> float:
        """Shadowing loss in dB for the (unordered) link between a and b."""
        if not self.enabled or self.std_db == 0.0:
            return 0.0
        key = tuple(sorted((str(a), str(b))))
        cached = self._cache.get(key)
        if cached is None:
            digest = hashlib.sha256(
                f"{self.seed}|{key[0]}|{key[1]}".encode()
            ).digest()
            link_seed = int.from_bytes(digest[:8], "big")
            cached = float(np.random.default_rng(link_seed).normal(0.0, self.std_db))
            self._cache[key] = cached
        return cached

    def clear(self) -> None:
        """Drop cached draws -- call between frames only if links should
        re-fade; by default a run keeps one static realisation."""
        self._cache.clear()


@dataclass(frozen=True)
class LinkBudget:
    """The computed state of one directed link.

    Attributes
    ----------
    distance_m    separation.
    path_loss_db  from the log-distance model.
    shadowing_db  the link's shadowing draw.
    sinr_db       received SINR.
    shannon_bps   continuous capacity of one subchannel at this SINR.
    rate_bps      the paper's stepped rate: 0 below threshold, else fixed.
    usable        rate_bps > 0.
    """

    distance_m: float
    path_loss_db: float
    shadowing_db: float
    sinr_db: float
    shannon_bps: float
    rate_bps: float
    usable: bool


class RateModel:
    """Table I PHY -> achievable rate, with the paper's threshold behaviour.

    Purpose
        The single authority on whether two CAVs can talk and how fast. Both
        the scheduler (which links exist) and the interference model (which
        links collide) key off this, so they cannot disagree.

    Inputs
    ------
    tx_power_dbm, noise_power_dbm, subchannel_bandwidth_hz,
    rate_threshold_bps, fixed_rate_bps  -- Table I / section VI-C.
    path_loss, shadowing                -- injected models.

    Outputs
    -------
    ``link(a, b, distance_m)`` -> LinkBudget
    ``max_range_m()``          -> the derived interference range (B6)

    Example
    -------
    >>> rm = RateModel(shadowing=ShadowingModel(enabled=False))
    >>> round(rm.sinr_threshold_db, 2)
    9.72
    >>> rm.link("a", "b", 50.0).usable
    True
    >>> rm.link("a", "b", 5000.0).usable
    False
    """

    def __init__(
        self,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        noise_power_dbm: float = DEFAULT_NOISE_POWER_DBM,
        subchannel_bandwidth_hz: float = DEFAULT_SUBCHANNEL_BANDWIDTH_HZ,
        rate_threshold_bps: float = DEFAULT_RATE_THRESHOLD_BPS,
        fixed_rate_bps: Optional[float] = None,
        path_loss: Optional[PathLossModel] = None,
        shadowing: Optional[ShadowingModel] = None,
    ) -> None:
        self.tx_power_dbm = float(tx_power_dbm)
        self.noise_power_dbm = float(noise_power_dbm)
        self.subchannel_bandwidth_hz = float(subchannel_bandwidth_hz)
        self.rate_threshold_bps = float(rate_threshold_bps)
        # Section VI-C: "transmission resumes at a fixed rate of 27 Mbps" --
        # the fixed rate IS the threshold unless overridden.
        self.fixed_rate_bps = float(
            rate_threshold_bps if fixed_rate_bps is None else fixed_rate_bps
        )
        self.path_loss = path_loss or PathLossModel()
        self.shadowing = shadowing if shadowing is not None else ShadowingModel()

    @property
    def sinr_threshold_db(self) -> float:
        """SINR implied by the 27 Mbps / 8 MHz threshold, via Shannon.

        Derived, not hardcoded: changing the subchannel bandwidth or rate
        threshold in config keeps the usability rule self-consistent.
        """
        spectral_efficiency = self.rate_threshold_bps / self.subchannel_bandwidth_hz
        linear = (2.0 ** spectral_efficiency) - 1.0
        return 10.0 * math.log10(linear)

    def sinr_db(self, sender: str, receiver: str, distance_m: float) -> float:
        """Received SINR in dB for one directed link."""
        pl = self.path_loss.loss_db(distance_m)
        sh = self.shadowing.for_link(sender, receiver)
        return self.tx_power_dbm - pl - sh - self.noise_power_dbm

    def link(self, sender: str, receiver: str, distance_m: float) -> LinkBudget:
        """Full link budget, including both the Shannon and stepped rates."""
        pl = self.path_loss.loss_db(distance_m)
        sh = self.shadowing.for_link(sender, receiver)
        sinr_db = self.tx_power_dbm - pl - sh - self.noise_power_dbm
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        shannon = self.subchannel_bandwidth_hz * math.log2(1.0 + sinr_linear)
        usable = shannon >= self.rate_threshold_bps
        return LinkBudget(
            distance_m=float(distance_m),
            path_loss_db=pl,
            shadowing_db=sh,
            sinr_db=sinr_db,
            shannon_bps=shannon,
            rate_bps=self.fixed_rate_bps if usable else 0.0,
            usable=usable,
        )

    def is_usable(self, sender: str, receiver: str, distance_m: float) -> bool:
        """Can these two CAVs exchange data at all?"""
        return self.link(sender, receiver, distance_m).usable

    def max_range_m(self) -> float:
        """Communication / interference range implied by Table I (B6).

        The separation at which received power falls to the threshold SINR,
        ignoring shadowing (i.e. the median link). Computed in closed form by
        inverting the path-loss model.

        Note for the reader: with Table I's numbers this comes out large
        relative to the 280 m x 80 m RoI, which means essentially every pair
        of CAVs in a scene interferes. That is a consequence of the paper's
        own parameters, not a modelling choice here -- and it is why the
        scheduler's effective concurrency is bounded by Z subchannels rather
        than by spatial reuse. Override with ``interference_range_m`` in
        config to study the spatial-reuse regime instead.
        """
        max_loss = self.tx_power_dbm - self.noise_power_dbm - self.sinr_threshold_db
        return self.path_loss.distance_for_loss_m(max_loss)

    def packet_bits(self, time_slot_s: float = DEFAULT_TIME_SLOT_S) -> float:
        """Bits carried by one time slot at the fixed rate.

        Table I's tau = 0.25 ms at 27 Mbps gives 6750 bits. Section V-B says
        "The time slot tau is set to the duration required to transmit one
        packet", and a packet encapsulates one area's features. Design doc
        derivation D2 puts an area-restricted feature at C * cells =
        256 * ~24 = ~6100 bits. The two agree to within a few percent, which
        is a useful independent check that D2's 1-bit-per-element reading of
        the paper's "2.16Mb" is right.
        """
        return self.fixed_rate_bps * float(time_slot_s)
