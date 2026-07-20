"""
combiner.py
-----------
Combine per-CAV area confidences into a group confidence.

Paper mapping
    Eq. 2:  F_i(V^_i) = 1 - PROD_{v_k in V^_i} (1 - F_i({v_k}))
    Eq. 3:  (1/N) SUM_i P_acc(a_i)  ~=  (1/N) SUM_i F_i(V^_i)
    Eq. 8:  F_i(V^_i U {v_j}) - F_i(V^_i)  >=  dg

This is a noisy-OR: each CAV independently "covers" the area with
probability f_k, and the group covers it unless every member fails.

Two consequences that the implementation exploits
-------------------------------------------------
(1) CLOSED-FORM GAIN. Writing S for the current group and f for a candidate's
    confidence,

        1 - F(S U {v}) = (1 - F(S)) * (1 - f)
    =>      F(S U {v}) = F(S) + (1 - F(S)) * f
    =>           gain  = (1 - F(S)) * f

    So Eq. 8's test needs no recomputation of the product -- it is one
    multiply. ``gain()`` implements this and a test asserts it equals the
    naive difference form to floating-point tolerance.

(2) PROVABLE DIMINISHING RETURNS, hence safe early termination. The paper
    argues informally that "the confidence gain from additional CAVs
    decreases as the group grows". The closed form makes it exact: F(S) is
    non-decreasing, so (1 - F(S)) is non-increasing. If candidates are also
    visited in descending f (assumption B2, matching Algorithm 1 line 2),
    then f is non-increasing too, so the gain is non-increasing over the
    whole scan. The moment one candidate fails Eq. 8, every later one fails.

    Algorithm 1 line 3 can therefore break out of its loop instead of
    scanning all |V| candidates, without changing the result. That is a
    property of the paper's own equations, not an approximation -- and
    ``selection/`` relies on it, so it is asserted by test here.

Order invariance
    The product is commutative, so F(S) does not depend on the order members
    were added. Order affects only WHICH members are admitted under dg.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

Number = Union[float, np.floating]


class NoisyOrCombiner:
    """Paper Eq. 2: independent-coverage combination of area confidences.

    Purpose
        The single implementation of Eq. 2. ``selection/`` calls ``gain`` on
        the hot path and ``combine`` when it needs the absolute value; the
        objective (Eq. 3, Eq. 7) calls ``combine``.

    Inputs
    ------
    clip  clamp inputs into [0, 1] instead of raising. Confidences come from
          a sigmoid so they are in range by construction -- but a control-
          plane fault injector deliberately falsifies reported confidences
          (design doc section 5, plane 3), and an injected value of 1.7 must
          not silently produce a negative group confidence. Default True:
          clamp, because the RSU in a real deployment cannot trust a report
          either. Set False in tests that want to catch upstream bugs.

    Example
    -------
    >>> c = NoisyOrCombiner()
    >>> round(c.combine([0.5, 0.5]), 4)
    0.75
    >>> round(c.gain(current=0.75, new=0.5), 4)
    0.125
    >>> round(c.combine([0.5, 0.5, 0.5]), 4)      # == 0.75 + 0.125
    0.875
    """

    def __init__(self, clip: bool = True) -> None:
        self.clip = clip

    # ------------------------------------------------------------------ #
    # Eq. 2
    # ------------------------------------------------------------------ #

    def combine(self, confidences: Union[Sequence[Number], np.ndarray]) -> float:
        """Group confidence F_i(V^_i) from member confidences.

        Inputs  confidences : (K,) values in [0, 1]. Empty -> 0.0, which is
                the correct reading: an area with no assigned group has no
                expected perception quality.
        Outputs scalar in [0, 1].
        """
        arr = self._prepare(confidences)
        if arr.size == 0:
            return 0.0
        return float(1.0 - np.prod(1.0 - arr))

    def combine_batch(self, confidences: np.ndarray, axis: int = -1) -> np.ndarray:
        """Vectorised ``combine`` over an axis.

        Inputs  (..., K) array; Outputs (...,) array.
        Used by the objective (Eq. 3) to score all areas at once.
        """
        arr = self._prepare(confidences)
        if arr.shape[axis] == 0:
            return np.zeros(np.delete(np.array(arr.shape), axis), dtype=float)
        return 1.0 - np.prod(1.0 - arr, axis=axis)

    # ------------------------------------------------------------------ #
    # Eq. 8, in closed form
    # ------------------------------------------------------------------ #

    def gain(self, current: Number, new: Number) -> float:
        """Marginal gain of adding one CAV -- ``(1 - current) * new``.

        Equivalent to ``combine(S + [new]) - combine(S)`` where
        ``combine(S) == current``, but O(1) and without recomputing the
        product. See the module docstring for the derivation.

        Inputs  current : F(S) in [0, 1];  new : f_v in [0, 1].
        Outputs marginal gain in [0, 1].
        """
        cur = self._scalar(current)
        nxt = self._scalar(new)
        return float((1.0 - cur) * nxt)

    def gain_batch(self, current: Number, new: np.ndarray) -> np.ndarray:
        """Vectorised ``gain`` over many candidates at one group state."""
        cur = self._scalar(current)
        return (1.0 - cur) * self._prepare(new)

    def update(self, current: Number, new: Number) -> float:
        """F(S U {v}) from F(S) -- ``current + gain(current, new)``.

        Lets Algorithm 1 carry the group confidence incrementally instead of
        rebuilding the product after every admission.
        """
        cur = self._scalar(current)
        return float(cur + self.gain(cur, new))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _prepare(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if self.clip:
            return np.clip(arr, 0.0, 1.0)
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise ValueError(
                f"confidences must lie in [0, 1], got range "
                f"[{arr.min():.4f}, {arr.max():.4f}]"
            )
        return arr

    def _scalar(self, value: Number) -> float:
        v = float(value)
        if self.clip:
            return min(1.0, max(0.0, v))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {v}")
        return v


def global_accuracy_proxy(group_confidences: Iterable[Number]) -> float:
    """Paper Eq. 3: the mean area confidence over all areas.

    This is the numerator of the objective P0 (Eq. 7). The paper uses it as a
    stand-in for true perception accuracy because the RSU cannot measure
    accuracy before collaboration happens -- it only ever sees confidences.

    Inputs  one F_i(V^_i) per area. Empty -> 0.0.
    Outputs scalar in [0, 1].

    Example
    -------
    >>> round(global_accuracy_proxy([0.8, 0.6, 1.0]), 4)
    0.8
    """
    arr = np.asarray(list(group_confidences), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())
