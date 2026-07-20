"""
pooling.py
----------
Reduce a per-cell confidence map to one scalar per area.

Paper mapping
    Eq. 1:  F_i({v_j}) = f_gen(f_i,j)

    The paper gives no pooling operator. ``f_gen`` (design doc derivation D1)
    produces a per-CELL confidence map -- Where2comm's spatial confidence --
    but Eq. 2's noisy-OR and Eq. 8's threshold both need a scalar per AREA.
    Something must collapse ~24 cells into one number, and the paper never
    says what. This is assumption B1, and it is made configurable rather than
    hardcoded because it materially changes which CAVs a group admits.

Why max is the default
    Where2comm itself reduces its anchor dimension with ``max``:

        ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(
            dim=1, keepdim=True)

    and the semantics match the paper's intent: area confidence should
    express "how well can this CAV perceive *something* in this area", which
    is a max over evidence, not an average diluted by empty road. Mean
    pooling systematically under-rates a CAV that sees one object very
    clearly in an otherwise empty area -- exactly the CAV a group wants.

Shapes
    Every pooling takes (V, h, w) -- one area's cells for V CAVs -- and
    returns (V,).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


@runtime_checkable
class AreaPooling(Protocol):
    """Collapse one area's confidence cells to one value per CAV.

    Inputs   patch (V, h, w) -- confidence in [0, 1].
    Outputs  (V,) -- area confidence in [0, 1].
    """

    def __call__(self, patch: torch.Tensor) -> torch.Tensor: ...


def _check(patch: torch.Tensor) -> torch.Tensor:
    if patch.dim() != 3:
        raise ValueError(f"expected (V, h, w) confidence patch, got {tuple(patch.shape)}")
    if patch.shape[1] == 0 or patch.shape[2] == 0:
        # An area smaller than one feature cell owns no evidence. Zero is the
        # honest answer: it contributes nothing to Eq. 2's noisy-OR.
        return patch.new_zeros(patch.shape[0])
    return patch


class MaxPooling:
    """Strongest single cell in the area (the default, B1).

    Example
    -------
    >>> MaxPooling()(torch.tensor([[[0.1, 0.8], [0.2, 0.3]]]))
    tensor([0.8000])
    """

    name = "max"

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        checked = _check(patch)
        if checked.dim() == 1:
            return checked
        return checked.amax(dim=(1, 2))


class MeanPooling:
    """Average confidence over the area.

    Conservative: an area where one object is seen clearly scores low if the
    rest of the area is empty road. Included for ablation, not recommended.

    Example
    -------
    >>> MeanPooling()(torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))
    tensor([0.5000])
    """

    name = "mean"

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        checked = _check(patch)
        if checked.dim() == 1:
            return checked
        return checked.mean(dim=(1, 2))


class TopKMeanPooling:
    """Mean of the k strongest cells -- a middle ground between max and mean.

    Purpose
        Max is sensitive to a single spuriously confident cell; mean is
        diluted by empty road. Averaging the top k trades one for the other.
        With k >= h*w this degenerates to mean; with k = 1 to max, and both
        degeneracies are asserted by test.

    Inputs  k >= 1.

    Example
    -------
    >>> TopKMeanPooling(k=2)(torch.tensor([[[0.0, 0.4], [0.6, 1.0]]]))
    tensor([0.8000])
    """

    name = "topk_mean"

    def __init__(self, k: int = 4) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        checked = _check(patch)
        if checked.dim() == 1:
            return checked
        v = checked.shape[0]
        flat = checked.reshape(v, -1)
        k = min(self.k, flat.shape[1])
        top, _ = flat.topk(k, dim=1)
        return top.mean(dim=1)


_POOLINGS: Dict[str, Any] = {
    "max": MaxPooling,
    "mean": MeanPooling,
    "topk_mean": TopKMeanPooling,
}


def make_pooling(name: str, **kwargs: Any) -> AreaPooling:
    """Build a pooling strategy by config name (assumption B1).

    Example
    -------
    >>> isinstance(make_pooling("topk_mean", k=2), TopKMeanPooling)
    True
    """
    try:
        cls = _POOLINGS[name]
    except KeyError:
        raise KeyError(
            f"unknown pooling {name!r}; expected one of {sorted(_POOLINGS)}"
        ) from None
    return cls(**kwargs)


def available_poolings() -> Sequence[str]:
    """Names accepted by ``make_pooling``."""
    return sorted(_POOLINGS)
