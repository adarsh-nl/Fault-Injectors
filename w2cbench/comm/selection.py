"""
selection.py
------------
Phi_select: turn a priority map into a binary message-selection matrix.

    M_{i->j}^(0) = Phi_select(C_i^(0))
    M_{i->j}^(k) = Phi_select(C_i^(k) (X) R_j^(k-1))      k > 0

This is the single most consequential module in the package, and the reason
is assumption A1: the paper and the released implementation define
``Phi_select`` differently, and the two behave in *opposite* ways under a
fault.

    ThresholdSelector   quality per selected cell is constant,
                        bandwidth floats            (released code)
    TopKSelector        bandwidth is constant,
                        quality floats              (the paper's budget)

Corrupt a collaborator's sensor. Its confidence map flattens. Under a
threshold, fewer cells clear the bar, so *the fault reduces measured
bandwidth* while perception degrades -- every efficiency number improves at
the moment the system starts failing. Under top-k, bandwidth is held fixed and
the same budget is spent on cells the agent is no longer confident about, so
the damage lands entirely in accuracy. Both are real deployments, so both are
implemented, and which one is configured decides what the fault benchmark can
see at all.

Training uses neither of them (the curriculum)
----------------------------------------------
The released ``Communication`` module takes a different branch in training: it
draws a random fraction of the map and keeps that many of the highest-priority
cells, ignoring the configured threshold entirely. This is not a shortcut, it
is the paper's curriculum -- "gradually increases communication bandwidth and
rounds, then randomly samples settings for robustness" -- and it is what lets
one checkpoint serve the whole accuracy-versus-bandwidth curve instead of
needing a checkpoint per operating point.

It also means **any communication measurement taken in training mode is
meaningless**: the volume reported would be a sample from the curriculum, not
the model's decision. ``lgcpbench``'s OpenCOOD adapter refuses to measure in
train mode for exactly this reason. Here the training branch lives in the base
class, shared by every strategy, because it belongs to the curriculum rather
than to any one selection rule -- and the accountant in step 6 refuses to
report a volume unless the model is in eval mode.

Gradients
---------
Every branch produces a hard {0, 1} mask, so no gradient reaches the
confidence head through selection. That is fine, and it is precisely why A11
matters: the head is supervised directly by the round-0 detection loss. If
that supervision were removed, the tensor that decides what gets transmitted
would be trained only through whatever survives the mask -- and the mask is
where its gradient was supposed to come from.

Assumption A6 -- the self-link is never masked
----------------------------------------------
``M_{i->i}`` is forced to all ones. An agent does not withhold cells from
itself: its own features are already local and cost nothing to "transmit", so
masking them would discard information for no bandwidth saving. The released
code does the same by overwriting the diagonal.
"""

from __future__ import annotations

import inspect
import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)

# Bytes of index metadata per transmitted cell. Must match
# ``cpbench.comms.MessageChannel.send(sparse=True)``, which charges
# ``n_cells * 4`` on top of the payload; a BudgetSelector that assumed a
# different figure would hand out a budget the channel then over-runs.
INDEX_BYTES_PER_CELL = 4


class Selector(nn.Module, ABC):
    """Base class for message-selection strategies.

    Purpose
        Decide which BEV cells of a sender's feature map cross the link.

    Inputs (to :meth:`forward`)
    ---------------------------
    priority  ``(..., L, L, H, W)`` -- ``priority[i, j]`` is the score for
              sending cell (h, w) from agent *i* to agent *j*. Leading batch
              dimensions are preserved untouched.
    taps      optional read-only observer.
    round_index  names the tap location only.

    Outputs
    -------
    ``(..., L, L, H, W)`` float mask in {0, 1}. Float rather than bool so it
    multiplies features directly and sums to a cell count without a cast --
    the two things every caller does with it.

    Subclass responsibility
    -----------------------
    Implement :meth:`_select_eval`, the strategy. The training branch, the
    self-link rule (A6) and every tap are handled here, so a new strategy
    cannot accidentally skip them.

    Inputs (to ``__init__``)
    ------------------------
    self_mask        ``"ones"`` (A6, the default) or ``"none"`` to disable,
                     which exists so the ablation is reachable and is not
                     recommended.
    train_bandwidth  ``(low, high)`` fraction of the map kept during training;
                     the released curriculum draws uniformly from (0.1, 1.0).
    """

    def __init__(self, self_mask: str = "ones",
                 train_bandwidth: Tuple[float, float] = (0.1, 1.0)) -> None:
        super().__init__()
        if self_mask not in ("ones", "none"):
            raise ValueError(
                f"self_mask must be 'ones' (A6) or 'none', got {self_mask!r}")
        low, high = float(train_bandwidth[0]), float(train_bandwidth[1])
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"train_bandwidth must satisfy 0 <= low <= high <= 1, "
                f"got {train_bandwidth!r}")
        self.self_mask = self_mask
        self.train_bandwidth = (low, high)

    # -- strategy -----------------------------------------------------------

    @abstractmethod
    def _select_eval(self, scores: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Strategy mask from flattened scores ``(..., L, L, H*W)``."""

    def _select_train(self, scores: torch.Tensor,
                      n_cells: int) -> torch.Tensor:
        """The curriculum branch: keep a random *fraction* of the map.

        The cells kept are still the highest-priority ones -- only how many is
        random. Calling it a "random mask" would be wrong and would suggest
        the model is trained on noise; it is trained on every bandwidth.

        Randomness goes through ``torch.rand`` so a run seeded by
        ``cpbench.logbook.seed_everything`` is reproducible.
        """
        low, high = self.train_bandwidth
        fraction = float(torch.rand(1).item()) * (high - low) + low
        k = int(max(1, round(fraction * n_cells)))
        return top_k_mask(scores, k)

    # -- forward ------------------------------------------------------------

    def forward(self, priority: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        """Select cells, apply A6, and report what was chosen.

        Every step is separately observable: the flattened scores entering the
        rule, the mask leaving it, and the per-link cell count -- which is the
        quantity that moves when a sensor fault reaches the protocol rather
        than only the features.
        """
        if priority.dim() < 4 or priority.shape[-4] != priority.shape[-3]:
            raise ValueError(
                f"priority must be (..., L, L, H, W) with a square agent "
                f"block; got {tuple(priority.shape)}. The two agent axes are "
                "sender and receiver, and A6's self-link rule needs them to "
                "index the same agents.")
        *lead, height, width = priority.shape
        n_cells = height * width
        scores = priority.reshape(*lead, n_cells)
        emit(taps, scores, module=type(self).__name__,
             location=f"comm/r{round_index}/selection_scores")

        flat = (self._select_train(scores, n_cells) if self.training
                else self._select_eval(scores, n_cells))
        mask = flat.reshape(*lead, height, width)
        mask = self._apply_self_mask(mask)

        emit(taps, mask, module=type(self).__name__,
             location=f"comm/r{round_index}/selection_mask")
        emit(taps, mask.sum(dim=(-2, -1)), module=type(self).__name__,
             location=f"comm/r{round_index}/selected_count")
        return mask

    def _apply_self_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Force ``M_{i->i} = 1`` (A6).

        Written as an out-of-place ``index_put`` rather than in-place
        assignment because ``mask`` may be a view of an autograd-tracked
        tensor in a subclass that makes selection soft.
        """
        if self.self_mask != "ones":
            return mask
        n_agents = mask.shape[-4]
        eye = torch.eye(n_agents, dtype=torch.bool, device=mask.device)
        return mask.masked_fill(eye[:, :, None, None], 1.0)

    def extra_repr(self) -> str:
        return (f"self_mask={self.self_mask!r}, "
                f"train_bandwidth={self.train_bandwidth}")


def top_k_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Binary mask keeping the `k` largest scores along the last axis.

    Inputs
    ------
    scores  ``(..., N)`` float.
    k       cells to keep; clamped to ``[0, N]``.

    Outputs
    -------
    ``(..., N)`` float mask with exactly ``min(k, N)`` ones per row.

    Example
    -------
    >>> import torch
    >>> top_k_mask(torch.tensor([[0.1, 0.9, 0.5, 0.2]]), k=2)
    tensor([[0., 1., 1., 0.]])
    >>> top_k_mask(torch.tensor([[0.1, 0.9]]), k=0)
    tensor([[0., 0.]])
    >>> top_k_mask(torch.tensor([[0.1, 0.9]]), k=99).sum()
    tensor(2.)
    """
    n = scores.shape[-1]
    k = int(min(max(k, 0), n))
    mask = torch.zeros_like(scores)
    if k == 0:
        return mask
    indices = scores.topk(k, dim=-1, sorted=False).indices
    return mask.scatter(-1, indices, 1.0)


class ThresholdSelector(Selector):
    """Keep every cell whose priority exceeds a fixed threshold.

    The released implementation's rule, and this package's default (A1).
    Bandwidth is an *output* here, not an input: a confident scene costs more
    than an empty one, and a degraded sensor costs less than a healthy one.
    That second property is the pathological case the benchmark exists to
    expose.

    Inputs
    ------
    threshold  released default 0.01. Note assumption A16: with the released
               un-normalised Gaussian kernel this value is effectively 1.28x
               larger, which is why this package normalises the kernel.

    Example
    -------
    >>> import torch
    >>> sel = ThresholdSelector(threshold=0.5).eval()
    >>> one = torch.tensor([0.9, 0.4, 0.6, 0.1]).reshape(1, 1, 2, 2)
    >>> sel(one).squeeze()                 # a self-link, so A6 forces ones
    tensor([[1., 1.],
            [1., 1.]])
    >>> two = one.expand(2, 2, 2, 2)
    >>> sel(two)[0, 1]                     # a cross-link is genuinely selected
    tensor([[1., 0.],
            [1., 0.]])
    """

    def __init__(self, threshold: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)

    def _select_eval(self, scores: torch.Tensor, n_cells: int) -> torch.Tensor:
        return (scores > self.threshold).to(scores.dtype)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, " + super().extra_repr()


class TopKSelector(Selector):
    """Keep a fixed number of the highest-priority cells per link.

    The paper's rule with the budget already resolved to a cell count.
    Bandwidth is an *input*: every link costs the same regardless of what the
    scene or the sensor is doing, so a fault's damage lands entirely in
    accuracy and never in the bandwidth column.

    Inputs
    ------
    k  cells per link, clamped to the map size.

    Example
    -------
    >>> import torch
    >>> sel = TopKSelector(k=2).eval()
    >>> priority = torch.rand(3, 3, 8, 8)
    >>> mask = sel(priority)
    >>> int(mask[0, 1].sum()), int(mask[2, 0].sum())
    (2, 2)
    >>> int(mask[1, 1].sum())              # A6: the self-link is unmasked
    64
    """

    def __init__(self, k: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if int(k) < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.k = int(k)

    def _select_eval(self, scores: torch.Tensor, n_cells: int) -> torch.Tensor:
        return top_k_mask(scores, self.k)

    def extra_repr(self) -> str:
        return f"k={self.k}, " + super().extra_repr()


def make_selector(kind: str, channels: Optional[int] = None,
                  **kwargs) -> "Selector":
    """Build a selection strategy by config name (A1).

    The three strategies take genuinely different arguments -- a threshold, a
    cell count, a byte budget -- so accepted keys are read from each
    constructor's signature rather than kept in a hand-maintained list. One
    config block then serves all three, and a sweep that varies
    ``budget_bytes`` while running ``threshold`` is ignored rather than
    raising.

    ``channels`` is the feature width, needed only by ``budget`` to convert
    bytes into cells. It is passed separately because it is a property of the
    *model*, not of the sweep.

    >>> make_selector("threshold", threshold=0.5).threshold
    0.5
    >>> make_selector("budget", channels=32, budget_bytes=1024).k
    7
    >>> make_selector("topk", k=12).k
    12
    """
    registry = {"threshold": ThresholdSelector, "topk": TopKSelector,
                "budget": BudgetSelector}
    try:
        cls = registry[kind]
    except KeyError:
        raise KeyError(
            f"unknown selector {kind!r}; expected one of "
            f"{sorted(registry)}") from None

    # A null in config means "not set", and passing it on would collide with a
    # value the target computes: BudgetSelector derives `k` from its budget, so
    # a forwarded `k=None` reaches TopKSelector.__init__ twice.
    supplied = {key: value for key, value in kwargs.items() if value is not None}
    if channels is not None:
        supplied["channels"] = channels
    accepted = set()
    for klass in cls.__mro__:
        if klass is object:
            break
        accepted |= set(inspect.signature(klass.__init__).parameters)
    accepted -= {"self", "kwargs"}
    missing = [name for name, parameter
               in inspect.signature(cls.__init__).parameters.items()
               if parameter.default is inspect.Parameter.empty
               and name not in ("self", "kwargs") and name not in supplied]
    if missing:
        raise ValueError(
            f"selector {kind!r} needs {missing}; got {sorted(supplied)}")
    return cls(**{k: v for k, v in supplied.items() if k in accepted})


class BudgetSelector(TopKSelector):
    """Top-k with `k` derived from a per-link byte budget.

    The paper's rule as the paper states it: select the largest elements
    *subject to a communication budget*. This is the selector that traces the
    accuracy-versus-bandwidth curve, because sweeping ``budget_bytes`` sweeps
    the x-axis directly instead of via a cell count that has to be converted
    by hand for every feature width.

    The byte arithmetic mirrors ``cpbench.comms.MessageChannel`` exactly --
    payload ``channels * bytes_per_element`` plus ``INDEX_BYTES_PER_CELL`` of
    coordinates, per selected cell. If the two ever disagreed the budget would
    be a number the channel quietly exceeds, which is the one failure a
    bandwidth-constrained benchmark cannot tolerate. A test runs a real
    ``MessageChannel`` over this selector's mask and asserts the budget holds.

    Inputs
    ------
    budget_bytes       bytes allowed per link per round.
    channels           D, the feature width being transmitted.
    bytes_per_element  transmission precision (A8: 4 for this package).

    Example
    -------
    >>> sel = BudgetSelector(budget_bytes=1024, channels=32,
    ...                      bytes_per_element=4)
    >>> sel.bytes_per_cell                 # 32 * 4 payload + 4 index
    132
    >>> sel.k                              # floor(1024 / 132)
    7
    """

    def __init__(self, budget_bytes: int, channels: int,
                 bytes_per_element: int = 4, **kwargs) -> None:
        if int(budget_bytes) < 0:
            raise ValueError(
                f"budget_bytes must be non-negative, got {budget_bytes}")
        if int(channels) <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        bytes_per_cell = (int(channels) * int(bytes_per_element)
                          + INDEX_BYTES_PER_CELL)
        super().__init__(k=int(budget_bytes) // bytes_per_cell, **kwargs)
        self.budget_bytes = int(budget_bytes)
        self.bytes_per_cell = bytes_per_cell
        logger.info("BudgetSelector: %d bytes / %d bytes-per-cell -> k=%d "
                    "(log2 budget %.2f)", self.budget_bytes, bytes_per_cell,
                    self.k, math.log2(max(self.budget_bytes, 1)))

    def extra_repr(self) -> str:
        return (f"budget_bytes={self.budget_bytes}, "
                f"bytes_per_cell={self.bytes_per_cell}, " + super().extra_repr())
