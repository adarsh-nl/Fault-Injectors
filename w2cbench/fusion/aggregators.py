"""
aggregators.py
--------------
Stage 4: collapse the agent axis into one fused ego map.

    F_i^(k+1) = FFN( sum_{j in N_i u {i}} W_{j->i}^(k) (X) Z_{j->i}^(k) )

Three implementations, because the released code ships three and the paper
describes the third (A4). Which one is configured is a research variable, so
each is its own class rather than a branch inside one.

    AttenFusion        parameter-free scaled dot-product over the agent axis
                       (the released default, and this package's)
    MaxFusion          elementwise maximum over agents
    TransformerFusion  learned multi-head attention + FFN, optional sensor
                       positional encoding, optional confidence weighting --
                       the paper's equation

A discrepancy worth stating plainly (A5)
----------------------------------------
The paper's fusion multiplies the attention weights by the sender's confidence,
``W_{j->i} = MHA_W(...) (X) C_j``. The released ``AttenFusion`` does not: it is
plain attention with no confidence term anywhere. So the default configuration
does **not** implement the equation the paper gives, and the difference is not
cosmetic -- confidence weighting is the mechanism by which a collaborator's own
assessment of its reliability is supposed to enter fusion, which is exactly the
behaviour a fault benchmark wants to test. ``with_scm`` on
:class:`TransformerFusion` is the paper's reading; the ablation between them is
one config key.

AttenFusion has no parameters at all
------------------------------------
Worth being explicit, because it is surprising. The released
``ScaledDotProductAttention`` is a raw ``bmm`` with a ``1/sqrt(d)`` scale and no
projections, and ``AttenFusion`` merely reshapes around it. So Where2comm's
default fusion learns nothing: every parameter in the model is in the encoder
and the detection head. That makes the encoder's features do all the work, and
it means a fault that degrades features has nowhere to be absorbed downstream.

The released implementation computes self-attention over all ``L`` agents and
then keeps row 0. Row 0 of a self-attention output *is* ego-as-query
cross-attention -- ``softmax(q_0 K^T) V`` -- so computing only that row is both
faithful and ``L`` times cheaper. That is what is done here.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

from .attention import FeedForward, MultiHeadAttention, ScaledDotProductAttention

logger = logging.getLogger(__name__)


def key_mask(valid: Optional[torch.Tensor] = None,
             graph: Optional[torch.Tensor] = None,
             shape: Optional[tuple] = None) -> Optional[torch.Tensor]:
    """Combine the warp's validity mask and the communication graph.

    Two different absences, one boolean: a collaborator that does not cover
    this cell, and one that sent nothing at all. Neither is a reading of zero,
    which is the whole reason a mask exists rather than relying on the zeros
    already in the tensor.

    Inputs
    ------
    valid  ``(B, L, 1, H, W)`` from :class:`~w2cbench.fusion.align.SpatialTransform`
    graph  ``(B, L)`` or ``(L,)`` incoming-link indicator for the receiver
    shape  ``(B, L, H, W)`` used when only `graph` is given

    Outputs
    -------
    ``(B, L, H, W)`` bool, or None when neither input was supplied.

    Example
    -------
    >>> import torch
    >>> valid = torch.ones(1, 3, 1, 2, 2)
    >>> valid[0, 2] = 0.0                       # agent 2 covers nothing here
    >>> graph = torch.tensor([[1.0, 0.0, 1.0]]) # agent 1 sent nothing
    >>> key_mask(valid, graph)[0, :, 0, 0]
    tensor([ True, False, False])
    """
    mask = None
    if valid is not None:
        mask = valid.squeeze(2).to(torch.bool)
    if graph is not None:
        link = graph.to(torch.bool)
        if link.dim() == 1:
            link = link.unsqueeze(0)
        link = link[..., None, None]
        mask = link.expand(*(shape or mask.shape)) if mask is None else mask & link
    return mask


class Aggregator(nn.Module, ABC):
    """Base class for the agent-axis collapse.

    Inputs (to :meth:`forward`)
    ---------------------------
    messages    ``(B, L, D, H, W)`` warped into the ego frame, ego at index 0.
    confidence  ``(B, L, 1, H, W)`` sender confidence, warped alongside; used
                only when confidence weighting is on (A5).
    mask        ``(B, L, H, W)`` bool from :func:`key_mask`; True means usable.
    distances   ``(B, L, H, W)`` metres, for the sensor positional encoding.

    Outputs
    -------
    ``(B, D, H, W)`` -- the fused ego map, ``F^(k+1)``.

    Every subclass emits ``fusion/r{k}/input``, ``/aggregated`` and
    ``/output``, which the base class handles, so a new aggregator gets the
    common observation points for free and cannot forget them.
    """

    def __init__(self, ego_index: int = 0) -> None:
        super().__init__()
        self.ego_index = int(ego_index)

    @abstractmethod
    def _aggregate(self, messages: torch.Tensor,
                   confidence: Optional[torch.Tensor],
                   mask: Optional[torch.Tensor],
                   distances: Optional[torch.Tensor],
                   taps: Optional[TapProtocol], round_index: int
                   ) -> torch.Tensor:
        """Collapse ``(B, L, D, H, W)`` to ``(B, D, H, W)``."""

    def forward(self, messages: torch.Tensor,
                confidence: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                distances: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        if messages.dim() != 5:
            raise ValueError(
                f"expected messages (B, L, D, H, W), got "
                f"{tuple(messages.shape)}")
        emit(taps, messages, module=type(self).__name__,
             location=f"fusion/r{round_index}/input")

        aggregated = self._aggregate(messages, confidence, mask, distances,
                                     taps, round_index)
        emit(taps, aggregated, module=type(self).__name__,
             location=f"fusion/r{round_index}/aggregated")
        emit(taps, aggregated, module=type(self).__name__,
             location=f"fusion/r{round_index}/output")
        return aggregated


class MaxFusion(Aggregator):
    """Elementwise maximum over the agent axis.

    The cheapest baseline the released code ships, and a useful control: it has
    no notion of which collaborator a value came from, so any robustness it
    shows is the encoder's rather than fusion's. A masked agent is pushed to
    ``-inf`` before the maximum rather than to zero, since features are signed
    and zero is a plausible value.

    Example
    -------
    >>> import torch
    >>> fuse = MaxFusion()
    >>> messages = torch.zeros(1, 2, 3, 2, 2)
    >>> messages[0, 1] = 5.0
    >>> float(fuse(messages).max())
    5.0
    """

    def _aggregate(self, messages, confidence, mask, distances, taps,
                   round_index):
        if mask is not None:
            neg_inf = torch.finfo(messages.dtype).min
            messages = messages.masked_fill(~mask.unsqueeze(2), neg_inf)
        fused = messages.amax(dim=1)
        # An all-masked cell maxes to finfo.min; zero is the honest stand-in.
        return torch.where(fused <= torch.finfo(messages.dtype).min / 2,
                           torch.zeros_like(fused), fused)


class AttenFusion(Aggregator):
    """Parameter-free scaled dot-product attention over the agent axis.

    The released default (A4), and this package's. Each BEV cell attends with
    the ego's own feature vector as query and every agent's as key and value,
    with no learned projections anywhere -- so this module has zero parameters
    and the attention is a pure similarity between raw feature vectors.

    Does **not** apply confidence weighting; that is A5, and
    :class:`TransformerFusion` is where the paper's equation lives.

    Example
    -------
    >>> import torch
    >>> fuse = AttenFusion(dim=4)
    >>> list(fuse.parameters())
    []
    >>> fuse(torch.randn(2, 3, 4, 5, 5)).shape
    torch.Size([2, 4, 5, 5])
    """

    def __init__(self, dim: int, ego_index: int = 0) -> None:
        super().__init__(ego_index=ego_index)
        self.dim = int(dim)
        self.attend = ScaledDotProductAttention(dim)

    def _aggregate(self, messages, confidence, mask, distances, taps,
                   round_index):
        batch, agents, channels, height, width = messages.shape
        # (B, L, D, H, W) -> (B*H*W, 1, L, D): one attention problem per cell.
        tokens = messages.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, 1, agents, channels)
        query = tokens[:, :, self.ego_index:self.ego_index + 1]

        flat_mask = None
        if mask is not None:
            flat_mask = mask.permute(0, 2, 3, 1).reshape(
                batch * height * width, 1, 1, agents)

        context, _ = self.attend(query, tokens, tokens, mask=flat_mask,
                                 taps=taps, round_index=round_index)
        return context.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2).contiguous()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, parameters=0"


class TransformerFusion(Aggregator):
    """Learned multi-head attention with the paper's confidence weighting.

    Purpose
        The paper's Stage-4 equation as written: multi-head attention whose
        weights are multiplied by the sender's spatial confidence, followed by
        a residual feed-forward block.

    Inputs
    ------
    dim        model width D.
    heads      attention heads.
    with_spe   add the sensor positional encoding to query and keys (off by
               default, matching the released config).
    with_scm   multiply the attention weights by ``C_j`` -- the paper's
               reading (A5). On by default *for this class*, because a
               transformer aggregator without it is just a parameterised
               AttenFusion and the ablation would have no contrast.
    mlp_dim, dropout   feed-forward geometry.

    Example
    -------
    >>> import torch
    >>> fuse = TransformerFusion(dim=8, heads=2).eval()
    >>> messages = torch.randn(1, 3, 8, 4, 4)
    >>> confidence = torch.rand(1, 3, 1, 4, 4)
    >>> fuse(messages, confidence=confidence).shape
    torch.Size([1, 8, 4, 4])
    """

    def __init__(self, dim: int, heads: int = 8, with_spe: bool = False,
                 with_scm: bool = True, mlp_dim: Optional[int] = None,
                 dropout: float = 0.0, ego_index: int = 0) -> None:
        super().__init__(ego_index=ego_index)
        self.dim = int(dim)
        self.with_spe = bool(with_spe)
        self.with_scm = bool(with_scm)
        self.attention = MultiHeadAttention(dim, heads=heads)
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_output = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout=dropout)
        if with_spe:
            from .spe import SensorPositionalEncoding
            self.spe = SensorPositionalEncoding(dim)
        else:
            self.spe = None

    def _aggregate(self, messages, confidence, mask, distances, taps,
                   round_index):
        if self.with_scm and confidence is None:
            raise ValueError(
                "TransformerFusion(with_scm=True) needs the senders' "
                "confidence maps; pass confidence=(B, L, 1, H, W), warped "
                "into the ego frame alongside the features. Without it the "
                "paper's W = MHA (X) C_j cannot be formed (A5).")

        if self.spe is not None:
            if distances is None:
                raise ValueError(
                    "TransformerFusion(with_spe=True) needs sensor distances; "
                    "pass distances=(B, L, H, W) from "
                    "w2cbench.fusion.spe.sensor_distances")
            messages = messages + self.spe(distances, taps=taps,
                                           round_index=round_index)

        batch, agents, channels, height, width = messages.shape
        cells = batch * height * width
        tokens = messages.permute(0, 3, 4, 1, 2).reshape(cells, agents, channels)
        tokens = self.norm_attention(tokens)
        query = tokens[:, self.ego_index:self.ego_index + 1]

        flat_mask = None
        if mask is not None:
            flat_mask = mask.permute(0, 2, 3, 1).reshape(cells, 1, 1, agents)

        gate = None
        if self.with_scm:
            gate = confidence.squeeze(2).permute(0, 2, 3, 1).reshape(
                cells, 1, 1, agents)

        context, weights = self.attention(query, tokens, mask=flat_mask,
                                          gate=gate, taps=taps,
                                          round_index=round_index)
        if gate is not None:
            emit(taps, weights, module="TransformerFusion",
                 location=f"fusion/r{round_index}/confidence_weighted")

        fused = query + context
        fused = fused + self.feed_forward(self.norm_output(fused), taps=taps,
                                          round_index=round_index)
        return fused.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2).contiguous()

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, with_spe={self.with_spe}, "
                f"with_scm={self.with_scm}")


_AGGREGATORS = {"atten": AttenFusion, "max": MaxFusion,
                "transformer": TransformerFusion}


def make_aggregator(name: str, dim: int, **kwargs) -> Aggregator:
    """Build an aggregator by config name (A4).

    One config block describes fusion for every strategy, but the strategies
    genuinely differ in what they accept -- ``MaxFusion`` has no width and no
    heads, ``AttenFusion`` has a width and no heads, only ``TransformerFusion``
    has both. Rather than a hand-maintained list of which keys to drop for
    which class (which rots the moment an aggregator gains an option), the
    accepted keys are read from each constructor's signature. A config setting
    ``heads`` while running ``atten`` is then silently ignored rather than
    raising, which is the right behaviour for a shared config group: the same
    YAML has to serve all three.

    >>> isinstance(make_aggregator("max", dim=8), MaxFusion)
    True
    >>> make_aggregator("transformer", dim=8, heads=2).with_scm
    True
    >>> make_aggregator("atten", dim=8, heads=2).dim   # heads: not applicable
    8
    """
    try:
        cls = _AGGREGATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown aggregator {name!r}; expected one of "
            f"{sorted(_AGGREGATORS)}") from None
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    supplied = {"dim": dim, **kwargs}
    return cls(**{k: v for k, v in supplied.items() if k in accepted})


def available_aggregators() -> list:
    """Names accepted by :func:`make_aggregator`."""
    return sorted(_AGGREGATORS)
