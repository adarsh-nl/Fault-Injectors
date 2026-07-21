"""
rel_pos_bias.py
---------------
Learned relative position bias over an N-dimensional token window.

CoBEVT adds a bias ``B`` to the attention logits before the softmax
(paper Eq. 4)::

    3D-Rel-Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + B) V

with ``B`` drawn from a table indexed by the *relative offset* between two
tokens. FuseBEVT uses a **3-D** window ``(agent, height, width)``; SinBEVT's
terminal self-attention uses a **2-D** window ``(height, width)``.

The agent axis is the interesting one
-------------------------------------
A bias indexed by agent offset is what lets the model learn something like
"the collaborator two slots away is systematically offset from me" rather
than treating the agents as an unordered set. It is also why the agent axis
must always be padded to exactly ``agent_size``: the table is allocated for a
fixed window extent, so a variable agent count would index outside it.
Dropped agents are handled by the attention mask, never by resizing this
table -- see ``fusion/fusebevt.py``.

One implementation, both ranks
------------------------------
The 2-D and 3-D cases differ only in how many coordinate axes the meshgrid
has, so there is one class over an arbitrary window rank rather than two
near-duplicates. (The design doc named ``RelativePositionBias3D`` and
``RelativePositionBias2D``; collapsing them removes the more likely bug,
which is the two copies drifting apart.)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class RelativePositionBias(nn.Module):
    """Learned bias over relative offsets within a token window.

    Purpose
        Give attention a sense of *where* two tokens sit relative to each
        other -- across space, and (in FuseBEVT) across agents.

    Inputs
    ------
    window_size  extent per axis, e.g. ``(5, 8, 8)`` for FuseBEVT's
                 (agent, h, w) window or ``(32, 32)`` for SinBEVT's terminal
                 self-attention. Any rank >= 1 is supported.
    num_heads    one bias table column per head; heads learn different
                 spatial preferences, which is most of the point.

    Outputs
    -------
    ``forward()`` -> ``(num_heads, T, T)`` where ``T = prod(window_size)``,
    ready to add to attention logits of shape ``(B, num_heads, T, T)``.

    Shapes
    ------
    table   (prod(2 * w - 1), num_heads) learnable
    index   (T, T) int64 buffer, constant

    Example
    -------
    >>> import torch
    >>> bias = RelativePositionBias(window_size=(5, 8, 8), num_heads=4)
    >>> bias.num_tokens, bias.table.num_embeddings
    (320, 2025)
    >>> bias().shape
    torch.Size([4, 320, 320])
    """

    def __init__(self, window_size: Sequence[int], num_heads: int) -> None:
        super().__init__()
        self.window_size: Tuple[int, ...] = tuple(int(w) for w in window_size)
        if not self.window_size or any(w < 1 for w in self.window_size):
            raise ValueError(
                f"window_size must be a non-empty tuple of positive ints, "
                f"got {tuple(window_size)}")
        self.num_heads = int(num_heads)
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.num_tokens = 1
        for extent in self.window_size:
            self.num_tokens *= extent

        n_offsets = 1
        for extent in self.window_size:
            n_offsets *= 2 * extent - 1
        self.table = nn.Embedding(n_offsets, self.num_heads)
        nn.init.trunc_normal_(self.table.weight, std=0.02)

        self.register_buffer("index", self._build_index(), persistent=False)

    # -- index construction -------------------------------------------------

    def _build_index(self) -> torch.Tensor:
        """(T, T) int64 lookup into the offset table.

        Offsets are shifted to be non-negative, then flattened with mixed
        radix -- the standard Swin construction, generalised to any rank.
        """
        axes = [torch.arange(extent) for extent in self.window_size]
        # indexing="ij" matters: the default would transpose the grid for
        # rank >= 2 and silently pair each token with the wrong offset.
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"))  # (R, *ws)
        flat = coords.flatten(1)                                    # (R, T)
        relative = flat[:, :, None] - flat[:, None, :]              # (R, T, T)
        relative = relative.permute(1, 2, 0).contiguous()           # (T, T, R)

        for axis, extent in enumerate(self.window_size):
            relative[..., axis] += extent - 1        # shift to [0, 2w-2]

        # Mixed-radix flatten, least significant axis last.
        stride = 1
        index = torch.zeros(relative.shape[:2], dtype=torch.long)
        for axis in reversed(range(len(self.window_size))):
            index += relative[..., axis] * stride
            stride *= 2 * self.window_size[axis] - 1
        return index

    # -- forward ------------------------------------------------------------

    def forward(self, taps: Optional[TapProtocol] = None,
                location: str = "attention/rel_pos_bias") -> torch.Tensor:
        """Materialise the bias as ``(num_heads, T, T)``.

        Takes ``taps`` like every other forward in this package: the bias is
        a first-class observation point, because "did attention learn to
        down-weight a particular agent offset?" is a question about this
        tensor and nothing else.
        """
        bias = self.table(self.index)                 # (T, T, heads)
        bias = bias.permute(2, 0, 1).contiguous()     # (heads, T, T)
        emit(taps, bias, module="RelativePositionBias", location=location)
        return bias

    def extra_repr(self) -> str:
        return (f"window_size={self.window_size}, num_heads={self.num_heads}, "
                f"num_tokens={self.num_tokens}")
