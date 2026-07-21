"""
The paper's contribution, isolated from CoBEVT itself.

This sub-package knows nothing about cameras, agents or bird's-eye views. It
imports only torch, einops and `cpbench.observation`. That is deliberate: the
subtle bugs in FAX live in the partition inverse, the relative-position-bias
indexing and the mask broadcast, and all three are only unit-testable if they
are separable from the model that uses them.

Contents
--------
partition       window / grid token grouping and their inverses
rel_pos_bias    learned bias over relative offsets (3-D in FuseBEVT, 2-D in
                SinBEVT's terminal self-attention)
qkv             fused and separate Q/K/V projections, head split/merge
attention       scaled dot-product attention with every intermediate tapped
mlp             the position-wise feed-forward network
fax_self        FAX self-attention: local half + global half (FuseBEVT)
fax_cross       FAX cross-attention: BEV query reads image features (SinBEVT)
"""

from .attention import ScaledDotProductAttention
from .fax_cross import CrossWindowAttention, FAXCrossAttentionBlock
from .fax_self import FAXAttentionHalf, FAXSelfAttentionBlock
from .mlp import FeedForward
from .partition import (GRID, WINDOW, as_window_size, pad_to_multiple,
                        partition, unpartition)
from .qkv import (FusedQKVProjection, SeparateQKVProjection, merge_heads,
                  split_heads)
from .rel_pos_bias import RelativePositionBias

__all__ = [
    "WINDOW", "GRID", "partition", "unpartition", "pad_to_multiple",
    "as_window_size",
    "RelativePositionBias",
    "FusedQKVProjection", "SeparateQKVProjection", "split_heads", "merge_heads",
    "ScaledDotProductAttention",
    "FeedForward",
    "FAXAttentionHalf", "FAXSelfAttentionBlock",
    "CrossWindowAttention", "FAXCrossAttentionBlock",
]
