"""The fusion stage -- what to do with whatever arrived.

Where ``comm/`` decides what crosses the link, this subpackage decides how the
receiver combines it. The split is the structural expression of the paper.

    align.py        warp collaborator maps into the ego frame (A12)
    attention.py    per-cell cross-agent attention          (step 8)
    spe.py          sensor positional encoding              (step 8)
    aggregators.py  AttenFusion | MaxFusion | TransformerFusion (step 8, A4)
"""

from .aggregators import (Aggregator, AttenFusion, MaxFusion,
                          TransformerFusion, available_aggregators, key_mask,
                          make_aggregator)
from .align import SpatialTransform, pairwise_to_ego
from .attention import (FeedForward, MultiHeadAttention,
                        ScaledDotProductAttention)
from .spe import SensorPositionalEncoding, sensor_distances

__all__ = ["SpatialTransform", "pairwise_to_ego",
           "ScaledDotProductAttention", "MultiHeadAttention", "FeedForward",
           "SensorPositionalEncoding", "sensor_distances",
           "Aggregator", "AttenFusion", "MaxFusion", "TransformerFusion",
           "make_aggregator", "available_aggregators", "key_mask"]
