"""The communication stage -- Where2comm's contribution.

Everything in this subpackage decides *what crosses the V2X link*, as opposed
to ``fusion/``, which decides what to do with whatever arrived. The split is
the structural expression of the paper, and it means the answer to "what can a
bandwidth fault touch" is a directory listing.

    smoothing.py   Gaussian low-pass over the confidence map (A9)
    request.py     R = 1 - C, the control payload            (step 5)
    selection.py   Phi_select: threshold / top-k / budget    (step 5)
    packing.py     Z = M (x) F, and the byte accounting      (step 6)
    graph.py       A_{i,j}, which links exist                (step 6)
    volume.py      per-round bookkeeping into CommVolumeMetrics (step 6)
"""

from .graph import CommunicationGraph, graph_density, incoming_links
from .packing import MessagePacker, message_statistics
from .request import RequestMapGenerator
from .selection import (BudgetSelector, Selector, ThresholdSelector,
                        TopKSelector, make_selector, top_k_mask)
from .smoothing import GaussianSmoother, gaussian_kernel_2d
from .volume import CommVolumeAccountant

__all__ = ["GaussianSmoother", "gaussian_kernel_2d", "RequestMapGenerator",
           "Selector", "ThresholdSelector", "TopKSelector", "BudgetSelector",
           "top_k_mask", "make_selector", "MessagePacker", "message_statistics",
           "CommunicationGraph", "graph_density", "incoming_links",
           "CommVolumeAccountant"]
