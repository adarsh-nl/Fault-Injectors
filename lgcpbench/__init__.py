"""
lgcpbench
=========
Reference implementation and fault-injection benchmark for

    Efficient Local-to-Global Collaborative Perception via Joint
    Communication and Computation Optimization
    Hui Zhang, Yuquan Yang, Zechuan Gong, Xiaohua Xu, Dan Keun Sung
    arXiv:2601.12749v1 [cs.DC], 19 Jan 2026

Design document: ``docs/lgcp_design.md``.

What LGCP is
------------
LGCP is *not* a neural network architecture. It is a distributed scheduling
and orchestration framework that wraps existing collaborative perception
models (Where2comm, CoBEVT, CoAlign). Its contributions are combinatorial:

    C1  RoI -> non-overlapping 10m x 6m areas          (lgcpbench.roi)
    C2  area confidence + noisy-OR combination         (lgcpbench.confidence)
    C3  greedy group selection under threshold dg      (lgcpbench.selection)
    C4  min-max load-balanced leader election          (lgcpbench.selection)
    C5  conflict-free packet scheduling over Z chans   (lgcpbench.network)
    C6  end-to-end latency model and objective         (lgcpbench.network)

The three-plane contract
------------------------
1. Corruption plane (physical, upstream). Faults hit raw poses, LiDAR,
   images and the V2X link, applied by ``DataFaultBridge`` BEFORE any tensor
   exists. No model code corrupts a tensor.
2. Measurement plane (passive). Every module accepts ``taps=None`` and calls
   ``emit``; observation cannot alter the forward pass.
3. Control plane (decisions). LGCP's own contribution lives in RSU decisions
   -- partitions, confidence reports, groups, leaders, schedules. Faults are
   applied only at the message boundary between protocol stages; the paper's
   algorithms are never fault-aware.
"""

__all__ = ["__version__", "PAPER"]

__version__ = "0.1.0"

PAPER = "arXiv:2601.12749v1"
