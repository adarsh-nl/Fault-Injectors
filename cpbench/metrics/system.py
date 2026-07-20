"""
system.py
---------
System metrics: inference latency, throughput, memory, communication volume.

Latency uses CUDA events on GPU (accurate around async kernels) and
perf_counter on CPU. All results are plain floats ready for EvalRecord.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch


class SystemProfiler:
    """Measure per-batch inference time and memory.

    Usage
    -----
    >>> prof = SystemProfiler(device)         # doctest: +SKIP
    >>> with prof.measure(n_frames=2):        # doctest: +SKIP
    ...     out = model(batch)                # doctest: +SKIP
    >>> prof.summary()                        # doctest: +SKIP
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.times_ms: List[float] = []
        self.frames: int = 0
        if self.is_cuda:  # pragma: no cover - GPU only
            torch.cuda.reset_peak_memory_stats(device)

    class _Ctx:
        def __init__(self, prof: "SystemProfiler", n_frames: int) -> None:
            self.prof = prof
            self.n_frames = n_frames

        def __enter__(self):
            if self.prof.is_cuda:  # pragma: no cover
                self.start_ev = torch.cuda.Event(enable_timing=True)
                self.end_ev = torch.cuda.Event(enable_timing=True)
                self.start_ev.record()
            else:
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if self.prof.is_cuda:  # pragma: no cover
                self.end_ev.record()
                torch.cuda.synchronize(self.prof.device)
                self.prof.times_ms.append(self.start_ev.elapsed_time(self.end_ev))
            else:
                self.prof.times_ms.append(
                    (time.perf_counter() - self.t0) * 1000.0)
            self.prof.frames += self.n_frames

    def measure(self, n_frames: int = 1) -> "_Ctx":
        return self._Ctx(self, n_frames)

    def summary(self) -> Dict[str, float]:
        total_ms = float(sum(self.times_ms))
        out = {
            "latency_ms_mean": total_ms / max(len(self.times_ms), 1),
            "latency_ms_p95": float(
                sorted(self.times_ms)[int(0.95 * (len(self.times_ms) - 1))])
            if self.times_ms else 0.0,
            "throughput_fps": self.frames / max(total_ms / 1000.0, 1e-9),
            "inference_time_s": total_ms / 1000.0,
        }
        if self.is_cuda:  # pragma: no cover
            out["gpu_mem_peak_mb"] = \
                torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        return out
