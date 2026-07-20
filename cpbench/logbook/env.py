"""
env.py
------
Reproducibility: seeding and environment capture.

`seed_everything` seeds python/numpy/torch (+CUDA) and optionally enables
deterministic kernels; `capture_environment` snapshots every version that
matters (python, torch, CUDA, cuDNN, git commit, hostname, GPU) into a plain
dict for `ExperimentMeta`.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch and CUDA; optionally force determinism.

    Deterministic mode sets cuDNN to deterministic single-algorithm mode and
    enables `torch.use_deterministic_algorithms(warn_only=True)` -- warn-only
    because a few ops (e.g. deform_conv2d backward) have no deterministic
    implementation; those warnings are logged, not fatal.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    logger.info("seeded everything with %d (deterministic=%s)", seed,
                deterministic)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=5, cwd=os.path.dirname(__file__))
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return ""


def capture_environment() -> Dict[str, Any]:
    """Snapshot versions, hardware and git state for the experiment record."""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda or "",
        "cudnn": torch.backends.cudnn.version() or 0,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():  # pragma: no cover - GPU only
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        env["torchvision"] = ""
    return env
