"""Shared fixtures for cobevtbench tests.

Everything here is CPU-only and small enough to run in milliseconds. No
dataset, no downloads, no GPU -- the same guarantee the other benchmark
packages make.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed every test.

    Autouse because several tests assert on values drawn from randomly
    initialised parameters (bias tables, projections). Without a fixed seed
    those tests would be flaky in a way that looks like a real regression.
    """
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def fuse_shape() -> dict:
    """FuseBEVT's tensor geometry, scaled down but structurally identical.

    Real CoBEVT is (B=1, L=5, C=128, 32, 32) with window 8. These tests use
    the same agent count and window size -- those two drive the relative
    position bias table and the masking logic, which is what the tests are
    about -- but a smaller grid and channel count for speed.
    """
    return {"batch": 2, "agents": 5, "channels": 32, "height": 16,
            "width": 16, "window": 8, "dim_head": 8}
