"""Shared fixtures for w2cbench tests.

Everything here is CPU-only and small enough to run in milliseconds. No
dataset, no downloads, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed every test.

    Autouse because several tests assert on values drawn from randomly
    initialised parameters (projections, depth heads). Without a fixed seed
    those tests would be flaky in a way that looks like a real regression.
    """
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def bev_shape() -> dict:
    """Where2comm's tensor geometry, scaled down but structurally identical.

    The real LiDAR track is (B=1, L=5, D=256, 100, 252) on OPV2V. These tests
    keep the agent count -- it drives the communication graph, the selection
    matrix and the attention sequence length, which is what the tests are
    about -- but shrink the grid and channels for speed.
    """
    return {"batch": 1, "agents": 5, "channels": 32, "height": 16,
            "width": 16, "anchors": 2, "heads": 4, "dim_head": 8}
