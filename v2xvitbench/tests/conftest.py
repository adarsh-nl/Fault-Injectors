"""Shared fixtures for v2xvitbench tests.

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
    initialised parameters (per-type projections, relation matrices).
    Without a fixed seed those tests would be flaky in a way that looks
    like a real regression.
    """
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def bev_shape() -> dict:
    """V2X-ViT's tensor geometry, scaled down but structurally identical.

    The real model runs (B=1, L=5, C=256, 48, 176) on V2XSet at fusion
    stride 4. These tests keep the agent count -- it drives the HMSA
    sequence length and the heterogeneity routing, which is what the tests
    are about -- but shrink the grid and channels for speed. The grid stays
    divisible by the test window sizes (2, 4).
    """
    return {"batch": 1, "agents": 5, "channels": 32, "height": 8,
            "width": 16, "anchors": 2, "heads": 4, "dim_head": 8}
