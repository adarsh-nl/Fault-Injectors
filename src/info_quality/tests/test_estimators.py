"""
test_estimators.py
------------------
Validate the MI estimators against a closed-form ground truth.

For Z, Y jointly Gaussian with per-dimension correlation rho,

    I(Z; Y) = -(dim / 2) * log(1 - rho^2)   nats.

These tests check the estimators recover the right magnitude and ordering on
this benchmark. They use synthetic data only: no model, no dataset. That is
also the point of the module under test being architecture- and data-agnostic.

Run with:  pytest src/info_quality/tests/test_estimators.py -v
"""

import numpy as np
import pytest

from src.info_quality.estimators import (
    InfoNCEEstimator,
    SMILEEstimator,
    correlated_gaussians,
    delta_information,
)


def test_true_mi_formula_matches_generator():
    _, _, mi = correlated_gaussians(n=10, dim=4, rho=0.8, seed=0)
    assert mi == pytest.approx(-0.5 * 4 * np.log(1 - 0.8 ** 2), rel=1e-6)


def test_infonce_is_a_lower_bound_and_tracks_signal():
    # Higher correlation must not produce a lower estimate.
    Z_lo, Y_lo, _ = correlated_gaussians(1000, dim=4, rho=0.3, seed=1)
    Z_hi, Y_hi, true_hi = correlated_gaussians(1000, dim=4, rho=0.8, seed=2)

    est = InfoNCEEstimator(epochs=50, batch_size=128, seed=0, device='cpu')
    mi_lo = est.estimate(Z_lo, Y_lo).mi_nats
    mi_hi = est.estimate(Z_hi, Y_hi).mi_nats

    assert mi_hi > mi_lo                 # tracks the signal
    assert mi_hi <= true_hi + 0.5        # stays a (slightly slack) lower bound
    assert mi_lo > 0.0                   # detects real dependence


def test_smile_recovers_magnitude():
    Z, Y, true_mi = correlated_gaussians(800, dim=4, rho=0.8, seed=3)
    est = SMILEEstimator(epochs=300, batch_size=128, warmup=100, avg_last=40,
                         eval_every=5, seed=0, device='cpu')
    mi = est.estimate(Z, Y).mi_nats
    # SMILE has no log(N) ceiling; expect the right ballpark, not exactness.
    assert mi == pytest.approx(true_mi, abs=0.7)


def test_holdout_removes_infonce_bias_on_independent_data():
    # True MI is 0. In-sample InfoNCE inflates it (the critic memorises the
    # pairing); a held-out split brings it back near 0. This is exactly why
    # absolute MI claims should use holdout > 0.
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((1000, 4)).astype(np.float32)
    Y = rng.standard_normal((1000, 4)).astype(np.float32)  # independent of Z

    in_sample = InfoNCEEstimator(epochs=50, batch_size=128, holdout=0.0,
                                 seed=0, device='cpu').estimate(Z, Y).mi_nats
    held_out = InfoNCEEstimator(epochs=50, batch_size=128, holdout=0.3,
                                seed=0, device='cpu').estimate(Z, Y).mi_nats

    assert held_out < 0.15               # ~0 once evaluated on unseen pairs
    assert held_out < in_sample          # holdout is the more honest estimate


def test_seed_makes_runs_reproducible():
    Z, Y, _ = correlated_gaussians(500, dim=4, rho=0.6, seed=5)
    est = InfoNCEEstimator(epochs=40, batch_size=64, seed=42, device='cpu')
    a = est.estimate(Z, Y).mi_nats
    b = est.estimate(Z, Y).mi_nats
    assert a == pytest.approx(b, abs=1e-6)


def test_delta_information():
    mi = {'camera': 1.0, 'lidar': 1.5, 'fused': 2.2}
    assert delta_information(mi, 'fused', ['camera', 'lidar']) == pytest.approx(0.7)
