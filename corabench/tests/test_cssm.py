"""CSSM: chunked scan exactness, shapes, gradients, scan modes."""

import torch

from corabench.fusion.cssm import CSSM, _chunked_selective_scan


def _naive(x, delta, A, B, C):
    bt, L, d = x.shape
    h = torch.zeros(bt, d, A.shape[1])
    ys = []
    for t in range(L):
        a = torch.exp(delta[:, t].unsqueeze(-1) * A)
        h = a * h + (delta[:, t] * x[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, 1)


def test_chunked_scan_matches_naive_recurrence():
    torch.manual_seed(0)
    x = torch.randn(2, 41, 5)
    delta = torch.rand(2, 41, 5) * 0.3
    A = -torch.rand(5, 3) * 2
    B, C = torch.randn(2, 41, 3), torch.randn(2, 41, 3)
    out, stats = _chunked_selective_scan(x, delta, A, B, C, chunk=7)
    assert torch.allclose(out, _naive(x, delta, A, B, C), atol=1e-5)
    # This regime is deliberately benign -- delta <= 0.3, |A| <= 2, so
    # |dA| <= 0.6 and logE never approaches the -30 floor over 7-position
    # chunks. The chunked closed form agrees with the naive scan exactly
    # HERE, and only here: agreement is what breaks once the clamp engages,
    # because a pinned E_t/E_s reads as 1 (RECON-4). Asserting the saturated
    # band is empty pins that this test covers the unclamped path and nothing
    # more -- it must not silently drift into validating the degenerate one.
    assert float(stats["saturated"]) == 0.0, (
        "the reference-agreement test has drifted into the clamped regime; "
        "it no longer validates what it claims to")
    assert abs(float(stats["saturated"]) + float(stats["healthy"])
               + float(stats["integrator"]) - 1.0) < 1e-6, \
        "the three logE bands must partition the entries exactly once"


def test_cssm_shapes_and_grad():
    m = CSSM(channels=16, d_inner=8, d_state=4, pool=2)
    zf = torch.rand(2, 16, 10, 12, requires_grad=True)
    y = m(zf, torch.rand(2, 16, 10, 12))
    assert y.shape == (2, 16, 10, 12)
    y.sum().backward()
    assert torch.isfinite(zf.grad).all()


def test_scan_modes_and_full_resolution():
    for scan in ("raster", "cross2d"):
        m = CSSM(channels=8, d_inner=4, d_state=2, scan=scan, pool=1)
        assert m(torch.rand(1, 8, 6, 6),
                 torch.rand(1, 8, 6, 6)).shape == (1, 8, 6, 6)


def test_ego_output_matrix_matters():
    """Eq. 8: Z_i parameterises the output matrix -> changing it changes y."""
    m = CSSM(channels=8, d_inner=4, d_state=2, pool=1)
    zf = torch.rand(1, 8, 6, 6)
    y1 = m(zf, torch.rand(1, 8, 6, 6))
    y2 = m(zf, torch.rand(1, 8, 6, 6))
    assert not torch.allclose(y1, y2)


def test_scan_stats_survive_tensors_above_the_quantile_limit():
    """torch.quantile refuses inputs above 2**24 elements.

    The scan's diagnostic statistics are computed over a (Bt, L, D, N) tensor,
    which is 2*8800*64*16 = 18.0M on the OPV2V grid but only 1.18M on the
    576-position synthetic one. A `.quantile(0.95)` there therefore passed
    every local and CI check and then killed jobs 549332 and 549333 in the
    first forward pass, with an EMPTY stderr and a bundle containing no
    metrics.csv at all -- the failure looked like the scale-floor abort it
    was not.

    This shape is the cheapest one that still crosses the limit (16.78M in
    0.6s), so the guard is affordable enough to keep in the default suite.
    Scale-dependent limits do not reproduce at test scale unless a test is
    written at scale.
    """
    torch.manual_seed(0)
    bt, length, d, n = 1, 4097, 64, 64
    assert bt * length * d * n > 2 ** 24, "shape no longer crosses the limit"
    x = torch.randn(bt, length, d)
    delta = torch.rand(bt, length, d) * 0.3
    A = -torch.rand(d, n) * 2
    B, C = torch.randn(bt, length, n), torch.randn(bt, length, n)

    y, stats = _chunked_selective_scan(x, delta, A, B, C, chunk=64)

    assert y.shape == (bt, length, d)
    assert all(torch.isfinite(v).all() for v in stats.values())
    assert float(stats["horizon_p95"]) >= float(stats["horizon_p50"]) > 0.0
