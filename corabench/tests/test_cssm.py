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
    out = _chunked_selective_scan(x, delta, A, B, C, chunk=7)
    assert torch.allclose(out, _naive(x, delta, A, B, C), atol=1e-5)


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
