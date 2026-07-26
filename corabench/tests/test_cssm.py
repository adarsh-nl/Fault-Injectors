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


def test_dt_max_is_a_soft_bound_with_a_live_gradient_where_it_matters():
    """The delta ceiling must not repeat the clamp trap -- and its limit.

    ``clamp(max=c)`` has EXACTLY zero gradient above ``c``, so an element whose
    pre-activation drifts past the bound is stuck with no restoring force:
    forward-safe, backward-dead. That shape has already cost three defects --
    the fp16 focal clamp (RECON-3), the asin hard clamp (96fbb2a), and it
    would have been a third here.

    ``dt_max * tanh(delta / dt_max)`` bounds just as hard and keeps a live
    gradient through the APPROACH to the ceiling, which is where a restoring
    force is actually needed. It is NOT immune: ``sech^2`` underflows to zero
    in float32 past raw delta ~2, so far above the bound it is backward-dead
    too. The bound's job is to stop delta ever getting there, not to recover
    from it -- b = (delta * x) (x) B scales linearly in delta, so capping
    delta caps the runaway that drove it up.
    """
    dt_max = 0.2
    raw = torch.tensor([0.001, 0.011, 0.1, 0.318, 1.0, 4.85],
                       requires_grad=True)
    bounded = dt_max * torch.tanh(raw / dt_max)
    bounded.sum().backward()

    assert float(bounded.max()) <= dt_max * (1 + 1e-6), "must actually bound"
    # In-range fidelity: dt_init samples [0.001, 0.1] and the bound must
    # barely touch it, or it is altering the initialisation it protects.
    for i, target in ((0, 0.001), (1, 0.011)):
        assert abs(float(bounded[i]) - target) / target < 0.01
    assert abs(float(bounded[2]) - 0.1) / 0.1 < 0.08      # <8% at dt_init's top

    # THE POINT: a live gradient through the approach, where clamp is already
    # dead. raw=0.318 is the delta actually observed at step 0 of job 549412.
    hard = torch.tensor([0.318], requires_grad=True)
    hard.clamp(max=dt_max).sum().backward()
    assert float(hard.grad) == 0.0, "clamp is expected to be backward-dead"
    assert float(raw.grad[3]) > 0.1, (
        "soft bound went backward-dead at the observed delta; there would be "
        "no restoring force exactly where one is needed")
    # Honest limit, pinned so nobody reads the bound as unconditionally safe.
    assert float(raw.grad[5]) == 0.0, (
        "sech^2 no longer underflows far above the bound -- the docstring's "
        "stated limitation is stale and should be updated")


def _float64_reference(x, delta, A, B, C):
    """Sequential recurrence in float64: h_t = exp(dA_t) h_{t-1} + b_t.

    No chunking, no closed form, no division -- the definition itself, at a
    precision where fp32 error cannot hide. This is the arbiter.
    """
    x, delta, A, B, C = (t.double() for t in (x, delta, A, B, C))
    bt, length, d = x.shape
    h = torch.zeros(bt, d, A.shape[1], dtype=torch.float64)
    ys = []
    for t in range(length):
        dA = delta[:, t].unsqueeze(-1) * A
        b = (delta[:, t] * x[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
        h = torch.exp(dA) * h + b
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, 1)


def test_scan_is_correct_at_the_549449_regime():
    """The regime that broke job 549449, against a float64 arbiter.

    Delta pinned at 0.2 by dt_max, A = -[1..16], x at the z_fused scale
    actually observed (~22). |dA| reaches 3.2, so logE crosses -30 at position
    ~9 of every 64-chunk and the old b/E form ran mostly on its clamp.

    FINITENESS IS NOT THE DISCRIMINATOR -- this was measured. The old form's
    backward at this regime is finite (grad amax ~26) because the 1/E and E
    factors cancel along the graph. What it does instead is compute the WRONG
    ANSWER: 381.3 against a true 139.8, a relative error of 2.68. That 2.7x
    inflation is what drove ssm_out past the fp16 ceiling at the island exit.

    So the assertion is accuracy, not health. Written against the b/E form
    first and observed to fail at rel err 2.68 before the rewrite landed.
    """
    torch.manual_seed(0)
    bt, length, d, n = 1, 192, 8, 16          # 3 chunks of 64
    x = torch.randn(bt, length, d) * 22.0     # observed z_fused scale
    delta = torch.full((bt, length, d), 0.2)  # pinned by dt_max
    A = -torch.arange(1, n + 1, dtype=torch.float32).repeat(d, 1)
    B, C = torch.randn(bt, length, n), torch.randn(bt, length, n)

    logE_min = float((delta.unsqueeze(-1) * A).min()) * 64
    assert logE_min < -30.0, "regime no longer crosses the -30 decay depth"

    reference = _float64_reference(x, delta, A, B, C)
    got, _ = _chunked_selective_scan(x, delta, A, B, C, chunk=64)

    rel = float((got.double() - reference).abs().max()
                / reference.abs().max())
    assert rel < 1e-5, (
        f"scan disagrees with the float64 reference by rel {rel:.3e}; the b/E "
        f"closed form scored 2.68 here (381.3 against a true 139.8)")


def test_scan_gradient_is_finite_and_bounded_at_the_549449_regime():
    """Complement to the accuracy test: the backward must also stay sane."""
    torch.manual_seed(0)
    bt, length, d, n = 1, 192, 8, 16
    x = torch.randn(bt, length, d, requires_grad=True) * 22.0
    x.retain_grad()
    delta = torch.full((bt, length, d), 0.2)
    A = -torch.arange(1, n + 1, dtype=torch.float32).repeat(d, 1)
    B, C = torch.randn(bt, length, n), torch.randn(bt, length, n)

    y, _ = _chunked_selective_scan(x, delta, A, B, C, chunk=64)
    y.abs().sum().backward()

    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    # The island casts back to fp16 on exit; a gradient that only survives in
    # fp32 would still become inf there.
    assert torch.isfinite(x.grad.half()).all(), \
        "gradient does not survive the float16 cast at the island boundary"


def test_inter_chunk_carry_error_does_not_accumulate():
    """The carry runs 138 times on the real grid; test it at that order.

    The reference-agreement test uses chunk=7 on L=41, so it does exercise
    the carry -- but only across 5 boundaries. The real OPV2V scan is 8800
    positions at chunk=64: 138 chunks, 137 boundaries. If per-boundary error
    compounded, a 3-chunk test would never show it.

    Measured flat: 3.3e-07 at 0 boundaries, 6.4e-07 at 63.
    """
    torch.manual_seed(0)
    bt, d, n = 1, 8, 16
    prev = None
    for length in (64, 1024, 4096):                  # 1, 16, 64 chunks
        x = torch.randn(bt, length, d) * 22.0
        delta = torch.full((bt, length, d), 0.2)
        A = -torch.arange(1, n + 1, dtype=torch.float32).repeat(d, 1)
        B, C = torch.randn(bt, length, n), torch.randn(bt, length, n)
        got, _ = _chunked_selective_scan(x, delta, A, B, C, chunk=64)
        reference = _float64_reference(x, delta, A, B, C)
        rel = float((got.double() - reference).abs().max()
                    / reference.abs().max())
        assert rel < 1e-5, f"L={length}: rel err {rel:.3e}"
        prev = rel if prev is None else prev
    assert rel < 10 * prev, (
        f"carry error grew from {prev:.3e} at 1 chunk to {rel:.3e} at 64 -- "
        f"it must stay flat, since the real grid runs 138")


def test_causal_mask_is_applied_in_log_space_before_the_exp():
    """For s > t the exponent is POSITIVE and can be huge.

    logE_t - logE_s > 0 above the diagonal, and it reached +1200 in a
    constructed check. `exp(ldiff) * mask` therefore overflows to inf and then
    inf * 0 = nan -- the overflow-then-mask trap, third cousin of the fp16
    focal clamp and the asin clamp. Masking in log space first makes exp yield
    an exact zero and never materialises anything large.
    """
    lc = 4
    logE = torch.tensor([[0., -400., -800., -1200.]]).reshape(1, 4, 1, 1)
    ldiff = logE.unsqueeze(2) - logE.unsqueeze(1)
    causal = torch.ones(lc, lc, dtype=torch.bool).tril()[None, :, :, None, None]

    assert float(ldiff.max()) > 700.0, "regime no longer overflows a naive exp"
    naive = torch.exp(ldiff) * causal
    assert not torch.isfinite(naive).all(), \
        "post-exp masking no longer overflows; this test has lost its point"

    correct = torch.exp(ldiff.masked_fill(~causal, float("-inf")))
    assert torch.isfinite(correct).all()
    assert bool((correct[0, :, :, 0, 0].triu(1) == 0).all())
