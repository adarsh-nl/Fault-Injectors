"""CSSM/SSD correctness + dynamics verification (read-only wrt production).

Phases (select with --phase, comma-separated, or 'smoke' / 'all'):

  smoke   Phase 0  minimum-size forward+backward through parallel AND
                   sequential paths, plus the _directions inspection.
  fwd     Phase 4  forward equivalence sweep, first-divergence tracing.
  bwd     Phase 5  backward equivalence (params + inputs), checkpoint
                   on/off, fp64 gradcheck, full-CSSM-level comparison.
  ref     Phase 6  third implementation: chunked state-passing SSD form
                   (re-derived from the Mamba-2 minimal listing, NOT an
                   import); mamba-ssm availability probe (no install).
  perm    Phase 8  permutation/causal-structure tests, including the
                   'col' scan-order semantics on h != w grids.
  dyn     Phase 9  long-horizon dynamics of the recurrence at measured
                   delta scales (delta rides the 0.2 bound in training).

Everything prints with flush=True. Run with python -u. No file output.
py3.7 / torch 1.12 compatible: no walrus, no math.prod.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nanjaiyalathaa/Fault-Injectors")

from corabench.fusion.cssm import CSSM, SelectiveScan  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731


# ────────────────────────────────────────────────────────────────────────
# Phase 3 — naive sequential reference. Reproduces the CURRENT CODE's math
# exactly (not the math we might wish it had):
#   dt   = dt_bound * tanh(softplus(dt_raw + dt_bias) / dt_bound)
#   A    = -exp(a_log)                       (D, N)
#   h_t  = exp(dt_t (x) A) * h_{t-1} + (dt_t * x_t) (x) B_t
#   y_t  = sum_n C_{t,n} h_{t,d,n}  ;  y += d_skip * x
# No chunking, no pairwise matrix, no mask, no einsum tricks: each step is
# individually inspectable.
# ────────────────────────────────────────────────────────────────────────
def sequential_scan(scan, x, dt_raw, b_in, c_in, return_states=False):
    dt_pre = F.softplus(dt_raw + scan.dt_bias.to(x.dtype))
    if scan.dt_bound:
        dt = scan.dt_bound * torch.tanh(dt_pre / scan.dt_bound)
    else:
        dt = dt_pre
    A = -torch.exp(scan.a_log.to(x.dtype))                   # (D, N)
    bsz, L, d = x.shape
    h = x.new_zeros(bsz, d, scan.d_state)
    ys, hs = [], []
    for t in range(L):
        A_bar = torch.exp(dt[:, t].unsqueeze(-1) * A)        # (B, D, N)
        B_bar_x = (dt[:, t] * x[:, t]).unsqueeze(-1) \
            * b_in[:, t].unsqueeze(1)                        # (B, D, N)
        h = A_bar * h + B_bar_x
        ys.append((h * c_in[:, t].unsqueeze(1)).sum(-1))     # (B, D)
        if return_states:
            hs.append(h)
    y = torch.stack(ys, dim=1) + scan.d_skip.to(x.dtype) * x
    if return_states:
        return y, torch.stack(hs, dim=1)                     # (B, L, D, N)
    return y


# ────────────────────────────────────────────────────────────────────────
# Phase 6 — third implementation, chunked state-passing SSD form.
# Re-derived from the Mamba-2 'ssd_minimal_discrete' listing (segsum +
# within-chunk diag block + chunkwise state recurrence), mapped onto THIS
# code's parameterization. Independent algorithmic path from both the
# pairwise-masked-exp parallel form and the per-step loop.
# ────────────────────────────────────────────────────────────────────────
def ssd_ref(scan, x, dt_raw, b_in, c_in, block=32):
    dt_pre = F.softplus(dt_raw + scan.dt_bias.to(x.dtype))
    if scan.dt_bound:
        dt = scan.dt_bound * torch.tanh(dt_pre / scan.dt_bound)
    else:
        dt = dt_pre
    A = -torch.exp(scan.a_log.to(x.dtype))                   # (D, N)
    dA = dt.unsqueeze(-1) * A                                # (B, L, D, N)
    b = (dt * x).unsqueeze(-1) * b_in.unsqueeze(2)           # (B, L, D, N)
    bsz, L, d = x.shape
    N = scan.d_state
    h = x.new_zeros(bsz, d, N)
    ys = []
    for s in range(0, L, block):
        dA_c, b_c = dA[:, s:s + block], b[:, s:s + block]
        c_c = c_in[:, s:s + block]
        Lc = dA_c.shape[1]
        seg = torch.cumsum(dA_c, dim=1)                      # (B, Lc, D, N)
        # within-chunk causal kernel via segsum difference, then exp
        diff = seg.unsqueeze(2) - seg.unsqueeze(1)           # (B, t, s, D, N)
        mask = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool,
                                     device=x.device))
        K = torch.where(mask.view(1, Lc, Lc, 1, 1), torch.exp(diff),
                        torch.zeros((), dtype=x.dtype, device=x.device))
        h_intra = torch.einsum("btsdn,bsdn->btdn", K, b_c)
        h_all = torch.exp(seg) * h.unsqueeze(1) + h_intra
        h = h_all[:, -1]
        ys.append(torch.einsum("bldn,bln->bld", h_all, c_c))
    return torch.cat(ys, dim=1) + scan.d_skip.to(x.dtype) * x


# ────────────────────────────────────────────────────────────────────────
# comparison helpers
# ────────────────────────────────────────────────────────────────────────
def cmp(name, a, b, eps=1e-12):
    if a.shape != b.shape:
        P("  %-28s SHAPE MISMATCH %s vs %s" % (name, tuple(a.shape),
                                               tuple(b.shape)))
        return float("inf")
    d = (a - b).abs()
    mx = float(d.max())
    ref = float(b.abs().max())
    rel = mx / (ref + eps)
    relel = float((d / (b.abs() + 1e-6)).max())
    loc = (d == d.max()).nonzero()[0].tolist() if mx > 0 else []
    fin = "finite" if bool(torch.isfinite(a).all()
                           and torch.isfinite(b).all()) else "NON-FINITE"
    P("  %-28s max|d|=%.3e mean|d|=%.3e rel=%.3e maxrel_el=%.3e "
      "ref_max=%.3e %s loc=%s"
      % (name, mx, float(d.mean()), rel, relel, ref, fin, loc))
    return rel


def gcmp(name, ga, gb):
    if ga is None or gb is None:
        P("  %-28s grad MISSING (a=%s b=%s)" % (name, ga is not None,
                                                gb is not None))
        return
    d = (ga - gb).abs()
    na, nb = float(ga.norm()), float(gb.norm())
    cos = float((ga.flatten() @ gb.flatten())
                / (ga.norm() * gb.norm() + 1e-30))
    fin = "finite" if bool(torch.isfinite(ga).all()
                           and torch.isfinite(gb).all()) else "NON-FINITE"
    P("  %-28s max|d|=%.3e mean|d|=%.3e rel=%.3e cos=%.10f "
      "|ref|=%.3e |par|=%.3e %s"
      % (name, float(d.max()), float(d.mean()),
         float(d.max()) / (nb + 1e-30), cos, nb, na, fin))


def mkscan(D, N, chunk, dt_bound=0.2, ckpt=True, dtype=torch.float32,
           seed=0):
    torch.manual_seed(seed)
    s = SelectiveScan(D, N, chunk=chunk, dt_bound=dt_bound,
                      fp32_island=False, checkpoint_chunks=ckpt)
    return s.to(dtype)


def mkinputs(B, L, D, N, dtype=torch.float32, scale=1.0, seed=1, grad=False):
    torch.manual_seed(seed)
    t = lambda *sh: (torch.randn(*sh, dtype=dtype)  # noqa: E731
                     * scale).requires_grad_(grad)
    return t(B, L, D), t(B, L, D), t(B, L, N), t(B, L, N)


# ────────────────────────────────────────────────────────────────────────
def phase_smoke():
    P("=" * 72)
    P("PHASE 0 -- SMOKE (B=1, L=8, D=4, N=2, chunk=4 -> 2 chunks, CPU)")
    P("=" * 72)
    scan = mkscan(4, 2, chunk=4)
    x, dtr, b, c = mkinputs(1, 8, 4, 2, grad=True)
    x2, dtr2, b2, c2 = (t.detach().clone().requires_grad_(True)
                        for t in (x, dtr, b, c))
    yp = scan(x, dtr, b, c)
    ys = sequential_scan(scan, x2, dtr2, b2, c2)
    P("forward:")
    cmp("parallel_vs_sequential", yp, ys)
    lp = yp.square().mean()
    ls = ys.square().mean()
    for p_ in scan.parameters():
        p_.grad = None
    lp.backward()
    gp = {n: p_.grad.clone() for n, p_ in scan.named_parameters()}
    gxp = [t.grad.clone() for t in (x, dtr, b, c)]
    for p_ in scan.parameters():
        p_.grad = None
    ls.backward()
    P("backward:")
    for n, p_ in scan.named_parameters():
        gcmp("param." + n, gp[n], p_.grad)
    for nm, ga, t in zip(("x", "dt_raw", "b_in", "c_in"), gxp,
                         (x2, dtr2, b2, c2)):
        gcmp("input." + nm, ga, t.grad)
    P("\nCSSM wrapper smoke (B=1, C=4, H=4, W=6, pool=2 -> h=2, w=3):")
    torch.manual_seed(3)
    m = CSSM(4, d_state=2, chunk=4, pool=2)
    zf = torch.randn(1, 4, 4, 6, requires_grad=True)
    zi = torch.randn(1, 4, 4, 6)
    out = m(zf, zi)
    out.square().mean().backward()
    P("  forward %s backward ok, out finite: %s, zf.grad finite: %s"
      % (tuple(out.shape), bool(torch.isfinite(out).all()),
         bool(torch.isfinite(zf.grad).all())))
    P("\nfp32_island=True path (CPU: no_autocast is a nullcontext, but the")
    P("enter/exit and cast plumbing must run):")
    torch.manual_seed(0)
    s2 = SelectiveScan(4, 2, chunk=4, dt_bound=0.2, fp32_island=True)
    x3, dtr3, b3, c3 = mkinputs(1, 8, 4, 2, grad=True, seed=1)
    y3 = s2(x3, dtr3, b3, c3)
    y3.square().mean().backward()
    P("  forward+backward ok, finite: %s"
      % bool(torch.isfinite(y3).all() and torch.isfinite(x3.grad).all()))

    P("\n_directions inspection (h=2, w=3):")
    dirs = CSSM._directions(torch.zeros(1, 6, 1), 2, 3)
    names = ["row", "row_r", "col", "col_r"]
    for nm, pm in zip(names, dirs):
        P("  %-6s %s" % (nm, pm.tolist()))
    true_col = [(k % 2) * 3 + k // 2 for k in range(6)]
    P("  true column-major gather perm (visit (r,c) col-by-col): %s"
      % true_col)
    P("  code col == true col-major: %s"
      % (dirs[2].tolist() == true_col))
    P("SMOKE DONE")


# ────────────────────────────────────────────────────────────────────────
def fwd_case(tag, B, L, D, N, chunk, scale=1.0, dt_shift=0.0,
             dtype=torch.float32, seed=1, trace=False):
    scan = mkscan(D, N, chunk, dtype=dtype)
    x, dtr, b, c = mkinputs(B, L, D, N, dtype=dtype, scale=scale, seed=seed)
    dtr = dtr + dt_shift
    with torch.no_grad():
        yp = scan(x, dtr, b, c)
        ys = sequential_scan(scan, x, dtr, b, c)
    P("case %-34s dt in [%.2e, %.2e]" % (
        tag, float(scan.last_delta_stats["delta_mean"]),
        float(scan.last_delta_stats["delta_max"])))
    rel = cmp("  y", yp, ys)
    if trace and rel > 1e-4:
        P("  TRACING first divergence:")
        trace_first_divergence(scan, x, dtr, b, c)
    return rel


def analytic_case(dtype):
    """Hand-crafted closed form, independent of BOTH implementations.

    x = 1, b_in = 1, c_in = 1, dt_raw = 0 -> dt is a per-channel constant
    dt_d = bound * tanh(softplus(dt_bias_d) / bound). Then
      h_t[d, n] = dt_d * (1 - r^t...) geometric sum with r = exp(dt_d A_dn):
      h_t = dt_d * (1 - r^(t+1)) / (1 - r) - dt_d ... careful: h_0 = dt
      (first step injects, no decay of anything prior), so
      h_t = dt_d * sum_{j=0..t} r^j - dt_d * r^t * 0 = dt * (1-r^(t+1))/(1-r)
      WRONG by one: h_t = A_bar h_{t-1} + dt -> h_t = dt * sum_{j=0..t} r^j?
      h_0 = dt; h_1 = r*dt + dt = dt(1+r); h_t = dt * sum_{j=0..t} r^j. Yes.
      y_t = sum_n h_t[d, n] + d_skip_d.
    """
    torch.manual_seed(2)
    scan = mkscan(4, 3, 4, dtype=dtype)
    B, L, D, N = 1, 12, 4, 3
    one = torch.ones(B, L, D, dtype=dtype)
    x, dtr = one, torch.zeros(B, L, D, dtype=dtype)
    b = torch.ones(B, L, N, dtype=dtype)
    c = torch.ones(B, L, N, dtype=dtype)
    with torch.no_grad():
        yp = scan(x, dtr, b, c)
        dt = scan.dt_bound * torch.tanh(
            F.softplus(scan.dt_bias) / scan.dt_bound)        # (D,)
        r = torch.exp(dt.unsqueeze(-1) * -torch.exp(scan.a_log))  # (D, N)
        t_idx = torch.arange(1, L + 1, dtype=dtype).view(L, 1, 1)
        h = dt.unsqueeze(-1) * (1 - r ** t_idx) / (1 - r)    # (L, D, N)
        y_closed = h.sum(-1).unsqueeze(0) + scan.d_skip * x
    P("case %-34s (closed-form geometric sum)" % "analytic x=1,b=1,c=1")
    cmp("  y vs closed form", yp, y_closed)


def trace_first_divergence(scan, x, dtr, b_in, c_in):
    """Recompute both paths intermediate-by-intermediate, in declared
    dependency order, and report the first tensor whose rel error > 1e-5."""
    with torch.no_grad():
        dt_pre = F.softplus(dtr + scan.dt_bias)
        dt = scan.dt_bound * torch.tanh(dt_pre / scan.dt_bound) \
            if scan.dt_bound else dt_pre
        A = -torch.exp(scan.a_log)
        dA = dt.unsqueeze(-1) * A
        b = (dt * x).unsqueeze(-1) * b_in.unsqueeze(2)
        # parallel h per chunk vs sequential h
        _, hs = sequential_scan(scan, x, dtr, b_in, c_in, return_states=True)
        h0 = x.new_zeros(x.shape[0], x.shape[2], scan.d_state)
        offs = 0
        for s in range(0, x.shape[1], scan.chunk):
            dA_c, b_c = dA[:, s:s + scan.chunk], b[:, s:s + scan.chunk]
            logE_c = torch.cumsum(dA_c, dim=1)
            h_all, h0 = scan._chunk_scan(logE_c, b_c, h0)
            r = cmp("h chunk@%d" % offs, h_all, hs[:, s:s + scan.chunk])
            if r > 1e-5:
                P("  FIRST DIVERGENCE inside chunk starting at t=%d" % offs)
                return
            offs += scan.chunk
        P("  no state divergence above 1e-5; divergence is in the C-readout"
          " or D-skip stage")


def phase_fwd():
    P("=" * 72)
    P("PHASE 4 -- FORWARD EQUIVALENCE (parallel vs sequential)")
    P("=" * 72)
    for dtype in (torch.float32, torch.float64):
        P("--- dtype=%s ---" % dtype)
        fwd_case("tiny random B1 L8 D4 N2 ck4", 1, 8, 4, 2, 4,
                 dtype=dtype, trace=True)
        analytic_case(dtype)
        fwd_case("near-zero delta (dt_raw-12)", 1, 64, 8, 4, 16,
                 dt_shift=-12.0, dtype=dtype, trace=True)
        fwd_case("delta at bound (dt_raw+12)", 1, 64, 8, 4, 16,
                 dt_shift=12.0, dtype=dtype, trace=True)
        fwd_case("large inputs x100", 1, 64, 8, 4, 16, scale=100.0,
                 dtype=dtype, trace=True)
        fwd_case("long seq L=1024", 1, 1024, 8, 4, 64, dtype=dtype)
        fwd_case("chunk not divides L (L=100)", 1, 100, 8, 4, 64,
                 dtype=dtype, trace=True)
    P("--- realistic CSSM shape, fp32 (B2 L8800 D256 N16 chunk64) ---")
    fwd_case("realistic", 2, 8800, 256, 16, 64, trace=True)
    P("PHASE 4 DONE")


# ────────────────────────────────────────────────────────────────────────
def bwd_pair(tag, B, L, D, N, chunk, scale=1.0, dt_shift=0.0,
             dtype=torch.float32, ckpt=True, seed=1):
    P("--- %s (ckpt=%s, %s) ---" % (tag, ckpt, dtype))
    scan = mkscan(D, N, chunk, ckpt=ckpt, dtype=dtype)
    xs = mkinputs(B, L, D, N, dtype=dtype, scale=scale, seed=seed, grad=True)
    x, dtr, b, c = xs
    dtr = (dtr + dt_shift).detach().requires_grad_(True)
    x2, dtr2, b2, c2 = (t.detach().clone().requires_grad_(True)
                        for t in (x, dtr, b, c))
    yp = scan(x, dtr, b, c)
    for p_ in scan.parameters():
        p_.grad = None
    yp.square().mean().backward()
    gpar = {n: p_.grad.clone() for n, p_ in scan.named_parameters()}
    gin = {nm: t.grad.clone() for nm, t in
           zip(("x", "dt_raw", "b_in", "c_in"), (x, dtr, b, c))}
    for p_ in scan.parameters():
        p_.grad = None
    ys = sequential_scan(scan, x2, dtr2, b2, c2)
    cmp("forward y", yp, ys)
    ys.square().mean().backward()
    for n, p_ in scan.named_parameters():
        gcmp("param." + n, gpar[n], p_.grad)
    for nm, t in zip(("x", "dt_raw", "b_in", "c_in"), (x2, dtr2, b2, c2)):
        gcmp("input." + nm, gin[nm], t.grad)


def phase_bwd():
    P("=" * 72)
    P("PHASE 5 -- BACKWARD EQUIVALENCE (highest priority)")
    P("=" * 72)
    bwd_pair("moderate B2 L128 D16 N8", 2, 128, 16, 8, 32)
    bwd_pair("moderate, checkpoint OFF", 2, 128, 16, 8, 32, ckpt=False)
    bwd_pair("delta at bound", 1, 64, 8, 4, 16, dt_shift=12.0)
    bwd_pair("large inputs x100", 1, 64, 8, 4, 16, scale=100.0)
    bwd_pair("fp64 moderate", 2, 128, 16, 8, 32, dtype=torch.float64)
    bwd_pair("realistic-shape slice B1 L1024 D256 N16",
             1, 1024, 256, 16, 64)

    P("--- checkpoint on vs off: same forward graph, grads must be"
      " bit-comparable ---")
    for dtype in (torch.float32, torch.float64):
        scan_a = mkscan(16, 8, 32, ckpt=True, dtype=dtype, seed=0)
        scan_b = mkscan(16, 8, 32, ckpt=False, dtype=dtype, seed=0)
        scan_b.load_state_dict(scan_a.state_dict())
        x, dtr, b, c = mkinputs(2, 128, 16, 8, dtype=dtype, grad=True)
        x2, dtr2, b2, c2 = (t.detach().clone().requires_grad_(True)
                            for t in (x, dtr, b, c))
        scan_a(x, dtr, b, c).square().mean().backward()
        scan_b(x2, dtr2, b2, c2).square().mean().backward()
        P("  [%s]" % dtype)
        for (n, pa), (_, pb) in zip(scan_a.named_parameters(),
                                    scan_b.named_parameters()):
            gcmp("ckpt_on_vs_off." + n, pa.grad, pb.grad)
        gcmp("ckpt_on_vs_off.x", x.grad, x2.grad)

    P("--- fp64 gradcheck (analytic vs numerical Jacobian), tiny ---")
    for ckpt in (False, True):
        scan = mkscan(3, 2, 4, ckpt=ckpt, dtype=torch.float64)
        x, dtr, b, c = mkinputs(1, 8, 3, 2, dtype=torch.float64, grad=True)
        inputs = (x, dtr, b, c)
        fn = lambda *i: scan(*i)  # noqa: E731
        try:
            ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-8,
                                          rtol=1e-6)
            P("  gradcheck(inputs, ckpt=%s): %s" % (ckpt, ok))
        except Exception as e:  # noqa: BLE001
            P("  gradcheck(inputs, ckpt=%s) FAILED: %s"
              % (ckpt, str(e)[:400]))
    scan = mkscan(3, 2, 4, ckpt=False, dtype=torch.float64)
    x, dtr, b, c = mkinputs(1, 8, 3, 2, dtype=torch.float64)
    prm = list(scan.parameters())
    fn = lambda al, db, ds: SelectiveScanFunctional(  # noqa: E731
        scan, al, db, ds, x, dtr, b, c)
    try:
        ok = torch.autograd.gradcheck(
            fn, tuple(p_.detach().clone().requires_grad_(True) for p_ in prm),
            eps=1e-6, atol=1e-8, rtol=1e-6)
        P("  gradcheck(parameters): %s" % ok)
    except Exception as e:  # noqa: BLE001
        P("  gradcheck(parameters) FAILED: %s" % str(e)[:400])

    P("--- full-CSSM-level backward: parallel scan vs sequential scan ---")
    cssm_level_bwd()
    P("PHASE 5 DONE")


def SelectiveScanFunctional(scan, a_log, dt_bias, d_skip, x, dtr, b, c):
    """Pure-functional rebuild of SelectiveScan.forward for param gradcheck
    (same math, params passed explicitly)."""
    dt_pre = F.softplus(dtr + dt_bias)
    dt = scan.dt_bound * torch.tanh(dt_pre / scan.dt_bound)
    A = -torch.exp(a_log)
    dA = dt.unsqueeze(-1) * A
    bb = (dt * x).unsqueeze(-1) * b.unsqueeze(2)
    h0 = x.new_zeros(x.shape[0], x.shape[2], scan.d_state)
    ys = []
    for s in range(0, x.shape[1], scan.chunk):
        logE_c = torch.cumsum(dA[:, s:s + scan.chunk], dim=1)
        b_c = bb[:, s:s + scan.chunk]
        Lc = logE_c.shape[1]
        pair = logE_c.unsqueeze(2) - logE_c.unsqueeze(1)
        tril = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool))
        pair = pair.masked_fill(~tril.view(1, Lc, Lc, 1, 1), float("-inf"))
        h_all = torch.exp(logE_c) * h0.unsqueeze(1) \
            + torch.einsum("btsdn,bsdn->btdn", torch.exp(pair), b_c)
        h0 = h_all[:, -1]
        ys.append(torch.einsum("bldn,bln->bld", h_all,
                               c[:, s:s + scan.chunk]))
    return torch.cat(ys, dim=1) + d_skip * x


class _SeqScanShim(torch.nn.Module):
    """Duck-typed stand-in for CSSM.scan that runs the sequential loop with
    the parallel module's parameters."""

    def __init__(self, scan):
        super().__init__()
        self.scan = scan
        self.last_delta_stats = None

    def forward(self, x, dtr, b, c):
        return sequential_scan(self.scan, x, dtr, b, c)


def cssm_level_bwd():
    torch.manual_seed(7)
    m1 = CSSM(8, d_state=4, chunk=8, pool=2)
    m2 = CSSM(8, d_state=4, chunk=8, pool=2)
    m2.load_state_dict(m1.state_dict())
    m2.scan = _SeqScanShim(m2.scan)
    zf = torch.randn(2, 8, 8, 12, requires_grad=True)
    zi = torch.randn(2, 8, 8, 12, requires_grad=True)
    zf2 = zf.detach().clone().requires_grad_(True)
    zi2 = zi.detach().clone().requires_grad_(True)
    y1 = m1(zf, zi)
    y2 = m2(zf2, zi2)
    cmp("CSSM forward", y1, y2)
    y1.square().mean().backward()
    y2.square().mean().backward()
    n2 = dict(m2.named_parameters())
    for n, p_ in m1.named_parameters():
        key = n.replace("scan.", "scan.scan.") if n.startswith("scan.") else n
        gcmp("CSSM." + n, p_.grad, n2[key].grad)
    gcmp("CSSM.z_fused", zf.grad, zf2.grad)
    gcmp("CSSM.z_i", zi.grad, zi2.grad)


# ────────────────────────────────────────────────────────────────────────
def phase_ref():
    P("=" * 72)
    P("PHASE 6 -- REFERENCE IMPLEMENTATION")
    P("=" * 72)
    try:
        import mamba_ssm  # noqa: F401
        P("mamba_ssm IS importable: %s" % mamba_ssm.__file__)
    except Exception as e:  # noqa: BLE001
        P("mamba_ssm unavailable in this environment (%s: %s)"
          % (type(e).__name__, str(e)[:200]))
        P("-> Reference implementation unavailable in the existing "
          "environment; sequential recurrence remains the independent "
          "correctness reference.")
    P("Third implementation (re-derived chunked SSD, block=32 != chunk=64):")
    for dtype in (torch.float32, torch.float64):
        scan = mkscan(16, 8, 64, dtype=dtype)
        x, dtr, b, c = mkinputs(2, 200, 16, 8, dtype=dtype)
        with torch.no_grad():
            yp = scan(x, dtr, b, c)
            ys = sequential_scan(scan, x, dtr, b, c)
            yr = ssd_ref(scan, x, dtr, b, c, block=32)
        P("  [%s]" % dtype)
        cmp("parallel vs ssd_ref", yp, yr)
        cmp("sequential vs ssd_ref", ys, yr)
    P("PHASE 6 DONE")


# ────────────────────────────────────────────────────────────────────────
def phase_perm():
    P("=" * 72)
    P("PHASE 8 -- PERMUTATION / CAUSAL STRUCTURE")
    P("=" * 72)
    P("A. What the four _directions perms actually traverse (h=50, w=176,")
    P("   the real pooled BEV shape). Spatial step = |dr|+|dc| between")
    P("   consecutively scanned cells (1 = raster-adjacent).")
    h, w = 50, 176
    dirs = CSSM._directions(torch.zeros(1, h * w, 1), h, w)
    names = ["row", "row_r", "col", "col_r"]
    true_col = torch.tensor([(k % h) * w + k // h for k in range(h * w)])
    for nm, pm in zip(names, dirs):
        r, c = pm // w, pm % w
        step = (r[1:] - r[:-1]).abs() + (c[1:] - c[:-1]).abs()
        P("  %-6s mean_step=%7.2f max_step=%4d frac_step==1: %.4f"
          % (nm, float(step.float().mean()), int(step.max()),
             float((step == 1).float().mean())))
    P("  col perm == TRUE column-major gather perm: %s"
      % bool((dirs[2] == true_col).all()))
    tc_r, tc_c = true_col // w, true_col % w
    st = (tc_r[1:] - tc_r[:-1]).abs() + (tc_c[1:] - tc_c[:-1]).abs()
    P("  (true column-major would give mean_step=%.2f, frac_step==1: %.4f)"
      % (float(st.float().mean()), float((st == 1).float().mean())))
    P("  square grid h=w=8: col == true col-major: %s"
      % bool((CSSM._directions(torch.zeros(1, 64, 1), 8, 8)[2]
              == torch.tensor([(k % 8) * 8 + k // 8
                               for k in range(64)])).all()))

    P("\nB. Round-trip: inverse perm restores order (per direction)")
    torch.manual_seed(11)
    m = CSSM(8, d_state=4, chunk=16, pool=2)
    for nm, pm in zip(names, CSSM._directions(torch.zeros(1, 24, 1), 4, 6)):
        inv = torch.empty_like(pm)
        inv[pm] = torch.arange(pm.numel())
        v = torch.arange(24)
        P("  %-6s perm-then-inv identity: %s"
          % (nm, bool((v[pm][inv] == v).all())))

    P("\nC. Single-direction scan is causal and order-sensitive (expected):")
    scan = mkscan(8, 4, 16)
    x, dtr, b, c = mkinputs(1, 64, 8, 4)
    with torch.no_grad():
        y = sequential_scan(scan, x, dtr, b, c)
        xr = x.flip(1)
        yr = sequential_scan(scan, xr, dtr.flip(1), b.flip(1), c.flip(1))
        P("  ||scan(x) - flip(scan(flip(x)))|| rel = %.3e  (0 would mean "
          "acausal)" % float((y - yr.flip(1)).norm() / y.norm()))
        x2 = x.clone()
        x2[:, 32] += 10.0
        y2 = sequential_scan(scan, x2, dtr, b, c)
        d = (y2 - y).abs().sum(-1)[0]
        P("  perturb t=32: max |dy| before t=32: %.3e, at/after: %.3e "
          "(strict causality)" % (float(d[:32].max()), float(d[32:].max())))

    P("\nD. Full-CSSM transpose equivariance (H<->W). If the 4-direction")
    P("   set were the intended symmetric cross-scan, CSSM(transposed in)")
    P("   would equal transpose(CSSM(in)) up to the pooling/upsample. ")
    torch.manual_seed(13)
    m = CSSM(8, d_state=4, chunk=16, pool=2)
    zf = torch.randn(1, 8, 12, 20)
    zi = torch.randn(1, 8, 12, 20)
    with torch.no_grad():
        a = m(zf, zi)
        bT = m(zf.transpose(-1, -2).contiguous(),
               zi.transpose(-1, -2).contiguous()).transpose(-1, -2)
    P("  rel diff = %.3e" % float((a - bT).norm() / a.norm()))

    P("\nE. Agent axis: CSSM does NOT scan agents. The agent axis is")
    P("   reduced upstream by AttFusion2 over the 2-token {ego, coll}")
    P("   stack; the scan axis is pooled BEV cells. (Code-inspection")
    P("   fact, printed here for the record.)")
    P("PHASE 8 DONE")


# ────────────────────────────────────────────────────────────────────────
def phase_dyn():
    P("=" * 72)
    P("PHASE 9 -- LONG-HORIZON DYNAMICS (sequential recurrence, fp32)")
    P("=" * 72)
    P("Measured training context: delta rides the tanh bound (p99 -> 0.197")
    P("of bound 0.2, sat_frac 0.13 by step 2k, job 565550). So the relevant")
    P("regime is dt ~= 0.2 on saturated channels, dt in init range else.")
    B, L, D, N = 1, 8800, 256, 16
    for scale in (1.0, 10.0, 100.0):
        for dt_shift, dt_tag in ((0.0, "init-range dt"),
                                 (12.0, "dt at bound 0.2")):
            scan = mkscan(D, N, 64)
            x, dtr, b, c = mkinputs(B, L, D, N, scale=scale, seed=5)
            dtr = dtr + dt_shift
            stats = dyn_run(scan, x, dtr, b, c)
            P("scale=%-6.0f %-16s %s" % (scale, dt_tag, stats))
    P("\nGain law: output/input scaling exponent (cubic path check).")
    P("y(a*x) ~ a^k. Measure k between successive decades, no D-skip:")
    scan = mkscan(D, N, 64)
    x, dtr, b, c = mkinputs(B, 512, D, N, seed=6)
    prev = None
    for a in (0.1, 1.0, 10.0, 100.0):
        with torch.no_grad():
            # scale the SOURCE activation: x, dt_raw, b_in, c_in all derive
            # from z_fused/z_i in the real model, so all scale together.
            y = sequential_scan(scan, a * x, a * dtr, a * b, a * c) \
                - scan.d_skip * (a * x)
        n = float(y.norm())
        if prev is not None:
            P("  a=%-6g ||y||=%.3e   k=log10(ratio)=%.2f"
              % (a, n, torch.log10(torch.tensor(n / prev)).item()))
        else:
            P("  a=%-6g ||y||=%.3e" % (a, n))
        prev = n
    P("  (dt saturates via softplus/tanh at large a, so k < 3 there;")
    P("   the un-saturated small-a regime shows the raw polynomial order.)")
    P("PHASE 9 DONE")


def dyn_run(scan, x, dtr, b_in, c_in):
    with torch.no_grad():
        dt_pre = F.softplus(dtr + scan.dt_bias)
        dt = scan.dt_bound * torch.tanh(dt_pre / scan.dt_bound)
        A = -torch.exp(scan.a_log)
        h = x.new_zeros(x.shape[0], x.shape[2], scan.d_state)
        hn, rn, an, bn, yn = [], [], [], [], []
        for t in range(x.shape[1]):
            A_bar = torch.exp(dt[:, t].unsqueeze(-1) * A)
            Ah = A_bar * h
            Bx = (dt[:, t] * x[:, t]).unsqueeze(-1) * b_in[:, t].unsqueeze(1)
            h = Ah + Bx
            y = (h * c_in[:, t].unsqueeze(1)).sum(-1)
            hn.append(float(h.norm()))
            an.append(float(Ah.norm()))
            bn.append(float(Bx.norm()))
            rn.append(float(Bx.norm() / (Ah.norm() + 1e-9)))
            yn.append(float(y.norm()))
        hn_t = torch.tensor(hn)
        half = hn_t[len(hn) // 2:]
        slope = float((torch.log(half[-1] + 1e-9)
                       - torch.log(half[0] + 1e-9)) / (len(half)))
        return ("||h||: t0=%.2e mid=%.2e end=%.2e max=%.2e | "
                "R=inject/recur: mean=%.2f p99=%.2f | ||Ch||max=%.2e | "
                "log-slope(2nd half)=%+.1e"
                % (hn[0], hn[len(hn) // 2], hn[-1], max(hn),
                   sum(rn) / len(rn),
                   sorted(rn)[int(0.99 * len(rn))], max(yn), slope))


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke",
                    help="comma list: smoke,fwd,bwd,ref,perm,dyn or all")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    P("torch %s, default threads %d" % (torch.__version__,
                                        torch.get_num_threads()))
    phases = (["smoke", "fwd", "bwd", "ref", "perm", "dyn"]
              if args.phase == "all" else args.phase.split(","))
    fns = {"smoke": phase_smoke, "fwd": phase_fwd, "bwd": phase_bwd,
           "ref": phase_ref, "perm": phase_perm, "dyn": phase_dyn}
    for ph in phases:
        fns[ph]()
    P("ALL REQUESTED PHASES DONE")
