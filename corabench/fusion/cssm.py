"""
cssm.py
-------
Collaborative State Space Model (paper Eq. 8), Mamba-style selective scan.

Role assignment follows the paper exactly:
    input sequence x   = Z_fused
    step size Delta    = softplus(Linear(Z_fused))
    input matrix B     = Linear_B(Z_fused)
    output matrix C    = Linear_C(Z_i)        <- the ego feature reads out
so the ego decides WHAT to read from the jointly-written state.

Discretised diagonal recurrence (per channel d, state n):
    h_t = exp(Delta_t * A) . h_{t-1} + (Delta_t * B_t) * x_t
    y_t = <C_t, h_t> + D_skip * x_t

Backend 'reference' is pure PyTorch: exact chunked scan using the log-space
cumulative-sum closed form inside fixed-size chunks with a carried hidden
state between chunks (numerically bounded; no Python loop over timesteps).
Backend 'cuda' delegates to `mamba-ssm`'s fused kernel when installed.

PRECISION: the reference scan is run in float32 with autocast explicitly
disabled, even when the surrounding model is in AMP, and casts back to the
caller's dtype at the boundary so the rest of the model stays in AMP.

The scan uses the DIVIDE-FREE (SSD) chunked form: within a chunk it forms the
pairwise decay exp(logE_t - logE_s) directly, which is bounded by 1 for
s <= t, instead of the b/E closed form it replaces. That form divided by a
decay term clamped at -30, and the clamp broke the identity it guarded --
against a float64 reference it returned 381.3 where the truth is 139.8, a
relative error of 2.68. See _chunked_selective_scan and RECON-4.

2-D scanning: 'cross2d' (VMamba-style: 4 directions -- row-major, reversed,
column-major, reversed -- averaged; assumption A9) or 'raster' (single pass).

Memory note: the reference scan materialises (B, L, D, N) activations for
autograd. `pool` runs the scan at reduced spatial resolution (avg-pool in,
bilinear out) -- the practical setting for full OPV2V grids without the
fused kernel; set pool=1 to scan at full resolution.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from cpbench.observation.taps import TapProtocol, emit

logger = logging.getLogger(__name__)


def _scan_chunk(xc: torch.Tensor, dc: torch.Tensor, Bc: torch.Tensor,
                A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One chunk of the divide-free (SSD) recurrence. Returns hc (Bt,Lc,D,N).

    Computes exactly the same recurrence as the closed form it replaces:

        h_t = E_t * (h_0 + sum_{s<=t} b_s / E_s)
            = exp(logE_t) * h_0 + sum_{s<=t} exp(logE_t - logE_s) * b_s

    but never forms 1/E. Since logE is a cumulative sum of dA = delta*A with
    A < 0 and delta > 0, logE is non-increasing, so for s <= t the pairwise
    exponent (logE_t - logE_s) is ALWAYS <= 0 and its exponential is bounded
    by 1. Nothing can overflow, and no clamp is needed: a decay span that
    underflows simply gives 0, which is the correct answer -- forget
    completely -- rather than the clamped form's E_t/E_s == 1, forget nothing.

    The clamp(max=0.0) is a guard for the s > t half of the pairwise matrix
    only, which the causal mask then zeroes; on the s <= t half it is inactive
    and passes gradient unchanged.
    """
    lc = xc.shape[1]
    logE = (dc.unsqueeze(-1) * A).cumsum(dim=1)              # (Bt, Lc, D, N)
    b = (dc * xc).unsqueeze(-1) * Bc.unsqueeze(2)            # (Bt, Lc, D, N)
    ldiff = logE.unsqueeze(2) - logE.unsqueeze(1)            # (Bt,Lc_t,Lc_s,D,N)
    causal = torch.ones(lc, lc, dtype=torch.bool,
                        device=xc.device).tril()[None, :, :, None, None]
    # MASK IN LOG SPACE, BEFORE THE EXP. For s > t the exponent
    # logE_t - logE_s is POSITIVE and can be large (it reached +1200 in a
    # constructed check), so `exp(ldiff) * mask` overflows to inf and then
    # inf * 0 = nan -- the same overflow-then-mask trap as the fp16 focal
    # clamp and the asin clamp. Filling with -inf first makes exp yield an
    # exact 0 with a 0 gradient, and nothing large is ever materialised.
    decay = torch.exp(ldiff.masked_fill(~causal, float("-inf")))
    intra = torch.einsum("btsdn,bsdn->btdn", decay, b)
    return torch.exp(logE) * h.unsqueeze(1) + intra


def _chunked_selective_scan(x: torch.Tensor, delta: torch.Tensor,
                            A: torch.Tensor, B: torch.Tensor,
                            C: torch.Tensor, chunk: int = 64,
                            collect_stats: bool = True,
                            checkpoint_chunks: bool = True):
    """Exact selective scan, divide-free chunked (SSD) form.

    Shapes: x, delta (Bt, L, D); A (D, N); B, C (Bt, L, N).
    Returns ``(y, stats)``: y is (Bt, L, D); stats is a dict of 0-d diagnostic
    tensors for the ``lc/ssm_*`` taps.

    WHY THIS REPLACED THE b/E CLOSED FORM. The previous implementation
    computed ``acc = cumsum(b / E)`` and then ``hc = E * (h + acc)``. That is
    algebraically the same recurrence, but it forms 1/E, which required
    clamping logE at -30 to keep the division finite -- and the clamp broke
    the identity it was guarding: once logE_t and logE_s are both pinned,
    E_t/E_s evaluates to exactly 1, so the chunk stops forgetting instead of
    forgetting completely.

    It was not merely fragile, it was WRONG. Against a float64 reference at
    the regime of job 549449 (delta pinned at 0.2 by dt_max, A = -[1..16],
    x at the observed z_fused scale), the b/E form returns amax 381.3 where
    the true value is 139.8 -- a RELATIVE ERROR OF 2.68, i.e. 268% -- while
    this form returns 139.775 with relative error 3.1e-07, which is fp32
    round-off. The 2.7x inflation it manufactured is what drove ssm_out past
    the fp16 ceiling at the island exit.

    So this is a CORRECTNESS fix, not a numerical workaround. It computes the
    same mathematics correctly, where the previous form computed it wrongly.

    MEMORY. The pairwise decay is (Bt, Lc, Lc, D, N) -- 32 MB per chunk at
    Lc=64 on the OPV2V grid, and autograd saves the clamp input and the exp
    output as well, so the naive cost across 138 chunks x 4 directions is well
    past the 46 GB card. ``checkpoint_chunks`` recomputes each chunk in the
    backward instead of storing it, which caps the scan at roughly one chunk
    of live pairwise state (~100 MB) for about 2x the scan compute -- and the
    scan is a small fraction of step time, so that is nearly free.

    Statistics are gathered OUTSIDE the checkpointed region, from a cheap
    no-grad recomputation of logE and b (0.5 MB each, no pairwise tensor).
    Collecting them inside would DOUBLE-COUNT: checkpointing runs the forward
    twice, once under no_grad and again during backward.
    """
    bt, L, d = x.shape
    n = A.shape[1]
    h = x.new_zeros((bt, d, n))
    ys = []

    # Three-band census of the cumulative log-decay, plus internal magnitudes.
    # Accumulated as TENSORS, never .item(), so the loop stays sync-free.
    n_sat = x.new_zeros(())
    n_int = x.new_zeros(())
    n_total = 0
    amax_b_term = x.new_zeros(())
    amax_hc = x.new_zeros(())
    if collect_stats:
        with torch.no_grad():
            inv = (1.0 / (delta.unsqueeze(-1) * A).abs().clamp(min=1e-12)
                   ).flatten()
            # kthvalue, NOT quantile: quantile refuses tensors above 2**24
            # elements and this one is 18.0M on the OPV2V grid (job 549332).
            horizon_p50 = inv.median()
            horizon_p95 = torch.kthvalue(
                inv, max(1, int(0.95 * inv.numel()))).values
            del inv
    else:
        horizon_p50 = x.new_zeros(())
        horizon_p95 = x.new_zeros(())

    use_ckpt = checkpoint_chunks and torch.is_grad_enabled()
    for s in range(0, L, chunk):
        xc = x[:, s:s + chunk]
        dc = delta[:, s:s + chunk]
        Bc = B[:, s:s + chunk]
        Cc = C[:, s:s + chunk]

        if collect_stats:
            with torch.no_grad():
                logE_ng = (dc.unsqueeze(-1) * A).cumsum(dim=1)
                n_sat += (logE_ng <= -30.0).sum()
                n_int += (logE_ng >= -0.01).sum()
                n_total += logE_ng.numel()
                b_ng = (dc * xc).unsqueeze(-1) * Bc.unsqueeze(2)
                amax_b_term = torch.maximum(amax_b_term, b_ng.abs().max())
                del logE_ng, b_ng

        if use_ckpt:
            hc = torch.utils.checkpoint.checkpoint(
                _scan_chunk, xc, dc, Bc, A, h, use_reentrant=False)
        else:
            hc = _scan_chunk(xc, dc, Bc, A, h)

        if collect_stats:
            with torch.no_grad():
                amax_hc = torch.maximum(amax_hc, hc.abs().max())

        ys.append(torch.einsum("bldn,bln->bld", hc, Cc))
        h = hc[:, -1]

    total = float(max(n_total, 1))
    sat, integ = n_sat / total, n_int / total
    stats = {
        # The bands still describe how much of the scan has effectively
        # forgotten -- but they are now DESCRIPTIVE, not diagnostic of a
        # defect: with no division there is no clamp and deep decay is simply
        # correct behaviour.
        "saturated": sat,
        "integrator": integ,
        "healthy": 1.0 - sat - integ,
        "horizon_p50": horizon_p50,
        "horizon_p95": horizon_p95,
        "b_term": amax_b_term,
        "hc": amax_hc,
    }
    return torch.cat(ys, dim=1), stats


class CSSM(nn.Module):
    """Collaborative SSM over BEV feature maps.

    Inputs   z_fused, z_i : (B, C, H, W).
    Output   x_ssm        : (B, C, H, W).

    Parameters
    ----------
    channels   C of the incoming feature maps.
    d_inner    scan width (input projection C -> d_inner; output back to C).
    d_state    state dimension N per channel.
    scan       'cross2d' | 'raster'.
    pool       spatial pooling factor for the scan (1 = full resolution).
    backend    'reference' | 'cuda' (requires mamba-ssm; falls back with a
               warning when unavailable).

    Example
    -------
    >>> m = CSSM(channels=32, d_inner=16, d_state=4, pool=1)
    >>> m(torch.rand(2, 32, 8, 8), torch.rand(2, 32, 8, 8)).shape
    torch.Size([2, 32, 8, 8])
    """

    def __init__(self, channels: int, d_inner: int = 64, d_state: int = 16,
                 scan: str = "cross2d", pool: int = 2,
                 backend: str = "reference", chunk: int = 64,
                 dt_init: str = "constant",
                 dt_max: Optional[float] = None) -> None:
        super().__init__()
        if scan not in ("cross2d", "raster"):
            raise ValueError(f"unknown scan mode: {scan!r}")
        self.scan = scan
        self.pool = max(1, int(pool))
        self.chunk = int(chunk)
        # None = unbounded, reproducing every run up to job 549427.
        self.dt_max = None if dt_max is None else float(dt_max)
        self.backend = backend
        if backend == "cuda":  # pragma: no cover - optional dependency
            try:
                import mamba_ssm  # noqa: F401
            except ImportError:
                logger.warning("mamba-ssm not installed; CSSM falls back to "
                               "the reference backend")
                self.backend = "reference"

        self.in_proj = nn.Conv2d(channels, d_inner, 1, bias=False)
        self.ego_proj = nn.Conv2d(channels, d_inner, 1, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.b_proj = nn.Linear(d_inner, d_state, bias=False)
        self.c_proj = nn.Linear(d_inner, d_state, bias=False)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float())
            .repeat(d_inner, 1))                       # A = -exp(a_log) < 0
        self.d_skip = nn.Parameter(torch.ones(d_inner))
        self._init_dt_bias(dt_init, d_inner)
        self.out_proj = nn.Conv2d(d_inner, channels, 1, bias=False)

    def _init_dt_bias(self, mode: str, d_inner: int,
                      dt_min: float = 1e-3, dt_max: float = 1e-1,
                      dt_floor: float = 1e-4, seed: int = 2026) -> None:
        """Initialise ``dt_proj.bias``; the ONLY thing dt_init changes.

        ``'constant'`` reproduces the original ``bias = -2.0``. That gives
        ``softplus(-2.0) = 0.1269``, already above Mamba's ``dt_max = 0.1``
        before the weight term is added at all.

        ``'mamba'`` follows the reference ``dt_init``: sample
        ``dt ~ exp(U(log dt_min, log dt_max))``, floor it, and set the bias to
        the inverse softplus ``dt + log(-expm1(-dt))``, so ``softplus(bias)``
        lands inside ``[dt_min, dt_max]`` by construction.

        The WEIGHT is deliberately left alone. ``nn.Linear``'s default
        kaiming-uniform bound is ``1/sqrt(fan_in) = d_inner**-0.5``, which is
        already exactly Mamba's ``dt_init_std = dt_rank**-0.5 * dt_scale`` at
        ``dt_scale = 1``. Re-initialising it would draw from the global RNG
        and shift every module constructed afterwards, so a dt_init run would
        differ from its baseline in far more than dt_proj -- which is the one
        thing this experiment must not do.

        Sampling likewise uses a LOCAL generator, so the global stream is
        untouched and every other parameter stays bit-identical.
        """
        if mode == "constant":
            nn.init.constant_(self.dt_proj.bias, -2.0)
            return
        if mode != "mamba":
            raise ValueError(f"unknown dt_init: {mode!r}; expected "
                             "'constant' or 'mamba'")
        gen = torch.Generator().manual_seed(seed)
        dt = torch.exp(
            torch.rand(d_inner, generator=gen)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))     # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    # -- sequence orderings -------------------------------------------------

    @staticmethod
    def _flatten(t: torch.Tensor, direction: int) -> torch.Tensor:
        """(B, D, H, W) -> (B, L, D) in one of 4 scan orders."""
        if direction in (2, 3):                        # column-major
            t = t.transpose(2, 3)
        b, d, h, w = t.shape
        seq = t.reshape(b, d, h * w).transpose(1, 2)
        if direction in (1, 3):                        # reversed
            seq = seq.flip(1)
        return seq

    @staticmethod
    def _unflatten(seq: torch.Tensor, direction: int,
                   hw: Tuple[int, int]) -> torch.Tensor:
        """(B, L, D) -> (B, D, H, W), inverse of `_flatten`."""
        h, w = hw
        if direction in (1, 3):
            seq = seq.flip(1)
        t = seq.transpose(1, 2)
        if direction in (2, 3):
            t = t.reshape(-1, t.shape[1], w, h).transpose(2, 3)
        else:
            t = t.reshape(-1, t.shape[1], h, w)
        return t

    # -- forward ------------------------------------------------------------

    def forward(self, z_fused: torch.Tensor, z_i: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        b, c, h, w = z_fused.shape
        x = self.in_proj(z_fused)
        e = self.ego_proj(z_i)
        if self.pool > 1:
            x = F.avg_pool2d(x, self.pool)
            e = F.avg_pool2d(e, self.pool)
        hp, wp = x.shape[-2:]

        A = -torch.exp(self.a_log)
        directions = (0, 1, 2, 3) if self.scan == "cross2d" else (0,)
        out = None
        for direction in directions:
            xs = self._flatten(x, direction)                     # (B, L, D)
            es = self._flatten(e, direction)
            # float32 island: see the PRECISION note in the module docstring.
            with torch.autocast(device_type=xs.device.type, enabled=False):
                xs32, es32 = xs.float(), es.float()
                delta = F.softplus(self.dt_proj(xs32))
                if self.dt_max is not None:
                    # SOFT ceiling, deliberately NOT clamp(max=...). clamp has
                    # exactly zero gradient above the bound, so an element
                    # whose pre-activation drifts past it is stuck there with
                    # no restoring force: dt_proj never learns to come back
                    # down. That is the same forward-safe / backward-dead
                    # shape as the asin hard clamp (96fbb2a) and the fp16
                    # focal clamp (RECON-3) -- the third instance of it.
                    # tanh saturates smoothly and keeps the gradient alive
                    # through the APPROACH to the ceiling -- 0.79 at the bound
                    # itself, 0.15 at the delta observed in 549412 -- which is
                    # where a restoring force is needed. It is NOT immune:
                    # sech^2 underflows to zero in float32 past raw delta ~2,
                    # so far above the bound it is backward-dead too. The
                    # bound's job is to stop delta ever getting there.
                    #
                    # WHY A CEILING AT ALL: dt_init sets where Delta STARTS;
                    # nothing held it during training. One landed step took it
                    # 0.318 -> 4.85 -> 11.1 (job 549412), and since
                    # b = (Delta * x) (x) B scales linearly in Delta, ssm_out
                    # followed 3.62 -> 192 -> 1612. Clamp saturation stayed at
                    # ~3% throughout, so the degeneracy was NOT the path.
                    #
                    # WHY 0.2 AND NOT 0.1: [0.001, 0.1] is Mamba's dt_init
                    # RANGE, not a runtime ceiling -- Mamba imposes none,
                    # because its scan never divides by E. A bound at 0.1
                    # distorts the top of that range by 24%; 0.2 distorts it
                    # by <2%. This is a WORKAROUND for our b/E closed form;
                    # the principled fix remains RECON-4 option 1.
                    delta = self.dt_max * torch.tanh(delta / self.dt_max)
                if direction == 0:
                    emit(taps, delta, module="CSSM", location="lc/ssm_delta")
                # Hoisted out of the call so they can be tapped: these are the
                # two learned projections whose product the scan is bilinear
                # in, and the existing taps jump straight from z_fused to
                # ssm_out with everything between invisible.
                b_map = self.b_proj(xs32)
                c_map = self.c_proj(es32)
                y, scan_stats = _chunked_selective_scan(
                    xs32, delta, A.float(), b_map, c_map, self.chunk,
                    collect_stats=(direction == 0))
                if direction == 0:
                    emit(taps, b_map, module="CSSM", location="lc/ssm_b_proj")
                    emit(taps, c_map, module="CSSM", location="lc/ssm_c_proj")
                    for key, loc in (
                            ("saturated", "lc/ssm_logE_saturated"),
                            ("healthy", "lc/ssm_logE_healthy"),
                            ("integrator", "lc/ssm_logE_integrator"),
                            ("horizon_p50", "lc/ssm_decay_horizon_p50"),
                            ("horizon_p95", "lc/ssm_decay_horizon_p95"),
                            ("b_term", "lc/ssm_b_term"),
                            ("hc", "lc/ssm_hc")):
                        emit(taps, scan_stats[key].reshape(1), module="CSSM",
                             location=loc)
                y = y + self.d_skip.float() * xs32
            y2d = self._unflatten(y.to(xs.dtype), direction, (hp, wp))
            out = y2d if out is None else out + y2d
        out = out / len(directions)

        if self.pool > 1:
            out = F.interpolate(out, size=(h, w), mode="bilinear",
                                align_corners=False)
        x_ssm = self.out_proj(out)
        emit(taps, x_ssm, module="CSSM", location="lc/ssm_out")
        return x_ssm
