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
disabled, even when the surrounding model is in AMP. The closed form divides
by the decay term E, whose exponent is clamped at -30, so the intermediate
`b / E` reaches ~1e13. float16 tops out at 65504, so under autocast that
intermediate overflows to inf, f_out becomes inf, the distillation loss
explodes and GradScaler then skips every step -- a failure that presents as
"the model does not learn" rather than as a crash. Casting back to the
caller's dtype at the boundary keeps the rest of the model in AMP.

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


def _chunked_selective_scan(x: torch.Tensor, delta: torch.Tensor,
                            A: torch.Tensor, B: torch.Tensor,
                            C: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Exact selective scan, chunked closed form.

    Shapes: x, delta (Bt, L, D); A (D, N); B, C (Bt, L, N).
    Returns ``(y, clamped_fraction)``: y is (Bt, L, D); clamped_fraction is a
    0-d tensor giving the share of cumulative log-decay entries pinned at the
    -30 floor, for the ``lc/ssm_logE_clamped`` tap.

    Within a chunk:  h_t = E_t * (h_0 + sum_{s<=t} b_s / E_s)  with
    E_t = exp(cumsum(delta*A)) decaying, exponents clamped to keep 1/E finite.

    NOTE the clamp is a correctness defect, not a guard -- see RECON-4 in
    docs/corabench_design.md. It exists only because this form divides by E;
    once logE_t and logE_s are both pinned, E_t/E_s reads as exactly 1 and the
    recurrence degenerates from near-total forgetting into NO forgetting, an
    undamped sum over the whole sequence. Deliberately unchanged for now:
    the tap measures how much of the scan is affected before anything moves.
    """
    bt, L, d = x.shape
    n = A.shape[1]
    h = x.new_zeros((bt, d, n))
    ys = []
    # Three-band census of the cumulative log-decay. The two pathological
    # regimes are OPPOSITE tails that coexist in one tensor, because logE is
    # per (D, N) and both delta and |A| vary across it:
    #   saturated  (logE <= -30)     pinned at the floor. E_t/E_s reads as 1
    #                                so the chunk stops forgetting -- but
    #                                h = E_last*(h_prev+acc) ANNIHILATES the
    #                                carried state, so this accumulation is
    #                                bounded by `chunk`, not by L.
    #   healthy    (-30 < logE < -0.01)  real decay.
    #   integrator (logE >= -0.01)   E ~ 1, nothing decays and nothing is
    #                                annihilated at the boundary, so the state
    #                                integrates across EVERY chunk -- an
    #                                L-fold accumulator. This is correct SSM
    #                                math for delta -> 0, and it is precisely
    #                                what Mamba's dt_init (dt_min = 0.001)
    #                                exists to prevent.
    # Accumulated as TENSORS, never .item(), so the chunk loop stays free of
    # device syncs: one `.item()` per chunk costs 552 syncs per forward here
    # (138 chunks x 4 directions) and dominated the step time when it was
    # first written this way.
    n_sat = x.new_zeros(())
    n_int = x.new_zeros(())
    n_total = 0
    # Decay horizon is a property of dA alone, so it is computed once over the
    # whole sequence rather than per chunk -- (Bt, L, D, N) under no_grad.
    with torch.no_grad():
        inv = 1.0 / (delta.unsqueeze(-1) * A).abs().clamp(min=1e-12)
        horizon_p50 = inv.median()
        horizon_p95 = inv.flatten().quantile(0.95)
        del inv
    for s in range(0, L, chunk):
        xc = x[:, s:s + chunk]                                   # (Bt, Lc, D)
        dc = delta[:, s:s + chunk]
        Bc = B[:, s:s + chunk]                                   # (Bt, Lc, N)
        Cc = C[:, s:s + chunk]
        dA = dc.unsqueeze(-1) * A                                # (Bt, Lc, D, N)
        # NOTE the cumsum is over the CHUNK SLICE: logE resets every `chunk`
        # positions. Any claim about accumulation "over the whole sequence"
        # must come from the integrator band, where the chunk boundary does
        # not annihilate h, and never from the saturated band, where it does.
        raw_logE = dA.cumsum(dim=1)
        with torch.no_grad():
            n_sat += (raw_logE <= -30.0).sum()
            n_int += (raw_logE >= -0.01).sum()
            n_total += raw_logE.numel()
        logE = raw_logE.clamp(min=-30.0, max=0.0)
        E = torch.exp(logE)
        b = (dc * xc).unsqueeze(-1) * Bc.unsqueeze(2)            # (Bt, Lc, D, N)
        # closed form: h_t = E_t * (h_0 + sum_{s<=t} b_s / E_s), E_t = prod a_r
        acc = (b / E.clamp(min=1e-30)).cumsum(dim=1)
        hc = E * (h.unsqueeze(1) + acc)                          # (Bt, Lc, D, N)
        ys.append(torch.einsum("bldn,bln->bld", hc, Cc))
        h = hc[:, -1]
    total = float(max(n_total, 1))
    sat, integ = n_sat / total, n_int / total
    stats = {
        "saturated": sat,
        "integrator": integ,
        "healthy": 1.0 - sat - integ,
        # p50 is paired with p95 deliberately: the integrator regime is a
        # TAIL, and a median sitting at ~1 position would hide a tail running
        # past L entirely.
        "horizon_p50": horizon_p50,
        "horizon_p95": horizon_p95,
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
                 dt_init: str = "constant") -> None:
        super().__init__()
        if scan not in ("cross2d", "raster"):
            raise ValueError(f"unknown scan mode: {scan!r}")
        self.scan = scan
        self.pool = max(1, int(pool))
        self.chunk = int(chunk)
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
                if direction == 0:
                    emit(taps, delta, module="CSSM", location="lc/ssm_delta")
                y, scan_stats = _chunked_selective_scan(
                    xs32, delta, A.float(), self.b_proj(xs32),
                    self.c_proj(es32), self.chunk)
                if direction == 0:
                    for key, loc in (
                            ("saturated", "lc/ssm_logE_saturated"),
                            ("healthy", "lc/ssm_logE_healthy"),
                            ("integrator", "lc/ssm_logE_integrator"),
                            ("horizon_p50", "lc/ssm_decay_horizon_p50"),
                            ("horizon_p95", "lc/ssm_decay_horizon_p95")):
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
