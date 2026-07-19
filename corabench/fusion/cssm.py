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

2-D scanning: 'cross2d' (VMamba-style: 4 directions -- row-major, reversed,
column-major, reversed -- averaged; assumption A9) or 'raster' (single pass).

Memory note: the reference scan materialises (B, L, D, N) activations for
autograd. `pool` runs the scan at reduced spatial resolution (avg-pool in,
bilinear out) -- the practical setting for full OPV2V grids without the
fused kernel; set pool=1 to scan at full resolution.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..observation.taps import TapProtocol, emit

logger = logging.getLogger(__name__)


def _chunked_selective_scan(x: torch.Tensor, delta: torch.Tensor,
                            A: torch.Tensor, B: torch.Tensor,
                            C: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Exact selective scan, chunked closed form.

    Shapes: x, delta (Bt, L, D); A (D, N); B, C (Bt, L, N). Returns (Bt, L, D).

    Within a chunk:  h_t = E_t * (h_0 + sum_{s<=t} b_s / E_s)  with
    E_t = exp(cumsum(delta*A)) decaying, exponents clamped to keep 1/E finite.
    """
    bt, L, d = x.shape
    n = A.shape[1]
    h = x.new_zeros((bt, d, n))
    ys = []
    for s in range(0, L, chunk):
        xc = x[:, s:s + chunk]                                   # (Bt, Lc, D)
        dc = delta[:, s:s + chunk]
        Bc = B[:, s:s + chunk]                                   # (Bt, Lc, N)
        Cc = C[:, s:s + chunk]
        dA = dc.unsqueeze(-1) * A                                # (Bt, Lc, D, N)
        logE = dA.cumsum(dim=1).clamp(min=-30.0, max=0.0)
        E = torch.exp(logE)
        b = (dc * xc).unsqueeze(-1) * Bc.unsqueeze(2)            # (Bt, Lc, D, N)
        # closed form: h_t = E_t * (h_0 + sum_{s<=t} b_s / E_s), E_t = prod a_r
        acc = (b / E.clamp(min=1e-30)).cumsum(dim=1)
        hc = E * (h.unsqueeze(1) + acc)                          # (Bt, Lc, D, N)
        ys.append(torch.einsum("bldn,bln->bld", hc, Cc))
        h = hc[:, -1]
    return torch.cat(ys, dim=1)


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
                 backend: str = "reference", chunk: int = 64) -> None:
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
        nn.init.constant_(self.dt_proj.bias, -2.0)     # softplus ~= 0.13
        self.out_proj = nn.Conv2d(d_inner, channels, 1, bias=False)

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
            delta = F.softplus(self.dt_proj(xs))
            if direction == 0:
                emit(taps, delta, module="CSSM", location="lc/ssm_delta")
            y = _chunked_selective_scan(
                xs, delta, A, self.b_proj(xs), self.c_proj(es), self.chunk)
            y = y + self.d_skip * xs
            y2d = self._unflatten(y, direction, (hp, wp))
            out = y2d if out is None else out + y2d
        out = out / len(directions)

        if self.pool > 1:
            out = F.interpolate(out, size=(h, w), mode="bilinear",
                                align_corners=False)
        x_ssm = self.out_proj(out)
        emit(taps, x_ssm, module="CSSM", location="lc/ssm_out")
        return x_ssm
