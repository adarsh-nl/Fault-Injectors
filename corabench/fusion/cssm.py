"""CSSM -- the Mamba-based selective SSM of paper Eq. 8 (spec §1.3.4, §3).

The paper gives one line: ``X_ssm = CSSM(Z_fused, Linear(Z_fused), Z_i)``.
Every numerical choice below is therefore spec §3's, not the paper's; the
load-bearing ones are the divide-free scan (the previous ``b/E`` closed form
was measured 268% wrong vs float64 -- not fragile, WRONG) and Mamba's dt
init range (spec §5.3).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..compat import checkpoint

from ..selfcheck import assert_dt_init, assert_nonpositive


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """x such that softplus(x) = y (y > 0)."""
    return y + torch.log(-torch.expm1(-y))


class SelectiveScan(nn.Module):
    """Divide-free (SSD-form) selective scan over one 1-D sequence.

    h_t = exp(logE_t) h_0 + sum_{s<=t} exp(logE_t - logE_s) b_s,
    logE = cumsum(dt * A)  (non-increasing since A < 0, dt > 0), so every
    pairwise exponent for s <= t is <= 0 and its exp is bounded by 1: no
    division, no clamp, underflow forgets completely (correct) rather than a
    clamped "forget nothing" (the measured defect of the b/E form).
    """

    def __init__(self, d_inner: int, d_state: int = 16, chunk: int = 64,
                 dt_min: float = 1e-3, dt_max: float = 1e-1,
                 checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.d_state = d_state
        self.chunk = chunk
        self.checkpoint_chunks = checkpoint_chunks
        # S4D-real init: A = -exp(a_log), negative for every real a_log.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(torch.log(a).repeat(d_inner, 1))
        self.d_skip = nn.Parameter(torch.ones(d_inner))
        # dt bias: softplus^-1 of dt ~ exp(U(log dt_min, log dt_max)) -- the
        # Mamba range (spec §3); asserted below, and the previous
        # implementation's primary defect class (integrator/saturation).
        dt = torch.exp(torch.empty(d_inner).uniform_(
            math.log(dt_min), math.log(dt_max)))
        self.dt_bias = nn.Parameter(_inv_softplus(dt))
        assert_dt_init(self.dt_bias, dt_min, dt_max, where="SelectiveScan")

    def _chunk_scan(self, logE_c: torch.Tensor, b_c: torch.Tensor,
                    h0: torch.Tensor) -> tuple:
        """One chunk. logE_c, b_c: (B, Lc, D, N); h0: (B, D, N)."""
        # pairwise exponent (B, Lc, Lc, D, N); lower triangle (s <= t) <= 0.
        # The upper triangle (s > t) is non-causal and its exponent is
        # POSITIVE -- masking must happen on the EXPONENT (-inf -> exp = 0
        # exactly), never after the exp: exp(+big) overflows to inf and
        # inf * 0 = NaN. (Caught by this package's own validation gate.)
        pair = logE_c.unsqueeze(2) - logE_c.unsqueeze(1)
        Lc = logE_c.shape[1]
        tril = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool,
                                     device=logE_c.device))
        assert_nonpositive(pair[:, tril], where="SelectiveScan.pairwise")
        pair = pair.masked_fill(~tril.view(1, Lc, Lc, 1, 1), float("-inf"))
        decay = torch.exp(pair)                        # causal, in [0, 1]
        acc = torch.einsum("btsdn,bsdn->btdn", decay, b_c)
        h_all = torch.exp(logE_c) * h0.unsqueeze(1) + acc      # (B, Lc, D, N)
        return h_all, h_all[:, -1]

    def forward(self, x: torch.Tensor, dt_raw: torch.Tensor,
                b_in: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, L, D)   sequence (Z_fused tokens)
        dt_raw : (B, L, D)   pre-softplus step size (Linear(Z_fused))
        b_in   : (B, L, N)   input matrix  B(x)
        c_in   : (B, L, N)   output matrix C  (from Z_i -- Eq. 8)
        returns (B, L, D)
        """
        bsz, L, d = x.shape
        dt = F.softplus(dt_raw + self.dt_bias)                  # > 0
        A = -torch.exp(self.a_log)                              # (D, N), < 0
        dA = dt.unsqueeze(-1) * A                               # (B, L, D, N)
        b = (dt * x).unsqueeze(-1) * b_in.unsqueeze(2)          # (B, L, D, N)

        h0 = x.new_zeros(bsz, d, self.d_state)
        ys = []
        for s in range(0, L, self.chunk):
            dA_c = dA[:, s:s + self.chunk]
            b_c = b[:, s:s + self.chunk]
            logE_c = torch.cumsum(dA_c, dim=1)                  # <= 0, falling
            if self.checkpoint_chunks and torch.is_grad_enabled():
                h_all, h0 = checkpoint(self._chunk_scan, logE_c, b_c, h0)
            else:
                h_all, h0 = self._chunk_scan(logE_c, b_c, h0)
            y_c = torch.einsum("bldn,bln->bld",
                               h_all, c_in[:, s:s + self.chunk])
            ys.append(y_c)
        y = torch.cat(ys, dim=1)
        return y + self.d_skip * x


class CSSM(nn.Module):
    """Eq. 8 wrapper: 2-D cross-scan (A9), 4 directions averaged, with
    avg_pool2d(2) before the scan and bilinear upsample after (spec §3)."""

    def __init__(self, channels: int, d_state: int = 16, chunk: int = 64,
                 pool: int = 2, checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.pool = pool
        self.dt_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.dt_proj.bias)     # bias lives in SelectiveScan
        self.b_proj = nn.Linear(channels, d_state)
        self.c_proj = nn.Linear(channels, d_state)
        self.scan = SelectiveScan(channels, d_state, chunk,
                                  checkpoint_chunks=checkpoint_chunks)
        self.out_norm = nn.LayerNorm(channels)

    @staticmethod
    def _directions(t: torch.Tensor, h: int, w: int) -> list:
        """(B, H*W, C) row-major -> 4 scan orders as index permutations."""
        idx = torch.arange(h * w, device=t.device)
        row = idx
        row_r = idx.flip(0)
        col = (idx % w) * h + idx // w        # column-major order of cells
        col_r = col.flip(0)
        return [row, row_r, col, col_r]

    def forward(self, z_fused: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        bsz, c, H, W = z_fused.shape
        zf = F.avg_pool2d(z_fused, self.pool)
        zi = F.avg_pool2d(z_i, self.pool)
        h, w = zf.shape[-2:]
        xf = zf.flatten(2).transpose(1, 2)                      # (B, L, C)
        xi = zi.flatten(2).transpose(1, 2)
        dt_raw = self.dt_proj(xf)
        b_in = self.b_proj(xf)
        c_in = self.c_proj(xi)                                  # C = Z_i (Eq. 8)

        y = 0.0
        for perm in self._directions(xf, h, w):
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel(), device=perm.device)
            y_dir = self.scan(xf[:, perm], dt_raw[:, perm], b_in[:, perm],
                              c_in[:, perm])
            y = y + y_dir[:, inv]
        y = self.out_norm(y / 4.0)
        y = y.transpose(1, 2).reshape(bsz, c, h, w)
        return F.interpolate(y, size=(H, W), mode="bilinear",
                             align_corners=False)
