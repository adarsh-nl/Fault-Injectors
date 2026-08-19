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
from ..compat import checkpoint, no_autocast

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
                 dt_bound: float = 0.2, fp32_island: bool = True,
                 checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.d_state = d_state
        self.chunk = chunk
        self.checkpoint_chunks = checkpoint_chunks
        # RUNTIME soft bound on delta (job 558108 post-mortem). `dt_max` above
        # is only the INIT range endpoint; nothing held delta there afterwards
        # and it reached 0.346 by step 2000 / 0.611 by step 4000.
        # tanh, never clamp: clamp has exactly zero gradient past the bound, so
        # a drifting channel is forward-safe and backward-dead. tanh bounds as
        # hard while keeping a restoring gradient through the approach.
        self.dt_bound = float(dt_bound)
        # fp32 island (job 558108 ROOT CAUSE). Under AMP the scan ran in fp16;
        # b = (delta*x) (x) B overflowed to inf, and the causal mask's EXACT
        # zeros (masked_fill(-inf) -> exp -> 0) then computed 0 * inf = NaN.
        # fp32's 3.4e38 ceiling removes the overflow, so the NaN cannot form.
        # Precision-only: no bound to saturate, no gradient dead zone.
        self.fp32_island = bool(fp32_island)
        # observables for the runtime gate (populated each forward, detached)
        self.last_delta_stats = None
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
        out_dtype = x.dtype
        # ── fp32 island ────────────────────────────────────────────────────
        # Everything from here to the exit cast runs in fp32 regardless of the
        # ambient autocast dtype. This is the fix for the job-558108 NaN: it
        # removes the overflow that produced `inf` in b, and therefore the
        # 0 * inf the causal mask's exact zeros formed from it.
        if self.fp32_island:
            dev = 'cuda' if x.is_cuda else 'cpu'
            island = no_autocast(dev)
            island.__enter__()
            x = x.float()
            dt_raw = dt_raw.float()
            b_in = b_in.float()
            c_in = c_in.float()
        else:
            island = None
        bsz, L, d = x.shape

        dt_pre = F.softplus(dt_raw + self.dt_bias.to(x.dtype))  # > 0, UNBOUNDED
        # soft bound: dt_bound * tanh(dt / dt_bound). Identity-like well below
        # the bound (tanh is linear near 0, and the init range [1e-3, 1e-1] is
        # entirely inside it), asymptotic above, gradient live through the
        # approach (it does die far past the bound, which is why the gate
        # aborts at 5x rather than relying on the bound alone).
        dt = (self.dt_bound * torch.tanh(dt_pre / self.dt_bound)
              if self.dt_bound else dt_pre)
        # gate observables -- detached, no graph, negligible cost.
        # sat_frac is measured on the PRE-bound delta: the post-tanh value can
        # never exceed dt_bound, so a post-bound ratio is identically <= 1 and
        # would always read 0.000 -- it would answer nothing. Pre-bound tells
        # us how hard the bound is actually working.
        with torch.no_grad():                                    # no-grad-ok
            ratio = (dt_pre / self.dt_bound) if self.dt_bound else dt_pre
            self.last_delta_stats = {
                'delta_max': float(dt.max()),
                'delta_p99': float(torch.quantile(dt.flatten().float(), 0.99)),
                'delta_mean': float(dt.mean()),
                # fraction of channels driven into the tanh's saturating region
                'sat_frac': float((ratio > 1.0).float().mean()),
            }

        A = -torch.exp(self.a_log.to(x.dtype))                  # (D, N), < 0
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
        y = y + self.d_skip.to(x.dtype) * x
        y = y.to(out_dtype)
        if island is not None:
            island.__exit__(None, None, None)
        return y


class CSSM(nn.Module):
    """Eq. 8 wrapper: 2-D cross-scan (A9), 4 directions averaged, with
    avg_pool2d(2) before the scan and bilinear upsample after (spec §3)."""

    def __init__(self, channels: int, d_state: int = 16, chunk: int = 64,
                 pool: int = 2, dt_bound: float = 0.2,
                 fp32_island: bool = True,
                 checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.pool = pool
        self.dt_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.dt_proj.bias)     # bias lives in SelectiveScan
        self.b_proj = nn.Linear(channels, d_state)
        self.c_proj = nn.Linear(channels, d_state)
        self.scan = SelectiveScan(channels, d_state, chunk,
                                  dt_bound=dt_bound, fp32_island=fp32_island,
                                  checkpoint_chunks=checkpoint_chunks)
        self.out_norm = nn.LayerNorm(channels)
        # sec 7.14 flow probe (off by default, zero cost when off): norms at
        # the three boundaries z_fused -> y_pre_ln -> y_post_ln plus the
        # GRADIENT norm crossing each, captured by tensor hooks during
        # backward. This is deliberately backward-side instrumentation: the
        # forward magnitudes were already shown non-predictive (spec 7.3).
        # NOTE for readers of the CSV: under AMP the hooks fire on the
        # GradScaler-SCALED gradients; the training script divides the
        # *_gradnorm values by the step's scale before logging.
        self.record_flow = False
        self.last_flow = {}

    def _flow_tap(self, key: str, t: torch.Tensor) -> None:
        """Record ||t|| and register a hook recording ||grad_t||. Read-only:
        the hook returns None, so the backward pass is unaltered."""
        with torch.no_grad():                                    # no-grad-ok
            self.last_flow[key] = float(t.detach().float().norm())
        self.last_flow[key + "_gradnorm"] = float("nan")
        if t.requires_grad:
            def _cap(g, _k=key):
                self.last_flow[_k + "_gradnorm"] = \
                    float(g.detach().float().norm())
            t.register_hook(_cap)

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
        y = y / 4.0
        if self.record_flow:
            self._flow_tap("y_pre_ln", y)
        y = self.out_norm(y)
        if self.record_flow:
            self._flow_tap("y_post_ln", y)
        y = y.transpose(1, 2).reshape(bsz, c, h, w)
        return F.interpolate(y, size=(H, W), mode="bilinear",
                             align_corners=False)
