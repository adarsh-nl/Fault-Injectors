"""Normalised spatial autocorrelation of a stack of BEV feature maps.

Measures rho(d) = <F(u), F(u-d)> / <F(u), F(u)> at REAL VALUED lags d, so the
feature's spatial support can be read off in metres and compared against a
registration displacement.

Pure numpy. No torch, no opencood. Importable and testable off the cluster.

Four properties that are load bearing, each of which fails silently if dropped:

1. FRACTIONAL LAGS. Severe pose displacement at the CoBEVT seam is 0.6 cells.
   An integer lag autocorrelation cannot resolve the regime the decision turns
   on, and would report the same number for 0.2 and 0.8 cells. rho is therefore
   evaluated from the power spectrum at arbitrary real d, never by indexing an
   inverse transform.
2. ZERO PAD PLUS MASK NORMALISATION. A raw circular FFT autocorrelation wraps
   the far edge of the map onto the near edge, which manufactures correlation at
   large lag out of nothing. The data is padded and the overlap count at each
   lag is divided out via the autocorrelation of the support mask.
3. PER CHANNEL MEAN SUBTRACTION before transforming. Without it rho is
   dominated by DC and every field measures as infinitely wide.
4. ACCUMULATE THE POWER SPECTRUM, evaluate rho once at the end. rho is a ratio
   of linear functionals of the spectrum, so this is exact rather than an
   approximation, and a full pass over a split costs one array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

R_PASS = 1.2
R_FAIL = 0.3


@dataclass(frozen=True)
class AutocorrConfig:
    """Geometry and sampling for one autocorrelation measurement.

    res_h_m and res_w_m stay SEPARATE FIELDS and nothing in this module may
    assume they are equal. The seam is 48 x 176, not square, and a row/column
    confusion is exactly what a shared resolution would hide.
    """

    res_h_m: float = 1.6
    res_w_m: float = 1.6
    pad_factor: int = 2
    max_lag_m: float = 8.0
    lag_step_m: float = 0.02
    n_directions: int = 8

    def __post_init__(self) -> None:
        if not (self.res_h_m > 0.0) or not np.isfinite(self.res_h_m):
            raise ValueError('res_h_m must be finite and positive, got %r'
                             % (self.res_h_m,))
        if not (self.res_w_m > 0.0) or not np.isfinite(self.res_w_m):
            raise ValueError('res_w_m must be finite and positive, got %r'
                             % (self.res_w_m,))
        if int(self.pad_factor) != self.pad_factor or self.pad_factor < 2:
            raise ValueError('pad_factor must be an integer >= 2 (a factor of '
                             '1 is a circular autocorrelation and wraps), got '
                             '%r' % (self.pad_factor,))
        if not (self.max_lag_m > 0.0) or not np.isfinite(self.max_lag_m):
            raise ValueError('max_lag_m must be finite and positive, got %r'
                             % (self.max_lag_m,))
        if not (self.lag_step_m > 0.0) or not np.isfinite(self.lag_step_m):
            raise ValueError('lag_step_m must be finite and positive, got %r'
                             % (self.lag_step_m,))
        if self.lag_step_m > self.max_lag_m:
            raise ValueError('lag_step_m %r exceeds max_lag_m %r'
                             % (self.lag_step_m, self.max_lag_m))
        if int(self.n_directions) != self.n_directions or self.n_directions < 1:
            raise ValueError('n_directions must be a positive integer, got %r'
                             % (self.n_directions,))


class PowerSpectrumAccumulator:
    """Accumulates the summed power spectrum of zero padded feature maps.

    State is one (h*pad, w*pad) float64 array plus a scalar count, so memory is
    flat no matter how many samples are added.
    """

    def __init__(self, h: int, w: int, cfg: AutocorrConfig) -> None:
        self.h = int(h)
        self.w = int(w)
        self.cfg = cfg
        self.hp = int(h) * int(cfg.pad_factor)
        self.wp = int(w) * int(cfg.pad_factor)
        self.power = np.zeros((self.hp, self.wp), dtype=np.float64)
        self.count = 0
        self.n_channels = 0
        # Support mask power, built once. Its inverse transform is the number
        # of overlapping samples at each lag, which is what turns a padded
        # circular correlation into an unbiased one.
        mask = np.zeros((self.hp, self.wp), dtype=np.float64)
        mask[:self.h, :self.w] = 1.0
        mf = np.fft.fft2(mask)
        self.mask_power = (mf.real ** 2 + mf.imag ** 2)

    def add(self, feat: np.ndarray) -> None:
        """Accumulate one (C, H, W) feature map."""
        arr = np.asarray(feat)
        if arr.ndim != 3:
            raise ValueError('feat must be (C, H, W), got %dD with shape %s'
                             % (arr.ndim, (arr.shape,)))
        c, fh, fw = arr.shape
        if (fh, fw) != (self.h, self.w):
            # TRANSPOSITION GUARD. Do not reshape, do not transpose, do not
            # accept and warn. A silently transposed map swaps the
            # longitudinal and lateral half widths, which inverts the fault to
            # axis mapping the verdict depends on.
            raise ValueError(
                'feat spatial shape %s does not match accumulator (%d, %d). '
                'Refusing to reshape or transpose: a swapped H/W would '
                'silently exchange the longitudinal and lateral half widths.'
                % ((fh, fw), self.h, self.w))
        arr = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(arr).all():
            # A NaN here propagates to every lag through the transform and is
            # invisible in the output curve.
            raise ValueError('feat contains %d non-finite values'
                             % int((~np.isfinite(arr)).sum()))
        # Per channel spatial mean removal. Without it the DC term dominates
        # and every field measures as infinitely wide.
        arr = arr - arr.mean(axis=(1, 2), keepdims=True)
        padded = np.zeros((c, self.hp, self.wp), dtype=np.float64)
        padded[:, :self.h, :self.w] = arr
        spec = np.fft.fft2(padded, axes=(1, 2))
        self.power += (spec.real ** 2 + spec.imag ** 2).sum(axis=0)
        self.count += 1
        self.n_channels = c

    def _max_lag_cells(self) -> float:
        return 0.4 * min(self.h, self.w)

    def rho_cells(self, d_h: float, d_w: float) -> float:
        """rho at a real valued lag in CELLS."""
        d_h = float(d_h)
        d_w = float(d_w)
        if self.count == 0:
            raise ValueError('no samples accumulated')
        r = float(np.hypot(d_h, d_w))
        lim = self._max_lag_cells()
        if r > lim:
            raise ValueError(
                'lag %.4f cells exceeds 0.4 * min(h, w) = %.4f cells; the '
                'mask normalisation is ill conditioned beyond that'
                % (r, lim))
        ky = np.fft.fftfreq(self.hp)[:, None]
        kx = np.fft.fftfreq(self.wp)[None, :]
        basis = np.cos(2.0 * np.pi * (ky * d_h + kx * d_w))
        num = float((self.power * basis).sum())
        den = float((self.mask_power * basis).sum())
        den0 = float(self.mask_power.sum())
        if abs(den) <= 1e-9 * abs(den0):
            raise ValueError('mask normalisation denominator %.3e is near '
                             'zero at lag (%.4f, %.4f) cells' % (den, d_h, d_w))
        zero = float(self.power.sum()) / den0
        if zero == 0.0:
            raise ValueError('zero lag autocorrelation is 0; the field is '
                             'constant after mean removal')
        return (num / den) / zero

    def rho_metres(self, d_h_m: float, d_w_m: float) -> float:
        """rho at a real valued lag in METRES, using the two axis resolutions."""
        return self.rho_cells(float(d_h_m) / self.cfg.res_h_m,
                              float(d_w_m) / self.cfg.res_w_m)

    def curve(self, theta_rad: float) -> Tuple[np.ndarray, np.ndarray]:
        """Sample rho along one direction.

        theta is measured from the +W axis toward the +H axis, so theta = 0 is
        longitudinal (along W) and theta = pi/2 is lateral (along H).
        """
        n = int(np.floor(self.cfg.max_lag_m / self.cfg.lag_step_m)) + 1
        lags = np.arange(n, dtype=np.float64) * self.cfg.lag_step_m
        ct = float(np.cos(theta_rad))
        st = float(np.sin(theta_rad))
        vals = np.empty(n, dtype=np.float64)
        for i, L in enumerate(lags):
            vals[i] = self.rho_metres(L * st, L * ct)
        return lags, vals


def half_width(lags_m: Sequence[float], rho_vals: Sequence[float],
               level: float = 0.5) -> float:
    """First crossing below `level`, linearly interpolated, in the units of
    lags_m.

    Returns NaN if the curve never crosses. Deliberately NOT max_lag_m: a curve
    that never falls to the level means the feature is AT LEAST that wide, and
    clamping to the sampling limit would understate the width and flatter APF.
    """
    lags = np.asarray(lags_m, dtype=np.float64)
    vals = np.asarray(rho_vals, dtype=np.float64)
    if lags.shape != vals.shape or lags.ndim != 1:
        raise ValueError('lags_m and rho_vals must be 1-D and the same length')
    if lags.size < 2:
        raise ValueError('need at least two samples')
    for i in range(1, lags.size):
        a, b = vals[i - 1], vals[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a > level >= b:
            if a == b:
                return float(lags[i])
            t = (a - level) / (a - b)
            return float(lags[i - 1] + t * (lags[i] - lags[i - 1]))
    return float('nan')


class AutocorrResult:
    """Half widths and curves for one tap."""

    def __init__(self, tap: str, cfg: AutocorrConfig, thetas: np.ndarray,
                 lags_m: np.ndarray, curves: List[np.ndarray],
                 n_samples: int, n_channels: int) -> None:
        self.tap = tap
        self.cfg = cfg
        self.thetas = np.asarray(thetas, dtype=np.float64)
        self.lags_m = np.asarray(lags_m, dtype=np.float64)
        self.curves = [np.asarray(c, dtype=np.float64) for c in curves]
        self.n_samples = int(n_samples)
        self.n_channels = int(n_channels)
        self.l_half: List[float] = [half_width(self.lags_m, c, 0.5)
                                    for c in self.curves]
        self.l_70: List[float] = [half_width(self.lags_m, c, 0.7)
                                  for c in self.curves]
        self.l_30: List[float] = [half_width(self.lags_m, c, 0.3)
                                  for c in self.curves]
        # Named axes. theta = 0 is along +W (longitudinal), pi/2 along +H
        # (lateral).
        self.axis_w_long = self._at_theta(0.0)
        self.axis_h_lat = self._at_theta(np.pi / 2.0)
        finite = [v for v in self.l_half if np.isfinite(v) and v > 0.0]
        self.anisotropy = (float(max(finite) / min(finite)) if len(finite) >= 2
                           else float('nan'))

    def _at_theta(self, theta: float) -> float:
        if self.thetas.size == 0:
            return float('nan')
        i = int(np.argmin(np.abs(self.thetas - theta)))
        if abs(float(self.thetas[i]) - theta) > 1e-9:
            return float('nan')
        return float(self.l_half[i])

    def to_json(self) -> Dict:
        return {
            'tap': self.tap,
            'n_samples': self.n_samples,
            'n_channels': self.n_channels,
            'res_h_m': float(self.cfg.res_h_m),
            'res_w_m': float(self.cfg.res_w_m),
            'thetas_rad': [float(t) for t in self.thetas],
            'thetas_deg': [float(np.degrees(t)) for t in self.thetas],
            'lags_m': [float(v) for v in self.lags_m],
            'curves': [[float(v) for v in c] for c in self.curves],
            'l_half_m': [float(v) for v in self.l_half],
            'l_70_m': [float(v) for v in self.l_70],
            'l_30_m': [float(v) for v in self.l_30],
            'axis_w_long_m': float(self.axis_w_long),
            'axis_h_lat_m': float(self.axis_h_lat),
            'anisotropy': float(self.anisotropy),
        }


def measure(acc: PowerSpectrumAccumulator, tap: str) -> AutocorrResult:
    """Curves at n_directions evenly spaced thetas over [0, pi)."""
    n = int(acc.cfg.n_directions)
    thetas = np.arange(n, dtype=np.float64) * (np.pi / n)
    lags = None
    curves: List[np.ndarray] = []
    for t in thetas:
        lg, vals = acc.curve(float(t))
        lags = lg
        curves.append(vals)
    return AutocorrResult(tap, acc.cfg, thetas, lags, curves,
                          acc.count, acc.n_channels)


def ratio_table(displacements_m: Dict[str, float],
                l_half_by_fault: Dict[str, float]) -> List[Dict]:
    """One row per fault: displacement, half width, ratio, verdict."""
    rows: List[Dict] = []
    for fault in displacements_m:
        disp = float(displacements_m[fault])
        lh = l_half_by_fault.get(fault, float('nan'))
        lh = float(lh) if lh is not None else float('nan')
        if not np.isfinite(lh) or lh <= 0.0:
            rows.append({'fault': fault, 'displacement_m': disp,
                         'l_half_m': lh, 'ratio': float('nan'),
                         'verdict': 'UNDETERMINED'})
            continue
        ratio = disp / lh
        if not np.isfinite(ratio):
            verdict = 'UNDETERMINED'
        elif ratio >= R_PASS:
            verdict = 'PASS'
        elif ratio <= R_FAIL:
            verdict = 'FAIL'
        else:
            verdict = 'MARGINAL'
        rows.append({'fault': fault, 'displacement_m': disp, 'l_half_m': lh,
                     'ratio': float(ratio), 'verdict': verdict})
    return rows


def overall_verdict(rows: Sequence[Dict], decisive: Sequence[str]) -> str:
    """PASS if any DECISIVE fault passes, else MARGINAL if any is marginal,
    else FAIL if all are FAIL, else UNDETERMINED.

    Deliberately per fault. There is no global threshold: at a 1.6 m seam the
    white noise grid floor already caps severe pose below the PASS ratio, so a
    pooled score would let an undecidable fault drag a decidable one down.
    """
    by_fault = {r['fault']: r for r in rows}
    picked = [by_fault[f] for f in decisive if f in by_fault]
    if not picked:
        return 'UNDETERMINED'
    verdicts = [r['verdict'] for r in picked]
    if 'PASS' in verdicts:
        return 'PASS'
    if 'MARGINAL' in verdicts:
        return 'MARGINAL'
    if all(v == 'FAIL' for v in verdicts):
        return 'FAIL'
    return 'UNDETERMINED'
