"""Gate for tools/apf/gate0/autocorr.py.

Every test runs on a 48 x 176 grid at 1.6 m in both axes. Never on a square
grid: a row/column or row-major/column-major confusion is invisible when h == w,
and the real seam is 48 x 176.

The expected half widths come from exact_half_width_m, which evaluates the
discrete lattice autocorrelation of the synthesised field in closed form. That
is used instead of the continuum formula 1.6651 * s because below roughly one
cell the discrete field is aliased and the continuum formula is wrong by more
than 20 percent, and the real measurement lands in exactly that sub-cell regime.

Run: python -u tools/apf/gate0/validate_autocorr.py
Exit 1 on any failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autocorr import (                                          # noqa: E402
    AutocorrConfig, PowerSpectrumAccumulator, half_width, measure,
    overall_verdict, ratio_table)

H, W = 48, 176
RES = 1.6

_FAILS = []


def check(name: str, ok: bool, detail: str = '') -> None:
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        _FAILS.append(name)
    print('%-4s %-46s %s' % (tag, name, detail), flush=True)


def otf(h: int, w: int, s_h: float, s_w: float) -> np.ndarray:
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    return np.exp(-2.0 * np.pi ** 2 * (s_h ** 2 * ky ** 2 + s_w ** 2 * kx ** 2))


def gaussian_field(rng, c: int, h: int, w: int, s_h: float,
                   s_w: float) -> np.ndarray:
    noise = rng.standard_normal((c, h, w))
    o = otf(h, w, s_h, s_w)[None, :, :]
    return np.fft.ifft2(np.fft.fft2(noise, axes=(1, 2)) * o, axes=(1, 2)).real


def exact_half_width_m(h: int, w: int, s_h: float, s_w: float, res: float,
                       axis: str) -> float:
    """Closed form 0.5 crossing of the DISCRETE lattice autocorrelation."""
    p = otf(h, w, s_h, s_w) ** 2
    if axis == 'h':
        k = np.fft.fftfreq(h)[:, None] * np.ones((1, w))
    elif axis == 'w':
        k = np.ones((h, 1)) * np.fft.fftfreq(w)[None, :]
    else:
        raise ValueError('axis must be h or w')
    total = p.sum()
    d = np.arange(0.0, 20.0, 0.0005)
    prev_d, prev_r = 0.0, 1.0
    for dd in d[1:]:
        r = float((p * np.cos(2.0 * np.pi * k * dd)).sum() / total)
        if prev_r > 0.5 >= r:
            t = (prev_r - 0.5) / (prev_r - r)
            return float(prev_d + t * (dd - prev_d)) * res
        prev_d, prev_r = dd, r
    return float('nan')


def build(rng, c: int, n: int, s_h: float, s_w: float, h: int = H,
          w: int = W, cfg: AutocorrConfig = None):
    cfg = cfg or AutocorrConfig(res_h_m=RES, res_w_m=RES)
    acc = PowerSpectrumAccumulator(h, w, cfg)
    for _ in range(n):
        acc.add(gaussian_field(rng, c, h, w, s_h, s_w))
    return acc


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------- T1
def t1():
    print('\n--- T1 isotropic s = 2.0 cells, 8 samples x 32 channels ---')
    rng = np.random.default_rng(0)
    acc = build(rng, 32, 8, 2.0, 2.0)
    res = measure(acc, 't1')
    ex_h = exact_half_width_m(H, W, 2.0, 2.0, RES, 'h')
    ex_w = exact_half_width_m(H, W, 2.0, 2.0, RES, 'w')
    print('     exact  H %.4f m   W %.4f m  (golden 5.3283)' % (ex_h, ex_w))
    print('     measured lat(H) %.4f m   long(W) %.4f m'
          % (res.axis_h_lat, res.axis_w_long))
    check('T1 exact H == golden 5.3283', abs(ex_h - 5.3283) < 5e-3,
          'exact %.4f' % ex_h)
    check('T1 exact W == golden 5.3283', abs(ex_w - 5.3283) < 5e-3,
          'exact %.4f' % ex_w)
    check('T1 measured long within 3 pct of exact',
          rel(res.axis_w_long, ex_w) < 0.03,
          '%.4f vs %.4f (%.2f pct)' % (res.axis_w_long, ex_w,
                                       100 * rel(res.axis_w_long, ex_w)))
    check('T1 measured lat within 3 pct of exact',
          rel(res.axis_h_lat, ex_h) < 0.03,
          '%.4f vs %.4f (%.2f pct)' % (res.axis_h_lat, ex_h,
                                       100 * rel(res.axis_h_lat, ex_h)))
    check('T1 anisotropy < 1.06', res.anisotropy < 1.06,
          'anisotropy %.4f' % res.anisotropy)


# ---------------------------------------------------------------- T2
def t2():
    print('\n--- T2 anisotropic s_h = 1.0, s_w = 3.0 ---')
    rng = np.random.default_rng(1)
    acc = build(rng, 32, 8, 1.0, 3.0)
    res = measure(acc, 't2')
    ex_h = exact_half_width_m(H, W, 1.0, 3.0, RES, 'h')
    ex_w = exact_half_width_m(H, W, 1.0, 3.0, RES, 'w')
    print('     exact  H %.4f m (golden 2.6642)   W %.4f m (golden 7.9927)'
          % (ex_h, ex_w))
    print('     measured lat(H) %.4f m   long(W) %.4f m'
          % (res.axis_h_lat, res.axis_w_long))
    check('T2 exact H == golden 2.6642', abs(ex_h - 2.6642) < 5e-3,
          'exact %.4f' % ex_h)
    check('T2 exact W == golden 7.9927', abs(ex_w - 7.9927) < 5e-3,
          'exact %.4f' % ex_w)
    check('T2 measured H within 4 pct', rel(res.axis_h_lat, ex_h) < 0.04,
          '%.4f vs %.4f (%.2f pct)' % (res.axis_h_lat, ex_h,
                                       100 * rel(res.axis_h_lat, ex_h)))
    check('T2 measured W within 4 pct', rel(res.axis_w_long, ex_w) < 0.04,
          '%.4f vs %.4f (%.2f pct)' % (res.axis_w_long, ex_w,
                                       100 * rel(res.axis_w_long, ex_w)))
    check('T2 axes not transposed (long > lat)',
          res.axis_w_long > res.axis_h_lat,
          'long %.4f > lat %.4f' % (res.axis_w_long, res.axis_h_lat))

    # transposed field into a (w, h) accumulator: the answer must flip
    rng2 = np.random.default_rng(1)
    cfg = AutocorrConfig(res_h_m=RES, res_w_m=RES)
    accT = PowerSpectrumAccumulator(W, H, cfg)
    for _ in range(8):
        f = gaussian_field(rng2, 32, H, W, 1.0, 3.0)
        accT.add(np.transpose(f, (0, 2, 1)))
    resT = measure(accT, 't2T')
    print('     transposed  lat(H-of-T) %.4f m   long(W-of-T) %.4f m'
          % (resT.axis_h_lat, resT.axis_w_long))
    check('T2 transposed answer flips',
          resT.axis_h_lat > resT.axis_w_long
          and abs(resT.axis_h_lat - res.axis_w_long) < 0.04 * res.axis_w_long
          and abs(resT.axis_w_long - res.axis_h_lat) < 0.04 * res.axis_h_lat,
          'T lat %.4f ~ orig long %.4f ; T long %.4f ~ orig lat %.4f'
          % (resT.axis_h_lat, res.axis_w_long, resT.axis_w_long,
             res.axis_h_lat))

    # wrong shape must raise
    acc2 = PowerSpectrumAccumulator(H, W, cfg)
    raised = False
    try:
        acc2.add(np.zeros((4, W, H)))
    except ValueError:
        raised = True
    check('T2 wrong shape raises ValueError', raised)
    raised2 = False
    try:
        acc2.add(np.zeros((H, W)))
    except ValueError:
        raised2 = True
    check('T2 non-3D raises ValueError', raised2)
    raised3 = False
    try:
        bad = np.zeros((4, H, W))
        bad[0, 0, 0] = np.nan
        acc2.add(bad)
    except ValueError:
        raised3 = True
    check('T2 non-finite raises ValueError', raised3)


# ---------------------------------------------------------------- T3
def t3():
    print('\n--- T3 sub-cell s = 0.35 ---')
    rng = np.random.default_rng(2)
    acc = build(rng, 64, 12, 0.35, 0.35)
    res = measure(acc, 't3')
    ex = exact_half_width_m(H, W, 0.35, 0.35, RES, 'w')
    print('     exact %.4f m (golden 1.1633) = %.4f cells' % (ex, ex / RES))
    print('     measured long %.4f m   lat %.4f m'
          % (res.axis_w_long, res.axis_h_lat))
    check('T3 exact == golden 1.1633', abs(ex - 1.1633) < 5e-3,
          'exact %.4f' % ex)
    check('T3 measured within 5 pct', rel(res.axis_w_long, ex) < 0.05,
          '%.4f vs %.4f (%.2f pct)' % (res.axis_w_long, ex,
                                       100 * rel(res.axis_w_long, ex)))
    check('T3 result below one cell (1.6 m)', res.axis_w_long < RES,
          '%.4f m < 1.6 m' % res.axis_w_long)
    cont = 1.66511 * 0.35 * RES
    check('T3 continuum formula would be wrong by > 20 pct', ex > 1.2 * cont,
          'exact %.4f > 1.2 * continuum %.4f = %.4f' % (ex, cont, 1.2 * cont))


# ---------------------------------------------------------------- T3b
def t3b():
    print('\n--- T3b GRID FLOOR, s = 0 (white noise) ---')
    fl_h = exact_half_width_m(H, W, 0.0, 0.0, RES, 'h')
    fl_w = exact_half_width_m(H, W, 0.0, 0.0, RES, 'w')
    print('     floor H %.4f m (%.4f cells)   floor W %.4f m (%.4f cells)'
          % (fl_h, fl_h / RES, fl_w, fl_w / RES))
    check('T3b floor H == 0.9653 m', abs(fl_h - 0.9653) < 0.01,
          '%.4f' % fl_h)
    check('T3b floor W == 0.9653 m', abs(fl_w - 0.9653) < 0.01,
          '%.4f' % fl_w)
    r_pose = 1.00 / fl_w
    r_lat = 3.60 / fl_w
    print('     severe pose 1.00 m -> R = %.4f ; latency 300 ms 3.60 m -> R '
          '= %.4f' % (r_pose, r_lat))
    check('T3b severe pose CANNOT reach PASS at this seam', r_pose < 1.2,
          'R = %.4f < 1.2' % r_pose)
    check('T3b 300 ms latency CAN', r_lat > 1.2, 'R = %.4f > 1.2' % r_lat)


# ---------------------------------------------------------------- T4
def t4():
    print('\n--- T4 spectral vs direct spatial overlap at integer lags ---')
    rng = np.random.default_rng(3)
    cfg = AutocorrConfig(res_h_m=RES, res_w_m=RES)
    acc = PowerSpectrumAccumulator(H, W, cfg)
    fields = []
    for _ in range(4):
        f = gaussian_field(rng, 16, H, W, 1.5, 2.5)
        acc.add(f)
        fields.append(f - f.mean(axis=(1, 2), keepdims=True))

    def direct(dh: int, dw: int) -> float:
        num = 0.0
        num0 = 0.0
        for f in fields:
            a = f[:, dh:, dw:]
            b = f[:, :f.shape[1] - dh if dh else None,
                  :f.shape[2] - dw if dw else None]
            num += float((a * b).sum())
            num0 += float((f * f).sum())
        cnt = (H - dh) * (W - dw)
        return (num / cnt) / (num0 / (H * W))

    worst = 0.0
    for dh, dw in [(1, 0), (0, 1), (2, 3), (0, 5), (4, 0)]:
        sp = acc.rho_cells(float(dh), float(dw))
        di = direct(dh, dw)
        worst = max(worst, abs(sp - di))
        print('     lag (%d, %d)  spectral %.6f  direct %.6f  |d| %.6f'
              % (dh, dw, sp, di, abs(sp - di)))
    check('T4 max |spectral - direct| < 0.02', worst < 0.02,
          'max %.6f' % worst)


# ---------------------------------------------------------------- T5
def t5():
    print('\n--- T5 half_width interpolation ---')
    hw = half_width([0.0, 1.0, 2.0], [1.0, 0.6, 0.4], 0.5)
    check('T5 interpolates to exactly 1.5', abs(hw - 1.5) < 1e-12,
          'got %.12f' % hw)
    hw2 = half_width([0.0, 1.0, 2.0], [1.0, 0.9, 0.8], 0.5)
    check('T5 returns NaN when never crossed', np.isnan(hw2), 'got %r' % hw2)


# ---------------------------------------------------------------- T6
def t6():
    print('\n--- T6 verdicts ---')
    disp = {'pose_severe': 1.0, 'latency_300ms': 3.6, 'rot_only_70m': 0.73}
    lh = {'pose_severe': 2.0, 'latency_300ms': 2.0, 'rot_only_70m': 3.0}
    rows = ratio_table(disp, lh)
    by = {r['fault']: r for r in rows}
    for r in rows:
        print('     %-16s disp %.2f  l_half %.2f  R %.4f  %s'
              % (r['fault'], r['displacement_m'], r['l_half_m'], r['ratio'],
                 r['verdict']))
    check('T6 pose_severe MARGINAL at 0.50',
          by['pose_severe']['verdict'] == 'MARGINAL'
          and abs(by['pose_severe']['ratio'] - 0.50) < 1e-12)
    check('T6 latency_300ms PASS at 1.80',
          by['latency_300ms']['verdict'] == 'PASS'
          and abs(by['latency_300ms']['ratio'] - 1.80) < 1e-12)
    check('T6 rot_only_70m FAIL at 0.243',
          by['rot_only_70m']['verdict'] == 'FAIL'
          and abs(by['rot_only_70m']['ratio'] - 0.73 / 3.0) < 1e-12)
    check('T6 overall(latency, rot) == PASS',
          overall_verdict(rows, ['latency_300ms', 'rot_only_70m']) == 'PASS',
          overall_verdict(rows, ['latency_300ms', 'rot_only_70m']))
    check('T6 overall(rot) == FAIL',
          overall_verdict(rows, ['rot_only_70m']) == 'FAIL',
          overall_verdict(rows, ['rot_only_70m']))
    check('T6 overall(pose) == MARGINAL',
          overall_verdict(rows, ['pose_severe']) == 'MARGINAL',
          overall_verdict(rows, ['pose_severe']))
    und = ratio_table({'x': 1.0}, {'x': float('nan')})
    check('T6 NaN half width -> UNDETERMINED',
          und[0]['verdict'] == 'UNDETERMINED')
    und2 = ratio_table({'x': 1.0}, {'x': 0.0})
    check('T6 zero half width -> UNDETERMINED',
          und2[0]['verdict'] == 'UNDETERMINED')


# ---------------------------------------------------------------- config
def t0():
    print('--- T0 config rejection ---')
    bad = [dict(res_h_m=0.0), dict(res_w_m=-1.0), dict(pad_factor=1),
           dict(max_lag_m=0.0), dict(lag_step_m=0.0), dict(lag_step_m=99.0),
           dict(n_directions=0)]
    ok = True
    for kw in bad:
        try:
            AutocorrConfig(**kw)
            ok = False
            print('     NOT rejected: %r' % kw)
        except ValueError:
            pass
    check('T0 bad configs rejected', ok)
    c = AutocorrConfig()
    check('T0 defaults 1.6 / 1.6 / pad 2 / 8 dirs',
          c.res_h_m == 1.6 and c.res_w_m == 1.6 and c.pad_factor == 2
          and c.n_directions == 8)


def main() -> int:
    print('APF GATE 0 :: autocorr validation')
    print('grid %d x %d at %.2f m per cell (deliberately non-square)'
          % (H, W, RES))
    t0()
    t1()
    t2()
    t3()
    t3b()
    t4()
    t5()
    t6()
    print('')
    if _FAILS:
        print('OVERALL FAIL  (%d failed: %s)' % (len(_FAILS),
                                                 ', '.join(_FAILS)))
        return 1
    print('OVERALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
