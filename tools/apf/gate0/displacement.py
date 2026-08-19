"""Per fault registration displacement at the CoBEVT fusion seam.

Dataset only. No model, no GPU, no opencood import.

Produces the displacement table Gate 0's verdict consumes. The latency number
is MEASURED from consecutive frame vehicle motion, not assumed from a nominal
speed.

Two matrices are reported per latency tier and they are NOT alternatives, they
are different rungs of the covariance source ladder:

  raw second moment  E[dd^T] = arr.T @ arr / N
      What a stage with NO velocity prior must blur by. It includes the
      coherent forward motion, so it is large and strongly anisotropic.
  central covariance Cov[d] = np.cov(arr.T)
      What a later stage WITH a velocity prior blurs by, after shifting by the
      mean. Smaller and less anisotropic.

The handoff brief quotes one anisotropy figure of about 22:1 without saying
which matrix it belongs to. This measurement decides that. Both are printed,
labelled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    _BASE = yaml.CSafeLoader
    _LOADER_NAME = 'CSafeLoader'
except AttributeError:  # pragma: no cover - depends on libyaml build
    _BASE = yaml.SafeLoader
    _LOADER_NAME = 'SafeLoader (no libyaml)'


class _TolerantLoader(_BASE):
    """CSafeLoader that neutralises embedded numpy pickle tags.

    Some V2XSet label files serialise CARLA planner output as raw numpy
    scalars, emitting tags like
    !!python/object/apply:numpy.core.multiarray.scalar. A plain safe load
    refuses those by design. They appear in 8 of the 19 test scenarios, so
    skipping the affected files is not an option: it would drop a large part
    of the test set and the result would not be comparable to anything else
    measured on it.

    Every occurrence measured is under plan_trajectory, which this file never
    reads. The mapping is deliberately narrow, only python tags, so a
    malformed PLAIN yaml value still raises. Same treatment as
    src/datasets/opv2v.py, kept local because this file must stay importable
    without the package.
    """


for _tag in ('tag:yaml.org,2002:python/object/apply:',
             'tag:yaml.org,2002:python/object/new:',
             'tag:yaml.org,2002:python/name:',
             'tag:yaml.org,2002:python/tuple'):
    _TolerantLoader.add_multi_constructor(
        _tag, lambda loader, suffix, node: None)

HZ = 10.0
LAGS = (1, 2, 3)
POSE_TIERS = (('mild', 0.2), ('moderate', 0.4), ('severe', 0.6))


def load_yaml(path: str) -> dict:
    with open(path, 'r') as fh:
        d = yaml.load(fh, Loader=_TolerantLoader)
    # GUARD. The tolerant loader maps neutralised tags to None. That is only
    # safe while those tags stay in fields this file does not read. If one
    # ever lands in lidar_pose or a vehicle location, None would propagate
    # into the displacement arithmetic, so refuse rather than measure it.
    pose = d.get('lidar_pose')
    if pose is not None and any(p is None for p in pose[:5]):
        raise ValueError('%s: lidar_pose contains a neutralised tag' % path)
    for vid, v in (d.get('vehicles') or {}).items():
        loc = v.get('location')
        if loc is not None and any(x is None for x in loc[:2]):
            raise ValueError('%s: vehicle %s location contains a neutralised '
                             'tag' % (path, vid))
    return d


def pick_cav(scenario_dir: str) -> Optional[str]:
    """First sorted cav directory that is not "-1".

    Infrastructure is static and its yaw would define a BEV frame that never
    turns, so a lon/lat split taken against it would be meaningless.
    """
    subs = sorted(d for d in os.listdir(scenario_dir)
                  if os.path.isdir(os.path.join(scenario_dir, d)))
    for d in subs:
        if d != '-1':
            return d
    return None


def frame_stems(cav_dir: str) -> List[str]:
    """Frame stems kept as STRINGS and filtered to isdigit.

    Parsing them as ints would drop the zero padding and reorder the sequence
    on any scenario whose stems are not uniformly wide.
    """
    stems = set()
    for f in os.listdir(cav_dir):
        if f.endswith('.yaml'):
            stem = f[:-5]
            if stem.isdigit():
                stems.add(stem)
    return sorted(stems)


def rotate_to_bev(dx: float, dy: float, yaw_deg: float) -> Tuple[float, float]:
    """World displacement into the observing agent's BEV frame.

    Rotating is not optional: the fusion seam is in ego BEV coordinates and an
    anisotropy expressed in world coordinates is meaningless there.
    """
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


def collect(root: str, max_scenarios: Optional[int] = None,
            verbose: bool = True) -> Dict:
    scenarios = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
    per_lag: Dict[int, List[Tuple[float, float]]] = {k: [] for k in LAGS}
    per_lag_pos: Dict[int, List[Tuple[float, float]]] = {k: [] for k in LAGS}
    per_lag_own: Dict[int, List[Tuple[float, float]]] = {k: [] for k in LAGS}
    speeds: List[float] = []
    ranges: List[float] = []
    n_frames = 0
    used = []
    for sc in scenarios:
        sdir = os.path.join(root, sc)
        cav = pick_cav(sdir)
        if cav is None:
            continue
        cdir = os.path.join(sdir, cav)
        stems = frame_stems(cdir)
        if len(stems) < 2:
            continue
        used.append((sc, cav, len(stems)))
        cache: Dict[str, dict] = {}
        for st in stems:
            try:
                cache[st] = load_yaml(os.path.join(cdir, st + '.yaml'))
            except Exception as exc:
                raise SystemExit('failed to read %s/%s.yaml: %s'
                                 % (cdir, st, exc))
        n_frames += len(stems)
        for st in stems:
            d = cache[st]
            pose = d.get('lidar_pose')
            veh = d.get('vehicles') or {}
            if not pose:
                continue
            ax, ay = float(pose[0]), float(pose[1])
            for vid, v in veh.items():
                loc = v.get('location')
                if not loc:
                    continue
                ranges.append(math.hypot(float(loc[0]) - ax,
                                         float(loc[1]) - ay))
                if v.get('speed') is not None:
                    speeds.append(float(v['speed']))
        for i, st in enumerate(stems):
            d0 = cache[st]
            pose0 = d0.get('lidar_pose')
            if not pose0:
                continue
            yaw0 = float(pose0[4])          # DEGREES, index 4
            v0 = d0.get('vehicles') or {}
            for k in LAGS:
                j = i + k
                if j >= len(stems):
                    continue
                v1 = cache[stems[j]].get('vehicles') or {}
                for vid, a in v0.items():
                    b = v1.get(vid)
                    if b is None:
                        continue
                    la, lb = a.get('location'), b.get('location')
                    if not la or not lb:
                        continue
                    dx = float(lb[0]) - float(la[0])
                    dy = float(lb[1]) - float(la[1])
                    per_lag[k].append(rotate_to_bev(dx, dy, yaw0))
                    # vehicle position in the EGO BEV frame, for spatial bins
                    per_lag_pos[k].append(rotate_to_bev(
                        float(la[0]) - float(pose0[0]),
                        float(la[1]) - float(pose0[1]), yaw0))
                    # same displacement in the VEHICLE'S OWN heading frame.
                    # angle is [roll, yaw, pitch] in degrees, so index 1.
                    ang = a.get('angle')
                    own_yaw = float(ang[1]) if ang else yaw0
                    per_lag_own[k].append(rotate_to_bev(dx, dy, own_yaw))
        if verbose and len(used) % 5 == 0:
            print('  scenarios %d, frames %d' % (len(used), n_frames),
                  flush=True)
    return {'per_lag': per_lag, 'per_lag_pos': per_lag_pos,
            'per_lag_own': per_lag_own,
            'speeds': speeds,
            'ranges': np.asarray(ranges, dtype=np.float64),
            'scenarios': used, 'n_frames': n_frames}


def eig2(m: np.ndarray) -> Dict:
    w, v = np.linalg.eigh(m)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    ratio = float(w[0] / w[1]) if w[1] > 0 else float('inf')
    ang = float(math.degrees(math.atan2(v[1, 0], v[0, 0])))
    return {'matrix': [[float(x) for x in r] for r in m],
            'eigenvalues': [float(x) for x in w],
            'eig_ratio': ratio,
            'principal_axis_deg_from_lon': ang}


def summarise(arr_list: List[Tuple[float, float]]) -> Dict:
    arr = np.asarray(arr_list, dtype=np.float64)
    n = arr.shape[0]
    if n < 2:
        return {'n': int(n)}
    mag = np.hypot(arr[:, 0], arr[:, 1])
    second = arr.T @ arr / n
    cov = np.cov(arr.T)
    return {
        'n': int(n),
        'rms_m': float(np.sqrt((mag ** 2).mean())),
        'mean_m': float(mag.mean()),
        'p50_m': float(np.percentile(mag, 50)),
        'p90_m': float(np.percentile(mag, 90)),
        'mean_lon_m': float(arr[:, 0].mean()),
        'mean_lat_m': float(arr[:, 1].mean()),
        'second_moment': eig2(second),
        'central_cov': eig2(cov),
        'residual_rms_m': float(np.sqrt(np.trace(cov))),
    }


def pose_rms(sigma_t: float, sigma_r_deg: float, mean_r2: float) -> float:
    """Analytic, over the EMPIRICAL range distribution.

    The rotation term grows with range, so a nominal range would misstate it.
    """
    return float(math.sqrt(2.0 * sigma_t ** 2
                           + math.radians(sigma_r_deg) ** 2 * mean_r2))


GRID_FLOOR_M = 0.9653
RANGE_BINS = ((0.0, 40.0), (40.0, 75.0), (75.0, 140.0))


def stratify(out: Dict) -> Dict:
    """Stratified pass over data already parsed. Prints and returns detail.

    (a) p50 / p90 and the FRACTION above the grid floor, per latency tier.
        The fraction is prediction 6.
    (b) Latency anisotropy on MOVING objects only, defined as magnitude above
        the floor at that tier, alongside the pooled figure. The hypothesis
        under test is that pooled 3.1:1 is dilution by stationary vehicles.
    (c) Pose displacement as a function of RANGE rather than pooled, because
        the rotation term grows with range and a pooled value hides where the
        fault is actually decidable.
    """
    det: Dict = {'grid_floor_m': GRID_FLOOR_M, 'latency': {}, 'pose': {}}
    ranges = out['ranges']

    print('\n=== (a) MAGNITUDE PERCENTILES AND FRACTION ABOVE THE FLOOR ===',
          flush=True)
    print('grid floor = %.4f m' % GRID_FLOOR_M, flush=True)
    print('%-16s %8s %9s %9s %9s %12s' % ('tier', 'n', 'p50', 'p90', 'rms',
                                          'frac > floor'), flush=True)
    for k in LAGS:
        ms = int(round(1000.0 * k / HZ))
        arr = np.asarray(out['per_lag'][k], dtype=np.float64)
        mag = np.hypot(arr[:, 0], arr[:, 1])
        frac = float((mag > GRID_FLOOR_M).mean())
        name = 'latency_%dms' % ms
        det['latency'][name] = {
            'n': int(mag.size),
            'p50_m': float(np.percentile(mag, 50)),
            'p90_m': float(np.percentile(mag, 90)),
            'rms_m': float(np.sqrt((mag ** 2).mean())),
            'frac_above_floor': frac,
        }
        print('%-16s %8d %9.4f %9.4f %9.4f %12.4f'
              % (name, mag.size, np.percentile(mag, 50),
                 np.percentile(mag, 90), np.sqrt((mag ** 2).mean()), frac),
              flush=True)
    f300 = det['latency']['latency_300ms']['frac_above_floor']
    verdict = 'CONFIRMED' if f300 > 0.35 else 'REFUTED'
    det['prediction_6'] = {'threshold': 0.35, 'measured': f300,
                           'verdict': verdict}
    print('\nPREDICTION 6: fraction above floor at 300 ms = %.4f, threshold '
          '0.35 -> %s' % (f300, verdict), flush=True)
    if verdict == 'REFUTED':
        print('  A spatially uniform stage 1 oracle would blur a stationary '
              'majority to recover a moving minority. Gate 1 must not use a '
              'pooled sigma.', flush=True)

    print('\n=== (b) ANISOTROPY: POOLED vs MOVING ONLY ===', flush=True)
    print('moving = magnitude > %.4f m at that tier' % GRID_FLOOR_M,
          flush=True)
    for k in LAGS:
        ms = int(round(1000.0 * k / HZ))
        name = 'latency_%dms' % ms
        arr = np.asarray(out['per_lag'][k], dtype=np.float64)
        mag = np.hypot(arr[:, 0], arr[:, 1])
        mv = arr[mag > GRID_FLOOR_M]
        pooled_sm = eig2(arr.T @ arr / arr.shape[0])
        pooled_cv = eig2(np.cov(arr.T))
        rec = {'n_pooled': int(arr.shape[0]), 'n_moving': int(mv.shape[0]),
               'pooled_second_moment_ratio': pooled_sm['eig_ratio'],
               'pooled_central_cov_ratio': pooled_cv['eig_ratio']}
        print('\n%s   n pooled %d | n moving %d (%.1f pct)'
              % (name, arr.shape[0], mv.shape[0],
                 100.0 * mv.shape[0] / arr.shape[0]), flush=True)
        print('  POOLED  E[dd^T] %6.2f : 1   Cov[d] %6.2f : 1'
              % (pooled_sm['eig_ratio'], pooled_cv['eig_ratio']), flush=True)
        if mv.shape[0] >= 2:
            mv_sm = eig2(mv.T @ mv / mv.shape[0])
            mv_cv = eig2(np.cov(mv.T))
            rec['moving_second_moment'] = mv_sm
            rec['moving_central_cov'] = mv_cv
            rec['moving_mean_lon_m'] = float(mv[:, 0].mean())
            rec['moving_mean_lat_m'] = float(mv[:, 1].mean())
            rec['moving_rms_m'] = float(
                np.sqrt((mv[:, 0] ** 2 + mv[:, 1] ** 2).mean()))
            print('  MOVING  E[dd^T] %6.2f : 1   Cov[d] %6.2f : 1'
                  % (mv_sm['eig_ratio'], mv_cv['eig_ratio']), flush=True)
            print('          moving rms %.4f m | mean lon %.4f | mean lat '
                  '%.4f | axis %+.2f deg'
                  % (rec['moving_rms_m'], rec['moving_mean_lon_m'],
                     rec['moving_mean_lat_m'],
                     mv_sm['principal_axis_deg_from_lon']), flush=True)
            print('          dilution factor (moving / pooled, E[dd^T]) '
                  '%.2fx' % (mv_sm['eig_ratio'] / pooled_sm['eig_ratio']),
                  flush=True)
        det['latency'][name].update(rec)

    print('\n=== (c) POSE vs RANGE (not pooled) ===', flush=True)
    n_r = ranges.size
    print('range bins over %d vehicle range samples; floor %.4f m'
          % (n_r, GRID_FLOOR_M), flush=True)
    for tier, sig in POSE_TIERS:
        print('\n  pose_%s (sigma_t %.1f m, sigma_r %.1f deg)' % (tier, sig,
                                                                  sig),
              flush=True)
        print('  %-14s %10s %10s %8s %10s' % ('bin (m)', 'frac', 'r_mid',
                                              'd(r) m', 'R'), flush=True)
        rows = []
        for lo, hi in RANGE_BINS:
            sel = (ranges >= lo) & (ranges < hi)
            frac = float(sel.mean())
            r_mid = float(ranges[sel].mean()) if sel.any() else float('nan')
            d = pose_rms(sig, sig, r_mid ** 2) if sel.any() else float('nan')
            R = d / GRID_FLOOR_M
            rows.append({'lo_m': lo, 'hi_m': hi, 'frac': frac,
                         'r_mean_m': r_mid, 'd_m': d, 'R': R})
            print('  %-14s %10.4f %10.2f %8.4f %10.4f'
                  % ('%g-%g' % (lo, hi), frac, r_mid, d, R), flush=True)
        # empirical crossover: smallest r with R >= 1.2
        target = 1.2 * GRID_FLOOR_M
        val = target ** 2 - 2.0 * sig ** 2
        if val <= 0:
            r_cross = 0.0
        else:
            sr = math.radians(sig)
            r_cross = float(math.sqrt(val) / sr) if sr > 0 else float('inf')
        beyond = float((ranges >= r_cross).mean()) if np.isfinite(r_cross) \
            else 0.0
        det['pose']['pose_%s' % tier] = {'bins': rows,
                                         'r_cross_m': r_cross,
                                         'frac_beyond_cross': beyond}
        print('  crossover R = 1.2 at r = %.2f m | fraction of objects beyond '
              '= %.4f' % (r_cross, beyond), flush=True)
    return det


BEV_LON = (-140.8, 140.8)
BEV_LAT = (-38.4, 38.4)
BIN_M = 8.0


def _ratios(m: np.ndarray) -> Dict:
    e = eig2(m)
    ev = e['eigenvalues']
    var_ratio = e['eig_ratio']
    std_ratio = float(math.sqrt(var_ratio)) if np.isfinite(var_ratio) \
        else float('inf')
    return {'var_ratio': var_ratio, 'std_ratio': std_ratio,
            'eigenvalues': ev,
            'principal_axis_deg_from_lon': e['principal_axis_deg_from_lon']}


def local_analysis(out: Dict) -> Dict:
    """Per-cell latency covariance field, plus a heading-aligned cross-check.

    Tests FRAME MIXING, which conditioning on speed cannot detect: vehicles on
    different road headings pooled into one ego frame average their directions
    away no matter how fast they move. Sigma_delta(u) is defined as a spatially
    varying field and the coherence argument is about traffic being coherent
    LOCALLY at u; the previous pass pooled globally.

    Every anisotropy is reported BOTH as an eigenvalue (variance) ratio and as
    a standard deviation ratio. The doc's 22.4 is a std ratio, so it compares
    against sqrt(var_ratio).
    """
    det: Dict = {'bin_m': BIN_M, 'tiers': {}}

    print('\n=== (d) MEASURED FLEET SPEED ===', flush=True)
    sp = np.asarray(out['speeds'], dtype=np.float64)
    det['fleet_speed'] = {
        'n': int(sp.size), 'mean_mps': float(sp.mean()),
        'p50_mps': float(np.percentile(sp, 50)),
        'p90_mps': float(np.percentile(sp, 90)),
        'frac_below_0p5': float((sp < 0.5).mean())}
    print('n = %d | mean %.4f m/s | p50 %.4f | p90 %.4f | fraction below '
          '0.5 m/s %.4f' % (sp.size, sp.mean(), np.percentile(sp, 50),
                            np.percentile(sp, 90), (sp < 0.5).mean()),
          flush=True)
    print('doc assumes 10 m/s; measured fleet mean is %.4f m/s' % sp.mean(),
          flush=True)

    for k in LAGS:
        ms = int(round(1000.0 * k / HZ))
        name = 'latency_%dms' % ms
        rec: Dict = {}
        d = np.asarray(out['per_lag'][k], dtype=np.float64)
        pos = np.asarray(out['per_lag_pos'][k], dtype=np.float64)
        own = np.asarray(out['per_lag_own'][k], dtype=np.float64)
        pooled = _ratios(d.T @ d / d.shape[0])
        pooled_cov = _ratios(np.cov(d.T))
        rec['pooled'] = {'n': int(d.shape[0]), 'second_moment': pooled,
                         'central_cov': pooled_cov}
        print('\n=== %s ===' % name, flush=True)
        print('POOLED ego frame   E[dd^T] var %.3f : 1  std %.3f : 1 | '
              'Cov[d] var %.3f : 1  std %.3f : 1'
              % (pooled['var_ratio'], pooled['std_ratio'],
                 pooled_cov['var_ratio'], pooled_cov['std_ratio']),
              flush=True)

        # (b) heading aligned: each displacement in ITS OWN vehicle frame
        ho = _ratios(own.T @ own / own.shape[0])
        ho_cov = _ratios(np.cov(own.T))
        rec['heading_aligned'] = {'n': int(own.shape[0]),
                                  'second_moment': ho, 'central_cov': ho_cov}
        print('HEADING ALIGNED    E[dd^T] var %.3f : 1  std %.3f : 1 | '
              'Cov[d] var %.3f : 1  std %.3f : 1'
              % (ho['var_ratio'], ho['std_ratio'],
                 ho_cov['var_ratio'], ho_cov['std_ratio']), flush=True)
        print('                   mean lon %.4f  mean lat %.4f  (own frame)'
              % (own[:, 0].mean(), own[:, 1].mean()), flush=True)

        # (a) spatial bins over the BEV extent
        lon_edges = np.arange(BEV_LON[0], BEV_LON[1] + BIN_M, BIN_M)
        lat_edges = np.arange(BEV_LAT[0], BEV_LAT[1] + BIN_M, BIN_M)
        il = np.digitize(pos[:, 0], lon_edges) - 1
        ia = np.digitize(pos[:, 1], lat_edges) - 1
        mag = np.hypot(d[:, 0], d[:, 1])
        bins = []
        for a in range(len(lon_edges) - 1):
            for b in range(len(lat_edges) - 1):
                sel = (il == a) & (ia == b)
                n = int(sel.sum())
                if n < 30:
                    continue
                dd = d[sel]
                sm = _ratios(dd.T @ dd / n)
                cv = _ratios(np.cov(dd.T))
                bins.append({
                    'lon_lo': float(lon_edges[a]),
                    'lat_lo': float(lat_edges[b]), 'n': n,
                    'sm_var': sm['var_ratio'], 'sm_std': sm['std_ratio'],
                    'cv_var': cv['var_ratio'], 'cv_std': cv['std_ratio'],
                    'axis_deg': sm['principal_axis_deg_from_lon'],
                    'frac_above_floor': float(
                        (mag[sel] > GRID_FLOOR_M).mean())})
        rec['n_bins'] = len(bins)
        rec['bins'] = bins
        if bins:
            w = np.asarray([b['n'] for b in bins], dtype=np.float64)
            for key, lab in (('sm_std', 'E[dd^T] std'),
                             ('cv_std', 'Cov[d] std')):
                v = np.asarray([b[key] for b in bins], dtype=np.float64)
                o = np.argsort(v)
                vs, ws = v[o], w[o]
                cw = np.cumsum(ws) / ws.sum()
                q25 = float(vs[np.searchsorted(cw, 0.25)])
                q50 = float(vs[np.searchsorted(cw, 0.50)])
                q75 = float(vs[np.searchsorted(cw, 0.75)])
                rec['%s_weighted' % key] = {'q25': q25, 'median': q50,
                                            'q75': q75}
                print('PER-BIN %-14s weighted median %.3f : 1  IQR [%.3f, '
                      '%.3f]  over %d bins' % (lab, q50, q25, q75, len(bins)),
                      flush=True)
            # (e) fraction above floor per bin
            fr = np.asarray([b['frac_above_floor'] for b in bins])
            rec['frac_above_floor_bins'] = {
                'min': float(fr.min()), 'p25': float(np.percentile(fr, 25)),
                'median': float(np.percentile(fr, 50)),
                'p75': float(np.percentile(fr, 75)),
                'max': float(fr.max()),
                'pooled': float((mag > GRID_FLOOR_M).mean())}
            print('FRAC ABOVE FLOOR per bin: min %.4f p25 %.4f median %.4f '
                  'p75 %.4f max %.4f   (pooled %.4f)'
                  % (fr.min(), np.percentile(fr, 25), np.percentile(fr, 50),
                     np.percentile(fr, 75), fr.max(),
                     (mag > GRID_FLOOR_M).mean()), flush=True)
            if fr.max() - fr.min() > 0.25:
                print('   SPATIALLY STRUCTURED: spread %.4f across bins, so a '
                      'spatially uniform sigma is actively wrong at this tier'
                      % (fr.max() - fr.min()), flush=True)
        det['tiers'][name] = rec

    print('\n=== PREDICTION: per-bin median std ratio AND heading-aligned '
          'pooled std ratio both > 3 : 1 ===', flush=True)
    for k in LAGS:
        ms = int(round(1000.0 * k / HZ))
        name = 'latency_%dms' % ms
        r = det['tiers'][name]
        pb = r.get('sm_std_weighted', {}).get('median', float('nan'))
        ha = r['heading_aligned']['second_moment']['std_ratio']
        ok = (pb > 3.0) and (ha > 3.0)
        print('  %-16s per-bin median std %.3f | heading-aligned pooled std '
              '%.3f -> %s' % (name, pb, ha,
                              'CONFIRMED' if ok else 'REFUTED'), flush=True)
        det['tiers'][name]['prediction'] = {
            'per_bin_median_std': pb, 'heading_aligned_std': ha,
            'verdict': 'CONFIRMED' if ok else 'REFUTED'}
    ha300 = det['tiers']['latency_300ms']['heading_aligned'][
        'second_moment']['std_ratio']
    if ha300 <= 3.0:
        print('\n  Heading-aligned pooled std ratio at 300 ms is %.3f <= 3. '
              'The mixing is removed by construction in that frame, so '
              'section 9.2 coherence does not hold on this data and rung 3 '
              'has no target on latency.' % ha300, flush=True)
    return det


def selftest(tmp_root: str) -> bool:
    """Synthetic tree with a fixed heading and a known speed.

    If the recovered mean longitudinal displacement is not speed * lag and the
    mean lateral is not about zero, the yaw rotation is wrong.
    """
    print('--- SELFTEST: synthetic tree at %s ---' % tmp_root, flush=True)
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    yaw_deg = 30.0
    speed = 12.0
    dt = 1.0 / HZ
    cdir = os.path.join(tmp_root, 'scenario_a', '1001')
    os.makedirs(cdir)
    yaw = math.radians(yaw_deg)
    for i in range(12):
        t = i * dt
        # two vehicles, both translating along the heading at `speed`
        veh = {}
        for vid, off in ((7001, 0.0), (7002, 25.0)):
            x = 100.0 + off + speed * t * math.cos(yaw)
            y = 50.0 + speed * t * math.sin(yaw)
            veh[vid] = {'location': [x, y, 0.0]}
        doc = {
            'lidar_pose': [100.0, 50.0, 1.9, 0.0, yaw_deg, 0.0],
            'vehicles': veh,
        }
        with open(os.path.join(cdir, '%06d.yaml' % i), 'w') as fh:
            yaml.safe_dump(doc, fh)
    out = collect(tmp_root, verbose=False)
    ok = True
    for k in LAGS:
        arr = np.asarray(out['per_lag'][k], dtype=np.float64)
        exp = speed * k * dt
        mlon = float(arr[:, 0].mean())
        mlat = float(arr[:, 1].mean())
        good = abs(mlon - exp) < 1e-6 and abs(mlat) < 1e-6
        ok = ok and good
        print('  lag %d: expected lon %.4f m | got lon %.6f lat %.6f  %s'
              % (k, exp, mlon, mlat, 'OK' if good else 'WRONG'), flush=True)
    # A deliberately wrong yaw must NOT reproduce the answer, otherwise the
    # test is not actually exercising the rotation.
    bad_lon, bad_lat = rotate_to_bev(speed * dt * math.cos(yaw),
                                     speed * dt * math.sin(yaw), 0.0)
    print('  control: with yaw forced to 0 the same motion gives lon %.6f '
          'lat %.6f (must NOT be %.4f / 0)' % (bad_lon, bad_lat, speed * dt),
          flush=True)
    ok = ok and abs(bad_lat) > 1e-6
    print('  SELFTEST %s' % ('PASS' if ok else 'FAIL'), flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out-dir', default='results/apf/gate0')
    ap.add_argument('--max-scenarios', type=int, default=None)
    ap.add_argument('--skip-selftest', action='store_true')
    ap.add_argument('--local', action='store_true', dest='local_mode',
                    help='per-cell latency covariance field plus the '
                         'heading-aligned cross-check. Writes local.json and '
                         'does NOT rewrite displacements.json.')
    ap.add_argument('--stratify', action='store_true',
                    help='stratified pass over the same parsed data. Writes '
                         'stratified.json and does NOT rewrite '
                         'displacements.json.')
    ap.add_argument('--tmp', default=os.environ.get('CLAUDE_JOB_DIR', '/tmp')
                    + '/apf_selftest')
    args = ap.parse_args()

    print('yaml loader: %s' % _LOADER_NAME, flush=True)
    if not args.skip_selftest:
        if not selftest(args.tmp):
            print('SELFTEST FAILED: the yaw rotation is wrong, refusing to '
                  'report real numbers', flush=True)
            return 1
    print('', flush=True)

    print('--- COLLECT from %s ---' % args.root, flush=True)
    out = collect(args.root, args.max_scenarios)
    ranges = out['ranges']
    mean_r2 = float((ranges ** 2).mean())
    print('scenarios used: %d | frames: %d | vehicle range samples: %d'
          % (len(out['scenarios']), out['n_frames'], ranges.size), flush=True)
    print('range r: mean %.3f m | rms %.3f m | p50 %.3f | p90 %.3f | max %.3f'
          % (ranges.mean(), math.sqrt(mean_r2), np.percentile(ranges, 50),
             np.percentile(ranges, 90), ranges.max()), flush=True)
    print('mean(r^2) = %.3f m^2  (used for the pose rotation term)' % mean_r2,
          flush=True)

    flat: Dict[str, float] = {}
    detail: Dict[str, Dict] = {'latency': {}, 'pose': {},
                               'range_stats': {
                                   'n': int(ranges.size),
                                   'mean_m': float(ranges.mean()),
                                   'rms_m': float(math.sqrt(mean_r2)),
                                   'p50_m': float(np.percentile(ranges, 50)),
                                   'p90_m': float(np.percentile(ranges, 90)),
                                   'max_m': float(ranges.max()),
                                   'mean_r2_m2': mean_r2},
                               'n_scenarios': len(out['scenarios']),
                               'n_frames': out['n_frames'],
                               'hz': HZ}

    print('\n--- LATENCY (measured from consecutive frame vehicle motion) ---',
          flush=True)
    for k in LAGS:
        ms = int(round(1000.0 * k / HZ))
        s = summarise(out['per_lag'][k])
        name = 'latency_%dms' % ms
        flat[name] = s['rms_m']
        detail['latency'][name] = s
        print('\n%s   n = %d' % (name, s['n']), flush=True)
        print('  magnitude: rms %.4f | mean %.4f | p50 %.4f | p90 %.4f m'
              % (s['rms_m'], s['mean_m'], s['p50_m'], s['p90_m']), flush=True)
        print('  mean lon %.4f m | mean lat %.4f m'
              % (s['mean_lon_m'], s['mean_lat_m']), flush=True)
        sm = s['second_moment']
        cv = s['central_cov']
        print('  RAW SECOND MOMENT E[dd^T] (no velocity prior):', flush=True)
        print('     [[%9.4f, %9.4f], [%9.4f, %9.4f]]'
              % (sm['matrix'][0][0], sm['matrix'][0][1],
                 sm['matrix'][1][0], sm['matrix'][1][1]), flush=True)
        print('     eigenvalues %.4f / %.4f   RATIO %.2f : 1   axis %+.2f deg '
              'from lon' % (sm['eigenvalues'][0], sm['eigenvalues'][1],
                            sm['eig_ratio'],
                            sm['principal_axis_deg_from_lon']), flush=True)
        print('  CENTRAL COVARIANCE Cov[d] (with velocity prior):', flush=True)
        print('     [[%9.4f, %9.4f], [%9.4f, %9.4f]]'
              % (cv['matrix'][0][0], cv['matrix'][0][1],
                 cv['matrix'][1][0], cv['matrix'][1][1]), flush=True)
        print('     eigenvalues %.4f / %.4f   RATIO %.2f : 1   axis %+.2f deg '
              'from lon' % (cv['eigenvalues'][0], cv['eigenvalues'][1],
                            cv['eig_ratio'],
                            cv['principal_axis_deg_from_lon']), flush=True)
        print('     residual rms %.4f m  (vs raw rms %.4f m)'
              % (s['residual_rms_m'], s['rms_m']), flush=True)

    print('\n--- POSE (analytic, over the empirical range distribution) ---',
          flush=True)
    for tier, sig in POSE_TIERS:
        full = pose_rms(sig, sig, mean_r2)
        rot = pose_rms(0.0, sig, mean_r2)
        flat['pose_%s' % tier] = full
        flat['rot_only_%s' % tier] = rot
        detail['pose']['pose_%s' % tier] = {'sigma_t_m': sig,
                                            'sigma_r_deg': sig,
                                            'rms_m': full}
        detail['pose']['rot_only_%s' % tier] = {'sigma_t_m': 0.0,
                                                'sigma_r_deg': sig,
                                                'rms_m': rot}
        print('  %-10s sigma_t %.1f m sigma_r %.1f deg -> rms %.4f m  '
              '(rotation only: %.4f m)' % (tier, sig, sig, full, rot),
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.local_mode:
        det = local_analysis(out)
        pl = os.path.join(args.out_dir, 'local.json')
        with open(pl, 'w') as fh:
            json.dump(det, fh, indent=1, sort_keys=True)
        print('\nwrote %s' % pl, flush=True)
        print('displacements.json and displacements_detail.json were NOT '
              'rewritten', flush=True)
        print('LOCAL DONE', flush=True)
        return 0
    if args.stratify:
        det = stratify(out)
        ps = os.path.join(args.out_dir, 'stratified.json')
        with open(ps, 'w') as fh:
            json.dump(det, fh, indent=1, sort_keys=True)
        print('\nwrote %s' % ps, flush=True)
        print('displacements.json and displacements_detail.json were NOT '
              'rewritten', flush=True)
        print('STRATIFY DONE', flush=True)
        return 0
    p1 = os.path.join(args.out_dir, 'displacements.json')
    p2 = os.path.join(args.out_dir, 'displacements_detail.json')
    with open(p1, 'w') as fh:
        json.dump(flat, fh, indent=1, sort_keys=True)
    with open(p2, 'w') as fh:
        json.dump(detail, fh, indent=1, sort_keys=True)
    print('\nwrote %s' % p1, flush=True)
    print('wrote %s' % p2, flush=True)
    print('\nFLAT TABLE (fault -> rms metres):', flush=True)
    for k in sorted(flat):
        print('  %-20s %.4f' % (k, flat[k]), flush=True)
    print('DISPLACEMENT DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
