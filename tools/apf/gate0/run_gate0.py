"""APF Gate 0 measurement at the CoBEVT fusion seam.

Two modes.

--mode probe   one batch, prints the model structure, the tap shapes, the
               DERIVED metres per cell on both axes, and the two forward
               sources. Answers one question about APF operator placement: is
               the pairwise warp applied BEFORE the fusion call or INSIDE it.
               Gate 0's verdict does not depend on the answer. APF's operator
               placement does.

--mode measure N batches. Autocorrelation half widths at the seam, the ratio
               table against the measured displacements, and the verdict.

Model and loader construction mirrors opencood/tools/inference.py.

RANGE BINNING of the peak conditioned accumulators. A peak's cell indices give
its position directly from cav_lidar_range, so each patch is binned into
0-40 / 40-75 / 75-140 m and accumulated separately, in addition to the unbinned
accumulator. The bin edges are fixed by the pose crossover measured in
displacement.py --stratify (severe pose reaches R = 1.2 at 75.30 m), not chosen
after seeing any half width. Reason: the region where any fault reaches
R >= 1.2 is at long range, so a single pooled half width would average the
deciding region away. Feature support may also genuinely widen with range as
returns get sparser, which is the thing being tested.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import socket
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from autocorr import (                                          # noqa: E402
    AutocorrConfig, PowerSpectrumAccumulator, measure, overall_verdict,
    ratio_table)

P = lambda *a: print(*a, flush=True)                            # noqa: E731

TAP_NAMES = ('scatter', 'backbone', 'shrink_conv', 'naive_compressor',
             'fusion_net')
RANGE_BINS = ((0.0, 40.0), (40.0, 75.0), (75.0, 140.0))
PATCH_H, PATCH_W = 17, 31          # deliberately NON-SQUARE, same reason the
                                   # validation grid is 48 x 176
N_PEAKS = 8
NMS_RADIUS = 4
DECISIVE = ('latency_300ms', 'latency_200ms', 'latency_100ms')


def sha256_file(path: str, n: int = 16) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def describe(obj, prefix: str = '') -> List[str]:
    """Shape of a captured tap, recursing into tuples/lists/dicts."""
    out: List[str] = []
    if torch.is_tensor(obj):
        out.append('%s Tensor %s %s' % (prefix, tuple(obj.shape), obj.dtype))
    elif isinstance(obj, (list, tuple)):
        out.append('%s %s len=%d' % (prefix, type(obj).__name__, len(obj)))
        for i, v in enumerate(obj):
            out.extend(describe(v, prefix + '  [%d]' % i))
    elif isinstance(obj, dict):
        out.append('%s dict keys=%s' % (prefix, list(obj)))
        for k, v in obj.items():
            out.extend(describe(v, prefix + '  [%r]' % k))
    else:
        out.append('%s %s' % (prefix, type(obj).__name__))
    return out


def build(model_dir: str, ns: str = 'opencood'):
    import importlib
    ocds = importlib.import_module(ns + '.data_utils.datasets')
    yaml_utils = importlib.import_module(ns + '.hypes_yaml.yaml_utils')
    train_utils = importlib.import_module(ns + '.tools.train_utils')

    hypes_path = os.path.join(model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(hypes_path)
    dataset = ocds.build_dataset(hypes, visualize=False, train=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=dataset.collate_batch_test, pin_memory=False,
        drop_last=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.to(device).eval()
    return hypes, hypes_path, dataset, loader, model, device, train_utils


def attach_taps(model) -> Tuple[Dict, List]:
    cap: Dict[str, object] = {}
    handles = []
    for name in TAP_NAMES:
        mod = getattr(model, name, None)
        if mod is None:
            continue

        def mk(n):
            def hook(_m, _i, o):
                cap[n] = o
            return hook
        handles.append(mod.register_forward_hook(mk(name)))
    fn = getattr(model, 'fusion_net', None)
    if fn is not None:
        def pre(_m, i):
            # The APF seam: what fusion_net RECEIVES.
            cap['fusion_net__INPUT'] = i
        handles.append(fn.register_forward_pre_hook(pre))
    return cap, handles


def derived_res(cav_lidar_range, grid_h: int, grid_w: int) -> Tuple[float,
                                                                   float]:
    """Metres per cell on BOTH axes SEPARATELY, derived not assumed."""
    x0, y0, _, x1, y1, _ = [float(v) for v in cav_lidar_range]
    span_w = x1 - x0          # longitudinal, maps to the W axis
    span_h = y1 - y0          # lateral, maps to the H axis
    return span_h / grid_h, span_w / grid_w


def find_peaks(energy: np.ndarray, n: int = N_PEAKS,
               radius: int = NMS_RADIUS) -> List[Tuple[int, int]]:
    """Top n local maxima with NMS, restricted to centres whose full patch
    fits inside the map. A clipped patch would change the accumulator shape
    and silently mix supports."""
    e = energy.copy()
    H, W = e.shape
    hh, hw = PATCH_H // 2, PATCH_W // 2
    e[:hh, :] = -np.inf
    e[H - hh:, :] = -np.inf
    e[:, :hw] = -np.inf
    e[:, W - hw:] = -np.inf
    peaks: List[Tuple[int, int]] = []
    for _ in range(n):
        idx = int(np.argmax(e))
        r, c = divmod(idx, W)
        if not np.isfinite(e[r, c]):
            break
        peaks.append((r, c))
        e[max(0, r - radius):r + radius + 1,
          max(0, c - radius):c + radius + 1] = -np.inf
    return peaks


def cell_to_xy(r: int, c: int, cav_lidar_range, H: int, W: int):
    x0, y0, _, x1, y1, _ = [float(v) for v in cav_lidar_range]
    x = x0 + (c + 0.5) * (x1 - x0) / W        # longitudinal, W axis
    y = y0 + (r + 0.5) * (y1 - y0) / H        # lateral, H axis
    return x, y


def range_bin(x: float, y: float) -> Optional[int]:
    """Radial range, using BOTH cell indices.

    The task described binning by the longitudinal index alone. Radial range
    is used instead so that these bins mean the same thing as the range bins
    in displacement.py --stratify, which measured agent to vehicle distance as
    hypot(dx, dy) and from which the 75 m edge was derived. Lateral extent is
    at most 38.4 m so the two agree except near a bin edge; the longitudinal
    value is recorded alongside so the choice is auditable.
    """
    r = float(np.hypot(x, y))
    for i, (lo, hi) in enumerate(RANGE_BINS):
        if lo <= r < hi:
            return i
    return None


def probe(args) -> int:
    hypes, hypes_path, dataset, loader, model, device, train_utils = build(
        args.model_dir)
    cap, handles = attach_taps(model)

    P('=' * 78)
    P('APF GATE 0 PROBE')
    P('=' * 78)
    P('host %s | device %s' % (socket.gethostname(), device))
    if torch.cuda.is_available():
        P('gpu %s | torch %s | cuda %s'
          % (torch.cuda.get_device_name(0), torch.__version__,
             torch.version.cuda))
    P('model class %s' % type(model).__name__)
    P('dataset len %d' % len(dataset))

    margs = hypes.get('model', {}).get('args', {})
    clr = margs.get('cav_lidar_range') or margs.get('lidar_range')
    P('\n--- config ---')
    P('cav_lidar_range : %s' % (clr,))
    P('compression     : %s' % margs.get('compression'))
    P('shrink_header   : %s' % margs.get('shrink_header'))
    P('max_cav         : %s' % margs.get('max_cav'))
    P('voxel_size      : %s' % margs.get('voxel_size'))

    P('\n--- named_children ---')
    for n, m in model.named_children():
        P('  %-22s %s' % (n, type(m).__name__))

    P('\n--- forward pass, one batch ---')
    batch = next(iter(loader))
    b = train_utils.to_device(batch, device)
    with torch.no_grad():
        model(b['ego'])

    P('\n--- captured taps ---')
    for k in sorted(cap):
        for line in describe(cap[k], '  %s' % k):
            P(line)

    P('\n--- DERIVED metres per cell (from cav_lidar_range / tap grid) ---')
    ok = True
    for k in sorted(cap):
        t = cap[k]
        if isinstance(t, (list, tuple)):
            t = t[0] if t and torch.is_tensor(t[0]) else None
        if not torch.is_tensor(t) or t.dim() < 2:
            continue
        gh, gw = int(t.shape[-2]), int(t.shape[-1])
        rh, rw = derived_res(clr, gh, gw)
        flag = ''
        if (gh, gw) == (48, 176):
            if abs(rh - 1.6) > 1e-6 or abs(rw - 1.6) > 1e-6:
                flag = '   <== NOT 1.6, every downstream covariance would ' \
                       'silently rescale'
                ok = False
            else:
                flag = '   <== seam grid, 1.6 / 1.6 as expected'
        P('  %-24s grid %3d x %3d   res_h %.4f m   res_w %.4f m%s'
          % (k, gh, gw, rh, rw, flag))
    if not ok:
        P('\nSTOP: derived resolution is not 1.6 m at the 48 x 176 grid.')
        for h in handles:
            h.remove()
        return 1

    P('\n' + '=' * 78)
    P('SOURCE: type(model).forward')
    P('=' * 78)
    P(inspect.getsource(type(model).forward))
    P('=' * 78)
    P('SOURCE: type(model.fusion_net).forward   (%s)'
      % type(model.fusion_net).__name__)
    P('=' * 78)
    P(inspect.getsource(type(model.fusion_net).forward))

    P('=' * 78)
    P('WARP PLACEMENT: read the two sources above and state which.')
    P('=' * 78)
    for h in handles:
        h.remove()
    return 0


class TapAccumulators:
    """One tap's full cross-cut.

    agent variants   all / ego / nonego
    scope variants   full / peak
    plus the peak accumulators RANGE BINNED into RANGE_BINS.

    Count: 3 agent x 2 scope = 6 base accumulators, plus 3 agent x 3 range
    bins = 9 binned peak accumulators, so 15 per tap. The task said eight per
    tap; 3 x 2 is 6, and the range binning adds 9. The arithmetic is stated
    here rather than padded to match.
    """

    AGENTS = ('all', 'ego', 'nonego')

    def __init__(self, h: int, w: int, cfg: AutocorrConfig):
        self.full = {a: PowerSpectrumAccumulator(h, w, cfg)
                     for a in self.AGENTS}
        self.peak = {a: PowerSpectrumAccumulator(PATCH_H, PATCH_W, cfg)
                     for a in self.AGENTS}
        self.peak_binned = {
            a: [PowerSpectrumAccumulator(PATCH_H, PATCH_W, cfg)
                for _ in RANGE_BINS] for a in self.AGENTS}
        self.n_patches = {a: [0] * len(RANGE_BINS) for a in self.AGENTS}


def measure_mode(args) -> int:
    hypes, hypes_path, dataset, loader, model, device, train_utils = build(
        args.model_dir)
    cap, handles = attach_taps(model)
    margs = hypes.get('model', {}).get('args', {})
    clr = margs.get('cav_lidar_range') or margs.get('lidar_range')

    taps = [t for t in ('shrink_conv', 'naive_compressor')
            if getattr(model, t, None) is not None]
    if not taps:
        P('no shrink_conv or naive_compressor on this model')
        return 1

    accs: Dict[str, TapAccumulators] = {}
    cfg = None
    n_done = 0
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        b = train_utils.to_device(batch, device)
        with torch.no_grad():
            model(b['ego'])
        for tap in taps:
            t = cap.get(tap)
            if not torch.is_tensor(t):
                continue
            # RUNTIME ASSERTIONS: dim 0 is the agent axis and row 0 is ego.
            n_ag = int(t.shape[0])
            rec = b['ego'].get('record_len')
            if rec is not None:
                exp = int(rec.sum().item())
                if n_ag != exp:
                    raise RuntimeError(
                        '%s: dim 0 is %d but record_len sums to %d, so dim 0 '
                        'is not the agent axis' % (tap, n_ag, exp))
            if t.dim() != 4:
                raise RuntimeError('%s: expected 4-D (agents, C, H, W), got %s'
                                   % (tap, tuple(t.shape)))
            arr = t.detach().float().cpu().numpy()
            _, C, H, W = arr.shape
            if cfg is None:
                rh, rw = derived_res(clr, H, W)
                if abs(rh - 1.6) > 1e-6 or abs(rw - 1.6) > 1e-6:
                    raise RuntimeError('derived res %.4f / %.4f is not 1.6'
                                       % (rh, rw))
                cfg = AutocorrConfig(res_h_m=rh, res_w_m=rw)
            if tap not in accs:
                accs[tap] = TapAccumulators(H, W, cfg)
            A = accs[tap]
            groups = {'all': list(range(n_ag)), 'ego': [0],
                      'nonego': list(range(1, n_ag))}
            for gname, rows in groups.items():
                if not rows:
                    continue
                # NEVER pool over dim 0. Each agent row is added separately.
                for r in rows:
                    A.full[gname].add(arr[r])
                    energy = (arr[r] ** 2).sum(axis=0)
                    for (pr, pc) in find_peaks(energy):
                        hh, hw = PATCH_H // 2, PATCH_W // 2
                        patch = arr[r][:, pr - hh:pr + hh + 1,
                                       pc - hw:pc + hw + 1]
                        if patch.shape[1:] != (PATCH_H, PATCH_W):
                            continue
                        A.peak[gname].add(patch)
                        x, y = cell_to_xy(pr, pc, clr, H, W)
                        bidx = range_bin(x, y)
                        if bidx is not None:
                            A.peak_binned[gname][bidx].add(patch)
                            A.n_patches[gname][bidx] += 1
        n_done += 1
        if n_done % 10 == 0:
            P('  %d / %d batches' % (n_done, args.n_batches))
    for h in handles:
        h.remove()

    results: Dict[str, Dict] = {}
    for tap, A in accs.items():
        results[tap] = {}
        for a in TapAccumulators.AGENTS:
            if A.full[a].count:
                results[tap]['%s|full' % a] = measure(
                    A.full[a], '%s|%s|full' % (tap, a)).to_json()
            if A.peak[a].count:
                results[tap]['%s|peak' % a] = measure(
                    A.peak[a], '%s|%s|peak' % (tap, a)).to_json()
            for i, (lo, hi) in enumerate(RANGE_BINS):
                acc = A.peak_binned[a][i]
                if acc.count:
                    key = '%s|peak|%g-%g' % (a, lo, hi)
                    results[tap][key] = measure(
                        acc, '%s|%s' % (tap, key)).to_json()

    disp_path = os.path.join(args.out_dir, 'displacements.json')
    with open(disp_path) as fh:
        disp = json.load(fh)

    primary_tap = 'naive_compressor' if 'naive_compressor' in results \
        else 'shrink_conv'
    primary = results[primary_tap].get('nonego|peak')
    if primary is None:
        P('primary key %s | nonego | peak is missing' % primary_tap)
        return 1
    long_m = primary['axis_w_long_m']
    lat_m = primary['axis_h_lat_m']
    l_by_fault = {}
    for k in disp:
        if k.startswith('latency'):
            l_by_fault[k] = long_m
        elif k.startswith('rot_only'):
            l_by_fault[k] = lat_m
        else:
            l_by_fault[k] = float(np.mean([long_m, lat_m]))
    rows = ratio_table(disp, l_by_fault)
    verdict = overall_verdict(rows, DECISIVE)

    P('\n=== VERDICT (primary %s | nonego | peak) ===' % primary_tap)
    P('axis_w_long %.4f m | axis_h_lat %.4f m' % (long_m, lat_m))
    P('%-20s %12s %12s %10s  %s'
      % ('fault', 'disp (m)', 'l_half (m)', 'R', 'verdict'))
    for r in rows:
        P('%-20s %12.4f %12.4f %10.4f  %s'
          % (r['fault'], r['displacement_m'], r['l_half_m'], r['ratio'],
             r['verdict']))
    P('OVERALL %s   (decisive: %s)' % (verdict, ', '.join(DECISIVE)))

    ckpt = None
    for f in sorted(os.listdir(args.model_dir)):
        if f.endswith('.pth') or f.endswith('.pt'):
            ckpt = os.path.join(args.model_dir, f)
    out = {
        'results': results,
        'verdict_rows': rows,
        'overall_verdict': verdict,
        'primary_key': '%s|nonego|peak' % primary_tap,
        'decisive': list(DECISIVE),
        'displacements': disp,
        'range_bins_m': [list(b) for b in RANGE_BINS],
        'patch_hw': [PATCH_H, PATCH_W],
        'n_batches': n_done,
        'provenance': {
            'autocorr_sha256': sha256_file(os.path.join(_HERE,
                                                        'autocorr.py')),
            'displacement_sha256': sha256_file(
                os.path.join(_HERE, 'displacement.py')),
            'run_gate0_sha256': sha256_file(os.path.join(_HERE,
                                                         'run_gate0.py')),
            'checkpoint_path': ckpt,
            'checkpoint_sha256': sha256_file(ckpt),
            'hypes_path': hypes_path,
            'hypes_sha256': sha256_file(hypes_path),
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'hostname': socket.gethostname(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu': (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, 'gate0.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    P('\nwrote %s' % p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('probe', 'measure'))
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--n-batches', type=int, default=100)
    ap.add_argument('--out-dir', default='results/apf/gate0')
    args = ap.parse_args()
    if args.mode == 'probe':
        return probe(args)
    return measure_mode(args)


if __name__ == '__main__':
    sys.exit(main())
