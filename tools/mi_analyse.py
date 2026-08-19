"""Estimate I(Z; Y) per condition and report the fusion gain.

ENFORCES THE FIXED-Y PROTOCOL: Y is taken from the CLEAN condition's file
and reused for every other condition. Each condition's own Y is loaded only
to be CHECKED (shape match) and then discarded, and any condition whose own
Y differs from clean is REPORTED -- that difference is exactly the GT-union
confound this protocol exists to avoid, so it is measured rather than
hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/nanjaiyalathaa/Fault-Injectors")

from src.info_quality.estimators import (                     # noqa: E402
    InfoNCEEstimator, SMILEEstimator)
from src.info_quality.feature_extraction import FeatureSet    # noqa: E402

P = lambda *a: print(*a, flush=True)                          # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--conditions', nargs='+',
                    default=['clean', 'pose_error', 'agent_drop'])
    ap.add_argument('--ego_tap', default='Z_ego_compressed')
    ap.add_argument('--fused_tap', default='Z_fused')
    ap.add_argument('--pca_dims', type=int, default=32)
    ap.add_argument('--epochs_infonce', type=int, default=100)
    ap.add_argument('--epochs_smile', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--holdout', type=float, default=0.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sets = {}
    for c in args.conditions:
        p = os.path.join(args.dir, 'features_%s.npz' % c)
        if not os.path.exists(p):
            raise SystemExit('missing %s' % p)
        sets[c] = FeatureSet.load(p)
        P('%-12s %s' % (c, sets[c]))

    if 'clean' not in sets:
        raise SystemExit('clean condition is required: it supplies Y')
    Y = sets['clean'].Y
    P('\n=== FIXED-Y PROTOCOL ===')
    P('Y taken from CLEAN, shape %s, reused verbatim for every condition.'
      % (Y.shape,))
    for c in args.conditions:
        own = sets[c].Y
        if own.shape != Y.shape:
            raise SystemExit('%s: Y shape %s != clean %s -- frame counts '
                             'differ, rows cannot be aligned'
                             % (c, own.shape, Y.shape))
        d = float(np.abs(own - Y).max())
        same = 'IDENTICAL' if d == 0.0 else 'DIFFERS max|d|=%.4g' % d
        P('  own-Y vs clean-Y for %-12s %s%s' % (
            c, same,
            '   <-- this is the GT-union confound the fixed-Y protocol '
            'avoids' if d > 0 else ''))

    ests = {
        'infonce': InfoNCEEstimator(epochs=args.epochs_infonce,
                                    batch_size=args.batch_size,
                                    seed=args.seed, holdout=args.holdout),
        'smile': SMILEEstimator(epochs=args.epochs_smile,
                                batch_size=args.batch_size,
                                seed=args.seed, holdout=args.holdout),
    }

    P('\n=== PROTOCOL (identical across every condition) ===')
    P('  pca_dims=%s  epochs=%d/%d  batch=%d  seed=%d  holdout=%.2f  N=%d'
      % (args.pca_dims, args.epochs_infonce, args.epochs_smile,
         args.batch_size, args.seed, args.holdout, Y.shape[0]))
    if args.holdout <= 0:
        P('  IN-SAMPLE: relative comparison only; absolute values are '
          'upward-biased and must not be quoted.')

    out = {}
    for est_name, est in ests.items():
        P('\n=== %s ===' % est_name.upper())
        P('%-12s %12s %12s %12s' % ('condition', 'I(ego;Y)', 'I(fused;Y)',
                                    'gain'))
        out[est_name] = {}
        for c in args.conditions:
            fs = sets[c]
            if args.ego_tap not in fs.features:
                raise SystemExit('%s: tap %s missing (have %s)'
                                 % (c, args.ego_tap, list(fs.features)))
            mi_e = est.estimate(fs.features[args.ego_tap], Y,
                                pca_dims=args.pca_dims).mi_nats
            mi_f = est.estimate(fs.features[args.fused_tap], Y,
                                pca_dims=args.pca_dims).mi_nats
            out[est_name][c] = {'ego': mi_e, 'fused': mi_f,
                                'gain': mi_f - mi_e}
            P('%-12s %12.4f %12.4f %+12.4f' % (c, mi_e, mi_f, mi_f - mi_e))

    P('\n=== FUSION GAIN, clean-relative ===')
    for est_name in out:
        base = out[est_name]['clean']['gain']
        P('  %s: clean %+.4f' % (est_name, base))
        for c in args.conditions:
            if c == 'clean':
                continue
            g = out[est_name][c]['gain']
            P('     %-12s %+.4f   (%+.4f vs clean, %s)'
              % (c, g, g - base,
                 'still positive' if g > 0 else 'NEGATIVE'))

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'protocol': vars(args), 'results': out}, fh, indent=1)
        P('\nsaved %s' % args.out)
    P('MI ANALYSE DONE')


if __name__ == '__main__':
    main()
