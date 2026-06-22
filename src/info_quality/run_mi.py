#!/usr/bin/env python3
"""
run_mi.py
---------
Command-line driver for information-quality estimation.

Loads a features file, estimates I(Z; Y) for every representation in it with one
or more estimators, prints a comparison table and (optionally) a fusion-gain
summary, then saves the numbers and an optional plot.

The features file is produced once (model-specific, via FeatureCollector) and
read many times here (model-agnostic). Two formats are accepted:

  * .npz : as written by FeatureSet.save (one array per representation + 'Y').
  * .pkl : a pickled dict, e.g. the notebook's {'Z_camera', 'Z_lidar',
           'Z_fused', 'Y'}. Every array-valued key except the target is treated
           as a representation.

Examples
--------
    # Both estimators on every representation, write a plot:
    python -m src.info_quality.run_mi features.npz --plot mi.png

    # InfoNCE only, and report fusion gain of Z_fused over the two modalities:
    python -m src.info_quality.run_mi feats.pkl --estimators infonce \\
        --fused Z_fused --unimodal Z_camera Z_lidar
"""

from __future__ import annotations

import argparse
import pickle
from typing import Dict, List

import numpy as np

from .estimators import InfoNCEEstimator, SMILEEstimator
from .reporting import format_table, fusion_summary, plot_comparison


def _load_features(path: str, target_key: str) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load representations and target Y from a .npz or pickled-dict file."""
    if path.endswith('.npz'):
        data = np.load(path)
        store = {k: data[k] for k in data.files}
    else:
        with open(path, 'rb') as fh:
            store = pickle.load(fh)

    if target_key not in store:
        raise KeyError(
            f"Target key '{target_key}' not in {path}. Available: {list(store.keys())}")
    Y = np.asarray(store[target_key])
    reps = {k: np.asarray(v) for k, v in store.items()
            if k != target_key and np.asarray(v).ndim == 2}
    if not reps:
        raise ValueError(f'No 2D representation arrays found in {path}.')
    return reps, Y


def _build_estimators(names: List[str], args) -> Dict[str, object]:
    built: Dict[str, object] = {}
    for name in names:
        if name == 'infonce':
            built['infonce'] = InfoNCEEstimator(
                temperature=args.temperature, epochs=args.epochs_infonce,
                batch_size=args.batch_size, seed=args.seed)
        elif name == 'smile':
            built['smile'] = SMILEEstimator(
                clip=args.clip, epochs=args.epochs_smile,
                batch_size=args.batch_size, seed=args.seed)
        else:
            raise ValueError(f"Unknown estimator '{name}'. Choose from infonce, smile.")
    return built


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('features', help='path to a .npz or pickled-dict features file')
    p.add_argument('--target-key', default='Y', help="key of the target array (default 'Y')")
    p.add_argument('--estimators', nargs='+', default=['infonce', 'smile'],
                   help='estimators to run (infonce, smile)')
    p.add_argument('--pca-dims', type=int, default=32,
                   help='reduce each representation to this many PCA dims (0 = off)')
    p.add_argument('--epochs-infonce', type=int, default=100)
    p.add_argument('--epochs-smile', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fused', default=None, help='representation key of the fused feature')
    p.add_argument('--unimodal', nargs='+', default=None,
                   help='unimodal representation keys for the fusion-gain summary')
    p.add_argument('--out', default='mi_results.pkl', help='where to save the results dict')
    p.add_argument('--plot', default=None, help='optional PNG path for a comparison plot')
    args = p.parse_args(argv)

    pca_dims = args.pca_dims if args.pca_dims and args.pca_dims > 0 else None
    reps, Y = _load_features(args.features, args.target_key)
    rep_names = list(reps.keys())
    print(f'Loaded {len(reps)} representation(s) from {args.features}: '
          f'{", ".join(f"{k}{v.shape}" for k, v in reps.items())}')
    print(f'Target {args.target_key}: {Y.shape} | PCA dims: {pca_dims} | seed: {args.seed}')

    estimators = _build_estimators(args.estimators, args)
    results: Dict[str, Dict[str, float]] = {}

    for est_name, est in estimators.items():
        print(f'\n=== {est_name} ===')
        results[est_name] = {}
        for rep in rep_names:
            res = est.estimate(reps[rep], Y, pca_dims=pca_dims)
            results[est_name][rep] = res.mi_nats
            print(f'  I({rep}; {args.target_key}) = {res.mi_nats:.4f} nats')

    print('\n' + format_table(results, rep_names))
    if args.fused and args.unimodal:
        print('\n' + fusion_summary(results, args.fused, args.unimodal))

    with open(args.out, 'wb') as fh:
        pickle.dump(results, fh)
    print(f'\nSaved results -> {args.out}')

    if args.plot:
        plot_comparison(results, args.plot, rep_names)
        print(f'Saved plot    -> {args.plot}')


if __name__ == '__main__':
    main()
