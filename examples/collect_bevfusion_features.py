"""
collect_bevfusion_features.py
-----------------------------
Worked example: collect camera / LiDAR / fused features from a BEVFusion model
using the architecture-agnostic FeatureCollector, then save them for the
mutual-information estimators.

This is the migration of the original notebook. The notebook hard-wired three
BEVFusion module paths inside the feature loop:

    encoders['camera'].neck      -> camera branch features
    encoders['lidar'].backbone   -> LiDAR branch features
    fuser                        -> fused BEV features

Here those paths are just entries in a `taps` dict handed to a generic
collector. Nothing about the InfoNCE / SMILE estimators is BEVFusion-specific;
only the few lines below that name modules and read targets out of a batch are.
Swap them for any other detector and the rest of the pipeline is unchanged.

Run (inside an mmdet3d / BEVFusion environment, with a built model + dataloader):

    python examples/collect_bevfusion_features.py \
        --config path/to/bevfusion.yaml \
        --checkpoint path/to/bevfusion.pth \
        --out features_bevfusion.npz \
        --n-samples 500

The script intentionally keeps the mmdet3d-specific construction in one place
(`build_model_and_loader`) so it is obvious what you would replace for a
different framework. The saved .npz feeds straight into:

    python -m src.info_quality.run_mi --input features_bevfusion.npz --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.info_quality.feature_extraction import FeatureCollector  # noqa: E402


# ── nuScenes target vector ──────────────────────────────────────────────────
# The notebook used a 10-dim per-sample label summarising the scene. The exact
# choice does not matter to the estimators as long as it is a fixed-length,
# row-aligned numeric vector that carries the task-relevant signal. Here we use
# a simple class-count histogram over the 10 nuScenes detection classes; replace
# with whatever target your study defines (e.g. a foreground/background ratio,
# an object-count vector, or a pooled GT-box embedding).
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]


def nuscenes_label_fn(batch) -> np.ndarray:
    """
    Build a (10,) class-count vector from a mmdet3d batch.

    This reaches into mmdet3d's data containers, which is exactly the kind of
    framework-specific code we want isolated in the adapter. Adjust the key path
    if your pipeline names things differently.
    """
    # NO try/except HERE, DELIBERATELY. This function used to swallow every
    # exception and return a zero vector "so collection does not crash". That
    # made a wrong key path indistinguishable from a legitimately empty scene:
    # Y came out constant, both estimators reported MI ~= 0 for EVERY
    # condition, and the result looked like a real null. A wrong key path must
    # fail on the first sample, loudly, not 500 samples later as a plausible
    # finding. FeatureCollector.collect() additionally asserts that Y is not
    # constant, so the guard survives even if someone re-adds a fallback here.
    data_samples = batch['data_samples']
    sample = (data_samples[0] if isinstance(data_samples, (list, tuple))
              else data_samples)
    labels = np.asarray(sample.gt_instances_3d.labels_3d.cpu().numpy())

    counts = np.zeros(len(NUSCENES_CLASSES), dtype=np.float32)
    for c in labels:
        if 0 <= int(c) < len(counts):
            counts[int(c)] += 1.0
    return counts


def bevfusion_forward(model, batch):
    """
    Run one BEVFusion forward pass. The return value is ignored; the hooks on
    the tapped modules capture the features. We only need a forward far enough
    that camera neck, LiDAR backbone and fuser have all executed.
    """
    import torch
    with torch.no_grad():
        # mmdet3d test-step signature; adapt to your model's API.
        return model.test_step(batch)


def resolve_module(model, dotted: str):
    """
    Resolve a dotted/bracketed path like "encoders['camera'].neck" against a
    model instance, returning the nn.Module at that path.

    Supports attribute access (.neck) and dict-style indexing (['camera']).
    """
    import re

    node = model
    # Split into tokens: attribute names and ['key'] indexers.
    tokens = re.findall(r"\['([^']+)'\]|\[\"([^\"]+)\"\]|([A-Za-z_]\w*)", dotted)
    for sq, dq, attr in tokens:
        key = sq or dq
        if key:
            node = node[key]
        else:
            node = getattr(node, attr)
    return node


def build_model_and_loader(config: str, checkpoint: str):
    """
    Construct a BEVFusion model and a batch-size-1 dataloader.

    This is the only block that assumes mmdet3d. Replace it wholesale for a
    different framework; everything else in this file stays the same.
    """
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS, DATASETS
        from mmengine.runner import load_checkpoint
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise SystemExit(
            'This example needs an mmdet3d / BEVFusion environment. '
            f'Import failed: {exc}'
        )

    cfg = Config.fromfile(config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    val_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=getattr(dataset, 'collate_fn', None))
    return model, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='BEVFusion config file.')
    parser.add_argument('--checkpoint', required=True, help='BEVFusion checkpoint.')
    parser.add_argument('--out', default='features_bevfusion.npz',
                        help='Where to save the collected FeatureSet (.npz).')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Stop after this many samples (default: whole loader).')
    parser.add_argument('--camera-tap', default="encoders['camera'].neck",
                        help='Dotted path to the camera-branch module to tap.')
    parser.add_argument('--lidar-tap', default="encoders['lidar'].backbone",
                        help='Dotted path to the LiDAR-branch module to tap.')
    parser.add_argument('--fuser-tap', default='fuser',
                        help='Dotted path to the fusion module to tap.')
    args = parser.parse_args()

    model, loader = build_model_and_loader(args.config, args.checkpoint)

    # Resolve the three notebook hook points against the live model. The names
    # become the representation names in the MI report (Z_camera etc.).
    taps = {
        'Z_camera': resolve_module(model, args.camera_tap),
        'Z_lidar': resolve_module(model, args.lidar_tap),
        'Z_fused': resolve_module(model, args.fuser_tap),
    }

    collector = FeatureCollector(model, taps=taps)
    feature_set = collector.collect(
        loader,
        forward_fn=bevfusion_forward,
        label_fn=nuscenes_label_fn,
        n_samples=args.n_samples,
        ignore_forward_errors=True,  # the detection head may error after fusion
    )

    feature_set.save(args.out)
    print(f'\nSaved features to {args.out}')
    for name, arr in feature_set.features.items():
        print(f'  {name:10s} {arr.shape}')
    print(f'  {"Y":10s} {feature_set.Y.shape}')
    print('\nNext:')
    print(f'  python -m src.info_quality.run_mi --input {args.out} --plot')


if __name__ == '__main__':
    main()
