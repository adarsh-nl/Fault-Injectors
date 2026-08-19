"""
feature_extraction.py
---------------------
Architecture-agnostic feature collection via forward hooks.

The mutual information estimators consume plain (N, d) arrays and know nothing
about any model. Getting those arrays out of a network is the *only*
model-specific step, and this module makes it generic: you say which modules to
tap and how to pool their outputs, and you get back one row per sample.

The original notebook hard-wired three BEVFusion module paths
(encoders['camera'].neck, encoders['lidar'].backbone, fuser). Here those paths
are just arguments. Nothing below mentions BEVFusion, nuScenes or Griffin.

Contract
--------
You provide:
  * model      : an nn.Module already on the right device, in eval mode.
  * taps       : dict name -> nn.Module. A forward hook is attached to each;
                 its output is pooled to a single (d,) vector per forward pass.
  * loader     : an iterable of batches.
  * forward_fn : forward_fn(model, batch) runs one forward pass. You write this
                 because calling conventions differ across frameworks
                 (model(**batch), model(batch['img']), ...).
  * label_fn   : label_fn(batch) -> (d_y,) array, the target row for the sample.

You get back a FeatureSet: one (N, d) array per tap name, plus Y of shape
(N, d_y), all row-aligned and ready for the estimators.

Pooling
-------
Network features are typically (V, C, H, W), (B, C, H, W), (B, C, L) or (B, C).
`global_pool` collapses everything except the channel axis to a single (C,)
vector by averaging, which is the modality-agnostic default. Pass your own
callable to `taps_pool` if you need something else (e.g. max-pool, CLS token).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    _TORCH = False


# ── Pooling ─────────────────────────────────────────────────────────────────

def global_pool(t: "torch.Tensor") -> "torch.Tensor":
    """
    Collapse a feature tensor to a single channel vector by averaging.

    Handles the common ranks:
        4D (V/B, C, H, W) -> mean over dims (0, 2, 3) -> (C,)
        3D (B, C, L)      -> mean over dims (0, 2)     -> (C,)
        2D (B, C)         -> mean over dim 0           -> (C,)
        1D (C,)           -> returned as is.

    Returns
    -------
    1D tensor (C,), detached, on CPU, float32.
    """
    if t.dim() == 4:
        t = t.mean(dim=(0, 2, 3))
    elif t.dim() == 3:
        t = t.mean(dim=(0, 2))
    elif t.dim() == 2:
        t = t.mean(dim=0)
    elif t.dim() != 1:
        raise ValueError(f'global_pool cannot handle a {t.dim()}D tensor.')
    return t.detach().to('cpu', torch.float32)


# ── Result container ─────────────────────────────────────────────────────────

class FeatureSet:
    """
    Collected features and aligned targets.

    Attributes
    ----------
    features : dict name -> np.ndarray (N, d). One entry per tap.
    Y        : np.ndarray (N, d_y) targets.
    """

    def __init__(self, features: Dict[str, np.ndarray], Y: np.ndarray):
        self.features = features
        self.Y = Y

    def __repr__(self) -> str:
        shapes = ', '.join(f'{k}={v.shape}' for k, v in self.features.items())
        return f'FeatureSet({shapes}, Y={self.Y.shape})'

    def save(self, path: str) -> None:
        """Save to a compressed .npz (keys: each feature name, plus 'Y')."""
        np.savez_compressed(path, Y=self.Y, **self.features)

    @classmethod
    def load(cls, path: str) -> 'FeatureSet':
        """Load a FeatureSet from a .npz written by `save`."""
        data = np.load(path)
        feats = {k: data[k] for k in data.files if k != 'Y'}
        return cls(feats, data['Y'])


# ── Collector ────────────────────────────────────────────────────────────────

class FeatureCollector:
    """
    Attach forward hooks to named modules and collect pooled features.

    Parameters
    ----------
    model     : nn.Module (already on device, in eval mode).
    taps      : dict name -> nn.Module to hook.
    taps_pool : pooling callable applied to each tapped output (default
                `global_pool`). Either one callable for all taps, or a dict
                name -> callable for per-tap control.

    Usage
    -----
        collector = FeatureCollector(model, {'feat': model.backbone})
        fs = collector.collect(loader, forward_fn, label_fn, n_samples=200)
        collector.remove()
        # fs.features['feat'] : (N, d),  fs.Y : (N, d_y)
    """

    def __init__(self, model, taps: Dict[str, object],
                 taps_pool: Optional[object] = None):
        if not _TORCH:
            raise ImportError('FeatureCollector requires PyTorch.')
        self.model = model
        self._buffers: Dict[str, List["torch.Tensor"]] = {}
        self._handles: List[object] = []
        if taps_pool is None:
            pools = {name: global_pool for name in taps}
        elif isinstance(taps_pool, dict):
            pools = taps_pool
        else:
            pools = {name: taps_pool for name in taps}

        for name, module in taps.items():
            handle = module.register_forward_hook(self._make_hook(name, pools[name]))
            self._handles.append(handle)

    def _make_hook(self, name: str, pool: Callable):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            if not isinstance(t, torch.Tensor):
                return
            self._buffers.setdefault(name, []).append(pool(t))
        return hook

    def _drain(self, name: str) -> Optional[np.ndarray]:
        if name not in self._buffers or not self._buffers[name]:
            return None
        return torch.stack(self._buffers[name], dim=0).numpy()

    def collect(self, loader: Iterable, forward_fn: Callable,
                label_fn: Callable, n_samples: Optional[int] = None,
                ignore_forward_errors: bool = False,
                progress: bool = True) -> FeatureSet:
        """
        Run inference over the loader and collect pooled features + targets.

        Parameters
        ----------
        model output is captured by the hooks; the forward return value is
        ignored. One row is recorded per batch, so use a loader with
        samples_per_gpu / batch_size = 1 for per-sample features.

        forward_fn : forward_fn(model, batch) -> runs one forward pass.
        label_fn   : label_fn(batch) -> (d_y,) target array for this sample.
        n_samples  : stop after this many batches (None = whole loader).
        ignore_forward_errors : if True, swallow exceptions raised *after* the
                       tapped modules have fired. Useful when a downstream head
                       errors but the features you need are already captured.
                       Off by default so real failures are not hidden.
        progress   : show a tqdm bar if tqdm is installed.

        Returns
        -------
        FeatureSet with row-aligned features and Y.
        """
        iterator = loader
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(loader, total=n_samples, desc='Collecting features')
            except ImportError:
                pass

        labels: List[np.ndarray] = []
        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                if n_samples is not None and idx >= n_samples:
                    break
                n_before = {k: len(v) for k, v in self._buffers.items()}
                try:
                    forward_fn(self.model, batch)
                except Exception:
                    if not ignore_forward_errors:
                        raise
                labels.append(np.asarray(label_fn(batch), dtype=np.float32).ravel())

                # Keep features and labels aligned: every tap must have advanced
                # by exactly one row this step, else the run is inconsistent.
                for name in self._buffers:
                    grew = len(self._buffers[name]) - n_before.get(name, 0)
                    if grew not in (0, 1):
                        raise RuntimeError(
                            f"Tap '{name}' produced {grew} rows in one step; "
                            'expected 0 or 1. Check that the tapped module fires '
                            'once per forward pass.')

        Y = np.stack(labels, axis=0)
        # ── DEGENERATE-TARGET GUARD ─────────────────────────────────────
        # A label_fn that silently returns a constant (the classic case is a
        # bare `except: return np.zeros(...)` fallback swallowing a wrong key
        # path) produces a Y with no variance. `standardise` then divides by
        # sd + 1e-8, every column collapses to ~0, and BOTH estimators report
        # MI ~= 0 for EVERY condition -- a plausible-looking null result that
        # is really an instrumentation failure. It is the worst failure mode
        # available here, because nothing about the output looks wrong.
        # Assert positively, in the COLLECTOR, so no adapter can reintroduce
        # it downstream.
        if Y.ndim != 2:
            raise RuntimeError(
                'label_fn must return a fixed-length 1-D vector per sample; '
                'stacked Y has shape %s.' % (Y.shape,))
        sd = Y.std(axis=0)
        n_live = int((sd > 0).sum())
        if not np.isfinite(Y).all():
            raise RuntimeError('Y contains non-finite values (%d of %d).'
                               % (int((~np.isfinite(Y)).sum()), Y.size))
        if float(sd.max()) <= 0.0:
            raise RuntimeError(
                'DEGENERATE TARGET: Y is CONSTANT across all %d samples '
                '(max per-column std = %.3e, %d/%d columns non-degenerate). '
                'Mutual information against a constant target is 0 by '
                'construction, so every condition would report MI ~= 0 and '
                'the comparison would be meaningless. This almost always '
                'means label_fn is failing silently -- check its key path. '
                'The run is aborted rather than producing a plausible null.'
                % (Y.shape[0], float(sd.max()), n_live, Y.shape[1]))
        features = {name: self._drain(name) for name in self._buffers}
        present = {k: v for k, v in features.items() if v is not None}
        missing = [k for k, v in features.items() if v is None]
        if missing:
            raise RuntimeError(
                f'Taps fired zero times: {missing}. The hooked module(s) were '
                'never reached during the forward pass.')
        for name, arr in present.items():
            if arr.shape[0] != Y.shape[0]:
                raise RuntimeError(
                    f"Tap '{name}' has {arr.shape[0]} rows but Y has {Y.shape[0]}; "
                    'features and labels are misaligned.')
        return FeatureSet(present, Y)

    def clear(self) -> None:
        """Drop all buffered features (keep the hooks attached)."""
        self._buffers.clear()

    def remove(self) -> None:
        """Detach all forward hooks. Always call this when done."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> 'FeatureCollector':
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
