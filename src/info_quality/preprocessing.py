"""
preprocessing.py
----------------
Shared, architecture- and dataset-agnostic preprocessing for the mutual
information estimators.

Both estimators (InfoNCE, SMILE) operate on two float matrices:

    Z : (N, d_z)  feature representation under test (one row per sample)
    Y : (N, d_y)  task-relevant target           (one row per sample)

Nothing here knows about BEVFusion, nuScenes, Griffin, LiDAR or cameras. The
only assumptions are: N rows, real-valued columns, rows aligned between Z and Y.

Two operations live here because both estimators need them and they must be
applied identically to every condition being compared (clean vs faulty):

  * standardise : zero-mean, unit-variance per column. Puts the critic /
                  statistics network at a sane scale regardless of feature
                  magnitudes, which differ wildly across modalities and layers.
  * pca_reduce  : optional linear dimensionality reduction. With small N a
                  high-dimensional Z makes the critic trivially memorise the
                  N samples, inflating the bound. Reducing d_z below N removes
                  that failure mode. PCA is linear, so it cannot *create*
                  information about Y; it only discards low-variance directions.

`set_seed` is provided so an entire estimation run (NumPy + PyTorch, CPU + CUDA)
is reproducible. Reproducibility is a default in this project, not an option.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: Optional[int]) -> None:
    """
    Seed NumPy and (if available) PyTorch for a reproducible estimation run.

    Parameters
    ----------
    seed : int to seed all RNGs, or None to leave them untouched (fresh stream).

    Notes
    -----
    Torch is imported lazily so that callers who only use the NumPy-side
    preprocessing do not pay the import cost. Seeding CUDA is a no-op when no
    GPU is present.
    """
    if seed is None:
        return
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Feature preprocessing ───────────────────────────────────────────────────

def standardise(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardise each column to zero mean and unit variance.

    Parameters
    ----------
    X   : np.ndarray (N, d)
    eps : float guard against division by zero for constant columns.

    Returns
    -------
    np.ndarray (N, d), float32.
    """
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def pca_reduce(X: Optional[np.ndarray], n_components: int) -> Optional[np.ndarray]:
    """
    Reduce X to at most `n_components` principal components.

    The effective number of components is capped at min(n_components, N - 1, d)
    so the call is always valid: PCA cannot return more components than the
    rank of the (centred) data, which is bounded by N - 1.

    Parameters
    ----------
    X            : np.ndarray (N, d), or None (passed through as None).
    n_components : int upper bound on the number of components to keep.

    Returns
    -------
    np.ndarray (N, k) with k = min(n_components, N - 1, d), or None.

    Raises
    ------
    ImportError if scikit-learn is not installed.
    """
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'pca_reduce requires scikit-learn. Install the info-quality extras: '
            'pip install -r requirements-info-quality.txt'
        ) from exc
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    if k < 1:
        raise ValueError(f'Cannot run PCA: need at least 2 samples, got N={X.shape[0]}.')
    return PCA(n_components=k).fit_transform(X).astype(np.float32)


def prepare(Z: np.ndarray, Y: np.ndarray, pca_dims: Optional[int] = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the standard preprocessing pipeline to a (Z, Y) pair.

    Pipeline: optional PCA on Z -> standardise Z -> standardise Y. Y is never
    PCA-reduced; it is the target and is usually already low dimensional.

    Parameters
    ----------
    Z        : np.ndarray (N, d_z)  feature matrix under test.
    Y        : np.ndarray (N, d_y)  target matrix.
    pca_dims : int to reduce Z to that many components first, or None to skip.

    Returns
    -------
    (Z_prepared, Y_prepared), both float32 with N rows.

    Raises
    ------
    ValueError if Z and Y have a different number of rows.
    """
    Z = np.asarray(Z, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if Z.shape[0] != Y.shape[0]:
        raise ValueError(
            f'Z and Y must have the same number of rows; got {Z.shape[0]} and {Y.shape[0]}.'
        )
    if pca_dims is not None:
        Z = pca_reduce(Z, pca_dims)
    return standardise(Z), standardise(Y)
