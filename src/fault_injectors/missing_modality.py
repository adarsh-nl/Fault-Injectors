"""
missing_modality.py
-------------------
Failure Mode 1: Missing Modality (Sensor Dropout).

A clean, perfectly synchronised sample from platform i at frame k is the pair
X = (I, P), where I is the RGB image and P is the LiDAR point cloud, both
captured at the same instant.

This injector models a sensor failing: when a modality is "dropped", the model
receives a structurally valid but information-free input.

  - Dropped image  -> a zero tensor of the same shape (a black frame), or
                      optionally a constant mean-fill (see `fill` below).
  - Dropped points -> an empty point set (N = 0 points).

Whether each modality is dropped on a given frame is decided by a Bernoulli
trial. With a per-modality drop probability p_drop, a sensor stays ALIVE with
probability (1 - p_drop):

    m_RGB   ~ Bernoulli(1 - p_drop_RGB)     # 1 = alive, 0 = dropped
    m_LiDAR ~ Bernoulli(1 - p_drop_LiDAR)

Each trial is independent per frame and per modality.

Severity sweep used in the spec: p_drop in {0.00, 0.25, 0.50, 0.75, 1.00}.

Design notes
------------
* Zero-fill is the default for images because it matches the spec and is the
  common convention. Be aware that after a network's input normalisation a zero
  pixel is NOT "neutral" -- it maps to a negative value. If you want the fill to
  be neutral *after* normalisation, use fill='mean' with the dataset mean, which
  maps to ~0 post-normalisation. Both are provided; zero is the default.
* Empty (N = 0) is the only sensible representation of a dropped point cloud.
  Downstream voxelisers and PointNet-style encoders accept an empty set as a
  valid (if degenerate) input.
* Randomness is reproducible by default (seeded). Pass seed=None for a fresh,
  unseeded stream so every run drops different frames.
"""

import numpy as np


# ── Low-level primitives ───────────────────────────────────────────────────

def bernoulli_mask(p_drop, rng):
    """
    Draw one Bernoulli availability gate.

    Parameters
    ----------
    p_drop : float in [0, 1]  probability the sensor is DROPPED this frame.
    rng    : np.random.Generator

    Returns
    -------
    int  1 if the sensor is ALIVE (kept), 0 if DROPPED.
         Implemented as Bernoulli(1 - p_drop): alive with prob (1 - p_drop).
    """
    return int(rng.random() >= p_drop)


def drop_image(image, fill='zero', mean_value=None):
    """
    Return a dropped (information-free) version of an RGB image.

    Parameters
    ----------
    image      : np.ndarray (H, W, 3)
    fill       : 'zero' (black frame) or 'mean' (constant mean-fill).
    mean_value : per-channel mean as a length-3 sequence, required if fill='mean'.
                 Use the dataset mean so that, after input normalisation, the
                 filled image maps to approximately zero (a neutral input).

    Returns
    -------
    np.ndarray (H, W, 3)  same shape and dtype as the input.
    """
    if fill == 'zero':
        return np.zeros_like(image)
    if fill == 'mean':
        if mean_value is None:
            raise ValueError("fill='mean' requires mean_value (length-3 per-channel mean).")
        out = np.empty_like(image)
        out[:] = np.asarray(mean_value, dtype=image.dtype)
        return out
    raise ValueError(f"Unknown fill '{fill}'. Use 'zero' or 'mean'.")


def drop_points(points):
    """
    Return a dropped (empty) version of a LiDAR point cloud.

    Parameters
    ----------
    points : np.ndarray (N, C)

    Returns
    -------
    np.ndarray (0, C)  an empty point set with the same number of columns.
    """
    return np.empty((0, points.shape[1]), dtype=points.dtype)


# ── Stateful injector ───────────────────────────────────────────────────────

class MissingModalityInjector:
    """
    Apply Bernoulli sensor dropout to a stream of (image, points) samples.

    Parameters
    ----------
    p_drop_rgb   : float in [0, 1]  per-frame drop probability for the camera.
    p_drop_lidar : float in [0, 1]  per-frame drop probability for the LiDAR.
    fill         : 'zero' or 'mean' fill style for dropped images.
    mean_value   : length-3 per-channel mean, required if fill='mean'.
    seed         : int for reproducible draws, or None for a fresh random stream.

    Usage
    -----
        inj = MissingModalityInjector(p_drop_lidar=0.5, seed=0)
        for image, points in stream:
            corrupt = inj(image, points)
            # corrupt.image, corrupt.points, corrupt.m_rgb, corrupt.m_lidar

    The returned object is a small dict-like result, see `inject`.
    """

    def __init__(self, p_drop_rgb=0.0, p_drop_lidar=0.0,
                 fill='zero', mean_value=None, seed=0):
        if not (0.0 <= p_drop_rgb <= 1.0):
            raise ValueError('p_drop_rgb must be in [0, 1].')
        if not (0.0 <= p_drop_lidar <= 1.0):
            raise ValueError('p_drop_lidar must be in [0, 1].')
        self.p_drop_rgb   = p_drop_rgb
        self.p_drop_lidar = p_drop_lidar
        self.fill         = fill
        self.mean_value   = mean_value
        self.rng          = np.random.default_rng(seed)

    def inject(self, image, points):
        """
        Corrupt one (image, points) sample.

        Returns
        -------
        dict with keys:
            image   : (H, W, 3) corrupted image (clean if RGB kept).
            points  : (N, C) or (0, C) corrupted points (clean if LiDAR kept).
            m_rgb   : 1 if the camera was kept, 0 if dropped.
            m_lidar : 1 if the LiDAR was kept, 0 if dropped.
        """
        m_rgb   = bernoulli_mask(self.p_drop_rgb,   self.rng)
        m_lidar = bernoulli_mask(self.p_drop_lidar, self.rng)

        out_image  = image  if m_rgb   else drop_image(image, self.fill, self.mean_value)
        out_points = points if m_lidar else drop_points(points)

        return {
            'image'  : out_image,
            'points' : out_points,
            'm_rgb'  : m_rgb,
            'm_lidar': m_lidar,
        }

    # Allow calling the injector directly: inj(image, points)
    __call__ = inject

    def simulate_sequence(self, n_frames):
        """
        Pre-draw the availability gates for n_frames without needing data.

        Useful for inspecting or plotting a dropout schedule (e.g. reproducing
        the 10-frame example table in the spec) before touching real samples.

        Returns
        -------
        dict with keys 'm_rgb' and 'm_lidar', each a length-n_frames int array
        of 1 (alive) / 0 (dropped).
        """
        m_rgb   = np.array([bernoulli_mask(self.p_drop_rgb,   self.rng) for _ in range(n_frames)])
        m_lidar = np.array([bernoulli_mask(self.p_drop_lidar, self.rng) for _ in range(n_frames)])
        return {'m_rgb': m_rgb, 'm_lidar': m_lidar}
