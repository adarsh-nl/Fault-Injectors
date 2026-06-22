"""
temporal_misalignment.py
------------------------
Failure Mode 2: Temporal Misalignment (stale image via index shifting).

A clean sample at frame k is the synchronised pair X = (I_k, P_k). This
injector corrupts the pairing in time: the model receives the CURRENT LiDAR
scan together with an OLDER image,

    X~_k = (I_{k - delta_k}, P_k)

where delta_k >= 0 is a discrete frame-index shift. The model is unaware that
the image is stale.

How delta_k is computed
-----------------------
Datasets are recorded at a fixed frame rate f (Hz), so consecutive frames are
separated by a known period:

    delta_t_frame = 1 / f          (0.1 s at 10 Hz)

A continuous time delay is sampled per frame from a Normal distribution and
quantised to the nearest integer number of frames:

    delta_t_{k} ~ Normal(mu_delay, sigma_jitter^2)
    delta_k     = round(delta_t_{k} / delta_t_frame)

The two parameters play distinct roles:

  * mu_delay     the SYSTEMATIC component: a miscalibrated clock or a fixed
                 transmission lag. Every frame is affected by about this much.
  * sigma_jitter the STOCHASTIC component: frame-to-frame variation from CPU
                 scheduling, network congestion, or hardware interrupts.

Example (the spec's worked example): at 10 Hz with mu_delay = 200 ms and
sigma_jitter = 50 ms, one draw might give delta_t = 213 ms, so
delta_k = round(2.13) = 2: LiDAR P_10 is paired with image I_8.

Physical interpretation
-----------------------
A vehicle moving at speed v covers d = v * delta_t during the delay. At
50 km/h (~13.9 m/s) a 200 ms delay means the image describes a world ~2.8 m
behind the LiDAR's. The fusion layer projects current LiDAR points onto stale
pixels, displacing boxes or losing detections entirely.

Why index shifting (and not, e.g., generative image warping)?
-------------------------------------------------------------
Index shifting gives physically authentic corrupted inputs with zero
computational overhead, and any accuracy change in evaluation is guaranteed to
come from the temporal mismatch rather than from image-quality degradation.

Design notes
------------
* delta_k is clamped at 0: a sensor cannot deliver a frame from the future.
  Negative sampled delays (possible when mu_delay is small relative to
  sigma_jitter) therefore degrade to the clean pairing.
* At the start of a sequence the stale index k - delta_k can fall before the
  first available frame; it is clamped to `k_min` (the earliest valid index).
* Reproducible by default (seeded). Pass seed=None for fresh randomness.
* Units are SECONDS throughout (mu_delay=0.2 is 200 ms).
"""

import numpy as np


# ── Low-level primitives ───────────────────────────────────────────────────

def sample_index_shift(mu_delay, sigma_jitter, frame_period, rng):
    """
    Draw one discrete index shift delta_k.

    Parameters
    ----------
    mu_delay     : float  systematic delay in seconds (e.g. 0.2 for 200 ms).
    sigma_jitter : float  jitter standard deviation in seconds.
    frame_period : float  time between frames in seconds (1 / fps).
    rng          : np.random.Generator

    Returns
    -------
    int  delta_k >= 0.  Sampled as round(Normal(mu, sigma) / frame_period),
         clamped at zero because a frame from the future cannot be delivered.
    """
    delta_t = rng.normal(mu_delay, sigma_jitter)
    return max(int(round(delta_t / frame_period)), 0)


def physical_displacement(velocity, delta_t):
    """
    Distance the platform travels during the delay: d = v * delta_t.

    Parameters
    ----------
    velocity : float  speed in m/s.
    delta_t  : float  delay in seconds.

    Returns
    -------
    float  displacement in metres.
    """
    return velocity * delta_t


# ── Stateful injector ───────────────────────────────────────────────────────

class TemporalMisalignmentInjector:
    """
    Pair the current LiDAR frame with a stale image via index shifting.

    Parameters
    ----------
    mu_delay     : float  systematic delay in seconds (default 0.2 = 200 ms).
    sigma_jitter : float  jitter std in seconds (default 0.05 = 50 ms).
    fps          : float  dataset frame rate in Hz (Griffin: 10).
    seed         : int or None  reproducible draws by default.

    Usage
    -----
        inj = TemporalMisalignmentInjector(mu_delay=0.2, sigma_jitter=0.05)

        # Index-level API (zero overhead, mirrors a data-loader modification):
        k_img, dk = inj.stale_index(k, k_min=0)
        corrupted_pair = (load_image(k_img), load_points(k))

        # Sequence-level API (lists already in memory, local 0-based indices):
        out = inj.inject(k_local, images, points_list)
        # out['image'] is stale, out['points'] is current, out['delta_k'] ...
    """

    def __init__(self, mu_delay=0.2, sigma_jitter=0.05, fps=10.0, seed=0):
        if mu_delay < 0:
            raise ValueError('mu_delay must be >= 0 seconds.')
        if sigma_jitter < 0:
            raise ValueError('sigma_jitter must be >= 0 seconds.')
        if fps <= 0:
            raise ValueError('fps must be positive.')
        self.mu_delay     = mu_delay
        self.sigma_jitter = sigma_jitter
        self.frame_period = 1.0 / fps
        self.rng          = np.random.default_rng(seed)

    # ── core draws ──────────────────────────────────────────────────────

    def sample_shift(self):
        """Draw one delta_k (>= 0) from the configured delay distribution."""
        return sample_index_shift(self.mu_delay, self.sigma_jitter,
                                  self.frame_period, self.rng)

    def stale_index(self, k, k_min=0):
        """
        For LiDAR frame k, return the index of the stale image to pair.

        Parameters
        ----------
        k     : int  current frame index.
        k_min : int  earliest valid index (sequence start); the stale index
                     is clamped so it never falls before it.

        Returns
        -------
        (k_image, delta_k) : tuple of int
            k_image = max(k - delta_k, k_min) is the image index to load;
            delta_k is the sampled shift before clamping to k_min.
        """
        delta_k = self.sample_shift()
        return max(k - delta_k, k_min), delta_k

    # ── convenience: corrupt directly from in-memory sequences ─────────

    def inject(self, k, images, points_seq, k_min=0):
        """
        Build the corrupted pair (I_{k-delta_k}, P_k) from sequences.

        Parameters
        ----------
        k          : int   local index of the CURRENT frame.
        images     : sequence of (H, W, 3) arrays, indexable by local index.
        points_seq : sequence of (N, C) arrays, indexable by local index.
        k_min      : int   earliest valid local index (default 0).

        Returns
        -------
        dict with keys:
            image    : the stale image I_{k_image}.
            points   : the current point cloud P_k.
            k_image  : the (clamped) index the image came from.
            delta_k  : the sampled shift.
        """
        k_image, delta_k = self.stale_index(k, k_min=k_min)
        return {
            'image'  : images[k_image],
            'points' : points_seq[k],
            'k_image': k_image,
            'delta_k': delta_k,
        }

    __call__ = inject

    # ── inspection without data ─────────────────────────────────────────

    def simulate_sequence(self, n_frames):
        """
        Pre-draw delta_k for n_frames without touching any data.

        Useful for plotting the shift schedule or its distribution before a
        run. Draws consume the same RNG stream as injection, so simulate and
        inject with SEPARATE injector instances if you need both to match.

        Returns
        -------
        np.ndarray of int, shape (n_frames,), each entry the delta_k drawn
        for that frame.
        """
        return np.array([self.sample_shift() for _ in range(n_frames)])
