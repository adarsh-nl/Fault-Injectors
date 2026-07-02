"""LidarSnowInjector - MultiCorrupt / Hahner LiDAR snowfall simulation on Griffin.

Snow is environmental: it must corrupt the LiDAR as well as the camera. This
injector runs MultiCorrupt's snow physics (which is Hahner et al.'s CVPR 2022
LiDAR snowfall simulation) on Griffin clouds. The full occlusion / pulse-power
machinery is MultiCorrupt's verbatim code in `_mc_snow.py`; the snowflake
particle files are generated with Hahner's verbatim `dart_throwing` sampler in
`snowflake_sampling.py`.

What the simulation does, per LiDAR channel:
  1. sample which laser beams intersect airborne snowflake disks (get_occlusions);
  2. run a pulse-power simulation over range bins where flakes reflect at 90%
     reflectivity and the original hard target keeps its (focal-corrected)
     intensity (process_single_channel);
  3. the strongest return wins: if it is still the hard target, the point is kept
     with attenuated intensity (flag 1); if a flake wins, the point is dragged to
     the flake's range as a scatter point (flag 2);
  4. finally, returns whose intensity falls below a range-adaptive noise floor
     (fitted on ground points) are dropped entirely.

Griffin has no nuScenes lidarseg and no ring column, so two inputs of the exact
pipeline are derived from the data itself. Each derivation is documented and
isolated; everything downstream of these two inputs is byte-identical MultiCorrupt
code:

  A. GROUND MASK (replaces nuScenes lidarseg driveable/other-flat/sidewalk):
     a geometric ground estimate. Candidate points in the lowest height band are
     used to fit a plane z = a*x + b*y + c by RANSAC (least squares on random
     minimal samples, best consensus kept); points within `ground_tol` of the
     plane are ground. On CARLA/AirSim towns the ground is near-planar, so this
     is a faithful stand-in for the semantic classes the noise-floor fit needs.

  B. CHANNEL COLUMN (replaces the nuScenes ring index):
     the simulation models an HDL-64 sensor (snow_laser.yaml, 64 channels with
     per-channel focal corrections). Griffin points are assigned to those 64
     channels by binning elevation angle into 64 equal bins. The channel column
     only groups points and selects per-channel particle files / focal constants;
     it does not alter the per-point geometry.

Intensity scale: the pipeline assumes nuScenes-style 0-255 intensity (focal
corrections, 90%-reflectivity flakes, 255 clip). If the input intensity maximum
is <= 1.5 (CARLA convention), intensities are scaled to 0-255 for the simulation
and scaled back afterwards.

Frames: beams must radiate from the sensor origin. If `T_lidar_to_ego` is given,
inputs are treated as EGO-frame (Griffin `load_lidar` default), converted to the
sensor frame internally, and converted back after simulation. If None, inputs are
assumed already sensor-centred.

Output: (M, 4) array [x, y, z, intensity], M <= N (sub-noise-floor returns are
removed). `last_flags` holds the per-output-point status (0 unchanged,
1 attenuated, 2 snow-scatter) and `last_stats` the (num_attenuated, num_removed,
avg_intensity_diff) tuple from the simulation.

Runtime: the occlusion test is per-point Python/NumPy (inherited from upstream);
expect roughly 1-3 minutes for a full Griffin frame. Particle files are generated
once per severity on first use (a few minutes) and cached under
src/fault_injectors/npy/.
"""
from __future__ import annotations

import numpy as np

from . import _mc_snow
from .snowflake_sampling import ensure_particle_files

# The snow simulation discretizes each beam into range bins over a fixed
# [0, lidar_range] window (lidar_range = 120 m, hard-coded in MultiCorrupt's
# process_single_channel; the pulse-power array has 120*10 + margin bins). It is
# the HDL-64 sensor model's valid domain. Griffin's simulated LiDAR returns points
# well beyond that (CARLA range extends to 200-300 m+), and those indices run past
# the pulse-power array. Points beyond this range are therefore outside the
# simulation's domain: they are passed through unchanged (snow attenuation is
# negligible at such ranges anyway - the beam is extinct long before). This is a
# domain restriction in the wrapper, NOT a change to the verbatim physics.
SIM_MAX_RANGE = 118.0    # m; a small margin below the 120 m bin ceiling


def geometric_ground_mask(pts_sensor, mount_height=None, band=0.5,
                          ground_tol=0.15, iters=200, seed=1000):
    """Boolean ground mask via RANSAC plane fit z = a*x + b*y + c (sensor frame).

    Candidates are points within `band` metres of the lowest height mode (or of
    -mount_height if given). The plane with the largest inlier consensus over
    `iters` random 3-point samples wins; inliers within `ground_tol` are ground.
    """
    z = pts_sensor[:, 2]
    z_ref = -abs(mount_height) if mount_height is not None else np.percentile(z, 2.0)
    cand = np.abs(z - z_ref) < band
    if cand.sum() < 50:                                   # degenerate scene: take lowest decile
        cand = z < np.percentile(z, 10)
    P = pts_sensor[cand, :3]
    rng = np.random.default_rng(seed)
    best_inl, best_plane = None, None
    for _ in range(iters):
        idx = rng.choice(P.shape[0], size=3, replace=False)
        A = np.c_[P[idx, 0], P[idx, 1], np.ones(3)]
        try:
            coef = np.linalg.solve(A, P[idx, 2])
        except np.linalg.LinAlgError:
            continue
        d = np.abs(P[:, 0] * coef[0] + P[:, 1] * coef[1] + coef[2] - P[:, 2])
        inl = (d < ground_tol).sum()
        if best_inl is None or inl > best_inl:
            best_inl, best_plane = inl, coef
    a, b, c0 = best_plane
    dist_all = np.abs(pts_sensor[:, 0] * a + pts_sensor[:, 1] * b + c0 - pts_sensor[:, 2])
    return dist_all < ground_tol


def elevation_channels(pts_sensor, num_channels=64):
    """Assign each point to one of `num_channels` elevation bins (sensor frame)."""
    r_xy = np.hypot(pts_sensor[:, 0], pts_sensor[:, 1])
    phi = np.arctan2(pts_sensor[:, 2], r_xy + 1e-12)
    lo, hi = float(phi.min()), float(phi.max())
    ch = np.floor((phi - lo) / (hi - lo + 1e-12) * num_channels).astype(int)
    return np.clip(ch, 0, num_channels - 1)


class LidarSnowInjector:
    """MultiCorrupt/Hahner LiDAR snowfall on Griffin (N,4) clouds.

    severity        : 1 / 2 / 3 -> (snowfall_rate, terminal_velocity) =
                      (0.5, 1.2) / (2.5, 1.6) / (1.5, 0.4)  [MultiCorrupt Table I:
                      5 / 35 / 70 mm/h equivalent].
    T_lidar_to_ego  : optional 4x4; if given, input/output are EGO-frame and the
                      simulation runs in the sensor frame internally.
    mount_height    : sensor height above ground in metres, used only to seed the
                      geometric ground search (Griffin lidar_top: 1.10).
    noise_floor,
    beam_divergence : passed through exactly as MultiCorrupt's converter does
                      (0.7 and degrees(0.003)).
    seed            : seeds numpy (channel shuffle, RANSAC noise fit) for
                      reproducibility; particle files are seeded independently.
    """

    def __init__(self, severity=2, T_lidar_to_ego=None, mount_height=1.10,
                 noise_floor=0.7, beam_divergence=float(np.degrees(0.003)),
                 shuffle=True, seed=1000, verbose=True):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity = severity
        self.T_lidar_to_ego = None if T_lidar_to_ego is None else np.asarray(T_lidar_to_ego, float)
        self.mount_height = mount_height
        self.noise_floor = noise_floor
        self.beam_divergence = beam_divergence
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.last_flags = None
        self.last_stats = None
        self.last_ground_fraction = None
        self.last_num_far_passthrough = None

    def __call__(self, points):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise ValueError(f"expected (N,>=4) [x,y,z,intensity], got {pts.shape}")
        ensure_particle_files(self.severity, seed=42, verbose=self.verbose)
        np.random.seed(self.seed)                       # channel shuffle + ransac_polyfit

        # ego -> sensor if a mount transform is provided
        if self.T_lidar_to_ego is not None:
            T_inv = np.linalg.inv(self.T_lidar_to_ego)
            xyz_h = np.c_[pts[:, :3], np.ones(len(pts))]
            xyz_s = (T_inv @ xyz_h.T).T[:, :3]
        else:
            xyz_s = pts[:, :3].copy()

        # intensity scale handling (CARLA 0-1 -> 0-255 for the simulation)
        inten = pts[:, 3].copy()
        scale = 255.0 if np.nanmax(inten) <= 1.5 else 1.0
        inten255 = np.clip(inten * scale, 0, 255)

        # Restrict to the simulation's valid range window (see SIM_MAX_RANGE).
        # Far points are outside the HDL-64 sensor model's domain and are passed
        # through unchanged (flag 0); only near points enter the verbatim physics.
        rng_sensor = np.linalg.norm(xyz_s, axis=1)
        near = rng_sensor <= SIM_MAX_RANGE
        self.last_num_far_passthrough = int((~near).sum())

        # Griffin-derived inputs A and B (see module docstring), computed on the
        # near points that will actually be simulated.
        xyz_near = xyz_s[near]
        ground_near = geometric_ground_mask(xyz_near, mount_height=self.mount_height, seed=self.seed)
        channels_near = elevation_channels(xyz_near, num_channels=64)
        self.last_ground_fraction = float(ground_near.mean()) if near.any() else 0.0

        pc5 = np.c_[xyz_near, inten255[near], channels_near.astype(np.float64)]
        stats, aug_pc, _ = _mc_snow.simulate_snow_griffin(
            pc5, ground_near, self.severity,
            beam_divergence=self.beam_divergence,
            shuffle=self.shuffle, noise_floor=self.noise_floor)
        self.last_stats = stats

        # near points: simulated geometry + intensity + flags
        near_xyz_s = aug_pc[:, :3]
        near_int = aug_pc[:, 3]
        near_flags = aug_pc[:, 4].astype(int)

        # far points: untouched, flagged 0 (unchanged / outside sim domain)
        far_xyz_s = xyz_s[~near]
        far_int = inten255[~near]
        far_flags = np.zeros(len(far_xyz_s), dtype=int)

        out_xyz_s = np.vstack([near_xyz_s, far_xyz_s]) if len(far_xyz_s) else near_xyz_s
        out_int255 = np.concatenate([near_int, far_int]) if len(far_xyz_s) else near_int
        self.last_flags = np.concatenate([near_flags, far_flags]) if len(far_xyz_s) else near_flags
        out_int = out_int255 / scale

        # sensor -> ego back-transform
        if self.T_lidar_to_ego is not None:
            xyz_h = np.c_[out_xyz_s, np.ones(len(out_xyz_s))]
            out_xyz = (self.T_lidar_to_ego @ xyz_h.T).T[:, :3]
        else:
            out_xyz = out_xyz_s

        return np.c_[out_xyz, out_int].astype(np.float32)
