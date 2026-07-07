"""
pose_error.py
-------------
Cooperative Failure Mode: localisation (pose) error on shared agent poses.

In cooperative perception every agent broadcasts its pose T_agent_to_world
alongside its sensor data; the ego uses that pose to warp the received data
(or features) into its own frame. GPS/IMU error therefore corrupts the
SPATIAL ALIGNMENT of everything an agent shares, which is the dominant
real-world failure of V2V/V2X fusion.

This is the standard robustness protocol of V2X-ViT (arXiv:2203.10638,
Sec. 5.3) and is reused by CoBEVT, Where2comm and CoAlign: add zero-mean
Gaussian noise to the 2-D position and heading of every NON-EGO agent,

    x' = x + e_x,   e_x ~ Normal(0, sigma_xy^2)
    y' = y + e_y,   e_y ~ Normal(0, sigma_xy^2)
    yaw' = yaw + e_h, e_h ~ Normal(0, sigma_heading^2)

with sigma_xy swept over e.g. {0, 0.1, 0.2, ..., 0.5} m and sigma_heading
over {0, 0.2, ..., 1.0} degrees. Optionally a Laplace distribution can be
used (CoAlign, arXiv:2211.07214, argues heavy-tailed GPS error).

The injector works on 4x4 homogeneous pose matrices (the common sample
model of `src.datasets`) and also on raw [x, y, z, roll, yaw, pitch] lists
(the OPV2V/OpenCOOD yaml convention), so it can be dropped into either
this repository's pipeline or an OpenCOOD data loader.

Design notes
------------
* The ego's own pose is NOT perturbed by `apply_to_sample` (protect_ego
  defaults to True): pose error is relative misalignment between sender
  and receiver, and the standard protocol perturbs the senders.
* Noise is applied around the z (up) axis only -- planar position + heading
  -- matching the published protocol. Roll/pitch error can be enabled
  explicitly with sigma_rollpitch for full 6-DoF studies.
* Reproducible by default (seeded). Pass seed=None for fresh randomness.
"""

import numpy as np


def _rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class PoseErrorInjector:
    """
    Add localisation noise to agent poses.

    Parameters
    ----------
    sigma_xy         : float  std of planar position noise in metres
                       (V2X-ViT sweeps 0.0 - 0.5).
    sigma_heading    : float  std of heading (yaw) noise in DEGREES
                       (V2X-ViT sweeps 0.0 - 1.0).
    sigma_z          : float  std of vertical position noise in metres
                       (0 in the standard protocol).
    sigma_rollpitch  : float  std of roll/pitch noise in degrees
                       (0 in the standard protocol).
    distribution     : 'gaussian' (default) or 'laplace' (CoAlign-style
                       heavy-tailed error; sigma_* are then the Laplace
                       scale parameters).
    seed             : int for reproducible draws, or None.

    Usage
    -----
        inj = PoseErrorInjector(sigma_xy=0.2, sigma_heading=0.2)

        T_noisy = inj(T_agent_to_world)          # 4x4 matrix in, 4x4 out
        pose6   = inj.perturb_pose6(pose6)       # [x,y,z,roll,yaw,pitch] deg
        sample  = inj.apply_to_sample(sample)    # src.datasets sample
    """

    def __init__(self, sigma_xy=0.2, sigma_heading=0.2, sigma_z=0.0,
                 sigma_rollpitch=0.0, distribution='gaussian', seed=0):
        for name, v in [('sigma_xy', sigma_xy), ('sigma_heading', sigma_heading),
                        ('sigma_z', sigma_z), ('sigma_rollpitch', sigma_rollpitch)]:
            if v < 0:
                raise ValueError(f'{name} must be >= 0.')
        if distribution not in ('gaussian', 'laplace'):
            raise ValueError("distribution must be 'gaussian' or 'laplace'.")
        self.sigma_xy        = sigma_xy
        self.sigma_heading   = sigma_heading
        self.sigma_z         = sigma_z
        self.sigma_rollpitch = sigma_rollpitch
        self.distribution    = distribution
        self.rng             = np.random.default_rng(seed)

    # ── core draws ──────────────────────────────────────────────────────

    def _draw(self, sigma, size=None):
        if sigma == 0:
            return np.zeros(size) if size else 0.0
        if self.distribution == 'laplace':
            return self.rng.laplace(0.0, sigma, size)
        return self.rng.normal(0.0, sigma, size)

    def sample_error(self):
        """
        Draw one pose error.

        Returns
        -------
        dict with keys dx, dy, dz (metres) and droll, dyaw, dpitch (degrees).
        """
        return {
            'dx'    : float(self._draw(self.sigma_xy)),
            'dy'    : float(self._draw(self.sigma_xy)),
            'dz'    : float(self._draw(self.sigma_z)),
            'dyaw'  : float(self._draw(self.sigma_heading)),
            'droll' : float(self._draw(self.sigma_rollpitch)),
            'dpitch': float(self._draw(self.sigma_rollpitch)),
        }

    # ── pose formats ────────────────────────────────────────────────────

    def perturb_matrix(self, T_agent_to_world, error=None):
        """
        Perturb a 4x4 agent->world pose matrix. Position noise is added to
        the translation; heading noise pre-rotates the orientation about the
        agent's up axis (a heading error rotates everything the agent shares).
        """
        T = np.asarray(T_agent_to_world, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f'expected a (4, 4) pose matrix, got {T.shape}')
        e = error if error is not None else self.sample_error()

        out = T.copy()
        R_err = _rot_z(np.radians(e['dyaw']))
        if e['droll'] or e['dpitch']:
            R_err = R_err @ _rot_y(np.radians(e['dpitch'])) @ _rot_x(np.radians(e['droll']))
        out[:3, :3] = T[:3, :3] @ R_err
        out[:3, 3] += [e['dx'], e['dy'], e['dz']]
        return out

    def perturb_pose6(self, pose, error=None):
        """
        Perturb an OPV2V/OpenCOOD-style pose list
        [x, y, z, roll, yaw, pitch] (metres / degrees). Returns a new list.
        """
        if len(pose) != 6:
            raise ValueError('pose must be [x, y, z, roll, yaw, pitch]')
        e = error if error is not None else self.sample_error()
        x, y, z, roll, yaw, pitch = pose
        return [x + e['dx'], y + e['dy'], z + e['dz'],
                roll + e['droll'], yaw + e['dyaw'], pitch + e['dpitch']]

    __call__ = perturb_matrix

    # ── sample-level API ────────────────────────────────────────────────

    def apply_to_sample(self, sample, protect_ego=True):
        """
        Perturb the pose of every (non-ego) agent in a cooperative sample
        from `src.datasets` (modified in place and returned). Each agent
        gets an independent error draw; the draw is recorded in
        `agent.faults['pose_error']` for logging/analysis.
        """
        for agent in sample.agents.values():
            if protect_ego and agent.is_ego:
                continue
            if agent.pose is None:
                continue
            e = self.sample_error()
            agent.pose = self.perturb_matrix(agent.pose, error=e)
            agent.faults['pose_error'] = e
        return sample
