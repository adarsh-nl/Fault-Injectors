"""
communication.py
----------------
Cooperative Failure Modes: the V2V/V2X communication channel.

In cooperative perception (V2VNet, V2X-ViT, CoBEVT, Where2comm) the ego
receives data from other agents over a wireless link. Three things go wrong
with that link, and each is a separate injector here:

1. LATENCY (CommLatencyInjector)
   The received data is STALE: at ego frame k the ego only has agent j's
   frame k - delta_jk. This generalises `temporal_misalignment` from
   "camera lags LiDAR on one platform" to "sender lags receiver across
   platforms" -- the asynchronous setting of V2X-ViT (their delay sweep is
   0 - 400 ms) and of SyncNet / CoBEVFlow. Delay is sampled per agent per
   frame as Normal(mu_delay, sigma_jitter), quantised to frames, clamped
   at 0. The sender's POSE travels with its data: a stale frame comes with
   the stale pose, which is exactly why latency hurts fusion.

2. PACKET LOSS / AGENT DROPOUT (AgentDropInjector)
   A transmission fails entirely: the ego simply does not receive agent j
   this frame. Bernoulli per agent per frame with drop probability p_drop
   (sweep {0, 0.25, 0.5, 0.75, 1.0}; p_drop=1 is the "no cooperation"
   lower baseline every paper reports). Optionally burst losses via a
   two-state Gilbert-Elliott channel (loss comes in bursts on real DSRC /
   C-V2X links).

3. BANDWIDTH LIMIT (BandwidthLimitInjector)
   The channel carries fewer bits than the sensor produces. Where2comm's
   whole premise is fusing under a communication budget. At the raw-data
   level the honest proxy is transmitting only a fraction of the point
   cloud (random subsampling) and/or quantising coordinates to a coarser
   grid before sharing. Applied to non-ego agents only: the ego's own
   sensors are local and free.

All three operate on the common sample model of `src.datasets` and also
expose low-level, dataset-free primitives (index maps, masks) so they can
be wired into an OpenCOOD dataloader without this repo's sample classes.

Reproducible by default (seeded); pass seed=None for fresh randomness.
"""

import numpy as np


# ── 1. Latency ──────────────────────────────────────────────────────────────

class CommLatencyInjector:
    """
    Per-agent transmission delay: at ego frame k, agent j's data comes from
    frame k - delta_jk.

    Parameters
    ----------
    mu_delay     : float  mean delay in seconds (e.g. 0.1 = 100 ms).
    sigma_jitter : float  delay jitter std in seconds.
    fps          : float  dataset frame rate in Hz (OPV2V/V2XSet: 10,
                   DAIR-V2X: 10, Griffin: 10). Used to quantise seconds
                   to frame indices.
    seed         : int or None.

    Usage
    -----
        inj = CommLatencyInjector(mu_delay=0.1, sigma_jitter=0.02, fps=10)

        # Index-level API (wire into any dataloader):
        k_send, delta = inj.stale_index('agent7', k, k_min=0)

        # Sample-level API (src.datasets):
        sample = inj.apply(dataset, k)   # rebuilds the cooperative sample
                                         # with each non-ego agent taken
                                         # from its stale frame
    """

    def __init__(self, mu_delay=0.1, sigma_jitter=0.0, fps=10.0, seed=0):
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

    def sample_shift(self):
        """Draw one delay in frames (int >= 0)."""
        delta_t = self.rng.normal(self.mu_delay, self.sigma_jitter)
        return max(int(round(delta_t / self.frame_period)), 0)

    def stale_index(self, agent_id, k, k_min=0):
        """
        Frame index agent `agent_id`'s data should be read from at ego
        frame k. Returns (k_stale, delta_frames). agent_id is accepted for
        signature symmetry / logging; each call draws independently.
        """
        delta = self.sample_shift()
        return max(k - delta, k_min), delta

    def apply(self, dataset, k, protect_ego=True, **frame_kwargs):
        """
        Build a latency-corrupted cooperative sample at ego frame k.

        The ego agent is loaded at frame k; every other agent at its own
        stale frame (with its stale pose -- the pose it transmitted). The
        applied delay is recorded in `agent.faults['comm_latency']`.
        """
        sample = dataset.get_sample(k, **frame_kwargs)
        for agent_id, agent in sample.agents.items():
            if protect_ego and agent.is_ego:
                continue
            k_stale, delta = self.stale_index(agent_id, k, k_min=0)
            if k_stale != k:
                stale = dataset.get_sample(k_stale, agents=[agent_id],
                                           **frame_kwargs)
                sample.agents[agent_id] = stale.agents[agent_id]
                agent = sample.agents[agent_id]
            agent.faults['comm_latency'] = {
                'delta_frames': delta,
                'delta_seconds': delta * self.frame_period,
                'frame_used': k_stale,
            }
        return sample


# ── 2. Packet loss / agent dropout ─────────────────────────────────────────

class AgentDropInjector:
    """
    Bernoulli (or bursty Gilbert-Elliott) loss of whole agent transmissions.

    Parameters
    ----------
    p_drop  : float in [0, 1]  probability an agent's transmission is lost
              at a given frame. p_drop=1.0 reproduces the single-vehicle
              (no cooperation) baseline.
    burst   : None for i.i.d. loss, or a dict {'p_bad': ..., 'p_recover': ...}
              for a Gilbert-Elliott two-state channel: from the GOOD state
              the link fails with prob p_bad; from the BAD state it recovers
              with prob p_recover; while BAD, every transmission is lost.
              State is tracked per agent across calls (call `reset` between
              sequences).
    seed    : int or None.

    Usage
    -----
        inj = AgentDropInjector(p_drop=0.5)
        keep = inj.keep_mask(['ego', 'cav1', 'cav2'], protect=['ego'])
        sample = inj.apply_to_sample(sample)      # removes dropped agents
    """

    def __init__(self, p_drop=0.0, burst=None, seed=0):
        if not (0.0 <= p_drop <= 1.0):
            raise ValueError('p_drop must be in [0, 1].')
        if burst is not None:
            for key in ('p_bad', 'p_recover'):
                if key not in burst or not (0.0 <= burst[key] <= 1.0):
                    raise ValueError(
                        "burst must be {'p_bad': p, 'p_recover': q} with "
                        "both in [0, 1].")
        self.p_drop = p_drop
        self.burst  = burst
        self.rng    = np.random.default_rng(seed)
        self._bad_state = {}          # agent_id -> bool (Gilbert-Elliott)

    def reset(self):
        """Clear per-agent channel state (call between sequences)."""
        self._bad_state.clear()

    def _lost(self, agent_id):
        if self.burst is None:
            return self.rng.random() < self.p_drop
        bad = self._bad_state.get(agent_id, False)
        if bad:
            if self.rng.random() < self.burst['p_recover']:
                bad = False
        else:
            if self.rng.random() < self.burst['p_bad']:
                bad = True
        self._bad_state[agent_id] = bad
        return bad

    def keep_mask(self, agent_ids, protect=()):
        """
        Draw the loss gates for a list of agent ids.

        Returns
        -------
        dict {agent_id: True (received) / False (lost)}. Agents in
        `protect` are always received.
        """
        return {a: True if a in protect else not self._lost(a)
                for a in agent_ids}

    def apply_to_sample(self, sample, protect_ego=True):
        """
        Remove dropped agents from a cooperative sample (in place, returned).
        Dropped ids are recorded in `sample.meta['dropped_agents']`.
        """
        protect = {a for a, ag in sample.agents.items()
                   if protect_ego and ag.is_ego}
        keep = self.keep_mask(list(sample.agents), protect=protect)
        dropped = [a for a, kept in keep.items() if not kept]
        for a in dropped:
            del sample.agents[a]
        sample.meta.setdefault('dropped_agents', []).extend(dropped)
        return sample


# ── 3. Bandwidth limit ──────────────────────────────────────────────────────

class BandwidthLimitInjector:
    """
    Transmit only part of each non-ego agent's point cloud.

    Parameters
    ----------
    keep_fraction  : float in (0, 1]  fraction of points transmitted
                     (uniform random subsampling without replacement).
    quantise_m     : float or None  if set, round transmitted x,y,z to this
                     grid (metres) -- a coarse proxy for lossy coordinate
                     compression. Duplicate points after rounding are merged.
    seed           : int or None.

    Usage
    -----
        inj = BandwidthLimitInjector(keep_fraction=0.25)
        pts_small = inj(points)                    # (N, C) -> (~N/4, C)
        sample = inj.apply_to_sample(sample)       # non-ego agents only
    """

    def __init__(self, keep_fraction=1.0, quantise_m=None, seed=0):
        if not (0.0 < keep_fraction <= 1.0):
            raise ValueError('keep_fraction must be in (0, 1].')
        if quantise_m is not None and quantise_m <= 0:
            raise ValueError('quantise_m must be positive.')
        self.keep_fraction = keep_fraction
        self.quantise_m    = quantise_m
        self.rng           = np.random.default_rng(seed)

    def inject(self, points):
        """Subsample (and optionally quantise) one (N, C) point cloud."""
        pts = np.asarray(points)
        n = len(pts)
        if n and self.keep_fraction < 1.0:
            m = max(int(round(n * self.keep_fraction)), 1)
            idx = self.rng.choice(n, size=m, replace=False)
            pts = pts[np.sort(idx)]
        if self.quantise_m is not None and len(pts):
            pts = pts.copy()
            pts[:, :3] = np.round(pts[:, :3] / self.quantise_m) * self.quantise_m
            _, uniq = np.unique(pts[:, :3], axis=0, return_index=True)
            pts = pts[np.sort(uniq)]
        return pts

    __call__ = inject

    def apply_to_sample(self, sample, protect_ego=True):
        """
        Reduce the LiDAR of every (non-ego) agent in a cooperative sample
        (in place, returned). Records kept fraction per agent in
        `agent.faults['bandwidth']`.
        """
        for agent in sample.agents.values():
            if protect_ego and agent.is_ego:
                continue
            if agent.lidar is None or len(agent.lidar) == 0:
                continue
            n_before = len(agent.lidar)
            agent.lidar = self.inject(agent.lidar)
            agent.faults['bandwidth'] = {
                'points_before': n_before,
                'points_after': len(agent.lidar),
            }
        return sample
