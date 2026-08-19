"""CoSDH adapter (CVPR 2025, arXiv 2503.03430, github.com/Xu2729/CoSDH).

CoSDH is an OpenCOOD *fork*, not vanilla OpenCOOD, and it diverges at every
point the OpenCOOD adapter hooks. This module is a separate adapter for that
reason; it deliberately does NOT reuse ``src/adapters/opencood.py``.

TWO THINGS THAT DO NOT LOOK LIKE THE OPENCOOD PATH -- DO NOT NORMALISE THEM
===========================================================================
Both of the following are deliberate divergences from
``src/adapters/opencood.py``. A reader who "restores" either by analogy will
produce a fault that fires, logs correct magnitudes, and changes nothing.

  (1) there is NO delta composition here                     -- see below
  (2) the pose hook is NOT in ``retrieve_base_data``          -- see below

(2) is the subtler one and it was found only by reading: the originally
approved hook point (subclassing ``retrieve_base_data``) is WRONG, because
CoSDH copies ``lidar_pose`` into ``lidar_pose_clean`` afterwards. A pose fault
injected there would be duplicated into the clean slot, both derived matrices
would agree, and the injected error would vanish while every fire-check and
magnitude check still passed.

WHY THE DELTA COMPOSITION IS ABSENT -- DO NOT "RESTORE" IT
==========================================================
``src/adapters/opencood.py::from_canonical`` composes a pose perturbation as a
right-multiplied delta onto an already-baked matrix::

    T_new = T_orig @ inv(A_clean) @ A_perturbed

That machinery exists for one reason: vanilla OpenCOOD's ``retrieve_base_data``
calls ``reform_param``, which BAKES ``params['transformation_matrix']`` before
any wrapper can intervene, so the only way to inject is to compose a delta onto
the baked matrix.

**CoSDH bakes nothing.** ``opv2v_basedataset.retrieve_base_data(self, idx)``
loads yaml/json params, cameras and LiDAR and returns; it never calls
``reform_param`` (which is not even defined for this path). Every matrix is
derived per item inside the *fusion* dataset's ``__getitem__``::

    intermediate_late_fusion_dataset.py:86  transformation_matrix       = x1_to_x2(cav['lidar_pose'],       ego['lidar_pose'])
    intermediate_late_fusion_dataset.py:89  transformation_matrix_clean = x1_to_x2(cav['lidar_pose_clean'], ego['lidar_pose_clean'])
    transformation_utils.py:72              pairwise_t_matrix           <- x_to_world(cav['params']['lidar_pose'])

So there is nothing to compose onto, and porting the delta machinery here would
be actively wrong: it would perturb a derived matrix while feature warping
recomputed itself from the untouched pose. Injecting on the POSE is both
simpler and strictly more correct -- one perturbation propagates to feature
warping, to the late-branch detection transform, and to the (inert, because
``proj_first: false``) point projection, because all three derive from it.

WHERE THE POSE INJECTION MUST LAND
==================================
NOT in ``retrieve_base_data``. CoSDH's own pose-noise injector,
``pose_utils.add_noise_data_dict``, runs LATER -- inside the fusion dataset's
item methods (``intermediate_late_fusion_dataset.py:152, 281``) -- and in the
``add_noise: false`` branch it does::

    cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose']

Perturbing ``lidar_pose`` in ``retrieve_base_data`` would therefore be copied
straight into ``lidar_pose_clean`` a moment later, and BOTH matrices would be
built from the perturbed pose -- a perturbation that fires, logs, and changes
nothing, because the noisy/clean pair would agree. That is exactly the failure
a naive fire-check cannot see.

The pose fault is therefore applied by WRAPPING ``add_noise_data_dict`` in the
fusion dataset's module namespace: the original runs first (establishing
``lidar_pose_clean`` = truth), then we perturb ``lidar_pose`` only.

Non-pose faults (LiDAR corruption, agent drop, missing modality, latency) act
on ``lidar_np`` / agent presence and are applied by subclassing
``OPV2VBaseDataset.retrieve_base_data``, which is the factory's intended
extension point (``build_dataset`` passes the base class into
``getIntermediatelateFusionDataset(cls)``).

PROTOCOL FACTS (also recorded in the sweep manifest)
====================================================
* CoSDH ships ``noise_setting: {add_noise: false}`` -- effectively PERFECT. Our
  injected sigma is therefore the TOTAL pose error, not an increment on a
  shipped perturbation as it is for V2X-ViT and Where2comm.
* It groups with CoBEVT (Perfect-shipped), not with the two Noisy models.
* The released checkpoints are RETRAINED (per the release README) and differ
  from the paper's numbers, so CoSDH is not gradeable against 96.83 / 92.99.
* ``agent_drop`` and ``missing_modality`` remove an agent from BOTH the feature
  channel and the late detection channel at once -- a strictly larger
  intervention than on the other three models. Those cells need a footnote.
"""

from __future__ import annotations

import copy
import hashlib
import os

import numpy as np

from .runtime import _InjectionLog, _seed

COSDH_CONTRACT_VERSION = 1

# Shared per-process frame index.
#
# CoSDH calls ``add_noise_data_dict(base_data_dict, noise_setting)`` with NO
# index, so a hook on it cannot see which frame it is perturbing. Seeding on a
# constant would give EVERY frame the identical offset -- a constant pose bias
# that fires, logs plausible magnitudes and passes a naive fire-check, while
# being nothing like per-frame localisation noise.
#
# ``retrieve_base_data(idx)`` DOES receive the index and always runs first for
# the same item in the same process, so the base subclass stashes it here and
# the hook reads it. A module global is correct under forked DataLoader
# workers: each worker has its own copy and processes items sequentially.
_FRAME = {"idx": 0}


def cosdh_fingerprint() -> dict:
    """Content hash of this adapter, mirroring ``wrapper_fingerprint``."""
    here = os.path.abspath(__file__)
    with open(here, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()[:16]
    return {"cosdh_contract_version": COSDH_CONTRACT_VERSION,
            "cosdh_adapter_sha256": h}


# ── pose ────────────────────────────────────────────────────────────────
def make_pose_noise_hook(original, spec, log=None):
    """Wrap CoSDH's ``add_noise_data_dict``.

    The original runs FIRST so ``lidar_pose_clean`` is set to the true pose;
    only then is ``lidar_pose`` perturbed. Reversing that order silently
    destroys the clean reference -- see the module docstring.

    ``lidar_pose`` is 6-DoF ``[x, y, z, roll, yaw, pitch]`` in CoSDH's OPV2V
    convention (``x_to_world`` reads indices 0,1,2 and 4 for yaw), so the
    perturbation is applied to x, y and yaw to match ``generate_noise``.
    """
    if spec is None or spec.pose_error is None:
        return original

    sigma_xy = float(spec.pose_error.get("sigma_xy", 0.0))
    sigma_yaw = float(spec.pose_error.get("sigma_heading", 0.0))

    def hook(data_dict, noise_setting):
        data_dict = original(data_dict, noise_setting)
        _idx = _FRAME["idx"]
        for cav_id, cav in data_dict.items():
            if getattr(spec, "pose_scope", "non-ego") == "non-ego" and cav.get("ego"):
                continue                      # a link fault is the SENDER's
            rng = np.random.default_rng(
                _seed(spec.seed, _idx, cav_id, "pose_error"))
            pose = list(cav["params"]["lidar_pose"])
            dx, dy = rng.normal(0.0, sigma_xy, 2)
            dyaw = rng.normal(0.0, sigma_yaw)
            pose[0] += float(dx)
            pose[1] += float(dy)
            pose[4] += float(dyaw)            # index 4 == yaw, per x_to_world
            cav["params"]["lidar_pose"] = pose
            if log is not None:
                log.write(idx=_idx, agent_id=cav_id,
                          is_ego=int(bool(cav.get("ego"))), stage="pose_error",
                          seed=_seed(spec.seed, _idx, cav_id, "pose_error"),
                          detail="dx=%.4f;dy=%.4f;dyaw=%.4f;theory_xy=%.4f"
                                 % (dx, dy, dyaw, sigma_xy))
        return data_dict

    return hook


# ── non-pose faults, on the base dataset ────────────────────────────────
def make_faulty_base(base_cls, spec, log=None):
    """Subclass ``OPV2VBaseDataset`` so ``retrieve_base_data`` returns a
    corrupted ``base_data_dict``.

    Pose is NOT handled here (see the module docstring) -- only faults that
    act on the point cloud or on agent presence, both of which are settled
    before ``add_noise_data_dict`` runs and are unaffected by it.
    """

    class FaultyCoSDHBase(base_cls):
        fi_spec = spec
        fi_log = log

        def retrieve_base_data(self, idx):
            # stash BEFORE returning: the pose hook fires later in the same
            # item, in this process, and has no other way to learn the frame.
            _FRAME["idx"] = int(idx)
            data = super().retrieve_base_data(idx)
            if spec is None:
                return data
            data = copy.deepcopy(data)
            ego = next((c for c in data if data[c].get("ego")), None)

            def rec(aid, stage, detail, s):
                if log is not None:
                    log.write(idx=idx, agent_id=aid,
                              is_ego=int(aid == ego), stage=stage,
                              seed=s, detail=detail)

            # ---- agent_drop: removes the agent ENTIRELY, which on CoSDH
            # takes out BOTH the feature channel and the late detection
            # channel at once -- a strictly larger intervention than on the
            # three pure-intermediate models. Recorded in the manifest.
            if spec.agent_drop:
                from ..fault_injectors.communication import AgentDropInjector
                for aid in [a for a in list(data) if a != ego]:
                    s = _seed(spec.seed, idx, aid, "agent_drop")
                    if AgentDropInjector(seed=s, **spec.agent_drop)._lost(aid):
                        del data[aid]
                        rec(aid, "agent_drop", "dropped", s)

            # ---- missing_lidar: same dual-channel caveat as agent_drop.
            if spec.missing_lidar:
                from ..fault_injectors.missing_modality import bernoulli_mask
                pdrop = spec.missing_lidar["p_drop_lidar"]
                for aid in [a for a in list(data) if a != ego]:
                    s = _seed(spec.seed, idx, aid, "missing_lidar")
                    if not bernoulli_mask(pdrop, np.random.default_rng(s)):
                        lp = data[aid].get("lidar_np")
                        if lp is not None:
                            data[aid]["lidar_np"] = np.zeros((0, lp.shape[1]),
                                                             dtype=lp.dtype)
                        rec(aid, "missing_lidar", "lidar_dropped", s)

            # ---- latency: swap the sender's whole entry for a STALE frame.
            # CoSDH's retrieve_base_data does its own per-cav load loop, so
            # rather than reimplement it we re-enter the BASE loader at the
            # stale index and take that agent's entry. delay = ts_index -
            # k_stale is <= ts_index by construction, so the stale frame can
            # never precede the start of the scenario.
            if spec.latency:
                from ..fault_injectors.communication import CommLatencyInjector
                scen = 0
                for i, ele in enumerate(self.len_record):
                    if idx < ele:
                        scen = i
                        break
                ts_index = idx if scen == 0 else idx - self.len_record[scen - 1]
                for aid in [a for a in list(data) if a != ego]:
                    s = _seed(spec.seed, idx, aid, "latency")
                    inj = CommLatencyInjector(seed=s, **spec.latency)
                    k_stale, drawn = inj.stale_index(str(aid), ts_index)
                    delay = int(max(0, ts_index - k_stale))
                    if delay <= 0:
                        continue
                    stale = super().retrieve_base_data(idx - delay)
                    if aid in stale:
                        data[aid] = copy.deepcopy(stale[aid])
                        rec(aid, "latency",
                            "delay_frames=%d;drawn=%d;ts_index=%d"
                            % (delay, drawn, ts_index), s)

            # ---- points_reduce (sensor degradation; default scope ALL)
            if spec.points_reduce:
                from ..fault_injectors.lidar_points_reduce import (
                    PointsReductionInjector)
                cfg = dict(spec.points_reduce)
                scope = cfg.pop("scope", "all")
                for aid in list(data):
                    if scope == "non-ego" and aid == ego:
                        continue
                    pts = data[aid].get("lidar_np")
                    if pts is None or not len(pts):
                        continue
                    s = _seed(spec.seed, idx, aid, "points_reduce")
                    out = PointsReductionInjector(seed=s, **cfg)(pts)
                    rec(aid, "points_reduce",
                        "pts=%d->%d" % (len(pts), len(out)), s)
                    data[aid]["lidar_np"] = out

            # ---- weather: environmental, so scope defaults to ALL agents
            for stage in ("lidar_fog", "lidar_snow"):
                cfg = getattr(spec, stage)
                if not cfg:
                    continue
                cfg = dict(cfg)
                scope = cfg.pop("scope", "all")
                if stage == "lidar_fog":
                    from ..fault_injectors.lidar_fog import (
                        LidarFogInjector as _W)
                else:
                    from ..fault_injectors.lidar_snow import (
                        LidarSnowInjector as _W)
                    cfg.setdefault("verbose", False)
                for aid in list(data):
                    if scope == "non-ego" and aid == ego:
                        continue
                    pts = data[aid].get("lidar_np")
                    if pts is None or not len(pts):
                        continue
                    s = _seed(spec.seed, idx, aid, stage)
                    out = _W(seed=s, **cfg)(pts)
                    rec(aid, stage,
                        "pts=%d->%d;meanI=%.4f->%.4f"
                        % (len(pts), len(out),
                           float(np.asarray(pts)[:, 3].mean()),
                           float(np.asarray(out)[:, 3].mean()) if len(out) else 0.0),
                        s)
                    data[aid]["lidar_np"] = out
            return data

    FaultyCoSDHBase.__name__ = "Faulty" + base_cls.__name__
    return FaultyCoSDHBase
