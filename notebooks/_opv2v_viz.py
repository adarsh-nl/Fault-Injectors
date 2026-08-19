"""Plotting + fault-application logic for the OPV2V visualisation notebook.

Split out of `opv2v_visualize.py` so it can be exercised without marimo
installed (it is not, on this account) and so the notebook cells stay short.

READ-ONLY with respect to core: this module IMPORTS `src.datasets` and
`src.fault_injectors` and never modifies them. Where a core injector cannot
run on OPV2V data the fact is recorded in `VIEWABLE` / returned in `notes`,
not worked around.

The governing idea: every plot is in the EGO's coordinate frame, because that
is what cooperative fusion actually operates on. A pose error corrupts the
ego<-collaborator transform, so the collaborator's points land in the WRONG
place in the ego frame -- that misplacement is the whole point of the fault
and it is invisible in world frame, where each agent's cloud is plotted
through its own (corrupted) pose and therefore looks self-consistent.
"""

from __future__ import annotations

import copy

import numpy as np

from src.datasets import load_dataset
from src.fault_injectors import (
    MissingModalityInjector,
    PoseErrorInjector,
    TemporalMisalignmentInjector,
)
from src.fault_injectors import severity as sev

# ── what can actually be shown on OPV2V ────────────────────────────────────
#
# `ok=False` entries are NOT bugs to fix here. They are recorded so the
# notebook can grey them out and say why, per the read-only rule.

VIEWABLE = {
    # fault                 plane        modality  ok    note
    "spatial_misalignment": ("geometry", "lidar", True, ""),
    "temporal_misalignment": ("geometry", "lidar", True, ""),
    "points_reducing": ("corruption", "lidar", True, ""),
    "missing_camera": ("corruption", "camera", True, ""),
    "darkness": ("corruption", "camera", True, ""),
    "brightness": ("corruption", "camera", True, ""),
    "fog_camera": ("corruption", "camera", True, ""),
    "snow_camera": ("corruption", "camera", True, ""),
    "beams_reducing": (
        "corruption", "lidar", False,
        "BeamReductionInjector needs a ring-index column at index 4 "
        "(nuScenes-style (N,5)). OPV2V clouds are (N,4) with no ring, so it "
        "raises ValueError. Not fixed here -- core is read-only.",
    ),
    "snow_lidar": (
        "corruption", "lidar", False,
        "LidarSnowInjector documents 1-3 min PER FRAME (per-point snowflake "
        "sampling). That breaks the login-node rule, so it is disabled in "
        "this notebook. Run it from an sbatch job instead. NOTE it is also "
        "intensity-multiplicative like fog_lidar, so it is likely a no-op on "
        "OPV2V for the same reason -- untested here because of the runtime.",
    ),
    "fog_lidar": (
        "corruption", "lidar", False,
        "Runs without error but is an EXACT no-op on OPV2V, verified at "
        "severity 1 and 3: 0/57171 points changed in xyz or intensity. "
        "OPV2V's intensity column is identically zero (min=max=mean=0.0) and "
        "both fog paths are multiplicative in intensity -- P_R_fog_hard "
        "computes round(exp(-2*alpha*r) * 0) = 0, and in P_R_fog_soft the "
        "response is fog_response * original_intensity[i] * ... = 0, so the "
        "gate `if fog_response > pc[i,3]` is 0 > 0 and never fires. Not a "
        "bug in the injector -- the physical model has nothing to attenuate. "
        "Disabled because an identical clean/faulted pair would read as "
        "'fog does not affect fusion', which is the wrong conclusion.",
    ),
}

LIDAR_FAULTS = [f for f, (_p, m, ok, _n) in VIEWABLE.items() if m == "lidar" and ok]
CAMERA_FAULTS = [f for f, (_p, m, ok, _n) in VIEWABLE.items() if m == "camera" and ok]
SELECTABLE = LIDAR_FAULTS + CAMERA_FAULTS

# Faults that act by corrupting the ego<-collaborator transform or the frame
# index, rather than by corrupting the sensor payload itself.
GEOMETRY_FAULTS = {"spatial_misalignment", "temporal_misalignment"}


# ── geometry ───────────────────────────────────────────────────────────────

def pose_yaw(T):
    """Yaw (radians) of a 4x4 agent->world pose."""
    return float(np.arctan2(T[1, 0], T[0, 0]))


def boxes_in_ego_frame(sample, agent_id=None):
    """
    GT boxes as (cx, cy, length, width, yaw_rad) in the EGO frame.

    OPV2V labels come out of the adapter with `frame='world'` (center =
    location + centre offset, yaw = angle[1] in DEGREES), so they must be
    warped by inv(T_ego_to_world) before they can be drawn next to
    ego-frame points. Boxes are the fixed reference: the injectors perturb
    sensor data and shared poses, never ground truth, so the boxes are
    IDENTICAL in the clean and faulted panels by construction -- which is
    exactly what makes the collaborator's misplacement legible.
    """
    agent = sample.agents[agent_id or sample.ego_id]
    T_world_to_ego = np.linalg.inv(sample.ego.pose)
    ego_yaw = pose_yaw(sample.ego.pose)

    out = []
    for b in agent.labels:
        if b.frame == "world":
            c = T_world_to_ego @ np.array([*b.center[:3], 1.0])
            yaw = np.deg2rad(b.yaw) - ego_yaw
        else:                                    # already agent-frame
            c = np.array([*b.center[:3], 1.0])
            yaw = np.deg2rad(b.yaw)
        out.append((float(c[0]), float(c[1]),
                    float(b.size[0]), float(b.size[1]), float(yaw)))
    return out


def box_corners(cx, cy, length, width, yaw):
    """Closed 2-D footprint (5, 2) for plotting."""
    l, w = length / 2.0, width / 2.0
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w], [l, w]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


# ── fault application ──────────────────────────────────────────────────────

def apply_fault(ds, k, fault, level, seed=0, sample=None, cams=None):
    """
    Return (clean, faulted, notes).

    Both are CooperativeSamples for frame `k` with the same ego. `faulted` is
    a deep copy, so the two can be plotted side by side. Faults act on
    NON-EGO agents only: the ego is the clean reference, matching the
    injectors' own scope (`PoseErrorInjector.apply_to_sample` defaults to
    `protect_ego=True`).
    """
    notes = []
    if fault in VIEWABLE and not VIEWABLE[fault][2]:
        raise ValueError(f"{fault} is not viewable here: {VIEWABLE[fault][3]}")

    def _cams(agent):
        """Only corrupt what will be displayed -- fog_camera costs ~6.6 s per
        camera, so all four would be ~26 s on a shared login node."""
        names = list(agent.images)
        # missing_camera is a Bernoulli draw costing ~0 s, and at p_drop=0.2
        # a single camera usually survives -- showing all four is both free
        # and the only way the drop is reliably visible.
        if cams is None or fault == "missing_camera":
            return names
        return names[:cams]

    load = ("lidar", "labels") if fault in LIDAR_FAULTS else ("lidar", "labels", "images")
    clean = sample if sample is not None else ds.get_sample(k, load=load)
    faulted = copy.deepcopy(clean)
    kwargs = sev.SEVERITY[fault][level]

    if fault == "spatial_misalignment":
        inj = PoseErrorInjector(seed=seed, **kwargs)
        inj.apply_to_sample(faulted, protect_ego=True)
        for aid, ag in faulted.agents.items():
            if "pose_error" in ag.faults:
                e = ag.faults["pose_error"]
                notes.append(f"{aid}: dx={e['dx']:+.2f} dy={e['dy']:+.2f} m, "
                             f"dyaw={e['dyaw']:+.2f} deg")

    elif fault == "temporal_misalignment":
        inj = TemporalMisalignmentInjector(seed=seed, **kwargs)
        # SEMANTIC DIVERGENCE, stated rather than hidden: the core injector's
        # own `inject()` pairs a STALE IMAGE with CURRENT POINTS -- an
        # intra-agent, camera-vs-lidar misalignment. That is not the
        # cooperative failure mode. Here only its shift SAMPLER
        # (`stale_index`) is reused, and the staleness is applied
        # inter-agent: the collaborator transmits an OLD frame, both payload
        # AND the pose it held at that time. Replacing the whole AgentFrame
        # is the faithful version -- a lagging agent does not send
        # yesterday's points stamped with today's pose.
        for aid in faulted.agents:
            if aid == faulted.ego_id:
                continue
            k_stale, _delta = inj.stale_index(k, k_min=0)   # returns a TUPLE
            if k_stale == k:
                notes.append(f"{aid}: shift 0 frames (clamped at k_min=0)")
                continue
            stale = ds.get_sample(k_stale, agents=[aid], load=load)
            faulted.agents[aid] = stale.agents[aid]
            faulted.agents[aid].faults["temporal"] = {"k": k, "k_stale": k_stale}
            dt_ms = (k - k_stale) / ds.fps * 1000
            notes.append(f"{aid}: frame {k} -> {k_stale} "
                         f"({k - k_stale} frames, {dt_ms:.0f} ms stale)")

    elif fault == "missing_camera":
        inj = MissingModalityInjector(seed=seed, **kwargs)
        for aid, ag in faulted.agents.items():
            if ag.is_ego:
                continue
            for cam in _cams(ag):
                r = inj.inject(ag.images[cam], np.zeros((0, 4), np.float32))
                ag.images[cam] = r["image"]
                notes.append(f"{aid}/{cam}: "
                             + ("kept" if r["m_rgb"] else "DROPPED"))

    else:
        inj = sev.make(fault, level, seed=seed)
        for aid, ag in faulted.agents.items():
            if ag.is_ego:
                continue
            if fault in LIDAR_FAULTS:
                n0 = len(ag.lidar)
                ag.lidar = inj(ag.lidar)
                notes.append(f"{aid}: {n0} -> {len(ag.lidar)} points "
                             f"({len(ag.lidar) / max(n0, 1):.1%} kept)")
            else:
                names = _cams(ag)
                for cam in names:
                    ag.images[cam] = inj(ag.images[cam])
                notes.append(f"{aid}: corrupted {', '.join(names)}")

    return clean, faulted, notes


# ── plotting ───────────────────────────────────────────────────────────────

EGO_COLOUR = "#111111"
COLLAB_COLOURS = ["#e4572e", "#2e86ab", "#8332ac", "#3f9b0b", "#d4a017"]


def _subsample(points, frac, rng):
    if frac >= 1.0 or len(points) == 0:
        return points
    n = max(1, int(len(points) * frac))
    return points[rng.choice(len(points), n, replace=False)]


def bev_panel(ax, sample, title, extent=100.0, y_extent=None, max_points=25_000,
              frac=None, show_boxes=True, point_size=0.4):
    """
    One BEV panel in the EGO frame.

    `frac` should be computed ONCE from the clean sample and passed to BOTH
    panels. Subsampling each panel to a fixed COUNT would normalise away the
    density loss that is the entire visible signature of `points_reducing`
    and `fog_lidar`, making a 90 %-dropout panel look identical to the clean
    one.
    """
    y_extent = y_extent or extent
    rng = np.random.default_rng(0)
    ego_id = sample.ego_id
    ids = [ego_id] + [a for a in sample.agents if a != ego_id]

    for i, aid in enumerate(ids):
        agent = sample.agents[aid]
        if agent.lidar is None or len(agent.lidar) == 0:
            continue
        pts = sample.lidar_in_ego_frame(aid)      # <- the fusion-side view
        pts = _subsample(pts, 1.0 if frac is None else frac, rng)
        is_ego = aid == ego_id
        # The ego is context, the collaborators are the subject: the fault acts
        # on them. Ego is drawn faint and small so it does not swamp the
        # collaborators with its dense near-field return.
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=point_size if is_ego else point_size * 2.4,
            c=EGO_COLOUR if is_ego else COLLAB_COLOURS[(i - 1) % len(COLLAB_COLOURS)],
            alpha=0.22 if is_ego else 0.75, linewidths=0, zorder=2 if is_ego else 3,
            label=f"ego {aid}" if is_ego else f"collab {aid}",
        )

    if show_boxes:
        for cx, cy, l, w, yaw in boxes_in_ego_frame(sample):
            if abs(cx) > extent + 10 or abs(cy) > y_extent + 10:
                continue
            c = box_corners(cx, cy, l, w, yaw)
            ax.plot(c[:, 0], c[:, 1], color="#00a878", lw=0.9, zorder=5)

    ax.plot(0, 0, marker="*", ms=15, color="#ffd400",
            mec="black", mew=0.6, zorder=6)          # ego origin
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m] (ego forward)", fontsize=8)
    ax.set_ylabel("y [m] (ego left)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, lw=0.4)
    return ax


def sampling_fraction(sample, max_points=25_000):
    """The one fraction to use for both panels (see `bev_panel`)."""
    biggest = max((len(a.lidar) for a in sample.agents.values()
                   if a.lidar is not None), default=0)
    return 1.0 if biggest <= max_points else max_points / biggest
