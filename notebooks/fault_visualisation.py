# -*- coding: utf-8 -*-
"""Fault visualisation -- every injector on every dataset at every tested
severity, as ONE self-contained marimo notebook + a CLI export driver.

Supersedes notebooks/fault_injection_visualisation.ipynb (kept in place).

Three ways to use this file:

    marimo edit notebooks/fault_visualisation.py      # interactive (.venv-hpc)
    python notebooks/fault_visualisation.py --export --datasets v2xset,opv2v
                                                      # opencood-official env
    python notebooks/fault_visualisation.py --export --datasets griffin
                                                      # .venv-hpc env

Everything the interactive layer shows is read from the exported artifacts
(results/fault_injector_visualisation/), so the notebook works in an env
where a dataset's loader is not importable, and results are viewable with no
Python at all (MP4s + PNGs + INDEX.md).

INJECTION IS NEVER REIMPLEMENTED HERE. OpenCOOD datasets go through the
sweep's exact seam (src.adapters.make_faulty_dataset over the unmodified
IntermediateFusionDataset); Griffin goes through FaultedGriffinDataset. Seed
1234 -- every visualised frame is a real injected sample from the sweep's
distribution.

Visual language (consistent everywhere):
    agents      fixed palette by agent order (ego always blue)
    intensity   viridis        depth   jet_r
    snow flags  blue unchanged / yellow attenuated / red snow-scatter
    BEV         dark bg, ego marker at origin, fixed +-BEV_R m axes
    titles      carry the quantitative fact (pts, fractions, metres)
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

# ── environment adaptivity (stated, never silent) ───────────────────────────
HAS_OPENCOOD = importlib.util.find_spec("opencood") is not None
HAS_PLYFILE = importlib.util.find_spec("plyfile") is not None
HAS_FFMPEG = os.system("ffmpeg -version > /dev/null 2>&1") == 0

OUT_ROOT = os.path.join(_REPO, "results", "fault_injector_visualisation")
PACKS = os.path.join(OUT_ROOT, "packs")
SWEEP_CSV = os.path.join(_REPO, "results", "sweep", "sweep_results.csv")

V2XSET_ROOT = "/datasets/eemcs/ps/cv/opencood/v2xset/test"
OPV2V_ROOT = "/datasets/eemcs/ps/cv/opencood/opv2v/test"
GRIFFIN_ROOT = ("/datasets/eemcs/ps/cv/huggingface/griffin/datasets/"
                "griffin_50scenes_25m/griffin_50scenes_25m/griffin-release")
OC_HYPES = {"v2xset": os.path.expanduser("~/opencood-eval/cobevt/config.yaml"),
            "opv2v": os.path.expanduser("~/opencood-eval/where2comm/config.yaml")}
OC_DATA = {"v2xset": V2XSET_ROOT, "opv2v": OPV2V_ROOT}

SEED = 1234
BEV_R = 50.0
ANIM_N = 30           # frames per animation
ANIM_FPS = 5
DARK = "#0d0d0d"
AGENT_COLORS = ["#3a86ff", "#ff2d55", "#ffbe0b", "#2dd4bf", "#a78bfa",
                "#f97316", "#84cc16"]          # ego first, always blue
FLAG_COLORS = np.array(["#3a86ff", "#ffbe0b", "#ff2d55"])
MAX_PTS = 9000        # per-agent subsample for packs / scatter speed

# The grid's tiers, restated for filenames and titles (kept in sync with
# tools/sweep/grid.py by the applicability test in the export driver).
TIERS = {
    "pose_error": [("sev1_0.2m", {"pose_error": {"sigma_xy": .2, "sigma_heading": .2}}),
                   ("sev2_0.4m", {"pose_error": {"sigma_xy": .4, "sigma_heading": .4}}),
                   ("sev3_0.6m", {"pose_error": {"sigma_xy": .6, "sigma_heading": .6}})],
    "comm_latency": [("sev1_100ms", {"latency": {"mu_delay": .1, "sigma_jitter": 0.}}),
                     ("sev2_200ms", {"latency": {"mu_delay": .2, "sigma_jitter": 0.}}),
                     ("sev3_300ms", {"latency": {"mu_delay": .3, "sigma_jitter": 0.}})],
    "agent_drop": [("p0.25", {"agent_drop": {"p_drop": .25}}),
                   ("p0.50", {"agent_drop": {"p_drop": .50}}),
                   ("p0.75", {"agent_drop": {"p_drop": .75}})],
    "missing_modality": [("p0.25", {"missing_lidar": {"p_drop_lidar": .25}}),
                         ("p0.50", {"missing_lidar": {"p_drop_lidar": .50}}),
                         ("p0.75", {"missing_lidar": {"p_drop_lidar": .75}})],
    "points_reduce": [("keep30", {"points_reduce": {"severity": 1}}),
                      ("keep20", {"points_reduce": {"severity": 2}}),
                      ("keep10", {"points_reduce": {"severity": 3}})],
    "lidar_fog": [("sev1", {"lidar_fog": {"severity": 1}}),
                  ("sev2", {"lidar_fog": {"severity": 2}}),
                  ("sev3", {"lidar_fog": {"severity": 3}})],
    "lidar_snow": [("sev1", {"lidar_snow": {"severity": 1, "mount_height": 1.9}}),
                   ("sev2", {"lidar_snow": {"severity": 2, "mount_height": 1.9}}),
                   ("sev3", {"lidar_snow": {"severity": 3, "mount_height": 1.9}})],
}

# ── the applicability matrix (the landing table; never silently skipped) ────
# dataset -> injector -> True | reason string for non-applicability
APPLICABILITY = {
    "v2xset": {
        "pose_error": True, "comm_latency": True, "agent_drop": True,
        "missing_modality": True, "points_reduce": True, "lidar_fog": True,
        "lidar_snow": True,
        "camera_faults": "LiDAR-only through OpenCOOD (camera PNGs never "
                         "loaded); blocked structurally by the modality gate",
        "occlusion": "camera fault on a LiDAR-only pipeline",
        "beam_reduce": "needs a ring/beam column these CARLA clouds lack",
    },
    "opv2v": None,   # same as v2xset (filled below)
    "griffin": {
        "pose_error": "NOT WIRED by the approved Griffin routing (spec: PAC-"
                      "era protocol excluded PoseError on Griffin)",
        "comm_latency": True, "agent_drop": True, "missing_modality": True,
        "points_reduce": True, "lidar_fog": True, "lidar_snow": True,
        "camera_faults": True, "occlusion": True, "beam_reduce": True,
    },
}
APPLICABILITY["opv2v"] = dict(APPLICABILITY["v2xset"])


def applicable(dataset: str, injector: str) -> bool:
    return APPLICABILITY[dataset][injector] is True


# ── sweep results, for section intros and the snow mechanism figure ─────────

def sweep_rows():
    if not os.path.exists(SWEEP_CSV):
        return []
    return list(csv.DictReader(open(SWEEP_CSV)))


def sweep_line(injector: str, model: str = "cobevt") -> str:
    """One-line measured summary for a section intro."""
    name = {"comm_latency": "latency"}.get(injector, injector)
    rows = [r for r in sweep_rows()
            if r["model"] == model and r["injector"] == name]
    if not rows:
        return "(sweep results not available in results/sweep yet)"
    by = {}
    for r in rows:
        by.setdefault(r["severity_tier"], []).append(float(r["delta_ap70"]))
    parts = ["%s %+0.3f" % (t, float(np.mean(v)))
             for t, v in by.items() if t != "clean"]
    return ("measured on CoBEVT/V2XSet, delta AP@0.7 vs clean: "
            + ", ".join(parts))


# ── loaders: the SWEEP's seams, verbatim ────────────────────────────────────

_OC_CACHE = {}


def _oc_dataset(dataset: str, spec_kwargs):
    """(Fault-wrapped) IntermediateFusionDataset via the sweep's exact seam.
    opencood-official env only."""
    assert HAS_OPENCOOD, "opencood not importable in this env"
    key = (dataset, json.dumps(spec_kwargs, sort_keys=True))
    if key in _OC_CACHE:
        return _OC_CACHE[key]
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import IntermediateFusionDataset
    from src.adapters import FaultSpec, make_faulty_dataset
    hypes = yaml_utils.load_yaml(OC_HYPES[dataset])
    hypes["root_dir"] = OC_DATA[dataset]
    hypes["validate_dir"] = OC_DATA[dataset]
    cls = IntermediateFusionDataset
    if spec_kwargs is not None:
        cls = make_faulty_dataset(cls, FaultSpec(seed=SEED, **spec_kwargs))
    ds = cls(params=hypes, visualize=False, train=False)
    _OC_CACHE[key] = ds
    return ds


def _oc_scene_offset(ds, min_cav: int = 3):
    """First frame of the scenario with the most cavs (>=min_cav preferred),
    so agent colours are interesting."""
    best, best_n = 0, 0
    prev = 0
    for i, upper in enumerate(ds.len_record):
        n = len(ds.scenario_database[i])
        if n > best_n:
            best, best_n = prev, n
        prev = upper
        if best_n >= min_cav:
            break
    return best, best_n


def oc_frame(dataset: str, spec_kwargs, idx: int):
    """One frame -> list of (agent_id, is_ego, pts_ego_frame(N,4), meta).
    Points are warped by params['transformation_matrix'] -- exactly the
    tensor the fused model consumes, so pose error shows as the model sees
    it."""
    ds = _oc_dataset(dataset, spec_kwargs)
    base = ds.retrieve_base_data(idx)
    out = []
    for cav_id, e in base.items():
        T = np.asarray(e["params"]["transformation_matrix"], dtype=np.float64)
        pts = np.asarray(e["lidar_np"], dtype=np.float64)
        if len(pts):
            xyz1 = np.c_[pts[:, :3], np.ones(len(pts))]
            w = pts.copy()
            w[:, :3] = (T @ xyz1.T).T[:, :3]
        else:
            w = pts
        out.append({"agent": str(cav_id), "ego": bool(e["ego"]), "pts": w,
                    "delay": int(e.get("time_delay", 0)),
                    "speed": float(e["params"].get("ego_speed", 0.0))})
    return out


_GRIFFIN = {}


def _griffin(spec_kwargs):
    key = json.dumps(spec_kwargs, sort_keys=True) if spec_kwargs else "none"
    if key in _GRIFFIN:
        return _GRIFFIN[key]
    from src.adapters.griffin import FaultedGriffinDataset, GriffinFaultSpec
    from src.datasets import load_dataset
    ds = load_dataset("griffin", veh_root=GRIFFIN_ROOT + "/vehicle-side",
                      drone_root=GRIFFIN_ROOT + "/drone-side")
    if spec_kwargs is None:
        _GRIFFIN[key] = ("plain", ds)
    else:
        _GRIFFIN[key] = ("faulted",
                         FaultedGriffinDataset(ds, GriffinFaultSpec(
                             seed=SEED, **spec_kwargs)))
    return _GRIFFIN[key]


def griffin_sample(spec_kwargs, k: int, load=("lidar", "images")):
    kind, ds = _griffin(spec_kwargs)
    return ds.get_sample(k, load=load) if kind != "plain" \
        else ds.get_sample(k, load=load)


# ── renderers ───────────────────────────────────────────────────────────────

def _sub(p, n=MAX_PTS):
    if len(p) <= n:
        return p
    idx = np.random.RandomState(0).choice(len(p), n, replace=False)
    return p[idx]


def bev_by_agent(ax, agents, title="", counter=True):
    """THE cooperative view: one colour per agent, ego blue at origin."""
    ax.set_facecolor(DARK)
    n_present = 0
    for i, a in enumerate(agents):
        p = _sub(np.asarray(a["pts"]))
        if len(p):
            n_present += 1
        ax.scatter(p[:, 0], p[:, 1], s=0.4,
                   c=AGENT_COLORS[i % len(AGENT_COLORS)],
                   label="%s%s (%d pts)" % ("ego " if a["ego"] else "agent ",
                                            a["agent"], len(a["pts"])))
    ax.plot(0, 0, "w^", ms=8)
    ax.set_xlim(-BEV_R, BEV_R)
    ax.set_ylim(-BEV_R, BEV_R)
    ax.set_aspect("equal")
    ax.axis("off")
    extra = ("   agents with points: %d/%d" % (n_present, len(agents))
             if counter else "")
    ax.set_title(title + extra, color="white", fontsize=10)
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.2,
                    labelcolor="white")
    leg.get_frame().set_facecolor(DARK)


def bev_scalar(ax, pts, title="", color="intensity"):
    ax.set_facecolor(DARK)
    p = _sub(np.asarray(pts))
    if color == "intensity":
        c, cmap = p[:, 3], "viridis"
    else:
        c, cmap = np.linalg.norm(p[:, :2], axis=1), "jet_r"
    ax.scatter(p[:, 0], p[:, 1], c=c, cmap=cmap, s=0.4)
    ax.plot(0, 0, "w^", ms=7)
    ax.set_xlim(-BEV_R, BEV_R)
    ax.set_ylim(-BEV_R, BEV_R)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("%s\n%d pts" % (title, len(pts)), color="white", fontsize=10)


def agent_status_strip(ax, statuses):
    """The node view: present-with-points / present-ZERO-points / absent.
    Present-but-empty is the missing-modality signature -- the node stays in
    the fusion graph (pose known, consumes attention) while contributing
    nothing; our sweep measured that this degrades slightly WORSE than
    dropping the agent outright."""
    ax.set_facecolor(DARK)
    ax.set_xlim(-0.5, len(statuses) - 0.5)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    for i, s in enumerate(statuses):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        if s["state"] == "absent":
            ax.plot(i, 0, "x", color="#666666", ms=14, mew=3)
            lbl = "DROPPED\n(node gone)"
        elif s["state"] == "empty":
            ax.plot(i, 0, "o", mfc="none", mec=col, ms=16, mew=2)
            lbl = "present\n0 pts"
        else:
            ax.plot(i, 0, "o", color=col, ms=16)
            lbl = "present\n%d pts" % s["pts"]
        ax.annotate("%s\n%s" % (s["agent"], lbl), (i, 0), (i, -0.85),
                    color="white", fontsize=8, ha="center")
    ax.set_title("fusion-graph nodes", color="white", fontsize=10)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return path


def animate(path, n_frames, draw, figsize=(12, 6), fps=ANIM_FPS):
    """draw(fig, k) renders frame k onto a cleared fig. mp4 (ffmpeg) or gif."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not HAS_FFMPEG and path.endswith(".mp4"):
        path = path[:-4] + ".gif"
    writer = (FFMpegWriter(fps=fps, bitrate=2500) if path.endswith(".mp4")
              else PillowWriter(fps=fps))
    fig = plt.figure(figsize=figsize, facecolor=DARK)
    with writer.saving(fig, path, dpi=100):
        for k in range(n_frames):
            fig.clf()
            draw(fig, k)
            writer.grab_frame()
    plt.close(fig)
    return path


# ── export driver ───────────────────────────────────────────────────────────

INDEX = []


def note(path, desc):
    INDEX.append((os.path.relpath(path, OUT_ROOT), desc))
    print("[export] %s" % os.path.relpath(path, OUT_ROOT), flush=True)


def pack_save(name, obj):
    os.makedirs(PACKS, exist_ok=True)
    path = os.path.join(PACKS, name + ".npz")
    np.savez_compressed(path, **obj)
    note(path, "frame pack for the interactive scrubbers")


def _oc_agents_pack(dataset, injector, n_frames=12):
    """Frames x severities of per-agent ego-frame clouds -> pack + arrays for
    rendering. Severity axis at FIXED frames = the dose-scrub mode."""
    off, ncav = _oc_scene_offset(_oc_dataset(dataset, None))
    sevs = TIERS[injector]
    data = {"n_frames": n_frames, "n_sev": len(sevs),
            "sev_names": np.array([s[0] for s in sevs])}
    for si, (sname, kw) in enumerate([("clean", None)] + sevs):
        for k in range(n_frames):
            agents = oc_frame(dataset, kw, off + k)
            for ai, a in enumerate(agents):
                p = _sub(np.asarray(a["pts"], dtype=np.float32), 6000)
                data["s%d_f%d_a%d" % (si, k, ai)] = p.astype(np.float16)
                data["s%d_f%d_a%d_meta" % (si, k, ai)] = np.array(
                    [1.0 if a["ego"] else 0.0, a["delay"], a["speed"],
                     len(a["pts"])], dtype=np.float32)
    return data, off, ncav


def export_pose_error(dataset):
    inj = "pose_error"
    if not applicable(dataset, inj):
        print("[n/a] %s on %s: %s" % (inj, dataset,
                                      APPLICABILITY[dataset][inj]))
        return
    off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
    clean0 = oc_frame(dataset, None, off)
    sheet = plt.figure(figsize=(19, 5), facecolor=DARK)
    for si, (sname, kw) in enumerate(TIERS[inj]):
        # animation: clean ego + faulty collaborators, per-agent colours
        def draw(fig, k, kw=kw, sname=sname):
            agents = oc_frame(dataset, kw, off + k)
            ax = fig.add_subplot(111)
            bev_by_agent(ax, agents,
                         "%s pose_error %s  frame %d -- collaborator clouds "
                         "ghost against the ego's" % (dataset, sname, k))
        p = animate(os.path.join(OUT_ROOT, inj, "%s_%s.mp4"
                                 % (dataset, sname)), ANIM_N, draw)
        note(p, "%s: per-agent BEV, collaborator misalignment at %s"
             % (dataset, sname))

        # contact-sheet panel with displacement vectors (same point order --
        # only the transform differs -- so vectors ARE the displacement field)
        fx = oc_frame(dataset, kw, off)
        ax = sheet.add_subplot(1, 3, si + 1)
        ax.set_facecolor(DARK)
        disp = []
        for ai, (ca, fa) in enumerate(zip(clean0, fx)):
            cp, fp = np.asarray(ca["pts"]), np.asarray(fa["pts"])
            n = min(len(cp), len(fp))
            cp, fp = cp[:n], fp[:n]
            col = AGENT_COLORS[ai % len(AGENT_COLORS)]
            ax.scatter(_sub(fp, 4000)[:, 0], _sub(fp, 4000)[:, 1], s=0.3,
                       c=col)
            if not ca["ego"] and n:
                d = np.linalg.norm(fp[:, :2] - cp[:, :2], axis=1)
                disp.append(d.mean())
                q = np.random.RandomState(1).choice(n, min(25, n), False)
                ax.quiver(cp[q, 0], cp[q, 1],
                          (fp - cp)[q, 0], (fp - cp)[q, 1],
                          color="white", width=0.003,
                          angles="xy", scale_units="xy", scale=1.0)
        ax.set_xlim(-BEV_R, BEV_R)
        ax.set_ylim(-BEV_R, BEV_R)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("%s  mean |disp| = %.2f m" % (sname, np.mean(disp)),
                     color="white", fontsize=11)
    p = save_fig(sheet, os.path.join(OUT_ROOT, inj,
                                     "%s_contact.png" % dataset))
    note(p, "%s: displacement-vector contact sheet, all severities" % dataset)
    pack_save("%s_%s" % (inj, dataset), _oc_agents_pack(dataset, inj)[0])


def export_comm_latency(dataset):
    inj = "comm_latency"
    if dataset == "griffin":
        return _export_latency_griffin()
    off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
    for sname, kw in TIERS[inj]:
        def draw(fig, k, kw=kw, sname=sname):
            agents = oc_frame(dataset, kw, off + k)
            stale = [a for a in agents if not a["ego"] and a["delay"] > 0]
            m = np.mean([a["speed"] * a["delay"] * 0.1 for a in stale]) \
                if stale else 0.0
            ax = fig.add_subplot(111)
            bev_by_agent(ax, agents,
                         "%s latency %s  frame %d -- stale collaborators "
                         "trail the live ego (staleness ~= %.1f m at their "
                         "speed)" % (dataset, sname, k, m))
        p = animate(os.path.join(OUT_ROOT, inj, "%s_%s.mp4"
                                 % (dataset, sname)), ANIM_N, draw)
        note(p, "%s: stale collaborator clouds trailing at %s"
             % (dataset, sname))
    pack_save("%s_%s" % (inj, dataset), _oc_agents_pack(dataset, inj)[0])


def _export_latency_griffin():
    """Griffin: the drone is the cooperating agent; latency = the ego pairs
    with the drone's STALE frame (scene-clamped). Visual: vehicle front
    camera (current) beside drone bottom camera (stale)."""
    inj = "comm_latency"
    for sname, kw in TIERS[inj]:
        gkw = {"latency": kw["latency"]}

        def draw(fig, k, gkw=gkw, sname=sname):
            s = griffin_sample(gkw, 20 + k, load=("images",))
            veh = s.agents["vehicle"].images.get("front")
            dr = s.agents["drone"].images.get("bottom")
            fault = s.agents["drone"].faults.get("comm_latency", {})
            for i, (img, t) in enumerate((
                    (veh, "vehicle front  (current frame %d)" % (20 + k)),
                    (dr, "drone bottom  (STALE frame %s, delta %s frames)"
                     % (fault.get("frame_used", "?"),
                        fault.get("delta_frames", "?"))))):
                ax = fig.add_subplot(1, 2, i + 1)
                if img is not None:
                    ax.imshow(img)
                ax.axis("off")
                ax.set_title(t, color="white", fontsize=10)
        p = animate(os.path.join(OUT_ROOT, inj, "griffin_%s.mp4" % sname),
                    ANIM_N, draw)
        note(p, "griffin: drone camera lags the vehicle by %s (scene-clamped)"
             % sname)


def export_agent_drop(dataset):
    inj = "agent_drop"
    if dataset == "griffin":
        return _export_drop_missing_griffin(inj)
    off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
    for sname, kw in TIERS[inj]:
        def draw(fig, k, kw=kw, sname=sname):
            clean = oc_frame(dataset, None, off + k)
            faulty = oc_frame(dataset, kw, off + k)
            fmap = {a["agent"]: a for a in faulty}
            agents, statuses = [], []
            for a in clean:
                fa = fmap.get(a["agent"])
                if fa is None:
                    agents.append({**a, "pts": a["pts"][:0]})
                    statuses.append({"agent": a["agent"], "state": "absent",
                                     "pts": 0})
                else:
                    agents.append(fa)
                    statuses.append({"agent": a["agent"], "state": "alive",
                                     "pts": len(fa["pts"])})
            gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
            bev_by_agent(fig.add_subplot(gs[0]), agents,
                         "%s agent_drop %s  frame %d" % (dataset, sname, k))
            agent_status_strip(fig.add_subplot(gs[1]), statuses)
        p = animate(os.path.join(OUT_ROOT, inj, "%s_%s.mp4"
                                 % (dataset, sname)), ANIM_N, draw,
                    figsize=(14, 6))
        note(p, "%s: agent colours vanish at %s; node strip shows DROPPED"
             % (dataset, sname))
    pack_save("%s_%s" % (inj, dataset), _oc_agents_pack(dataset, inj)[0])


def export_missing_modality(dataset):
    inj = "missing_modality"
    if dataset == "griffin":
        return _export_drop_missing_griffin(inj)
    off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
    for sname, kw in TIERS[inj]:
        def draw(fig, k, kw=kw, sname=sname):
            faulty = oc_frame(dataset, kw, off + k)
            statuses = [{"agent": a["agent"],
                         "state": ("alive" if len(a["pts"]) else "empty"),
                         "pts": len(a["pts"])} for a in faulty]
            gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
            bev_by_agent(fig.add_subplot(gs[0]), faulty,
                         "%s missing_modality %s  frame %d -- LiDAR gone, "
                         "agent still PRESENT" % (dataset, sname, k))
            agent_status_strip(fig.add_subplot(gs[1]), statuses)
        p = animate(os.path.join(OUT_ROOT, inj, "%s_%s.mp4"
                                 % (dataset, sname)), ANIM_N, draw,
                    figsize=(14, 6))
        note(p, "%s at %s: hollow node = present with 0 pts (contrast "
             "agent_drop's absent node)" % (dataset, sname))
    pack_save("%s_%s" % (inj, dataset), _oc_agents_pack(dataset, inj)[0])


def _export_drop_missing_griffin(inj):
    """Griffin drop/missing target the DRONE (only droppable agent). Visual:
    vehicle BEV + drone bottom camera + node strip."""
    key = "agent_drop" if inj == "agent_drop" else "missing_modality"
    tiers = TIERS[key]
    for sname, kw in tiers:
        gkw = ({"agent_drop": kw["agent_drop"]} if key == "agent_drop"
               else {"missing_modality":
                     {"p_drop_rgb": kw["missing_lidar"]["p_drop_lidar"]}})

        def draw(fig, k, gkw=gkw, sname=sname):
            s = griffin_sample(gkw, 20 + k, load=("lidar", "images"))
            veh = s.agents["vehicle"]
            drone = s.agents.get("drone")
            gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1])
            bev_by_agent(fig.add_subplot(gs[0]),
                         [{"agent": "vehicle", "ego": True,
                           "pts": veh.lidar, "delay": 0, "speed": 0}],
                         "griffin %s %s  frame %d" % (key, sname, k))
            ax = fig.add_subplot(gs[1])
            img = drone.images.get("bottom") if drone else None
            if img is not None and img.any():
                ax.imshow(img)
                dstate = "alive"
            elif drone is not None:
                ax.imshow(np.zeros((10, 10, 3), np.uint8))
                dstate = "empty"
            else:
                ax.set_facecolor(DARK)
                dstate = "absent"
            ax.axis("off")
            ax.set_title("drone bottom camera", color="white", fontsize=10)
            agent_status_strip(fig.add_subplot(gs[2]), [
                {"agent": "vehicle", "state": "alive",
                 "pts": len(veh.lidar) if veh.lidar is not None else 0},
                {"agent": "drone", "state": dstate,
                 "pts": 0 if dstate != "alive" else 1}])
        p = animate(os.path.join(OUT_ROOT, key, "griffin_%s.mp4" % sname),
                    ANIM_N, draw, figsize=(16, 5))
        note(p, "griffin %s at %s (drone is the cooperating agent)"
             % (key, sname))


def export_lidar_scalar(dataset, inj):
    """points_reduce / lidar_fog on any dataset: intensity/depth BEV."""
    color = "intensity" if inj == "lidar_fog" else "depth"
    if dataset == "griffin":
        clean = griffin_sample(None, 20, load=("lidar",)).agents["vehicle"].lidar
        get = lambda kw, k: griffin_sample(
            {inj: kw[inj]} if inj in kw else
            {"points_reduce": kw["points_reduce"]},
            20 + k, load=("lidar",)).agents["vehicle"].lidar
    else:
        off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
        clean = next(a for a in oc_frame(dataset, None, off) if a["ego"])["pts"]
        get = lambda kw, k: next(a for a in oc_frame(dataset, kw, off + k)
                                 if a["ego"])["pts"]
    sheet = plt.figure(figsize=(19, 5), facecolor=DARK)
    ax = None
    for si, (sname, kw) in enumerate(TIERS[inj]):
        def draw(fig, k, kw=kw, sname=sname):
            pts = get(kw, k)
            ax = fig.add_subplot(111)
            bev_scalar(ax, pts, "%s %s %s  frame %d"
                       % (dataset, inj, sname, k), color)
        p = animate(os.path.join(OUT_ROOT, inj, "%s_%s.mp4"
                                 % (dataset, sname)), ANIM_N, draw,
                    figsize=(7, 7))
        note(p, "%s %s at %s (%s-coloured BEV)" % (dataset, inj, sname, color))
        bev_scalar(sheet.add_subplot(1, 3, si + 1), get(kw, 0),
                   "%s" % sname, color)
    p = save_fig(sheet, os.path.join(OUT_ROOT, inj,
                                     "%s_contact.png" % dataset))
    note(p, "%s %s severity contact sheet" % (dataset, inj))


def export_lidar_snow(dataset):
    """Snow: stills per severity + flags + the removal-vs-AP mechanism
    figure + one short animation. Compute-heavy (20-45 s/cloud) -- progress
    printed per severity."""
    inj = "lidar_snow"
    if dataset == "griffin":
        gclean = griffin_sample(None, 20, load=("lidar",))
        clean = gclean.agents["vehicle"].lidar

        def snowed(kw, k):
            s = griffin_sample({"lidar_snow": {"severity":
                                kw["lidar_snow"]["severity"]}},
                               20 + k, load=("lidar",))
            return s.agents["vehicle"].lidar
    else:
        off, _ = _oc_scene_offset(_oc_dataset(dataset, None))
        clean = next(a for a in oc_frame(dataset, None, off) if a["ego"])["pts"]

        def snowed(kw, k):
            return next(a for a in oc_frame(dataset, kw, off + k)
                        if a["ego"])["pts"]

    sheet = plt.figure(figsize=(24, 5.6), facecolor=DARK)
    bev_scalar(sheet.add_subplot(1, 4, 1), clean, "clean", "intensity")
    removed = []
    for si, (sname, kw) in enumerate(TIERS[inj]):
        print("[snow] %s %s computing..." % (dataset, sname), flush=True)
        pts = snowed(kw, 0)
        removed.append(1.0 - len(pts) / max(len(clean), 1))
        bev_scalar(sheet.add_subplot(1, 4, si + 2), pts,
                   "%s  removed %.0f%%" % (sname, removed[-1] * 100),
                   "intensity")
    p = save_fig(sheet, os.path.join(OUT_ROOT, inj,
                                     "%s_contact.png" % dataset))
    note(p, "%s snow: clean + all severities side by side" % dataset)

    # THE mechanism figure: removed_frac inverts, AP degrades monotonically
    rows = [r for r in sweep_rows() if r["injector"] == "lidar_snow"
            and r["model"] == "cobevt"]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    xs = [1, 2, 3]
    if rows:
        rf = [float(r["extra_metric"].split("removed_frac=")[1])
              for r in sorted(rows, key=lambda r: r["severity_value"])]
        d70 = [float(r["delta_ap70"])
               for r in sorted(rows, key=lambda r: r["severity_value"])]
        ax1.plot(xs, rf, "o-", color="#3a86ff", label="removed_frac (sweep)")
        ax2 = ax1.twinx()
        ax2.plot(xs, d70, "s-", color="#ff2d55", label="delta AP@0.7 (sweep)")
        ax2.set_ylabel("delta AP@0.7", color="#ff2d55")
    ax1.plot(xs, removed, "^--", color="#2dd4bf",
             label="removed_frac (this %s cloud)" % dataset)
    ax1.set_xticks(xs)
    ax1.set_xlabel("snow severity")
    ax1.set_ylabel("fraction of points removed", color="#3a86ff")
    ax1.set_title("Snow: removal INVERTS with severity while AP degrades "
                  "monotonically\n-- severity acts through scatter "
                  "corruption, not removal volume")
    fig.legend(loc="lower left", fontsize=8)
    p = save_fig(fig, os.path.join(OUT_ROOT, inj,
                                   "%s_mechanism.png" % dataset))
    note(p, "%s: the paper figure -- removal fraction vs AP by severity"
         % dataset)

    def draw(fig, k):
        pts = snowed(TIERS[inj][1][1], k)         # sev2
        bev_scalar(fig.add_subplot(111), pts,
                   "%s snow sev2  frame %d" % (dataset, k), "intensity")
    p = animate(os.path.join(OUT_ROOT, inj, "%s_sev2.mp4" % dataset), 5,
                draw, figsize=(7, 7), fps=2)
    note(p, "%s snow sev2 short animation (compute-bound: 5 frames)"
         % dataset)


def export_griffin_only_faults():
    """Camera faults / occlusion / beam_reduce -- Griffin per applicability;
    ports of the old notebook's matrices and status views."""
    from src.data_loaders import (load_calib_griffin, load_image, load_lidar,
                                  load_sensor_extrinsic)
    from src.fault_injectors import (BeamReductionInjectorGriffin,
                                     BrightnessInjector, DarknessInjector,
                                     FogInjector, SnowInjector)
    from src.fault_injectors.sensor_occlusion import (OcclusionConfig,
                                                      SensorOcclusionInjector)
    veh = GRIFFIN_ROOT + "/vehicle-side"
    dr = GRIFFIN_ROOT + "/drone-side"
    k = 20
    vf = load_image(sorted(glob.glob(veh + "/camera/front/*.png"))[k])
    db = load_image(sorted(glob.glob(dr + "/camera/bottom/*.png"))[k])
    lid = load_lidar(sorted(glob.glob(veh + "/lidar/lidar_top/*.ply"))[k])

    # camera-fault matrices: rows [veh front, drone bottom] x [clean, s1-3]
    for name, Inj in (("brightness", BrightnessInjector),
                      ("darkness", DarknessInjector),
                      ("fog_camera", FogInjector),
                      ("snow_camera", SnowInjector)):
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), facecolor=DARK)
        for r, (cam, img) in enumerate((("vehicle front", vf),
                                        ("drone bottom", db))):
            panels = [("clean", img)] + [("sev %d" % s, Inj(s)(img))
                                         for s in (1, 2, 3)]
            for c, (lbl, im) in enumerate(panels):
                axes[r, c].imshow(im)
                axes[r, c].axis("off")
                axes[r, c].set_title("%s -- %s" % (cam, lbl), color="white",
                                     fontsize=9)
        p = save_fig(fig, os.path.join(OUT_ROOT, "camera_faults",
                                       "griffin_%s.png" % name))
        note(p, "griffin %s matrix: cameras x severities" % name)

    # occlusion: dirt / scratch / crack still + one fixed-contaminant anim
    # OcclusionConfig expresses coverage as severity_mode="coverage" plus
    # `severity` (the target coverage c) -- there is no `coverage=` kwarg.
    effects = [("dirt 35% coverage",
                dict(kind="dirt", severity_mode="coverage", severity=0.35)),
               ("scratch 25% coverage",
                dict(kind="scratch", severity_mode="coverage", severity=0.25)),
               ("crack", dict(kind="crack"))]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), facecolor=DARK)
    axes[0].imshow(vf)
    axes[0].axis("off")
    axes[0].set_title("clean", color="white")
    for i, (nm, kwv) in enumerate(effects):
        out = SensorOcclusionInjector(OcclusionConfig(seed=42, **kwv))(vf)
        img = out[0] if isinstance(out, tuple) else out
        axes[i + 1].imshow(img)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(nm, color="white")
    p = save_fig(fig, os.path.join(OUT_ROOT, "occlusion",
                                   "griffin_contact.png"))
    note(p, "griffin occlusion: clean | dirt | scratch | crack")

    files = sorted(glob.glob(veh + "/camera/front/*.png"))
    inj_fix = SensorOcclusionInjector(OcclusionConfig(
        kind="dirt", severity_mode="coverage", severity=0.35,
        temporal="persistent", seed=42))

    def draw(fig, kk):
        img = load_image(files[k + kk])
        out = inj_fix(img)
        img2 = out[0] if isinstance(out, tuple) else out
        for i, (t, im) in enumerate((("clean", img),
                                     ("dirt (FIXED on the lens)", img2))):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.imshow(im)
            ax.axis("off")
            ax.set_title(t, color="white", fontsize=10)
    p = animate(os.path.join(OUT_ROOT, "occlusion", "griffin_dirt.mp4"),
                ANIM_N, draw)
    note(p, "griffin: persistent lens dirt while the scene moves")

    # beam reduce (griffin variant; N/A on CARLA pcds -- stated in INDEX)
    sheet = plt.figure(figsize=(19, 5), facecolor=DARK)
    for si, s in enumerate((1, 2, 3)):
        out = BeamReductionInjectorGriffin(s)(lid)
        bev_scalar(sheet.add_subplot(1, 3, si + 1), out,
                   "beam_reduce sev %d" % s, "depth")
    p = save_fig(sheet, os.path.join(OUT_ROOT, "beam_reduce",
                                     "griffin_contact.png"))
    note(p, "griffin beam reduction (elevation binning); N/A on "
         "OPV2V/V2XSet -- no ring column")


def describe(rel: str) -> str:
    """Self-describing filename -> one-line description (used to rebuild the
    index from DISK, so a crashed export still ends up fully catalogued)."""
    inj, _, fname = rel.partition("/")
    stem = fname.rsplit(".", 1)[0]
    ds = stem.split("_")[0]
    sev = stem[len(ds) + 1:] or "-"
    what = {
        "pose_error": "collaborator clouds ghost against the ego's; "
                      "displacement grows with sigma",
        "comm_latency": "stale collaborator clouds trail the live ego",
        "agent_drop": "an agent's colour vanishes; node strip shows DROPPED",
        "missing_modality": "agent PRESENT with zero points (hollow node) -- "
                            "contrast agent_drop's absent node",
        "points_reduce": "uniform dropout; kept fraction in the title",
        "lidar_fog": "intensity dims and near-range scatter appears",
        "lidar_snow": "attenuate / scatter / remove; removal fraction in title",
        "camera_faults": "camera x severity matrix (vehicle front, drone bottom)",
        "occlusion": "lens contamination (dirt / scratch / crack)",
        "beam_reduce": "scan-line thinning by elevation binning",
        "packs": "frame pack driving the interactive scrubbers",
    }.get(inj, "")
    if fname == "INDEX.md":
        return ""
    kind = ("contact sheet, all severities" if stem.endswith("contact")
            else "mechanism figure (removal fraction vs AP)"
            if stem.endswith("mechanism")
            else "frame pack" if inj == "packs"
            else "animation" if fname.endswith((".mp4", ".gif")) else "still")
    return "%s | %s | severity %s | %s -- %s" % (inj, ds, sev, kind, what)


def write_index():
    """Rebuild INDEX.md by SCANNING the export tree, not from this run's
    in-memory list: a crashed export (job 558212) left 21 files uncatalogued
    when the index depended on run state."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "INDEX.md")
    lines = ["# Fault-injection visual exports\n",
             "Generated by notebooks/fault_visualisation.py --export "
             "(seed %d, the sweep's own injectors and adapters). Rebuilt "
             "from the files on disk.\n" % SEED,
             "\n## Applicability matrix\n",
             "| injector | v2xset | opv2v | griffin |", "|---|---|---|---|"]
    for injr in list(TIERS) + ["camera_faults", "occlusion", "beam_reduce"]:
        row = ["`%s`" % injr]
        for d in ("v2xset", "opv2v", "griffin"):
            v = APPLICABILITY[d].get(injr, True)
            row.append("yes" if v is True else "NO -- %s" % v)
        lines.append("| " + " | ".join(row) + " |")
    files = sorted(os.path.relpath(f, OUT_ROOT)
                   for f in glob.glob(os.path.join(OUT_ROOT, "*", "*")))
    lines.append("\n## Files (%d)\n" % len(files))
    lines += ["- `%s` -- %s" % (f, describe(f)) for f in files]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[export] INDEX.md rebuilt from disk: %d files" % len(files))


def run_exports(datasets):
    t_note = "ffmpeg" if HAS_FFMPEG else "NO ffmpeg -> GIF fallback"
    print("exporting for datasets=%s | opencood=%s plyfile=%s | %s"
          % (datasets, HAS_OPENCOOD, HAS_PLYFILE, t_note), flush=True)
    for d in datasets:
        if d in ("v2xset", "opv2v") and not HAS_OPENCOOD:
            print("[skip-env] %s needs the opencood env" % d)
            continue
        if d == "griffin-rest":
            # resume path: the camera/occlusion/beam block only (the rest of
            # griffin already exported in job 558212)
            export_griffin_only_faults()
            continue
        if d == "griffin" and not HAS_PLYFILE:
            print("[skip-env] griffin needs plyfile (.venv-hpc)")
            continue
        if d != "griffin":
            export_pose_error(d)
            export_comm_latency(d)
            export_agent_drop(d)
            export_missing_modality(d)
            export_lidar_scalar(d, "points_reduce")
            export_lidar_scalar(d, "lidar_fog")
            export_lidar_snow(d)
        else:
            print("[n/a] pose_error on griffin: %s"
                  % APPLICABILITY["griffin"]["pose_error"])
            _export_latency_griffin()
            _export_drop_missing_griffin("agent_drop")
            _export_drop_missing_griffin("missing_modality")
            export_lidar_scalar(d, "points_reduce")
            export_lidar_scalar(d, "lidar_fog")
            export_lidar_snow(d)
            export_griffin_only_faults()
    write_index()
    print("DONE: %d files exported to %s" % (len(INDEX), OUT_ROOT))


# ── marimo app (interactive layer; feeds on the exported packs) ─────────────

try:
    import marimo
    app = marimo.App(width="medium")
    _M = True
except ImportError:                                # export-env fallback
    _M = False

    class _Shim:
        def cell(self, f=None, **k):
            return (lambda g: g) if f is None else f
    app = _Shim()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    rows = sweep_rows()
    have = "loaded %d sweep rows" % len(rows) if rows else \
        "sweep_results.csv not found -- intros show visuals only"
    matrix = ["| injector | v2xset | opv2v | griffin |", "|---|---|---|---|"]
    for injr in list(TIERS) + ["camera_faults", "occlusion", "beam_reduce"]:
        r = ["`%s`" % injr]
        for d in ("v2xset", "opv2v", "griffin"):
            v = APPLICABILITY[d].get(injr, True)
            r.append("yes" if v is True else "**NO** (%s)" % v)
        matrix.append("| " + " | ".join(r) + " |")
    mo.md("\n".join([
        "# Fault injection -- every injector, every dataset, every severity",
        "All visuals come from the SWEEP's own injectors and adapters "
        "(seed 1234); exported artifacts live in "
        "`results/fault_injector_visualisation/` and are viewable without "
        "running anything (see `INDEX.md`). %s." % have,
        "## Applicability matrix (non-applicability is stated, never "
        "silently skipped)"] + matrix))
    return


@app.cell
def _(mo):
    injector = mo.ui.dropdown(
        list(TIERS), value="pose_error", label="injector")
    dataset = mo.ui.dropdown(["v2xset", "opv2v"], value="v2xset",
                             label="dataset")
    mode = mo.ui.radio(["sweep frames (hold severity)",
                        "sweep severity (hold frame)"],
                       value="sweep frames (hold severity)", label="scrub mode")
    mo.hstack([injector, dataset, mode])
    return dataset, injector, mode


@app.cell
def _(dataset, injector, mo, mode):
    import numpy as _np
    pack_path = os.path.join(PACKS, "%s_%s.npz"
                             % (injector.value, dataset.value))
    pack = _np.load(pack_path) if os.path.exists(pack_path) else None
    if pack is None:
        mo.md("**No frame pack for `%s` on `%s`.** Either not applicable "
              "(see matrix) or exports have not been generated yet -- run "
              "the CLI export first." % (injector.value, dataset.value))
    n_frames = int(pack["n_frames"]) if pack is not None else 1
    n_sev = int(pack["n_sev"]) if pack is not None else 1
    frame = mo.ui.slider(0, max(n_frames - 1, 0), value=0, label="frame")
    sev = mo.ui.slider(0, n_sev, value=1,
                       label="severity (0=clean)")
    mo.hstack([frame, sev])
    return frame, n_frames, n_sev, pack, sev


@app.cell
def _(frame, injector, mo, mode, n_frames, n_sev, pack, sev, dataset):
    import matplotlib.pyplot as _plt
    if pack is None:
        out = mo.md("")
    else:
        sweep_second = mode.value.startswith("sweep severity")
        panels = (range(n_sev + 1) if sweep_second else [sev.value])
        fig, axes = _plt.subplots(1, len(panels),
                                  figsize=(6.2 * len(panels), 6),
                                  facecolor=DARK, squeeze=False)
        names = ["clean"] + list(pack["sev_names"])
        for pi, s in enumerate(panels):
            agents = []
            ai = 0
            while "s%d_f%d_a%d" % (s, frame.value, ai) in pack:
                meta = pack["s%d_f%d_a%d_meta" % (s, frame.value, ai)]
                agents.append({
                    "agent": str(ai), "ego": bool(meta[0]),
                    "pts": pack["s%d_f%d_a%d"
                                % (s, frame.value, ai)].astype("float32"),
                    "delay": int(meta[1]), "speed": float(meta[2])})
                ai += 1
            bev_by_agent(axes[0][pi], agents,
                         "%s %s -- %s, frame %d"
                         % (dataset.value, injector.value, names[s],
                            frame.value))
        out = fig
    out
    return


@app.cell
def _(mo):
    lines = ["## Section guide (what each fault is, what to look for, what "
             "we measured)"]
    intro = {
        "pose_error": ("GPS/IMU error corrupts the SHARED pose, so a "
                       "collaborator's whole cloud lands misaligned in the "
                       "ego frame. Look for: rigid ghosting of one colour "
                       "against the ego's, growing with sigma."),
        "comm_latency": ("The collaborator's data is one-to-three frames "
                         "STALE. Look for: moving objects in the stale "
                         "colour trailing their live positions; staleness "
                         "in metres = speed x delay."),
        "agent_drop": ("The transmission is lost entirely: node AND points "
                       "gone. Look for: a colour vanishing; the node strip "
                       "shows an X."),
        "missing_modality": ("The sensor dies but the AGENT remains in the "
                             "fusion graph: pose known, zero points. Look "
                             "for: hollow node, empty colour. Measured "
                             "slightly WORSE than agent_drop at equal p -- "
                             "the empty agent still consumes fusion "
                             "attention."),
        "points_reduce": ("Uniform dropout keeps 30/20/10%% of points."),
        "lidar_fog": ("Beer-Lambert attenuation + fog backscatter. Watch "
                      "intensity (viridis) dim and near-range scatter "
                      "appear. CoBEVT collapses under this."),
        "lidar_snow": ("Hahner snowfall physics: attenuate, scatter, "
                       "remove. removed_frac INVERTS with severity while "
                       "AP degrades monotonically -- see the mechanism "
                       "figure."),
    }
    for k, v in intro.items():
        lines.append("**`%s`** -- %s %s\n" % (k, v, sweep_line(k)))
    mo.md("\n".join(lines))
    return


if __name__ == "__main__":
    if "--export" in sys.argv:
        ds_arg = "v2xset,opv2v,griffin"
        for i, a in enumerate(sys.argv):
            if a == "--datasets" and i + 1 < len(sys.argv):
                ds_arg = sys.argv[i + 1]
        run_exports([d.strip() for d in ds_arg.split(",") if d.strip()])
    elif _M:
        app.run()
