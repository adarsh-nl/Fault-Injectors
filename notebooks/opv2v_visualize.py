import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OPV2V — what each fault does to cooperative fusion

        Every plot here is in the **ego's coordinate frame**, because that is what
        fusion actually operates on. This matters most for pose error: it corrupts
        the `ego <- collaborator` transform, so the collaborator's points land in the
        **wrong place** in the ego frame. That misplacement *is* the fault. In world
        frame it is invisible — each agent's cloud is drawn through its own corrupted
        pose and therefore looks perfectly self-consistent.

        **Ego is the clean reference.** All injectors are applied with non-ego scope,
        matching `PoseErrorInjector.apply_to_sample(..., protect_ego=True)`. The ego's
        LiDAR and images are byte-identical between the two panels — asserted below,
        and it must always hold.

        **Ground-truth boxes are identical in both panels by construction** — the
        injectors perturb sensor data and shared poses, never labels. The boxes are
        the fixed reference against which the collaborator's drift becomes legible.

        Severity levels come from `src/fault_injectors/severity.py`, so what is drawn
        here is what the benchmark actually runs.
        """
    )
    return


@app.cell
def _():
    import glob
    import os
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Repo root on the path so `notebooks.` and `src.` both import regardless of
    # where marimo was launched from.
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:                              # defensive
        _root = os.path.abspath("..")
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from notebooks import _opv2v_viz as V
    from src.fault_injectors import severity as sev

    OPV2V_ROOT = "/datasets/eemcs/ps/cv/opencood/opv2v"
    return OPV2V_ROOT, V, glob, mo, np, os, plt, sev, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ego selection

        OPV2V's ego is **not** ambiguous once you follow the adapter: OpenCOOD sorts
        the CAV folder names **as strings** and takes the first. `src/datasets/opv2v.py`
        reproduces that deliberately — sorting *numerically* would pick a different
        ego whenever ids have different digit counts (`'1045'` vs `'650'`), silently
        changing the viewpoint of every frame and making results incomparable with
        published OPV2V numbers. On top of that, V2X-ViT's convention applies:
        negative ids are roadside units, which sort first as strings (`'-' < '0'`) but
        are moved to the **end** so they are never the default ego.

        The adapter accepts `ego_id=` directly, so the dropdown below just passes it
        through — the convention is not reimplemented here.
        """
    )
    return


@app.cell
def _(OPV2V_ROOT, glob, mo, os):
    scenario_paths = sorted(
        p for p in glob.glob(os.path.join(OPV2V_ROOT, "train", "*"))
        if os.path.isdir(p))
    scenario = mo.ui.dropdown(
        options={os.path.basename(p): p for p in scenario_paths},
        value=os.path.basename(scenario_paths[0]),
        label="scenario",
    )
    scenario
    return scenario, scenario_paths


@app.cell
def _(V, scenario):
    # Cheap: the adapter only globs folder names and timestamps in __init__,
    # it does not read any point cloud until get_sample().
    probe = V.load_dataset("opv2v", scenario_dir=scenario.value)
    agent_ids = probe.agent_ids()
    default_ego = probe.ego_id
    return agent_ids, default_ego, probe


@app.cell
def _(agent_ids, default_ego, mo, probe):
    ego_pick = mo.ui.dropdown(
        options=list(agent_ids), value=default_ego,
        label=f"ego (adapter default: {default_ego})")
    frame = mo.ui.slider(
        0, len(probe) - 1, value=min(60, len(probe) - 1),
        label="frame", show_value=True)
    mo.hstack([ego_pick, frame], justify="start", gap=2)
    return ego_pick, frame


@app.cell
def _(V, ego_pick, mo, scenario):
    ds = V.load_dataset("opv2v", scenario_dir=scenario.value,
                        ego_id=ego_pick.value)
    mo.md(
        f"**{len(ds)} frames**, agents `{ds.agent_ids()}`, "
        f"ego **`{ds.ego_id}`**, {ds.fps:g} Hz &nbsp;·&nbsp; `{scenario.value}`"
    )
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fault and severity

        Two faults act on **geometry** — they corrupt the `ego <- collab` transform or
        the frame index, so the collaborator's points move in the ego frame while the
        points themselves are untouched. The rest corrupt the **payload**.
        """
    )
    return


@app.cell
def _(V, mo):
    fault = mo.ui.dropdown(options=V.SELECTABLE, value="spatial_misalignment",
                           label="fault")
    level = mo.ui.radio(options={"1": 1, "2": 2, "3": 3}, value="3",
                        label="severity", inline=True)
    seed = mo.ui.number(0, 999, value=0, label="seed")
    extent = mo.ui.slider(20, 140, value=100, step=10,
                          label="BEV extent [m]", show_value=True)
    mo.hstack([fault, level, seed, extent], justify="start", gap=2)
    return extent, fault, level, seed


@app.cell
def _(V, ds, fault, frame, level, mo, seed, sev):
    clean, faulted, notes = V.apply_fault(
        ds, frame.value, fault.value, level.value, seed=seed.value, cams=1)

    plane, modality, _ok, _why = V.VIEWABLE[fault.value]
    kwargs_md = ", ".join(
        f"`{k}`={v}" for k, v in sev.SEVERITY[fault.value][level.value].items())

    mo.md(
        f"**{fault.value}** · level {level.value} · {plane} plane · {modality} · "
        f"group **{sev.GROUP[fault.value]}** "
        f"({'transcribed' if sev.GROUP[fault.value] == 'A' else 'injector-calibrated'})\n\n"
        f"constructor kwargs: {kwargs_md}\n\n"
        + "\n".join(f"- {n}" for n in notes)
    )
    return clean, faulted, kwargs_md, modality, notes, plane


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Clean vs faulted — same ego frame, same extent

        Black = ego (clean reference). Colours = collaborators. Green = GT boxes.
        The yellow star is the ego origin.

        Both panels are subsampled with the **same fraction**, computed once from the
        clean sample. Subsampling each panel to a fixed *count* would normalise away
        the density loss that is the entire visible signature of `points_reducing`,
        making a 90 %-dropout panel look identical to the clean one.
        """
    )
    return


@app.cell
def _(V, clean, extent, fault, faulted, level, plt):
    frac = V.sampling_fraction(clean)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    V.bev_panel(ax[0], clean, "clean", extent=extent.value, frac=frac)
    V.bev_panel(ax[1], faulted, f"{fault.value} · level {level.value}",
                extent=extent.value, frac=frac)
    ax[0].legend(loc="upper right", fontsize=7, markerscale=12, framealpha=0.9)
    fig.tight_layout()
    fig
    return ax, fig, frac


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How big is the displacement, really

        A pose error of a degree or two is a sub-metre shift at short range and a
        multi-metre one far out, so the **mean over the whole cloud understates it
        badly** — OPV2V's mean point range is only ~14 m. The farthest-decile figure
        below is the honest one. Narrow the BEV extent above to watch the
        collaborator's points peel away from the GT boxes.
        """
    )
    return


@app.cell
def _(V, clean, faulted, mo, np):
    rows = []
    for aid in clean.agents:
        if aid == clean.ego_id:
            continue
        a = clean.lidar_in_ego_frame(aid)
        b = faulted.lidar_in_ego_frame(aid)
        if a.shape == b.shape:
            d = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
            r = np.linalg.norm(a[:, :3], axis=1)
            far = d[r > np.percentile(r, 90)]
            rows.append(
                f"- **{aid}**: mean ego-frame displacement **{d.mean():.3f} m**, "
                f"median {np.median(d):.3f} m, max {d.max():.2f} m; "
                f"farthest 10 % of points shift **{far.mean():.2f} m** on average")
        else:
            rows.append(
                f"- **{aid}**: {len(a)} → {len(b)} points "
                f"({len(b) / max(len(a), 1):.1%} kept) — the point set itself "
                f"changed, so per-point displacement is undefined")

    # Both invariants are checked, not merely asserted in prose above. Eyeballing
    # two panels side by side is not reliable at this scale -- the boxes look
    # like they move under pose error and they do not.
    ego_untouched = np.array_equal(clean.ego.lidar, faulted.ego.lidar)
    boxes_clean = np.array(V.boxes_in_ego_frame(clean))
    boxes_faulted = np.array(V.boxes_in_ego_frame(faulted))
    boxes_identical = (boxes_clean.shape == boxes_faulted.shape
                       and np.allclose(boxes_clean, boxes_faulted))
    rows.append(f"\n**ego LiDAR untouched: `{ego_untouched}`** — non-ego scope; "
                f"must always be True")
    rows.append(f"**GT boxes identical across panels: `{boxes_identical}`** "
                f"({len(boxes_clean)} boxes) — injectors never touch labels, so "
                f"the boxes are the fixed reference; must always be True")
    mo.md("\n".join(rows))
    return (a, aid, b, boxes_clean, boxes_faulted, boxes_identical, d,
            ego_untouched, far, r, rows)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cameras

        Shown only for camera-plane faults. One camera per agent is corrupted by
        default: `fog_camera` costs ~5.4 s per camera and this runs on a shared login
        node. `missing_camera` is exempt — it is a Bernoulli draw costing ~0 s, and at
        `p_drop=0.2` a single camera usually survives, so all four are sampled.
        """
    )
    return


@app.cell
def _(V, clean, fault, faulted, mo, plt):
    if fault.value not in V.CAMERA_FAULTS:
        cam_out = mo.md("*(LiDAR-plane fault — no camera panel.)*")
    else:
        collabs = [x for x in clean.agents if x != clean.ego_id]
        fig2, ax2 = plt.subplots(len(collabs), 2,
                                 figsize=(11, 3.2 * len(collabs)),
                                 squeeze=False)
        for i, cid in enumerate(collabs):
            cam = list(clean.agents[cid].images)[0]
            for j, (s, t) in enumerate(((clean, "clean"), (faulted, fault.value))):
                ax2[i][j].imshow(s.agents[cid].images[cam])
                ax2[i][j].set_title(f"collab {cid} · {cam} · {t}", fontsize=9)
                ax2[i][j].axis("off")
        fig2.tight_layout()
        cam_out = fig2
    cam_out
    return cam_out


@app.cell(hide_code=True)
def _(V, mo):
    mo.md(
        "## What does not load\n\n"
        "Recorded, not worked around — core is read-only in this notebook.\n\n"
        + "\n\n".join(
            f"**`{f}`** — {why}"
            for f, (_p, _m, ok, why) in V.VIEWABLE.items() if not ok)
    )
    return


if __name__ == "__main__":
    app.run()
