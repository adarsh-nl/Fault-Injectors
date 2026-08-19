"""figures.py -- publication figure set for the fault-robustness benchmark.

Single source: ``results/sweep/sweep_results.csv`` (93 rows) and
``results/sweep/sweep_manifest.json``. Nothing is recomputed from raw bundles,
with ONE approved, documented exception -- see FIG 5 / OPV2V_CONTRAST below.

    .venv-hpc/bin/python tools/sweep/figures.py --outdir results/figures
    .venv-hpc/bin/python tools/sweep/figures.py --only fig1,fig5

Deterministic: one pass over the CSV into dicts, no RNG, no wall-clock in the
output. Re-running produces identical PDFs.

SCIENTIFIC CONSTRAINTS ENCODED HERE (they are not stylistic):

* Absolute AP is NOT comparable across models -- each runs at its authors'
  shipped wild_setting on its native dataset. Only Fig 1 (one model per panel)
  and Fig 4/5 (per-model series) show absolute AP. The ONLY cross-model panel,
  Fig 2, uses rel_drop exclusively.
* Where2comm is UNGRADED on provenance grounds: its checkpoint is a
  third-party retraining on V2XSet (md5-identical to
  point_pillar_where2comm_v2xset.zip), a dataset the Where2comm authors never
  used, and the official repo releases no checkpoints at all. It is marked on
  every panel it appears in and must not read as equivalent evidence to
  CoBEVT and V2X-ViT.
* AgentDrop is NEVER pooled -- in a 2-agent scene dropping one collaborator is
  ablation to single-agent perception, not graded degradation. It is excluded
  from Fig 1 and Fig 2 and gets its own stratified Fig 4.
* latency/mild on NOISY-shipped models is EXACTLY 0.000000: the hook REPLACES
  the shipped 100 ms delay rather than stacking on it, so the run is
  bit-identical to clean. CoBEVT (Perfect, no shipped delay to replace) shows
  a real -0.0676 and is the control that demonstrates the mechanism.
* Shipped setting differs by model and is printed on every panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

# ── craft spec ──────────────────────────────────────────────────────────
COL_W, DBL_W = 3.35, 7.0          # inches: one- and two-column widths
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed Type 42, not Type 3
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False,
})

# Okabe-Ito, colourblind-safe. ONE fixed injector->colour map, reused by every
# figure so a colour means the same thing across the whole set.
INJ_COLOR = {
    "lidar_snow":       "#0072B2",   # blue
    "lidar_fog":        "#56B4E9",   # sky
    "points_reduce":    "#009E73",   # green
    "pose_error":       "#D55E00",   # vermillion
    "latency":          "#E69F00",   # orange
    "missing_modality": "#CC79A7",   # purple
    "agent_drop":       "#666666",   # grey (stratified; Fig 4 only)
}
INJ_LABEL = {
    "lidar_snow": "LiDAR snow", "lidar_fog": "LiDAR fog",
    "points_reduce": "Points reduce", "pose_error": "Pose error",
    "latency": "Comm. latency", "missing_modality": "Missing LiDAR",
    "agent_drop": "Agent drop",
}
# Fig 1/2/3 order: corrupted-data injectors first, then lost-collaborator.
# lidar_fog was excluded from the ranked panels while its cells were produced
# by the pre-2026-08-10 injector, which binarised normalised intensity and
# saturated every tier (EXACTLY -1.00 at severe on all three models). After
# the intensity-scale fix (jobs 560218/560219/560220) fog has real dynamic
# range -- -0.23..-0.46 mild, -0.32..-0.62 moderate, -0.88..-0.99 severe, all
# three models monotone, no exact zeros -- so it is a ranked entry again and
# the standalone saturation figure has been deleted.
CORRUPTED = ["lidar_snow", "lidar_fog", "points_reduce", "pose_error",
             "latency"]
LOST = ["missing_modality", "agent_drop"]
FIG1_INJ = CORRUPTED + ["missing_modality"]        # agent_drop -> Fig 4
TIERS = ["mild", "moderate", "severe"]
MODELS = ["cobevt", "v2xvit", "where2comm", "cosdh"]
MODEL_LABEL = {"cobevt": "CoBEVT", "v2xvit": "V2X-ViT",
               "where2comm": "Where2comm", "cosdh": "CoSDH"}
UNGRADED = {"where2comm", "cosdh"}
FLAG = "⚑"                                     # provenance marker
# Ungraded for DIFFERENT reasons -- the flag alone would conflate them.
UNGRADED_WHY = ("Ungraded⚑ for different reasons: Where2comm = PROVENANCE "
                "(third-party retrain, no official OPV2V/V2XSet release); "
                "CoSDH = RETRAIN (author release, but README states the "
                "checkpoints were retrained and 'differ slightly' from the "
                "paper; measured clean 0.9660/0.9289 vs published "
                "0.9683/0.9299). ")
# CoSDH evaluates on OPV2V test; the other three on V2XSet test. rel_drop
# normalises away different clean baselines but NOT a different dataset.
XDATA_WHY = ("DATASET CAVEAT: CoSDH runs on OPV2V test (2,170 frames); the "
             "other three on V2XSet test (2,834). rel_drop normalises away "
             "different clean baselines but not a different dataset -- "
             "cross-model comparisons involving CoSDH are NOT like-for-like. ")
# The six CoSDH not-applicable cells (agent_drop x3, missing_modality x3):
# the released inference path crashes when a collaborator is removed
# (record_len over-counts; manifest not_applicable). Never 0.0, never ranked.
COSDH_NA = ("CoSDH agent_drop and missing_modality are NOT APPLICABLE (6 "
            "cells): the released inference path CRASHES when a collaborator "
            "is removed, rather than degrading -- which is itself the "
            "finding. Shown as N/A, never as 0.0, excluded from every "
            "ranking. ")
# Injectors CoSDH has cells for (the 5 corrupted-data ones); used to skip
# rather than crash the exactly-one lookups.
COSDH_INJ = set(CORRUPTED)

# Provenance line for every panel that shows fog. The 9 fog cells were
# regenerated after the intensity-binarisation fix; everything else predates
# injector stamping. Stated on the figures because a reader cannot otherwise
# tell that one injector's cells have a different lineage.
GT_CONFOUND = (
    "GT-DENOMINATOR CAVEAT (agent_drop only): ground truth is the UNION over "
    "CAVs, so dropping an agent removes the boxes only it observed. Measured "
    "on 200 frames of V2XSet: GT shrinks -2.8% / -5.8% / -8.5% at "
    "p=0.25/0.50/0.75, and with EVERY collaborator dropped -12.1%. Recall = "
    "TP/GT, so a smaller denominator inflates recall and agent_drop's "
    "degradation is biased toward looking MILDER than it is. "
    "missing_modality is NOT affected -- it empties a cloud but KEEPS the "
    "agent, so its denominator is stable. ")

FOG_PROV = ("LiDAR fog PROVENANCE: the 9 V2XSet fog cells were regenerated "
            "by the CORRECTED injector (lidar_fog: bdc5c3f7) after the "
            "intensity-binarisation fix; CoSDH's 3 fog cells predate driver "
            "stamping (fog code byte-identical per adapter digest), making "
            "fog the ONE injector the census flags as version-mixed -- do "
            "not pool CoSDH fog with the other three. Every non-fog cell "
            "predates stamping uniformly. ")

# Determinism floors, stated in captions rather than drawn: at these axis
# ranges a 1e-4 band is sub-pixel and would imply false precision.
FLOOR_NOTE = ("Determinism floors: CoBEVT 6.4e-05 within-session; "
              "Where2comm and V2X-ViT 0.0 within a node, ~4.3e-05 across "
              "GPU SKUs. All are >=3 orders below every delta plotted.")


# ── data ────────────────────────────────────────────────────────────────
def parse_extra(s):
    """extra_metric is a ';'- or ','-joined k=v blob."""
    out = {}
    for part in re.split(r"[;,]", s or ""):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load(csv_path):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("clean_ap50", "clean_ap70", "faulty_ap50", "faulty_ap70",
                  "delta_ap50", "delta_ap70", "rel_drop_ap50", "rel_drop_ap70",
                  "n_frames", "n_injections"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                r[k] = float("nan")
        r["_x"] = parse_extra(r.get("extra_metric", ""))
    return rows


def one(rows, **kw):
    """Exactly-one row lookup. Raises rather than silently taking the first --
    a wrong row here becomes a wrong number in a paper."""
    hits = [r for r in rows
            if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    if len(hits) != 1:
        raise SystemExit("expected 1 row for %s, got %d" % (kw, len(hits)))
    return hits[0]


def setting_of(rows, model):
    return one(rows, model=model, injector="none",
               severity_tier="clean")["setting"].capitalize()


def clean70(rows, model):
    return one(rows, model=model, injector="none",
               severity_tier="clean")["clean_ap70"]


def panel_tag(rows, model):
    """Dataset, shipped setting, n -- on every panel, per the brief."""
    r = one(rows, model=model, injector="none", severity_tier="clean")
    tag = "%s · %s · n=%d" % (r["dataset"].upper(),
                                        r["setting"].capitalize(),
                                        int(r["n_frames"]))
    if model in UNGRADED:
        tag += " · UNGRADED" + FLAG
    return tag


def save(fig, outdir, name):
    """PDF + PNG. CreationDate is suppressed: matplotlib stamps wall-clock
    into PDF metadata, which makes byte-identical regeneration impossible and
    silently breaks the determinism guarantee this script advertises."""
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "%s.%s" % (name, ext))
        meta = ({"CreationDate": None} if ext == "pdf"
                else {"Software": None})
        fig.savefig(p, metadata=meta)
        paths.append(p)
    plt.close(fig)
    return paths


def footnote(fig, text, width=118, y=0.012, bottom=0.24):
    """Wrapped note placed BELOW the axes, growing the canvas downward.

    Two failure modes, both hit and both fixed here:
      * UNWRAPPED text at negative y -- bbox_inches='tight' expands the canvas
        to contain one enormous line (fig5 rendered 5193 px wide).
      * WRAPPED text placed INSIDE the box with subplots_adjust -- a long
        caption then overruns the axes, because a colorbar steals width and
        the reserved fraction does not track the line count.
    Wrapped text at a small negative y does neither: the canvas grows downward
    by exactly the text block's height and the axes are untouched.

    `bottom` is accepted and ignored, so call sites need not be rewritten.
    """
    lines = textwrap.wrap(text, width)
    fig.text(0.005, -0.03, "\n".join(lines), fontsize=6, color="#444444",
             va="top", ha="left")


def mark_ungraded(ax, model):
    """Hatching + flag, never colour alone."""
    if model not in UNGRADED:
        return
    for s in ("bottom", "left"):
        ax.spines[s].set_linestyle((0, (3, 2)))
    ax.patch.set_facecolor("#f4f4f4")
    ax.patch.set_alpha(0.6)


# ── FIG 1 ───────────────────────────────────────────────────────────────
def fig1(rows, outdir, assertions):
    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.6), sharex=True)
    for ax, model in zip(axes, MODELS):
        c70 = clean70(rows, model)
        ax.axhline(c70, ls=(0, (4, 2)), lw=0.9, color="#222222", zorder=1)
        ax.text(2.60, c70, "clean\n%.3f" % c70, va="center", ha="right",
                fontsize=5.8, color="#222222",
                bbox=dict(fc="white", ec="none", pad=0.8))
        for inj in FIG1_INJ:
            if model == "cosdh" and inj not in COSDH_INJ:
                # missing_modality: the ONLY not-applicable gap visible in
                # this figure (agent_drop is already excluded for everyone).
                # Annotated rather than silently absent.
                ax.text(0.03, 0.035,
                        "Missing LiDAR: N/A —\ncrashes, does not degrade",
                        transform=ax.transAxes, fontsize=5.5,
                        color=INJ_COLOR["missing_modality"])
                assertions.append(("fig1", model, inj, ["N/A (crash)"]))
                continue
            ys = [one(rows, model=model, injector=inj,
                      severity_tier=t)["faulty_ap70"] for t in TIERS]
            ax.plot(range(3), ys, marker="o", color=INJ_COLOR[inj],
                    label=INJ_LABEL[inj], zorder=3)
            assertions.append(("fig1", model, inj,
                               ["%.4f" % v for v in ys]))
        # latency/mild == clean by construction on Noisy-shipped models
        lat = one(rows, model=model, injector="latency",
                  severity_tier="mild")
        if abs(lat["delta_ap70"]) == 0.0:
            # label sits AT the point: a long leader line reads as a
            # spurious latency series plunging to zero.
            ax.annotate("≡ clean", xy=(0, lat["faulty_ap70"]),
                        xytext=(4, -9), textcoords="offset points",
                        fontsize=5.5, color=INJ_COLOR["latency"])
        mark_ungraded(ax, model)
        ax.set_xticks(range(3))
        ax.set_xticklabels([t.capitalize() for t in TIERS])
        ax.set_title("(%s) %s" % ("abcd"[MODELS.index(model)],
                                  MODEL_LABEL[model]), loc="left")
        # TWO lines: at 4 panels each is ~1.55 in wide and the one-line tag
        # of panel (c) overran into panel (d)'s.
        ax.text(0.0, -0.20, panel_tag(rows, model).replace(" · n=", "\nn="),
                transform=ax.transAxes, fontsize=6, color="#444444",
                va="top", linespacing=1.4)
        ax.grid(axis="y", color="#dddddd", zorder=0)
        ax.set_xlim(-0.15, 2.62)
    axes[0].set_ylabel("AP@0.7")
    # explicit limits: autoscale plus an overlapping legend clipped the fog
    # severe point (v2xvit 0.0685) below the axis.
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.0) - 0.01, hi)
    handles = [Line2D([], [], color=INJ_COLOR[i], marker="o",
                      label=INJ_LABEL[i]) for i in FIG1_INJ]
    handles.append(Line2D([], [], color="#222222", ls=(0, (4, 2)),
                          label="Clean control"))
    # footnote() no longer reserves bottom space (it grows the canvas
    # downward instead), so a fig.legend must reserve its own.
    # bottom widened for the two-line panel tags: at 0.30 the tags'
    # second line struck through the legend row.
    fig.subplots_adjust(bottom=0.40)
    fig.legend(handles=handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, 0.01))
    footnote(fig, "Latency/mild on the Noisy-shipped models is EXACTLY "
             "0.000000 — the hook replaces the shipped 100 ms rather than "
             "stacking on it, so the run is bit-identical to clean; CoBEVT "
             "and CoSDH (Perfect-shipped, no delay to replace) show real "
             "drops (-0.0676, -0.3086) and are the controls for that. "
             "CoSDH ships add_noise:false, so it GROUPS WITH COBEVT and its "
             "injected pose sigma is TOTAL error, not an increment on "
             "shipped noise. Agent drop is excluded — it is stratified by "
             "scene agent count (Fig. 4). "
             + XDATA_WHY + UNGRADED_WHY + FOG_PROV, bottom=0.40)
    return save(fig, outdir, "fig1_degradation_curves")


# ── FIG 2 ───────────────────────────────────────────────────────────────
def fig2(rows, outdir, assertions):
    injs = CORRUPTED + ["missing_modality"]

    def cell(m, i):
        if m == "cosdh" and i not in COSDH_INJ:
            return float("nan")                     # N/A, drawn explicitly
        return one(rows, model=m, injector=i,
                   severity_tier="severe")["rel_drop_ap70"]

    grid = [[cell(m, i) for m in MODELS] for i in injs]
    fig, ax = plt.subplots(figsize=(COL_W * 1.15, 2.5))
    vmax = max(abs(v) for row in grid for v in row if v == v)
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for r, inj in enumerate(injs):
        for c, m in enumerate(MODELS):
            v = grid[r][c]
            if v != v:
                # Per-MODEL not-applicable cell: DISTINCT from the hatched
                # exclusion band below. The band ("///") means "excluded for
                # ALL models -- not poolable"; this cell ("xxx") means "THIS
                # model could not run it -- crashes rather than degrades".
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           hatch="xxx", facecolor="white",
                                           edgecolor="#999999", lw=0.5))
                ax.text(c, r, "N/A", ha="center", va="center", fontsize=6,
                        color="#B22222")
                assertions.append(("fig2", m, inj, "N/A (crash)"))
                continue
            ax.text(c, r, "%.2f" % v, ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 * vmax else "black")
            assertions.append(("fig2", m, inj, "%.4f" % v))
    ax.set_xticks(range(len(MODELS)))
    # dataset ON the axis: a reader must not read this as like-for-like
    ax.set_xticklabels(["%s%s\n%s · %s" % (
        MODEL_LABEL[m], FLAG if m in UNGRADED else "",
        one(rows, model=m, injector="none",
            severity_tier="clean")["dataset"].upper(),
        setting_of(rows, m)) for m in MODELS], fontsize=6)
    ax.set_yticks(range(len(injs)))
    ax.set_yticklabels([INJ_LABEL[i] for i in injs])
    ax.axvline(1.5, color="black", lw=1.0)      # graded | ungraded boundary
    # Excluded injectors get EXPLICIT hatched bands, never silent omission.
    ax.set_ylim(len(injs) - 0.5, -1.5)
    ax.add_patch(plt.Rectangle((-0.5, -1.5), len(MODELS), 1.0, hatch="///",
                               facecolor="#eeeeee", edgecolor="#999999",
                               lw=0.5))
    ax.text((len(MODELS) - 1) / 2.0, -1.0,
            "agent drop — not poolable, see Fig. 4 (CoSDH: no cells at all)",
            ha="center", va="center", fontsize=6, color="#444444")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Relative degradation, AP@0.7\n(from each model's own clean "
                 "control)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    ax.set_title("Severe tier", loc="left")
    footnote(fig, "TWO different exclusion marks: the '///' band = excluded "
             "for ALL models (agent drop, not poolable); the 'xxx' N/A cell "
             "= THIS model could not run it. " + COSDH_NA + XDATA_WHY
             + UNGRADED_WHY +
             "Agent drop is a hatched band rather than silently dropped: "
             "stratified by scene agent count (Fig. 4). LiDAR fog is a "
             "normal ranked row: monotone on all four models, no exact "
             "zeros (CoBEVT -0.875, V2X-ViT -0.889, Where2comm -0.988, "
             "CoSDH -0.743 on OPV2V). Where2comm at severe is AP 0.0067 — "
             "near the floor and to be read as near-saturated in isolation, "
             "though it ranks distinctly. " + FOG_PROV,
             width=70, bottom=0.46)
    return save(fig, outdir, "fig2_crossmodel_heatmap")


# ── FIG 3 ───────────────────────────────────────────────────────────────
def fig3(rows, outdir, assertions):
    def val(m, i):
        if m == "cosdh" and i not in COSDH_INJ:
            return None                             # N/A: never a number
        if i == "agent_drop":                       # n=2 stratum ONLY
            hits = [r for r in rows if r["model"] == m
                    and r["injector"] == "agent_drop"
                    and r["severity_tier"] == "severe"
                    and r["_x"].get("scene_agent_count") == "2"]
            if len(hits) != 1:
                raise SystemExit("agent_drop n=2 severe: %d rows" % len(hits))
            return hits[0]["rel_drop_ap70"]
        return one(rows, model=m, injector=i,
                   severity_tier="severe")["rel_drop_ap70"]

    # fog is a ranked entry again since the intensity-scale fix. The mean
    # for ordering runs over the models that HAVE the cell: CoSDH's two
    # not-applicable injectors must not enter any ranking, so on those rows
    # the mean is over the three V2XSet models only.
    def rank_key(i):
        vs = [val(m, i) for m in MODELS]
        vs = [v for v in vs if v is not None]
        return sum(vs) / len(vs)

    order = sorted(CORRUPTED + LOST, key=rank_key)
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    h, off = 0.20, [-0.30, -0.10, 0.10, 0.30]
    alphas = [1.0, 0.78, 0.56, 0.38]
    hatches = {"where2comm": "///", "cosdh": "xx"}
    for k, m in enumerate(MODELS):
        for r, inj in enumerate(order):
            v = val(m, inj)
            if v is None:
                assertions.append(("fig3", m, inj, "N/A (crash)"))
                continue
            ax.barh(r + off[k], v, height=h, color=INJ_COLOR[inj],
                    alpha=alphas[k], edgecolor="#333333", lw=0.4,
                    hatch=hatches.get(m, ""))
            assertions.append(("fig3", m, inj, "%.4f" % v))
    # the absence IS the finding; say it inside the lost-collaborator band.
    # LEFT side: the lost-collaborator bars are short and live at the right
    # edge, so the band's left half is the only collision-free region.
    ax.text(0.02, max(order.index(i) for i in LOST) + 0.38,
            "CoSDH bars absent: crashes rather than degrades (N/A)",
            transform=ax.get_yaxis_transform(), fontsize=5.5,
            color="#B22222", ha="left", va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([INJ_LABEL[i] + ("*\u2020" if i == "agent_drop" else "")
                        for i in order])
    # on-figure denominator warning: this row is not comparable to the one
    # above it, and the figure must say so without the caption.
    if "agent_drop" in order:
        ax.text(0.015, order.index("agent_drop") - 0.42,
                "\u2020 GT denominator shrinks -2.8/-5.8/-8.5% with severity",
                transform=ax.get_yaxis_transform(), fontsize=5.5,
                color="#B22222", va="center")
    ax.set_xlabel("Relative degradation, AP@0.7 (severe tier)")
    ax.grid(axis="x", color="#dddddd", zorder=0)
    # regime band: lost-collaborator vs corrupted-data
    lost_rows = [r for r, i in enumerate(order) if i in LOST]
    if lost_rows and max(lost_rows) - min(lost_rows) == len(lost_rows) - 1:
        ax.axhspan(min(lost_rows) - 0.5, max(lost_rows) + 0.5,
                   color="#f0f0f0", zorder=0)
        # below the GT-dagger line, which sits at agent_drop_row - 0.42:
        # stacking all three band texts at distinct heights on the LEFT is
        # what keeps them off each other and off the right-edge bars.
        ax.text(0.02, min(lost_rows) - 0.15,
                "lost collaborators", transform=ax.get_yaxis_transform(),
                fontsize=6, va="center", color="#555555")
    handles = [Patch(facecolor="#888888", alpha=a, hatch=hatches.get(m, ""),
                     edgecolor="#333333", lw=0.4,
                     label=MODEL_LABEL[m] + (FLAG if m in UNGRADED else ""))
               for m, a in zip(MODELS, alphas)]
    # OUTSIDE the axes: at loc="lower left" it was drawn on top of the
    # LiDAR snow and LiDAR fog bars. Space is reserved EXPLICITLY via
    # subplots_adjust rather than relying on footnote() -- footnote() now
    # grows the canvas downward and reserves nothing, and depending on it is
    # exactly what clipped the fig1 fog series.
    fig.subplots_adjust(top=0.80)
    ax.legend(handles=handles, fontsize=6, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0, columnspacing=1.2, handlelength=1.4)
    footnote(fig, COSDH_NA + XDATA_WHY +
             "CoSDH therefore contributes to fog, snow, points_reduce, pose "
             "and latency only, and the ordering mean on the two "
             "lost-collaborator rows runs over the three V2XSet models. "
             + GT_CONFOUND +
             "Consequently agent drop's POSITION RELATIVE TO MISSING LIDAR IS "
             "NOT RELIABLE and could plausibly reverse: the gap between them "
             "is 0.06/0.03/0.04 while the denominator shift is 8.5% at "
             "severe. The BAND-level claim -- lost-collaborator faults are "
             "milder than corrupted-data ones, which sit at -0.3 to -0.9 -- "
             "is unaffected and is what this figure asserts. "
             "*agent drop: n=2 stratum only (Fig. 4). NOTE V2X-ViT pose error (-0.16) sits BELOW its own latency "
             "(-0.19) and near the lost-collaborator band, vs -0.46 (CoBEVT) "
             "and -0.40 (Where2comm). V2X-ViT is the only model shipping "
             "loc_err=true. Consistent with noise-trained pose defence, but "
             "an ASSOCIATION, not a proven cause — the three models differ "
             "on architecture, dataset pairing and compression too. LiDAR "
             "fog now ranks first legitimately (monotone, no exact zeros) "
             "rather than as the floor tie it was before the fix. " + FOG_PROV,
             width=62, bottom=0.52)
    return save(fig, outdir, "fig3_robustness_ordering")


# ── FIG 4 ───────────────────────────────────────────────────────────────
def fig4(rows, outdir, assertions):
    """Two rows: bars from ZERO (top, honest baseline) and the same data
    zoomed to the working range (bottom). Truncating the bar axis would
    exaggerate the differences; a paired zoom shows them without lying."""
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 4.2),
                             gridspec_kw={"height_ratios": [1.0, 0.85],
                                          "hspace": 0.78})
    tier_alpha = {"mild": 0.45, "moderate": 0.72, "severe": 1.0}
    # CoSDH CANNOT appear here: it has no agent_drop cells at all (the
    # released inference path crashes when a collaborator is removed). Its
    # absence is stated in the footnote rather than shown as an empty panel.
    fig4_models = [m for m in MODELS if m != "cosdh"]
    for col, model in enumerate(fig4_models):
        ax, axz = axes[0][col], axes[1][col]
        strata = sorted({int(r["_x"]["scene_agent_count"]) for r in rows
                         if r["model"] == model
                         and r["injector"] == "agent_drop"})
        w, lo, hi = 0.26, 1e9, -1e9
        for si, n in enumerate(strata):
            for ti, t_ in enumerate(TIERS):
                hits = [r for r in rows if r["model"] == model
                        and r["injector"] == "agent_drop"
                        and r["severity_tier"] == t_
                        and r["_x"].get("scene_agent_count") == str(n)]
                if len(hits) != 1:
                    raise SystemExit("agent_drop %s n=%s %s: %d rows"
                                     % (model, n, t_, len(hits)))
                v = hits[0]["faulty_ap70"]
                lo, hi = min(lo, v), max(hi, v)
                for a in (ax, axz):
                    a.bar(si + (ti - 1) * w, v, width=w,
                          color=INJ_COLOR["agent_drop"], alpha=tier_alpha[t_],
                          edgecolor="#333333", lw=0.4, zorder=2)
                assertions.append(("fig4", model, "n=%d/%s" % (n, t_),
                                   "%.4f" % v))
            cl = hits[0]["clean_ap70"]
            lo, hi = min(lo, cl), max(hi, cl)
            for a in (ax, axz):
                a.plot([si - 1.6 * w, si + 1.6 * w], [cl, cl],
                       color="#D55E00", lw=1.0, zorder=3)
                a.plot([si], [cl], marker="o", ms=3, color="#D55E00", zorder=4)
            nf, ns = int(hits[0]["n_frames"]), int(hits[0]["_x"]["n_scenes"])
            thin = ns <= 2
            # TWO lines, placed well BELOW the "n=2..n=5" tick labels.
            # Stacked at -0.06 they were struck through by the tick text; on
            # ONE line at -0.20 they cleared the ticks but ran into each
            # other horizontally ("3sc" overran the red "308f / 2sc").
            # Two narrow lines at -0.20 clear both.
            ax.text(si, -0.20, "%df\n%dsc%s" % (nf, ns, " ⚠" if thin else ""),
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=5.5, linespacing=1.4,
                    color="#B22222" if thin else "#444444")
            if thin and model == "where2comm" and n == 5:
                d = [r for r in rows if r["model"] == model
                     and r["injector"] == "agent_drop"
                     and r["severity_tier"] == "mild"
                     and r["_x"].get("scene_agent_count") == "5"][0]
                axz.annotate("mild Δ=%+.4f\n(within noise)" % d["delta_ap70"],
                             xy=(si - w, d["faulty_ap70"]),
                             xytext=(-4, 10), textcoords="offset points",
                             fontsize=5.5, color="#B22222", ha="center")
                assertions.append(("fig4", model, "n=5/mild delta",
                                   "%+.4f" % d["delta_ap70"]))
        for a in (ax, axz):
            mark_ungraded(a, model)
            a.set_xticks(range(len(strata)))
            a.set_xticklabels(["n=%d" % n for n in strata])
            a.grid(axis="y", color="#dddddd", zorder=0)
        # tag folded INTO the title: the band under the axis already holds
        # the tick labels and the two-line counts, and every placement of a
        # long tag in it collided -- with the tick labels, with the counts'
        # second line, or (worst) with the red thin-stratum warning.
        # TWO lines: on one line the tag made the title wider than the
        # panel and adjacent titles ran into each other.
        ax.set_title("(%s) %s\n%s" % ("abc"[col], MODEL_LABEL[model],
                                      panel_tag(rows, model)),
                     loc="left", fontsize=7, linespacing=1.5)
        if col == len(fig4_models) - 1:
            ax.text(1.02, 0.5, "CoSDH: no agent_drop\ncells exist — N/A\n"
                    "(crashes; see Fig. 2)", transform=ax.transAxes,
                    fontsize=6, color="#B22222", va="center", rotation=90)
        pad = 0.02
        axz.set_ylim(max(0.0, lo - pad), hi + pad)
        axz.set_title("zoom: %.2f–%.2f" % (axz.get_ylim()[0],
                                           axz.get_ylim()[1]),
                      loc="left", fontsize=6.5, color="#555555")
    axes[0][0].set_ylabel("AP@0.7  (from 0)")
    axes[1][0].set_ylabel("AP@0.7  (zoom)")
    handles = [Patch(facecolor=INJ_COLOR["agent_drop"], alpha=tier_alpha[t_],
                     edgecolor="#333333", lw=0.4, label=t_.capitalize())
               for t_ in TIERS]
    handles.append(Line2D([], [], color="#D55E00", marker="o",
                          label="Per-stratum clean"))
    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.02))
    footnote(fig, "CoSDH IS ABSENT FROM THIS FIGURE AND CANNOT APPEAR: it "
             "has no agent_drop cells at all -- the released inference path "
             "crashes when a collaborator is removed rather than degrading "
             "(manifest not_applicable; Fig. 2 N/A cells). The absence is "
             "the datum. " + GT_CONFOUND +
             "STRATUM DEPENDENCE: the strata are NOT equally biased. With "
             "every collaborator dropped, GT falls -11.2% at n=2 but -22.7% "
             "at n=3 -- more collaborators means more of GT is uniquely "
             "theirs -- so the bias is LARGEST exactly where collaboration "
             "should look most valuable, and it runs in the direction that "
             "FLATTERS collaboration. n=4 and n=5 magnitudes are UNMEASURED: "
             "those strata did not occur in the 200 frames sampled. "
             "Top row: bars from zero (honest baseline). Bottom row: the "
             "SAME data zoomed to the working range — the axis is not "
             "truncated in the top row, which would exaggerate differences. "
             "x-axis is scene agent count (post max_cav truncation); f = ego "
             "frames, sc = scenarios; ⚠ thin stratum, unreliable. "
             "Per-stratum clean references differ by up to 10.2 AP (CoBEVT "
             "n=2 0.616 -> n=5 0.718), which is why a pooled agent-drop "
             "number would be meaningless. At n=2, dropping a collaborator "
             "is ablation to single-agent perception, not graded "
             "degradation.", bottom=0.30)
    return save(fig, outdir, "fig4_agentdrop_stratified")


# ── FIG 5 ───────────────────────────────────────────────────────────────
# APPROVED EXCEPTION to CSV-as-single-source. The Where2comm-on-OPV2V series
# is read from the SUPERSEDED bundle tree, which the aggregator deliberately
# cannot see (it iterates MODELS.keys(), so a directory not named by a model
# key is unreachable -- that is what quarantines it).
#
# Why the exception is justified rather than a leak: the superseded status IS
# the point. It is a labelled contrast series showing the SAME checkpoint was
# MONOTONE in AP on the other dataset, which is the strongest available
# evidence that the V2XSet AP inversion is specific to Where2comm-on-V2XSet
# rather than intrinsic to the snow injector or to the architecture. Values
# are read at run time, never hard-coded, and the series is marked as
# superseded on the figure itself. Absent bundles FAIL LOUDLY -- silently
# dropping the series would remove the contrast without removing the claim.
OPV2V_CONTRAST = "results/sweep/where2comm_SUPERSEDED_opv2v_eval"


def _opv2v_snow():
    out = []
    for t in TIERS:
        p = os.path.join(OPV2V_CONTRAST, "lidar_snow", t, "fi_result.json")
        if not os.path.exists(p):
            raise SystemExit(
                "FIG5: superseded OPV2V contrast bundle missing: %s\n"
                "This series is an approved, documented exception to "
                "CSV-as-single-source and must not be silently dropped." % p)
        with open(p) as fh:
            out.append(json.load(fh)["ap_70"])
    return out


def fig5(rows, outdir, assertions):
    """Stacked, shared x. Twin axes were tried first and rejected: the three
    removal curves are near-coincident by construction, so overlaying them on
    a second scale put the legend and both annotations on top of the data.
    Stacking also matches the claim -- universal above, model-dependent below.
    """
    fig, (axr, axa) = plt.subplots(
        2, 1, figsize=(COL_W * 1.32, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.30], "hspace": 0.46})
    x = list(range(3))
    RF_C = "#0072B2"
    RF_C2 = "#994F00"                      # CoSDH removal series: OPV2V data
    AP_C = {"cobevt": "#444444", "v2xvit": "#8c8c8c",
            "where2comm": "#D55E00", "cosdh": "#CC79A7",
            "_opv2v": "#009E73"}

    # ── (a) removal fraction: model-independent WITHIN a dataset ───────
    for model in MODELS:
        rf = [float(one(rows, model=model, injector="lidar_snow",
                        severity_tier=t_)["_x"]["removed_frac"])
              for t_ in TIERS]
        if model == "cosdh":
            # OPV2V point clouds: same inverted SHAPE, different level --
            # its own dataset band, never overlaid as if comparable.
            axr.plot(x, rf, marker="D", ms=4, lw=1.4, color=RF_C2,
                     ls=(0, (3, 1)))
        else:
            solid = (model == "cobevt")
            axr.plot(x, rf, marker="s", ms=4, lw=1.8, color=RF_C,
                     alpha=1.0 if solid else 0.45,
                     ls="-" if solid else (0, (5, 1.5)))
        assertions.append(("fig5", model, "removed_frac",
                           ["%.4f" % v for v in rf]))
    axr.set_ylabel("Removal fraction")
    axr.set_title("(a) Corruption plane — model-independent within a "
                  "dataset", loc="left", fontsize=7.5)
    axr.set_ylim(0.27, 0.66)
    # DIRECT line labels, no legend box: the two dataset bands leave no
    # rectangle wide enough for a 3-entry legend without covering data
    # (upper right covered CoSDH moderate; the mid band clipped the V2XSet
    # lines at their moderate points).
    axr.text(1.0, 0.585, "CoSDH (Perfect) — OPV2V clouds", fontsize=6,
             color=RF_C2, ha="center", va="bottom")
    axr.text(0.30, 0.432, "V2XSet clouds: CoBEVT solid (Perfect); "
             "V2X-ViT / Where2comm dashed (Noisy)",
             fontsize=6, color=RF_C, va="bottom")
    axr.annotate("inverted on BOTH datasets: mild strips MOST",
                 xy=(0.03, 0.055), xycoords="axes fraction", fontsize=6,
                 color=RF_C)
    axr.annotate("CoBEVT +0.002 (Perfect ships async=false)",
                 xy=(0.03, 0.01), xycoords="axes fraction", fontsize=6,
                 color=RF_C)
    axr.grid(axis="y", color="#eeeeee", zorder=0)

    # ── (b) AP@0.7: model-dependent ────────────────────────────────────
    styles = {"cobevt": "o", "v2xvit": "s", "where2comm": "^",
              "cosdh": "D"}
    for model in MODELS:
        ap = [one(rows, model=model, injector="lidar_snow",
                  severity_tier=t_)["faulty_ap70"] for t_ in TIERS]
        inv = not (ap[0] > ap[1] > ap[2])
        ds = ("OPV2V" if model == "cosdh" else "V2XSet")
        axa.plot(x, ap, marker=styles[model], ms=4, ls=(0, (2, 1.5)),
                 color=AP_C[model], lw=2.0 if inv else 1.0,
                 zorder=5 if inv else 2,
                 label="%s / %s%s" % (MODEL_LABEL[model], ds,
                                      "  ★ INVERTED" if inv else ""))
        assertions.append(("fig5", model, "ap70 " + ds,
                           ["%.4f" % v for v in ap] +
                           ["INVERTED" if inv else "monotone"]))
        if model == "cosdh":
            # the density-hypothesis test, answered ON the figure. Placed in
            # the empty band BETWEEN the CoSDH line (~0.52-0.65) and the
            # V2XSet cluster (<=0.30): anchored above it collided with the
            # panel title.
            axa.text(0.03, 0.64,
                     "CoSDH: MONOTONE despite explicit density thresholding\n"
                     "(points_map<4 of 32) — the density hypothesis'\n"
                     "predicted strongest witness does not show the inversion",
                     transform=axa.transAxes, fontsize=5.5,
                     color=AP_C[model], va="top")
        if inv:
            # offset RIGHT, not up: directly above the Where2comm moderate
            # point sits CoBEVT's marker, and the star landed on it.
            axa.annotate("★", xy=(1, ap[1]), xytext=(13, -3),
                         textcoords="offset points", fontsize=8,
                         color=AP_C[model], ha="center")
    ap_op = _opv2v_snow()
    inv_op = not (ap_op[0] > ap_op[1] > ap_op[2])
    axa.plot(x, ap_op, marker="v", ms=4, ls=(0, (1, 2)), color=AP_C["_opv2v"],
             lw=1.0, label="Where2comm / OPV2V (superseded)")
    assertions.append(("fig5", "where2comm", "ap70 OPV2V (superseded tree)",
                       ["%.4f" % v for v in ap_op] +
                       ["INVERTED" if inv_op else "monotone"]))
    axa.set_ylabel("AP@0.7")
    axa.set_title("(b) Perception outcome — model-dependent", loc="left",
                  fontsize=7.5)
    # the caption grows the canvas downward from y=-0.03 in FIGURE coords;
    # the legend must sit in its own band above that, not share it.
    fig.subplots_adjust(bottom=0.30)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
               fontsize=6, ncol=2, columnspacing=1.0)
    axa.grid(axis="y", color="#eeeeee", zorder=0)
    axa.set_xticks(x)
    axa.set_xticklabels(["Mild (sev1)", "Moderate (sev2)", "Severe (sev3)"])
    axa.set_xlim(-0.12, 2.12)

    footnote(fig, "Removal fraction is model-independent but SETTING- and "
             "DATASET-dependent: identical to 4 dp for the two Noisy models "
             "(0.3970/0.3205/0.2877) vs 0.3991/0.3227/0.2900 for "
             "Perfect-shipped CoBEVT, and 0.5815/0.5197/0.4932 on CoSDH's "
             "OPV2V clouds — same inverted shape, different level. "
             "DENSITY-HYPOTHESIS TEST: the density explanation for the "
             "Where2comm AP inversion predicted CoSDH — whose demand mask "
             "thresholds EXPLICITLY on point density (points_map < 4 of 32, "
             "inference-only) — would show the inversion MORE strongly. It "
             "does not: CoSDH is strictly monotone (0.647/0.604/0.524) "
             "under a corruption plane that is ITSELF inverted "
             "(0.58/0.52/0.49 removed). The prediction FAILS and the "
             "hypothesis is UNDERMINED — not conclusively refuted, since "
             "CoSDH differs in dataset (OPV2V vs V2XSet) and architecture, "
             "so this is a directional test, not a controlled one. "
             + XDATA_WHY, width=74, bottom=0.40)
    return save(fig, outdir, "fig5_snow_mechanism")


# ── main ────────────────────────────────────────────────────────────────
FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
        "fig5": fig5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep/sweep_results.csv")
    ap.add_argument("--manifest", default="results/sweep/sweep_manifest.json")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--only", default="",
                    help="comma-separated subset, e.g. fig1,fig5")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.csv)
    with open(args.manifest) as fh:
        man = json.load(fh)
    # provenance the captions depend on
    assert man["graded"]["where2comm"] is False, \
        "manifest says where2comm is graded; the UNGRADED marking is wrong"
    assert man["graded"]["cosdh"] is False, \
        "manifest says cosdh is graded; the UNGRADED marking is wrong"
    assert set(man["not_applicable"]) >= {"cosdh/agent_drop",
                                          "cosdh/missing_modality"} \
        or any("cosdh" in k for k in man["not_applicable"]), \
        "manifest not_applicable no longer lists the cosdh cells this " \
        "figure set draws as N/A"

    want = [w.strip() for w in args.only.split(",") if w.strip()] or list(FIGS)
    assertions, written = [], []
    for name in want:
        if name not in FIGS:
            raise SystemExit("unknown figure %r (have %s)"
                             % (name, ", ".join(FIGS)))
        written += FIGS[name](rows, args.outdir, assertions)

    print("wrote %d files to %s" % (len(written), args.outdir))
    for p in written:
        print("  %s" % p)
    print("\n--- numbers asserted by each figure ---")
    for fig_name, model, key, val in assertions:
        print("  %-5s %-11s %-24s %s" % (fig_name, model, key, val))
    print("\n%s" % FLOOR_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
